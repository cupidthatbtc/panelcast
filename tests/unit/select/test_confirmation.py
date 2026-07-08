"""Multi-seed confirmation: per-seed pairing and the holds-on-every-seed rule."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from panelcast.select.confirmation import (
    ConfirmationResult,
    SeedResult,
    _confirmation_timeout,
    render_confirmation,
    run_confirmation,
)
from panelcast.select.runner import SweepConfig

# Reference log-lik: zeros -> elpd_i = 0. Winner (good): varying positive
# densities -> a positive, finite-variance paired diff (finite z).
_REF_LL = np.zeros((2, 4))
_WINNER_GOOD_LL = np.log(np.array([[2.0, 4.0, 8.0, 16.0]] * 2))
_WINNER_FLAT_LL = np.log(np.array([[1.001, 1.0, 1.0, 1.0]] * 2))  # ~zero diff -> tiny z


def _write_ll(path: Path, ll: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        ll[None, :, :],
        dims=["chain", "draw", "obs"],
        coords={"chain": [0], "draw": range(ll.shape[0]), "obs": range(ll.shape[1])},
    )
    az.InferenceData(log_likelihood=xr.Dataset({"y": da})).to_netcdf(str(path))


def _fake_env(tmp_path, monkeypatch, winner_ll_for_seed=None, winner_passed_for_seed=None):
    """Fake launcher that writes per-run log-lik snapshots; patches resolve_latest."""
    counter = {"n": 0}
    winner_ll_for_seed = winner_ll_for_seed or (lambda seed: _WINNER_GOOD_LL)
    winner_passed_for_seed = winner_passed_for_seed or (lambda seed: True)

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        name = Path(config_path).stem  # confirm_<label>_seed<seed>
        label = "winner" if "winner" in name else "reference"
        seed = int(name.split("seed")[-1])
        counter["n"] += 1
        run_dir = tmp_path / "outputs" / f"run_{counter['n']:03d}"
        ll = winner_ll_for_seed(seed) if label == "winner" else _REF_LL
        _write_ll(run_dir / "evaluation" / "log_likelihood.nc", ll)
        if label == "winner":
            (run_dir / "evaluation" / "diagnostics.json").write_text(
                json.dumps({"passed": winner_passed_for_seed(seed)}), encoding="utf-8"
            )
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_dir": run_dir.name}), encoding="utf-8"
        )
        return 0, "ok"

    import panelcast.paths as paths_mod

    def _latest(output_base=Path("outputs")):
        data = json.loads((tmp_path / "outputs" / "latest.json").read_text(encoding="utf-8"))
        return tmp_path / "outputs" / data["run_dir"]

    monkeypatch.setattr(paths_mod, "resolve_latest", _latest)
    cfg = SweepConfig(sweep_id="c", output_root=tmp_path / "select", panelcast_bin="pc")
    return cfg, launch


class TestRunConfirmation:
    def test_all_seeds_positive_confirms(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43, 44), promote_z=2.0, launch=launch
        )
        assert result.confirmed
        assert len(result.seeds) == 3
        assert all(s.elpd["z"] > 2.0 for s in result.seeds)

    def test_one_flat_seed_fails_confirmation(self, tmp_path, monkeypatch):
        def winner_ll(seed):
            return _WINNER_FLAT_LL if seed == 43 else _WINNER_GOOD_LL

        cfg, launch = _fake_env(tmp_path, monkeypatch, winner_ll)
        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43, 44), promote_z=2.0, launch=launch
        )
        assert not result.confirmed

    def test_nonconverged_winner_seed_fails_confirmation(self, tmp_path, monkeypatch):
        # z clears the bar on every seed, but the winner fails the convergence
        # gate on one — confirmation must still fail (publication-scale gate).
        cfg, launch = _fake_env(
            tmp_path, monkeypatch, winner_passed_for_seed=lambda seed: seed != 43
        )
        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43, 44), promote_z=2.0, launch=launch
        )
        assert not result.confirmed
        seed43 = next(s for s in result.seeds if s.seed == 43)
        assert seed43.winner_converged is False
        assert seed43.elpd["z"] > 2.0

    def test_checkpoint_written_each_seed(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        payload = json.loads((cfg.sweep_dir / "confirmation.json").read_text(encoding="utf-8"))
        assert payload["winner_knobs"] == {"latent_process": "ar1"}
        assert len(payload["seeds"]) == 1

    def test_fit_failure_recorded_not_raised(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)

        def flaky(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
            if "seed43" in Path(config_path).stem:
                return 1, "boom"
            return launch(config_path, panelcast_bin, timeout_seconds)

        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=flaky
        )
        seed43 = next(s for s in result.seeds if s.seed == 43)
        assert seed43.error is not None
        assert not result.confirmed

    def test_sampler_overrides_written_to_config(self, tmp_path, monkeypatch):
        import yaml

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation(
            {"latent_process": "ar1"},
            cfg,
            seeds=(42,),
            sampler_overrides={"num_samples": 5000},
            launch=launch,
        )
        config = yaml.safe_load(
            (cfg.sweep_dir / "confirm_winner_seed42.yaml").read_text(encoding="utf-8")
        )
        assert config["num_samples"] == 5000
        assert config["seed"] == 42
        assert config["stages"] == ["splits", "features", "train", "evaluate"]


class TestConfirmationTimeout:
    def _captured_timeouts(self, tmp_path, monkeypatch, cfg, sampler_overrides):
        base_cfg, launch = _fake_env(tmp_path, monkeypatch)
        timeouts: list = []

        def capturing(config_path, panelcast_bin, timeout_seconds=None):
            timeouts.append(timeout_seconds)
            return launch(config_path, panelcast_bin, timeout_seconds)

        run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42,),
            sampler_overrides=sampler_overrides, launch=capturing,
        )
        return timeouts

    def test_timeout_scaled_by_publication_sampler_ratio(self, tmp_path, monkeypatch):
        cfg = SweepConfig(
            sweep_id="c", output_root=tmp_path / "select", panelcast_bin="pc",
            num_samples=1000, num_warmup=1000, arm_timeout_seconds=1800.0,
        )
        timeouts = self._captured_timeouts(
            tmp_path, monkeypatch, cfg,
            {"num_chains": 4, "num_samples": 5000, "num_warmup": 5000},
        )
        # (5000+5000)/(1000+1000) = 5x the screening timeout, on every fit.
        assert timeouts == [9000.0, 9000.0]

    def test_screening_timeout_is_the_floor_without_overrides(self, tmp_path, monkeypatch):
        cfg = SweepConfig(
            sweep_id="c", output_root=tmp_path / "select", panelcast_bin="pc",
            num_samples=1000, num_warmup=1000, arm_timeout_seconds=1800.0,
        )
        timeouts = self._captured_timeouts(tmp_path, monkeypatch, cfg, None)
        assert timeouts == [1800.0, 1800.0]

    def test_no_timeout_when_arm_timeout_unset(self, tmp_path, monkeypatch):
        cfg = SweepConfig(
            sweep_id="c", output_root=tmp_path / "select", panelcast_bin="pc",
            num_samples=1000, num_warmup=1000,
        )
        timeouts = self._captured_timeouts(
            tmp_path, monkeypatch, cfg, {"num_samples": 5000, "num_warmup": 5000}
        )
        assert timeouts == [None, None]


class TestConfirmationAutoTimeout:
    def _auto_cfg(self, tmp_path) -> SweepConfig:
        return SweepConfig(
            sweep_id="c", output_root=tmp_path / "select", panelcast_bin="pc",
            num_samples=1000, num_warmup=1000, arm_timeout_seconds="auto",
        )

    def test_auto_base_is_the_resolved_timeout_scaled(self, tmp_path, monkeypatch):
        import panelcast.gpu_memory.runtime_predictor as rp

        monkeypatch.setattr(
            rp, "predict_fit_seconds",
            lambda *a, **k: rp.RuntimePrediction(seconds=1000.0, source="stub"),
        )
        timeout = _confirmation_timeout(
            self._auto_cfg(tmp_path), {"num_samples": 5000, "num_warmup": 5000},
            winner_knobs={"latent_process": "ar1"}, dims={"n_observations": 5000},
        )
        # base max(1800 floor, 3x1000) = 3000, x5 publication sampler ratio.
        assert timeout == 15000.0

    def test_auto_without_dims_scales_the_floor(self, tmp_path):
        timeout = _confirmation_timeout(
            self._auto_cfg(tmp_path), {"num_samples": 5000, "num_warmup": 5000},
            winner_knobs={}, dims=None,
        )
        assert timeout == 1800.0 * 5

    def test_run_confirmation_threads_auto_timeout(self, tmp_path, monkeypatch):
        _, launch = _fake_env(tmp_path, monkeypatch)
        cfg = self._auto_cfg(tmp_path)
        timeouts: list = []

        def capturing(config_path, panelcast_bin, timeout_seconds=None):
            timeouts.append(timeout_seconds)
            return launch(config_path, panelcast_bin, timeout_seconds)

        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=capturing)
        # No dims: auto resolves to the floor, unscaled without overrides.
        assert timeouts == [1800.0, 1800.0]


class TestSelfPairGuard:
    def test_winner_resolving_to_reference_run_is_an_error(self, tmp_path, monkeypatch):
        # A failed latest-pointer write after the winner fit leaves the pointer
        # on the reference run: pairing it against itself would fake a z=0 seed.
        cfg, launch = _fake_env(tmp_path, monkeypatch)

        def sticky_pointer(config_path, panelcast_bin, timeout_seconds=None):
            if "winner" in Path(config_path).stem:
                return 0, "ok"  # succeeds but never re-points latest.json
            return launch(config_path, panelcast_bin, timeout_seconds)

        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42,), launch=sticky_pointer
        )
        assert not result.confirmed
        assert "self-pair" in result.seeds[0].error


class TestConfirmationResult:
    def test_unmeasured_seed_prevents_confirmation(self):
        result = ConfirmationResult(
            winner_knobs={},
            seeds=[
                SeedResult(seed=42, elpd={"z": 5.0}),
                SeedResult(seed=43, error="crash"),
            ],
        )
        assert not result.confirmed

    def test_empty_is_not_confirmed(self):
        assert not ConfirmationResult(winner_knobs={}).confirmed


class TestRender:
    def test_confirmed_block(self):
        result = ConfirmationResult(
            winner_knobs={"latent_process": "ar1"},
            seeds=[
                SeedResult(seed=s, elpd={"diff": 20.0, "dse": 4.0, "z": 5.0}, winner_converged=True)
                for s in (42, 43)
            ],
        )
        md = render_confirmation(result)
        assert "CONFIRMED" in md
        assert "manual PR" in md
        assert md.count("| 4") >= 2

    def test_convergence_failure_block(self):
        result = ConfirmationResult(
            winner_knobs={"latent_process": "ar1"},
            seeds=[
                SeedResult(seed=42, elpd={"diff": 20.0, "dse": 4.0, "z": 5.0}, winner_converged=True),
                SeedResult(seed=43, elpd={"diff": 20.0, "dse": 4.0, "z": 5.0}, winner_converged=False),
            ],
        )
        md = render_confirmation(result)
        assert "NOT CONFIRMED" in md
        assert "convergence gate" in md
        assert "FAIL" in md

    def test_not_confirmed_block(self):
        result = ConfirmationResult(
            winner_knobs={},
            seeds=[SeedResult(seed=42, error="failed")],
        )
        md = render_confirmation(result)
        assert "NOT CONFIRMED" in md


class TestConfirmationResume:
    """Re-entry reuses prior seeds from persisted snapshots (#165)."""

    def _counting(self, launch):
        calls = {"n": 0}

        def counting(config_path, panelcast_bin, timeout_seconds=None):
            calls["n"] += 1
            return launch(config_path, panelcast_bin, timeout_seconds)

        return calls, counting

    def test_rerun_refits_nothing(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=counting)
        assert calls["n"] == 4
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=counting)
        assert calls["n"] == 4  # both seeds re-paired from snapshots
        assert result.confirmed
        assert all(s.elpd["z"] > 2.0 for s in result.seeds)

    def test_interrupt_resumes_at_missing_seed(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        # Same protocol, one more seed: 42 reused, only 43 fits (2 launches).
        calls, counting = self._counting(launch)
        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=counting
        )
        # seeds tuple is part of the identity: (42,) != (42, 43) archives and refits all
        assert calls["n"] == 4
        assert result.confirmed
        assert any(p.name.startswith("confirmation_") and p.name != "confirmation.json"
                   for p in cfg.sweep_dir.iterdir())

    def test_protocol_change_archives_and_refits(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), promote_z=3.0,
                         launch=counting)
        assert calls["n"] == 4  # z bar changed: no reuse
        archived = [p for p in cfg.sweep_dir.iterdir()
                    if p.name.startswith("confirmation_") and p.suffix == ".json"]
        assert archived

    def test_missing_snapshot_refits_that_seed(self, tmp_path, monkeypatch):
        import shutil

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=launch)
        victim = Path(next(s for s in result.seeds if s.seed == 43).winner_run)
        shutil.rmtree(victim)
        calls, counting = self._counting(launch)
        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=counting
        )
        assert calls["n"] == 2  # seed 42 reused; seed 43 refit both sides
        assert result.confirmed

    def test_failed_seed_is_not_reused(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)

        def flaky(config_path, panelcast_bin, timeout_seconds=None):
            if "seed43" in Path(config_path).stem:
                return 1, "boom"
            return launch(config_path, panelcast_bin, timeout_seconds)

        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=flaky)
        calls, counting = self._counting(launch)
        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=counting
        )
        assert calls["n"] == 2  # only the failed seed refits
        assert result.confirmed

    def test_reused_seed_rechecks_convergence(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        win = Path(result.seeds[0].winner_run)
        (win / "evaluation" / "diagnostics.json").write_text(
            json.dumps({"passed": False}), encoding="utf-8"
        )
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        assert result.seeds[0].winner_converged is False
        assert not result.confirmed


class TestConfirmationAlwaysCold:
    def test_confirmation_configs_never_carry_warmup_transfer(self, tmp_path, monkeypatch):
        import yaml

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.warmup_transfer = True  # even when the sweep transferred, confirmation is cold
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        for p in cfg.sweep_dir.glob("confirm_*.yaml"):
            payload = yaml.safe_load(p.read_text(encoding="utf-8"))
            assert "warmup_import_path" not in payload
            assert "warmup_export_path" not in payload
