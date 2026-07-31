"""Multi-seed confirmation: per-seed pairing and the holds-on-every-seed rule."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import xarray as xr

from panelcast.select.confirmation import (
    ConfirmationResult,
    SeedResult,
    _cached_run_mismatch,
    _confirmation_timeout,
    _descriptor_hash,
    _identity_changes,
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


def _fake_env(
    tmp_path, monkeypatch, winner_ll_for_seed=None, winner_passed_for_seed=None, dataset=None
):
    """Fake launcher that writes per-run log-lik snapshots into the named run dir.

    It also writes the manifest the orchestrator would: the cache identity is
    only reusable when each run can prove which experiment it belongs to.
    """
    import yaml as _yaml

    winner_ll_for_seed = winner_ll_for_seed or (lambda seed: _WINNER_GOOD_LL)
    winner_passed_for_seed = winner_passed_for_seed or (lambda seed: True)

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        name = Path(config_path).stem  # confirm_<label>_seed<seed>
        label = "winner" if "winner" in name else "reference"
        seed = int(name.split("seed")[-1])
        payload = _yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        run_dir = tmp_path / "outputs" / payload["run_id"]
        ll = winner_ll_for_seed(seed) if label == "winner" else _REF_LL
        _write_ll(run_dir / "evaluation" / "log_likelihood.nc", ll)
        _write_manifest(run_dir, payload, dataset)
        if label == "winner":
            (run_dir / "evaluation" / "diagnostics.json").write_text(
                json.dumps({"passed": winner_passed_for_seed(seed)}), encoding="utf-8"
            )
        return 0, "ok"

    cfg = SweepConfig(
        sweep_id="c",
        dataset=dataset,
        output_root=tmp_path / "select",
        panelcast_bin="pc",
        pipeline_output_base=tmp_path / "outputs",
    )
    return cfg, launch


def _write_manifest(run_dir: Path, config_payload: dict, dataset) -> None:
    """The manifest fields the confirmation cache checks a reused run against."""
    from panelcast.config.descriptor import load_descriptor

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "success": True,
                "experiment_identity": {
                    "descriptor_hash": load_descriptor(dataset).descriptor_hash(),
                    # run_id is execution mechanics and a None value is unset:
                    # both are excluded exactly as the orchestrator's recorder
                    # excludes them.
                    "config_payload": {
                        k: v
                        for k, v in config_payload.items()
                        if k != "run_id" and v is not None
                    },
                },
            }
        ),
        encoding="utf-8",
    )


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


class TestHandshakeGuard:
    def test_winner_fit_without_named_run_dir_is_an_error(self, tmp_path, monkeypatch):
        # A winner child that exits 0 without creating its NAMED run dir must
        # fail the seed, never silently pair against something else (#167).
        cfg, launch = _fake_env(tmp_path, monkeypatch)

        def no_dir_winner(config_path, panelcast_bin, timeout_seconds=None):
            if "winner" in Path(config_path).stem:
                return 0, "ok"  # succeeds but never creates its run dir
            return launch(config_path, panelcast_bin, timeout_seconds)

        result = run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42,), launch=no_dir_winner
        )
        assert not result.confirmed
        assert "not resolved" in result.seeds[0].error


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

    def test_a_different_dataset_is_a_different_confirmation(self, tmp_path, monkeypatch):
        """Same sweep id, same winner knobs, other domain: nothing is reused."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2

        other_cfg, other_launch = _fake_env(tmp_path, monkeypatch, dataset="aero")
        assert other_cfg.sweep_dir == cfg.sweep_dir
        calls, counting = self._counting(other_launch)
        result = run_confirmation(
            {"latent_process": "ar1"}, other_cfg, seeds=(42,), launch=counting
        )
        assert calls["n"] == 2
        assert result.dataset_descriptor_hash is not None
        assert result.dataset_descriptor_hash != _descriptor_hash(cfg.dataset)
        assert [
            p
            for p in cfg.sweep_dir.iterdir()
            if p.name.startswith("confirmation_") and p.suffix == ".json"
        ]

    @pytest.mark.parametrize(
        ("tamper", "reason", "key"),
        [
            (None, "no readable manifest", None),
            (
                lambda m: m.update(success=False),
                "manifest does not record a successful run",
                None,
            ),
            (
                lambda m: m.update(experiment_identity={}),
                "manifest records no experiment identity",
                None,
            ),
            (
                lambda m: m["experiment_identity"].update(config_payload=None),
                "manifest records no config payload",
                None,
            ),
            (
                lambda m: m["experiment_identity"].pop("descriptor_hash"),
                "manifest records no dataset descriptor hash",
                None,
            ),
            (
                lambda m: m["experiment_identity"].update(descriptor_hash="0" * 64),
                "the run was fit on another dataset",
                "dataset_descriptor_hash",
            ),
            (
                lambda m: m["experiment_identity"]["config_payload"].update(stages=["train"]),
                "the run was fit with another value",
                "stages",
            ),
            (
                # A value neither arm holds, so it disagrees with either side.
                lambda m: m["experiment_identity"]["config_payload"].update(latent_process="ou"),
                "the run was fit with another value",
                "latent_process",
            ),
            (
                lambda m: m["experiment_identity"]["config_payload"].pop("seed"),
                "manifest does not record the key that identifies this fit",
                "seed",
            ),
            (
                lambda m: m["experiment_identity"]["config_payload"].pop("latent_process"),
                "manifest does not record the key that identifies this fit",
                "latent_process",
            ),
        ],
    )
    @pytest.mark.parametrize("label", ["reference", "winner"])
    def test_a_cached_run_that_cannot_prove_itself_refits(
        self, tmp_path, monkeypatch, tamper, reason, key, label
    ):
        """The identity file matching is not enough — each run's manifest must too."""
        import structlog

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        path = Path(getattr(result.seeds[0], f"{label}_run")) / "manifest.json"
        if tamper is None:
            path.unlink()
        else:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            tamper(manifest)
            path.write_text(json.dumps(manifest), encoding="utf-8")

        calls, counting = self._counting(launch)
        with structlog.testing.capture_logs() as logs:
            run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2  # both sides of the seed refit
        rejected = [e for e in logs if e["event"] == "confirmation_cached_run_rejected"]
        assert [(e["label"], e["reason"], e["key"]) for e in rejected] == [(label, reason, key)]

    def test_swapping_the_paired_runs_refits(self, tmp_path, monkeypatch):
        """Each side is checked against its own arm, so the pair cannot be reversed."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        out = cfg.sweep_dir / "confirmation.json"
        payload = json.loads(out.read_text(encoding="utf-8"))
        seed = payload["seeds"][0]
        seed["reference_run"], seed["winner_run"] = seed["winner_run"], seed["reference_run"]
        out.write_text(json.dumps(payload), encoding="utf-8")
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2

    def test_crossing_a_run_between_seeds_refits(self, tmp_path, monkeypatch):
        """Pairing is within a seed, so a run dir from another one is not this seed's."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=launch)
        out = cfg.sweep_dir / "confirmation.json"
        payload = json.loads(out.read_text(encoding="utf-8"))
        first, second = payload["seeds"]
        first["winner_run"] = second["winner_run"]
        out.write_text(json.dumps(payload), encoding="utf-8")
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42, 43), launch=counting)
        assert calls["n"] == 2  # seed 43 still reuses; only the crossed seed refits

    def test_a_changed_base_option_is_a_different_confirmation(self, tmp_path, monkeypatch):
        """The identity's other half: an option the run gate would never see."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        calls, counting = self._counting(launch)
        first = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2
        cfg.extra_config = {"max_events": 500}
        calls, counting = self._counting(launch)
        second = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2
        assert first.fit_config_hash is not None
        assert second.fit_config_hash != first.fit_config_hash
        assert [
            p
            for p in cfg.sweep_dir.iterdir()
            if p.name.startswith("confirmation_") and p.suffix == ".json"
        ]

    def test_a_changed_reference_arm_is_a_different_confirmation(self, tmp_path, monkeypatch):
        """The winner knobs are a delta; the identity has to hold what they vary from."""
        import structlog

        from panelcast.select import confirmation as mod
        from panelcast.select.space import default_arm

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        calls, counting = self._counting(launch)
        # A shipped default moving off None is invisible to the run gate: the
        # recorder omits an unset value, so no manifest contradicts it.
        with monkeypatch.context() as moved, structlog.testing.capture_logs() as logs:
            moved.setattr(
                mod, "default_arm", lambda: {**default_arm(), "entity_group_pooling": True}
            )
            run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2
        assert [
            e["changed"] for e in logs if e["event"] == "confirmation_cache_archived"
        ] == [["fit_config_hash"]]

    def test_a_regenerated_dataset_is_a_different_confirmation(self, tmp_path, monkeypatch):
        """The descriptor's content, not its name: same `aero`, re-extracted."""
        import structlog

        from panelcast.config.descriptor import load_descriptor

        cfg, launch = _fake_env(tmp_path, monkeypatch, dataset="aero")
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)

        class _Regenerated:
            def __init__(self, inner):
                self._inner = inner

            def descriptor_hash(self):
                return "1" * 64

        calls, counting = self._counting(launch)
        with monkeypatch.context() as moved, structlog.testing.capture_logs() as logs:
            moved.setattr(
                "panelcast.select.confirmation.load_descriptor",
                lambda ref: _Regenerated(load_descriptor(ref)),
            )
            run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 2
        assert [
            e["changed"] for e in logs if e["event"] == "confirmation_cache_archived"
        ] == [["dataset_descriptor_hash"]]

    def test_a_dataset_only_extra_config_names_is_the_one_resolved(self, tmp_path, monkeypatch):
        """cfg is one of three writers of the key; what the fits run on is what is written."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.extra_config = {"dataset": "aero"}
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        assert result.dataset_descriptor_hash == _descriptor_hash("aero")
        assert result.dataset_descriptor_hash != _descriptor_hash(None)

    def test_a_knob_the_winner_overrides_still_reuses(self, tmp_path, monkeypatch):
        """An extra_config value an arm overrides is compared as the arm's, not the base's."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.extra_config = {"latent_process": "rw"}
        first = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 0
        # Both arms set it, so the extra_config value reaches no fit: editing it
        # must not cost a publication-scale refit that rewrites the same configs.
        cfg.extra_config = {"latent_process": "ar1"}
        calls, counting = self._counting(launch)
        second = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 0
        assert second.fit_config_hash == first.fit_config_hash

    def test_a_winner_knob_a_base_option_overwrites_is_refused(self, tmp_path, monkeypatch):
        """Both sides would run one configuration; the pairing would self-compare."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.num_samples = 200
        with pytest.raises(ValueError, match="overwritten by a base option"):
            run_confirmation({"num_samples": 4000}, cfg, seeds=(42,), launch=launch)

    def test_an_option_the_manifest_never_recorded_still_reuses(self, tmp_path, monkeypatch):
        """A key with no recorded value has nothing to disagree with."""
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        for run in (result.seeds[0].reference_run, result.seeds[0].winner_run):
            path = Path(run) / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["experiment_identity"]["config_payload"].pop("stages")
            path.write_text(json.dumps(manifest), encoding="utf-8")
        calls, counting = self._counting(launch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 0

    def test_an_unloadable_descriptor_resolves_to_no_hash(self):
        """An unloadable descriptor is an absence, not a value another absence matches."""
        assert _descriptor_hash("not-a-dataset") is None

    def test_a_seed_override_is_refused(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="must not set seed"):
            run_confirmation(
                {"latent_process": "ar1"},
                cfg,
                seeds=(42,),
                sampler_overrides={"seed": 7},
                launch=launch,
            )

    def test_a_ledger_written_blind_survives_the_healthy_call(self):
        """A stale mount must not cost the refit twice — once now, once when it heals."""
        healthy = {"promote_z": 2.0, "dataset_descriptor_hash": "abc"}
        blind = {"promote_z": 2.0, "dataset_descriptor_hash": None}
        assert _identity_changes(blind, healthy) == []
        # The mirror case still archives: this call reuses nothing either way,
        # and the checkpoint would overwrite the ledger without a copy.
        assert _identity_changes(healthy, blind) == ["dataset_descriptor_hash"]
        assert _identity_changes({"dataset_descriptor_hash": "xyz"}, healthy) == [
            "dataset_descriptor_hash",
            "promote_z",
        ]

    def test_a_blind_descriptor_refits_every_seed_without_raising(
        self, tmp_path, monkeypatch
    ):
        """The whole path, not its helpers: a descriptor that will not load."""
        import structlog

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)

        def unloadable(_ref):
            raise OSError("stale mount")

        calls, counting = self._counting(launch)
        # Scoped, so undoing the blindness cannot revert whatever else the
        # fixture patched.
        with monkeypatch.context() as blind, structlog.testing.capture_logs() as logs:
            blind.setattr("panelcast.select.confirmation.load_descriptor", unloadable)
            result = run_confirmation(
                {"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting
            )
        assert calls["n"] == 2
        # Not "the dataset changed", which is what a stale mount must never be
        # reported as — and the only thing distinguishing the two guard orders.
        assert [
            (e["reason"], e["changed"]) for e in logs if e["event"] == "confirmation_cache_archived"
        ] == [("unresolved dataset descriptor", None)]
        assert result.dataset_descriptor_hash is None
        persisted = json.loads(
            (cfg.sweep_dir / "confirmation.json").read_text(encoding="utf-8")
        )
        assert persisted["dataset_descriptor_hash"] is None
        archived = [
            p
            for p in cfg.sweep_dir.iterdir()
            if p.name.startswith("confirmation_") and p.suffix == ".json"
        ]
        assert archived

        # The blink costs one refit, not two: the ledger it wrote is not
        # archived again when the descriptor comes back.
        calls, counting = self._counting(launch)
        result = run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=counting)
        assert calls["n"] == 0
        assert result.confirmed
        assert [
            p
            for p in cfg.sweep_dir.iterdir()
            if p.name.startswith("confirmation_") and p.suffix == ".json"
        ] == archived

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


class TestManifestContract:
    """The gate reads a real manifest, not the shape the fixture happens to write."""

    @staticmethod
    def _record_run(cfg, arm, seed, run_dir):
        """A run dir carrying the manifest the orchestrator would have written."""
        import yaml as _yaml
        from panelcast.config.descriptor import load_descriptor
        from panelcast.config.pipeline_yaml import (
            apply_yaml_overrides,
            experiment_config_payload,
        )
        from panelcast.pipelines.manifest import (
            EnvironmentInfo,
            GitStateModel,
            RunManifest,
            save_run_manifest,
        )
        from panelcast.pipelines.orchestrator import PipelineConfig
        from panelcast.select.confirmation import _write_config

        config_path = cfg.sweep_dir / f"{run_dir.name}.yaml"
        _write_config(cfg, arm, seed, config_path, run_id=run_dir.name)
        # The layering a real fit goes through: the YAML select wrote, mapped
        # onto PipelineConfig, recorded by the orchestrator's own recorder.
        written = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = PipelineConfig(**apply_yaml_overrides({}, written))
        recorded = experiment_config_payload(config)
        # The fixture's own manifest excludes run_id on the claim that the real
        # recorder does; this is where that claim gets checked.
        assert config.run_id == run_dir.name
        assert "run_id" not in recorded
        save_run_manifest(
            RunManifest(
                run_id=config.run_id,
                created_at="2026-07-31T00:00:00",
                command="panelcast run",
                flags={},
                seed=config.seed,
                git=GitStateModel(commit="a" * 40, branch="main", dirty=False, untracked_count=0),
                environment=EnvironmentInfo(
                    python_version="3.14",
                    jax_version="0.8.2",
                    numpyro_version=None,
                    arviz_version=None,
                    platform="Linux",
                    pixi_lock_hash=None,
                ),
                input_hashes={},
                stage_hashes={},
                stages_completed=[],
                stages_skipped=[],
                outputs={},
                success=True,
                experiment_identity={
                    "descriptor_hash": load_descriptor(config.dataset).descriptor_hash(),
                    "config_payload": recorded,
                },
            ),
            run_dir,
        )
        return recorded

    @pytest.mark.parametrize("dataset", [None, "aero"])
    # entity_group_pooling is the tri-state knob whose default is None: the
    # recorder omits an unset value, so the reference's manifest cannot carry
    # the key the winner differs on.
    @pytest.mark.parametrize(
        "winner_knobs", [{"latent_process": "ar1"}, {"entity_group_pooling": True}]
    )
    def test_a_manifest_written_the_orchestrators_way_verifies(
        self, tmp_path, dataset, winner_knobs
    ):
        from panelcast.select.confirmation import _discriminating_keys, _fit_config_payload
        from panelcast.select.space import default_arm

        cfg = SweepConfig(
            sweep_id="c",
            dataset=dataset,
            output_root=tmp_path / "select",
            num_samples=200,
            num_warmup=100,
            # The last compared key whose round trip would otherwise be unpinned.
            extra_config={"max_events": 500},
        )
        cfg.sweep_dir.mkdir(parents=True)
        arms = {"reference": dict(default_arm()), "winner": {**default_arm(), **winner_knobs}}
        payloads = {
            label: _fit_config_payload(cfg, arm, None, 42) for label, arm in arms.items()
        }
        runs = {
            label: tmp_path / "outputs" / f"sel_c_confirm_{label}_seed42_x"
            for label in arms
        }
        recorded = {
            label: self._record_run(cfg, arms[label], 42, runs[label]) for label in arms
        }

        # Not vacuous, and not only the knobs this arm happens to set: every
        # knob a winner can differ on has to survive the round trip, or the
        # presence floor would reject a real run forever.
        assert {k for k, v in default_arm().items() if v is not None} <= set(recorded["winner"])
        assert {"stages", "num_samples", "seed", "max_events"} <= set(recorded["winner"])
        assert ("dataset" in recorded["winner"]) == (dataset is not None)

        discriminating = _discriminating_keys(
            payloads["reference"], payloads["winner"], cfg.dataset
        )
        assert set(winner_knobs) <= discriminating
        descriptor_hash = _descriptor_hash(dataset)
        assert descriptor_hash is not None
        for label, run_dir in runs.items():
            assert (
                _cached_run_mismatch(run_dir, descriptor_hash, payloads[label], discriminating)
                is None
            )
        # The pair is not interchangeable: the other side's payload disagrees.
        assert _cached_run_mismatch(
            runs["reference"], descriptor_hash, payloads["winner"], discriminating
        ) is not None


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
