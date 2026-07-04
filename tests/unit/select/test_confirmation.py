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


def _fake_env(tmp_path, monkeypatch, winner_ll_for_seed=None):
    """Fake launcher that writes per-run log-lik snapshots; patches resolve_latest."""
    counter = {"n": 0}
    winner_ll_for_seed = winner_ll_for_seed or (lambda seed: _WINNER_GOOD_LL)

    def launch(config_path: Path, panelcast_bin: str) -> tuple[int, str]:
        name = Path(config_path).stem  # confirm_<label>_seed<seed>
        label = "winner" if "winner" in name else "reference"
        seed = int(name.split("seed")[-1])
        counter["n"] += 1
        run_dir = tmp_path / "outputs" / f"run_{counter['n']:03d}"
        ll = winner_ll_for_seed(seed) if label == "winner" else _REF_LL
        _write_ll(run_dir / "evaluation" / "log_likelihood.nc", ll)
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

    def test_checkpoint_written_each_seed(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        payload = json.loads((cfg.sweep_dir / "confirmation.json").read_text(encoding="utf-8"))
        assert payload["winner_knobs"] == {"latent_process": "ar1"}
        assert len(payload["seeds"]) == 1

    def test_fit_failure_recorded_not_raised(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)

        def flaky(config_path: Path, panelcast_bin: str) -> tuple[int, str]:
            if "seed43" in Path(config_path).stem:
                return 1, "boom"
            return launch(config_path, panelcast_bin)

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
            seeds=[SeedResult(seed=s, elpd={"diff": 20.0, "dse": 4.0, "z": 5.0}) for s in (42, 43)],
        )
        md = render_confirmation(result)
        assert "CONFIRMED" in md
        assert "manual PR" in md
        assert md.count("| 4") >= 2

    def test_not_confirmed_block(self):
        result = ConfirmationResult(
            winner_knobs={},
            seeds=[SeedResult(seed=42, error="failed")],
        )
        md = render_confirmation(result)
        assert "NOT CONFIRMED" in md
