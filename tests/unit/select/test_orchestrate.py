"""Orchestration: pre-run plan, cost prediction, and the full run_select flow."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from panelcast.config.descriptor import DatasetDescriptor, load_descriptor
from panelcast.select.orchestrate import build_plan, render_plan, run_select
from panelcast.select.rules import DecisionRules
from panelcast.select.runner import SweepConfig, ofat_arms
from panelcast.select.tiers import EffortTier, tier_to_sweep_config

AOTY = DatasetDescriptor()
REPO_ROOT = Path(__file__).resolve().parents[3]
STANDARD = EffortTier("standard", (1, 2), 4, 1000, 1000, confirm=True)


def _cfg(tmp_path, **kw) -> SweepConfig:
    return tier_to_sweep_config(STANDARD, sweep_id="s", output_root=tmp_path / "select", **kw)


class TestBuildPlan:
    def test_stage1_count_matches_ofat(self, tmp_path):
        plan = build_plan(AOTY, STANDARD, _cfg(tmp_path), dataset_label="aoty")
        assert plan.n_stage1_arms == 1 + len(ofat_arms(AOTY))

    def test_full_space_enumerated(self, tmp_path):
        plan = build_plan(AOTY, STANDARD, _cfg(tmp_path), dataset_label="aoty")
        assert len(plan.space["likelihood_family"]) == 9
        assert plan.space["entity_group_pooling"] == (None, True, False)
        assert plan.pruned == {}  # AOTY prunes nothing structurally

    def test_aero_shows_structural_pruning(self):
        aero = load_descriptor(REPO_ROOT / "examples" / "aerospace" / "descriptor.yaml")
        plan = build_plan(aero, STANDARD, SweepConfig(sweep_id="s"), dataset_label="aero")
        assert "beta_binomial" in plan.pruned["likelihood_family"]

    def test_cost_note_without_data(self, tmp_path):
        plan = build_plan(AOTY, STANDARD, _cfg(tmp_path), dataset_label="aoty")
        assert plan.predicted_gpu_hours is None
        assert "no prepared data" in plan.cost_source

    def test_cost_predicted_with_dims(self, tmp_path):
        dims = {"n_observations": 5000, "n_features": 40, "n_artists": 900, "max_seq": 30}
        plan = build_plan(
            AOTY, STANDARD, _cfg(tmp_path), dataset_label="aoty", dims=dims,
            calibration_store_path=tmp_path / "cal.json",
        )
        assert plan.predicted_gpu_hours > 0
        assert plan.predicted_peak_gb > 0
        assert "cold-start" in plan.cost_source

    def test_max_fits_caps_plan(self, tmp_path):
        plan = build_plan(AOTY, STANDARD, _cfg(tmp_path, max_fits=5), dataset_label="aoty")
        assert plan.max_fits_planned == 5


class TestRenderPlan:
    def test_contains_space_and_pruned(self):
        aero = load_descriptor(REPO_ROOT / "examples" / "aerospace" / "descriptor.yaml")
        plan = build_plan(aero, STANDARD, SweepConfig(sweep_id="s"), dataset_label="aero")
        text = render_plan(plan)
        assert "effort=standard" in text
        assert "likelihood_family:" in text
        assert "pruned:" in text
        assert "beta_binomial" in text

    def test_shows_predicted_cost(self, tmp_path):
        dims = {"n_observations": 5000, "n_features": 40, "n_artists": 900, "max_seq": 30}
        plan = build_plan(
            AOTY, STANDARD, _cfg(tmp_path), dataset_label="aoty", dims=dims,
            calibration_store_path=tmp_path / "c.json",
        )
        assert "GPU-h" in render_plan(plan)


def _write_scored_run(run_dir: Path, ll: np.ndarray, *, converged: bool = True) -> None:
    ev = run_dir / "evaluation"
    ev.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        ll[None, :, :],
        dims=["chain", "draw", "obs"],
        coords={"chain": [0], "draw": range(ll.shape[0]), "obs": range(ll.shape[1])},
    )
    az.InferenceData(log_likelihood=xr.Dataset({"y": da})).to_netcdf(str(ev / "log_likelihood.nc"))
    (ev / "metrics.json").write_text(
        json.dumps(
            {
                "calibration": {
                    "coverages": {
                        "0.80": {"empirical": 0.80, "nominal": 0.80},
                        "0.95": {"empirical": 0.95, "nominal": 0.95},
                    },
                    "pit": {"max_abs_dev_from_uniform": 0.03},
                },
                "ppc": {"extreme_statistics": []},
            }
        ),
        encoding="utf-8",
    )
    (ev / "diagnostics.json").write_text(
        json.dumps(
            {
                "passed": converged,
                "rhat_max": 1.005 if converged else 1.3,
                "ess_bulk_min": 800 if converged else 40,
                "divergences": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"stage_durations": {"train": 120.0}, "resources": {}}), encoding="utf-8"
    )


def _fake_env(tmp_path, monkeypatch):
    """Fake pipeline: reference scores flat, every other arm beats it (finite z)."""
    counter = {"n": 0}
    ref_ll = np.zeros((2, 6))
    good_ll = np.log(np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), (2, 1)))

    def launch(config_path: Path, panelcast_bin: str) -> tuple[int, str]:
        counter["n"] += 1
        run_dir = tmp_path / "outputs" / f"run_{counter['n']:03d}"
        _write_scored_run(run_dir, ref_ll if counter["n"] == 1 else good_ll)
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_dir": run_dir.name}), encoding="utf-8"
        )
        return 0, "ok"

    import panelcast.paths as paths_mod

    def _latest(output_base=Path("outputs")):
        data = json.loads((tmp_path / "outputs" / "latest.json").read_text(encoding="utf-8"))
        return tmp_path / "outputs" / data["run_dir"]

    monkeypatch.setattr(paths_mod, "resolve_latest", _latest)
    return launch


class TestRunSelect:
    def test_produces_report_and_verdicts(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=4)
        cfg.include_stage2 = False
        result = run_select(None, STANDARD, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit")

        report_md = Path(result["report_dir"]) / "report.md"
        report_json = Path(result["report_dir"]) / "report.json"
        assert report_md.exists() and report_json.exists()
        text = report_md.read_text(encoding="utf-8")
        assert "panelcast select" in text
        assert "Promotion verdicts" in text
        assert result["n_arms_scored"] == 4

    def test_winner_recommended_when_bar_cleared(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=4)
        cfg.include_stage2 = False
        # good_ll vs ref gives z well above 2 with in-tolerance coverage + convergence.
        result = run_select(None, STANDARD, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit")
        assert result["winner_arm"] is not None
        assert result["promotable"]

    def test_report_json_lists_arms(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=3)
        cfg.include_stage2 = False
        result = run_select(None, STANDARD, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit")
        payload = json.loads((Path(result["report_dir"]) / "report.json").read_text())
        assert len(payload["arms"]) == 3
