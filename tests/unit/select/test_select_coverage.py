"""Coverage hardening for the merged select modules: subprocess launch, sweep
stage-2/3 and budget edges, scoring error paths, and the baseline-floor report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.select.confirmation import ConfirmationResult, SeedResult
from panelcast.select.runner import (
    ArmRecord,
    SweepConfig,
    _default_panelcast_bin,
    _write_arm_config,
    launch_arm,
    ofat_arms,
    run_sweep,
)
from panelcast.select.scoring import ArmScore, _load_json, render_report

AOTY = DatasetDescriptor()


# --- runner: subprocess launch + config plumbing ---------------------------
class TestLaunchArm:
    def test_sets_save_log_likelihood_and_invokes_bin(self, monkeypatch, tmp_path):
        import subprocess as sp

        captured = {}

        class _Proc:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(cmd, capture_output, text, env, timeout=None):
            captured["cmd"] = cmd
            captured["env"] = env
            return _Proc()

        monkeypatch.setattr(sp, "run", fake_run)
        code, tail = launch_arm(tmp_path / "arm.yaml", "panelcast")
        assert code == 0
        assert captured["env"]["PANELCAST_SAVE_LOG_LIKELIHOOD"] == "1"
        assert captured["cmd"][:2] == ["panelcast", "run"]

    def test_returns_nonzero_and_tail_on_failure(self, monkeypatch, tmp_path):
        import subprocess as sp

        class _Proc:
            returncode = 2
            stdout = "boom-out"
            stderr = "boom-err"

        monkeypatch.setattr(sp, "run", lambda *a, **k: _Proc())
        code, tail = launch_arm(tmp_path / "arm.yaml", "panelcast")
        assert code == 2
        assert "boom-err" in tail

    def test_default_bin_resolves(self):
        assert isinstance(_default_panelcast_bin(), str)


class TestWriteArmConfig:
    def test_dataset_written_when_set(self, tmp_path):
        cfg = SweepConfig(sweep_id="s", dataset="aero", num_samples=250)
        path = tmp_path / "arm.yaml"
        _write_arm_config(cfg, {"latent_process": "ar1"}, ["train", "evaluate"], path)
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["dataset"] == "aero"
        assert payload["num_samples"] == 250
        assert payload["stages"] == ["train", "evaluate"]


def _fake_env(tmp_path, monkeypatch):
    counter = {"n": 0}

    def launch(config_path, panelcast_bin, timeout_seconds=None):
        counter["n"] += 1
        run_dir = tmp_path / "outputs" / f"run_{counter['n']:03d}"
        run_dir.mkdir(parents=True)
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_dir": run_dir.name}), encoding="utf-8"
        )
        return 0, "ok"

    import panelcast.paths as paths_mod

    monkeypatch.setattr(
        paths_mod,
        "resolve_latest",
        lambda output_base=Path("outputs"): tmp_path
        / "outputs"
        / json.loads((tmp_path / "outputs" / "latest.json").read_text())["run_dir"],
    )
    return SweepConfig(sweep_id="s", output_root=tmp_path / "select"), launch


class TestSweepEdges:
    def test_diagnostics_written_with_train_df(self, tmp_path, monkeypatch):
        import pandas as pd

        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 2
        df = pd.DataFrame(
            {
                "Artist": [f"a{i % 5}" for i in range(40)],
                "User_Score": np.random.default_rng(0).normal(70, 8, 40),
                "User_Ratings": np.full(40, 100),
            }
        )
        run_sweep(cfg, AOTY, train_df=df, launch=launch)
        assert (cfg.sweep_dir / "diagnostics.json").exists()

    def test_budget_hours_truncates_on_resume(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.budget_hours = 0.5
        # A prior session already burned the budget: the first check truncates.
        cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
        (cfg.sweep_dir / "ledger.json").write_text(
            json.dumps(
                {"arms": [
                    {"arm_id": "prior", "knobs": {"latent_process": "ar1"}, "stage": 1,
                     "status": "completed", "run_dir": None, "wall_clock_seconds": 3600.0,
                     "error": None, "score": None, "note": None}
                ]}
            ),
            encoding="utf-8",
        )
        launches_before = []

        def counting_launch(config_path, panelcast_bin, timeout_seconds=None):
            launches_before.append(config_path)
            return launch(config_path, panelcast_bin, timeout_seconds)

        run_sweep(cfg, AOTY, launch=counting_launch)
        assert launches_before == []  # truncated before any new fit

    def test_scorer_exception_recorded_not_raised(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.max_fits = 2

        def boom(run_dir, reference_run):
            raise RuntimeError("scorer down")

        ledger = run_sweep(cfg, AOTY, launch=launch, scorer=boom)
        assert any("scoring failed" in (r.note or "") for r in ledger.records.values())
        assert all(r.status == "completed" for r in ledger.records.values())

    def test_stage3_runs(self, tmp_path, monkeypatch):
        cfg, launch = _fake_env(tmp_path, monkeypatch)
        cfg.include_stage2 = False
        cfg.stage3_fits = 3
        cfg.max_fits = 1 + len(ofat_arms(AOTY)) + 3
        ledger = run_sweep(cfg, AOTY, launch=launch)
        assert any(r.stage == 3 for r in ledger.records.values())


# --- scoring: error paths + baseline floor ---------------------------------
class TestLoadJsonErrors:
    def test_unreadable_json_noted(self, tmp_path):
        bad = tmp_path / "b.json"
        bad.write_text("{not valid", encoding="utf-8")
        notes: list[str] = []
        assert _load_json(bad, notes, "metrics.json") == {}
        assert any("unreadable" in n for n in notes)

    def test_non_object_json_noted(self, tmp_path):
        arr = tmp_path / "a.json"
        arr.write_text("[1, 2, 3]", encoding="utf-8")
        notes: list[str] = []
        assert _load_json(arr, notes, "metrics.json") == {}
        assert any("not a JSON object" in n for n in notes)


class TestBaselineFloor:
    def _scores(self):
        return [ArmScore(arm="a1", elpd_diff=5.0, elpd_dse=1.0, elpd_z=5.0)]

    def test_beats_gbm(self):
        block = {"rows": [
            {"model": "bayes", "split": "within", "mae": 5.1},
            {"model": "gbm", "split": "within", "mae": 5.4},
        ]}
        md, payload = render_report(self._scores(), "reference", baseline_block=block)
        assert "Baseline floor" in md
        assert "beats the GBM" in md
        assert payload["baseline_floor"] is not None

    def test_does_not_beat_gbm(self):
        block = {"rows": [
            {"model": "bayes", "split": "within", "mae": 5.9},
            {"model": "gbm", "split": "within", "mae": 5.4},
        ]}
        md, _ = render_report(self._scores(), "reference", baseline_block=block)
        assert "does not beat the GBM" in md

    def test_empty_baseline_rows(self):
        md, _ = render_report(self._scores(), "reference", baseline_block={"rows": []})
        assert "No baseline rows provided" in md

    def test_no_bayes_row_no_conclusion(self):
        block = {"rows": [{"model": "gbm", "split": "within", "mae": 5.4}]}
        md, _ = render_report(self._scores(), "reference", baseline_block=block)
        assert "Baseline floor" in md
        assert "the GBM floor on MAE" not in md

    def test_no_gbm_row_no_conclusion(self):
        block = {"rows": [{"model": "bayes", "split": "within", "mae": 5.1}]}
        md, _ = render_report(self._scores(), "reference", baseline_block=block)
        assert "Baseline floor" in md
        assert "the GBM floor on MAE" not in md

    def test_losing_top_arm_verdict(self):
        scores = [ArmScore(arm="a1", elpd_diff=-3.0, elpd_dse=1.0, elpd_z=-3.0)]
        md, _ = render_report(scores, "reference")
        assert "No arm beats" in md


class TestScoreArmManifest:
    def test_duration_seconds_fallback(self, tmp_path):
        from panelcast.select.scoring import score_arm

        run_dir = tmp_path / "run"
        (run_dir / "evaluation").mkdir(parents=True)
        (run_dir / "evaluation" / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "evaluation" / "diagnostics.json").write_text(
            json.dumps({"passed": True, "rhat_max": 1.0, "ess_bulk_min": 900, "divergences": 0}),
            encoding="utf-8",
        )
        # No resources.train wall-clock and no stage_durations.train: fall back
        # to the manifest's top-level duration_seconds.
        (run_dir / "manifest.json").write_text(
            json.dumps({"resources": {}, "stage_durations": {}, "duration_seconds": 314.0}),
            encoding="utf-8",
        )
        score = score_arm(run_dir, arm="a1")
        assert score.wall_clock_seconds == 314.0


# --- confirmation: the confirmed-property edges ----------------------------
class TestConfirmedProperty:
    def test_all_seeds_hold(self):
        result = ConfirmationResult(
            winner_knobs={},
            seeds=[SeedResult(seed=s, elpd={"z": 5.0}) for s in (42, 43)],
            promote_z=2.0,
        )
        assert result.confirmed

    def test_one_seed_below_threshold(self):
        result = ConfirmationResult(
            winner_knobs={},
            seeds=[
                SeedResult(seed=42, elpd={"z": 5.0}),
                SeedResult(seed=43, elpd={"z": 0.5}),
            ],
            promote_z=2.0,
        )
        assert not result.confirmed

    def test_null_z_blocks(self):
        result = ConfirmationResult(
            winner_knobs={}, seeds=[SeedResult(seed=42, elpd={"z": None})], promote_z=2.0
        )
        assert not result.confirmed


class TestConfirmationConfig:
    def test_dataset_and_sampler_written(self, tmp_path, monkeypatch):
        from panelcast.select.confirmation import run_confirmation

        counter = {"n": 0}

        def launch(config_path, panelcast_bin):
            counter["n"] += 1
            run_dir = tmp_path / "outputs" / f"r{counter['n']}"
            ev = run_dir / "evaluation"
            ev.mkdir(parents=True)
            import arviz as az
            import xarray as xr

            ll = np.zeros((1, 2, 3)) if counter["n"] % 2 else np.log(
                np.array([[[2.0, 3.0, 4.0]]])
            )
            da = xr.DataArray(
                ll, dims=["chain", "draw", "obs"],
                coords={"chain": [0], "draw": range(ll.shape[1]), "obs": range(3)},
            )
            az.InferenceData(log_likelihood=xr.Dataset({"y": da})).to_netcdf(
                str(ev / "log_likelihood.nc")
            )
            (tmp_path / "outputs" / "latest.json").write_text(
                json.dumps({"run_dir": run_dir.name}), encoding="utf-8"
            )
            return 0, "ok"

        import panelcast.paths as paths_mod

        monkeypatch.setattr(
            paths_mod, "resolve_latest",
            lambda output_base=Path("outputs"): tmp_path / "outputs" / json.loads(
                (tmp_path / "outputs" / "latest.json").read_text()
            )["run_dir"],
        )
        cfg = SweepConfig(
            sweep_id="c", dataset="aero", output_root=tmp_path / "select",
            num_chains=4, num_samples=1000, num_warmup=1000,
        )
        run_confirmation({"latent_process": "ar1"}, cfg, seeds=(42,), launch=launch)
        import yaml

        payload = yaml.safe_load(
            (cfg.sweep_dir / "confirm_reference_seed42.yaml").read_text(encoding="utf-8")
        )
        assert payload["dataset"] == "aero"
        assert payload["num_samples"] == 1000
