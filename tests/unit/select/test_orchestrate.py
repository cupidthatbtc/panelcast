"""Orchestration: pre-run plan, cost prediction, and the full run_select flow."""

from __future__ import annotations

import json
from datetime import datetime
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
NOCONFIRM = EffortTier("screen", (1, 2), 4, 1000, 1000, confirm=False)


def _run_id_from_config(config_path) -> str:
    import yaml as _yaml

    return _yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["run_id"]


def _cfg(tmp_path, **kw) -> SweepConfig:
    cfg = tier_to_sweep_config(STANDARD, sweep_id="s", output_root=tmp_path / "select", **kw)
    cfg.pipeline_output_base = tmp_path / "outputs"
    return cfg


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

    def test_cost_predicted_with_rung_ladder(self, tmp_path):
        from panelcast.select.tiers import Rung

        tier = EffortTier(
            "standard", (1, 2), 4, 1000, 1000, confirm=True,
            rungs=(Rung(2, 500, 500, 0.4), Rung(4, 1000, 1000, None)),
        )
        dims = {"n_observations": 5000, "n_features": 40, "n_artists": 900, "max_seq": 30}
        plan = build_plan(
            AOTY, tier, _cfg(tmp_path), dataset_label="aoty", dims=dims,
            calibration_store_path=tmp_path / "cal.json",
        )
        assert plan.predicted_gpu_hours > 0
        assert any("rung ladder" in n for n in plan.notes)

    def test_cap_below_baseline_keeps_floor_le_ceiling(self, tmp_path):
        # --max-fits below the stage-1 count truncates mid-stage-1; the range
        # must not render backwards (floor > ceiling).
        plan = build_plan(AOTY, STANDARD, _cfg(tmp_path, max_fits=3), dataset_label="aoty")
        assert plan.min_fits <= plan.max_fits_planned == 3

    def test_auto_timeout_noted_in_plan(self, tmp_path):
        cfg = _cfg(tmp_path, arm_timeout_seconds="auto")
        plan = build_plan(AOTY, STANDARD, cfg, dataset_label="aoty")
        assert any("per-arm timeout: auto" in n for n in plan.notes)
        assert "per-arm timeout: auto" in render_plan(plan)

    def test_numeric_timeout_not_noted(self, tmp_path):
        cfg = _cfg(tmp_path, arm_timeout_seconds=1800.0)
        plan = build_plan(AOTY, STANDARD, cfg, dataset_label="aoty")
        assert not any("per-arm timeout" in n for n in plan.notes)


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


class TestResolveDims:
    def test_from_features_parquet(self, tmp_path):
        import pandas as pd

        from panelcast.select.orchestrate import resolve_dims

        feats = tmp_path / "f.parquet"
        pd.DataFrame(
            {"original_row_id": [0, 1, 2], "x": [1.0, 2, 3], "y": [4.0, 5, 6]}
        ).to_parquet(feats)
        dims = resolve_dims({"features": feats, "n_artists": 7})
        assert dims["n_observations"] == 3
        assert dims["n_features"] == 2  # original_row_id excluded
        assert dims["n_artists"] == 7

    def test_none_without_hint(self):
        from panelcast.select.orchestrate import resolve_dims

        assert resolve_dims(None) is None

    def test_missing_file_returns_none(self, tmp_path):
        from panelcast.select.orchestrate import resolve_dims

        assert resolve_dims({"features": tmp_path / "missing.parquet"}) is None

    def test_coarse_entity_estimate_without_hint(self, tmp_path):
        import pandas as pd

        from panelcast.select.orchestrate import resolve_dims

        feats = tmp_path / "f.parquet"
        pd.DataFrame({"a": range(50)}).to_parquet(feats)
        dims = resolve_dims({"features": feats})
        assert dims["n_artists"] == 10  # n_obs // 5


PUB_OVERRIDES = {"num_chains": 4, "num_samples": 5000, "num_warmup": 5000}
STANDARD_PUB = EffortTier(
    "standard", (1, 2), 4, 1000, 1000, confirm=True, publication_confirm=PUB_OVERRIDES
)


class TestPlanMatchesExecution:
    def test_stage2_bound_matches_runner_cap(self, tmp_path):
        # 1 composed + C(3, 2) pairwise for the capped winner set — not the old
        # hand-waved 6 that stage2_arms could silently exceed.
        plan = build_plan(AOTY, NOCONFIRM, _cfg(tmp_path), dataset_label="aoty")
        assert plan.max_fits_planned == plan.n_stage1_arms + 4
        assert any("stage 2" in n for n in plan.notes)
        assert "stage 2" in render_plan(plan)

    def test_no_phantom_publication_fits(self, tmp_path):
        # run_confirmation runs ALL 2xN confirmation fits at publication scale;
        # there is no separate 2-fit publication pass to price.
        plan = build_plan(
            AOTY, STANDARD_PUB, _cfg(tmp_path), dataset_label="aoty", n_confirmation_seeds=3
        )
        assert plan.min_fits == plan.n_stage1_arms + 6
        assert plan.max_fits_planned == plan.n_stage1_arms + 4 + 6

    def test_cost_prices_confirmation_at_publication_scale(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        import panelcast.gpu_memory.calibration_store as cs
        import panelcast.gpu_memory.runtime_predictor as rp

        def fake_predict(num_chains, num_samples, num_warmup, n_obs, transform, store_path):
            seconds = 1000.0 if num_samples == 5000 else 100.0
            return SimpleNamespace(seconds=seconds, source="fake")

        monkeypatch.setattr(rp, "predict_fit_seconds", fake_predict)
        monkeypatch.setattr(
            cs,
            "estimate_with_calibration",
            lambda *a, **k: (SimpleNamespace(total_gb=8.0), "fake-mem"),
        )
        dims = {"n_observations": 5000, "n_features": 40, "n_artists": 900, "max_seq": 30}
        plan = build_plan(
            AOTY, STANDARD_PUB, _cfg(tmp_path), dataset_label="aoty",
            n_confirmation_seeds=3, dims=dims,
        )
        n_screen = plan.n_stage1_arms + 4
        expected_hours = (n_screen * 100.0 + 6 * 1000.0) / 3600.0
        assert plan.predicted_gpu_hours == expected_hours

    def test_confirmation_launch_count_matches_plan(self, tmp_path, monkeypatch):
        import yaml

        import panelcast.paths as paths_mod
        from panelcast.select.confirmation import run_confirmation

        monkeypatch.setattr(paths_mod, "resolve_latest", lambda output_base=Path("outputs"): None)
        launched: list[Path] = []

        def launch(config_path, panelcast_bin, timeout_seconds=None):
            launched.append(Path(config_path))
            return 0, "ok"

        cfg = _cfg(tmp_path)
        plan = build_plan(
            AOTY, STANDARD_PUB, cfg, dataset_label="aoty", n_confirmation_seeds=3
        )
        run_confirmation(
            {"latent_process": "ar1"}, cfg, seeds=(42, 43, 44),
            sampler_overrides=STANDARD_PUB.publication_confirm, launch=launch,
        )
        planned_confirm = plan.max_fits_planned - (plan.n_stage1_arms + 4)
        assert len(launched) == planned_confirm == 6
        for path in launched:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert payload["num_samples"] == 5000  # every fit at publication scale


class TestThoroughCost:
    def test_publication_fit_included(self, tmp_path):
        thorough = EffortTier(
            "thorough", (1, 2, 3), 4, 1000, 1000, stage3_fits=8, confirm=True,
            publication_confirm={"num_chains": 4, "num_samples": 5000, "num_warmup": 5000},
        )
        cfg = tier_to_sweep_config(thorough, "s", output_root=tmp_path / "s")
        dims = {"n_observations": 5000, "n_features": 40, "n_artists": 900, "max_seq": 30}
        plan = build_plan(
            AOTY, thorough, cfg, dataset_label="aoty", dims=dims,
            calibration_store_path=tmp_path / "c.json",
        )
        assert plan.predicted_gpu_hours > 0


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
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "stage_durations": {"train": 120.0},
                "resources": {},
            }
        ),
        encoding="utf-8",
    )


def _fake_env(tmp_path, monkeypatch):
    """Fake pipeline: reference scores flat, every other arm beats it (finite z)."""
    counter = {"n": 0}
    ref_ll = np.zeros((2, 6))
    good_ll = np.log(np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), (2, 1)))

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        counter["n"] += 1
        run_dir = tmp_path / "outputs" / _run_id_from_config(config_path)
        _write_scored_run(run_dir, ref_ll if counter["n"] == 1 else good_ll)
        return 0, "ok"

    return launch


class TestRunSelect:
    def test_produces_report_and_verdicts(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=4)
        cfg.include_stage2 = False
        result = run_select(None, NOCONFIRM, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit")

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
        result = run_select(None, NOCONFIRM, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit")
        assert result["winner_arm"] is not None
        assert result["promotable"]

    def test_report_json_lists_arms(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=3)
        cfg.include_stage2 = False
        result = run_select(None, NOCONFIRM, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit")
        payload = json.loads((Path(result["report_dir"]) / "report.json").read_text())
        assert len(payload["arms"]) == 3

    def test_scoring_failure_records_unscored_arm(self, tmp_path, monkeypatch):
        """A bad snapshot in the post-sweep scoring loop must not crash the
        report after the whole sweep is paid for: the arm is recorded unscored
        with the failure note, and the report still renders."""
        import panelcast.select.scoring as scoring_mod

        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=3)
        cfg.include_stage2 = False

        def _boom(*args, **kwargs):
            raise ValueError("corrupt snapshot")

        monkeypatch.setattr(scoring_mod, "score_arm", _boom)
        result = run_select(
            None, NOCONFIRM, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit"
        )

        report_json = Path(result["report_dir"]) / "report.json"
        assert report_json.exists()
        payload = json.loads(report_json.read_text())
        assert payload["arms"], "unscored arms must still be listed"
        assert all(
            any("scoring failed" in note for note in arm["notes"]) for arm in payload["arms"]
        )
        assert result["winner_arm"] is None
        assert not result["promotable"]

    def test_ladder_run_writes_screening_appendix(self, tmp_path, monkeypatch):
        import panelcast.select.runner as runner_mod

        launch = _fake_env(tmp_path, monkeypatch)
        monkeypatch.setattr(
            runner_mod, "ofat_arms",
            lambda d, available_columns=None: [
                ({"latent_process": "ar1"}, None),
                ({"ar_center": "none"}, None),
            ],
        )
        cfg = _cfg(tmp_path, max_fits=8)
        cfg.include_stage2 = False
        cfg.rungs = [
            {"num_chains": 2, "num_samples": 500, "num_warmup": 500, "keep_fraction": 0.5},
            {"num_chains": 4, "num_samples": 1000, "num_warmup": 1000},
        ]
        result = run_select(
            None, NOCONFIRM, DecisionRules(), cfg, launch=launch, audit_root=tmp_path / ".audit"
        )
        text = (Path(result["report_dir"]) / "report.md").read_text(encoding="utf-8")
        assert "Screening rungs (appendix)" in text
        payload = json.loads((Path(result["report_dir"]) / "report.json").read_text())
        assert payload["screening_rungs"]
        assert all(r["rung"] == 0 for r in payload["screening_rungs"])

    def test_prior_screen_block_when_frame_given(self, tmp_path, monkeypatch):
        import pandas as pd

        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=2)
        cfg.include_stage2 = False
        rng = np.random.default_rng(0)
        rows = []
        for a in range(6):
            base = rng.normal(72, 6)
            for _ in range(5):
                rows.append(
                    {
                        "Artist": f"a{a}",
                        "User_Score": float(np.clip(base + rng.normal(0, 4), 0, 100)),
                        "User_Ratings": int(rng.integers(20, 400)),
                        "f_one": rng.normal(),
                        "f_two": rng.normal(),
                    }
                )
        train_df = pd.DataFrame(rows)
        result = run_select(
            None, NOCONFIRM, DecisionRules(), cfg, train_df=train_df,
            feature_cols=["f_one", "f_two"], launch=launch, audit_root=tmp_path / ".audit",
        )
        assert (Path(result["report_dir"]) / "prior_screen.json").exists()
        assert "Prior-predictive screen" in (Path(result["report_dir"]) / "report.md").read_text()


def _nonconverged_env(tmp_path, monkeypatch):
    """Every non-reference arm beats the reference but FAILS the convergence gate,
    so it is screenable yet held by the strict per-arm verdict."""
    counter = {"n": 0}
    ref_ll = np.zeros((2, 6))
    good_ll = np.log(np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), (2, 1)))

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        counter["n"] += 1
        run_dir = tmp_path / "outputs" / _run_id_from_config(config_path)
        is_ref = counter["n"] == 1
        _write_scored_run(run_dir, ref_ll if is_ref else good_ll, converged=is_ref)
        return 0, "ok"

    return launch


class TestScreeningCandidate:
    def test_nonconverged_screenable_arm_confirmed_at_publication_scale(
        self, tmp_path, monkeypatch
    ):
        import panelcast.select.confirmation as confirm_mod

        launch = _nonconverged_env(tmp_path, monkeypatch)
        captured: dict = {}

        class _Result:
            confirmed = True

            def to_dict(self):
                return {"confirmed": True}

        def fake_confirm(winner_knobs, cfg, seeds, promote_z, sampler_overrides, launch, dims=None):
            captured["overrides"] = sampler_overrides
            captured["knobs"] = winner_knobs
            return _Result()

        monkeypatch.setattr(confirm_mod, "run_confirmation", fake_confirm)
        monkeypatch.setattr(confirm_mod, "render_confirmation", lambda r: "confirmation block\n")

        tier = EffortTier(
            "standard", (1, 2), 4, 1000, 1000, confirm=True,
            publication_confirm={"num_chains": 4, "num_samples": 5000, "num_warmup": 5000},
        )
        cfg = _cfg(tmp_path, max_fits=3)
        cfg.include_stage2 = False
        result = run_select(
            None, tier, DecisionRules(confirmation_seeds=(42,)), cfg,
            launch=launch, audit_root=tmp_path / ".audit",
        )
        # No arm converged, so the strict verdicts promote nobody...
        assert result["promotable"] == []
        # ...yet a screenable candidate was confirmed at publication scale (5000).
        assert captured["overrides"] == {"num_chains": 4, "num_samples": 5000, "num_warmup": 5000}
        assert captured["knobs"]  # a real, non-reference arm was chosen
        assert result["confirmed"] is True
        assert result["winner_arm"] is not None


def _confirm_env(tmp_path, monkeypatch):
    """Fake pipeline that distinguishes reference vs winner fits by config name,
    so the multi-seed confirmation resolves deterministically to confirmed."""
    state = {"n": 0, "sweep_seen": False}
    ref_ll = np.zeros((2, 6))
    good_ll = np.log(np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), (2, 1)))

    def launch(config_path: Path, panelcast_bin: str, timeout_seconds=None) -> tuple[int, str]:
        state["n"] += 1
        name = Path(config_path).stem
        if "confirm_reference" in name:
            ll = ref_ll
        elif "confirm_winner" in name:
            ll = good_ll
        elif not state["sweep_seen"]:
            ll = ref_ll
            state["sweep_seen"] = True
        else:
            ll = good_ll
        run_dir = tmp_path / "outputs" / _run_id_from_config(config_path)
        _write_scored_run(run_dir, ll)
        return 0, "ok"

    return launch


class TestConfirmationWiring:
    def test_confirmed_winner_recommended(self, tmp_path, monkeypatch):
        launch = _confirm_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=2)
        cfg.include_stage2 = False
        result = run_select(
            None, STANDARD, DecisionRules(confirmation_seeds=(42, 43)), cfg,
            launch=launch, audit_root=tmp_path / ".audit",
        )
        assert result["confirmed"] is True
        assert result["winner_arm"] is not None
        report = (Path(result["report_dir"]) / "report.md").read_text()
        assert "Multi-seed confirmation" in report
        assert "CONFIRMED" in report
        assert (cfg.sweep_dir / "confirmation.json").exists()

    def test_unconfirmed_winner_not_recommended(self, tmp_path, monkeypatch):
        # Winner passes the sweep rules but the confirmation fits come back flat
        # (reference == winner), so it must NOT be recommended.
        state = {"n": 0, "sweep_seen": False}
        ref_ll = np.zeros((2, 6))
        good_ll = np.log(np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), (2, 1)))

        def launch(config_path, panelcast_bin, timeout_seconds=None):
            state["n"] += 1
            name = Path(config_path).stem
            if name.startswith("confirm_"):
                ll = ref_ll  # both reference and winner flat -> not confirmed
            elif not state["sweep_seen"]:
                ll = ref_ll
                state["sweep_seen"] = True
            else:
                ll = good_ll
            run_dir = tmp_path / "outputs" / _run_id_from_config(config_path)
            _write_scored_run(run_dir, ll)
            return 0, "ok"

        cfg = _cfg(tmp_path, max_fits=2)
        cfg.include_stage2 = False
        result = run_select(
            None, STANDARD, DecisionRules(confirmation_seeds=(42,)), cfg,
            launch=launch, audit_root=tmp_path / ".audit",
        )
        assert result["confirmed"] is False
        assert result["winner_arm"] is None
        assert result["promotable"]  # it cleared the rules, just not confirmation


class TestReferenceBaselineRow:
    def test_reference_scores_zero_not_missing_snapshot(self, tmp_path, monkeypatch):
        # The reference paired against its own snapshot is the z=0 baseline row,
        # not a "no snapshot" caveat sinking below every scored arm.
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=3)
        cfg.include_stage2 = False
        result = run_select(
            None, NOCONFIRM, DecisionRules(), cfg, launch=launch,
            audit_root=tmp_path / ".audit",
        )
        report = (Path(result["report_dir"]) / "report.md").read_text(encoding="utf-8")
        assert "No pointwise log-likelihood snapshot" not in report
        payload = json.loads((Path(result["report_dir"]) / "report.json").read_text())
        reference = next(a for a in payload["arms"] if a["knobs"] == {})
        assert reference["elpd_z"] == 0.0
        assert reference["elpd_diff"] == 0.0
        # z=0 ranks with the scored arms, below the genuine winners.
        assert payload["arms"][-1]["knobs"] == {}


class TestNotEvaluatedSection:
    def test_failed_arms_surface_in_report_and_summary(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        calls = {"n": 0}

        def flaky(config_path, panelcast_bin, timeout_seconds=None):
            calls["n"] += 1
            if calls["n"] == 3:
                return 1, "cuda OOM: out of memory"
            return launch(config_path, panelcast_bin, timeout_seconds)

        cfg = _cfg(tmp_path, max_fits=4)
        cfg.include_stage2 = False
        result = run_select(
            None, NOCONFIRM, DecisionRules(), cfg, launch=flaky,
            audit_root=tmp_path / ".audit",
        )
        assert result["n_arms_not_evaluated"] == 1
        assert result["not_evaluated"] == {"failed": 1}
        report = (Path(result["report_dir"]) / "report.md").read_text(encoding="utf-8")
        assert "## Not evaluated" in report
        assert "cuda OOM" in report
        payload = json.loads((Path(result["report_dir"]) / "report.json").read_text())
        assert len(payload["not_evaluated"]) == 1
        assert payload["not_evaluated"][0]["status"] == "failed"
        assert payload["not_evaluated"][0]["knobs"]

    def test_clean_sweep_has_no_section_but_zero_counts(self, tmp_path, monkeypatch):
        launch = _fake_env(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, max_fits=2)
        cfg.include_stage2 = False
        result = run_select(
            None, NOCONFIRM, DecisionRules(), cfg, launch=launch,
            audit_root=tmp_path / ".audit",
        )
        assert result["n_arms_not_evaluated"] == 0
        report = (Path(result["report_dir"]) / "report.md").read_text(encoding="utf-8")
        assert "## Not evaluated" not in report
