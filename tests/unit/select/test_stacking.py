"""Predictive stacking over the arm ledger (#154): weights, mixture, report."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pytest
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.select.stacking import (
    allocate_mixture_draws,
    load_predictive_snapshot,
    load_stack_arms,
    mixture_predictive,
    pseudo_bma_plus_weights,
    render_stack_report,
    run_stack,
    score_predictive,
    stacking_weights,
)

runner = CliRunner()


def _const_elpd(values: list[float]) -> np.ndarray:
    """(n_arms, n_obs) matrix where each arm has a constant per-obs elpd."""
    return np.array(values)


class TestStackingWeights:
    def test_single_arm_gets_full_weight(self):
        assert stacking_weights(np.full((1, 10), -2.0)).tolist() == [1.0]

    def test_dominant_arm_takes_the_weight(self):
        elpd = np.vstack([np.full(20, -1.0), np.full(20, -5.0)])
        w = stacking_weights(elpd)
        assert w.shape == (2,)
        assert w[0] > 0.95
        assert w.sum() == pytest.approx(1.0)

    def test_complementary_arms_split_the_weight(self):
        # Arm 0 is good on the first half, arm 1 on the second: the optimal
        # mixture is interior — the whole point of stacking over selection.
        a = np.concatenate([np.full(10, -1.0), np.full(10, -5.0)])
        elpd = np.vstack([a, a[::-1]])
        w = stacking_weights(elpd)
        assert w[0] == pytest.approx(0.5, abs=0.05)
        assert w[1] == pytest.approx(0.5, abs=0.05)

    def test_deterministic(self):
        rng = np.random.default_rng(7)
        elpd = rng.normal(-2.0, 1.0, size=(4, 30))
        np.testing.assert_array_equal(stacking_weights(elpd), stacking_weights(elpd))

    def test_rejects_bad_shapes_and_nonfinite(self):
        with pytest.raises(ValueError, match="n_arms, n_obs"):
            stacking_weights(np.zeros(5))
        with pytest.raises(ValueError, match="non-finite"):
            stacking_weights(np.array([[0.0, np.nan]]))


class TestPseudoBmaPlus:
    def test_simplex_and_ordering(self):
        elpd = np.vstack([np.full(20, -1.0), np.full(20, -3.0)])
        w = pseudo_bma_plus_weights(elpd, seed=0)
        assert w.sum() == pytest.approx(1.0)
        assert w[0] > w[1]

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(3)
        elpd = rng.normal(-2.0, 1.0, size=(3, 25))
        np.testing.assert_array_equal(
            pseudo_bma_plus_weights(elpd, seed=11), pseudo_bma_plus_weights(elpd, seed=11)
        )

    def test_single_arm(self):
        assert pseudo_bma_plus_weights(np.full((1, 5), -1.0)).tolist() == [1.0]


class TestMixture:
    def test_allocation_is_exact_and_deterministic(self):
        counts = allocate_mixture_draws(np.array([0.5, 0.3, 0.2]), 10)
        assert counts.tolist() == [5, 3, 2]
        counts = allocate_mixture_draws(np.array([0.55, 0.45]), 9)
        assert counts.sum() == 9
        assert counts.tolist() == [5, 4]

    def test_zero_weight_contributes_nothing(self):
        counts = allocate_mixture_draws(np.array([1.0, 0.0]), 7)
        assert counts.tolist() == [7, 0]

    def test_mixture_shape_and_content(self):
        draws_a = np.full((10, 3), 1.0)
        draws_b = np.full((20, 3), 2.0)
        mix = mixture_predictive([draws_a, draws_b], np.array([0.5, 0.5]))
        # Sized at the smallest snapshot (10): 5 draws from each arm.
        assert mix.shape == (10, 3)
        assert (mix == 1.0).sum() == 15
        assert (mix == 2.0).sum() == 15

    def test_requires_one_weight_per_arm(self):
        with pytest.raises(ValueError, match="per weight"):
            mixture_predictive([np.zeros((5, 2))], np.array([0.5, 0.5]))


class TestScorePredictive:
    def test_all_metrics_present_and_finite(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(70, 5, size=40)
        draws = y_true[None, :] + rng.normal(0, 2, size=(200, 40))
        scores = score_predictive(y_true, draws)
        for key in ("crps", "mae", "rmse", "r2", "cov80", "cov95", "wis"):
            assert np.isfinite(scores[key]), key
        assert 0.5 < scores["cov95"] <= 1.0


def _write_run(
    run_dir: Path,
    elpd_value: float,
    n_obs: int = 12,
    predictive: bool = True,
    center: float = 70.0,
) -> None:
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    arr = np.full((2, 50, n_obs), elpd_value)
    az.from_dict(log_likelihood={"y": arr}).to_netcdf(str(eval_dir / "log_likelihood.nc"))
    if predictive:
        rng = np.random.default_rng(int(center))
        y_true = np.linspace(60, 80, n_obs).astype(np.float32)
        arrays = {}
        for split in ("primary", "secondary"):
            arrays[f"{split}_draws"] = (
                y_true[None, :] + rng.normal(center - 70.0, 2.0, size=(100, n_obs))
            ).astype(np.float32)
            arrays[f"{split}_y_true"] = y_true
        np.savez_compressed(eval_dir / "predictive.npz", **arrays)


def _write_sweep(tmp_path: Path, arms: list[dict]) -> Path:
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "ledger.json").write_text(json.dumps({"arms": arms}), encoding="utf-8")
    return sweep_dir


def _make_sweep(tmp_path: Path, predictive: bool = True) -> Path:
    ref_run = tmp_path / "runs" / "ref"
    arm_run = tmp_path / "runs" / "armb"
    _write_run(ref_run, elpd_value=-2.0, predictive=predictive, center=70.0)
    _write_run(arm_run, elpd_value=-1.5, predictive=predictive, center=71.0)
    return _write_sweep(
        tmp_path,
        [
            {"arm_id": "ref", "knobs": {}, "stage": 1, "status": "completed",
             "run_dir": str(ref_run)},
            {"arm_id": "armb", "knobs": {"target_transform": "logit"}, "stage": 1,
             "status": "completed", "run_dir": str(arm_run)},
            {"arm_id": "dead", "knobs": {"likelihood_family": "student_t"}, "stage": 1,
             "status": "timeout", "run_dir": None},
        ],
    )


def _make_ladder_sweep(tmp_path: Path) -> Path:
    """A 2-rung ladder (#164): reference + one promoted arm, each with a
    completed record per rung — screening records must never reach the stack."""
    runs = tmp_path / "runs"
    _write_run(runs / "ref_r0", elpd_value=-3.0, center=69.0)
    _write_run(runs / "armb_r0", elpd_value=-2.5, center=72.0)
    _write_run(runs / "ref_r1", elpd_value=-2.0, center=70.0)
    _write_run(runs / "armb_r1", elpd_value=-1.5, center=71.0)
    return _write_sweep(
        tmp_path,
        [
            {"arm_id": "ref", "knobs": {}, "stage": 1, "status": "completed",
             "run_dir": str(runs / "ref_r0"), "rung": 0},
            {"arm_id": "armb", "knobs": {"target_transform": "logit"}, "stage": 1,
             "status": "completed", "run_dir": str(runs / "armb_r0"), "rung": 0},
            {"arm_id": "ref", "knobs": {}, "stage": 1, "status": "completed",
             "run_dir": str(runs / "ref_r1"), "rung": 1},
            {"arm_id": "armb", "knobs": {"target_transform": "logit"}, "stage": 1,
             "status": "completed", "run_dir": str(runs / "armb_r1"), "rung": 1},
        ],
    )


class TestLoadStackArms:
    def test_loads_completed_arms_and_excludes_the_rest(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        arms, excluded = load_stack_arms(sweep_dir)
        assert [a.arm_id for a in arms] == ["ref", "armb"]
        assert arms[0].total_elpd == pytest.approx(-2.0 * 12)
        assert ("dead", "status timeout") in excluded

    def test_missing_snapshot_is_excluded_not_substituted(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        bare_run = tmp_path / "runs" / "bare"
        (bare_run / "evaluation").mkdir(parents=True)
        arms_json = json.loads((sweep_dir / "ledger.json").read_text())["arms"]
        arms_json.append(
            {"arm_id": "bare", "knobs": {"x": 1}, "stage": 1, "status": "completed",
             "run_dir": str(bare_run)}
        )
        _write_sweep(tmp_path, arms_json)
        arms, excluded = load_stack_arms(sweep_dir)
        assert [a.arm_id for a in arms] == ["ref", "armb"]
        assert ("bare", "no pointwise log_likelihood snapshot") in excluded

    def test_obs_dimension_mismatch_excluded(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        odd_run = tmp_path / "runs" / "odd"
        _write_run(odd_run, elpd_value=-1.0, n_obs=5)
        arms_json = json.loads((sweep_dir / "ledger.json").read_text())["arms"]
        arms_json.append(
            {"arm_id": "odd", "knobs": {"y": 2}, "stage": 1, "status": "completed",
             "run_dir": str(odd_run)}
        )
        _write_sweep(tmp_path, arms_json)
        arms, excluded = load_stack_arms(sweep_dir)
        assert [a.arm_id for a in arms] == ["ref", "armb"]
        assert any(aid == "odd" and "differs" in reason for aid, reason in excluded)

    def test_ladder_sweep_stacks_only_final_rung(self, tmp_path):
        sweep_dir = _make_ladder_sweep(tmp_path)
        arms, excluded = load_stack_arms(sweep_dir)
        assert [a.arm_id for a in arms] == ["ref", "armb"]
        assert {a.run_dir.name for a in arms} == {"ref_r1", "armb_r1"}
        reasons = [reason for _, reason in excluded]
        assert sum("screening rung 0" in r for r in reasons) == 2

    def test_sweep_config_rungs_is_authoritative(self, tmp_path):
        """An interrupted ladder with no completed final-rung records must not
        silently fall back to stacking screening fits."""
        sweep_dir = _make_ladder_sweep(tmp_path)
        arms_json = [
            e for e in json.loads((sweep_dir / "ledger.json").read_text())["arms"]
            if e.get("rung", 0) == 0
        ]
        _write_sweep(tmp_path, arms_json)
        (sweep_dir / "sweep_config.json").write_text(
            json.dumps({"rungs": [{"num_samples": 500}, {"num_samples": 1000}]}),
            encoding="utf-8",
        )
        arms, excluded = load_stack_arms(sweep_dir)
        assert arms == []
        assert all("screening rung 0 (final is 1)" in reason for _, reason in excluded)

    def test_malformed_sweep_config_falls_back_to_ledger_rungs(self, tmp_path):
        sweep_dir = _make_ladder_sweep(tmp_path)
        (sweep_dir / "sweep_config.json").write_text("{not json", encoding="utf-8")
        arms, _ = load_stack_arms(sweep_dir)
        assert [a.arm_id for a in arms] == ["ref", "armb"]
        assert {a.run_dir.name for a in arms} == {"ref_r1", "armb_r1"}

    def test_missing_ledger_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stack_arms(tmp_path)


class TestLoadPredictiveSnapshot:
    def test_roundtrip(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_run(run_dir, elpd_value=-1.0)
        snap = load_predictive_snapshot(run_dir)
        assert set(snap) == {"primary", "secondary"}
        draws, y_true = snap["primary"]
        assert draws.shape == (100, 12)
        assert y_true.shape == (12,)

    def test_absent_file_is_empty(self, tmp_path):
        assert load_predictive_snapshot(tmp_path) == {}


class TestRunStack:
    def test_end_to_end_writes_report_with_honest_headline(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        result = run_stack(sweep_dir)
        assert result["n_arms_stacked"] == 2
        assert result["n_excluded"] == 1
        report = (sweep_dir / "stacking.md").read_text(encoding="utf-8")
        assert "## Weights" in report
        assert "honest headline" in report
        assert "secondary" in result["verdict"]
        payload = json.loads((sweep_dir / "stacking.json").read_text(encoding="utf-8"))
        assert {a["arm"] for a in payload["arms"]} == {"ref", "armb"}
        assert payload["splits"]["secondary"]["stacked mixture"]["crps"] > 0

    def test_weights_deterministic_across_calls(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        run_stack(sweep_dir)
        first = json.loads((sweep_dir / "stacking.json").read_text(encoding="utf-8"))
        run_stack(sweep_dir)
        second = json.loads((sweep_dir / "stacking.json").read_text(encoding="utf-8"))
        assert first["arms"] == second["arms"]

    def test_no_predictive_snapshots_still_reports_weights(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path, predictive=False)
        result = run_stack(sweep_dir)
        assert "no honest headline" in result["verdict"]
        report = (sweep_dir / "stacking.md").read_text(encoding="utf-8")
        assert "Not scored" in report

    def test_partial_snapshots_renormalize_with_disclosed_dropped_mass(self, tmp_path):
        """A low-weight arm without predictive.npz must not nuke the headline."""
        sweep_dir = _make_sweep(tmp_path)
        bare_run = tmp_path / "runs" / "nosnap"
        _write_run(bare_run, elpd_value=-3.0, predictive=False)
        arms_json = json.loads((sweep_dir / "ledger.json").read_text())["arms"]
        arms_json.append(
            {"arm_id": "nosnap", "knobs": {"latent_process": "ar1"}, "stage": 1,
             "status": "completed", "run_dir": str(bare_run)}
        )
        _write_sweep(tmp_path, arms_json)
        result = run_stack(sweep_dir)
        assert "beats" in result["verdict"] or "does not beat" in result["verdict"]
        report = (sweep_dir / "stacking.md").read_text(encoding="utf-8")
        assert "renormalized over the rest" in report
        assert "nosnap" in report
        payload = json.loads((sweep_dir / "stacking.json").read_text(encoding="utf-8"))
        assert "renormalized" in payload["splits"]["secondary"]["note"]

    def test_champion_reference_row_deduped(self, tmp_path):
        """When the reference IS the champion, the table shows one row, not two."""
        ref_run = tmp_path / "runs" / "ref"
        arm_run = tmp_path / "runs" / "armb"
        _write_run(ref_run, elpd_value=-1.5, center=70.0)  # reference wins
        _write_run(arm_run, elpd_value=-2.0, center=71.0)
        sweep_dir = _write_sweep(
            tmp_path,
            [
                {"arm_id": "ref", "knobs": {}, "stage": 1, "status": "completed",
                 "run_dir": str(ref_run)},
                {"arm_id": "armb", "knobs": {"target_transform": "logit"}, "stage": 1,
                 "status": "completed", "run_dir": str(arm_run)},
            ],
        )
        run_stack(sweep_dir)
        payload = json.loads((sweep_dir / "stacking.json").read_text(encoding="utf-8"))
        labels = set(payload["splits"]["secondary"])
        assert "champion (reference)" in labels
        assert "reference (reference)" not in labels

    def test_ladder_sweep_has_single_reference_row(self, tmp_path):
        """Rung-blind loading listed the reference once per rung and split its
        weight across screening-scale duplicates of the same model."""
        sweep_dir = _make_ladder_sweep(tmp_path)
        result = run_stack(sweep_dir)
        assert result["n_arms_stacked"] == 2
        report = (sweep_dir / "stacking.md").read_text(encoding="utf-8")
        assert report.count("| reference |") == 1

    def test_fewer_than_two_arms_raises(self, tmp_path):
        run_dir = tmp_path / "runs" / "only"
        _write_run(run_dir, elpd_value=-1.0)
        sweep_dir = _write_sweep(
            tmp_path,
            [{"arm_id": "only", "knobs": {}, "stage": 1, "status": "completed",
              "run_dir": str(run_dir)}],
        )
        with pytest.raises(ValueError, match="at least two arms"):
            run_stack(sweep_dir)

    def test_baseline_block_rendered(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        baselines = tmp_path / "baseline_comparison.json"
        baselines.write_text(
            json.dumps([
                {"model": "bayes", "split": "primary", "mae": 5.3},
                {"model": "gbm", "split": "primary", "mae": 5.5},
            ]),
            encoding="utf-8",
        )
        run_stack(sweep_dir, baselines_path=baselines)
        report = (sweep_dir / "stacking.md").read_text(encoding="utf-8")
        assert "## Baseline floor" in report
        assert "beats the GBM floor" in report


class TestRenderStackReport:
    def test_verdict_without_champion_snapshot(self):
        md, payload = render_stack_report(
            arms=[], excluded=[("x", "status failed")],
            split_rows={"primary": None, "secondary": {"stacked mixture": {
                c: 1.0 for c in ("crps", "mae", "rmse", "r2", "cov80", "cov95", "wis")
            }}},
            split_notes={"primary": "nope"},
        )
        assert "no champion snapshot" in payload["verdict"]
        assert "- x: status failed" in md


class TestStackCli:
    def test_cli_end_to_end(self, tmp_path):
        sweep_dir = _make_sweep(tmp_path)
        result = runner.invoke(app, ["stack", str(sweep_dir)])
        assert result.exit_code == 0, result.stdout
        assert "Stacked 2 arms" in result.stdout
        assert "Report:" in result.stdout

    def test_cli_missing_ledger(self, tmp_path):
        result = runner.invoke(app, ["stack", str(tmp_path)])
        assert result.exit_code != 0

    def test_cli_single_arm_exits_nonzero(self, tmp_path):
        run_dir = tmp_path / "runs" / "only"
        _write_run(run_dir, elpd_value=-1.0)
        sweep_dir = _write_sweep(
            tmp_path,
            [{"arm_id": "only", "knobs": {}, "stage": 1, "status": "completed",
              "run_dir": str(run_dir)}],
        )
        result = runner.invoke(app, ["stack", str(sweep_dir)])
        assert result.exit_code == 1
        assert "at least two arms" in result.stdout
