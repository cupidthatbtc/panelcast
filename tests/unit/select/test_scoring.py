"""Behavior of arm scoring: pointwise/paired elpd, ArmScore assembly, ranking, report."""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pytest
from scipy.special import logsumexp

from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    save_run_manifest,
)
from panelcast.select.scoring import (
    ArmScore,
    paired_elpd,
    pointwise_elpd,
    rank_arms,
    render_report,
    score_arm,
)

METRICS = {
    "calibration": {
        "coverages": {
            "0.80": {"nominal": 0.8, "empirical": 0.85},
            "0.95": {"nominal": 0.95, "empirical": 0.957},
        },
        "pit": {"max_abs_dev_from_uniform": 0.056},
    },
    "ppc": {"extreme_statistics": ["max", "q50", "q90"]},
    # Present on purpose: score_arm must never fall back to this estimator.
    "info_criteria": {"loo": {"elpd": -99.0, "se": 1.0}},
}
DIAGNOSTICS = {"passed": False, "rhat_max": 1.01, "ess_bulk_min": 787.0, "divergences": 0}
RESOURCES = {"expected_gb": 1.5, "actual_peak_gb": 2.0, "ratio": 1.333, "wall_clock_seconds": 60.0}


def _write_loglik(path: Path, arr: np.ndarray) -> Path:
    idata = az.from_dict(log_likelihood={"y": arr})
    idata.to_netcdf(str(path))
    return path


def _manifest(**overrides) -> RunManifest:
    base = dict(
        run_id="run-A",
        created_at="2026-07-03T00:00:00",
        command="test",
        flags={},
        seed=42,
        git=GitStateModel(commit="abc", branch="main", dirty=False, untracked_count=0),
        environment=EnvironmentInfo(
            python_version="3.11",
            jax_version="0.0",
            numpyro_version=None,
            arviz_version=None,
            platform="test",
            pixi_lock_hash=None,
        ),
        input_hashes={},
        stage_hashes={},
        stages_completed=[],
        stages_skipped=[],
        outputs={},
        success=True,
    )
    base.update(overrides)
    return RunManifest(**base)


def _make_run_dir(tmp_path: Path, loglik: np.ndarray | None, metrics: dict = METRICS) -> Path:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (eval_dir / "diagnostics.json").write_text(json.dumps(DIAGNOSTICS), encoding="utf-8")
    if loglik is not None:
        _write_loglik(eval_dir / "log_likelihood.nc", loglik)
    save_run_manifest(
        _manifest(stage_durations={"train": 12.5}, resources={"train": RESOURCES}), run_dir
    )
    return run_dir


class TestPointwiseElpd:
    def test_constant_loglik_returns_that_constant(self, tmp_path):
        # logsumexp of n identical values minus log(n) is the value itself.
        nc = _write_loglik(tmp_path / "ll.nc", np.full((2, 10, 6), -1.3))
        elpd_i = pointwise_elpd(nc)
        assert elpd_i.shape == (6,)
        np.testing.assert_allclose(elpd_i, -1.3)

    def test_matches_manual_computation(self, tmp_path):
        arr = np.random.default_rng(0).normal(-2.0, 0.5, size=(2, 10, 6))
        nc = _write_loglik(tmp_path / "ll.nc", arr)
        expected = logsumexp(arr.reshape(20, 6), axis=0) - np.log(20)
        np.testing.assert_allclose(pointwise_elpd(nc), expected)


class TestPairedElpd:
    def test_matches_manual_numpy(self, tmp_path):
        rng = np.random.default_rng(1)
        a = rng.normal(-2.0, 0.5, size=(2, 10, 6))
        b = rng.normal(-2.2, 0.5, size=(2, 10, 6))
        nc_a = _write_loglik(tmp_path / "a.nc", a)
        nc_b = _write_loglik(tmp_path / "b.nc", b)

        e_a = logsumexp(a.reshape(20, 6), axis=0) - np.log(20)
        e_b = logsumexp(b.reshape(20, 6), axis=0) - np.log(20)
        d = e_a - e_b
        diff = float(np.sum(d))
        dse = float(np.sqrt(6 * np.var(d, ddof=1)))

        pair = paired_elpd(nc_a, nc_b)
        assert pair.diff == pytest.approx(diff)
        assert pair.dse == pytest.approx(dse)
        assert pair.z == pytest.approx(diff / dse)
        assert pair.n == 6

    def test_mismatched_obs_raises(self, tmp_path):
        nc_a = _write_loglik(tmp_path / "a.nc", np.zeros((2, 10, 6)))
        nc_b = _write_loglik(tmp_path / "b.nc", np.zeros((2, 10, 5)))
        with pytest.raises(ValueError, match="obs dimensions differ"):
            paired_elpd(nc_a, nc_b)

    def test_self_pairing_is_zero_not_none(self, tmp_path):
        # The reference paired against its own snapshot: diff identically 0,
        # dse 0 — an explicit z=0 baseline row, not "unscored".
        arr = np.random.default_rng(3).normal(-2.0, 0.5, size=(2, 10, 6))
        nc = _write_loglik(tmp_path / "same.nc", arr)
        pair = paired_elpd(nc, nc)
        assert pair.diff == 0.0
        assert pair.dse == 0.0
        assert pair.z == 0.0
        assert "identically zero" in pair.note

    def test_single_observation_pair_is_none_with_note(self, tmp_path):
        # n=1: var(d, ddof=1) is NaN; z must be None, not NaN (NaN sorts
        # nondeterministically in rank_arms).
        nc_a = _write_loglik(tmp_path / "a.nc", np.full((2, 10, 1), -1.0))
        nc_b = _write_loglik(tmp_path / "b.nc", np.full((2, 10, 1), -2.0))
        pair = paired_elpd(nc_a, nc_b)
        assert pair.n == 1
        assert pair.z is None
        assert "single observation" in pair.note

    def test_zero_variance_nonzero_diff_is_none_with_note(self, tmp_path):
        nc_a = _write_loglik(tmp_path / "a.nc", np.full((2, 10, 6), -1.0))
        nc_b = _write_loglik(tmp_path / "b.nc", np.full((2, 10, 6), -2.0))
        pair = paired_elpd(nc_a, nc_b)
        assert pair.z is None
        assert pair.diff == pytest.approx(6.0)
        assert "zero variance" in pair.note


class TestScoreArm:
    def test_full_run_dir(self, tmp_path):
        rng = np.random.default_rng(2)
        cell = rng.normal(-2.0, 0.5, size=(2, 10, 6))
        ref = rng.normal(-2.1, 0.5, size=(2, 10, 6))
        run_dir = _make_run_dir(tmp_path, cell)
        ref_nc = _write_loglik(tmp_path / "ref.nc", ref)

        score = score_arm(
            run_dir, arm="cell", knobs={"latent_process": "ar1"}, reference_nc=ref_nc
        )
        expected = paired_elpd(run_dir / "evaluation" / "log_likelihood.nc", ref_nc)
        assert score.arm == "cell"
        assert score.knobs == {"latent_process": "ar1"}
        assert score.elpd_diff == pytest.approx(expected.diff)
        assert score.elpd_dse == pytest.approx(expected.dse)
        assert score.elpd_z == pytest.approx(expected.z)
        assert score.cov80 == 0.85
        assert score.cov80_delta == pytest.approx(0.05)
        assert score.cov95_delta == pytest.approx(0.007)
        assert score.pit_dev == 0.056
        assert score.ppc_pinned == 3
        assert score.ppc_pinned_names == ("max", "q50", "q90")
        assert score.converged is False
        assert score.rhat_max == 1.01
        assert score.ess_bulk_min == 787.0
        assert score.divergences == 0
        assert score.wall_clock_seconds == 60.0
        assert score.expected_gb == 1.5
        assert score.actual_peak_gb == 2.0
        assert score.resource_ratio == 1.333
        assert score.notes == []

    def test_missing_loglik_leaves_elpd_none(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, loglik=None)
        ref_nc = _write_loglik(tmp_path / "ref.nc", np.zeros((2, 10, 6)))
        score = score_arm(run_dir, arm="cell", reference_nc=ref_nc)
        # metrics.json carries a PSIS-LOO block; it must not leak in as elpd.
        assert score.elpd_diff is None
        assert score.elpd_dse is None
        assert score.elpd_z is None
        assert any("log_likelihood.nc" in n for n in score.notes)
        assert score.cov95 == 0.957  # the rest of the scorecard still fills

    def test_missing_reference_leaves_elpd_none(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, np.zeros((2, 10, 6)))
        score = score_arm(run_dir, arm="cell", reference_nc=tmp_path / "absent.nc")
        assert score.elpd_diff is None
        assert any("reference" in n for n in score.notes)

    def test_partial_metrics_none_fill_with_note(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, loglik=None, metrics={"point_metrics": {"mae": 5.6}})
        score = score_arm(run_dir, arm="cell")
        assert score.cov80 is None
        assert score.pit_dev is None
        assert score.ppc_pinned is None
        assert any(n.startswith("metrics.json missing:") for n in score.notes)

    def test_empty_run_dir_never_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        score = score_arm(empty, arm="cell")
        assert score.elpd_diff is None
        assert score.converged is None
        assert score.wall_clock_seconds is None
        assert "metrics.json missing" in score.notes
        assert "diagnostics.json missing" in score.notes
        assert "manifest.json missing" in score.notes

    def test_wall_clock_falls_back_to_stage_duration(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_run_manifest(_manifest(stage_durations={"train": 12.5}), run_dir)
        score = score_arm(run_dir, arm="cell")
        assert score.wall_clock_seconds == 12.5
        assert score.expected_gb is None

    def test_degenerate_pair_note_propagates(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, np.full((2, 10, 6), -1.0))
        ref_nc = _write_loglik(tmp_path / "ref.nc", np.full((2, 10, 6), -1.0))
        score = score_arm(run_dir, arm="cell", reference_nc=ref_nc)
        assert score.elpd_z == 0.0
        assert any("identically zero" in n for n in score.notes)


class TestRankArms:
    def test_ordering_and_none_sinking(self):
        a = ArmScore(arm="a", elpd_z=2.0)
        b = ArmScore(arm="b")
        c = ArmScore(arm="c", elpd_z=-1.0, converged=False)
        d = ArmScore(arm="d", elpd_z=5.0)
        ranked = rank_arms([a, b, c, d])
        assert [s.arm for s in ranked] == ["d", "a", "c", "b"]

    def test_convergence_failures_kept(self):
        failed = ArmScore(arm="bad", elpd_z=9.0, converged=False)
        ranked = rank_arms([ArmScore(arm="ok", elpd_z=1.0, converged=True), failed])
        assert ranked[0] is failed


class TestRenderReport:
    def _scores(self):
        return [
            ArmScore(
                arm="offset_logit_rw",
                elpd_diff=22.2,
                elpd_dse=4.5,
                elpd_z=4.91,
                converged=False,
                rhat_max=1.01,
                ess_bulk_min=615.0,
                divergences=0,
            ),
            ArmScore(arm="identity_ar1", elpd_diff=-2.2, elpd_dse=0.93, elpd_z=-2.39,
                     converged=True),
            ArmScore(arm="offset_logit_ar1"),
        ]

    def test_structure_and_verdict(self):
        md, payload = render_report(self._scores(), "identity_rw", title="Sweep")
        lines = md.splitlines()
        assert lines[0] == "# Sweep"
        header = next(line for line in lines if line.startswith("| arm |"))
        for label in ("elpd_diff", "dse", "z", "d_cov80", "conv", "wall_s", "peak_gb"):
            assert f" {label} " in header
        for arm in ("offset_logit_rw", "identity_ar1", "offset_logit_ar1"):
            assert any(line.startswith(f"| {arm} |") for line in lines)
        assert "**Verdict:**" in md
        assert "offset_logit_rw leads" in md
        assert "failed the convergence gate" in md
        assert "offset_logit_ar1" in md.split("**Verdict:**")[1]  # unscored caveat
        assert "## Baseline floor" not in md
        assert payload["reference"] == "identity_rw"
        assert payload["arms"][0]["arm"] == "offset_logit_rw"
        assert payload["baseline_floor"] is None
        json.dumps(payload)

    def test_rows_render_dash_for_missing(self):
        md, _ = render_report([ArmScore(arm="bare")], "ref")
        row = next(line for line in md.splitlines() if line.startswith("| bare |"))
        assert " - " in row

    def test_baseline_floor_section(self):
        block = {
            "rows": [
                {"model": "gbm", "split": "s", "mae": 5.65, "runtime_s": float("nan")},
                {"model": "bayes (current)", "split": "s", "mae": 5.31},
            ]
        }
        md, payload = render_report(self._scores(), "identity_rw", baseline_block=block)
        assert "## Baseline floor" in md
        assert "| model | split | mae | runtime_s |" in md
        assert "beats the GBM floor" in md
        assert payload["baseline_floor"][0]["runtime_s"] is None
        json.dumps(payload)

    def test_no_scored_arms_verdict(self):
        md, _ = render_report([ArmScore(arm="only")], "ref")
        assert "nothing is scored against ref" in md

    def test_degenerate_and_missing_caveats_are_distinct(self):
        # An arm that WAS paired but has no defined z (dse 0) must not be
        # reported as "no snapshot" — that caveat is for unpaired arms only.
        scores = [
            ArmScore(arm="winner", elpd_diff=5.0, elpd_dse=1.0, elpd_z=5.0, converged=True),
            ArmScore(arm="no_snapshot"),
            ArmScore(arm="degen", elpd_diff=0.0, elpd_dse=0.0),
        ]
        md, _ = render_report(scores, "ref")
        verdict = md.split("**Verdict:**")[1]
        assert "No pointwise log-likelihood snapshot for no_snapshot;" in verdict
        assert "Paired elpd degenerate (z undefined) for degen." in verdict
