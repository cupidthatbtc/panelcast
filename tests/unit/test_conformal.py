"""Conformal wrapper (#156): CQR adjustment, PIT recalibration, level maps."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.evaluation.conformal import (
    conformal_adjustment,
    conformalize,
    conformalized_bounds,
    recalibrated_levels,
)


def _cal_set(
    n=300, n_draws=400, pred_scale=5.0, true_scale=5.0, seed=0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(70, true_scale, size=n)
    samples = rng.normal(70, pred_scale, size=(n_draws, n))
    return y, samples


class TestConformalAdjustment:
    def test_too_narrow_intervals_get_widened(self):
        # Predictive claims sd 1 while truth has sd 5: intervals must widen.
        y, samples = _cal_set(pred_scale=1.0, true_scale=5.0)
        adj = conformal_adjustment(y, samples, prob=0.9)
        assert adj > 0
        lo, hi = conformalized_bounds(samples, 0.9, adj)
        assert float(np.mean((y >= lo) & (y <= hi))) >= 0.88

    def test_too_wide_intervals_get_tightened(self):
        y, samples = _cal_set(pred_scale=20.0, true_scale=5.0)
        assert conformal_adjustment(y, samples, prob=0.9) < 0

    def test_calibrated_intervals_barely_move(self):
        y, samples = _cal_set(pred_scale=5.0, true_scale=5.0, n=2000)
        adj = conformal_adjustment(y, samples, prob=0.8)
        lo, hi = conformalized_bounds(samples, 0.8, 0.0)
        assert abs(adj) < 0.15 * float(np.mean(hi - lo))

    def test_empty_calibration_raises(self):
        with pytest.raises(ValueError, match="empty calibration"):
            conformal_adjustment(np.array([]), np.zeros((10, 0)), 0.9)


class TestRecalibratedLevels:
    def test_identity_when_calibrated(self):
        # Uniform PIT (perfect calibration) maps levels to themselves.
        pit = np.linspace(0.001, 0.999, 5000)
        out = recalibrated_levels(pit, [0.05, 0.5, 0.95])
        np.testing.assert_allclose(out, [0.05, 0.5, 0.95], atol=0.01)

    def test_overconfident_model_pushes_levels_outward(self):
        # PIT mass in the tails (intervals too narrow): the 95% level must
        # move above 0.95 to actually cover 95%.
        pit = np.concatenate([np.random.default_rng(1).uniform(0, 1, 200),
                              np.zeros(150), np.ones(150)])
        lo, hi = recalibrated_levels(pit, [0.05, 0.95])
        assert lo < 0.05 or hi > 0.95


class TestConformalize:
    def test_block_shape_and_grid(self):
        y_cal, cal_samples = _cal_set(seed=2)
        y_test, test_samples = _cal_set(seed=3)
        block = conformalize(y_cal, cal_samples, y_test, test_samples, (0.8, 0.95))
        assert block["n_calibration"] == len(y_cal)
        assert set(block["levels"]) == {"0.80", "0.95"}
        for lv in block["levels"].values():
            assert set(lv) >= {
                "cqr_adjustment", "cqr_coverage", "cqr_mean_width",
                "recalibrated_levels", "recalibrated_coverage",
            }
        grid = block["pit_quantile_grid"]
        assert len(grid["levels"]) == len(grid["values"]) == 101
        assert "exchangeability" in block["note"]

    def test_conformalized_coverage_hits_nominal_on_misspecified_model(self):
        # Same misspecification in cal and test: conformal fixes coverage.
        y_cal, cal_samples = _cal_set(pred_scale=1.5, true_scale=5.0, seed=4)
        y_test, test_samples = _cal_set(pred_scale=1.5, true_scale=5.0, seed=5)
        block = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,))
        lv = block["levels"]["0.90"]
        raw_lo = np.percentile(test_samples, 5, axis=0)
        raw_hi = np.percentile(test_samples, 95, axis=0)
        raw_coverage = float(np.mean((y_test >= raw_lo) & (y_test <= raw_hi)))
        assert raw_coverage < 0.7  # the raw intervals really are broken
        # CQR widens additively, so it restores coverage even here.
        assert lv["cqr_coverage"] >= 0.85
        # Recalibration can only stretch to the predictive's sample range, so
        # under severe narrowness it improves but saturates below nominal.
        assert lv["recalibrated_coverage"] > raw_coverage

    def test_recalibration_recovers_nominal_under_mild_misspecification(self):
        y_cal, cal_samples = _cal_set(pred_scale=3.5, true_scale=5.0, seed=6)
        y_test, test_samples = _cal_set(pred_scale=3.5, true_scale=5.0, seed=7)
        block = conformalize(y_cal, cal_samples, y_test, test_samples, (0.9,))
        assert block["levels"]["0.90"]["recalibrated_coverage"] >= 0.85


class TestPredictNextLevelMap:
    def test_levels_interpolated_from_metrics(self, tmp_path):
        import json

        from panelcast.pipelines.predict_next import _load_conformal_levels

        grid_levels = np.round(np.linspace(0.0, 1.0, 101), 2)
        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "calibration": {
                        "conformal": {
                            "pit_quantile_grid": {
                                "levels": grid_levels.tolist(),
                                # Identity map: recalibration is a no-op.
                                "values": grid_levels.tolist(),
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        levels = _load_conformal_levels(tmp_path)
        assert levels == pytest.approx((0.05, 0.95))

    def test_missing_block_returns_none(self, tmp_path):
        from panelcast.pipelines.predict_next import _load_conformal_levels

        assert _load_conformal_levels(tmp_path) is None
