"""Tests for residual autocorrelation diagnostic and PIT values."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.evaluation.calibration import compute_pit_values
from panelcast.models.bayes.diagnostics import compute_residual_autocorrelation


class TestResidualAutocorrelation:
    def test_recovers_ar_coefficient(self):
        """Synthetic AR(0.5) residuals within one entity are detected."""
        rng = np.random.default_rng(42)
        n = 5000
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = 0.5 * r[t - 1] + rng.normal(0, 1)
        entity = np.zeros(n, dtype=int)
        result = compute_residual_autocorrelation(r, entity)
        assert result["lag1_acf"] == pytest.approx(0.5, abs=0.05)
        assert result["n_pairs"] == n - 1

    def test_iid_residuals_near_zero(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0, 1, 5000)
        entity = np.repeat(np.arange(500), 10)
        result = compute_residual_autocorrelation(r, entity)
        assert abs(result["lag1_acf"]) < 0.05
        # 10 obs per entity -> 9 pairs each.
        assert result["n_pairs"] == 500 * 9

    def test_no_pairs_across_entity_boundaries(self):
        """Two entities with perfectly correlated boundary must contribute 0 pairs."""
        r = np.array([1.0, 2.0])
        entity = np.array([0, 1])
        result = compute_residual_autocorrelation(r, entity)
        assert result["lag1_acf"] is None
        assert result["n_pairs"] == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_residual_autocorrelation(np.zeros(3), np.zeros(4, dtype=int))

    def test_singleton_entities_contribute_nothing(self):
        r = np.array([1.0, -1.0, 2.0, -2.0])
        entity = np.array([0, 1, 2, 3])
        result = compute_residual_autocorrelation(r, entity)
        assert result["n_pairs"] == 0
        assert result["n_entities_multi"] == 0


class TestPitValues:
    def test_calibrated_forecast_is_uniform(self):
        """Draws and observations from the same distribution -> uniform PIT."""
        rng = np.random.default_rng(42)
        n_obs, n_draws = 2000, 400
        y_true = rng.normal(0, 1, n_obs)
        y_samples = rng.normal(0, 1, (n_draws, n_obs))
        pit = compute_pit_values(y_true, y_samples)
        assert pit["mean"] == pytest.approx(0.5, abs=0.02)
        # Uniform std = sqrt(1/12) ~= 0.2887
        assert pit["std"] == pytest.approx(np.sqrt(1 / 12), abs=0.02)
        assert pit["max_abs_dev_from_uniform"] < 0.03

    def test_overconfident_forecast_is_u_shaped(self):
        """Predictive spread too narrow -> mass piles at the PIT extremes."""
        rng = np.random.default_rng(7)
        n_obs, n_draws = 2000, 400
        y_true = rng.normal(0, 2.0, n_obs)  # truth twice as wide
        y_samples = rng.normal(0, 1.0, (n_draws, n_obs))
        pit = compute_pit_values(y_true, y_samples, n_bins=10)
        counts = np.asarray(pit["counts"], dtype=float)
        freq = counts / counts.sum()
        # Edge bins clearly above uniform, middle clearly below.
        assert freq[0] > 0.15 and freq[-1] > 0.15
        assert freq[4] < 0.08 and freq[5] < 0.08

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="incompatible"):
            compute_pit_values(np.zeros(5), np.zeros((10, 4)))
