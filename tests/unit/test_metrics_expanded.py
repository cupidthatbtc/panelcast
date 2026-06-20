"""Expanded tests for evaluation metrics: CRPSResult, PointMetrics edge cases."""

import numpy as np
import pytest

from panelcast.evaluation.metrics import (
    CRPSResult,
    PointMetrics,
    compute_crps,
    compute_point_metrics,
    posterior_mean,
)


class TestCRPSResultDataclass:
    """Tests for CRPSResult dataclass."""

    def test_fields_accessible(self):
        values = np.array([0.5, 1.0])
        result = CRPSResult(crps_values=values, mean_crps=0.75, n_obs=2)
        assert result.n_obs == 2
        assert result.mean_crps == 0.75
        np.testing.assert_array_equal(result.crps_values, values)

    def test_n_obs_matches_values_length(self):
        values = np.array([0.1, 0.2, 0.3, 0.4])
        result = CRPSResult(crps_values=values, mean_crps=0.25, n_obs=4)
        assert result.n_obs == len(result.crps_values)


class TestPointMetricsDataclass:
    """Tests for PointMetrics dataclass."""

    def test_all_fields(self):
        result = PointMetrics(
            mae=2.0,
            rmse=2.5,
            r2=0.85,
            median_ae=1.8,
            n_observations=100,
            mean_bias=-0.3,
        )
        assert result.mae == 2.0
        assert result.rmse == 2.5
        assert result.r2 == 0.85
        assert result.median_ae == 1.8
        assert result.n_observations == 100
        assert result.mean_bias == -0.3

    def test_zero_metrics(self):
        result = PointMetrics(
            mae=0.0,
            rmse=0.0,
            r2=1.0,
            median_ae=0.0,
            n_observations=5,
            mean_bias=0.0,
        )
        assert result.mae == 0.0
        assert result.r2 == 1.0


class TestComputeCRPSEdgeCases:
    """Edge case tests for compute_crps."""

    def test_single_observation(self):
        y_true = np.array([50.0])
        y_samples = np.random.default_rng(42).normal(50, 1, (100, 1))
        result = compute_crps(y_true, y_samples)
        assert result.n_obs == 1
        assert result.crps_values.shape == (1,)

    def test_single_sample(self):
        y_true = np.array([50.0, 60.0, 70.0])
        y_samples = np.array([[50.0, 60.0, 70.0]])  # (1, 3)
        result = compute_crps(y_true, y_samples)
        assert result.n_obs == 3

    def test_constant_predictions(self):
        y_true = np.array([50.0, 60.0, 70.0])
        # All samples are the same value
        y_samples = np.full((100, 3), 55.0)
        result = compute_crps(y_true, y_samples)
        # CRPS should equal the absolute error for constant predictions
        assert result.crps_values[0] == pytest.approx(5.0)
        assert result.crps_values[1] == pytest.approx(5.0)
        assert result.crps_values[2] == pytest.approx(15.0)


class TestComputePointMetricsEdgeCases:
    """Edge case tests for compute_point_metrics."""

    def test_single_observation(self):
        y_true = np.array([50.0])
        y_pred = np.array([52.0])
        result = compute_point_metrics(y_true, y_pred)
        assert result.mae == pytest.approx(2.0)
        assert result.rmse == pytest.approx(2.0)
        assert result.n_observations == 1

    def test_large_errors(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1000.0, -1000.0, 500.0])
        result = compute_point_metrics(y_true, y_pred)
        assert result.mae > 0
        assert result.rmse >= result.mae

    def test_negative_bias(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([5.0, 15.0, 25.0])
        result = compute_point_metrics(y_true, y_pred)
        assert result.mean_bias == pytest.approx(-5.0)

    def test_zero_bias(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([15.0, 15.0])  # +5, -5 = 0 bias
        result = compute_point_metrics(y_true, y_pred)
        assert result.mean_bias == pytest.approx(0.0)


class TestPosteriorMeanEdgeCases:
    """Edge case tests for posterior_mean."""

    def test_two_samples(self):
        y_samples = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = posterior_mean(y_samples)
        np.testing.assert_array_almost_equal(result, [20.0, 30.0])

    def test_large_n_samples(self):
        rng = np.random.default_rng(42)
        y_samples = rng.normal(50, 1, (10000, 5))
        result = posterior_mean(y_samples)
        assert result.shape == (5,)
        # With many samples, mean should be close to 50
        np.testing.assert_allclose(result, 50.0, atol=0.1)
