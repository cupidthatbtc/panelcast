"""Unit tests for the metrics module.

Tests CRPS computation, point metrics, and the posterior_mean helper.
Uses synthetic data with known statistical properties.
"""

import numpy as np
import pytest

from panelcast.evaluation.metrics import (
    CRPSResult,
    PointMetrics,
    compute_crps,
    compute_point_metrics,
    posterior_mean,
)


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def perfect_prediction_data(rng):
    """Data where predictions perfectly match true values (CRPS near 0)."""
    n_obs = 100
    y_true = rng.normal(50, 10, n_obs)

    # "Perfect" samples: very narrow distribution centered exactly on y_true
    n_samples = 1000
    sigma = 0.01  # Very small noise
    y_samples = y_true + rng.normal(0, sigma, (n_samples, n_obs))

    return y_true, y_samples


@pytest.fixture
def noisy_prediction_data(rng):
    """Data where predictions are noisy (higher CRPS)."""
    n_obs = 100
    y_true = rng.normal(50, 10, n_obs)

    # Noisy samples: wide distribution with offset from y_true
    n_samples = 1000
    sigma = 20.0
    offset = 5.0  # Bias
    y_samples = y_true + offset + rng.normal(0, sigma, (n_samples, n_obs))

    return y_true, y_samples


@pytest.fixture
def point_prediction_perfect():
    """Perfect point predictions (MAE=0, R2=1)."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def point_prediction_imperfect():
    """Imperfect point predictions with known errors."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    # Errors: -2, +2, -2, +2, -2 -> MAE = 2.0
    y_pred = np.array([52.0, 58.0, 72.0, 78.0, 92.0])
    return y_true, y_pred


class TestComputeCRPS:
    """Tests for the compute_crps function."""

    def test_returns_crps_result(self, perfect_prediction_data):
        """Should return a CRPSResult instance."""
        y_true, y_samples = perfect_prediction_data
        result = compute_crps(y_true, y_samples)

        assert isinstance(result, CRPSResult)

    def test_compute_crps_perfect_prediction(self, perfect_prediction_data):
        """CRPS should be near zero for perfect predictions."""
        y_true, y_samples = perfect_prediction_data
        result = compute_crps(y_true, y_samples)

        # With very narrow samples centered on truth, CRPS should be small
        assert result.mean_crps < 0.1
        assert result.n_obs == len(y_true)

    def test_compute_crps_worse_prediction(self, perfect_prediction_data, noisy_prediction_data):
        """CRPS should be higher for worse predictions."""
        y_true_perfect, y_samples_perfect = perfect_prediction_data
        y_true_noisy, y_samples_noisy = noisy_prediction_data

        crps_perfect = compute_crps(y_true_perfect, y_samples_perfect)
        crps_noisy = compute_crps(y_true_noisy, y_samples_noisy)

        # Noisy predictions should have higher CRPS
        assert crps_noisy.mean_crps > crps_perfect.mean_crps

    def test_compute_crps_shape(self, perfect_prediction_data):
        """crps_values should have correct length."""
        y_true, y_samples = perfect_prediction_data
        result = compute_crps(y_true, y_samples)

        assert len(result.crps_values) == len(y_true)
        assert result.crps_values.shape == (len(y_true),)

    def test_crps_values_non_negative(self, noisy_prediction_data):
        """CRPS values should always be non-negative."""
        y_true, y_samples = noisy_prediction_data
        result = compute_crps(y_true, y_samples)

        assert all(v >= 0 for v in result.crps_values)
        assert result.mean_crps >= 0

    def test_crps_mean_consistency(self, noisy_prediction_data):
        """Mean CRPS should be consistent with crps_values."""
        y_true, y_samples = noisy_prediction_data
        result = compute_crps(y_true, y_samples)

        expected_mean = result.crps_values.mean()
        assert np.isclose(result.mean_crps, expected_mean)

    def test_crps_input_validation_y_true_shape(self, rng):
        """Should raise error for non-1D y_true."""
        y_true_2d = rng.normal(0, 1, (10, 5))
        y_samples = rng.normal(0, 1, (100, 50))

        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_crps(y_true_2d, y_samples)

    def test_crps_input_validation_y_samples_shape(self, rng):
        """Should raise error for non-2D y_samples."""
        y_true = rng.normal(0, 1, 50)
        y_samples_1d = rng.normal(0, 1, 50)

        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_crps(y_true, y_samples_1d)

    def test_crps_input_validation_shape_mismatch(self, rng):
        """Should raise error when shapes don't match."""
        y_true = rng.normal(0, 1, 50)
        y_samples = rng.normal(0, 1, (100, 30))  # Wrong n_obs

        with pytest.raises(ValueError, match="observations"):
            compute_crps(y_true, y_samples)


class TestComputePointMetrics:
    """Tests for the compute_point_metrics function."""

    def test_returns_point_metrics(self, point_prediction_imperfect):
        """Should return a PointMetrics instance."""
        y_true, y_pred = point_prediction_imperfect
        result = compute_point_metrics(y_true, y_pred)

        assert isinstance(result, PointMetrics)

    def test_compute_point_metrics_perfect(self, point_prediction_perfect):
        """Perfect predictions should have MAE=0, RMSE=0, R2=1."""
        y_true, y_pred = point_prediction_perfect
        result = compute_point_metrics(y_true, y_pred)

        assert result.mae == 0.0
        assert result.rmse == 0.0
        assert result.r2 == 1.0
        assert result.median_ae == 0.0
        assert result.n_observations == 5
        assert result.mean_bias == 0.0

    def test_compute_point_metrics_imperfect(self, point_prediction_imperfect):
        """Imperfect predictions should have reasonable metric values."""
        y_true, y_pred = point_prediction_imperfect
        result = compute_point_metrics(y_true, y_pred)

        # MAE should be 2.0 (all errors are +/- 2)
        assert np.isclose(result.mae, 2.0)

        # RMSE should also be 2.0 (since all errors have same magnitude)
        assert np.isclose(result.rmse, 2.0)

        # R2 should be high but not 1.0
        assert 0.9 < result.r2 < 1.0

        # Median AE should be 2.0
        assert np.isclose(result.median_ae, 2.0)

        # n_observations should be 5
        assert result.n_observations == 5

        # mean_bias = mean([52-50, 58-60, 72-70, 78-80, 92-90])
        #           = mean([+2, -2, +2, -2, +2]) = 2/5 = 0.4
        assert np.isclose(result.mean_bias, 0.4)

    def test_compute_point_metrics_all_fields(self, point_prediction_imperfect):
        """All PointMetrics fields should be populated."""
        y_true, y_pred = point_prediction_imperfect
        result = compute_point_metrics(y_true, y_pred)

        # All fields should be numeric (not None or NaN)
        assert isinstance(result.mae, float)
        assert isinstance(result.rmse, float)
        assert isinstance(result.r2, float)
        assert isinstance(result.median_ae, float)
        assert isinstance(result.n_observations, int)
        assert isinstance(result.mean_bias, float)
        assert not np.isnan(result.mae)
        assert not np.isnan(result.rmse)
        assert not np.isnan(result.r2)
        assert not np.isnan(result.median_ae)
        assert not np.isnan(result.mean_bias)

    def test_mae_always_positive_or_zero(self, rng):
        """MAE should never be negative."""
        y_true = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        result = compute_point_metrics(y_true, y_pred)

        assert result.mae >= 0

    def test_rmse_always_positive_or_zero(self, rng):
        """RMSE should never be negative."""
        y_true = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        result = compute_point_metrics(y_true, y_pred)

        assert result.rmse >= 0

    def test_rmse_gte_mae(self, rng):
        """RMSE should always be >= MAE (due to Jensen's inequality)."""
        y_true = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        result = compute_point_metrics(y_true, y_pred)

        # RMSE >= MAE always (equality when all errors have same magnitude)
        assert result.rmse >= result.mae - 1e-10  # Small tolerance for floating point

    def test_r2_can_be_negative(self, rng):
        """R2 can be negative for very poor predictions."""
        y_true = rng.normal(50, 1, 100)  # Low variance
        y_pred = rng.normal(0, 50, 100)  # Very wrong predictions

        result = compute_point_metrics(y_true, y_pred)

        # R2 can be negative when predictions are worse than mean
        assert result.r2 < 0

    def test_r2_identical_values(self):
        """R2 should handle the edge case of identical true values."""
        y_true = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        y_pred_perfect = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
        y_pred_imperfect = np.array([49.0, 51.0, 49.0, 51.0, 49.0])

        # Perfect predictions with zero variance should give R2=1.0
        result_perfect = compute_point_metrics(y_true, y_pred_perfect)
        assert result_perfect.r2 == 1.0

        # Imperfect predictions with zero variance in y_true
        result_imperfect = compute_point_metrics(y_true, y_pred_imperfect)
        # When SS_tot = 0, we return 1.0 if perfect, 0.0 otherwise
        assert result_imperfect.r2 == 0.0

    def test_point_metrics_input_validation(self, rng):
        """Should raise error for invalid inputs."""
        y_true_2d = rng.normal(0, 1, (10, 5))
        y_pred = rng.normal(0, 1, 50)

        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_point_metrics(y_true_2d, y_pred)

    def test_point_metrics_mean_bias_calculation(self):
        """Test mean_bias calculation with known asymmetric errors."""
        # y_pred consistently 3 units higher than y_true
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([13.0, 23.0, 33.0, 43.0, 53.0])  # All +3 bias

        result = compute_point_metrics(y_true, y_pred)

        # mean_bias = mean(y_pred - y_true) = mean([3, 3, 3, 3, 3]) = 3.0
        assert np.isclose(result.mean_bias, 3.0)

        # Test negative bias
        y_pred_under = np.array([8.0, 18.0, 28.0, 38.0, 48.0])  # All -2 bias
        result_under = compute_point_metrics(y_true, y_pred_under)

        # mean_bias = mean([-2, -2, -2, -2, -2]) = -2.0
        assert np.isclose(result_under.mean_bias, -2.0)


class TestPosteriorMean:
    """Tests for the posterior_mean helper function."""

    def test_posterior_mean_shape(self, rng):
        """Output should have shape (n_obs,)."""
        n_samples, n_obs = 1000, 50
        y_samples = rng.normal(0, 1, (n_samples, n_obs))

        result = posterior_mean(y_samples)

        assert result.shape == (n_obs,)

    def test_posterior_mean_values(self, rng):
        """Posterior mean should be the mean across samples."""
        n_samples, n_obs = 1000, 50
        y_samples = rng.normal(0, 1, (n_samples, n_obs))

        result = posterior_mean(y_samples)
        expected = y_samples.mean(axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_posterior_mean_known_values(self):
        """Test with known values."""
        y_samples = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
            ]
        )  # Shape: (3, 3)

        result = posterior_mean(y_samples)
        expected = np.array([3.0, 4.0, 5.0])  # Mean across axis 0

        np.testing.assert_array_almost_equal(result, expected)

    def test_posterior_mean_input_validation(self, rng):
        """Should raise error for non-2D input."""
        y_samples_1d = rng.normal(0, 1, 50)

        with pytest.raises(ValueError, match="y_samples must be 2D"):
            posterior_mean(y_samples_1d)

    def test_posterior_mean_single_sample(self):
        """Should work with a single sample (edge case)."""
        y_samples = np.array([[1.0, 2.0, 3.0]])  # Shape: (1, 3)

        result = posterior_mean(y_samples)
        expected = np.array([1.0, 2.0, 3.0])

        np.testing.assert_array_almost_equal(result, expected)


# --- from unit/test_metrics_expanded.py ---


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


# --- from unit/test_metrics_new.py ---


class TestComputePointMetricsNaN:
    """Cover NaN validation paths."""

    def test_nan_in_y_true_raises(self):
        """NaN in y_true should raise ValueError."""
        y_true = np.array([50.0, float("nan"), 70.0])
        y_pred = np.array([50.0, 60.0, 70.0])
        with pytest.raises(ValueError, match="NaN"):
            compute_point_metrics(y_true, y_pred)

    def test_nan_in_y_pred_raises(self):
        """NaN in y_pred_mean should raise ValueError."""
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([50.0, float("nan"), 70.0])
        with pytest.raises(ValueError, match="NaN"):
            compute_point_metrics(y_true, y_pred)


class TestComputePointMetricsValidation:
    """Cover validation error paths."""

    def test_y_pred_2d_raises(self):
        """2D y_pred_mean should raise ValueError."""
        y_true = np.array([50.0, 60.0])
        y_pred = np.array([[50.0], [60.0]])
        with pytest.raises(ValueError, match="y_pred_mean must be 1D"):
            compute_point_metrics(y_true, y_pred)

    def test_length_mismatch_raises(self):
        """Length mismatch between y_true and y_pred should raise."""
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([50.0, 60.0])
        with pytest.raises(ValueError, match="observations"):
            compute_point_metrics(y_true, y_pred)


class TestComputeCrpsListInput:
    """Cover automatic conversion from lists."""

    def test_list_inputs(self):
        """Lists should be converted to arrays."""
        y_true = [50.0, 60.0, 70.0]
        y_samples = [[48.0, 58.0, 68.0], [52.0, 62.0, 72.0]]
        result = compute_crps(y_true, y_samples)
        assert result.n_obs == 3
        assert result.crps_values.shape == (3,)


class TestPosteriorMean3D:
    """Cover 3D input validation."""

    def test_3d_input_raises(self):
        """3D input should raise ValueError."""
        y_samples = np.ones((10, 5, 3))
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            posterior_mean(y_samples)

    def test_0d_input_raises(self):
        """0D input should raise ValueError."""
        y_samples = np.array(5.0)
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            posterior_mean(y_samples)
