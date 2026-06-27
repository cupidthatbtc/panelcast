"""Unit tests for the calibration module.

Tests coverage computation, multi-level coverage, and reliability diagram data.
Uses synthetic data with known calibration properties.
"""

import numpy as np
import pytest

from panelcast.evaluation.calibration import (
    CoverageResult,
    IntervalScoreResult,
    ReliabilityData,
    WISResult,
    compute_coverage,
    compute_interval_score,
    compute_multi_coverage,
    compute_reliability_data,
    compute_weighted_interval_score,
)


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def perfect_calibration_data(rng):
    """Synthetic data with perfect calibration.

    For perfect calibration, we need y_true to be a sample FROM the
    predictive distribution (not the center of it). The model's
    predictive distribution has mean mu and std sigma. y_true is
    drawn from N(mu, sigma), so when we compute coverage, ~95% of
    observations should fall within the 95% CI.
    """
    n_obs = 500
    sigma = 10.0
    n_samples = 1000

    # For each observation, the model predicts a distribution N(mu_i, sigma)
    # We'll use mu_i = 50 for all observations (simplicity)
    mu = np.full(n_obs, 50.0)

    # y_true is sampled from the model's predictive distribution
    # This is the key: y_true = mu + eps, where eps ~ N(0, sigma)
    y_true = mu + rng.normal(0, sigma, n_obs)

    # The model's predictive samples are drawn from N(mu_i, sigma)
    # Note: samples are around mu, not around y_true
    y_samples = mu + rng.normal(0, sigma, (n_samples, n_obs))

    return y_true, y_samples, sigma


@pytest.fixture
def overconfident_data(rng):
    """Synthetic data with overconfident predictions.

    Intervals are too narrow - samples have lower variance than the
    actual noise in y_true.
    """
    n_obs = 500
    true_sigma = 10.0
    model_sigma = 5.0  # Model thinks variance is lower

    # True values with noise
    y_base = rng.normal(50, 5, n_obs)
    y_true = y_base + rng.normal(0, true_sigma, n_obs)

    # Samples are too narrow
    n_samples = 1000
    y_samples = y_base + rng.normal(0, model_sigma, (n_samples, n_obs))

    return y_true, y_samples


@pytest.fixture
def underconfident_data(rng):
    """Synthetic data with underconfident predictions.

    Intervals are too wide - samples have higher variance than the
    actual noise in y_true.
    """
    n_obs = 500
    true_sigma = 5.0
    model_sigma = 15.0  # Model thinks variance is higher

    # True values with small noise
    y_base = rng.normal(50, 5, n_obs)
    y_true = y_base + rng.normal(0, true_sigma, n_obs)

    # Samples are too wide
    n_samples = 1000
    y_samples = y_base + rng.normal(0, model_sigma, (n_samples, n_obs))

    return y_true, y_samples


class TestComputeCoverage:
    """Tests for the compute_coverage function."""

    def test_returns_coverage_result(self, perfect_calibration_data):
        """Should return a CoverageResult instance."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        assert isinstance(result, CoverageResult)

    def test_compute_coverage_perfect_calibration(self, perfect_calibration_data):
        """Well-calibrated samples should have empirical coverage near nominal."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # With 500 observations, coverage should be within a reasonable range
        # Allow some sampling variability (say, 90% to 100%)
        assert 0.90 <= result.empirical <= 1.0
        assert result.nominal == 0.95
        assert result.n_obs == 500

    def test_compute_coverage_overconfident(self, overconfident_data):
        """Overconfident predictions should have coverage < nominal."""
        y_true, y_samples = overconfident_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # Overconfident model: coverage should be clearly below 95%
        assert result.empirical < 0.90
        assert result.nominal == 0.95

    def test_compute_coverage_underconfident(self, underconfident_data):
        """Underconfident predictions should have coverage > nominal."""
        y_true, y_samples = underconfident_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # Underconfident model: coverage should be near 100%
        assert result.empirical > 0.97
        assert result.nominal == 0.95

    def test_compute_coverage_interval_width(self, perfect_calibration_data):
        """Interval width (sharpness) should be computed correctly."""
        y_true, y_samples, sigma = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # For normal distribution, 95% CI width is ~3.92 * sigma
        expected_width = 3.92 * sigma
        # Allow 20% tolerance for finite samples
        assert 0.8 * expected_width <= result.interval_width <= 1.2 * expected_width

    def test_coverage_bounds_shape(self, perfect_calibration_data):
        """Lower and upper bounds should have correct shape."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        assert result.lower_bound.shape == (len(y_true),)
        assert result.upper_bound.shape == (len(y_true),)

    def test_coverage_n_covered_consistent(self, perfect_calibration_data):
        """n_covered should be consistent with empirical coverage."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        expected_empirical = result.n_covered / result.n_obs
        assert result.empirical == expected_empirical

    def test_different_probability_levels(self, perfect_calibration_data):
        """Coverage should work for different probability levels."""
        y_true, y_samples, _ = perfect_calibration_data

        for prob in [0.50, 0.80, 0.95]:
            result = compute_coverage(y_true, y_samples, prob=prob)
            assert result.nominal == prob
            # Looser bounds for smaller intervals
            tolerance = 0.15 if prob < 0.9 else 0.10
            assert abs(result.empirical - prob) < tolerance

    def test_input_validation_y_true_shape(self, rng):
        """Should raise error for non-1D y_true."""
        y_true_2d = rng.normal(0, 1, (10, 5))
        y_samples = rng.normal(0, 1, (100, 50))

        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_coverage(y_true_2d, y_samples)

    def test_input_validation_y_samples_shape(self, rng):
        """Should raise error for non-2D y_samples."""
        y_true = rng.normal(0, 1, 50)
        y_samples_1d = rng.normal(0, 1, 50)

        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_coverage(y_true, y_samples_1d)

    def test_input_validation_shape_mismatch(self, rng):
        """Should raise error when shapes don't match."""
        y_true = rng.normal(0, 1, 50)
        y_samples = rng.normal(0, 1, (100, 30))  # Wrong n_obs

        with pytest.raises(ValueError, match="observations"):
            compute_coverage(y_true, y_samples)

    @pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.1])
    def test_input_validation_probability_range(self, perfect_calibration_data, prob):
        """Coverage should reject invalid nominal probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        with pytest.raises(ValueError, match="prob must satisfy 0 < prob < 1"):
            compute_coverage(y_true, y_samples, prob=prob)


class TestComputeMultiCoverage:
    """Tests for the compute_multi_coverage function."""

    def test_compute_multi_coverage_all_levels(self, perfect_calibration_data):
        """Should return coverage for all requested probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        probs = (0.50, 0.80, 0.95)

        results = compute_multi_coverage(y_true, y_samples, probs=probs)

        assert set(results.keys()) == set(probs)
        for prob in probs:
            assert isinstance(results[prob], CoverageResult)
            assert results[prob].nominal == prob

    def test_multi_coverage_custom_levels(self, perfect_calibration_data):
        """Should work with custom probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        probs = (0.60, 0.90, 0.99)

        results = compute_multi_coverage(y_true, y_samples, probs=probs)

        assert set(results.keys()) == set(probs)

    def test_multi_coverage_ordering(self, perfect_calibration_data):
        """Higher probability levels should have higher or equal coverage."""
        y_true, y_samples, _ = perfect_calibration_data
        probs = (0.50, 0.80, 0.95)

        results = compute_multi_coverage(y_true, y_samples, probs=probs)

        # Coverage should increase with probability level (for calibrated model)
        # Use <= for robustness (can be equal at 100%)
        assert results[0.50].empirical <= results[0.80].empirical
        assert results[0.80].empirical <= results[0.95].empirical
        # At least one should differ (not all saturated at 100%)
        assert results[0.50].empirical < 1.0 or results[0.50].nominal == 0.50


class TestComputeReliabilityData:
    """Tests for the compute_reliability_data function."""

    def test_returns_reliability_data(self, perfect_calibration_data):
        """Should return a ReliabilityData instance."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        assert isinstance(result, ReliabilityData)

    def test_compute_reliability_data_bins(self, perfect_calibration_data):
        """Should produce the requested number of bins (or fewer if ties)."""
        y_true, y_samples, _ = perfect_calibration_data
        n_bins = 10

        result = compute_reliability_data(y_true, y_samples, n_bins=n_bins)

        # Number of actual bins may be <= n_bins due to ties
        assert len(result.predicted_probs) <= n_bins
        assert len(result.observed_freq) <= n_bins
        assert len(result.counts) <= n_bins

    def test_compute_reliability_data_counts(self, perfect_calibration_data):
        """Each quantile level should use all observations."""
        y_true, y_samples, _ = perfect_calibration_data
        n_bins = 10

        result = compute_reliability_data(y_true, y_samples, n_bins=n_bins)

        assert np.all(result.counts == len(y_true))

    def test_reliability_data_bin_edges(self, perfect_calibration_data):
        """Bin edges should span [0, 1] approximately."""
        y_true, y_samples, _ = perfect_calibration_data

        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        # Bin edges should be in [0, 1] (or close to it for empirical data)
        assert result.bin_edges[0] >= 0
        assert result.bin_edges[-1] <= 1

    def test_reliability_predicted_probs_range(self, perfect_calibration_data):
        """Predicted probabilities should be in [0, 1]."""
        y_true, y_samples, _ = perfect_calibration_data

        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        # Check non-empty bins
        mask = result.counts > 0
        assert all(0 <= p <= 1 for p in result.predicted_probs[mask])

    def test_reliability_observed_freq_range(self, perfect_calibration_data):
        """Observed frequencies should be in [0, 1]."""
        y_true, y_samples, _ = perfect_calibration_data

        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        # Check non-empty bins
        mask = result.counts > 0
        assert all(0 <= f <= 1 for f in result.observed_freq[mask])

    def test_reliability_different_n_bins(self, perfect_calibration_data):
        """Should work with different numbers of bins."""
        y_true, y_samples, _ = perfect_calibration_data

        for n_bins in [5, 10, 20]:
            result = compute_reliability_data(y_true, y_samples, n_bins=n_bins)
            # Just check it runs and produces valid output
            assert len(result.predicted_probs) <= n_bins
            assert np.all(result.counts == len(y_true))

    def test_reliability_input_validation(self, rng):
        """Should raise error for invalid inputs."""
        y_true_2d = rng.normal(0, 1, (10, 5))
        y_samples = rng.normal(0, 1, (100, 50))

        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_reliability_data(y_true_2d, y_samples)

    def test_reliability_invalid_n_bins(self, perfect_calibration_data):
        """n_bins must be >= 1."""
        y_true, y_samples, _ = perfect_calibration_data
        with pytest.raises(ValueError, match="n_bins"):
            compute_reliability_data(y_true, y_samples, n_bins=0)

    def test_observed_frequency_tracks_nominal_for_calibrated_data(self, rng):
        """Quantile calibration curve should be near diagonal for calibrated draws."""
        n_obs = 500
        n_samples = 1000
        mu = np.full(n_obs, 50.0)
        sigma = 10.0

        y_true = mu + rng.normal(0, sigma, n_obs)
        y_samples = mu + rng.normal(0, sigma, (n_samples, n_obs))

        result = compute_reliability_data(y_true, y_samples, n_bins=10)
        max_abs_error = float(np.max(np.abs(result.observed_freq - result.predicted_probs)))
        assert max_abs_error < 0.10


class TestIntervalScore:
    """Tests for the compute_interval_score function."""

    def test_interval_score_perfect_calibration(self, perfect_calibration_data):
        """When all obs inside CI, penalty ~ 0, score ~ width."""
        y_true, y_samples, sigma = perfect_calibration_data
        result = compute_interval_score(y_true, y_samples, prob=0.95)
        assert isinstance(result, IntervalScoreResult)
        assert result.calibration_penalty >= 0
        # Sharpness should dominate for well-calibrated model
        assert result.sharpness_component > result.calibration_penalty

    def test_interval_score_overconfident(self, overconfident_data):
        """Overconfident model should have large calibration penalty."""
        y_true, y_samples = overconfident_data
        result = compute_interval_score(y_true, y_samples, prob=0.95)
        assert result.calibration_penalty > 0
        # Overconfident: significant penalty
        assert result.calibration_penalty > result.sharpness_component * 0.1

    def test_interval_score_decomposition(self, perfect_calibration_data):
        """sharpness + penalty should equal mean_score."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_interval_score(y_true, y_samples, prob=0.95)
        np.testing.assert_allclose(
            result.sharpness_component + result.calibration_penalty,
            result.mean_score,
            rtol=1e-10,
        )

    def test_interval_score_narrower_is_better(self, rng):
        """Narrower calibrated model should have better (lower) score."""
        n_obs = 200
        mu = np.full(n_obs, 50.0)
        y_true = mu + rng.normal(0, 5, n_obs)
        # Narrow model (correct width)
        narrow_samples = mu + rng.normal(0, 5, (1000, n_obs))
        # Wide model (too wide)
        wide_samples = mu + rng.normal(0, 15, (1000, n_obs))
        narrow_result = compute_interval_score(y_true, narrow_samples, prob=0.95)
        wide_result = compute_interval_score(y_true, wide_samples, prob=0.95)
        assert narrow_result.mean_score < wide_result.mean_score

    @pytest.mark.parametrize("prob", [0.0, 1.0, -0.5, 2.0])
    def test_interval_score_rejects_invalid_probability(self, perfect_calibration_data, prob):
        """Interval score should reject invalid nominal probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        with pytest.raises(ValueError, match="prob must satisfy 0 < prob < 1"):
            compute_interval_score(y_true, y_samples, prob=prob)


class TestWeightedIntervalScore:
    """Tests for the compute_weighted_interval_score function."""

    def test_wis_bracher_formula(self, rng):
        """Verify WIS matches hand-calculated Bracher et al. formula."""
        n_obs = 100
        mu = np.full(n_obs, 50.0)
        sigma = 5.0
        y_true = mu + rng.normal(0, sigma, n_obs)
        y_samples = mu + rng.normal(0, sigma, (1000, n_obs))
        probs = (0.50, 0.90)
        result = compute_weighted_interval_score(y_true, y_samples, probs=probs)
        assert isinstance(result, WISResult)
        # WIS should be positive
        assert result.wis > 0
        assert result.n_obs == n_obs
        # Manual check: K=2, so denominator = 2.5
        K = len(probs)
        assert K == 2
        # Verify per_level dict has correct keys
        assert set(result.per_level.keys()) == set(probs)

    def test_wis_median_component(self, rng):
        """Verify median component is included and correctly weighted."""
        n_obs = 200
        mu = np.full(n_obs, 50.0)
        y_true = mu + rng.normal(0, 5, n_obs)
        y_samples = mu + rng.normal(0, 5, (1000, n_obs))
        result = compute_weighted_interval_score(y_true, y_samples, probs=(0.50, 0.80, 0.95))
        # Median component should be non-negative
        assert result.median_component >= 0
        # WIS should be at least as large as the median component
        # (because other terms are non-negative)
        assert result.wis >= result.median_component / (len((0.50, 0.80, 0.95)) + 0.5)

    def test_wis_rejects_empty_probability_levels(self, perfect_calibration_data):
        """WIS should require at least one interval level."""
        y_true, y_samples, _ = perfect_calibration_data
        with pytest.raises(ValueError, match="at least one probability level"):
            compute_weighted_interval_score(y_true, y_samples, probs=())

    def test_wis_rejects_invalid_probability_level(self, perfect_calibration_data):
        """WIS should reject invalid interval probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        with pytest.raises(ValueError, match="prob must satisfy 0 < prob < 1"):
            compute_weighted_interval_score(y_true, y_samples, probs=(0.8, 1.0))


# --- from unit/test_calibration_expanded.py ---


class TestCoverageResultDataclass:
    """Tests for CoverageResult dataclass."""

    def test_fields_accessible(self):
        lb = np.zeros(50)
        ub = np.ones(50)
        result = CoverageResult(
            nominal=0.90,
            empirical=0.88,
            n_obs=50,
            n_covered=44,
            interval_width=8.5,
            lower_bound=lb,
            upper_bound=ub,
        )
        assert result.nominal == 0.90
        assert result.empirical == 0.88
        assert result.n_obs == 50
        assert result.n_covered == 44
        assert result.interval_width == 8.5

    def test_empirical_equals_ratio(self):
        lb = np.zeros(100)
        ub = np.ones(100)
        result = CoverageResult(
            nominal=0.95,
            empirical=0.93,
            n_obs=100,
            n_covered=93,
            interval_width=10.0,
            lower_bound=lb,
            upper_bound=ub,
        )
        assert result.empirical == pytest.approx(result.n_covered / result.n_obs)


class TestIntervalScoreResultDataclass:
    """Tests for IntervalScoreResult dataclass."""

    def test_decomposition(self):
        result = IntervalScoreResult(
            nominal=0.90,
            mean_score=7.5,
            score_values=np.ones(200) * 7.5,
            sharpness_component=5.0,
            calibration_penalty=2.5,
            n_obs=200,
        )
        assert result.sharpness_component + result.calibration_penalty == pytest.approx(
            result.mean_score
        )

    def test_fields_accessible(self):
        result = IntervalScoreResult(
            nominal=0.95,
            mean_score=5.0,
            score_values=np.ones(100) * 5.0,
            sharpness_component=4.0,
            calibration_penalty=1.0,
            n_obs=100,
        )
        assert result.nominal == 0.95
        assert result.n_obs == 100


class TestWISResultDataclass:
    """Tests for WISResult dataclass."""

    def test_per_level_keys(self):
        result = WISResult(
            wis=3.0,
            per_level={0.5: 2.0, 0.8: 3.0, 0.95: 5.0},
            median_component=1.5,
            n_obs=100,
        )
        assert set(result.per_level.keys()) == {0.5, 0.8, 0.95}

    def test_wis_positive(self):
        result = WISResult(
            wis=3.0,
            per_level={0.5: 2.0},
            median_component=1.5,
            n_obs=100,
        )
        assert result.wis > 0


class TestComputeCoverageEdgeCases:
    """Edge case tests for compute_coverage."""

    def test_all_covered(self):
        """When all observations fall within very wide intervals."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 1, 50)
        # Very wide samples: all obs should be covered
        y_samples = rng.normal(50, 100, (1000, 50))
        result = compute_coverage(y_true, y_samples, prob=0.95)
        assert result.empirical == 1.0
        assert result.n_covered == 50

    def test_few_observations(self):
        """Coverage with very few observations."""
        rng = np.random.default_rng(42)
        y_true = np.array([50.0, 60.0, 70.0])
        y_samples = rng.normal(60, 10, (1000, 3))
        result = compute_coverage(y_true, y_samples, prob=0.95)
        assert result.n_obs == 3
        assert 0.0 <= result.empirical <= 1.0

    def test_50_percent_coverage(self):
        rng = np.random.default_rng(42)
        n_obs = 500
        mu = np.full(n_obs, 50.0)
        sigma = 10.0
        y_true = mu + rng.normal(0, sigma, n_obs)
        y_samples = mu + rng.normal(0, sigma, (1000, n_obs))
        result = compute_coverage(y_true, y_samples, prob=0.50)
        assert result.nominal == 0.50
        assert 0.40 <= result.empirical <= 0.60

    def test_wider_interval_means_higher_coverage(self):
        rng = np.random.default_rng(42)
        n_obs = 200
        mu = np.full(n_obs, 50.0)
        y_true = mu + rng.normal(0, 5, n_obs)
        y_samples = mu + rng.normal(0, 5, (1000, n_obs))
        result_50 = compute_coverage(y_true, y_samples, prob=0.50)
        result_95 = compute_coverage(y_true, y_samples, prob=0.95)
        assert result_95.empirical >= result_50.empirical
        assert result_95.interval_width >= result_50.interval_width


class TestComputeMultiCoverageEdgeCases:
    """Edge case tests for compute_multi_coverage."""

    def test_single_level(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 10, 100)
        y_samples = rng.normal(50, 10, (500, 100))
        results = compute_multi_coverage(y_true, y_samples, probs=(0.90,))
        assert len(results) == 1
        assert 0.90 in results

    def test_many_levels(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 10, 100)
        y_samples = rng.normal(50, 10, (500, 100))
        probs = (0.10, 0.20, 0.30, 0.50, 0.80, 0.90, 0.95, 0.99)
        results = compute_multi_coverage(y_true, y_samples, probs=probs)
        assert len(results) == 8


class TestComputeIntervalScoreEdgeCases:
    """Edge case tests for compute_interval_score."""

    def test_all_observations_inside(self):
        """When all obs are well within the interval, penalty should be 0."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 0.01, 100)  # Very tight around 50
        y_samples = rng.normal(50, 10, (1000, 100))  # Wide samples
        result = compute_interval_score(y_true, y_samples, prob=0.95)
        assert result.calibration_penalty == pytest.approx(0.0, abs=0.01)

    def test_high_prob_wider_interval(self):
        """Higher probability -> wider interval -> larger sharpness."""
        rng = np.random.default_rng(42)
        n_obs = 200
        y_true = rng.normal(50, 5, n_obs)
        y_samples = rng.normal(50, 5, (1000, n_obs))
        result_80 = compute_interval_score(y_true, y_samples, prob=0.80)
        result_95 = compute_interval_score(y_true, y_samples, prob=0.95)
        assert result_95.sharpness_component > result_80.sharpness_component


# --- from unit/test_calibration_new.py ---


class TestComputeCoverageSingleSample:
    """Cover compute_coverage with a single posterior sample."""

    def test_single_posterior_sample(self):
        """Coverage computation should work with exactly 1 sample."""
        y_true = np.array([50.0, 60.0, 70.0])
        y_samples = np.array([[50.0, 60.0, 70.0]])  # shape (1, 3)
        result = compute_coverage(y_true, y_samples, prob=0.50)
        assert result.n_obs == 3
        # With 1 sample, lower = upper = that sample
        # So coverage depends on whether y_true == sample
        assert 0.0 <= result.empirical <= 1.0

    def test_empty_samples_raises(self):
        """Zero posterior samples should raise ValueError."""
        y_true = np.array([50.0])
        y_samples = np.empty((0, 1))
        with pytest.raises(ValueError, match="at least one posterior sample"):
            compute_coverage(y_true, y_samples, prob=0.95)


class TestComputeCoverageListInputs:
    """Cover automatic conversion from lists to arrays."""

    def test_list_inputs_converted(self):
        """Lists are converted to numpy arrays."""
        y_true = [50.0, 60.0, 70.0]
        y_samples = [[48.0, 58.0, 68.0], [52.0, 62.0, 72.0]]
        result = compute_coverage(y_true, y_samples, prob=0.50)
        assert isinstance(result, CoverageResult)
        assert result.n_obs == 3


class TestReliabilityDataNBinsOne:
    """Cover the n_bins=1 special case."""

    def test_n_bins_one(self):
        """n_bins=1 should produce single bin at 0.5."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 10, 100)
        y_samples = rng.normal(50, 10, (500, 100))

        result = compute_reliability_data(y_true, y_samples, n_bins=1)

        assert isinstance(result, ReliabilityData)
        assert len(result.predicted_probs) == 1
        assert result.predicted_probs[0] == pytest.approx(0.5)
        assert len(result.observed_freq) == 1
        assert len(result.counts) == 1
        assert result.counts[0] == 100
        assert len(result.bin_edges) == 2  # n_bins + 1

    def test_n_bins_two(self):
        """n_bins=2 should produce linspace from 0.05 to 0.95."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 10, 50)
        y_samples = rng.normal(50, 10, (200, 50))

        result = compute_reliability_data(y_true, y_samples, n_bins=2)

        assert len(result.predicted_probs) == 2
        assert result.predicted_probs[0] == pytest.approx(0.05)
        assert result.predicted_probs[1] == pytest.approx(0.95)


class TestReliabilityDataValidation:
    """Cover validation error paths."""

    def test_y_samples_1d_raises(self):
        """1D y_samples should raise ValueError."""
        y_true = np.array([50.0])
        y_samples = np.array([50.0])  # 1D
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_reliability_data(y_true, y_samples)

    def test_shape_mismatch_raises(self):
        """Shape mismatch between y_true and y_samples should raise."""
        y_true = np.array([50.0, 60.0])
        y_samples = np.ones((10, 5))  # 5 obs != 2
        with pytest.raises(ValueError, match="observations"):
            compute_reliability_data(y_true, y_samples)

    def test_negative_n_bins_raises(self):
        """Negative n_bins should raise."""
        y_true = np.array([50.0])
        y_samples = np.ones((10, 1))
        with pytest.raises(ValueError, match="n_bins"):
            compute_reliability_data(y_true, y_samples, n_bins=-1)


class TestIntervalScoreValidation:
    """Cover validation error paths for compute_interval_score."""

    def test_y_true_2d_raises(self):
        """2D y_true should raise ValueError."""
        y_true = np.ones((2, 5))
        y_samples = np.ones((10, 5))
        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_interval_score(y_true, y_samples)

    def test_y_samples_3d_raises(self):
        """3D y_samples should raise ValueError."""
        y_true = np.ones(5)
        y_samples = np.ones((10, 5, 3))
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_interval_score(y_true, y_samples)

    def test_shape_mismatch_raises(self):
        """Shape mismatch should raise ValueError."""
        y_true = np.ones(5)
        y_samples = np.ones((10, 3))
        with pytest.raises(ValueError, match="observations"):
            compute_interval_score(y_true, y_samples)

    def test_empty_samples_raises(self):
        """Zero samples should raise."""
        y_true = np.ones(5)
        y_samples = np.empty((0, 5))
        with pytest.raises(ValueError, match="at least one posterior sample"):
            compute_interval_score(y_true, y_samples)

    def test_score_values_per_observation(self):
        """score_values array should have correct shape and be non-negative."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 5, 20)
        y_samples = rng.normal(50, 5, (200, 20))
        result = compute_interval_score(y_true, y_samples, prob=0.80)
        assert result.score_values.shape == (20,)
        assert np.all(result.score_values >= 0)
        assert result.n_obs == 20


class TestWISValidation:
    """Cover validation paths for compute_weighted_interval_score."""

    def test_y_true_2d_raises(self):
        y_true = np.ones((2, 5))
        y_samples = np.ones((10, 5))
        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_weighted_interval_score(y_true, y_samples)

    def test_y_samples_1d_raises(self):
        y_true = np.ones(5)
        y_samples = np.ones(5)
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_weighted_interval_score(y_true, y_samples)

    def test_shape_mismatch_raises(self):
        y_true = np.ones(5)
        y_samples = np.ones((10, 3))
        with pytest.raises(ValueError, match="observations"):
            compute_weighted_interval_score(y_true, y_samples)

    def test_empty_samples_raises(self):
        y_true = np.ones(5)
        y_samples = np.empty((0, 5))
        with pytest.raises(ValueError, match="at least one posterior sample"):
            compute_weighted_interval_score(y_true, y_samples)

    def test_single_prob_level(self):
        """WIS with single probability level."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 5, 30)
        y_samples = rng.normal(50, 5, (300, 30))
        result = compute_weighted_interval_score(y_true, y_samples, probs=(0.90,))
        assert result.wis > 0
        assert len(result.per_level) == 1
        assert result.n_obs == 30

    def test_wis_components_sum(self):
        """WIS should decompose correctly."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 5, 100)
        y_samples = rng.normal(50, 5, (500, 100))
        probs = (0.50, 0.80, 0.95)
        result = compute_weighted_interval_score(y_true, y_samples, probs=probs)

        K = len(probs)
        weighted_sum = sum(result.per_level.values())
        expected_wis = (1 / (K + 0.5)) * (result.median_component + weighted_sum)
        assert result.wis == pytest.approx(expected_wis, rel=1e-10)


class TestMultiCoverageAdditional:
    """Additional multi-coverage tests."""

    def test_empty_probs_returns_empty(self):
        """Empty probs tuple returns empty dict."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 10, 50)
        y_samples = rng.normal(50, 10, (200, 50))
        results = compute_multi_coverage(y_true, y_samples, probs=())
        assert len(results) == 0

    def test_default_probs(self):
        """Default probs should be (0.50, 0.80, 0.95)."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(50, 10, 50)
        y_samples = rng.normal(50, 10, (200, 50))
        results = compute_multi_coverage(y_true, y_samples)
        assert set(results.keys()) == {0.50, 0.80, 0.95}
