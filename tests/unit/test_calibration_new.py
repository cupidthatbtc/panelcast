"""New coverage tests for evaluation/calibration.py.

Targets uncovered code paths (91% -> higher):
- compute_coverage: single posterior sample edge case, empty samples validation
- compute_reliability_data: n_bins=1 special case, y_samples 2D validation,
  shape mismatch validation
- compute_interval_score: shape validation (y_true 2D, y_samples 1D/3D,
  shape mismatch), empty samples validation
- compute_weighted_interval_score: y_true not 1D, y_samples not 2D,
  shape mismatch, empty samples
"""

import numpy as np
import pytest

from panelcast.evaluation.calibration import (
    CoverageResult,
    ReliabilityData,
    compute_coverage,
    compute_interval_score,
    compute_multi_coverage,
    compute_reliability_data,
    compute_weighted_interval_score,
)

# ===========================================================================
# compute_coverage: edge cases
# ===========================================================================


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


# ===========================================================================
# compute_reliability_data: edge cases
# ===========================================================================


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


# ===========================================================================
# compute_interval_score: validation edge cases
# ===========================================================================


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


# ===========================================================================
# compute_weighted_interval_score: validation edge cases
# ===========================================================================


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


# ===========================================================================
# compute_multi_coverage: additional
# ===========================================================================


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
