"""Expanded tests for calibration module: CoverageResult, IntervalScoreResult, WISResult."""

import numpy as np
import pytest

from panelcast.evaluation.calibration import (
    CoverageResult,
    IntervalScoreResult,
    WISResult,
    compute_coverage,
    compute_interval_score,
    compute_multi_coverage,
    compute_weighted_interval_score,
)


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
