"""Unit tests for posterior predictive checks module."""

import numpy as np
import pytest

from panelcast.evaluation.ppc import (
    DEFAULT_PPC_STATISTICS,
    PPCResult,
    PPCStatistic,
    compute_ppc_statistics,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPPCWellCalibrated:
    def test_ppc_well_calibrated(self, rng):
        """y_rep from same distribution → no extreme p-values."""
        n_obs = 500
        n_samples = 1000
        y_obs = rng.normal(50, 10, n_obs)
        y_rep = rng.normal(50, 10, (n_samples, n_obs))
        result = compute_ppc_statistics(y_obs, y_rep)
        extreme = result.check_extreme(lower=0.005, upper=0.995)
        assert len(extreme) == 0, f"Unexpected extreme statistics: {extreme}"

    def test_ppc_misspecified_mean(self, rng):
        """Shifted y_obs → 'mean' should be extreme."""
        n_obs = 200
        n_samples = 500
        y_obs = rng.normal(70, 10, n_obs)  # shifted up
        y_rep = rng.normal(50, 10, (n_samples, n_obs))
        result = compute_ppc_statistics(y_obs, y_rep)
        extreme = result.check_extreme(lower=0.01, upper=0.99)
        assert "mean" in extreme

    def test_ppc_misspecified_variance(self, rng):
        """Wrong variance → 'sd' should be extreme."""
        n_obs = 200
        n_samples = 500
        y_obs = rng.normal(50, 20, n_obs)  # much wider
        y_rep = rng.normal(50, 5, (n_samples, n_obs))
        result = compute_ppc_statistics(y_obs, y_rep)
        extreme = result.check_extreme(lower=0.01, upper=0.99)
        assert "sd" in extreme


class TestPPCResultFields:
    def test_ppc_result_fields(self, rng):
        """Correct n_obs, n_samples, len(statistics)."""
        n_obs = 100
        n_samples = 200
        y_obs = rng.normal(0, 1, n_obs)
        y_rep = rng.normal(0, 1, (n_samples, n_obs))
        result = compute_ppc_statistics(y_obs, y_rep)
        assert result.n_obs == n_obs
        assert result.n_samples == n_samples
        assert len(result.statistics) == len(DEFAULT_PPC_STATISTICS)

    def test_ppc_summary_dict(self, rng):
        """Summary property should return correct structure."""
        y_obs = rng.normal(0, 1, 50)
        y_rep = rng.normal(0, 1, (100, 50))
        result = compute_ppc_statistics(y_obs, y_rep)
        summary = result.summary
        assert isinstance(summary, dict)
        for name, entry in summary.items():
            assert "observed" in entry
            assert "p_value" in entry
            assert "mc_se" in entry


class TestPPCCheckExtreme:
    def test_ppc_check_extreme_configurable(self, rng):
        """Custom thresholds change which stats are flagged."""
        n_obs = 200
        n_samples = 500
        y_obs = rng.normal(50, 10, n_obs)
        y_rep = rng.normal(50, 10, (n_samples, n_obs))
        result = compute_ppc_statistics(y_obs, y_rep)
        # With very tight thresholds, more stats flagged
        extreme_tight = result.check_extreme(lower=0.40, upper=0.60)
        extreme_loose = result.check_extreme(lower=0.01, upper=0.99)
        assert len(extreme_tight) >= len(extreme_loose)


class TestPPCMCSE:
    def test_ppc_mc_se(self, rng):
        """Verify mc_se = sqrt(p*(1-p)/n_samples)."""
        y_obs = rng.normal(0, 1, 100)
        y_rep = rng.normal(0, 1, (500, 100))
        result = compute_ppc_statistics(y_obs, y_rep)
        for stat in result.statistics:
            p = stat.bayesian_p_value
            expected_se = np.sqrt(p * (1 - p) / 500)
            np.testing.assert_allclose(stat.mc_se, expected_se, rtol=1e-10)


class TestPPCCustomStatistics:
    def test_ppc_custom_statistics(self, rng):
        """Custom dict honored."""
        y_obs = rng.normal(0, 1, 50)
        y_rep = rng.normal(0, 1, (100, 50))
        custom_stats = {"my_mean": np.mean, "my_max": np.max}
        result = compute_ppc_statistics(y_obs, y_rep, statistics=custom_stats)
        assert len(result.statistics) == 2
        names = [s.name for s in result.statistics]
        assert "my_mean" in names
        assert "my_max" in names


class TestPPCConstantVector:
    def test_ppc_constant_vector_skewness(self, rng):
        """All-same y_rep row → skewness returns 0.0 (not NaN)."""
        n_obs = 50
        y_obs = rng.normal(0, 1, n_obs)
        y_rep = np.zeros((100, n_obs))  # all constant rows
        # Add one non-constant row to avoid all p-values being 0/1
        y_rep[0] = rng.normal(0, 1, n_obs)
        result = compute_ppc_statistics(y_obs, y_rep)
        skew_stat = next(s for s in result.statistics if s.name == "skewness")
        # All replicated skewness should be finite (0.0 for constant rows)
        assert np.all(np.isfinite(skew_stat.replicated_distribution))


class TestPPCInputValidation:
    def test_invalid_y_obs_shape(self, rng):
        with pytest.raises(ValueError, match="y_obs must be 1D"):
            compute_ppc_statistics(rng.normal(0, 1, (10, 5)), rng.normal(0, 1, (100, 50)))

    def test_invalid_y_rep_shape(self, rng):
        with pytest.raises(ValueError, match="y_rep must be 2D"):
            compute_ppc_statistics(rng.normal(0, 1, 50), rng.normal(0, 1, 50))

    def test_shape_mismatch(self, rng):
        with pytest.raises(ValueError, match="observations"):
            compute_ppc_statistics(rng.normal(0, 1, 50), rng.normal(0, 1, (100, 30)))

    def test_rejects_zero_replicated_samples(self, rng):
        """PPC should fail fast when y_rep has zero posterior draws."""
        y_obs = rng.normal(0, 1, 50)
        y_rep = np.empty((0, 50))
        with pytest.raises(ValueError, match="at least one replicated sample"):
            compute_ppc_statistics(y_obs, y_rep)
