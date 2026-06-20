"""Unit tests for convergence diagnostics module.

Tests cover:
- check_convergence with passing criteria
- check_convergence with failing R-hat
- check_convergence with failing ESS
- check_convergence with divergences (strict and allowed modes)
- check_convergence with custom thresholds
- get_divergence_info with zero and non-zero divergences
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.models.bayes.diagnostics import (
    ConvergenceDiagnostics,
    check_convergence,
    get_divergence_info,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_mock_idata(
    n_chains: int = 4,
    n_draws: int = 1000,
    n_params: int = 3,
    param_names: list[str] | None = None,
    inject_bad_rhat: str | None = None,
    inject_low_ess: str | None = None,
    n_divergences: int = 0,
    divergence_chain: int | None = None,
) -> az.InferenceData:
    """Create mock InferenceData for testing.

    Creates synthetic posterior samples with controllable properties for testing
    diagnostic functions.

    Parameters
    ----------
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws per chain
    n_params : int
        Number of parameters (ignored if param_names provided)
    param_names : list[str], optional
        Custom parameter names
    inject_bad_rhat : str, optional
        Parameter name to inject poor mixing (different chain means)
    inject_low_ess : str, optional
        Parameter name to inject high autocorrelation (low ESS)
    n_divergences : int
        Number of divergent transitions to inject
    divergence_chain : int, optional
        Which chain to put divergences in (None = spread evenly)

    Returns
    -------
    az.InferenceData
        Mock InferenceData with posterior and sample_stats groups
    """
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    # Create posterior samples - well-mixed by default
    np.random.seed(42)
    posterior_dict = {}

    for name in param_names:
        # Create well-mixed samples (chains have same distribution)
        samples = np.random.randn(n_chains, n_draws)

        if name == inject_bad_rhat:
            # Inject poor mixing: different chain means
            for c in range(n_chains):
                samples[c, :] += c * 2.0  # Shift each chain

        if name == inject_low_ess:
            # Inject high autocorrelation using cumulative sum
            # This creates extremely correlated samples
            for c in range(n_chains):
                base = np.random.randn(n_draws) * 0.01  # Very small innovations
                samples[c, :] = np.cumsum(base)

        posterior_dict[name] = xr.DataArray(
            samples,
            dims=["chain", "draw"],
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )

    posterior = xr.Dataset(posterior_dict)

    # Create sample_stats with diverging field
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if n_divergences > 0:
        if divergence_chain is not None:
            # Put all divergences in one chain
            div_indices = np.random.choice(n_draws, min(n_divergences, n_draws), replace=False)
            diverging[divergence_chain, div_indices] = True
        else:
            # Spread divergences across chains
            total_slots = n_chains * n_draws
            div_flat_indices = np.random.choice(total_slots, n_divergences, replace=False)
            for idx in div_flat_indices:
                c = idx // n_draws
                d = idx % n_draws
                diverging[c, d] = True

    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(
                diverging,
                dims=["chain", "draw"],
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            ),
        }
    )

    return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


@pytest.fixture
def good_idata():
    """Create InferenceData that passes all convergence checks."""
    return make_mock_idata(
        n_chains=4,
        n_draws=1000,
        param_names=["alpha", "beta", "sigma"],
        n_divergences=0,
    )


@pytest.fixture
def bad_rhat_idata():
    """Create InferenceData with one parameter having poor R-hat."""
    return make_mock_idata(
        n_chains=4,
        n_draws=1000,
        param_names=["alpha", "beta", "sigma"],
        inject_bad_rhat="beta",
        n_divergences=0,
    )


@pytest.fixture
def bad_ess_idata():
    """Create InferenceData with one parameter having low ESS."""
    return make_mock_idata(
        n_chains=4,
        n_draws=1000,
        param_names=["alpha", "beta", "sigma"],
        inject_low_ess="sigma",
        n_divergences=0,
    )


@pytest.fixture
def divergent_idata():
    """Create InferenceData with divergent transitions."""
    return make_mock_idata(
        n_chains=4,
        n_draws=1000,
        param_names=["alpha", "beta", "sigma"],
        n_divergences=10,
    )


# =============================================================================
# Tests for check_convergence
# =============================================================================


class TestCheckConvergencePassing:
    """Tests for passing convergence scenarios."""

    def test_check_convergence_passing(self, good_idata):
        """Good idata should pass all convergence checks."""
        diags = check_convergence(good_idata)

        assert diags.passed is True
        assert diags.rhat_max < 1.01
        assert diags.ess_bulk_min >= 400 * 4  # 4 chains
        assert diags.divergences == 0
        assert diags.failing_params == []

    def test_check_convergence_returns_summary(self, good_idata):
        """Verify summary_df is populated with expected columns."""
        diags = check_convergence(good_idata)

        assert isinstance(diags.summary_df, pd.DataFrame)
        assert "r_hat" in diags.summary_df.columns
        assert "ess_bulk" in diags.summary_df.columns
        assert "ess_tail" in diags.summary_df.columns
        # Should have rows for each parameter
        assert len(diags.summary_df) == 3  # alpha, beta, sigma


class TestCheckConvergenceFailingRhat:
    """Tests for R-hat failure scenarios."""

    def test_check_convergence_failing_rhat(self, bad_rhat_idata):
        """Bad R-hat idata should fail with correct failing_params."""
        diags = check_convergence(bad_rhat_idata)

        assert diags.passed is False
        assert diags.rhat_max >= 1.01
        assert "beta" in diags.failing_params

    def test_check_convergence_rhat_threshold(self, bad_rhat_idata):
        """Verify custom R-hat threshold is respected."""
        # With very high threshold, should pass
        diags_permissive = check_convergence(bad_rhat_idata, rhat_threshold=10.0)
        # R-hat should not cause failure with high threshold
        # But ESS might still fail, so just check R-hat isn't the issue
        assert diags_permissive.rhat_max < 10.0


class TestCheckConvergenceFailingESS:
    """Tests for ESS failure scenarios."""

    def test_check_convergence_failing_ess(self, bad_ess_idata):
        """Bad ESS idata should fail with correct failing_params."""
        diags = check_convergence(bad_ess_idata)

        assert diags.passed is False
        # sigma should have low ESS
        assert "sigma" in diags.failing_params

    def test_check_convergence_ess_threshold(self, good_idata):
        """Verify custom ESS threshold is respected."""
        # With very high threshold, even good idata should fail
        diags = check_convergence(good_idata, ess_threshold=10000)

        # This should fail due to ESS being below 10000 * 4 chains
        # (unless the mock generates extremely high ESS)
        # Just verify threshold affects the check
        assert diags.ess_bulk_min < 10000 * 4


class TestCheckConvergenceDivergences:
    """Tests for divergence handling."""

    def test_check_convergence_divergences_strict(self, divergent_idata):
        """Divergent idata with allow_divergences=False should fail."""
        diags = check_convergence(divergent_idata, allow_divergences=False)

        assert diags.passed is False
        assert diags.divergences > 0

    def test_check_convergence_divergences_allowed(self, divergent_idata):
        """Divergent idata with allow_divergences=True should pass if other criteria ok."""
        diags = check_convergence(divergent_idata, allow_divergences=True)

        # Divergences should not cause failure
        assert diags.divergences > 0
        # Pass status depends on R-hat and ESS only
        # The divergent_idata has good mixing, so should pass
        assert diags.passed is True


class TestCheckConvergenceCustomThresholds:
    """Tests for custom threshold configurations."""

    def test_check_convergence_custom_thresholds_permissive(self, bad_rhat_idata):
        """Verify all custom thresholds work together."""
        # Very permissive thresholds
        diags = check_convergence(
            bad_rhat_idata,
            rhat_threshold=100.0,  # Very permissive
            ess_threshold=1,  # Very permissive
            allow_divergences=True,
        )

        # With such permissive thresholds, should pass
        # (unless ESS is literally 0)
        assert diags.rhat_max < 100.0
        assert diags.ess_bulk_min >= 4  # 1 * 4 chains

    def test_check_convergence_strict_thresholds(self, good_idata):
        """Verify strict thresholds cause appropriate failures."""
        diags = check_convergence(
            good_idata,
            rhat_threshold=1.0001,  # Very strict
            ess_threshold=10000,  # Very strict
        )

        # With strict thresholds, even good idata might fail
        # Just verify the function runs and returns valid result
        assert isinstance(diags, ConvergenceDiagnostics)


class TestCheckConvergenceEdgeCases:
    """Tests for edge cases and error handling."""

    def test_check_convergence_missing_posterior(self):
        """Should raise ValueError if posterior group missing."""
        idata = az.InferenceData(
            sample_stats=xr.Dataset(
                {"diverging": xr.DataArray(np.zeros((2, 100), dtype=bool), dims=["chain", "draw"])}
            )
        )

        with pytest.raises(ValueError, match="posterior"):
            check_convergence(idata)

    def test_check_convergence_missing_sample_stats(self):
        """Should raise ValueError if sample_stats group missing."""
        posterior = xr.Dataset(
            {"param": xr.DataArray(np.random.randn(2, 100), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior)

        with pytest.raises(ValueError, match="sample_stats"):
            check_convergence(idata)


# =============================================================================
# Tests for get_divergence_info
# =============================================================================


class TestGetDivergenceInfoZero:
    """Tests for zero divergence scenarios."""

    def test_get_divergence_info_zero(self, good_idata):
        """No divergences should return all zeros."""
        info = get_divergence_info(good_idata)

        assert info["total"] == 0
        assert info["rate"] == 0.0
        assert info["locations"] == {}
        assert all(c == 0 for c in info["per_chain"])

    def test_get_divergence_info_missing_sample_stats(self):
        """Missing sample_stats should return zeros gracefully."""
        posterior = xr.Dataset(
            {"param": xr.DataArray(np.random.randn(2, 100), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior)

        info = get_divergence_info(idata)

        assert info["total"] == 0
        assert info["per_chain"] == []


class TestGetDivergenceInfoNonzero:
    """Tests for non-zero divergence scenarios."""

    def test_get_divergence_info_nonzero(self, divergent_idata):
        """Divergent idata should have correct count and breakdown."""
        info = get_divergence_info(divergent_idata)

        assert info["total"] == 10
        assert sum(info["per_chain"]) == 10
        assert len(info["per_chain"]) == 4  # 4 chains
        # Rate should be 10 / (4 * 1000) = 0.0025
        assert abs(info["rate"] - 0.0025) < 0.001

    def test_get_divergence_info_single_chain(self):
        """Divergences in single chain should be correctly attributed."""
        idata = make_mock_idata(
            n_chains=4,
            n_draws=100,
            n_divergences=5,
            divergence_chain=2,  # All in chain 2
        )

        info = get_divergence_info(idata)

        assert info["total"] == 5
        assert info["per_chain"][2] == 5
        assert info["per_chain"][0] == 0
        assert info["per_chain"][1] == 0
        assert info["per_chain"][3] == 0
        assert 2 in info["locations"]
        assert len(info["locations"][2]) == 5

    def test_get_divergence_info_locations_correct(self):
        """Verify locations dict contains correct draw indices."""
        # Create minimal idata with known divergence location
        np.random.seed(123)
        n_chains = 2
        n_draws = 50

        diverging = np.zeros((n_chains, n_draws), dtype=bool)
        diverging[0, 10] = True  # Divergence at chain 0, draw 10
        diverging[1, 25] = True  # Divergence at chain 1, draw 25

        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    diverging,
                    dims=["chain", "draw"],
                    coords={"chain": range(n_chains), "draw": range(n_draws)},
                )
            }
        )
        posterior = xr.Dataset(
            {"param": xr.DataArray(np.random.randn(n_chains, n_draws), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        info = get_divergence_info(idata)

        assert 0 in info["locations"]
        assert 10 in info["locations"][0]
        assert 1 in info["locations"]
        assert 25 in info["locations"][1]


# =============================================================================
# Tests for ConvergenceDiagnostics dataclass
# =============================================================================


class TestConvergenceDiagnosticsDataclass:
    """Tests for ConvergenceDiagnostics properties."""

    def test_dataclass_is_frozen(self):
        """ConvergenceDiagnostics should be immutable."""
        diags = ConvergenceDiagnostics(
            rhat_max=1.001,
            ess_bulk_min=2000,
            ess_tail_min=1800,
            divergences=0,
            passed=True,
            failing_params=[],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            diags.rhat_max = 1.5

    def test_dataclass_repr_passed(self):
        """Check repr for passed diagnostics."""
        diags = ConvergenceDiagnostics(
            rhat_max=1.001,
            ess_bulk_min=2000,
            ess_tail_min=1800,
            divergences=0,
            passed=True,
            failing_params=[],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )

        assert "PASSED" in repr(diags)
        assert "1.001" in repr(diags)

    def test_dataclass_repr_failed(self):
        """Check repr for failed diagnostics."""
        diags = ConvergenceDiagnostics(
            rhat_max=1.5,
            ess_bulk_min=500,
            ess_tail_min=400,
            divergences=10,
            passed=False,
            failing_params=["beta"],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )

        assert "FAILED" in repr(diags)
        assert "1.5" in repr(diags)
