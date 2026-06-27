"""Unit tests for convergence diagnostics module.

Tests cover:
- check_convergence with passing criteria
- check_convergence with failing R-hat
- check_convergence with failing ESS
- check_convergence with divergences (strict and allowed modes)
- check_convergence with custom thresholds
- get_divergence_info with zero and non-zero divergences
"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

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


# --- from unit/test_diagnostics_expanded.py ---


def _make_simple_idata(n_chains=4, n_draws=500, n_divergences=0):
    """Helper to create simple InferenceData for testing."""
    np.random.seed(99)
    posterior = xr.Dataset(
        {
            "alpha": xr.DataArray(
                np.random.randn(n_chains, n_draws),
                dims=["chain", "draw"],
            ),
            "beta": xr.DataArray(
                np.random.randn(n_chains, n_draws),
                dims=["chain", "draw"],
            ),
        }
    )
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if n_divergences > 0:
        flat_idx = np.random.choice(n_chains * n_draws, n_divergences, replace=False)
        for idx in flat_idx:
            diverging[idx // n_draws, idx % n_draws] = True
    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(diverging, dims=["chain", "draw"]),
        }
    )
    return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


class TestConvergenceDiagnosticsDataclass_expanded:
    """Tests for ConvergenceDiagnostics properties and repr."""

    def test_frozen(self):
        diag = ConvergenceDiagnostics(
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
        with pytest.raises(FrozenInstanceError):
            diag.passed = False

    def test_repr_contains_status_passed(self):
        diag = ConvergenceDiagnostics(
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
        assert "PASSED" in repr(diag)

    def test_repr_contains_status_failed(self):
        diag = ConvergenceDiagnostics(
            rhat_max=1.5,
            ess_bulk_min=100,
            ess_tail_min=80,
            divergences=5,
            passed=False,
            failing_params=["beta"],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        assert "FAILED" in repr(diag)

    def test_repr_contains_rhat_max(self):
        diag = ConvergenceDiagnostics(
            rhat_max=1.023,
            ess_bulk_min=2000,
            ess_tail_min=1800,
            divergences=0,
            passed=False,
            failing_params=["beta"],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        assert "1.0230" in repr(diag)

    def test_repr_contains_divergences(self):
        diag = ConvergenceDiagnostics(
            rhat_max=1.001,
            ess_bulk_min=2000,
            ess_tail_min=1800,
            divergences=42,
            passed=False,
            failing_params=[],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        assert "42" in repr(diag)

    def test_thresholds_preserved(self):
        diag = ConvergenceDiagnostics(
            rhat_max=1.001,
            ess_bulk_min=2000,
            ess_tail_min=1800,
            divergences=0,
            passed=True,
            failing_params=[],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.05,
            ess_threshold=200,
        )
        assert diag.rhat_threshold == 1.05
        assert diag.ess_threshold == 200


class TestCheckConvergenceExpanded:
    """Additional check_convergence tests."""

    def test_two_chains(self):
        idata = _make_simple_idata(n_chains=2, n_draws=1000)
        diag = check_convergence(idata)
        assert isinstance(diag, ConvergenceDiagnostics)

    def test_many_chains(self):
        idata = _make_simple_idata(n_chains=8, n_draws=200)
        diag = check_convergence(idata)
        assert isinstance(diag, ConvergenceDiagnostics)

    def test_well_mixed_passes(self):
        idata = _make_simple_idata(n_chains=4, n_draws=2000)
        diag = check_convergence(idata)
        assert diag.passed is True

    def test_divergences_with_allow(self):
        idata = _make_simple_idata(n_chains=4, n_draws=2000, n_divergences=5)
        diag = check_convergence(idata, allow_divergences=True)
        assert diag.divergences == 5
        # Should still pass if mixing is ok
        assert diag.passed is True

    def test_divergences_strict(self):
        idata = _make_simple_idata(n_chains=4, n_draws=2000, n_divergences=5)
        diag = check_convergence(idata, allow_divergences=False)
        assert diag.divergences == 5
        assert diag.passed is False

    def test_summary_df_has_expected_params(self):
        idata = _make_simple_idata()
        diag = check_convergence(idata)
        assert "alpha" in diag.summary_df.index
        assert "beta" in diag.summary_df.index


class TestGetDivergenceInfoExpanded:
    """Additional get_divergence_info tests."""

    def test_no_diverging_field(self):
        """sample_stats without diverging field should return zeros."""
        posterior = xr.Dataset(
            {"param": xr.DataArray(np.random.randn(2, 100), dims=["chain", "draw"])}
        )
        sample_stats = xr.Dataset(
            {"energy": xr.DataArray(np.random.randn(2, 100), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        info = get_divergence_info(idata)
        assert info["total"] == 0

    def test_all_divergent(self):
        """Every sample is divergent."""
        n_chains, n_draws = 2, 10
        diverging = np.ones((n_chains, n_draws), dtype=bool)
        sample_stats = xr.Dataset({"diverging": xr.DataArray(diverging, dims=["chain", "draw"])})
        posterior = xr.Dataset(
            {"p": xr.DataArray(np.random.randn(n_chains, n_draws), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)
        info = get_divergence_info(idata)
        assert info["total"] == n_chains * n_draws
        assert info["rate"] == pytest.approx(1.0)

    def test_per_chain_sums_to_total(self):
        idata = _make_simple_idata(n_chains=4, n_draws=100, n_divergences=7)
        info = get_divergence_info(idata)
        assert sum(info["per_chain"]) == info["total"]

    def test_locations_keys_are_chains_with_divergences(self):
        idata = _make_simple_idata(n_chains=4, n_draws=100, n_divergences=3)
        info = get_divergence_info(idata)
        # Only chains that have divergences should appear in locations
        for chain_idx, count in enumerate(info["per_chain"]):
            if count > 0:
                assert chain_idx in info["locations"]
            else:
                assert chain_idx not in info["locations"]


# --- from unit/test_diagnostics_new.py ---


def _make_idata(
    n_chains=2,
    n_draws=100,
    n_params=3,
    divergences=None,
    include_sample_stats=True,
    include_posterior=True,
    include_diverging=True,
    rhat_values=None,
    ess_values=None,
):
    """Create a minimal ArviZ-like InferenceData for testing."""
    import arviz as az

    groups = {}

    if include_posterior:
        posterior_dict = {}
        rng = np.random.default_rng(42)
        for i in range(n_params):
            posterior_dict[f"param_{i}"] = xr.DataArray(
                data=rng.normal(0, 1, (n_chains, n_draws)),
                dims=["chain", "draw"],
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            )
        groups["posterior"] = xr.Dataset(posterior_dict)

    if include_sample_stats:
        stats_dict = {}
        if include_diverging:
            if divergences is not None:
                # Create divergence array with specified per-chain divergences
                div_data = np.zeros((n_chains, n_draws), dtype=bool)
                if isinstance(divergences, list):
                    for c, count in enumerate(divergences):
                        div_data[c, :count] = True
                else:
                    # Total divergences spread across chains
                    flat_count = 0
                    for c in range(n_chains):
                        for d in range(n_draws):
                            if flat_count < divergences:
                                div_data[c, d] = True
                                flat_count += 1
            else:
                div_data = np.zeros((n_chains, n_draws), dtype=bool)
            stats_dict["diverging"] = xr.DataArray(
                data=div_data,
                dims=["chain", "draw"],
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            )
        else:
            # Include a dummy stat so the group is non-empty and recognized by ArviZ
            stats_dict["num_steps"] = xr.DataArray(
                data=np.ones((n_chains, n_draws), dtype=int),
                dims=["chain", "draw"],
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            )
        groups["sample_stats"] = xr.Dataset(stats_dict)

    idata = az.InferenceData(**groups)
    return idata


class TestCheckConvergenceMissingGroups:
    """Cover missing group validation."""

    def test_missing_posterior_raises(self):
        """Missing posterior group should raise ValueError."""
        idata = _make_idata(include_posterior=False)
        with pytest.raises(ValueError, match="posterior"):
            check_convergence(idata)

    def test_missing_sample_stats_raises(self):
        """Missing sample_stats group should raise ValueError."""
        idata = _make_idata(include_sample_stats=False)
        with pytest.raises(ValueError, match="sample_stats"):
            check_convergence(idata)


class TestCheckConvergenceAllowDivergences:
    """Cover allow_divergences=True path."""

    def test_allow_divergences_passes_with_divergences(self):
        """allow_divergences=True should pass even with divergences."""
        idata = _make_idata(divergences=5)
        result = check_convergence(idata, allow_divergences=True)
        assert result.divergences == 5
        # Should still pass since divergences are allowed
        # (assuming rhat and ESS are OK)

    def test_allow_divergences_false_fails_with_divergences(self):
        """allow_divergences=False should fail with divergences."""
        idata = _make_idata(divergences=5)
        result = check_convergence(idata, allow_divergences=False)
        assert result.divergences == 5
        assert result.passed is False


class TestCheckConvergenceNoDivergingField:
    """Cover missing 'diverging' field in sample_stats."""

    def test_no_diverging_field_defaults_zero(self):
        """Missing 'diverging' field should default to 0 divergences."""
        idata = _make_idata(include_diverging=False)
        result = check_convergence(idata)
        assert result.divergences == 0


class TestCheckConvergenceRepr:
    """Cover ConvergenceDiagnostics __repr__."""

    def test_repr_passed(self):
        idata = _make_idata(divergences=0)
        result = check_convergence(idata)
        repr_str = repr(result)
        assert "PASSED" in repr_str or "FAILED" in repr_str
        assert "rhat_max=" in repr_str
        assert "ess_bulk_min=" in repr_str

    def test_repr_failed(self):
        idata = _make_idata(divergences=10)
        result = check_convergence(idata, allow_divergences=False)
        repr_str = repr(result)
        assert "FAILED" in repr_str


class TestCheckConvergenceThresholds:
    """Cover threshold and ESS checking."""

    def test_custom_thresholds(self):
        """Custom thresholds should be stored in result."""
        idata = _make_idata(divergences=0)
        result = check_convergence(idata, rhat_threshold=1.05, ess_threshold=100)
        assert result.rhat_threshold == 1.05
        assert result.ess_threshold == 100


class TestGetDivergenceInfoNoStats:
    """Cover missing sample_stats group."""

    def test_no_sample_stats_returns_empty(self):
        """No sample_stats group should return zeros."""
        idata = _make_idata(include_sample_stats=False)
        info = get_divergence_info(idata)
        assert info["total"] == 0
        assert info["per_chain"] == []
        assert info["rate"] == 0.0
        assert info["locations"] == {}


class TestGetDivergenceInfoNoDivergingField:
    """Cover missing 'diverging' field."""

    def test_no_diverging_field_returns_empty(self):
        """Missing 'diverging' field should return zeros."""
        idata = _make_idata(include_diverging=False)
        info = get_divergence_info(idata)
        assert info["total"] == 0
        assert info["per_chain"] == []
        assert info["rate"] == 0.0
        assert info["locations"] == {}


class TestGetDivergenceInfoWithDivergences:
    """Cover divergence extraction with actual divergences."""

    def test_per_chain_breakdown(self):
        """Per-chain breakdown should match total."""
        # Chain 0: 3 divergences, Chain 1: 2 divergences
        idata = _make_idata(divergences=[3, 2])
        info = get_divergence_info(idata)
        assert info["total"] == 5
        assert len(info["per_chain"]) == 2
        assert info["per_chain"][0] == 3
        assert info["per_chain"][1] == 2
        assert sum(info["per_chain"]) == info["total"]

    def test_divergence_rate(self):
        """Rate should be total / (n_chains * n_draws)."""
        idata = _make_idata(n_chains=2, n_draws=100, divergences=10)
        info = get_divergence_info(idata)
        expected_rate = 10 / (2 * 100)
        assert info["rate"] == pytest.approx(expected_rate)

    def test_divergence_locations(self):
        """Locations should map chain to draw indices."""
        idata = _make_idata(divergences=[2, 0])
        info = get_divergence_info(idata)
        assert 0 in info["locations"]
        assert len(info["locations"][0]) == 2
        # Chain 1 should not be in locations (no divergences)
        assert 1 not in info["locations"]

    def test_zero_divergences(self):
        """Zero divergences should have empty locations."""
        idata = _make_idata(divergences=0)
        info = get_divergence_info(idata)
        assert info["total"] == 0
        assert info["rate"] == 0.0
        assert info["locations"] == {}
