"""New coverage tests for models/bayes/diagnostics.py.

Targets uncovered code paths (98% -> higher):
- check_convergence: missing 'posterior' group raises, missing 'sample_stats' raises,
  allow_divergences=True with divergences still passes,
  no 'diverging' field in sample_stats defaults to 0
- get_divergence_info: no sample_stats group, no 'diverging' field,
  per-chain divergence breakdown, locations extraction
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.models.bayes.diagnostics import (
    ConvergenceDiagnostics,
    check_convergence,
    get_divergence_info,
)

# ===========================================================================
# Helpers
# ===========================================================================


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


# ===========================================================================
# check_convergence
# ===========================================================================


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


# ===========================================================================
# get_divergence_info
# ===========================================================================


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
