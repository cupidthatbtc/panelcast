"""Expanded tests for convergence diagnostics: ConvergenceDiagnostics, check_convergence."""

from dataclasses import FrozenInstanceError

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


class TestConvergenceDiagnosticsDataclass:
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
