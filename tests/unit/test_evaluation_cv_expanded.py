"""Expanded tests for evaluation/cv.py: LOOResult, compute_loo, add_log_likelihood_to_idata."""

import arviz as az
import numpy as np
import pytest
import xarray as xr

from panelcast.evaluation.cv import (
    LOOResult,
    add_log_likelihood_to_idata,
    compute_loo,
)


def _make_idata_with_loglik(n_chains=4, n_draws=500, n_obs=50):
    """Helper to create InferenceData with log_likelihood group."""
    np.random.seed(42)
    posterior = xr.Dataset(
        {
            "mu": xr.DataArray(
                np.random.randn(n_chains, n_draws),
                dims=["chain", "draw"],
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            )
        }
    )
    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(
                np.zeros((n_chains, n_draws), dtype=bool),
                dims=["chain", "draw"],
            )
        }
    )
    # Stable log-likelihood to avoid noisy diagnostics
    obs_effect = np.linspace(-0.2, 0.2, n_obs)[None, None, :]
    log_lik = -100.0 + obs_effect + np.random.randn(n_chains, n_draws, n_obs) * 0.05
    log_likelihood = xr.Dataset(
        {
            "y": xr.DataArray(
                log_lik,
                dims=["chain", "draw", "y_dim_0"],
                coords={
                    "chain": range(n_chains),
                    "draw": range(n_draws),
                    "y_dim_0": range(n_obs),
                },
            )
        }
    )
    return az.InferenceData(
        posterior=posterior,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
    )


class TestLOOResult:
    """Tests for LOOResult dataclass."""

    def test_fields_accessible(self):
        result = LOOResult(
            loo=None,
            elpd_loo=-500.0,
            se_elpd=20.0,
            p_loo=10.5,
            n_high_pareto_k=2,
            high_pareto_k_indices=np.array([5, 10]),
            warning="Some warning",
        )
        assert result.elpd_loo == -500.0
        assert result.se_elpd == 20.0
        assert result.p_loo == 10.5
        assert result.n_high_pareto_k == 2
        assert len(result.high_pareto_k_indices) == 2
        assert result.warning == "Some warning"

    def test_no_warning(self):
        result = LOOResult(
            loo=None,
            elpd_loo=-300.0,
            se_elpd=15.0,
            p_loo=5.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        assert result.warning is None
        assert result.n_high_pareto_k == 0

    def test_mutable(self):
        result = LOOResult(
            loo=None,
            elpd_loo=-300.0,
            se_elpd=15.0,
            p_loo=5.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        result.elpd_loo = -400.0
        assert result.elpd_loo == -400.0


class TestAddLogLikelihoodToIdata:
    """Tests for add_log_likelihood_to_idata."""

    def test_adds_group(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray(
                    np.random.randn(2, 100),
                    dims=["chain", "draw"],
                )
            }
        )
        idata = az.InferenceData(posterior=posterior)
        log_lik_da = xr.DataArray(
            np.random.randn(2, 100, 50),
            dims=["chain", "draw", "obs"],
        )
        result = add_log_likelihood_to_idata(idata, log_lik_da)
        assert "log_likelihood" in result.groups()

    def test_custom_var_name(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray(
                    np.random.randn(2, 100),
                    dims=["chain", "draw"],
                )
            }
        )
        idata = az.InferenceData(posterior=posterior)
        log_lik_da = xr.DataArray(
            np.random.randn(2, 100, 30),
            dims=["chain", "draw", "obs"],
        )
        result = add_log_likelihood_to_idata(idata, log_lik_da, var_name="custom")
        assert "custom" in result.log_likelihood

    def test_returns_same_idata(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray(
                    np.random.randn(2, 100),
                    dims=["chain", "draw"],
                )
            }
        )
        idata = az.InferenceData(posterior=posterior)
        log_lik_da = xr.DataArray(
            np.random.randn(2, 100, 20),
            dims=["chain", "draw", "obs"],
        )
        result = add_log_likelihood_to_idata(idata, log_lik_da)
        assert result is idata  # should modify in place


class TestComputeLoo:
    """Tests for compute_loo."""

    def test_returns_loo_result(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata)
        assert isinstance(result, LOOResult)

    def test_elpd_is_negative(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata)
        assert result.elpd_loo < 0

    def test_se_positive(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata)
        assert result.se_elpd > 0

    def test_p_loo_nonnegative(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata)
        assert result.p_loo >= 0

    def test_high_pareto_k_count(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata)
        assert result.n_high_pareto_k >= 0
        assert len(result.high_pareto_k_indices) == result.n_high_pareto_k

    def test_missing_log_likelihood_raises(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray(
                    np.random.randn(2, 100),
                    dims=["chain", "draw"],
                )
            }
        )
        idata = az.InferenceData(posterior=posterior)
        with pytest.raises(ValueError, match="log_likelihood"):
            compute_loo(idata)

    def test_with_var_name(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata, var_name="y")
        assert isinstance(result, LOOResult)

    def test_loo_data_attribute(self):
        idata = _make_idata_with_loglik()
        result = compute_loo(idata)
        assert result.loo is not None
