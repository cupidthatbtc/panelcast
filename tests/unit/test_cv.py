"""Tests for cross-validation and predictive check functions.

These tests verify the LOO-CV and predictive check functionality
using mock/synthetic data. Full integration tests with actual
MCMC fitting are deferred to integration tests.
"""

from unittest.mock import MagicMock, patch

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.evaluation.cv import (
    LOOResult,
    add_log_likelihood_to_idata,
    compare_models,
    compute_log_likelihood,
    compute_loo,
    generate_prior_predictive,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_log_likelihood_da():
    """Create mock log-likelihood xr.DataArray with known values."""
    n_chains = 4
    n_draws = 100
    n_obs = 50

    # Create realistic-looking log-likelihood values (negative)
    np.random.seed(42)
    log_lik_values = np.random.normal(loc=-2.0, scale=0.5, size=(n_chains, n_draws, n_obs))

    return xr.DataArray(
        log_lik_values,
        dims=["chain", "draw", "obs"],
        coords={
            "chain": range(n_chains),
            "draw": range(n_draws),
            "obs": range(n_obs),
        },
    )


@pytest.fixture
def mock_idata_with_log_lik(mock_log_likelihood_da):
    """Create mock InferenceData with log_likelihood group."""
    # Create minimal posterior for InferenceData
    posterior = xr.Dataset(
        {
            "beta": xr.DataArray(
                np.random.normal(size=(4, 100, 5)),
                dims=["chain", "draw", "beta_dim"],
            ),
        }
    )

    idata = az.InferenceData(posterior=posterior)

    # Add log_likelihood group
    log_lik_ds = xr.Dataset({"y": mock_log_likelihood_da})
    idata.add_groups(log_likelihood=log_lik_ds)

    return idata


@pytest.fixture
def mock_idata_without_log_lik():
    """Create mock InferenceData WITHOUT log_likelihood group."""
    posterior = xr.Dataset(
        {
            "beta": xr.DataArray(
                np.random.normal(size=(4, 100, 5)),
                dims=["chain", "draw", "beta_dim"],
            ),
        }
    )

    return az.InferenceData(posterior=posterior)


@pytest.fixture
def mock_mcmc():
    """Create a mock MCMC object for testing."""
    mcmc = MagicMock()
    mcmc.num_chains = 4
    mcmc.num_samples = 100

    # Mock get_samples to return flattened samples
    np.random.seed(42)
    mock_samples = {
        "user_beta": np.random.normal(size=(400, 5)),  # 4 chains * 100 samples
        "user_rho": np.random.normal(size=(400,)),
    }
    mcmc.get_samples.return_value = mock_samples

    return mcmc


# ============================================================================
# Tests for compute_log_likelihood
# ============================================================================


class TestComputeLogLikelihood:
    """Tests for log-likelihood computation."""

    def test_compute_log_likelihood_shape(self, mock_mcmc):
        """Verify output has (chain, draw, obs) dims."""
        n_obs = 50

        # Mock model and log_likelihood function
        def mock_model(*args, **kwargs):
            pass

        mock_log_lik = {
            "user_y": np.random.normal(size=(400, n_obs)),  # flattened samples
        }

        with patch("panelcast.evaluation.cv.log_likelihood", return_value=mock_log_lik):
            model_args = {
                "artist_idx": np.zeros(n_obs, dtype=int),
                "album_seq": np.ones(n_obs, dtype=int),
                "prev_score": np.zeros(n_obs),
                "X": np.random.normal(size=(n_obs, 5)),
                "y": np.random.normal(size=n_obs),
                "n_artists": 10,
                "max_seq": 5,
            }

            result = compute_log_likelihood(mock_model, mock_mcmc, model_args, obs_name="user_y")

        # Check dims
        assert result.dims == ("chain", "draw", "obs")
        assert result.shape == (4, 100, n_obs)

    def test_compute_log_likelihood_dtype(self, mock_mcmc):
        """Verify output has float dtype."""
        n_obs = 30

        def mock_model(*args, **kwargs):
            pass

        mock_log_lik = {
            "user_y": np.random.normal(size=(400, n_obs)),
        }

        with patch("panelcast.evaluation.cv.log_likelihood", return_value=mock_log_lik):
            model_args = {
                "artist_idx": np.zeros(n_obs, dtype=int),
                "album_seq": np.ones(n_obs, dtype=int),
                "prev_score": np.zeros(n_obs),
                "X": np.random.normal(size=(n_obs, 5)),
                "y": np.random.normal(size=n_obs),
                "n_artists": 10,
                "max_seq": 5,
            }

            result = compute_log_likelihood(mock_model, mock_mcmc, model_args, obs_name="user_y")

        assert np.issubdtype(result.dtype, np.floating)

    def test_compute_log_likelihood_missing_site_raises(self, mock_mcmc):
        """Verify KeyError when observation site not found."""

        def mock_model(*args, **kwargs):
            pass

        # Return log-likelihood with different site name
        mock_log_lik = {
            "other_y": np.random.normal(size=(400, 30)),
        }

        with patch("panelcast.evaluation.cv.log_likelihood", return_value=mock_log_lik):
            model_args = {
                "artist_idx": np.zeros(30, dtype=int),
                "album_seq": np.ones(30, dtype=int),
                "prev_score": np.zeros(30),
                "X": np.random.normal(size=(30, 5)),
                "y": np.random.normal(size=30),
                "n_artists": 10,
                "max_seq": 5,
            }

            with pytest.raises(KeyError) as exc_info:
                compute_log_likelihood(mock_model, mock_mcmc, model_args, obs_name="user_y")

            assert "user_y" in str(exc_info.value)
            assert "other_y" in str(exc_info.value)


# ============================================================================
# Tests for add_log_likelihood_to_idata
# ============================================================================


class TestAddLogLikelihoodToIdata:
    """Tests for adding log-likelihood to InferenceData."""

    def test_add_log_likelihood_creates_group(
        self, mock_idata_without_log_lik, mock_log_likelihood_da
    ):
        """idata without log_likelihood -> adds it."""
        idata = mock_idata_without_log_lik
        assert "log_likelihood" not in idata.groups()

        result = add_log_likelihood_to_idata(idata, mock_log_likelihood_da, var_name="y")

        assert "log_likelihood" in result.groups()
        assert "y" in result.log_likelihood.data_vars

    def test_add_log_likelihood_preserves_existing(
        self, mock_idata_with_log_lik, mock_log_likelihood_da
    ):
        """idata with log_likelihood -> preserves and adds new variable."""
        idata = mock_idata_with_log_lik
        assert "log_likelihood" in idata.groups()
        assert "y" in idata.log_likelihood.data_vars

        # Create a new log-likelihood with different name
        new_log_lik = mock_log_likelihood_da.copy()

        result = add_log_likelihood_to_idata(idata, new_log_lik, var_name="new_y")

        assert "log_likelihood" in result.groups()
        assert "y" in result.log_likelihood.data_vars  # Original preserved
        assert "new_y" in result.log_likelihood.data_vars  # New added


# ============================================================================
# Tests for LOOResult and compute_loo
# ============================================================================


class TestComputeLoo:
    """Tests for LOO-CV computation."""

    def test_compute_loo_returns_result(self, mock_idata_with_log_lik):
        """Verify compute_loo returns LOOResult with populated fields."""
        result = compute_loo(mock_idata_with_log_lik, var_name="y")

        assert isinstance(result, LOOResult)
        assert result.loo is not None
        assert isinstance(result.elpd_loo, float)
        assert isinstance(result.se_elpd, float)
        assert isinstance(result.p_loo, float)
        assert isinstance(result.n_high_pareto_k, int)
        assert isinstance(result.high_pareto_k_indices, np.ndarray)

    def test_compute_loo_pareto_k_detection(self, mock_idata_with_log_lik):
        """Mock high k values -> n_high_pareto_k correct."""
        # Get result - with random data, we may or may not have high k
        result = compute_loo(mock_idata_with_log_lik)

        # Verify n_high_pareto_k matches indices length
        assert result.n_high_pareto_k == len(result.high_pareto_k_indices)

    def test_compute_loo_no_log_likelihood_raises(self, mock_idata_without_log_lik):
        """Raise ValueError when log_likelihood group missing."""
        with pytest.raises(ValueError) as exc_info:
            compute_loo(mock_idata_without_log_lik)

        assert "log_likelihood" in str(exc_info.value)

    def test_compute_loo_no_high_k_with_good_data(self):
        """All k < 0.7 -> n_high_pareto_k == 0 (with well-behaved data)."""
        # Create well-behaved log-likelihood data with small variation across samples
        # but variation across observations to avoid degenerate importance weights
        n_chains = 4
        n_draws = 1000  # More samples for stable PSIS
        n_obs = 20

        # Create realistic log-likelihood: different base values per observation
        # with small variation across samples (well-fitting model)
        np.random.seed(42)
        # Base log-likelihood varies by observation
        base_log_lik = np.linspace(-1.5, -0.5, n_obs)
        # Add small random variation across samples
        log_lik_values = base_log_lik[None, None, :] + np.random.normal(
            0, 0.01, size=(n_chains, n_draws, n_obs)
        )

        log_lik_da = xr.DataArray(
            log_lik_values,
            dims=["chain", "draw", "obs"],
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "obs": range(n_obs),
            },
        )

        posterior = xr.Dataset(
            {
                "beta": xr.DataArray(
                    np.random.normal(size=(4, 1000, 5)),
                    dims=["chain", "draw", "beta_dim"],
                ),
            }
        )

        idata = az.InferenceData(posterior=posterior)
        log_lik_ds = xr.Dataset({"y": log_lik_da})
        idata.add_groups(log_likelihood=log_lik_ds)

        result = compute_loo(idata)

        # With well-behaved log-likelihood (small sample variation, no outliers),
        # Pareto-k should be low for all observations
        assert result.n_high_pareto_k == 0


# ============================================================================
# Tests for compare_models
# ============================================================================


class TestCompareModels:
    """Tests for model comparison."""

    def test_compare_models_returns_dataframe(self, mock_idata_with_log_lik):
        """Verify compare_models returns DataFrame with expected columns."""
        # Create two models with different log-likelihoods
        idata1 = mock_idata_with_log_lik

        # Create second model with different log-likelihood
        n_chains = 4
        n_draws = 100
        n_obs = 50

        np.random.seed(123)
        log_lik_values2 = np.random.normal(loc=-2.5, scale=0.5, size=(n_chains, n_draws, n_obs))

        log_lik_da2 = xr.DataArray(
            log_lik_values2,
            dims=["chain", "draw", "obs"],
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "obs": range(n_obs),
            },
        )

        posterior2 = xr.Dataset(
            {
                "beta": xr.DataArray(
                    np.random.normal(size=(4, 100, 5)),
                    dims=["chain", "draw", "beta_dim"],
                ),
            }
        )

        idata2 = az.InferenceData(posterior=posterior2)
        log_lik_ds2 = xr.Dataset({"y": log_lik_da2})
        idata2.add_groups(log_likelihood=log_lik_ds2)

        result = compare_models({"model_a": idata1, "model_b": idata2})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "model_a" in result.index
        assert "model_b" in result.index
        # Check for expected columns (ArviZ compare returns these)
        assert "elpd_loo" in result.columns
        assert "p_loo" in result.columns

    def test_compare_models_ranks_by_elpd(self, mock_idata_with_log_lik):
        """Model with higher elpd ranked first."""
        # Create "better" model with higher (less negative) log-likelihood
        n_chains = 4
        n_draws = 100
        n_obs = 50

        # Better model: higher log-likelihood (closer to 0)
        np.random.seed(42)
        log_lik_better = np.random.normal(loc=-1.0, scale=0.3, size=(n_chains, n_draws, n_obs))

        log_lik_da_better = xr.DataArray(
            log_lik_better,
            dims=["chain", "draw", "obs"],
            coords={
                "chain": range(n_chains),
                "draw": range(n_draws),
                "obs": range(n_obs),
            },
        )

        posterior_better = xr.Dataset(
            {
                "beta": xr.DataArray(
                    np.random.normal(size=(4, 100, 5)),
                    dims=["chain", "draw", "beta_dim"],
                ),
            }
        )

        idata_better = az.InferenceData(posterior=posterior_better)
        idata_better.add_groups(log_likelihood=xr.Dataset({"y": log_lik_da_better}))

        # Use the fixture as "worse" model (mean=-2.0)
        result = compare_models(
            {
                "better_model": idata_better,
                "worse_model": mock_idata_with_log_lik,
            }
        )

        # Better model should be ranked first (rank 0)
        assert result.loc["better_model", "rank"] == 0
        assert result.loc["worse_model", "rank"] == 1


# ============================================================================
# Tests for generate_prior_predictive
# ============================================================================


class TestGeneratePriorPredictive:
    """Tests for prior predictive sampling."""

    def test_generate_prior_predictive_shape(self):
        """Verify num_samples dimension."""
        # Create a simple mock model
        import numpyro
        import numpyro.distributions as dist

        def simple_model(X, y=None, n_artists=10, max_seq=5, **kwargs):
            """Simple model for testing."""
            beta = numpyro.sample("user_beta", dist.Normal(0, 1).expand([X.shape[1]]).to_event(1))
            mu = X @ beta
            sigma = numpyro.sample("user_sigma", dist.HalfNormal(1))
            with numpyro.plate("obs", X.shape[0]):
                numpyro.sample("user_y", dist.Normal(mu, sigma), obs=y)

        n_obs = 30
        n_features = 5
        model_args = {
            "X": np.random.normal(size=(n_obs, n_features)),
            "y": None,
            "n_artists": 10,
            "max_seq": 5,
        }

        result = generate_prior_predictive(simple_model, model_args, num_samples=500)

        assert "user_y" in result
        assert result["user_y"].shape[0] == 500  # num_samples
        assert result["user_y"].shape[1] == n_obs

    def test_generate_prior_predictive_reproducible(self):
        """Same seed -> same output."""
        import numpyro
        import numpyro.distributions as dist

        def simple_model(X, y=None, **kwargs):
            """Simple model for testing."""
            beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1]]).to_event(1))
            mu = X @ beta
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))
            with numpyro.plate("obs", X.shape[0]):
                numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

        model_args = {
            "X": np.random.normal(size=(20, 3)),
            "y": None,
        }

        result1 = generate_prior_predictive(simple_model, model_args, num_samples=100, seed=42)
        result2 = generate_prior_predictive(simple_model, model_args, num_samples=100, seed=42)

        np.testing.assert_array_equal(result1["y"], result2["y"])
        np.testing.assert_array_equal(result1["beta"], result2["beta"])

    def test_generate_prior_predictive_different_seeds(self):
        """Different seeds -> different output."""
        import numpyro
        import numpyro.distributions as dist

        def simple_model(X, y=None, **kwargs):
            """Simple model for testing."""
            beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1]]).to_event(1))
            mu = X @ beta
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))
            with numpyro.plate("obs", X.shape[0]):
                numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

        model_args = {
            "X": np.random.normal(size=(20, 3)),
            "y": None,
        }

        result1 = generate_prior_predictive(simple_model, model_args, num_samples=100, seed=42)
        result2 = generate_prior_predictive(simple_model, model_args, num_samples=100, seed=123)

        # Different seeds should produce different results
        assert not np.allclose(result1["y"], result2["y"])
