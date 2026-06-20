"""New coverage tests for models/bayes/fit.py.

Targets uncovered code paths (90% -> higher):
- fit_model: full execution with heavily mocked MCMC/NUTS,
  exclude_from_idata filtering, exclude_from_idata removing all sites,
  config=None default, divergence warning logging, n_reviews in model_args,
  n_ref + n_ref_method in model_args, extra_fields with multi-dim data,
  1D parameter variables, multi-dimensional parameter variables
- get_gpu_info: nvidia-smi with malformed output (single column),
  GPU present but no gpu devices in fallback path
"""

from unittest.mock import MagicMock, patch

import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from panelcast.models.bayes.fit import (
    FitResult,
    MCMCConfig,
    fit_model,
    get_gpu_info,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_mcmc(
    n_chains=2, n_draws=10, n_features=3, n_artists=5, divergences=0, include_extra_dims=False
):
    """Create a mock MCMC object that returns realistic sample shapes."""
    mock_mcmc = MagicMock()

    # get_samples(group_by_chain=True) -> dict with (chains, draws, ...) shapes
    samples = {
        "user_beta": np.random.randn(n_chains, n_draws, n_features),
        "user_rho": np.random.randn(n_chains, n_draws),
        "user_sigma_obs": np.abs(np.random.randn(n_chains, n_draws)) + 0.1,
        "user_mu_artist": np.random.randn(n_chains, n_draws),
        "user_sigma_artist": np.abs(np.random.randn(n_chains, n_draws)) + 0.1,
        "user_init_artist_effect": np.random.randn(n_chains, n_draws, n_artists),
    }
    if include_extra_dims:
        # 3D parameter: rw_raw with (chains, draws, n_artists, max_seq-1)
        samples["user_rw_raw"] = np.random.randn(n_chains, n_draws, n_artists, 3)

    mock_mcmc.get_samples.return_value = samples

    # get_extra_fields() -> flat arrays (n_chains * n_draws,)
    total = n_chains * n_draws
    div_arr = np.zeros(total, dtype=bool)
    if divergences > 0:
        div_arr[:divergences] = True
    extra_fields = {
        "diverging": div_arr,
        "num_steps": np.ones(total, dtype=int) * 5,
    }
    mock_mcmc.get_extra_fields.return_value = extra_fields

    return mock_mcmc


def _make_model_args(n_obs=20, n_features=3, include_n_reviews=False, include_n_ref=False):
    """Create model_args dict for testing."""
    rng = np.random.default_rng(42)
    args = {
        "artist_idx": np.repeat(np.arange(5), n_obs // 5),
        "album_seq": np.tile(np.arange(1, n_obs // 5 + 1), 5),
        "prev_score": rng.normal(50, 10, n_obs),
        "X": rng.normal(0, 1, (n_obs, n_features)),
        "y": rng.normal(60, 10, n_obs),
        "n_artists": 5,
        "max_seq": n_obs // 5,
    }
    if include_n_reviews:
        args["n_reviews"] = rng.integers(1, 500, n_obs).astype(float)
    if include_n_ref:
        args["n_ref"] = 50.0
        args["n_ref_method"] = "median"
    return args


# ===========================================================================
# fit_model
# ===========================================================================


class TestFitModelBasic:
    """Test fit_model execution with mocked MCMC infrastructure."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_basic_execution(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """fit_model completes and returns FitResult."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2, seed=42)

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
        )

        assert isinstance(result, FitResult)
        assert result.divergences == 0
        assert result.gpu_info == "CPU only"
        assert result.runtime_seconds >= 0

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_config_none_uses_default(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """config=None should use MCMCConfig() defaults."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=None,
            progress_bar=False,
        )

        assert isinstance(result, FitResult)

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_divergence_logged(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """Divergences should be counted and returned."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc(divergences=3)
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2)

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
        )

        assert result.divergences == 3


class TestFitModelExcludeFromIdata:
    """Cover exclude_from_idata parameter."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_exclude_sites(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """Excluded sites should not appear in InferenceData."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc(include_extra_dims=True)
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2)

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
            exclude_from_idata=("user_rw_raw",),
        )

        # rw_raw should not be in posterior
        assert "user_rw_raw" not in result.idata.posterior

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_exclude_all_sites_raises(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """Excluding all sites should raise ValueError."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        # Create MCMC with only one sample site
        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {
            "only_var": np.random.randn(2, 10),
        }
        mock_mcmc.get_extra_fields.return_value = {
            "diverging": np.zeros(20, dtype=bool),
        }
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()

        with pytest.raises(ValueError, match="removed all sample sites"):
            fit_model(
                model=lambda **kw: None,
                model_args=model_args,
                config=MCMCConfig(num_warmup=5, num_samples=10, num_chains=2),
                progress_bar=False,
                exclude_from_idata=("only_var",),
            )


class TestFitModelWithNReviews:
    """Cover n_reviews in model_args for heteroscedastic models."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_n_reviews_included_in_constant_data(
        self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls
    ):
        """n_reviews should appear in constant_data when present."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args(include_n_reviews=True)
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2)

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
        )

        assert "n_reviews" in result.idata.constant_data


class TestFitModelWithNRef:
    """Cover n_ref and n_ref_method in model_args."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_n_ref_included_in_constant_data(
        self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls
    ):
        """n_ref and n_ref_method should appear in constant_data."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args(include_n_ref=True)
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2)

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
        )

        assert "n_ref" in result.idata.constant_data
        assert "n_ref_method" in result.idata.constant_data


class TestFitModelMissingRequiredKeys:
    """Cover missing required keys validation."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_missing_y_raises(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """Missing 'y' in model_args should raise ValueError."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()
        del model_args["y"]

        with pytest.raises(ValueError, match="missing required key 'y'"):
            fit_model(
                model=lambda **kw: None,
                model_args=model_args,
                config=MCMCConfig(num_warmup=5, num_samples=10, num_chains=2),
                progress_bar=False,
            )

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_missing_X_raises(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """Missing 'X' in model_args should raise ValueError."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()
        del model_args["X"]

        with pytest.raises(ValueError, match="missing required key 'X'"):
            fit_model(
                model=lambda **kw: None,
                model_args=model_args,
                config=MCMCConfig(num_warmup=5, num_samples=10, num_chains=2),
                progress_bar=False,
            )


class TestFitModelMetadataKeys:
    """Cover metadata key exclusion from mcmc.run()."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_n_ref_method_excluded_from_run(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """n_ref_method should not be passed to mcmc.run()."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args(include_n_ref=True)
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2)

        fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
        )

        # Check that mcmc.run was called without n_ref_method
        run_call_kwargs = mock_mcmc.run.call_args
        # run is called as mcmc.run(rng_key, extra_fields=..., **run_args)
        if run_call_kwargs.kwargs:
            assert "n_ref_method" not in run_call_kwargs.kwargs


class TestFitModelIdataGroups:
    """Cover InferenceData group construction."""

    @patch("panelcast.models.bayes.fit.MCMC")
    @patch("panelcast.models.bayes.fit.NUTS")
    @patch("panelcast.models.bayes.fit.get_gpu_info")
    @patch("panelcast.models.bayes.fit.jax.default_backend")
    def test_idata_has_required_groups(self, mock_backend, mock_gpu, mock_nuts, mock_mcmc_cls):
        """InferenceData should have posterior, sample_stats, observed_data, constant_data."""
        mock_backend.return_value = "cpu"
        mock_gpu.return_value = "CPU only"
        mock_nuts.return_value = MagicMock()

        mock_mcmc = _make_mock_mcmc()
        mock_mcmc_cls.return_value = mock_mcmc

        model_args = _make_model_args()
        config = MCMCConfig(num_warmup=5, num_samples=10, num_chains=2)

        result = fit_model(
            model=lambda **kw: None,
            model_args=model_args,
            config=config,
            progress_bar=False,
        )

        groups = set(result.idata.groups())
        assert "posterior" in groups
        assert "sample_stats" in groups
        assert "observed_data" in groups
        assert "constant_data" in groups


# ===========================================================================
# get_gpu_info: additional edge cases
# ===========================================================================


class TestGetGpuInfoAdditional:
    """Additional edge cases for get_gpu_info."""

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_malformed_output(self, mock_devices, mock_run):
        """nvidia-smi with malformed output (missing comma) uses line as-is."""
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_devices.return_value = [mock_device]

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Some Malformed GPU Info"
        mock_run.return_value = mock_result

        result = get_gpu_info()
        assert "Some Malformed GPU Info" in result

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_gpu_devices_empty_after_filter(self, mock_devices, mock_run):
        """GPU platform exists but nvidia-smi fails, fallback to device_kind."""
        mock_gpu = MagicMock()
        mock_gpu.platform = "gpu"
        mock_gpu.device_kind = "CUDA GPU"

        mock_cpu = MagicMock()
        mock_cpu.platform = "cpu"
        mock_cpu.device_kind = "cpu"

        mock_devices.return_value = [mock_gpu, mock_cpu]
        mock_run.side_effect = FileNotFoundError()

        result = get_gpu_info()
        assert "CUDA GPU" in result
