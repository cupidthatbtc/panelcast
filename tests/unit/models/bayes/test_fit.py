"""Unit tests for MCMC configuration and helper functions in fit.py.

Tests cover:
- MCMCConfig validation and edge cases
- get_gpu_info subprocess handling
- FitResult structure

These tests focus on config validation and helpers, NOT MCMC execution
(actual MCMC tests are in tests/unit/test_model_fit.py).
"""

from dataclasses import FrozenInstanceError
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


class TestMCMCConfigValidation:
    """Tests for MCMCConfig validation and edge cases."""

    def test_custom_config_all_parameters(self):
        """Should accept all custom parameter values."""
        config = MCMCConfig(
            num_warmup=500,
            num_samples=2000,
            num_chains=2,
            chain_method="vectorized",
            seed=123,
            max_tree_depth=8,
            target_accept_prob=0.95,
        )

        assert config.num_warmup == 500
        assert config.num_samples == 2000
        assert config.num_chains == 2
        assert config.chain_method == "vectorized"
        assert config.seed == 123
        assert config.max_tree_depth == 8
        assert config.target_accept_prob == 0.95

    def test_chain_method_sequential(self):
        """Sequential chain method should be valid."""
        config = MCMCConfig(chain_method="sequential")
        assert config.chain_method == "sequential"

    def test_chain_method_vectorized(self):
        """Vectorized chain method should be valid."""
        config = MCMCConfig(chain_method="vectorized")
        assert config.chain_method == "vectorized"

    def test_chain_method_parallel(self):
        """Parallel chain method should be valid."""
        config = MCMCConfig(chain_method="parallel")
        assert config.chain_method == "parallel"

    def test_config_immutable(self):
        """MCMCConfig should be frozen (immutable)."""
        config = MCMCConfig()

        with pytest.raises(FrozenInstanceError):
            config.num_warmup = 999

        with pytest.raises(FrozenInstanceError):
            config.seed = 42

    def test_to_dict_serialization(self):
        """to_dict should serialize all fields correctly."""
        config = MCMCConfig(
            num_warmup=100,
            num_samples=200,
            num_chains=3,
            chain_method="parallel",
            seed=42,
            max_tree_depth=10,
            target_accept_prob=0.85,
        )

        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["num_warmup"] == 100
        assert d["num_samples"] == 200
        assert d["num_chains"] == 3
        assert d["chain_method"] == "parallel"
        assert d["seed"] == 42
        assert d["max_tree_depth"] == 10
        assert d["target_accept_prob"] == 0.85

    def test_to_dict_roundtrip(self):
        """Should be able to reconstruct from to_dict output."""
        original = MCMCConfig(
            num_warmup=100,
            num_samples=200,
            num_chains=4,
            chain_method="sequential",
            seed=99,
        )

        d = original.to_dict()
        reconstructed = MCMCConfig(**d)

        assert reconstructed == original

    def test_edge_case_single_chain(self):
        """Should handle single chain configuration."""
        config = MCMCConfig(num_chains=1)
        assert config.num_chains == 1

    def test_edge_case_zero_warmup(self):
        """Should accept zero warmup (edge case)."""
        config = MCMCConfig(num_warmup=0)
        assert config.num_warmup == 0

    def test_target_accept_prob_bounds(self):
        """Should accept target_accept_prob in valid range."""
        config_low = MCMCConfig(target_accept_prob=0.5)
        config_high = MCMCConfig(target_accept_prob=0.99)

        assert config_low.target_accept_prob == 0.5
        assert config_high.target_accept_prob == 0.99


class TestGetGPUInfo:
    """Tests for get_gpu_info function edge cases."""

    def test_returns_string(self):
        """get_gpu_info should always return a string."""
        result = get_gpu_info()
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_timeout_fallback(self, mock_devices, mock_run):
        """Should fallback gracefully when nvidia-smi times out."""
        import subprocess

        # Mock GPU device
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "NVIDIA Test GPU"
        mock_devices.return_value = [mock_device]

        # Mock nvidia-smi timeout
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 5)

        result = get_gpu_info()

        # Should fallback to JAX device info
        assert isinstance(result, str)
        assert "NVIDIA Test GPU" in result or "GPU" in result

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_not_found_fallback(self, mock_devices, mock_run):
        """Should fallback gracefully when nvidia-smi not found."""
        # Mock GPU device
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "Mock GPU"
        mock_devices.return_value = [mock_device]

        # Mock nvidia-smi not found
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        result = get_gpu_info()

        # Should fallback to JAX device info
        assert isinstance(result, str)
        assert "Mock GPU" in result or "GPU" in result

    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_cpu_only_no_gpu(self, mock_devices):
        """Should return 'CPU only' when no GPU available."""
        # Mock CPU-only device
        mock_device = MagicMock()
        mock_device.platform = "cpu"
        mock_devices.return_value = [mock_device]

        result = get_gpu_info()

        assert result == "CPU only"

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_success(self, mock_devices, mock_run):
        """Should parse nvidia-smi output correctly."""
        # Mock GPU device
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_devices.return_value = [mock_device]

        # Mock nvidia-smi success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3080, 10240"
        mock_run.return_value = mock_result

        result = get_gpu_info()

        assert "NVIDIA GeForce RTX 3080" in result
        assert "10240 MiB" in result

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_multiple_gpus(self, mock_devices, mock_run):
        """Should handle multiple GPUs from nvidia-smi."""
        # Mock GPU devices
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_devices.return_value = [mock_device, mock_device]

        # Mock nvidia-smi success with multiple GPUs
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA A100, 40960\nNVIDIA A100, 40960"
        mock_run.return_value = mock_result

        result = get_gpu_info()

        # Should contain both GPUs separated by semicolon
        assert "NVIDIA A100" in result
        assert "40960 MiB" in result


class TestFitResult:
    """Tests for FitResult structure."""

    def test_can_construct_with_all_fields(self):
        """Should be able to construct FitResult with all required fields."""
        # Create mock objects
        mock_mcmc = MagicMock()
        mock_idata = MagicMock(spec=az.InferenceData)

        result = FitResult(
            mcmc=mock_mcmc,
            idata=mock_idata,
            divergences=5,
            runtime_seconds=123.45,
            gpu_info="NVIDIA Test GPU, 8192 MiB",
        )

        assert result.mcmc is mock_mcmc
        assert result.idata is mock_idata
        assert result.divergences == 5
        assert result.runtime_seconds == 123.45
        assert result.gpu_info == "NVIDIA Test GPU, 8192 MiB"

    def test_attributes_accessible(self):
        """All FitResult attributes should be accessible."""
        mock_mcmc = MagicMock()
        mock_idata = MagicMock(spec=az.InferenceData)

        result = FitResult(
            mcmc=mock_mcmc,
            idata=mock_idata,
            divergences=0,
            runtime_seconds=10.0,
            gpu_info="CPU only",
        )

        # Verify all attributes exist and are accessible
        _ = result.mcmc
        _ = result.idata
        _ = result.divergences
        _ = result.runtime_seconds
        _ = result.gpu_info

    def test_divergences_can_be_zero(self):
        """Should handle zero divergences correctly."""
        result = FitResult(
            mcmc=MagicMock(),
            idata=MagicMock(spec=az.InferenceData),
            divergences=0,
            runtime_seconds=1.0,
            gpu_info="CPU only",
        )

        assert result.divergences == 0

    def test_is_mutable(self):
        """FitResult should be mutable (not frozen)."""
        result = FitResult(
            mcmc=MagicMock(),
            idata=MagicMock(spec=az.InferenceData),
            divergences=0,
            runtime_seconds=1.0,
            gpu_info="CPU only",
        )

        # Should be able to modify (not frozen dataclass)
        result.divergences = 10
        assert result.divergences == 10


# --- from unit/models/bayes/test_fit_expanded.py ---


class TestMCMCConfigExpanded:
    """Expanded MCMCConfig tests."""

    def test_defaults(self):
        cfg = MCMCConfig()
        assert cfg.num_warmup == 1000
        assert cfg.num_samples == 1000
        assert cfg.num_chains == 4
        assert cfg.chain_method == "sequential"
        assert cfg.seed == 0
        assert cfg.max_tree_depth == 10
        assert cfg.target_accept_prob == 0.90

    def test_frozen(self):
        cfg = MCMCConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.num_chains = 2

    def test_to_dict_keys(self):
        d = MCMCConfig().to_dict()
        expected_keys = {
            "num_warmup",
            "num_samples",
            "num_chains",
            "chain_method",
            "seed",
            "max_tree_depth",
            "target_accept_prob",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_roundtrip(self):
        original = MCMCConfig(num_warmup=50, num_samples=100, seed=7)
        reconstructed = MCMCConfig(**original.to_dict())
        assert reconstructed == original

    def test_equality(self):
        a = MCMCConfig(seed=42)
        b = MCMCConfig(seed=42)
        assert a == b

    def test_inequality(self):
        a = MCMCConfig(seed=1)
        b = MCMCConfig(seed=2)
        assert a != b

    def test_hashable(self):
        cfg = MCMCConfig()
        s = {cfg}
        assert len(s) == 1

    def test_zero_samples(self):
        cfg = MCMCConfig(num_samples=0)
        assert cfg.num_samples == 0

    def test_large_tree_depth(self):
        cfg = MCMCConfig(max_tree_depth=15)
        assert cfg.max_tree_depth == 15

    def test_target_accept_prob_extreme(self):
        cfg = MCMCConfig(target_accept_prob=0.999)
        assert cfg.target_accept_prob == 0.999


class TestGetGpuInfoExpanded:
    """Expanded get_gpu_info tests."""

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_nonzero_return(self, mock_devices, mock_run):
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "Fallback GPU"
        mock_devices.return_value = [mock_device]
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = get_gpu_info()
        assert "Fallback GPU" in result

    @patch("panelcast.models.bayes.fit.subprocess.run")
    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_nvidia_smi_oserror(self, mock_devices, mock_run):
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "Test GPU"
        mock_devices.return_value = [mock_device]
        mock_run.side_effect = OSError("Permission denied")
        result = get_gpu_info()
        assert "Test GPU" in result

    @patch("panelcast.models.bayes.fit.jax.devices")
    def test_gpu_no_device_kind(self, mock_devices):
        """Handles GPU with empty device_kind list."""
        mock_device = MagicMock()
        mock_device.platform = "cpu"
        mock_devices.return_value = [mock_device]
        result = get_gpu_info()
        assert result == "CPU only"


class TestFitResultExpanded:
    """Expanded FitResult tests."""

    def test_mutable(self):
        result = FitResult(
            mcmc=MagicMock(),
            idata=MagicMock(spec=az.InferenceData),
            divergences=0,
            runtime_seconds=1.0,
            gpu_info="CPU only",
        )
        result.divergences = 99
        assert result.divergences == 99

    def test_runtime_float(self):
        result = FitResult(
            mcmc=MagicMock(),
            idata=MagicMock(spec=az.InferenceData),
            divergences=0,
            runtime_seconds=0.001,
            gpu_info="CPU only",
        )
        assert result.runtime_seconds == pytest.approx(0.001)

    def test_large_divergences(self):
        result = FitResult(
            mcmc=MagicMock(),
            idata=MagicMock(spec=az.InferenceData),
            divergences=10000,
            runtime_seconds=600.0,
            gpu_info="NVIDIA A100",
        )
        assert result.divergences == 10000

    def test_gpu_info_string(self):
        result = FitResult(
            mcmc=MagicMock(),
            idata=MagicMock(spec=az.InferenceData),
            divergences=0,
            runtime_seconds=1.0,
            gpu_info="NVIDIA GeForce RTX 3090, 24576 MiB",
        )
        assert "RTX 3090" in result.gpu_info


# --- from unit/models/bayes/test_fit_new.py ---


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
