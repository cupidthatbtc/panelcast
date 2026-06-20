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
import pytest

from panelcast.models.bayes.fit import FitResult, MCMCConfig, get_gpu_info

# =============================================================================
# Tests for MCMCConfig
# =============================================================================


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


# =============================================================================
# Tests for get_gpu_info
# =============================================================================


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


# =============================================================================
# Tests for FitResult
# =============================================================================


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
