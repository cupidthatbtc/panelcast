"""Expanded tests for models/bayes/fit.py: MCMCConfig, get_gpu_info, FitResult."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import arviz as az
import pytest

from panelcast.models.bayes.fit import FitResult, MCMCConfig, get_gpu_info


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
