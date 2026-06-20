"""Tests for JAX memory statistics module."""

from __future__ import annotations

from unittest import mock

import pytest

from panelcast.gpu_memory.measure import JaxMemoryStats, get_jax_memory_stats


class TestJaxMemoryStatsDataclass:
    """Tests for JaxMemoryStats dataclass properties."""

    def test_jax_memory_stats_peak_gb(self):
        """peak_gb converts bytes to GB (1024^3)."""
        stats = JaxMemoryStats(
            bytes_in_use=1 * 1024**3,
            peak_bytes_in_use=4 * 1024**3,
            bytes_limit=8 * 1024**3,
            bytes_reserved=2 * 1024**3,
        )
        assert stats.peak_gb == 4.0

    def test_jax_memory_stats_limit_gb(self):
        """limit_gb converts bytes to GB (1024^3)."""
        stats = JaxMemoryStats(
            bytes_in_use=1 * 1024**3,
            peak_bytes_in_use=4 * 1024**3,
            bytes_limit=8 * 1024**3,
            bytes_reserved=2 * 1024**3,
        )
        assert stats.limit_gb == 8.0

    def test_jax_memory_stats_in_use_gb(self):
        """in_use_gb converts bytes to GB (1024^3)."""
        stats = JaxMemoryStats(
            bytes_in_use=2 * 1024**3,
            peak_bytes_in_use=4 * 1024**3,
            bytes_limit=8 * 1024**3,
            bytes_reserved=1 * 1024**3,
        )
        assert stats.in_use_gb == 2.0

    def test_jax_memory_stats_reserved_gb(self):
        """reserved_gb converts bytes to GB (1024^3)."""
        stats = JaxMemoryStats(
            bytes_in_use=2 * 1024**3,
            peak_bytes_in_use=4 * 1024**3,
            bytes_limit=8 * 1024**3,
            bytes_reserved=3 * 1024**3,
        )
        assert stats.reserved_gb == 3.0

    def test_jax_memory_stats_zero_values(self):
        """Zero values produce 0.0 GB for all properties."""
        stats = JaxMemoryStats(
            bytes_in_use=0,
            peak_bytes_in_use=0,
            bytes_limit=0,
            bytes_reserved=0,
        )
        assert stats.peak_gb == 0.0
        assert stats.limit_gb == 0.0
        assert stats.in_use_gb == 0.0
        assert stats.reserved_gb == 0.0

    def test_jax_memory_stats_fractional_gb(self):
        """Fractional GB values calculated correctly."""
        # 4.5 GB in bytes
        stats = JaxMemoryStats(
            bytes_in_use=0,
            peak_bytes_in_use=int(4.5 * 1024**3),
            bytes_limit=0,
            bytes_reserved=0,
        )
        assert stats.peak_gb == pytest.approx(4.5)

    def test_jax_memory_stats_frozen(self):
        """JaxMemoryStats is immutable (frozen=True)."""
        stats = JaxMemoryStats(
            bytes_in_use=1000,
            peak_bytes_in_use=2000,
            bytes_limit=3000,
            bytes_reserved=500,
        )
        with pytest.raises(AttributeError):
            stats.peak_bytes_in_use = 5000  # type: ignore[misc]


class TestGetJaxMemoryStats:
    """Tests for get_jax_memory_stats function."""

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_no_gpu_runtime_error(self, mock_devices):
        """Raises RuntimeError with message about no GPU when jax.devices fails."""
        mock_devices.side_effect = RuntimeError("CUDA_ERROR_NO_DEVICE")

        with pytest.raises(RuntimeError) as exc_info:
            get_jax_memory_stats()

        assert "No GPU devices available for JAX" in str(exc_info.value)

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_empty_device_list(self, mock_devices):
        """Raises RuntimeError when jax.devices returns empty list."""
        mock_devices.return_value = []

        with pytest.raises(RuntimeError) as exc_info:
            get_jax_memory_stats()

        assert "No GPU devices available for JAX" in str(exc_info.value)

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_index_out_of_range(self, mock_devices):
        """Raises RuntimeError when device_index exceeds available GPUs."""
        # Single device at index 0
        mock_device = mock.Mock()
        mock_devices.return_value = [mock_device]

        with pytest.raises(RuntimeError) as exc_info:
            get_jax_memory_stats(device_index=5)

        assert "GPU index 5 out of range" in str(exc_info.value)
        assert "Available GPUs: 0-0" in str(exc_info.value)

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_index_out_of_range_multiple_gpus(self, mock_devices):
        """Error message shows correct range for multiple GPUs."""
        # Three devices at index 0, 1, 2
        mock_devices.return_value = [mock.Mock(), mock.Mock(), mock.Mock()]

        with pytest.raises(RuntimeError) as exc_info:
            get_jax_memory_stats(device_index=5)

        assert "GPU index 5 out of range" in str(exc_info.value)
        assert "Available GPUs: 0-2" in str(exc_info.value)

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_none_memory_stats(self, mock_devices):
        """Raises RuntimeError when device.memory_stats() returns None."""
        mock_device = mock.Mock()
        mock_device.memory_stats.return_value = None
        mock_devices.return_value = [mock_device]

        with pytest.raises(RuntimeError) as exc_info:
            get_jax_memory_stats()

        assert "does not support memory_stats()" in str(exc_info.value)

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_success(self, mock_devices):
        """Returns JaxMemoryStats on success."""
        mock_device = mock.Mock()
        mock_device.memory_stats.return_value = {
            "bytes_in_use": 1 * 1024**3,
            "peak_bytes_in_use": 4 * 1024**3,
            "bytes_limit": 8 * 1024**3,
            "bytes_reserved": 2 * 1024**3,
        }
        mock_devices.return_value = [mock_device]

        result = get_jax_memory_stats()

        assert isinstance(result, JaxMemoryStats)
        assert result.bytes_in_use == 1 * 1024**3
        assert result.peak_bytes_in_use == 4 * 1024**3
        assert result.bytes_limit == 8 * 1024**3
        assert result.bytes_reserved == 2 * 1024**3

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_missing_keys_use_defaults(self, mock_devices):
        """Missing keys in memory_stats dict default to 0."""
        mock_device = mock.Mock()
        # Partial stats - only peak_bytes_in_use provided
        mock_device.memory_stats.return_value = {
            "peak_bytes_in_use": 4 * 1024**3,
        }
        mock_devices.return_value = [mock_device]

        result = get_jax_memory_stats()

        assert result.bytes_in_use == 0  # Default
        assert result.peak_bytes_in_use == 4 * 1024**3
        assert result.bytes_limit == 0  # Default
        assert result.bytes_reserved == 0  # Default

    @mock.patch("panelcast.gpu_memory.measure.jax.devices")
    def test_get_jax_memory_stats_device_index_parameter(self, mock_devices):
        """device_index parameter selects correct device."""
        mock_device_0 = mock.Mock()
        mock_device_0.memory_stats.return_value = {
            "bytes_in_use": 0,
            "peak_bytes_in_use": 2 * 1024**3,
            "bytes_limit": 8 * 1024**3,
            "bytes_reserved": 0,
        }
        mock_device_1 = mock.Mock()
        mock_device_1.memory_stats.return_value = {
            "bytes_in_use": 0,
            "peak_bytes_in_use": 6 * 1024**3,
            "bytes_limit": 8 * 1024**3,
            "bytes_reserved": 0,
        }
        mock_devices.return_value = [mock_device_0, mock_device_1]

        result = get_jax_memory_stats(device_index=1)

        assert result.peak_bytes_in_use == 6 * 1024**3
        mock_device_1.memory_stats.assert_called_once()
