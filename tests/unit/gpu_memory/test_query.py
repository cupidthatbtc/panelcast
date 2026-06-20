"""Tests for GPU memory query module."""

from __future__ import annotations

from unittest import mock

import pytest

from panelcast.gpu_memory.query import GpuMemoryInfo, query_gpu_memory
from panelcast.pipelines.errors import GpuMemoryError

# Check if NVML is available (for conditional test skipping)
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class TestGpuMemoryInfo:
    """Tests for GpuMemoryInfo dataclass."""

    @pytest.fixture
    def sample_info(self) -> GpuMemoryInfo:
        """Create sample GPU info with 8GB total, 2GB used, 6GB free."""
        return GpuMemoryInfo(
            device_name="NVIDIA GeForce RTX 3080",
            total_bytes=8 * 1024**3,  # 8 GB
            used_bytes=2 * 1024**3,  # 2 GB
            free_bytes=6 * 1024**3,  # 6 GB
        )

    def test_frozen_dataclass(self, sample_info: GpuMemoryInfo):
        """GpuMemoryInfo is immutable."""
        with pytest.raises(AttributeError):
            sample_info.total_bytes = 0  # type: ignore[misc]

    def test_total_gb(self, sample_info: GpuMemoryInfo):
        """total_gb converts bytes to GB correctly."""
        assert sample_info.total_gb == 8.0

    def test_used_gb(self, sample_info: GpuMemoryInfo):
        """used_gb converts bytes to GB correctly."""
        assert sample_info.used_gb == 2.0

    def test_free_gb(self, sample_info: GpuMemoryInfo):
        """free_gb converts bytes to GB correctly."""
        assert sample_info.free_gb == 6.0

    def test_free_percent(self, sample_info: GpuMemoryInfo):
        """free_percent calculates percentage correctly."""
        assert sample_info.free_percent == 75.0

    def test_used_percent(self, sample_info: GpuMemoryInfo):
        """used_percent calculates percentage correctly."""
        assert sample_info.used_percent == 25.0

    def test_free_percent_zero_total(self):
        """free_percent returns 0 when total is 0 (avoid division by zero)."""
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
        )
        assert info.free_percent == 0.0

    def test_used_percent_zero_total(self):
        """used_percent returns 0 when total is 0 (avoid division by zero)."""
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
        )
        assert info.used_percent == 0.0

    def test_format_display(self, sample_info: GpuMemoryInfo):
        """format_display returns human-readable string."""
        display = sample_info.format_display()
        assert "6.0 GB free" in display
        assert "75%" in display
        assert "8.0 GB" in display
        # Check bytes are formatted with commas
        assert "6,442,450,944 bytes" in display

    def test_format_display_fractional(self):
        """format_display handles fractional GB values."""
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=int(8.5 * 1024**3),
            used_bytes=int(2.3 * 1024**3),
            free_bytes=int(6.2 * 1024**3),
        )
        display = info.format_display()
        assert "6.2 GB free" in display


class TestQueryGpuMemoryNvmlNotAvailable:
    """Tests for query_gpu_memory() when NVML is not available."""

    def test_raises_when_nvml_not_available(self):
        """query_gpu_memory raises GpuMemoryError when NVML is not available."""
        with mock.patch("panelcast.gpu_memory.query._NVML_AVAILABLE", False):
            with mock.patch(
                "panelcast.gpu_memory.query._NVML_IMPORT_ERROR",
                "No module named 'pynvml'",
            ):
                with pytest.raises(GpuMemoryError) as exc_info:
                    query_gpu_memory()

                assert "NVML library not available" in str(exc_info.value)
                assert "nvidia-ml-py" in str(exc_info.value)


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestQueryGpuMemory:
    """Tests for query_gpu_memory() function.

    These tests require pynvml to be installed so the NVML functions
    exist in the module namespace for patching.
    """

    @pytest.fixture
    def mock_nvml(self):
        """Mock all NVML functions for testing."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit") as init,
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown") as shutdown,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount") as count,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetHandleByIndex") as handle,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetMemoryInfo") as memory,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetName") as name,
        ):
            yield {
                "init": init,
                "shutdown": shutdown,
                "count": count,
                "handle": handle,
                "memory": memory,
                "name": name,
            }

    def test_returns_gpu_memory_info(self, mock_nvml):
        """query_gpu_memory returns GpuMemoryInfo on success."""
        mock_nvml["count"].return_value = 1
        mock_nvml["handle"].return_value = "handle"

        mem_info = mock.Mock()
        mem_info.total = 8 * 1024**3
        mem_info.used = 2 * 1024**3
        mem_info.free = 6 * 1024**3
        mock_nvml["memory"].return_value = mem_info
        mock_nvml["name"].return_value = "NVIDIA GeForce RTX 3080"

        result = query_gpu_memory()

        assert isinstance(result, GpuMemoryInfo)
        assert result.device_name == "NVIDIA GeForce RTX 3080"
        assert result.total_bytes == 8 * 1024**3
        assert result.used_bytes == 2 * 1024**3
        assert result.free_bytes == 6 * 1024**3

    def test_handles_bytes_device_name(self, mock_nvml):
        """query_gpu_memory handles device name returned as bytes."""
        mock_nvml["count"].return_value = 1
        mock_nvml["handle"].return_value = "handle"

        mem_info = mock.Mock()
        mem_info.total = 8 * 1024**3
        mem_info.used = 0
        mem_info.free = 8 * 1024**3
        mock_nvml["memory"].return_value = mem_info
        mock_nvml["name"].return_value = b"NVIDIA RTX 4090"  # bytes, not str

        result = query_gpu_memory()

        assert result.device_name == "NVIDIA RTX 4090"

    def test_raises_on_no_gpu(self, mock_nvml):
        """query_gpu_memory raises GpuMemoryError when no GPU detected."""
        mock_nvml["count"].return_value = 0

        with pytest.raises(GpuMemoryError) as exc_info:
            query_gpu_memory()

        assert "No NVIDIA GPU detected" in str(exc_info.value)
        assert "--force-run" in str(exc_info.value)
        assert exc_info.value.exit_code == 6

    def test_raises_on_invalid_device_index(self, mock_nvml):
        """query_gpu_memory raises GpuMemoryError for invalid device index."""
        mock_nvml["count"].return_value = 2  # 2 GPUs (index 0, 1)

        with pytest.raises(GpuMemoryError) as exc_info:
            query_gpu_memory(device_index=5)

        assert "out of range" in str(exc_info.value)
        assert "0-1" in str(exc_info.value)

    def test_device_index_parameter(self, mock_nvml):
        """query_gpu_memory passes device_index to NVML."""
        mock_nvml["count"].return_value = 4
        mock_nvml["handle"].return_value = "handle2"

        mem_info = mock.Mock()
        mem_info.total = mem_info.used = mem_info.free = 1024
        mock_nvml["memory"].return_value = mem_info
        mock_nvml["name"].return_value = "GPU 2"

        query_gpu_memory(device_index=2)

        mock_nvml["handle"].assert_called_once_with(2)

    def test_nvml_shutdown_called_on_success(self, mock_nvml):
        """nvmlShutdown is called even on success."""
        mock_nvml["count"].return_value = 1
        mock_nvml["handle"].return_value = "handle"

        mem_info = mock.Mock()
        mem_info.total = mem_info.used = mem_info.free = 1024
        mock_nvml["memory"].return_value = mem_info
        mock_nvml["name"].return_value = "GPU"

        query_gpu_memory()

        mock_nvml["shutdown"].assert_called_once()

    def test_nvml_shutdown_called_on_error(self, mock_nvml):
        """nvmlShutdown is called even when an error occurs."""
        mock_nvml["count"].return_value = 0  # Will raise GpuMemoryError

        with pytest.raises(GpuMemoryError):
            query_gpu_memory()

        mock_nvml["shutdown"].assert_called_once()


class _MockNVMLError(Exception):
    """Mock NVMLError that behaves like real NVMLError for testing.

    The real pynvml.NVMLError requires NVML library to be initialized to get
    error strings, which breaks tests. This mock provides the same interface.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestQueryGpuMemoryErrorHandling:
    """Tests for NVML error handling in query_gpu_memory().

    These tests require pynvml to be installed for patching.
    """

    def test_library_not_found_error(self):
        """query_gpu_memory converts library not found to GpuMemoryError."""
        # Use mock exception that looks like NVMLError
        error = _MockNVMLError("NVML Shared Library Not Found")

        with (
            mock.patch(
                "panelcast.gpu_memory.query.NVMLError",
                _MockNVMLError,
            ),
            mock.patch(
                "panelcast.gpu_memory.query.nvmlInit",
                side_effect=error,
            ),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()

            assert "NVML library not found" in str(exc_info.value)
            assert "--force-run" in str(exc_info.value)

    def test_driver_not_loaded_error(self):
        """query_gpu_memory converts driver not loaded to GpuMemoryError."""
        error = _MockNVMLError("Driver Not Loaded")

        with (
            mock.patch(
                "panelcast.gpu_memory.query.NVMLError",
                _MockNVMLError,
            ),
            mock.patch(
                "panelcast.gpu_memory.query.nvmlInit",
                side_effect=error,
            ),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()

            assert "driver not loaded" in str(exc_info.value).lower()
            assert "nvidia-smi" in str(exc_info.value)

    def test_permission_error(self):
        """query_gpu_memory converts permission error to GpuMemoryError."""
        error = _MockNVMLError("Permission Denied")

        with (
            mock.patch(
                "panelcast.gpu_memory.query.NVMLError",
                _MockNVMLError,
            ),
            mock.patch(
                "panelcast.gpu_memory.query.nvmlInit",
                side_effect=error,
            ),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()

            assert "permission" in str(exc_info.value).lower()

    def test_generic_nvml_error(self):
        """query_gpu_memory wraps generic NVML errors."""
        error = _MockNVMLError("Unknown Error XYZ")

        with (
            mock.patch(
                "panelcast.gpu_memory.query.NVMLError",
                _MockNVMLError,
            ),
            mock.patch(
                "panelcast.gpu_memory.query.nvmlInit",
                side_effect=error,
            ),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()

            assert "GPU memory query failed" in str(exc_info.value)
            assert "Unknown Error XYZ" in str(exc_info.value)
