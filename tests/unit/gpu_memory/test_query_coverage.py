"""Coverage-targeted tests for gpu_memory/query.py.

Tests target missed lines/branches:
- _nvml_context: shutdown always called (even on exception)
- query_gpu_memory: negative device_index, NVML not available path,
  NVMLError sub-branches (library, driver, permission, generic),
  bytes vs str device name, device_index >= count
- GpuMemoryInfo: edge case properties (zero total, format_display)
"""

from __future__ import annotations

from unittest import mock

import pytest

from panelcast.gpu_memory.query import GpuMemoryInfo, query_gpu_memory
from panelcast.pipelines.errors import GpuMemoryError

# Check if pynvml is available
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


# =============================================================================
# Helper: mock NVMLError for tests
# =============================================================================


class _MockNVMLError(Exception):
    """Lightweight NVMLError mock."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


# =============================================================================
# TestNvmlNotAvailable
# =============================================================================


class TestNvmlNotAvailable:
    """Tests for query_gpu_memory when NVML is not importable."""

    def test_raises_with_import_error_message(self):
        """Error message includes the original import error."""
        with (
            mock.patch("panelcast.gpu_memory.query._NVML_AVAILABLE", False),
            mock.patch(
                "panelcast.gpu_memory.query._NVML_IMPORT_ERROR",
                "No module named 'pynvml'",
            ),
        ):
            with pytest.raises(GpuMemoryError, match="NVML library not available"):
                query_gpu_memory()

    def test_error_suggests_pip_install(self):
        """Error message suggests installing nvidia-ml-py."""
        with (
            mock.patch("panelcast.gpu_memory.query._NVML_AVAILABLE", False),
            mock.patch(
                "panelcast.gpu_memory.query._NVML_IMPORT_ERROR",
                "some error",
            ),
        ):
            with pytest.raises(GpuMemoryError, match="nvidia-ml-py"):
                query_gpu_memory()


# =============================================================================
# TestNegativeDeviceIndex
# =============================================================================


class TestNegativeDeviceIndex:
    """Tests for negative device_index validation."""

    def test_negative_one(self):
        """device_index=-1 should raise before any NVML calls."""
        with pytest.raises(GpuMemoryError, match="non-negative"):
            query_gpu_memory(device_index=-1)

    def test_very_negative(self):
        """Large negative device_index should raise."""
        with pytest.raises(GpuMemoryError, match="non-negative"):
            query_gpu_memory(device_index=-999)


# =============================================================================
# TestNvmlContextManager
# =============================================================================


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestNvmlContextManager:
    """Tests for _nvml_context ensuring shutdown is called."""

    def test_shutdown_on_normal_exit(self):
        """nvmlShutdown called after successful context body."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown") as mock_shutdown,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=1),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetHandleByIndex", return_value="h"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetMemoryInfo") as mock_mem,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetName", return_value="GPU"),
        ):
            mem = mock.Mock()
            mem.total = mem.used = mem.free = 1024
            mock_mem.return_value = mem

            query_gpu_memory()
            mock_shutdown.assert_called_once()

    def test_shutdown_on_exception(self):
        """nvmlShutdown called even when body raises."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown") as mock_shutdown,
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=0),
        ):
            with pytest.raises(GpuMemoryError):
                query_gpu_memory()

            mock_shutdown.assert_called_once()


# =============================================================================
# TestNvmlErrorBranches
# =============================================================================


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestNvmlErrorBranches:
    """Tests for specific NVMLError message-based error handling branches."""

    def test_library_not_found(self):
        """'library' or 'not found' in error triggers library-not-found message."""
        error = _MockNVMLError("NVML Shared Library Not Found")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVML library not found"):
                query_gpu_memory()

    def test_driver_not_loaded(self):
        """'driver' or 'not loaded' in error triggers driver message."""
        error = _MockNVMLError("Driver Not Loaded")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVIDIA driver not loaded"):
                query_gpu_memory()

    def test_permission_denied(self):
        """'permission' in error triggers permission message."""
        error = _MockNVMLError("Permission Denied accessing device")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="Permission denied"):
                query_gpu_memory()

    def test_generic_error(self):
        """Unknown NVMLError triggers generic fallback message."""
        error = _MockNVMLError("Something unexpected happened")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()

    def test_error_during_device_query(self):
        """NVMLError raised during device query (not init) is also caught."""
        error = _MockNVMLError("Unknown device error")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()


# =============================================================================
# TestDeviceNameEncoding
# =============================================================================


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestDeviceNameEncoding:
    """Tests for bytes vs str device name handling."""

    @pytest.fixture
    def mock_nvml_success(self):
        """Standard mock NVML setup returning 1 GPU."""
        mem = mock.Mock()
        mem.total = 8 * 1024**3
        mem.used = 2 * 1024**3
        mem.free = 6 * 1024**3

        return {
            "init": mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            "shutdown": mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            "count": mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=1),
            "handle": mock.patch(
                "panelcast.gpu_memory.query.nvmlDeviceGetHandleByIndex", return_value="h"
            ),
            "memory": mock.patch(
                "panelcast.gpu_memory.query.nvmlDeviceGetMemoryInfo", return_value=mem
            ),
        }

    def test_str_name(self, mock_nvml_success):
        """String device name is used directly."""
        with (
            mock_nvml_success["init"],
            mock_nvml_success["shutdown"],
            mock_nvml_success["count"],
            mock_nvml_success["handle"],
            mock_nvml_success["memory"],
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetName", return_value="RTX 3090"),
        ):
            result = query_gpu_memory()
            assert result.device_name == "RTX 3090"

    def test_bytes_name_decoded(self, mock_nvml_success):
        """Bytes device name is decoded to str."""
        with (
            mock_nvml_success["init"],
            mock_nvml_success["shutdown"],
            mock_nvml_success["count"],
            mock_nvml_success["handle"],
            mock_nvml_success["memory"],
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetName", return_value=b"RTX 4090"),
        ):
            result = query_gpu_memory()
            assert result.device_name == "RTX 4090"


# =============================================================================
# TestDeviceIndexOutOfRange
# =============================================================================


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestDeviceIndexOutOfRange:
    """Tests for device_index >= count error."""

    def test_index_equals_count(self):
        """device_index == count should raise (0-based indexing)."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=2),
        ):
            with pytest.raises(GpuMemoryError, match="out of range"):
                query_gpu_memory(device_index=2)

    def test_index_way_above_count(self):
        """device_index much larger than count should raise."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=1),
        ):
            with pytest.raises(GpuMemoryError, match="out of range"):
                query_gpu_memory(device_index=10)

    def test_error_message_shows_range(self):
        """Error message should show available GPU range."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=3),
        ):
            with pytest.raises(GpuMemoryError, match="0-2"):
                query_gpu_memory(device_index=5)


# =============================================================================
# TestGpuMemoryInfoEdgeCases
# =============================================================================


class TestGpuMemoryInfoEdgeCases:
    """Additional edge cases for GpuMemoryInfo."""

    def test_very_small_memory(self):
        """Single byte of memory."""
        info = GpuMemoryInfo(
            device_name="tiny",
            total_bytes=1,
            used_bytes=0,
            free_bytes=1,
        )
        assert info.total_gb == pytest.approx(1 / 1024**3)
        assert info.free_percent == 100.0

    def test_format_display_with_zero_free(self):
        """Format display when all memory is used."""
        info = GpuMemoryInfo(
            device_name="full",
            total_bytes=8 * 1024**3,
            used_bytes=8 * 1024**3,
            free_bytes=0,
        )
        display = info.format_display()
        assert "0.0 GB free" in display
        assert "0%" in display

    def test_format_display_comma_separated_bytes(self):
        """Format display includes comma-separated byte count."""
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=16 * 1024**3,
            used_bytes=4 * 1024**3,
            free_bytes=12 * 1024**3,
        )
        display = info.format_display()
        assert "," in display  # Comma-separated bytes
        assert "12.0 GB free" in display
