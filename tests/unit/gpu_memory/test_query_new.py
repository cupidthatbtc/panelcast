"""New coverage-targeted tests for gpu_memory/query.py.

Targets missed lines/branches:
- _nvml_context: ensures shutdown always called
- query_gpu_memory: negative device_index early return
- query_gpu_memory: NVML not available path
- query_gpu_memory: no GPUs detected (count==0)
- query_gpu_memory: device_index >= count
- query_gpu_memory: bytes vs str device name
- NVMLError branches: library, driver, permission, generic
- GpuMemoryInfo: zero total_bytes edge case for percent properties
- GpuMemoryInfo: format_display various cases
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


class _MockNVMLError(Exception):
    """Mock NVMLError for testing without real NVML."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class TestQueryNvmlNotAvailable:
    """Tests for query_gpu_memory when NVML library is not importable."""

    def test_nvml_not_available_includes_original_error(self):
        """Error message includes the original import error string."""
        with (
            mock.patch("panelcast.gpu_memory.query._NVML_AVAILABLE", False),
            mock.patch("panelcast.gpu_memory.query._NVML_IMPORT_ERROR", "No module named 'pynvml'"),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            error_msg = str(exc_info.value)
            assert "NVML library not available" in error_msg
            assert "No module named 'pynvml'" in error_msg
            assert "nvidia-ml-py" in error_msg

    def test_nvml_not_available_suggests_force_run(self):
        """Error message suggests --force-run alternative."""
        with (
            mock.patch("panelcast.gpu_memory.query._NVML_AVAILABLE", False),
            mock.patch("panelcast.gpu_memory.query._NVML_IMPORT_ERROR", "err"),
        ):
            with pytest.raises(GpuMemoryError, match="--force-run"):
                query_gpu_memory()


class TestQueryNegativeDeviceIndex:
    """Tests for negative device_index validation (before NVML calls)."""

    def test_negative_one_raises(self):
        """device_index=-1 raises GpuMemoryError with non-negative message."""
        with pytest.raises(GpuMemoryError, match="non-negative"):
            query_gpu_memory(device_index=-1)

    def test_very_negative_raises(self):
        """Large negative device_index also raises."""
        with pytest.raises(GpuMemoryError, match="non-negative"):
            query_gpu_memory(device_index=-100)

    def test_zero_index_does_not_raise_validation(self):
        """device_index=0 passes the validation check (may fail later on NVML)."""
        # This will fail on NVML init, but should not fail on the negative check
        if not NVML_AVAILABLE:
            with mock.patch("panelcast.gpu_memory.query._NVML_AVAILABLE", False):
                with pytest.raises(GpuMemoryError, match="NVML library not available"):
                    query_gpu_memory(device_index=0)
        else:
            # Just verify no "non-negative" error
            try:
                query_gpu_memory(device_index=0)
            except GpuMemoryError as e:
                assert "non-negative" not in str(e)


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestNvmlErrorBranchesNew:
    """Tests for NVMLError message-based branching."""

    def test_library_keyword_triggers_library_message(self):
        """'library' in error string triggers library-not-found message."""
        error = _MockNVMLError("NVML Shared Library Not Found")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "NVML library not found" in str(exc_info.value)
            assert "GPU_SETUP.md" in str(exc_info.value)

    def test_not_found_keyword_triggers_library_message(self):
        """'not found' in error string triggers library-not-found message."""
        error = _MockNVMLError("Module Not Found")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVML library not found"):
                query_gpu_memory()

    def test_driver_keyword_triggers_driver_message(self):
        """'driver' in error string triggers driver-not-loaded message."""
        error = _MockNVMLError("NVIDIA Driver Not Loaded Properly")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "NVIDIA driver not loaded" in str(exc_info.value)
            assert "nvidia-smi" in str(exc_info.value)

    def test_not_loaded_keyword_triggers_driver_message(self):
        """'not loaded' in error string triggers driver-not-loaded message."""
        error = _MockNVMLError("Kernel module not loaded")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVIDIA driver not loaded"):
                query_gpu_memory()

    def test_permission_keyword_triggers_permission_message(self):
        """'permission' in error string triggers permission-denied message."""
        error = _MockNVMLError("Permission Denied accessing /dev/nvidia0")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "Permission denied" in str(exc_info.value)
            assert "/dev/nvidia" in str(exc_info.value)

    def test_generic_nvml_error(self):
        """Unknown NVMLError message triggers generic fallback."""
        error = _MockNVMLError("Something completely unexpected")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()

    def test_nvml_error_during_device_get_count(self):
        """NVMLError raised during nvmlDeviceGetCount is caught."""
        error = _MockNVMLError("Internal Error")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError),
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestQueryNoGpuDetected:
    """Tests for count==0 path."""

    def test_no_gpu_raises_with_message(self):
        """count==0 raises GpuMemoryError suggesting --force-run."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=0),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "No NVIDIA GPU detected" in str(exc_info.value)
            assert "--force-run" in str(exc_info.value)


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestQueryDeviceIndexOutOfRange:
    """Tests for device_index >= count path."""

    def test_index_at_count_raises(self):
        """device_index == count should raise (0-indexed)."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=1),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory(device_index=1)
            assert "out of range" in str(exc_info.value)
            assert "0-0" in str(exc_info.value)

    def test_index_above_count_shows_range(self):
        """Error message shows available GPU range."""
        with (
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=4),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory(device_index=7)
            assert "0-3" in str(exc_info.value)


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestQueryDeviceNameEncoding:
    """Tests for bytes vs str device name handling."""

    def _make_mock_nvml(self, name_value):
        """Create mock NVML patches for a successful query."""
        mem = mock.Mock()
        mem.total = 8 * 1024**3
        mem.used = 2 * 1024**3
        mem.free = 6 * 1024**3

        patches = {
            "init": mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            "shutdown": mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            "count": mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", return_value=1),
            "handle": mock.patch(
                "panelcast.gpu_memory.query.nvmlDeviceGetHandleByIndex", return_value="h"
            ),
            "memory": mock.patch(
                "panelcast.gpu_memory.query.nvmlDeviceGetMemoryInfo", return_value=mem
            ),
            "name": mock.patch(
                "panelcast.gpu_memory.query.nvmlDeviceGetName", return_value=name_value
            ),
        }
        return patches

    def test_str_name_used_directly(self):
        """String device name is used as-is."""
        patches = self._make_mock_nvml("NVIDIA RTX 3090")
        with (
            patches["init"],
            patches["shutdown"],
            patches["count"],
            patches["handle"],
            patches["memory"],
            patches["name"],
        ):
            result = query_gpu_memory()
            assert result.device_name == "NVIDIA RTX 3090"

    def test_bytes_name_decoded(self):
        """Bytes device name is decoded to UTF-8."""
        patches = self._make_mock_nvml(b"NVIDIA A100")
        with (
            patches["init"],
            patches["shutdown"],
            patches["count"],
            patches["handle"],
            patches["memory"],
            patches["name"],
        ):
            result = query_gpu_memory()
            assert result.device_name == "NVIDIA A100"


class TestGpuMemoryInfoProperties:
    """Tests for GpuMemoryInfo edge case properties."""

    def test_zero_total_bytes_free_percent(self):
        """free_percent returns 0 when total_bytes is 0."""
        info = GpuMemoryInfo("test", total_bytes=0, used_bytes=0, free_bytes=0)
        assert info.free_percent == 0.0

    def test_zero_total_bytes_used_percent(self):
        """used_percent returns 0 when total_bytes is 0."""
        info = GpuMemoryInfo("test", total_bytes=0, used_bytes=0, free_bytes=0)
        assert info.used_percent == 0.0

    def test_full_gpu_percentages(self):
        """100% used, 0% free."""
        total = 8 * 1024**3
        info = GpuMemoryInfo("GPU", total_bytes=total, used_bytes=total, free_bytes=0)
        assert info.used_percent == 100.0
        assert info.free_percent == 0.0

    def test_empty_gpu_percentages(self):
        """0% used, 100% free."""
        total = 8 * 1024**3
        info = GpuMemoryInfo("GPU", total_bytes=total, used_bytes=0, free_bytes=total)
        assert info.used_percent == 0.0
        assert info.free_percent == 100.0

    def test_format_display_includes_all_parts(self):
        """format_display includes free GB, percent, total GB, and bytes."""
        info = GpuMemoryInfo(
            "RTX 3080", total_bytes=8 * 1024**3, used_bytes=2 * 1024**3, free_bytes=6 * 1024**3
        )
        display = info.format_display()
        assert "6.0 GB free" in display
        assert "75%" in display
        assert "8.0 GB" in display
        assert "bytes" in display

    def test_format_display_zero_memory(self):
        """format_display handles zero memory gracefully."""
        info = GpuMemoryInfo("test", total_bytes=0, used_bytes=0, free_bytes=0)
        display = info.format_display()
        assert "0.0 GB free" in display
        assert "0 bytes" in display

    def test_gb_conversions(self):
        """All GB properties convert correctly."""
        info = GpuMemoryInfo(
            "GPU",
            total_bytes=16 * 1024**3,
            used_bytes=4 * 1024**3,
            free_bytes=12 * 1024**3,
        )
        assert info.total_gb == pytest.approx(16.0)
        assert info.used_gb == pytest.approx(4.0)
        assert info.free_gb == pytest.approx(12.0)

    def test_frozen(self):
        """GpuMemoryInfo is frozen dataclass."""
        info = GpuMemoryInfo("GPU", total_bytes=1, used_bytes=0, free_bytes=1)
        with pytest.raises(AttributeError):
            info.total_bytes = 2
