"""Tests for GPU memory query module."""

from __future__ import annotations

from unittest import mock

import pytest

from panelcast.gpu_memory.query import GpuMemoryInfo, query_gpu_memory
from panelcast.pipelines.errors import GpuMemoryError

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


# --- from unit/gpu_memory/test_query_coverage.py ---


class _MockNVMLError_coverage(Exception):
    """Lightweight NVMLError mock."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


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


@pytest.mark.skipif(not NVML_AVAILABLE, reason="pynvml not installed")
class TestNvmlErrorBranches:
    """Tests for specific NVMLError message-based error handling branches."""

    def test_library_not_found(self):
        """'library' or 'not found' in error triggers library-not-found message."""
        error = _MockNVMLError_coverage("NVML Shared Library Not Found")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_coverage),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVML library not found"):
                query_gpu_memory()

    def test_driver_not_loaded(self):
        """'driver' or 'not loaded' in error triggers driver message."""
        error = _MockNVMLError_coverage("Driver Not Loaded")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_coverage),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVIDIA driver not loaded"):
                query_gpu_memory()

    def test_permission_denied(self):
        """'permission' in error triggers permission message."""
        error = _MockNVMLError_coverage("Permission Denied accessing device")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_coverage),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="Permission denied"):
                query_gpu_memory()

    def test_generic_error(self):
        """Unknown NVMLError triggers generic fallback message."""
        error = _MockNVMLError_coverage("Something unexpected happened")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_coverage),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()

    def test_error_during_device_query(self):
        """NVMLError raised during device query (not init) is also caught."""
        error = _MockNVMLError_coverage("Unknown device error")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_coverage),
            mock.patch("panelcast.gpu_memory.query.nvmlInit"),
            mock.patch("panelcast.gpu_memory.query.nvmlShutdown"),
            mock.patch("panelcast.gpu_memory.query.nvmlDeviceGetCount", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()


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


# --- from unit/gpu_memory/test_query_expanded.py ---


class TestGpuMemoryInfoConversions:
    """Additional tests for GpuMemoryInfo byte-to-GB conversions."""

    def test_fractional_gb(self):
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=int(6.5 * 1024**3),
            used_bytes=int(2.3 * 1024**3),
            free_bytes=int(4.2 * 1024**3),
        )
        assert info.total_gb == pytest.approx(6.5)
        assert info.used_gb == pytest.approx(2.3)
        assert info.free_gb == pytest.approx(4.2)

    def test_small_bytes(self):
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=1024,
            used_bytes=512,
            free_bytes=512,
        )
        assert info.total_gb == pytest.approx(1024 / 1024**3)

    def test_large_bytes(self):
        """80 GB GPU."""
        info = GpuMemoryInfo(
            device_name="A100",
            total_bytes=80 * 1024**3,
            used_bytes=10 * 1024**3,
            free_bytes=70 * 1024**3,
        )
        assert info.total_gb == pytest.approx(80.0)
        assert info.free_gb == pytest.approx(70.0)

    def test_all_used(self):
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=8 * 1024**3,
            used_bytes=8 * 1024**3,
            free_bytes=0,
        )
        assert info.free_percent == 0.0
        assert info.used_percent == 100.0

    def test_all_free(self):
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=8 * 1024**3,
            used_bytes=0,
            free_bytes=8 * 1024**3,
        )
        assert info.free_percent == 100.0
        assert info.used_percent == 0.0


class TestGpuMemoryInfoFormatDisplay:
    """Additional tests for format_display."""

    def test_includes_bytes_count(self):
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=8 * 1024**3,
            used_bytes=2 * 1024**3,
            free_bytes=6 * 1024**3,
        )
        display = info.format_display()
        assert "bytes" in display

    def test_zero_total_format(self):
        info = GpuMemoryInfo(
            device_name="test",
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
        )
        display = info.format_display()
        assert "0.0 GB" in display

    def test_device_name_not_in_display(self):
        """format_display does not include device name (shown separately)."""
        info = GpuMemoryInfo(
            device_name="NVIDIA RTX 3080",
            total_bytes=8 * 1024**3,
            used_bytes=2 * 1024**3,
            free_bytes=6 * 1024**3,
        )
        display = info.format_display()
        assert "NVIDIA" not in display


class TestQueryGpuMemoryValidation:
    """Tests for input validation in query_gpu_memory."""

    def test_negative_device_index_raises(self):
        with pytest.raises(GpuMemoryError, match="non-negative"):
            from panelcast.gpu_memory.query import query_gpu_memory

            query_gpu_memory(device_index=-1)

    def test_negative_device_index_minus_ten(self):
        with pytest.raises(GpuMemoryError, match="non-negative"):
            from panelcast.gpu_memory.query import query_gpu_memory

            query_gpu_memory(device_index=-10)


# --- from unit/gpu_memory/test_query_new.py ---


class _MockNVMLError_new(Exception):
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
        error = _MockNVMLError_new("NVML Shared Library Not Found")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "NVML library not found" in str(exc_info.value)
            assert "GPU_SETUP.md" in str(exc_info.value)

    def test_not_found_keyword_triggers_library_message(self):
        """'not found' in error string triggers library-not-found message."""
        error = _MockNVMLError_new("Module Not Found")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVML library not found"):
                query_gpu_memory()

    def test_driver_keyword_triggers_driver_message(self):
        """'driver' in error string triggers driver-not-loaded message."""
        error = _MockNVMLError_new("NVIDIA Driver Not Loaded Properly")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "NVIDIA driver not loaded" in str(exc_info.value)
            assert "nvidia-smi" in str(exc_info.value)

    def test_not_loaded_keyword_triggers_driver_message(self):
        """'not loaded' in error string triggers driver-not-loaded message."""
        error = _MockNVMLError_new("Kernel module not loaded")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="NVIDIA driver not loaded"):
                query_gpu_memory()

    def test_permission_keyword_triggers_permission_message(self):
        """'permission' in error string triggers permission-denied message."""
        error = _MockNVMLError_new("Permission Denied accessing /dev/nvidia0")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError) as exc_info:
                query_gpu_memory()
            assert "Permission denied" in str(exc_info.value)
            assert "/dev/nvidia" in str(exc_info.value)

    def test_generic_nvml_error(self):
        """Unknown NVMLError message triggers generic fallback."""
        error = _MockNVMLError_new("Something completely unexpected")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
            mock.patch("panelcast.gpu_memory.query.nvmlInit", side_effect=error),
        ):
            with pytest.raises(GpuMemoryError, match="GPU memory query failed"):
                query_gpu_memory()

    def test_nvml_error_during_device_get_count(self):
        """NVMLError raised during nvmlDeviceGetCount is caught."""
        error = _MockNVMLError_new("Internal Error")
        with (
            mock.patch("panelcast.gpu_memory.query.NVMLError", _MockNVMLError_new),
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
