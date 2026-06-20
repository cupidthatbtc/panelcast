"""Expanded tests for GpuMemoryInfo dataclass properties."""

import pytest

from panelcast.gpu_memory.query import GpuMemoryInfo
from panelcast.pipelines.errors import GpuMemoryError


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
