"""Expanded tests for JaxMemoryStats dataclass."""

from dataclasses import FrozenInstanceError

import pytest

from panelcast.gpu_memory.measure import JaxMemoryStats


class TestJaxMemoryStatsProperties:
    """Tests for JaxMemoryStats properties."""

    @pytest.fixture
    def stats(self):
        return JaxMemoryStats(
            bytes_in_use=4 * 1024**3,
            peak_bytes_in_use=6 * 1024**3,
            bytes_limit=24 * 1024**3,
            bytes_reserved=8 * 1024**3,
        )

    def test_peak_gb(self, stats):
        assert stats.peak_gb == pytest.approx(6.0)

    def test_limit_gb(self, stats):
        assert stats.limit_gb == pytest.approx(24.0)

    def test_in_use_gb(self, stats):
        assert stats.in_use_gb == pytest.approx(4.0)

    def test_reserved_gb(self, stats):
        assert stats.reserved_gb == pytest.approx(8.0)

    def test_frozen(self, stats):
        with pytest.raises(FrozenInstanceError):
            stats.bytes_in_use = 0

    def test_zero_values(self):
        stats = JaxMemoryStats(bytes_in_use=0, peak_bytes_in_use=0, bytes_limit=0, bytes_reserved=0)
        assert stats.peak_gb == 0.0
        assert stats.limit_gb == 0.0
        assert stats.in_use_gb == 0.0
        assert stats.reserved_gb == 0.0

    def test_fractional_gb(self):
        stats = JaxMemoryStats(
            bytes_in_use=int(0.5 * 1024**3),
            peak_bytes_in_use=int(1.5 * 1024**3),
            bytes_limit=int(8.0 * 1024**3),
            bytes_reserved=int(2.0 * 1024**3),
        )
        assert stats.peak_gb == pytest.approx(1.5)
        assert stats.in_use_gb == pytest.approx(0.5)

    def test_large_values(self):
        """80 GB A100."""
        stats = JaxMemoryStats(
            bytes_in_use=40 * 1024**3,
            peak_bytes_in_use=60 * 1024**3,
            bytes_limit=80 * 1024**3,
            bytes_reserved=70 * 1024**3,
        )
        assert stats.peak_gb == pytest.approx(60.0)
        assert stats.limit_gb == pytest.approx(80.0)

    def test_equality(self):
        a = JaxMemoryStats(
            bytes_in_use=100, peak_bytes_in_use=200, bytes_limit=300, bytes_reserved=150
        )
        b = JaxMemoryStats(
            bytes_in_use=100, peak_bytes_in_use=200, bytes_limit=300, bytes_reserved=150
        )
        assert a == b

    def test_inequality(self):
        a = JaxMemoryStats(
            bytes_in_use=100, peak_bytes_in_use=200, bytes_limit=300, bytes_reserved=150
        )
        b = JaxMemoryStats(
            bytes_in_use=100, peak_bytes_in_use=999, bytes_limit=300, bytes_reserved=150
        )
        assert a != b
