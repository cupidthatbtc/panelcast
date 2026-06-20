"""Tests for full preflight output rendering."""

from __future__ import annotations

import pytest

from panelcast.preflight import (
    ExtrapolationResult,
    FullPreflightResult,
    PreflightStatus,
    render_extrapolation_result,
    render_full_preflight_result,
)
from panelcast.preflight.calibrate import CalibrationResult


def _make_full_result(**overrides: object) -> FullPreflightResult:
    """Create FullPreflightResult with sensible defaults.

    Provides a PASS result with typical values. Override any field
    by passing keyword arguments.
    """
    defaults = {
        "status": PreflightStatus.PASS,
        "measured_peak_gb": 4.0,
        "available_gb": 12.0,
        "total_gpu_gb": 16.0,
        "headroom_percent": 65.0,
        "mini_run_seconds": 10.0,
        "message": "Test message",
        "suggestions": (),
        "device_name": "Test GPU",
    }
    defaults.update(overrides)
    return FullPreflightResult(**defaults)


class TestRenderFullPreflightStatus:
    """Tests for full preflight status text rendering."""

    @pytest.fixture
    def pass_result(self) -> FullPreflightResult:
        """Create a PASS result for testing."""
        return FullPreflightResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=4.0,
            available_gb=12.0,
            total_gpu_gb=16.0,
            headroom_percent=66.7,
            mini_run_seconds=10.5,
            message="Full preflight passed: 4.00 GB measured peak, 12.0 GB available",
            suggestions=(),
            device_name="NVIDIA RTX 4090",
        )

    @pytest.fixture
    def fail_result(self) -> FullPreflightResult:
        """Create a FAIL result for testing."""
        return FullPreflightResult(
            status=PreflightStatus.FAIL,
            measured_peak_gb=8.0,
            available_gb=6.0,
            total_gpu_gb=8.0,
            headroom_percent=-33.3,
            mini_run_seconds=15.0,
            message="Full preflight failed: 8.00 GB measured peak exceeds 6.0 GB available",
            suggestions=("Try reducing --num-chains",),
            device_name="NVIDIA GTX 1080",
        )

    @pytest.fixture
    def warning_result(self) -> FullPreflightResult:
        """Create a WARNING result for testing."""
        return FullPreflightResult(
            status=PreflightStatus.WARNING,
            measured_peak_gb=7.0,
            available_gb=8.0,
            total_gpu_gb=10.0,
            headroom_percent=12.5,
            mini_run_seconds=12.0,
            message="Full preflight warning: 7.00 GB measured peak, low headroom",
            suggestions=("Memory is tight; consider reducing --num-chains",),
            device_name="NVIDIA RTX 3070",
        )

    @pytest.fixture
    def cannot_check_result(self) -> FullPreflightResult:
        """Create a CANNOT_CHECK result for testing."""
        return FullPreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            measured_peak_gb=0.0,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            mini_run_seconds=0.0,
            message="Cannot query GPU: No GPU detected",
            suggestions=("Use --preflight for estimation without GPU query",),
        )

    def test_render_pass_contains_pass(self, capsys, pass_result: FullPreflightResult):
        """Output contains 'PASS' for PASS status."""
        render_full_preflight_result(pass_result)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_render_fail_contains_fail(self, capsys, fail_result: FullPreflightResult):
        """Output contains 'FAIL' for FAIL status."""
        render_full_preflight_result(fail_result)
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_render_warning_contains_warning(self, capsys, warning_result: FullPreflightResult):
        """Output contains 'WARNING' for WARNING status."""
        render_full_preflight_result(warning_result)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_render_cannot_check_contains_cannot(
        self, capsys, cannot_check_result: FullPreflightResult
    ):
        """Output contains 'CANNOT' for CANNOT_CHECK status."""
        render_full_preflight_result(cannot_check_result)
        captured = capsys.readouterr()
        assert "CANNOT" in captured.out


class TestRenderFullPreflightMeasuredPeak:
    """Tests for measured peak display."""

    def test_render_shows_measured_peak(self, capsys):
        """Output shows measured peak GB value."""
        result = _make_full_result(measured_peak_gb=4.25)

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        assert "4.25 GB" in captured.out
        assert "Measured" in captured.out

    def test_render_shows_mini_run_time(self, capsys):
        """Output shows mini-run execution time."""
        result = _make_full_result(mini_run_seconds=15.3)

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        assert "15.3" in captured.out
        assert "seconds" in captured.out


class TestRenderFullPreflightVerbose:
    """Tests for verbose mode output."""

    def test_verbose_shows_mini_run_description(self, capsys):
        """verbose=True shows mini-run description."""
        result = _make_full_result()

        render_full_preflight_result(result, verbose=True)
        captured = capsys.readouterr()

        # Should mention mini-run details
        assert "mini-run" in captured.out.lower()
        # Should mention warmup and sample configuration
        assert "1-chain" in captured.out or "1 chain" in captured.out.lower()

    def test_non_verbose_has_basic_info(self, capsys):
        """verbose=False still shows essential information."""
        result = _make_full_result()

        render_full_preflight_result(result, verbose=False)
        captured = capsys.readouterr()

        # Should have essential info
        assert "PASS" in captured.out
        assert "4.00 GB" in captured.out or "4.0 GB" in captured.out


class TestRenderFullPreflightGpuInfo:
    """Tests for GPU info display."""

    def test_render_shows_gpu_name(self, capsys):
        """Output shows GPU device name."""
        result = _make_full_result(device_name="NVIDIA RTX 4090")

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        assert "NVIDIA RTX 4090" in captured.out

    def test_render_shows_available_memory(self, capsys):
        """Output shows available GB from total GB."""
        result = _make_full_result()

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        assert "12.0 GB" in captured.out
        assert "16.0 GB" in captured.out

    def test_render_no_gpu_info_when_none(self, capsys):
        """Output omits GPU section when device_name is None."""
        result = _make_full_result(
            status=PreflightStatus.CANNOT_CHECK,
            message="Cannot query GPU",
            device_name=None,
        )

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        # Should not have GPU: section
        assert "GPU:" not in captured.out


class TestRenderFullPreflightSuggestions:
    """Tests for suggestions display."""

    def test_render_suggestions(self, capsys):
        """Suggestions are displayed."""
        result = FullPreflightResult(
            status=PreflightStatus.FAIL,
            measured_peak_gb=10.0,
            available_gb=8.0,
            total_gpu_gb=8.0,
            headroom_percent=-25.0,
            mini_run_seconds=15.0,
            message="Memory exceeded",
            suggestions=(
                "Need 2.0 GB more GPU memory",
                "Try reducing --num-chains (most effective)",
            ),
            device_name="Test GPU",
        )

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        assert "Suggestions" in captured.out
        assert "--num-chains" in captured.out

    def test_no_suggestions_section_when_empty(self, capsys):
        """No suggestions section when suggestions tuple is empty."""
        result = _make_full_result()

        render_full_preflight_result(result)
        captured = capsys.readouterr()

        # Should not have Suggestions section when empty
        assert "Suggestions:" not in captured.out


def _make_extrapolation_result(**overrides: object) -> ExtrapolationResult:
    """Create ExtrapolationResult with sensible defaults.

    Provides a PASS result with typical values. Override any field
    by passing keyword arguments.
    """
    dummy_calibration = CalibrationResult(
        fixed_overhead_gb=2.0,
        per_sample_gb=0.001,
        calibration_points=((10, 2.01), (50, 2.05)),
        config_hash="abc123",
        calibration_time=5.0,
    )

    defaults = {
        "status": PreflightStatus.PASS,
        "measured_peak_gb": 2.05,
        "projected_gb": 4.0,
        "target_samples": 2000,
        "calibration_samples": 60,
        "uncertainty_percent": 10.0,
        "available_gb": 12.0,
        "total_gpu_gb": 16.0,
        "headroom_percent": 66.7,
        "calibration": dummy_calibration,
        "from_cache": False,
        "message": "Test message",
        "suggestions": (),
        "device_name": "Test GPU",
    }
    defaults.update(overrides)
    return ExtrapolationResult(**defaults)


class TestRenderExtrapolationResult:
    """Tests for render_extrapolation_result function."""

    def test_renders_both_measured_and_projected(self, capsys):
        """Output contains BOTH measured peak AND projected estimate."""
        result = _make_extrapolation_result(
            measured_peak_gb=2.05,
            projected_gb=4.0,
            target_samples=2000,
        )

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        # CRITICAL: Both measured and projected must appear
        assert "2.05 GB" in captured.out  # Measured peak
        assert "4.0 GB" in captured.out  # Projected
        assert "Memory Projection" in captured.out

    def test_measured_peak_shows_calibration_context(self, capsys):
        """Output mentions calibration context (10+50 samples)."""
        result = _make_extrapolation_result()

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        # Should mention calibration at 10+50 samples
        assert "10+50 samples" in captured.out or "calibration" in captured.out.lower()
        assert "Measured peak" in captured.out

    def test_projected_shows_target_samples(self, capsys):
        """Projected line shows target sample count."""
        result = _make_extrapolation_result(
            target_samples=2000,
            projected_gb=4.0,
        )

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        # Should show target samples in projected line
        assert "2,000 samples" in captured.out or "2000 samples" in captured.out
        assert "Projected" in captured.out

    def test_renders_uncertainty(self, capsys):
        """Output shows uncertainty percentage."""
        result = _make_extrapolation_result(
            projected_gb=4.0,
            uncertainty_percent=10.0,
        )

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        # Should show +/-10%
        assert "+/-10%" in captured.out

    def test_verbose_shows_calibration_coefficients(self, capsys):
        """verbose=True shows fixed and per_sample coefficients."""
        dummy_calibration = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.001,
            calibration_points=((10, 2.01), (50, 2.05)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        result = _make_extrapolation_result(calibration=dummy_calibration)

        render_extrapolation_result(result, verbose=True)
        captured = capsys.readouterr()

        # Should show calibration formula
        assert "fixed=" in captured.out or "2.00 GB" in captured.out
        assert "GB/sample" in captured.out

    def test_shows_cache_status_from_cache(self, capsys):
        """Output shows 'from cache' when calibration was cached."""
        result = _make_extrapolation_result(from_cache=True)

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        assert "from cache" in captured.out

    def test_shows_cache_status_fresh(self, capsys):
        """Output shows 'fresh' when calibration was not cached."""
        result = _make_extrapolation_result(from_cache=False)

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        assert "fresh" in captured.out

    def test_pass_status_shows_pass(self, capsys):
        """Output contains 'PASS' for PASS status."""
        result = _make_extrapolation_result(status=PreflightStatus.PASS)

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        assert "PASS" in captured.out

    def test_fail_status_shows_fail(self, capsys):
        """Output contains 'FAIL' for FAIL status."""
        result = _make_extrapolation_result(
            status=PreflightStatus.FAIL,
            projected_gb=15.0,
            available_gb=10.0,
            message="Extrapolation failed: 15.0 GB projected exceeds 10.0 GB available",
        )

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        assert "FAIL" in captured.out

    def test_shows_gpu_info(self, capsys):
        """Output shows GPU name and memory info."""
        result = _make_extrapolation_result(
            device_name="NVIDIA RTX 4090",
            available_gb=12.0,
            total_gpu_gb=16.0,
        )

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        assert "NVIDIA RTX 4090" in captured.out
        assert "12.0 GB" in captured.out
        assert "16.0 GB" in captured.out

    def test_shows_suggestions(self, capsys):
        """Output shows suggestions when present."""
        result = _make_extrapolation_result(
            status=PreflightStatus.FAIL,
            suggestions=("Try reducing --num-chains",),
        )

        render_extrapolation_result(result)
        captured = capsys.readouterr()

        assert "Suggestions" in captured.out
        assert "--num-chains" in captured.out
