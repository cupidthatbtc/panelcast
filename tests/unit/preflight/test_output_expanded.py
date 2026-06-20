"""Expanded tests for preflight output rendering: format_uncertainty, _format_status_line."""

import pytest

from panelcast.gpu_memory.estimate import MemoryEstimate
from panelcast.preflight import (
    ExtrapolationResult,
    FullPreflightResult,
    PreflightResult,
    PreflightStatus,
)
from panelcast.preflight.calibrate import CalibrationResult
from panelcast.preflight.output import (
    _format_status_line,
    format_uncertainty,
    render_extrapolation_result,
    render_full_preflight_result,
    render_preflight_result,
)


class TestFormatUncertainty:
    """Tests for format_uncertainty function."""

    def test_basic_format(self):
        result = format_uncertainty(15.8, 10.0)
        assert result == "15.8 GB +/-10%"

    def test_zero_uncertainty(self):
        result = format_uncertainty(5.0, 0.0)
        assert result == "5.0 GB +/-0%"

    def test_large_uncertainty(self):
        result = format_uncertainty(3.0, 50.0)
        assert result == "3.0 GB +/-50%"

    def test_fractional_gb(self):
        result = format_uncertainty(0.5, 15.0)
        assert result == "0.5 GB +/-15%"

    def test_large_projected(self):
        result = format_uncertainty(100.0, 5.0)
        assert result == "100.0 GB +/-5%"

    def test_contains_gb(self):
        result = format_uncertainty(1.0, 1.0)
        assert "GB" in result

    def test_contains_plus_minus(self):
        result = format_uncertainty(1.0, 10.0)
        assert "+/-" in result


class TestFormatStatusLine:
    """Tests for _format_status_line function."""

    def test_pass_contains_green(self):
        line = _format_status_line(PreflightStatus.PASS, "check passed")
        assert "green" in line
        assert "PASS" in line
        assert "check passed" in line

    def test_fail_contains_red(self):
        line = _format_status_line(PreflightStatus.FAIL, "check failed")
        assert "red" in line
        assert "FAIL" in line

    def test_warning_contains_yellow(self):
        line = _format_status_line(PreflightStatus.WARNING, "low headroom")
        assert "yellow" in line
        assert "WARNING" in line

    def test_cannot_check_contains_yellow(self):
        line = _format_status_line(PreflightStatus.CANNOT_CHECK, "no gpu")
        assert "yellow" in line
        assert "CANNOT CHECK" in line

    def test_message_preserved(self):
        msg = "custom message with details"
        line = _format_status_line(PreflightStatus.PASS, msg)
        assert msg in line


class TestRenderFullPreflightResult:
    """Tests for render_full_preflight_result function."""

    @pytest.fixture
    def full_result(self):
        return FullPreflightResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=3.5,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=65.0,
            mini_run_seconds=45.2,
            message="Full preflight passed",
            suggestions=(),
            device_name="NVIDIA Test GPU",
        )

    def test_renders_without_error(self, capsys, full_result):
        render_full_preflight_result(full_result)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_shows_measured_peak(self, capsys, full_result):
        render_full_preflight_result(full_result)
        captured = capsys.readouterr()
        assert "3.50 GB" in captured.out

    def test_shows_mini_run_time(self, capsys, full_result):
        render_full_preflight_result(full_result)
        captured = capsys.readouterr()
        assert "45.2" in captured.out

    def test_shows_gpu_name(self, capsys, full_result):
        render_full_preflight_result(full_result)
        captured = capsys.readouterr()
        assert "NVIDIA Test GPU" in captured.out

    def test_verbose_mode(self, capsys, full_result):
        render_full_preflight_result(full_result, verbose=True)
        captured = capsys.readouterr()
        assert "actual measurement" in captured.out or "mini-run" in captured.out

    def test_no_device_name(self, capsys):
        result = FullPreflightResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=2.0,
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=80.0,
            mini_run_seconds=20.0,
            message="ok",
            suggestions=(),
        )
        render_full_preflight_result(result)
        captured = capsys.readouterr()
        # Should not crash with no device name
        assert "PASS" in captured.out

    def test_with_suggestions(self, capsys):
        result = FullPreflightResult(
            status=PreflightStatus.WARNING,
            measured_peak_gb=8.0,
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=20.0,
            mini_run_seconds=30.0,
            message="low headroom",
            suggestions=("Try --num-chains 2", "Reduce samples"),
        )
        render_full_preflight_result(result)
        captured = capsys.readouterr()
        assert "--num-chains 2" in captured.out


class TestRenderExtrapolationResult:
    """Tests for render_extrapolation_result function."""

    @pytest.fixture
    def calibration(self):
        return CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.005,
            calibration_points=((10, 1.05), (50, 1.25)),
            config_hash="testhash12345678",
            calibration_time=30.0,
        )

    @pytest.fixture
    def extrap_result(self, calibration):
        return ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.25,
            projected_gb=6.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=40.0,
            calibration=calibration,
            from_cache=False,
            message="Projected OK",
            suggestions=(),
            device_name="Test GPU",
        )

    def test_renders_without_error(self, capsys, extrap_result):
        render_extrapolation_result(extrap_result)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_shows_measured_peak(self, capsys, extrap_result):
        render_extrapolation_result(extrap_result)
        captured = capsys.readouterr()
        assert "1.25 GB" in captured.out

    def test_shows_projected(self, capsys, extrap_result):
        render_extrapolation_result(extrap_result)
        captured = capsys.readouterr()
        assert "6.0 GB" in captured.out

    def test_shows_target_samples(self, capsys, extrap_result):
        render_extrapolation_result(extrap_result)
        captured = capsys.readouterr()
        assert "1,000" in captured.out

    def test_verbose_shows_coefficients(self, capsys, extrap_result):
        render_extrapolation_result(extrap_result, verbose=True)
        captured = capsys.readouterr()
        assert "fixed=" in captured.out or "Calibration:" in captured.out

    def test_fresh_cache_message(self, capsys, extrap_result):
        render_extrapolation_result(extrap_result)
        captured = capsys.readouterr()
        assert "fresh" in captured.out

    def test_from_cache_message(self, capsys, calibration):
        result = ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.0,
            projected_gb=5.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=50.0,
            calibration=calibration,
            from_cache=True,
            message="ok",
            suggestions=(),
        )
        render_extrapolation_result(result)
        captured = capsys.readouterr()
        assert "cache" in captured.out
