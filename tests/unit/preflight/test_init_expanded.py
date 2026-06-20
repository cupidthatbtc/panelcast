"""Expanded tests for preflight __init__ module: PreflightStatus, PreflightResult, etc."""

from dataclasses import FrozenInstanceError

import pytest

from panelcast.gpu_memory.estimate import MemoryEstimate
from panelcast.preflight import (
    ExtrapolationResult,
    FullPreflightResult,
    PreflightResult,
    PreflightStatus,
)
from panelcast.preflight.calibrate import CalibrationResult


class TestPreflightStatus:
    """Tests for PreflightStatus enum."""

    def test_pass_value(self):
        assert PreflightStatus.PASS.value == "pass"

    def test_fail_value(self):
        assert PreflightStatus.FAIL.value == "fail"

    def test_warning_value(self):
        assert PreflightStatus.WARNING.value == "warning"

    def test_cannot_check_value(self):
        assert PreflightStatus.CANNOT_CHECK.value == "cannot_check"

    def test_member_count(self):
        assert len(PreflightStatus) == 4

    def test_members_are_unique(self):
        values = [s.value for s in PreflightStatus]
        assert len(values) == len(set(values))


class TestPreflightResult:
    """Tests for PreflightResult frozen dataclass."""

    @pytest.fixture
    def estimate(self):
        return MemoryEstimate(base_model_gb=1.0, per_chain_gb=0.5, jit_buffer_gb=0.8, num_chains=4)

    @pytest.fixture
    def pass_result(self, estimate):
        return PreflightResult(
            status=PreflightStatus.PASS,
            estimate=estimate,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=62.0,
            message="Memory check passed",
            suggestions=(),
            device_name="Test GPU",
        )

    def test_frozen(self, pass_result):
        with pytest.raises(FrozenInstanceError):
            pass_result.status = PreflightStatus.FAIL

    def test_exit_code_pass(self, pass_result):
        assert pass_result.exit_code == 0

    def test_exit_code_fail(self, estimate):
        result = PreflightResult(
            status=PreflightStatus.FAIL,
            estimate=estimate,
            available_gb=1.0,
            total_gpu_gb=1.0,
            headroom_percent=-200.0,
            message="failed",
            suggestions=(),
        )
        assert result.exit_code == 1

    def test_exit_code_warning(self, estimate):
        result = PreflightResult(
            status=PreflightStatus.WARNING,
            estimate=estimate,
            available_gb=5.0,
            total_gpu_gb=5.0,
            headroom_percent=5.0,
            message="warning",
            suggestions=(),
        )
        assert result.exit_code == 2

    def test_exit_code_cannot_check(self):
        result = PreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            estimate=None,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            message="cannot check",
            suggestions=(),
        )
        assert result.exit_code == 2

    def test_device_name_optional(self, estimate):
        result = PreflightResult(
            status=PreflightStatus.PASS,
            estimate=estimate,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=62.0,
            message="ok",
            suggestions=(),
        )
        assert result.device_name is None

    def test_estimate_can_be_none(self):
        result = PreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            estimate=None,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            message="no gpu",
            suggestions=("Try cpu",),
        )
        assert result.estimate is None

    def test_suggestions_tuple(self, pass_result):
        assert isinstance(pass_result.suggestions, tuple)


class TestFullPreflightResult:
    """Tests for FullPreflightResult frozen dataclass."""

    @pytest.fixture
    def full_result(self):
        return FullPreflightResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=3.5,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=65.0,
            mini_run_seconds=45.0,
            message="Full preflight passed",
            suggestions=(),
            device_name="Test GPU",
        )

    def test_frozen(self, full_result):
        with pytest.raises(FrozenInstanceError):
            full_result.measured_peak_gb = 0.0

    def test_exit_code_pass(self, full_result):
        assert full_result.exit_code == 0

    def test_exit_code_fail(self):
        result = FullPreflightResult(
            status=PreflightStatus.FAIL,
            measured_peak_gb=15.0,
            available_gb=10.0,
            total_gpu_gb=12.0,
            headroom_percent=-50.0,
            mini_run_seconds=30.0,
            message="failed",
            suggestions=(),
        )
        assert result.exit_code == 1

    def test_fields_accessible(self, full_result):
        assert full_result.measured_peak_gb == 3.5
        assert full_result.mini_run_seconds == 45.0
        assert full_result.device_name == "Test GPU"

    def test_device_name_optional(self):
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
        assert result.device_name is None


class TestExtrapolationResult:
    """Tests for ExtrapolationResult frozen dataclass."""

    @pytest.fixture
    def calibration(self):
        return CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.005,
            calibration_points=((10, 1.05), (50, 1.25)),
            config_hash="extraptest12345",
            calibration_time=30.0,
        )

    @pytest.fixture
    def extrap_result(self, calibration):
        return ExtrapolationResult(
            status=PreflightStatus.WARNING,
            measured_peak_gb=1.25,
            projected_gb=6.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=8.0,
            total_gpu_gb=10.0,
            headroom_percent=25.0,
            calibration=calibration,
            from_cache=False,
            message="Projected memory close to available",
            suggestions=("Try reducing chains",),
            device_name="Test GPU",
        )

    def test_frozen(self, extrap_result):
        with pytest.raises(FrozenInstanceError):
            extrap_result.projected_gb = 0.0

    def test_exit_code_warning(self, extrap_result):
        assert extrap_result.exit_code == 2

    def test_fields_accessible(self, extrap_result):
        assert extrap_result.projected_gb == 6.0
        assert extrap_result.target_samples == 1000
        assert extrap_result.calibration_samples == 60
        assert extrap_result.uncertainty_percent == 10.0
        assert extrap_result.from_cache is False

    def test_calibration_accessible(self, extrap_result):
        assert extrap_result.calibration.fixed_overhead_gb == 1.0
        assert extrap_result.calibration.per_sample_gb == 0.005

    def test_device_name_optional(self, calibration):
        result = ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.0,
            projected_gb=3.0,
            target_samples=500,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=70.0,
            calibration=calibration,
            from_cache=True,
            message="ok",
            suggestions=(),
        )
        assert result.device_name is None

    def test_from_cache_true(self, calibration):
        result = ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.0,
            projected_gb=3.0,
            target_samples=500,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=10.0,
            headroom_percent=70.0,
            calibration=calibration,
            from_cache=True,
            message="ok",
            suggestions=(),
        )
        assert result.from_cache is True
