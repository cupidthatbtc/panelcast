"""Expanded unit tests for preflight calibration module."""

import pytest

from panelcast.preflight.calibrate import (
    CALIBRATION_SAMPLES,
    CalibrationError,
    CalibrationResult,
    calculate_calibration,
)


class TestCalibrationSamplesConstant:
    """Tests for CALIBRATION_SAMPLES constant."""

    def test_is_tuple_of_two_ints(self):
        assert isinstance(CALIBRATION_SAMPLES, tuple)
        assert len(CALIBRATION_SAMPLES) == 2
        assert all(isinstance(s, int) for s in CALIBRATION_SAMPLES)

    def test_values_are_10_and_50(self):
        assert CALIBRATION_SAMPLES == (10, 50)

    def test_ascending_order(self):
        assert CALIBRATION_SAMPLES[0] < CALIBRATION_SAMPLES[1]


class TestCalibrationError:
    """Tests for CalibrationError exception."""

    def test_is_exception(self):
        assert issubclass(CalibrationError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(CalibrationError):
            raise CalibrationError("test error")

    def test_message_preserved(self):
        with pytest.raises(CalibrationError, match="specific message"):
            raise CalibrationError("specific message")


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        return CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.005,
            calibration_points=((10, 1.05), (50, 1.25)),
            config_hash="abc123",
            calibration_time=30.0,
        )

    def test_fields_accessible(self, sample_result):
        assert sample_result.fixed_overhead_gb == 1.0
        assert sample_result.per_sample_gb == 0.005
        assert sample_result.config_hash == "abc123"
        assert sample_result.calibration_time == 30.0

    def test_is_frozen(self, sample_result):
        with pytest.raises(AttributeError):
            sample_result.fixed_overhead_gb = 2.0

    def test_extrapolate_basic(self, sample_result):
        result = sample_result.extrapolate(100)
        expected = 1.0 + 0.005 * 100
        assert result == pytest.approx(expected)

    def test_extrapolate_zero_samples(self, sample_result):
        result = sample_result.extrapolate(0)
        assert result == pytest.approx(1.0)

    def test_extrapolate_large_samples(self, sample_result):
        result = sample_result.extrapolate(10000)
        expected = 1.0 + 0.005 * 10000
        assert result == pytest.approx(expected)

    def test_extrapolate_matches_calibration_point(self):
        """Extrapolation at calibration points should match measured values."""
        result = CalibrationResult(
            fixed_overhead_gb=0.5,
            per_sample_gb=0.01,
            calibration_points=((10, 0.6), (50, 1.0)),
            config_hash="test",
            calibration_time=1.0,
        )
        assert result.extrapolate(10) == pytest.approx(0.6)
        assert result.extrapolate(50) == pytest.approx(1.0)

    def test_calibration_points_tuple(self, sample_result):
        points = sample_result.calibration_points
        assert isinstance(points, tuple)
        assert len(points) == 2
        assert len(points[0]) == 2
        assert len(points[1]) == 2


class TestCalculateCalibration:
    """Tests for calculate_calibration function."""

    def test_basic_calculation(self):
        point1 = (10, 1.05)
        point2 = (50, 1.25)
        fixed, per_sample = calculate_calibration(point1, point2)
        # slope = (1.25 - 1.05) / (50 - 10) = 0.005
        assert per_sample == pytest.approx(0.005)
        # intercept = 1.05 - 0.005 * 10 = 1.0
        assert fixed == pytest.approx(1.0)

    def test_exact_linear_points(self):
        """Two points on y = 2 + 0.01x should give exact coefficients."""
        point1 = (10, 2.1)
        point2 = (100, 3.0)
        fixed, per_sample = calculate_calibration(point1, point2)
        assert per_sample == pytest.approx(0.01)
        assert fixed == pytest.approx(2.0)

    def test_raises_for_identical_samples(self):
        with pytest.raises(CalibrationError, match="identical sample counts"):
            calculate_calibration((10, 1.0), (10, 2.0))

    def test_raises_for_negative_intercept(self):
        """Negative fixed overhead indicates invalid fit."""
        # point1: (10, 0.1), point2: (50, 3.0)
        # slope = 2.9/40 = 0.0725
        # intercept = 0.1 - 0.0725*10 = -0.625 (negative)
        with pytest.raises(CalibrationError, match="negative fixed overhead"):
            calculate_calibration((10, 0.1), (50, 3.0))

    def test_zero_intercept_passes(self):
        """Zero fixed overhead should not raise."""
        # point1: (10, 0.1), point2: (50, 0.5)
        # slope = 0.4/40 = 0.01
        # intercept = 0.1 - 0.01*10 = 0.0
        fixed, per_sample = calculate_calibration((10, 0.1), (50, 0.5))
        assert fixed == pytest.approx(0.0)
        assert per_sample == pytest.approx(0.01)

    def test_reversed_point_order(self):
        """Should work regardless of which point comes first."""
        p1 = (10, 1.05)
        p2 = (50, 1.25)
        fixed_a, slope_a = calculate_calibration(p1, p2)
        fixed_b, slope_b = calculate_calibration(p2, p1)
        assert fixed_a == pytest.approx(fixed_b)
        assert slope_a == pytest.approx(slope_b)

    def test_steep_slope(self):
        point1 = (10, 1.0)
        point2 = (50, 5.0)
        fixed, per_sample = calculate_calibration(point1, point2)
        assert per_sample == pytest.approx(0.1)
        assert fixed == pytest.approx(0.0)

    def test_flat_slope(self):
        """Zero slope means all memory is overhead."""
        point1 = (10, 2.0)
        point2 = (50, 2.0)
        fixed, per_sample = calculate_calibration(point1, point2)
        assert per_sample == pytest.approx(0.0)
        assert fixed == pytest.approx(2.0)
