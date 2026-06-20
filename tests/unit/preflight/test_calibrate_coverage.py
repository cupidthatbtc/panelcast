"""Coverage-targeted tests for preflight calibrate module.

Targets missed lines and branches in preflight/calibrate.py:
- run_calibration function: orchestration with mocked subprocess calls
- CalibrationError on mini-run failure
- Temp file cleanup in finally block
- calculate_calibration with edge case inputs
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from panelcast.preflight.calibrate import (
    CALIBRATION_SAMPLES,
    CalibrationError,
    CalibrationResult,
    calculate_calibration,
    run_calibration,
)


class TestRunCalibrationSuccess:
    """Tests for run_calibration with successful mini-run results."""

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_successful_calibration(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """Successful two-point calibration returns valid CalibrationResult."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        # Two calibration points: 10 samples -> 2.2GB, 50 samples -> 3.0GB
        mock_subprocess.side_effect = [
            {
                "success": True,
                "peak_memory_bytes": int(2.2 * 1024**3),
                "runtime_seconds": 10.0,
            },
            {
                "success": True,
                "peak_memory_bytes": int(3.0 * 1024**3),
                "runtime_seconds": 20.0,
            },
        ]

        mock_derive.return_value = (100, 10, 5, 3)
        mock_hash.return_value = "abcdef1234567890"

        model_args = {"y": np.ones(100), "X": np.ones((100, 5)), "n_artists": 10, "max_seq": 3}

        result = run_calibration(model_args, timeout_seconds=60)

        assert isinstance(result, CalibrationResult)
        assert result.fixed_overhead_gb >= 0
        assert result.per_sample_gb >= 0
        assert result.config_hash == "abcdef1234567890"
        assert result.calibration_time > 0
        assert len(result.calibration_points) == 2

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_calibration_calls_subprocess_twice(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """run_calibration calls _run_mini_mcmc_subprocess for each calibration point."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(2.0 * 1024**3), "runtime_seconds": 5.0},
            {"success": True, "peak_memory_bytes": int(3.0 * 1024**3), "runtime_seconds": 8.0},
        ]

        mock_derive.return_value = (50, 5, 3, 2)
        mock_hash.return_value = "hash1234________"

        model_args = {"y": np.ones(50), "X": np.ones((50, 3)), "n_artists": 5, "max_seq": 2}

        run_calibration(model_args)

        assert mock_subprocess.call_count == 2
        # First call: 10 samples
        first_call = mock_subprocess.call_args_list[0]
        assert first_call[1]["num_samples"] == CALIBRATION_SAMPLES[0]
        # Second call: 50 samples
        second_call = mock_subprocess.call_args_list[1]
        assert second_call[1]["num_samples"] == CALIBRATION_SAMPLES[1]

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_timeout_forwarded_to_subprocess(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """timeout_seconds is forwarded to each subprocess call."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(1.5 * 1024**3), "runtime_seconds": 3.0},
            {"success": True, "peak_memory_bytes": int(2.0 * 1024**3), "runtime_seconds": 5.0},
        ]

        mock_derive.return_value = (20, 3, 2, 1)
        mock_hash.return_value = "hash5678________"

        model_args = {"y": np.ones(20), "X": np.ones((20, 2)), "n_artists": 3, "max_seq": 1}

        run_calibration(model_args, timeout_seconds=300)

        for call in mock_subprocess.call_args_list:
            assert call[1]["timeout_seconds"] == 300


class TestRunCalibrationFailure:
    """Tests for run_calibration when mini-run fails."""

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_first_point_failure_raises(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """CalibrationError raised when first mini-run fails."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.return_value = {
            "success": False,
            "error": "CUDA out of memory",
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        with pytest.raises(CalibrationError, match="Calibration failed at 10 samples"):
            run_calibration(model_args)

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_second_point_failure_raises(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """CalibrationError raised when second mini-run fails."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(2.0 * 1024**3), "runtime_seconds": 5.0},
            {
                "success": False,
                "error": "Timeout exceeded",
                "peak_memory_bytes": 0,
                "runtime_seconds": 120.0,
            },
        ]

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        with pytest.raises(CalibrationError, match="Calibration failed at 50 samples"):
            run_calibration(model_args)

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_failure_error_message_preserved(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """Error message from mini-run is included in CalibrationError."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.return_value = {
            "success": False,
            "error": "JAX compilation failed: invalid shapes",
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        with pytest.raises(CalibrationError, match="JAX compilation failed"):
            run_calibration(model_args)

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_failure_with_missing_success_key(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """Missing 'success' key in result is treated as failure."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.return_value = {
            "error": "Some error",
            "peak_memory_bytes": 0,
        }

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        with pytest.raises(CalibrationError, match="Calibration failed"):
            run_calibration(model_args)

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_failure_with_unknown_error(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """Missing 'error' key in failed result uses 'Unknown error' fallback."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.return_value = {
            "success": False,
            "peak_memory_bytes": 0,
        }

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        with pytest.raises(CalibrationError, match="Unknown error"):
            run_calibration(model_args)


class TestRunCalibrationCleanup:
    """Tests for temp file cleanup in run_calibration."""

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_temp_file_cleaned_on_success(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """Temp file is removed after successful calibration."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(2.0 * 1024**3), "runtime_seconds": 5.0},
            {"success": True, "peak_memory_bytes": int(2.5 * 1024**3), "runtime_seconds": 8.0},
        ]

        mock_derive.return_value = (10, 2, 2, 1)
        mock_hash.return_value = "cleanup_test____"

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        run_calibration(model_args)

        assert not temp_file.exists()

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_temp_file_cleaned_on_failure(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """Temp file is removed even when calibration raises."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.return_value = {
            "success": False,
            "error": "Failed",
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        with pytest.raises(CalibrationError):
            run_calibration(model_args)

        assert not temp_file.exists()


class TestCalculateCalibrationEdgeCases:
    """Additional edge cases for calculate_calibration."""

    def test_very_small_slope(self):
        """Very small per-sample cost is calculated correctly."""
        # (10, 2.001) and (50, 2.005) -> slope = 0.004/40 = 0.0001
        fixed, per_sample = calculate_calibration((10, 2.001), (50, 2.005))
        assert per_sample == pytest.approx(0.0001)
        assert fixed == pytest.approx(2.0)

    def test_large_sample_counts(self):
        """Works with large sample counts."""
        fixed, per_sample = calculate_calibration((1000, 5.0), (5000, 9.0))
        assert per_sample == pytest.approx(0.001)
        assert fixed == pytest.approx(4.0)

    def test_identical_samples_different_message(self):
        """Identical sample count error mentions the count."""
        with pytest.raises(CalibrationError, match="50"):
            calculate_calibration((50, 1.0), (50, 2.0))

    def test_negative_slope_with_positive_intercept(self):
        """Negative slope is valid if intercept is positive."""
        # (10, 5.0) and (50, 3.0) -> slope = -2/40 = -0.05
        # intercept = 5.0 - (-0.05)*10 = 5.5
        fixed, per_sample = calculate_calibration((10, 5.0), (50, 3.0))
        assert per_sample == pytest.approx(-0.05)
        assert fixed == pytest.approx(5.5)


class TestCalibrationResultExtrapolateEdgeCases:
    """Edge cases for CalibrationResult.extrapolate."""

    def test_extrapolate_with_zero_per_sample(self):
        """Zero per_sample_gb means extrapolation always returns fixed overhead."""
        result = CalibrationResult(
            fixed_overhead_gb=3.0,
            per_sample_gb=0.0,
            calibration_points=((10, 3.0), (50, 3.0)),
            config_hash="zero_slope",
            calibration_time=10.0,
        )
        assert result.extrapolate(0) == pytest.approx(3.0)
        assert result.extrapolate(1000) == pytest.approx(3.0)
        assert result.extrapolate(100000) == pytest.approx(3.0)

    def test_extrapolate_negative_per_sample(self):
        """Negative per_sample_gb (unusual but valid) extrapolates correctly."""
        result = CalibrationResult(
            fixed_overhead_gb=5.0,
            per_sample_gb=-0.01,
            calibration_points=((10, 4.9), (50, 4.5)),
            config_hash="neg_slope",
            calibration_time=10.0,
        )
        assert result.extrapolate(100) == pytest.approx(4.0)

    def test_extrapolate_very_large_target(self):
        """Extrapolation with very large target sample count."""
        result = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.001,
            calibration_points=((10, 1.01), (50, 1.05)),
            config_hash="large_target",
            calibration_time=10.0,
        )
        projected = result.extrapolate(1_000_000)
        assert projected == pytest.approx(1001.0)


class TestCalibrationResultFields:
    """Tests for CalibrationResult field access and immutability."""

    def test_all_fields_accessible(self):
        """All CalibrationResult fields are accessible."""
        result = CalibrationResult(
            fixed_overhead_gb=1.5,
            per_sample_gb=0.02,
            calibration_points=((10, 1.7), (50, 2.5)),
            config_hash="fields_test",
            calibration_time=45.3,
        )
        assert result.fixed_overhead_gb == 1.5
        assert result.per_sample_gb == 0.02
        assert result.calibration_points == ((10, 1.7), (50, 2.5))
        assert result.config_hash == "fields_test"
        assert result.calibration_time == 45.3

    def test_calibration_points_are_tuples(self):
        """calibration_points contains tuples, not lists."""
        result = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.01,
            calibration_points=((10, 1.1), (50, 1.5)),
            config_hash="tuple_test",
            calibration_time=20.0,
        )
        assert isinstance(result.calibration_points[0], tuple)
        assert isinstance(result.calibration_points[1], tuple)
