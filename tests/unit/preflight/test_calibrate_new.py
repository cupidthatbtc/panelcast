"""New coverage-targeted tests for preflight/calibrate.py.

Targets the run_calibration function which orchestrates:
- serialize_model_args call
- Two _run_mini_mcmc_subprocess calls at CALIBRATION_SAMPLES points
- calculate_calibration for linear fit
- _derive_dimensions_from_model_args + compute_config_hash
- Temp file cleanup in finally block
- Error paths: subprocess failure, negative intercept
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


class TestRunCalibrationNegativeIntercept:
    """Tests for run_calibration when linear fit produces negative intercept."""

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_negative_intercept_raises_calibration_error(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """When two calibration points produce negative intercept, CalibrationError is raised."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        # Points that produce negative intercept:
        # (10, 1.0 GB) and (50, 5.0 GB)
        # slope = 4.0/40 = 0.1, intercept = 1.0 - 0.1*10 = 0.0 (borderline ok)
        # Let's use (10, 0.5 GB) and (50, 4.5 GB) => slope=0.1, intercept=0.5-1.0=-0.5
        mock_subprocess.side_effect = [
            {
                "success": True,
                "peak_memory_bytes": int(0.5 * 1024**3),
                "runtime_seconds": 5.0,
            },
            {
                "success": True,
                "peak_memory_bytes": int(4.5 * 1024**3),
                "runtime_seconds": 10.0,
            },
        ]

        model_args = {
            "y": np.ones(10),
            "X": np.ones((10, 2)),
            "n_artists": 2,
            "max_seq": 1,
        }

        with pytest.raises(CalibrationError, match="negative fixed overhead"):
            run_calibration(model_args)

        # Temp file should still be cleaned up
        assert not temp_file.exists()

    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_cleanup_on_negative_intercept(
        self,
        mock_serialize,
        mock_subprocess,
        tmp_path,
    ):
        """Temp file cleanup happens even when calculate_calibration raises."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(0.1 * 1024**3), "runtime_seconds": 2.0},
            {"success": True, "peak_memory_bytes": int(3.0 * 1024**3), "runtime_seconds": 5.0},
        ]

        model_args = {"y": np.ones(5), "X": np.ones((5, 1)), "n_artists": 1, "max_seq": 1}

        with pytest.raises(CalibrationError):
            run_calibration(model_args)

        assert not temp_file.exists()


class TestRunCalibrationSubprocessArgs:
    """Tests verifying subprocess call arguments."""

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_warmup_always_10(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """Each subprocess call uses num_warmup=10."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(2.0 * 1024**3), "runtime_seconds": 5.0},
            {"success": True, "peak_memory_bytes": int(2.5 * 1024**3), "runtime_seconds": 8.0},
        ]

        mock_derive.return_value = (10, 2, 2, 1)
        mock_hash.return_value = "warmup_test_hash"

        model_args = {"y": np.ones(10), "X": np.ones((10, 2)), "n_artists": 2, "max_seq": 1}

        run_calibration(model_args)

        for call in mock_subprocess.call_args_list:
            assert call[1]["num_warmup"] == 10

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_calibration_result_extrapolation_consistent(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """CalibrationResult.extrapolate at calibration points matches measurements."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        # Exact linear: y = 2.0 + 0.02*x
        # At 10 samples: 2.2 GB, at 50 samples: 3.0 GB
        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(2.2 * 1024**3), "runtime_seconds": 5.0},
            {"success": True, "peak_memory_bytes": int(3.0 * 1024**3), "runtime_seconds": 10.0},
        ]

        mock_derive.return_value = (100, 10, 5, 3)
        mock_hash.return_value = "extrap_test_hash"

        model_args = {"y": np.ones(100), "X": np.ones((100, 5)), "n_artists": 10, "max_seq": 3}

        result = run_calibration(model_args)

        # Verify extrapolation at calibration points
        assert result.extrapolate(10) == pytest.approx(2.2, rel=0.01)
        assert result.extrapolate(50) == pytest.approx(3.0, rel=0.01)


class TestRunCalibrationFilePath:
    """Tests for serialize path handling in run_calibration."""

    @mock.patch("panelcast.preflight.cache.compute_config_hash")
    @mock.patch("panelcast.preflight.full_check._derive_dimensions_from_model_args")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    @mock.patch("panelcast.preflight.full_check.serialize_model_args")
    def test_same_path_used_for_both_calls(
        self,
        mock_serialize,
        mock_subprocess,
        mock_derive,
        mock_hash,
        tmp_path,
    ):
        """Both subprocess calls receive the same serialized model args path."""
        temp_file = tmp_path / "args.json"
        temp_file.write_text("{}")
        mock_serialize.return_value = temp_file

        mock_subprocess.side_effect = [
            {"success": True, "peak_memory_bytes": int(1.5 * 1024**3), "runtime_seconds": 3.0},
            {"success": True, "peak_memory_bytes": int(2.0 * 1024**3), "runtime_seconds": 5.0},
        ]

        mock_derive.return_value = (20, 3, 2, 1)
        mock_hash.return_value = "path_test______"

        model_args = {"y": np.ones(20), "X": np.ones((20, 2)), "n_artists": 3, "max_seq": 1}

        run_calibration(model_args)

        # Both calls should receive the same args_path
        first_call_path = mock_subprocess.call_args_list[0][0][0]
        second_call_path = mock_subprocess.call_args_list[1][0][0]
        assert first_call_path == second_call_path == temp_file


class TestCalculateCalibrationAdditional:
    """Additional tests for calculate_calibration edge cases."""

    def test_very_large_values(self):
        """Works with very large memory values (e.g., A100 80GB)."""
        fixed, per_sample = calculate_calibration((10, 40.0), (50, 60.0))
        assert per_sample == pytest.approx(0.5)
        assert fixed == pytest.approx(35.0)

    def test_both_same_value_zero_slope(self):
        """Identical memory at different samples gives zero slope."""
        fixed, per_sample = calculate_calibration((10, 5.0), (100, 5.0))
        assert per_sample == pytest.approx(0.0)
        assert fixed == pytest.approx(5.0)

    def test_identical_samples_raises_with_count(self):
        """Error message for identical samples includes the sample count."""
        with pytest.raises(CalibrationError, match="both points have 10"):
            calculate_calibration((10, 1.0), (10, 2.0))

    def test_exact_zero_intercept_is_valid(self):
        """Exactly zero intercept does not raise (only negative raises)."""
        # y = 0.01 * x => intercept = 0
        fixed, per_sample = calculate_calibration((10, 0.1), (50, 0.5))
        assert fixed == pytest.approx(0.0)
        assert per_sample == pytest.approx(0.01)
