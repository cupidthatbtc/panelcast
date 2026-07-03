"""Tests for full preflight check module."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from panelcast.gpu_memory import GpuMemoryInfo
from panelcast.pipelines.errors import GpuMemoryError
from panelcast.preflight import PreflightStatus
from panelcast.preflight.full_check import (
    _create_dummy_calibration,
    _derive_dimensions_from_model_args,
    _generate_extrapolation_message,
    _generate_extrapolation_suggestions,
    _generate_message,
    _generate_suggestions,
    _run_mini_mcmc_subprocess,
    calculate_headroom_percent,
    run_extrapolated_preflight_check,
    run_full_preflight_check,
    serialize_model_args,
)


class TestSerializeModelArgs:
    """Tests for serialize_model_args function."""

    def test_serialize_model_args_converts_arrays(self):
        """JAX arrays are converted to Python lists."""
        import jax.numpy as jnp

        model_args = {
            "artist_idx": jnp.array([0, 1, 2], dtype=jnp.int32),
            "y": jnp.array([70.0, 80.0, 75.0], dtype=jnp.float32),
            "n_artists": 3,
        }

        args_path = serialize_model_args(model_args)
        try:
            with open(args_path) as f:
                loaded = json.load(f)

            # Arrays should be lists
            assert loaded["artist_idx"] == [0, 1, 2]
            assert loaded["y"] == pytest.approx([70.0, 80.0, 75.0])
            # Scalars preserved
            assert loaded["n_artists"] == 3
        finally:
            args_path.unlink(missing_ok=True)

    def test_serialize_model_args_handles_scalars(self):
        """Scalar values are preserved directly."""
        model_args = {
            "n_artists": 50,
            "max_seq": 10,
            "n_exponent": 0.33,
            "learn_n_exponent": True,
        }

        args_path = serialize_model_args(model_args)
        try:
            with open(args_path) as f:
                loaded = json.load(f)

            assert loaded["n_artists"] == 50
            assert loaded["max_seq"] == 10
            assert loaded["n_exponent"] == 0.33
            assert loaded["learn_n_exponent"] is True
        finally:
            args_path.unlink(missing_ok=True)

    def test_serialize_model_args_handles_2d_arrays(self):
        """2D arrays (feature matrices) are converted correctly."""
        import jax.numpy as jnp

        model_args = {
            "X": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        }

        args_path = serialize_model_args(model_args)
        try:
            with open(args_path) as f:
                loaded = json.load(f)

            assert loaded["X"] == [[1.0, 2.0], [3.0, 4.0]]
        finally:
            args_path.unlink(missing_ok=True)

    def test_serialize_model_args_returns_path(self):
        """Returns Path object to temp file."""
        model_args = {"n_artists": 10}

        args_path = serialize_model_args(model_args)
        try:
            assert isinstance(args_path, Path)
            assert args_path.exists()
            assert args_path.suffix == ".json"
        finally:
            args_path.unlink(missing_ok=True)


class TestRunFullPreflightCheckStatusDetermination:
    """Tests for run_full_preflight_check status determination."""

    # Sample model args for testing
    SAMPLE_MODEL_ARGS: ClassVar[dict[str, Any]] = {
        "artist_idx": [0, 1, 0],
        "album_seq": [1, 1, 2],
        "prev_score": [0.0, 0.0, 75.0],
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "y": [70.0, 80.0, 75.0],
        "n_artists": 2,
        "max_seq": 2,
    }

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_run_full_preflight_check_pass(self, mock_subprocess, mock_query):
        """PASS when measured peak < available * (1 - headroom_target)."""
        # 16 GB total, 12 GB free
        mock_query.return_value = GpuMemoryInfo(
            device_name="NVIDIA RTX 4090",
            total_bytes=16 * 1024**3,
            used_bytes=4 * 1024**3,
            free_bytes=12 * 1024**3,
        )

        # 4 GB measured peak (4 < 12 * 0.8 = 9.6 GB threshold)
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": 4 * 1024**3,
            "runtime_seconds": 10.5,
        }

        result = run_full_preflight_check(self.SAMPLE_MODEL_ARGS)

        assert result.status == PreflightStatus.PASS
        assert result.measured_peak_gb == 4.0
        assert result.available_gb == 12.0
        assert result.total_gpu_gb == 16.0
        assert result.device_name == "NVIDIA RTX 4090"

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_run_full_preflight_check_fail(self, mock_subprocess, mock_query):
        """FAIL when measured peak > available."""
        # 8 GB total, 6 GB free
        mock_query.return_value = GpuMemoryInfo(
            device_name="NVIDIA GTX 1080",
            total_bytes=8 * 1024**3,
            used_bytes=2 * 1024**3,
            free_bytes=6 * 1024**3,
        )

        # 8 GB measured peak (8 > 6 GB available)
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": 8 * 1024**3,
            "runtime_seconds": 15.0,
        }

        result = run_full_preflight_check(self.SAMPLE_MODEL_ARGS)

        assert result.status == PreflightStatus.FAIL
        assert result.measured_peak_gb == 8.0
        assert result.available_gb == 6.0
        # Should have suggestions for FAIL
        assert len(result.suggestions) > 0

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_run_full_preflight_check_warning(self, mock_subprocess, mock_query):
        """WARNING when fits but headroom < headroom_target."""
        # 10 GB total, 8 GB free
        mock_query.return_value = GpuMemoryInfo(
            device_name="NVIDIA RTX 3070",
            total_bytes=10 * 1024**3,
            used_bytes=2 * 1024**3,
            free_bytes=8 * 1024**3,
        )

        # 7 GB measured peak (7 < 8 but 7 > 8 * 0.8 = 6.4 -> low headroom)
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": 7 * 1024**3,
            "runtime_seconds": 12.0,
        }

        result = run_full_preflight_check(self.SAMPLE_MODEL_ARGS)

        assert result.status == PreflightStatus.WARNING
        assert result.measured_peak_gb == 7.0
        # Should have suggestions for WARNING
        assert len(result.suggestions) > 0

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    def test_run_full_preflight_check_gpu_error(self, mock_query):
        """CANNOT_CHECK when GPU query fails."""
        mock_query.side_effect = GpuMemoryError("No GPU detected")

        result = run_full_preflight_check(self.SAMPLE_MODEL_ARGS)

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert "Cannot query GPU" in result.message
        assert result.measured_peak_gb == 0.0
        assert result.exit_code == 2

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_run_full_preflight_check_subprocess_failure(self, mock_subprocess, mock_query):
        """CANNOT_CHECK when subprocess fails."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="NVIDIA RTX 4090",
            total_bytes=16 * 1024**3,
            used_bytes=0,
            free_bytes=16 * 1024**3,
        )

        mock_subprocess.return_value = {
            "success": False,
            "error": "CUDA out of memory",
            "peak_memory_bytes": 0,
            "runtime_seconds": 5.0,
        }

        result = run_full_preflight_check(self.SAMPLE_MODEL_ARGS)

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert "Mini-run failed" in result.message

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_run_full_preflight_check_custom_headroom(self, mock_subprocess, mock_query):
        """Custom headroom_target affects PASS/WARNING threshold."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        # 7.5 GB peak, 10 GB available
        # Default 20% headroom: 7.5 < 10 * 0.8 = 8 -> PASS
        # With 30% headroom: 7.5 > 10 * 0.7 = 7 -> WARNING
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": int(7.5 * 1024**3),
            "runtime_seconds": 10.0,
        }

        # With 30% headroom target, should be WARNING
        result = run_full_preflight_check(self.SAMPLE_MODEL_ARGS, headroom_target=0.30)

        assert result.status == PreflightStatus.WARNING


class TestSubprocessEnvironment:
    """Tests for subprocess environment configuration."""

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("subprocess.run")
    def test_subprocess_environment_includes_prealloc_disable(self, mock_run, mock_query):
        """XLA_PYTHON_CLIENT_PREALLOCATE=false is set in subprocess env."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=16 * 1024**3,
            used_bytes=0,
            free_bytes=16 * 1024**3,
        )

        # Return a CompletedProcess with valid JSON output
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"success": true, "peak_memory_bytes": 1073741824, "runtime_seconds": 5.0}',
            stderr="",
        )

        # Import to trigger subprocess
        from panelcast.preflight.full_check import _run_mini_mcmc_subprocess

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": 1}, f)
            temp_path = Path(f.name)

        try:
            _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)

            # Verify subprocess.run was called with env containing the flag
            call_kwargs = mock_run.call_args[1]
            env = call_kwargs["env"]
            assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
            assert env["TF_CPP_MIN_LOG_LEVEL"] == "2"
        finally:
            temp_path.unlink(missing_ok=True)


class TestSubprocessTimeout:
    """Tests for subprocess timeout handling."""

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("subprocess.run")
    def test_subprocess_timeout_returns_cannot_check(self, mock_run, mock_query):
        """Subprocess timeout results in CANNOT_CHECK status."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=16 * 1024**3,
            used_bytes=0,
            free_bytes=16 * 1024**3,
        )

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=120)

        result = run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            },
            timeout_seconds=120,
        )

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert "timeout" in result.message.lower()


class TestTempFileCleanup:
    """Tests for temp file cleanup."""

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_temp_file_cleaned_up_on_success(self, mock_subprocess, mock_query):
        """Temp file is cleaned up after successful run."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=16 * 1024**3,
            used_bytes=0,
            free_bytes=16 * 1024**3,
        )

        # Track what path was used
        captured_path = None

        def capture_path(args_path, timeout_seconds):
            nonlocal captured_path
            captured_path = args_path
            return {
                "success": True,
                "peak_memory_bytes": 1 * 1024**3,
                "runtime_seconds": 5.0,
            }

        mock_subprocess.side_effect = capture_path

        run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            }
        )

        # Temp file should be cleaned up
        assert captured_path is not None
        assert not captured_path.exists()

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_temp_file_cleaned_up_on_subprocess_failure(self, mock_subprocess, mock_query):
        """Temp file is cleaned up even when subprocess fails."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=16 * 1024**3,
            used_bytes=0,
            free_bytes=16 * 1024**3,
        )

        captured_path = None

        def capture_and_fail(args_path, timeout_seconds):
            nonlocal captured_path
            captured_path = args_path
            return {
                "success": False,
                "error": "Test failure",
                "peak_memory_bytes": 0,
                "runtime_seconds": 0.0,
            }

        mock_subprocess.side_effect = capture_and_fail

        run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            }
        )

        # Temp file should still be cleaned up
        assert captured_path is not None
        assert not captured_path.exists()


class TestFullPreflightResultExitCodes:
    """Tests for FullPreflightResult exit codes."""

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_exit_code_pass(self, mock_subprocess, mock_query):
        """PASS status -> exit_code 0."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=16 * 1024**3,
            used_bytes=0,
            free_bytes=16 * 1024**3,
        )
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": 1 * 1024**3,
            "runtime_seconds": 5.0,
        }

        result = run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            }
        )

        assert result.status == PreflightStatus.PASS
        assert result.exit_code == 0

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_exit_code_fail(self, mock_subprocess, mock_query):
        """FAIL status -> exit_code 1."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=8 * 1024**3,
            used_bytes=0,
            free_bytes=8 * 1024**3,
        )
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": 10 * 1024**3,  # Exceeds available
            "runtime_seconds": 5.0,
        }

        result = run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            }
        )

        assert result.status == PreflightStatus.FAIL
        assert result.exit_code == 1

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.full_check._run_mini_mcmc_subprocess")
    def test_exit_code_warning(self, mock_subprocess, mock_query):
        """WARNING status -> exit_code 2."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )
        mock_subprocess.return_value = {
            "success": True,
            "peak_memory_bytes": 9 * 1024**3,  # 9 GB of 10 GB = low headroom
            "runtime_seconds": 5.0,
        }

        result = run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            }
        )

        assert result.status == PreflightStatus.WARNING
        assert result.exit_code == 2

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    def test_exit_code_cannot_check(self, mock_query):
        """CANNOT_CHECK status -> exit_code 2."""
        mock_query.side_effect = GpuMemoryError("No GPU")

        result = run_full_preflight_check(
            {
                "artist_idx": [0],
                "album_seq": [1],
                "prev_score": [0.0],
                "X": [[1.0]],
                "y": [70.0],
                "n_artists": 1,
                "max_seq": 1,
            }
        )

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert result.exit_code == 2


class TestRunExtrapolatedPreflightCheck:
    """Tests for run_extrapolated_preflight_check function."""

    SAMPLE_MODEL_ARGS: ClassVar[dict[str, Any]] = {
        "artist_idx": [0, 1, 0],
        "album_seq": [1, 1, 2],
        "prev_score": [0.0, 0.0, 75.0],
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "y": [70.0, 80.0, 75.0],
        "n_artists": 2,
        "max_seq": 2,
    }

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.cache.save_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_uses_calibration(self, mock_run_cal, _mock_save, mock_load, mock_query):
        """Calibration is used to extrapolate memory."""
        from panelcast.preflight.calibrate import CalibrationResult

        mock_query.return_value = GpuMemoryInfo("Test GPU", 16 * 1024**3, 4 * 1024**3, 12 * 1024**3)
        mock_load.return_value = None  # Cache miss

        # Create a calibration result
        calibration = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.001,
            calibration_points=((10, 2.01), (50, 2.05)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        mock_run_cal.return_value = calibration

        result = run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=2000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
        )

        # Verify extrapolation was used: 2.0 + 0.001 * 2000 = 4.0 GB
        assert result.projected_gb == pytest.approx(4.0, rel=0.01)
        assert result.target_samples == 2000
        mock_run_cal.assert_called_once()

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_uses_cache_when_available(self, mock_run_cal, mock_load, mock_query):
        """Cached calibration is used when available."""
        from panelcast.preflight.calibrate import CalibrationResult

        mock_query.return_value = GpuMemoryInfo("Test GPU", 16 * 1024**3, 4 * 1024**3, 12 * 1024**3)

        # Cache hit
        cached_calibration = CalibrationResult(
            fixed_overhead_gb=1.5,
            per_sample_gb=0.002,
            calibration_points=((10, 1.52), (50, 1.6)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        mock_load.return_value = cached_calibration

        result = run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=1000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
        )

        # run_calibration should NOT have been called
        mock_run_cal.assert_not_called()
        assert result.from_cache is True

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.cache.save_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_recalibrate_bypasses_cache(
        self, mock_run_cal, _mock_save, mock_load, mock_query
    ):
        """recalibrate=True forces fresh calibration even if cache exists."""
        from panelcast.preflight.calibrate import CalibrationResult

        mock_query.return_value = GpuMemoryInfo("Test GPU", 16 * 1024**3, 4 * 1024**3, 12 * 1024**3)

        # Even with cache hit, should NOT be used
        cached_calibration = CalibrationResult(
            fixed_overhead_gb=1.5,
            per_sample_gb=0.002,
            calibration_points=((10, 1.52), (50, 1.6)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        mock_load.return_value = cached_calibration

        fresh_calibration = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.001,
            calibration_points=((10, 2.01), (50, 2.05)),
            config_hash="abc123",
            calibration_time=6.0,
        )
        mock_run_cal.return_value = fresh_calibration

        result = run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=1000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
            recalibrate=True,
        )

        # load_calibration_cache should NOT have been called
        mock_load.assert_not_called()
        # run_calibration SHOULD have been called
        mock_run_cal.assert_called_once()
        assert result.from_cache is False

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.cache.save_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_saves_to_cache(self, mock_run_cal, mock_save, mock_load, mock_query):
        """Fresh calibration is saved to cache."""
        from panelcast.preflight.calibrate import CalibrationResult

        mock_query.return_value = GpuMemoryInfo("Test GPU", 16 * 1024**3, 4 * 1024**3, 12 * 1024**3)
        mock_load.return_value = None  # Cache miss

        calibration = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.001,
            calibration_points=((10, 2.01), (50, 2.05)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        mock_run_cal.return_value = calibration

        run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=1000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
        )

        # save_calibration_cache should have been called with the calibration
        mock_save.assert_called_once_with(calibration)

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.cache.save_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_status_from_projected_not_measured(
        self, mock_run_cal, _mock_save, mock_load, mock_query
    ):
        """Status is based on projected memory, not measured peak."""
        from panelcast.preflight.calibrate import CalibrationResult

        # 8 GB available
        mock_query.return_value = GpuMemoryInfo("Test GPU", 10 * 1024**3, 2 * 1024**3, 8 * 1024**3)
        mock_load.return_value = None

        # Measured peak is only 1 GB, but projection to 2000 samples = 10 GB
        # 1.0 + 0.0045 * 2000 = 10.0 GB
        calibration = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.0045,
            calibration_points=((10, 1.045), (50, 1.225)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        mock_run_cal.return_value = calibration

        result = run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=2000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
        )

        # Measured peak is low (1.225 GB), but projected is high (10 GB)
        assert result.measured_peak_gb == pytest.approx(1.225, rel=0.01)
        assert result.projected_gb == pytest.approx(10.0, rel=0.01)
        # Status should be FAIL because projected (10 GB) > available (8 GB)
        assert result.status == PreflightStatus.FAIL

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.cache.save_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_status_pass_when_projected_fits(
        self, mock_run_cal, _mock_save, mock_load, mock_query
    ):
        """Status is PASS when projected memory fits with headroom."""
        from panelcast.preflight.calibrate import CalibrationResult

        # 12 GB available
        mock_query.return_value = GpuMemoryInfo("Test GPU", 16 * 1024**3, 4 * 1024**3, 12 * 1024**3)
        mock_load.return_value = None

        # Projected to 2000 samples = 5 GB (fits with plenty of headroom)
        # 3.0 + 0.001 * 2000 = 5.0 GB
        calibration = CalibrationResult(
            fixed_overhead_gb=3.0,
            per_sample_gb=0.001,
            calibration_points=((10, 3.01), (50, 3.05)),
            config_hash="abc123",
            calibration_time=5.0,
        )
        mock_run_cal.return_value = calibration

        result = run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=2000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
        )

        assert result.projected_gb == pytest.approx(5.0, rel=0.01)
        # Status should be PASS because 5 GB < 12 * 0.8 = 9.6 GB
        assert result.status == PreflightStatus.PASS

    @mock.patch("panelcast.preflight.full_check.query_gpu_memory")
    @mock.patch("panelcast.preflight.cache.load_calibration_cache")
    @mock.patch("panelcast.preflight.calibrate.run_calibration")
    def test_extrapolation_calibration_error_returns_cannot_check(
        self, mock_run_cal, mock_load, mock_query
    ):
        """CalibrationError results in CANNOT_CHECK status."""
        from panelcast.preflight.calibrate import CalibrationError

        mock_query.return_value = GpuMemoryInfo("Test GPU", 16 * 1024**3, 4 * 1024**3, 12 * 1024**3)
        mock_load.return_value = None  # Cache miss

        mock_run_cal.side_effect = CalibrationError("Model doesn't fit at 10 samples")

        result = run_extrapolated_preflight_check(
            self.SAMPLE_MODEL_ARGS,
            target_samples=2000,
            n_observations=100,
            n_artists=10,
            n_features=2,
            max_seq=10,
        )

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert "Calibration failed" in result.message


class TestExtrapolationResult:
    """Tests for ExtrapolationResult dataclass."""

    def test_exit_code_from_status(self):
        """exit_code property maps status correctly."""
        from panelcast.preflight import ExtrapolationResult
        from panelcast.preflight.calibrate import CalibrationResult

        dummy_calibration = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.001,
            calibration_points=((10, 1.01), (50, 1.05)),
            config_hash="abc",
            calibration_time=5.0,
        )

        pass_result = ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.0,
            projected_gb=2.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=16.0,
            headroom_percent=80.0,
            calibration=dummy_calibration,
            from_cache=False,
            message="Test",
            suggestions=(),
        )
        assert pass_result.exit_code == 0

        fail_result = ExtrapolationResult(
            status=PreflightStatus.FAIL,
            measured_peak_gb=1.0,
            projected_gb=15.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=16.0,
            headroom_percent=-50.0,
            calibration=dummy_calibration,
            from_cache=False,
            message="Test",
            suggestions=(),
        )
        assert fail_result.exit_code == 1

        warning_result = ExtrapolationResult(
            status=PreflightStatus.WARNING,
            measured_peak_gb=1.0,
            projected_gb=9.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=16.0,
            headroom_percent=10.0,
            calibration=dummy_calibration,
            from_cache=False,
            message="Test",
            suggestions=(),
        )
        assert warning_result.exit_code == 2

    def test_from_cache_field(self):
        """from_cache field is stored correctly."""
        from panelcast.preflight import ExtrapolationResult
        from panelcast.preflight.calibrate import CalibrationResult

        dummy_calibration = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.001,
            calibration_points=((10, 1.01), (50, 1.05)),
            config_hash="abc",
            calibration_time=5.0,
        )

        cached_result = ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.0,
            projected_gb=2.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=16.0,
            headroom_percent=80.0,
            calibration=dummy_calibration,
            from_cache=True,
            message="Test",
            suggestions=(),
        )
        assert cached_result.from_cache is True

        fresh_result = ExtrapolationResult(
            status=PreflightStatus.PASS,
            measured_peak_gb=1.0,
            projected_gb=2.0,
            target_samples=1000,
            calibration_samples=60,
            uncertainty_percent=10.0,
            available_gb=10.0,
            total_gpu_gb=16.0,
            headroom_percent=80.0,
            calibration=dummy_calibration,
            from_cache=False,
            message="Test",
            suggestions=(),
        )
        assert fresh_result.from_cache is False


# --- from unit/preflight/test_full_check_expanded.py ---


class TestDeriveDimensionsFromModelArgs:
    """Tests for _derive_dimensions_from_model_args."""

    def test_basic_extraction(self):
        model_args = {
            "y": np.ones(100),
            "X": np.ones((100, 5)),
            "n_artists": 20,
            "max_seq": 8,
        }
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)
        assert n_obs == 100
        assert n_art == 20
        assert n_feat == 5
        assert max_s == 8

    def test_missing_y(self):
        model_args = {"X": np.ones((10, 3)), "n_artists": 5, "max_seq": 3}
        n_obs, _, _, _ = _derive_dimensions_from_model_args(model_args)
        assert n_obs == 0

    def test_missing_x(self):
        model_args = {"y": np.ones(50), "n_artists": 10, "max_seq": 5}
        _, _, n_feat, _ = _derive_dimensions_from_model_args(model_args)
        assert n_feat == 0

    def test_missing_n_artists(self):
        model_args = {"y": np.ones(50), "X": np.ones((50, 3)), "max_seq": 5}
        _, n_art, _, _ = _derive_dimensions_from_model_args(model_args)
        assert n_art == 0

    def test_missing_max_seq(self):
        model_args = {"y": np.ones(50), "X": np.ones((50, 3)), "n_artists": 10}
        _, _, _, max_s = _derive_dimensions_from_model_args(model_args)
        assert max_s == 0

    def test_1d_x_raises(self):
        model_args = {
            "y": np.ones(50),
            "X": np.ones(50),
            "n_artists": 10,
            "max_seq": 5,
        }
        with pytest.raises(ValueError, match="2D array"):
            _derive_dimensions_from_model_args(model_args)

    def test_list_x(self):
        model_args = {
            "y": [1, 2, 3],
            "X": [[1, 2], [3, 4], [5, 6]],
            "n_artists": 2,
            "max_seq": 3,
        }
        n_obs, _, n_feat, _ = _derive_dimensions_from_model_args(model_args)
        assert n_obs == 3
        assert n_feat == 2

    def test_empty_x_list(self):
        model_args = {"y": [], "X": [], "n_artists": 0, "max_seq": 0}
        n_obs, _, n_feat, _ = _derive_dimensions_from_model_args(model_args)
        assert n_obs == 0
        assert n_feat == 0


class TestCalculateHeadroomPercent:
    """Tests for calculate_headroom_percent."""

    def test_full_headroom(self):
        result = calculate_headroom_percent(10.0, 0.0)
        assert result == 100.0

    def test_no_headroom(self):
        result = calculate_headroom_percent(10.0, 10.0)
        assert result == 0.0

    def test_negative_headroom(self):
        result = calculate_headroom_percent(10.0, 15.0)
        assert result == -50.0

    def test_partial_headroom(self):
        result = calculate_headroom_percent(10.0, 8.0)
        assert result == pytest.approx(20.0)

    def test_zero_available(self):
        result = calculate_headroom_percent(0.0, 5.0)
        assert result == -100.0

    def test_zero_both(self):
        result = calculate_headroom_percent(0.0, 0.0)
        assert result == -100.0


class TestSerializeModelArgs_expanded:
    """Tests for serialize_model_args."""

    def test_creates_temp_file(self):
        model_args = {"n_artists": 10, "max_seq": 5}
        path = serialize_model_args(model_args)
        try:
            assert path.exists()
            assert path.suffix == ".json"
        finally:
            path.unlink(missing_ok=True)

    def test_serializes_scalars(self):
        model_args = {"n_artists": 10, "max_seq": 5}
        path = serialize_model_args(model_args)
        try:
            data = json.loads(path.read_text())
            assert data["n_artists"] == 10
            assert data["max_seq"] == 5
        finally:
            path.unlink(missing_ok=True)

    def test_serializes_numpy_arrays(self):
        model_args = {"y": np.array([1.0, 2.0, 3.0])}
        path = serialize_model_args(model_args)
        try:
            data = json.loads(path.read_text())
            assert data["y"] == [1.0, 2.0, 3.0]
        finally:
            path.unlink(missing_ok=True)

    def test_serializes_2d_array(self):
        model_args = {"X": np.ones((3, 2))}
        path = serialize_model_args(model_args)
        try:
            data = json.loads(path.read_text())
            assert len(data["X"]) == 3
            assert len(data["X"][0]) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_returns_path_object(self):
        model_args = {"n": 1}
        path = serialize_model_args(model_args)
        try:
            assert isinstance(path, Path)
        finally:
            path.unlink(missing_ok=True)


class TestGenerateMessageFullCheck:
    """Tests for _generate_message in full_check module."""

    def test_pass_message(self):
        msg = _generate_message(PreflightStatus.PASS, 2.0, 10.0, 80.0)
        assert "passed" in msg
        assert "2.00 GB" in msg

    def test_warning_message(self):
        msg = _generate_message(PreflightStatus.WARNING, 8.0, 10.0, 20.0)
        assert "warning" in msg
        assert "low headroom" in msg

    def test_fail_message(self):
        msg = _generate_message(PreflightStatus.FAIL, 12.0, 10.0, -20.0)
        assert "failed" in msg
        assert "exceeds" in msg

    def test_cannot_check_message(self):
        msg = _generate_message(PreflightStatus.CANNOT_CHECK, 0.0, 0.0, 0.0)
        assert "Cannot" in msg


class TestGenerateSuggestionsFullCheck:
    """Tests for _generate_suggestions in full_check module."""

    def test_pass_no_suggestions(self):
        result = _generate_suggestions(PreflightStatus.PASS, 2.0, 10.0)
        assert result == ()

    def test_fail_has_deficit(self):
        result = _generate_suggestions(PreflightStatus.FAIL, 12.0, 10.0)
        assert any("2.0 GB more" in s for s in result)

    def test_fail_suggests_reduce_chains(self):
        result = _generate_suggestions(PreflightStatus.FAIL, 12.0, 10.0)
        assert any("--num-chains" in s for s in result)

    def test_warning_suggests_close_apps(self):
        result = _generate_suggestions(PreflightStatus.WARNING, 8.5, 10.0)
        assert any("Close" in s for s in result)

    def test_returns_tuple(self):
        result = _generate_suggestions(PreflightStatus.FAIL, 12.0, 10.0)
        assert isinstance(result, tuple)


class TestGenerateExtrapolationMessage:
    """Tests for _generate_extrapolation_message."""

    def test_pass_message(self):
        msg = _generate_extrapolation_message(PreflightStatus.PASS, 3.0, 10.0, 70.0, 2000)
        assert "passed" in msg
        assert "2,000 samples" in msg

    def test_fail_message(self):
        msg = _generate_extrapolation_message(PreflightStatus.FAIL, 15.0, 10.0, -50.0, 4000)
        assert "failed" in msg
        assert "exceeds" in msg

    def test_warning_message(self):
        msg = _generate_extrapolation_message(PreflightStatus.WARNING, 8.0, 10.0, 20.0, 1000)
        assert "warning" in msg
        assert "low headroom" in msg

    def test_cannot_check_message(self):
        msg = _generate_extrapolation_message(PreflightStatus.CANNOT_CHECK, 0.0, 0.0, 0.0, 100)
        assert "Cannot" in msg


class TestGenerateExtrapolationSuggestions:
    """Tests for _generate_extrapolation_suggestions."""

    def test_pass_empty(self):
        result = _generate_extrapolation_suggestions(PreflightStatus.PASS, 3.0, 10.0)
        assert result == ()

    def test_fail_suggests_reduce_samples(self):
        result = _generate_extrapolation_suggestions(PreflightStatus.FAIL, 15.0, 10.0)
        assert any("--num-samples" in s for s in result)

    def test_fail_suggests_reduce_chains(self):
        result = _generate_extrapolation_suggestions(PreflightStatus.FAIL, 15.0, 10.0)
        assert any("--num-chains" in s for s in result)

    def test_warning_suggests_caution(self):
        result = _generate_extrapolation_suggestions(PreflightStatus.WARNING, 8.5, 10.0)
        assert any("OOM" in s for s in result)

    def test_fail_shows_deficit(self):
        result = _generate_extrapolation_suggestions(PreflightStatus.FAIL, 12.0, 10.0)
        assert any("2.0 GB" in s for s in result)


# --- from unit/preflight/test_full_check_new.py ---


class TestRunMiniMcmcSubprocess:
    """Tests for _run_mini_mcmc_subprocess error paths."""

    @mock.patch("subprocess.run")
    def test_nonzero_returncode_returns_failure(self, mock_run):
        """Non-zero return code produces failure result with stderr message."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="CUDA error: out of memory"
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "CUDA error" in result["error"]
            assert result["peak_memory_bytes"] == 0
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_nonzero_returncode_empty_stderr(self, mock_run):
        """Non-zero return code with empty stderr uses default message."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "Unknown subprocess error" in result["error"]
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_timeout_returns_timeout_message(self, mock_run):
        """Subprocess timeout produces timeout error message."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=60)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
            assert result["runtime_seconds"] == 60.0
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_json_decode_error_returns_failure(self, mock_run):
        """Invalid JSON stdout produces parse error result."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json at all {{{", stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is False
            assert "parse" in result["error"].lower()
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_success_parses_json(self, mock_run):
        """Successful subprocess returns parsed JSON output."""
        output = {"success": True, "peak_memory_bytes": 4294967296, "runtime_seconds": 10.5}
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(output), stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            result = _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            assert result["success"] is True
            assert result["peak_memory_bytes"] == 4294967296
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_default_command_has_no_new_flags(self, mock_run):
        """Default kwargs keep the legacy command line byte-for-byte."""
        output = {"success": True, "peak_memory_bytes": 1, "runtime_seconds": 0.1}
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(output), stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            _run_mini_mcmc_subprocess(temp_path, timeout_seconds=60)
            command = mock_run.call_args[0][0]
            assert "--target-transform" not in command
            assert "--chain-method" not in command
            assert "--entity-group-pooling" not in command
        finally:
            temp_path.unlink(missing_ok=True)

    @mock.patch("subprocess.run")
    def test_new_flags_threaded_to_command(self, mock_run):
        """Non-default transform/chain-method/pooling appear on the command line."""
        output = {"success": True, "peak_memory_bytes": 1, "runtime_seconds": 0.1}
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(output), stderr=""
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            temp_path = Path(f.name)
        try:
            _run_mini_mcmc_subprocess(
                temp_path,
                timeout_seconds=60,
                target_transform="offset_logit",
                chain_method="vectorized",
                entity_group_pooling=True,
            )
            command = mock_run.call_args[0][0]
            i = command.index("--target-transform")
            assert command[i + 1] == "offset_logit"
            i = command.index("--chain-method")
            assert command[i + 1] == "vectorized"
            assert "--entity-group-pooling" in command
        finally:
            temp_path.unlink(missing_ok=True)


class TestCreateDummyCalibration:
    """Tests for _create_dummy_calibration."""

    def test_returns_zeroed_calibration(self):
        """Dummy calibration has all zero values."""
        cal = _create_dummy_calibration()
        assert cal.fixed_overhead_gb == 0.0
        assert cal.per_sample_gb == 0.0
        assert cal.calibration_points == ((0, 0.0), (0, 0.0))
        assert cal.config_hash == ""
        assert cal.calibration_time == 0.0

    def test_accepts_config_hash(self):
        """config_hash parameter is passed through."""
        cal = _create_dummy_calibration(config_hash="test_hash")
        assert cal.config_hash == "test_hash"

    def test_extrapolate_returns_zero(self):
        """Dummy calibration extrapolation returns 0.0 for any target."""
        cal = _create_dummy_calibration()
        assert cal.extrapolate(0) == 0.0
        assert cal.extrapolate(1000) == 0.0
        assert cal.extrapolate(100000) == 0.0


class TestGenerateMessageUnknownStatus:
    """Tests for _generate_message with edge-case status values."""

    def test_cannot_check_message(self):
        """CANNOT_CHECK status generates appropriate message."""
        msg = _generate_message(PreflightStatus.CANNOT_CHECK, 0.0, 0.0, 0.0)
        assert "Cannot" in msg

    def test_pass_includes_headroom(self):
        """PASS message includes headroom percentage."""
        msg = _generate_message(PreflightStatus.PASS, 2.0, 10.0, 80.0)
        assert "80%" in msg


class TestGenerateExtrapolationMessageEdgeCases:
    """Tests for _generate_extrapolation_message edge cases."""

    def test_cannot_check_message(self):
        """CANNOT_CHECK returns appropriate message."""
        msg = _generate_extrapolation_message(PreflightStatus.CANNOT_CHECK, 0.0, 0.0, 0.0, 100)
        assert "Cannot" in msg

    def test_pass_includes_sample_count(self):
        """PASS message includes formatted sample count."""
        msg = _generate_extrapolation_message(PreflightStatus.PASS, 3.0, 10.0, 70.0, 2000)
        assert "2,000" in msg

    def test_fail_includes_projected(self):
        """FAIL message includes projected memory and available."""
        msg = _generate_extrapolation_message(PreflightStatus.FAIL, 15.0, 10.0, -50.0, 5000)
        assert "15.0" in msg
        assert "10.0" in msg


class TestGenerateExtrapolationSuggestionsEdgeCases:
    """Tests for _generate_extrapolation_suggestions edge cases."""

    def test_warning_suggests_reducing(self):
        """WARNING suggests reducing samples or chains."""
        result = _generate_extrapolation_suggestions(PreflightStatus.WARNING, 9.0, 10.0)
        assert len(result) > 0
        assert any("--num-samples" in s or "--num-chains" in s or "Close" in s for s in result)

    def test_fail_shows_deficit(self):
        """FAIL shows how much memory is over."""
        result = _generate_extrapolation_suggestions(PreflightStatus.FAIL, 15.0, 10.0)
        assert any("5.0 GB" in s for s in result)
