"""Tests for calibration and caching modules."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from panelcast.preflight.cache import (
    CACHE_DIR,
    compute_config_hash,
    load_calibration_cache,
    save_calibration_cache,
)
from panelcast.preflight.calibrate import (
    CALIBRATION_SAMPLES,
    CalibrationError,
    CalibrationResult,
    calculate_calibration,
    run_calibration,
)


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_extrapolate_calculates_linear_projection(self):
        """extrapolate() returns fixed + per_sample * target."""
        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=((10, 2.5), (50, 4.5)),
            config_hash="abc123",
            calibration_time=30.0,
        )

        # 100 samples: 2.0 + 0.05 * 100 = 7.0
        projected = result.extrapolate(100)
        assert projected == pytest.approx(7.0)

        # 1000 samples: 2.0 + 0.05 * 1000 = 52.0
        projected = result.extrapolate(1000)
        assert projected == pytest.approx(52.0)

        # 0 samples: should return just the fixed overhead
        projected = result.extrapolate(0)
        assert projected == pytest.approx(2.0)

    def test_calibration_result_is_frozen(self):
        """CalibrationResult is immutable (frozen dataclass)."""
        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=((10, 2.5), (50, 4.5)),
            config_hash="abc123",
            calibration_time=30.0,
        )

        with pytest.raises(FrozenInstanceError):
            result.fixed_overhead_gb = 3.0

        with pytest.raises(FrozenInstanceError):
            result.per_sample_gb = 0.1

    def test_calibration_points_stored(self):
        """calibration_points tuple matches input."""
        points = ((10, 2.5), (50, 4.5))
        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=points,
            config_hash="abc123",
            calibration_time=30.0,
        )

        assert result.calibration_points == points
        assert result.calibration_points[0] == (10, 2.5)
        assert result.calibration_points[1] == (50, 4.5)


class TestCalculateCalibration:
    """Tests for calculate_calibration function."""

    def test_calculate_calibration_slope(self):
        """Slope is calculated correctly from two points."""
        # (20, 2.0) and (60, 4.0)
        # slope = (4.0 - 2.0) / (60 - 20) = 2.0 / 40 = 0.05
        _fixed, per_sample = calculate_calibration((20, 2.0), (60, 4.0))

        assert per_sample == pytest.approx(0.05)

    def test_calculate_calibration_intercept(self):
        """Intercept is calculated correctly from two points."""
        # (20, 2.0) and (60, 4.0)
        # slope = 0.05
        # intercept = 2.0 - 0.05 * 20 = 2.0 - 1.0 = 1.0
        fixed, _per_sample = calculate_calibration((20, 2.0), (60, 4.0))

        assert fixed == pytest.approx(1.0)

    def test_calculate_calibration_negative_intercept_raises(self):
        """CalibrationError raised when intercept is negative."""
        # Points where line crosses y=0 before x=0
        # (10, 1.0) and (20, 3.0)
        # slope = (3.0 - 1.0) / (20 - 10) = 0.2
        # intercept = 1.0 - 0.2 * 10 = -1.0
        with pytest.raises(CalibrationError) as excinfo:
            calculate_calibration((10, 1.0), (20, 3.0))

        assert "negative fixed overhead" in str(excinfo.value).lower()

    def test_calculate_calibration_with_actual_sample_counts(self):
        """Works with actual calibration sample counts (10, 50)."""
        # Realistic values: ~2GB fixed overhead, ~0.02 GB/sample
        # At 10 samples: 2.0 + 0.02 * 10 = 2.2 GB
        # At 50 samples: 2.0 + 0.02 * 50 = 3.0 GB
        fixed, per_sample = calculate_calibration((10, 2.2), (50, 3.0))

        assert fixed == pytest.approx(2.0)
        assert per_sample == pytest.approx(0.02)

    def test_calculate_calibration_zero_slope(self):
        """Handles case where memory doesn't scale with samples."""
        # Same memory at both points -> slope = 0
        fixed, per_sample = calculate_calibration((10, 2.5), (50, 2.5))

        assert fixed == pytest.approx(2.5)
        assert per_sample == pytest.approx(0.0)


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_hash_is_16_chars(self):
        """Hash is exactly 16 characters."""
        h = compute_config_hash(1000, 100, 20, 10)

        assert len(h) == 16

    def test_hash_is_hex(self):
        """Hash contains only hex characters."""
        h = compute_config_hash(1000, 100, 20, 10)

        # All characters should be valid hex
        int(h, 16)  # This will raise ValueError if not valid hex

    def test_hash_deterministic(self):
        """Same inputs produce same hash."""
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 100, 20, 10)

        assert h1 == h2

    def test_hash_differs_on_n_observations_change(self):
        """Different n_observations gives different hash."""
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(2000, 100, 20, 10)

        assert h1 != h2

    def test_hash_differs_on_n_artists_change(self):
        """Different n_artists gives different hash."""
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 200, 20, 10)

        assert h1 != h2

    def test_hash_differs_on_n_features_change(self):
        """Different n_features gives different hash."""
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 100, 40, 10)

        assert h1 != h2

    def test_hash_differs_on_max_seq_change(self):
        """Different max_seq gives different hash."""
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 100, 20, 15)

        assert h1 != h2


class TestCalibrationCache:
    """Tests for calibration cache operations."""

    def test_load_missing_file_returns_none(self, tmp_path):
        """Non-existent cache file returns None."""
        with mock.patch("panelcast.preflight.cache.CACHE_DIR", tmp_path / "calibration"):
            result = load_calibration_cache("nonexistent_hash")

        assert result is None

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save CalibrationResult, load it back, verify equality."""
        cache_dir = tmp_path / "calibration"

        result = CalibrationResult(
            fixed_overhead_gb=2.5,
            per_sample_gb=0.04,
            calibration_points=((10, 2.9), (50, 4.5)),
            config_hash="test_hash_1234",
            calibration_time=25.5,
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            save_calibration_cache(result)
            loaded = load_calibration_cache("test_hash_1234")

        assert loaded is not None
        assert loaded.fixed_overhead_gb == pytest.approx(2.5)
        assert loaded.per_sample_gb == pytest.approx(0.04)
        assert loaded.calibration_points == ((10, 2.9), (50, 4.5))
        assert loaded.config_hash == "test_hash_1234"
        assert loaded.calibration_time == pytest.approx(25.5)

    def test_atomic_write_creates_final_file(self, tmp_path):
        """Atomic write creates final .json file, not .tmp."""
        cache_dir = tmp_path / "calibration"

        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=((10, 2.5), (50, 4.5)),
            config_hash="atomic_test",
            calibration_time=30.0,
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            save_calibration_cache(result)

        # Final file should exist
        assert (cache_dir / "atomic_test.json").exists()

        # Temp file should not exist
        assert not (cache_dir / "atomic_test.tmp").exists()

    def test_load_invalid_json_returns_none(self, tmp_path):
        """Invalid JSON file returns None."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        # Write invalid JSON
        cache_file = cache_dir / "invalid.json"
        cache_file.write_text("not valid json {{{")

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("invalid")

        assert result is None

    def test_load_missing_keys_returns_none(self, tmp_path):
        """Cache file missing required keys returns None."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        # Write JSON missing required keys
        cache_file = cache_dir / "missing_keys.json"
        cache_file.write_text(json.dumps({"fixed_overhead_gb": 2.0}))

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("missing_keys")

        assert result is None

    def test_cache_dir_created_if_missing(self, tmp_path):
        """Cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache" / "calibration"
        assert not cache_dir.exists()

        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=((10, 2.5), (50, 4.5)),
            config_hash="new_cache_test",
            calibration_time=30.0,
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            save_calibration_cache(result)

        assert cache_dir.exists()
        assert (cache_dir / "new_cache_test.json").exists()


class TestCalibrationSamplesConstant:
    """Tests for CALIBRATION_SAMPLES constant."""

    def test_calibration_samples_is_tuple(self):
        """CALIBRATION_SAMPLES is a tuple of two ints."""
        assert isinstance(CALIBRATION_SAMPLES, tuple)
        assert len(CALIBRATION_SAMPLES) == 2
        assert all(isinstance(x, int) for x in CALIBRATION_SAMPLES)

    def test_calibration_samples_values(self):
        """CALIBRATION_SAMPLES contains expected values (10, 50)."""
        assert CALIBRATION_SAMPLES == (10, 50)


class TestCacheDirConstant:
    """Tests for CACHE_DIR constant."""

    def test_cache_dir_is_path(self):
        """CACHE_DIR is a Path object."""
        assert isinstance(CACHE_DIR, Path)

    def test_cache_dir_ends_with_calibration(self):
        """CACHE_DIR ends with calibration directory."""
        assert CACHE_DIR.name == "calibration"
        assert CACHE_DIR.parent.name == "panelcast"


# --- from unit/preflight/test_calibrate_coverage.py ---


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


# --- from unit/preflight/test_calibrate_expanded.py ---


class TestCalibrationSamplesConstant_expanded:
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


class TestCalibrationResult_expanded:
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


class TestCalculateCalibration_expanded:
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


# --- from unit/preflight/test_calibrate_new.py ---


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
