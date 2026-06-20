"""Tests for calibration and caching modules."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

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
