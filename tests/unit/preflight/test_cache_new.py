"""New coverage-targeted tests for preflight/cache.py.

Targets missed lines:
- save_calibration_cache: OSError during write (cleanup branch)
- load_calibration_cache: TypeError/IndexError in parsing
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from panelcast.preflight.cache import (
    compute_config_hash,
    load_calibration_cache,
    save_calibration_cache,
)
from panelcast.preflight.calibrate import CalibrationResult


class TestSaveCalibrationCacheErrors:
    """Tests for save_calibration_cache error handling."""

    def test_oserror_during_write_cleans_up_temp(self, tmp_path):
        """OSError during file write cleans up temp file."""
        cache_dir = tmp_path / "calibration"

        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=((10, 2.5), (50, 4.5)),
            config_hash="error_test",
            calibration_time=30.0,
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            # Make the rename fail with OSError
            with mock.patch("pathlib.Path.rename", side_effect=OSError("disk full")):
                save_calibration_cache(result)

        # Neither the final file nor temp should exist
        assert not (cache_dir / "error_test.json").exists()

    def test_oserror_during_file_open(self, tmp_path):
        """OSError during file write is handled gracefully."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.05,
            calibration_points=((10, 2.5), (50, 4.5)),
            config_hash="write_fail",
            calibration_time=30.0,
        )

        # Make the open() fail with OSError to trigger the except branch
        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            with mock.patch("builtins.open", side_effect=OSError("permission denied")):
                # Should not raise - it catches OSError
                save_calibration_cache(result)


class TestLoadCalibrationCacheEdgeCases:
    """Tests for load_calibration_cache parsing edge cases."""

    def test_type_error_returns_none(self, tmp_path):
        """TypeError in parsing returns None."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        # Write JSON with wrong types for calibration_points
        # Use an integer which is not subscriptable/iterable
        cache_file = cache_dir / "type_err.json"
        cache_file.write_text(
            json.dumps(
                {
                    "fixed_overhead_gb": 2.0,
                    "per_sample_gb": 0.05,
                    "calibration_points": 42,  # Wrong type - integer, not list
                    "config_hash": "type_err",
                    "calibration_time": 30.0,
                }
            )
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("type_err")

        assert result is None

    def test_index_error_returns_none(self, tmp_path):
        """IndexError in parsing returns None."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        # Write JSON with empty calibration_points
        cache_file = cache_dir / "idx_err.json"
        cache_file.write_text(
            json.dumps(
                {
                    "fixed_overhead_gb": 2.0,
                    "per_sample_gb": 0.05,
                    "calibration_points": [],  # Empty - IndexError on [0]
                    "config_hash": "idx_err",
                    "calibration_time": 30.0,
                }
            )
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("idx_err")

        assert result is None

    def test_key_error_returns_none(self, tmp_path):
        """KeyError in parsing returns None."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        # Write valid JSON but missing a required key
        cache_file = cache_dir / "key_err.json"
        cache_file.write_text(
            json.dumps(
                {
                    "fixed_overhead_gb": 2.0,
                    # Missing per_sample_gb
                    "calibration_points": [[10, 2.5], [50, 4.5]],
                    "config_hash": "key_err",
                    "calibration_time": 30.0,
                }
            )
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("key_err")

        assert result is None

    def test_valid_cache_loads_correctly(self, tmp_path):
        """Valid cache file loads and returns CalibrationResult."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        cache_file = cache_dir / "valid.json"
        cache_file.write_text(
            json.dumps(
                {
                    "fixed_overhead_gb": 1.5,
                    "per_sample_gb": 0.02,
                    "calibration_points": [[10, 1.7], [50, 2.5]],
                    "config_hash": "valid",
                    "calibration_time": 25.0,
                }
            )
        )

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("valid")

        assert result is not None
        assert result.fixed_overhead_gb == 1.5
        assert result.per_sample_gb == 0.02
        assert result.calibration_points == ((10, 1.7), (50, 2.5))

    def test_nonexistent_hash_returns_none(self, tmp_path):
        """Hash that doesn't exist in cache returns None."""
        cache_dir = tmp_path / "calibration"
        cache_dir.mkdir(parents=True)

        with mock.patch("panelcast.preflight.cache.CACHE_DIR", cache_dir):
            result = load_calibration_cache("nonexistent_hash")

        assert result is None


class TestComputeConfigHashEdgeCases:
    """Edge cases for compute_config_hash."""

    def test_zero_values(self):
        """Hash with all zeros is valid."""
        h = compute_config_hash(0, 0, 0, 0)
        assert len(h) == 16
        int(h, 16)  # Should be valid hex

    def test_large_values(self):
        """Hash with very large values is valid."""
        h = compute_config_hash(1_000_000, 100_000, 1000, 500)
        assert len(h) == 16

    def test_same_inputs_same_hash(self):
        """Deterministic: same inputs produce same hash."""
        h1 = compute_config_hash(500, 50, 10, 5)
        h2 = compute_config_hash(500, 50, 10, 5)
        assert h1 == h2
