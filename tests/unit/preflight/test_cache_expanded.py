"""Expanded unit tests for preflight calibration cache."""

import json

import pytest

from panelcast.preflight.cache import (
    CACHE_DIR,
    compute_config_hash,
    load_calibration_cache,
    save_calibration_cache,
)
from panelcast.preflight.calibrate import CalibrationResult


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_returns_16_char_hex(self):
        h = compute_config_hash(1000, 100, 20, 10)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 100, 20, 10)
        assert h1 == h2

    def test_different_n_observations(self):
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(2000, 100, 20, 10)
        assert h1 != h2

    def test_different_n_artists(self):
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 200, 20, 10)
        assert h1 != h2

    def test_different_n_features(self):
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 100, 30, 10)
        assert h1 != h2

    def test_different_max_seq(self):
        h1 = compute_config_hash(1000, 100, 20, 10)
        h2 = compute_config_hash(1000, 100, 20, 15)
        assert h1 != h2

    def test_small_values(self):
        h = compute_config_hash(1, 1, 1, 1)
        assert len(h) == 16

    def test_large_values(self):
        h = compute_config_hash(100000, 10000, 100, 50)
        assert len(h) == 16

    def test_zero_values(self):
        h = compute_config_hash(0, 0, 0, 0)
        assert len(h) == 16


class TestCacheDir:
    """Tests for CACHE_DIR constant."""

    def test_is_path(self):
        from pathlib import Path

        assert isinstance(CACHE_DIR, Path)

    def test_ends_with_calibration(self):
        assert CACHE_DIR.name == "calibration"

    def test_contains_panelcast(self):
        assert "panelcast" in str(CACHE_DIR)


class TestLoadCalibrationCache:
    """Tests for load_calibration_cache function."""

    def test_returns_none_for_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        result = load_calibration_cache("nonexistent_hash")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        cache_file = tmp_path / "badhash.json"
        cache_file.write_text("{ invalid json }")
        result = load_calibration_cache("badhash")
        assert result is None

    def test_returns_none_for_missing_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        cache_file = tmp_path / "incomplete.json"
        cache_file.write_text(json.dumps({"fixed_overhead_gb": 1.0}))
        result = load_calibration_cache("incomplete")
        assert result is None

    def test_loads_valid_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        data = {
            "fixed_overhead_gb": 1.0,
            "per_sample_gb": 0.005,
            "calibration_points": [[10, 1.05], [50, 1.25]],
            "config_hash": "testhash12345678",
            "calibration_time": 30.0,
        }
        cache_file = tmp_path / "testhash12345678.json"
        cache_file.write_text(json.dumps(data))
        result = load_calibration_cache("testhash12345678")
        assert result is not None
        assert result.fixed_overhead_gb == 1.0
        assert result.per_sample_gb == 0.005


class TestSaveCalibrationCache:
    """Tests for save_calibration_cache function."""

    def test_saves_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        result = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.005,
            calibration_points=((10, 1.05), (50, 1.25)),
            config_hash="savehash12345678",
            calibration_time=30.0,
        )
        save_calibration_cache(result)
        cache_file = tmp_path / "savehash12345678.json"
        assert cache_file.exists()

    def test_saved_file_is_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        result = CalibrationResult(
            fixed_overhead_gb=2.0,
            per_sample_gb=0.01,
            calibration_points=((10, 2.1), (50, 2.5)),
            config_hash="jsonhash12345678",
            calibration_time=15.0,
        )
        save_calibration_cache(result)
        cache_file = tmp_path / "jsonhash12345678.json"
        data = json.loads(cache_file.read_text())
        assert data["fixed_overhead_gb"] == 2.0
        assert data["per_sample_gb"] == 0.01

    def test_creates_cache_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "nested" / "dir"
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", nested)
        result = CalibrationResult(
            fixed_overhead_gb=1.0,
            per_sample_gb=0.005,
            calibration_points=((10, 1.05), (50, 1.25)),
            config_hash="nestedhash123456",
            calibration_time=30.0,
        )
        save_calibration_cache(result)
        assert nested.exists()

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("panelcast.preflight.cache.CACHE_DIR", tmp_path)
        original = CalibrationResult(
            fixed_overhead_gb=1.5,
            per_sample_gb=0.008,
            calibration_points=((10, 1.58), (50, 1.9)),
            config_hash="roundtrip1234567",
            calibration_time=25.0,
        )
        save_calibration_cache(original)
        loaded = load_calibration_cache("roundtrip1234567")
        assert loaded is not None
        assert loaded.fixed_overhead_gb == pytest.approx(1.5)
        assert loaded.per_sample_gb == pytest.approx(0.008)
        assert loaded.config_hash == "roundtrip1234567"
