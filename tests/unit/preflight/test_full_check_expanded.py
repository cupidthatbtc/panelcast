"""Expanded tests for preflight/full_check.py helper functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from panelcast.preflight import PreflightStatus
from panelcast.preflight.full_check import (
    _derive_dimensions_from_model_args,
    _generate_extrapolation_message,
    _generate_extrapolation_suggestions,
    _generate_message,
    _generate_suggestions,
    calculate_headroom_percent,
    serialize_model_args,
)


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


class TestSerializeModelArgs:
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
