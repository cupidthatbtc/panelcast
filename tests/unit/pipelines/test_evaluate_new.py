"""Additional coverage tests for evaluate pipeline.

Targets uncovered helper functions and branches:
- _json_safe: numpy types, sets, inf/nan, ndarray, generic
- _write_json: disk write with NaN replacement
- _resolve_feature_split_dir: backward-compatibility fallback
- _prepare_test_model_args: overlap columns, sorting, unknown artists,
  horizon clamping, feature scaler, n_reviews validation
- _prepare_disjoint_inputs: overlap columns, cold-start prev_score,
  n_reviews validation, missing n_reviews column
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.evaluate import (
    _json_safe,
    _prepare_disjoint_inputs,
    _prepare_test_model_args,
    _resolve_feature_split_dir,
    _write_json,
)

# ============================================================================
# Tests: _json_safe
# ============================================================================


class TestJsonSafe:
    """Tests for _json_safe conversion utility."""

    def test_dict_keys_converted_to_str(self):
        """Dict keys should be stringified."""
        result = _json_safe({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_nested_dict(self):
        """Nested dicts should be recursively converted."""
        result = _json_safe({"a": {"b": np.float64(1.5)}})
        assert result == {"a": {"b": 1.5}}

    def test_list_converted(self):
        """Lists should have elements converted."""
        result = _json_safe([np.int64(1), np.float32(2.5)])
        assert result == [1, 2.5]

    def test_tuple_converted_to_list(self):
        """Tuples should be converted to lists."""
        result = _json_safe((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set_converted_to_list(self):
        """Sets should be converted to lists."""
        result = _json_safe({1, 2})
        assert isinstance(result, list)
        assert set(result) == {1, 2}

    def test_numpy_array(self):
        """Numpy arrays should be converted to lists."""
        result = _json_safe(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_scalar(self):
        """Numpy scalars should be converted to Python types."""
        assert _json_safe(np.float64(3.14)) == 3.14
        assert _json_safe(np.int32(42)) == 42

    def test_inf_replaced_with_none(self):
        """Inf values should be replaced with None."""
        assert _json_safe(float("inf")) is None
        assert _json_safe(float("-inf")) is None

    def test_nan_replaced_with_none(self):
        """NaN values should be replaced with None."""
        assert _json_safe(float("nan")) is None

    def test_finite_float_preserved(self):
        """Finite floats should be preserved."""
        assert _json_safe(3.14) == 3.14

    def test_string_passthrough(self):
        """Strings should pass through unchanged."""
        assert _json_safe("hello") == "hello"

    def test_none_passthrough(self):
        """None should pass through unchanged."""
        assert _json_safe(None) is None

    def test_bool_passthrough(self):
        """Booleans should pass through unchanged."""
        assert _json_safe(True) is True
        assert _json_safe(False) is False

    def test_numpy_bool(self):
        """Numpy booleans should be converted."""
        result = _json_safe(np.bool_(True))
        assert result is True or result == True  # noqa: E712


# ============================================================================
# Tests: _write_json
# ============================================================================


class TestWriteJson:
    """Tests for _write_json helper."""

    def test_writes_valid_json(self, tmp_path):
        """Should write valid JSON to disk."""
        path = tmp_path / "test.json"
        _write_json(path, {"key": "value", "num": 42})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"key": "value", "num": 42}

    def test_nan_replaced_with_null(self, tmp_path):
        """NaN values should be written as null."""
        path = tmp_path / "test.json"
        _write_json(path, {"val": float("nan")})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["val"] is None

    def test_indent_parameter(self, tmp_path):
        """Indent parameter should be passed through."""
        path = tmp_path / "test.json"
        _write_json(path, {"a": 1}, indent=2)
        content = path.read_text(encoding="utf-8")
        assert "  " in content  # indented

    def test_numpy_values_converted(self, tmp_path):
        """Numpy values should be converted before writing."""
        path = tmp_path / "test.json"
        _write_json(path, {"arr": np.array([1, 2, 3]), "scalar": np.float64(3.14)})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["arr"] == [1, 2, 3]
        assert data["scalar"] == 3.14


# ============================================================================
# Tests: _resolve_feature_split_dir
# ============================================================================


class TestResolveFeatureSplitDir:
    """Tests for _resolve_feature_split_dir."""

    def test_existing_split_dir_returned(self, tmp_path, monkeypatch):
        """When split dir exists, it should be returned."""
        split_dir = tmp_path / "data" / "features" / "within_entity_temporal"
        split_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.Path",
            lambda p: tmp_path / p,
        )

        result = _resolve_feature_split_dir("within_entity_temporal")
        assert result == split_dir

    def test_primary_split_fallback(self, tmp_path, monkeypatch):
        """Primary split should fall back to data/features when split dir missing."""
        # Don't create the split-specific dir
        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.Path",
            lambda p: tmp_path / p,
        )

        result = _resolve_feature_split_dir("within_entity_temporal")
        assert result == features_dir

    def test_secondary_split_no_fallback(self, tmp_path, monkeypatch):
        """Non-primary split should return candidate path even if missing."""
        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.Path",
            lambda p: tmp_path / p,
        )

        result = _resolve_feature_split_dir("entity_disjoint")
        # Should return the candidate path, not fall back
        assert "entity_disjoint" in str(result)


# ============================================================================
# Tests: _prepare_test_model_args
# ============================================================================


class TestPrepareTestModelArgs:
    """Tests for _prepare_test_model_args helper."""

    def _make_summary(self):
        return {
            "artist_to_idx": {"A": 0, "B": 1},
            "max_seq": 5,
            "max_albums": 50,
            "min_albums_filter": 2,
            "global_mean_score": 75.0,
            "feature_cols": ["f1"],
            "feature_scaler": {
                "mean": [0.0],
                "std": [1.0],
            },
            "n_artists": 2,
            "n_exponent": 0.0,
            "learn_n_exponent": False,
            "n_exponent_prior": "logit-normal",
            "n_ref": None,
            "priors": {
                "mu_artist_scale": 1.0,
                "sigma_artist_scale": 0.5,
                "sigma_rw_scale": 0.1,
                "rho_scale": 0.3,
                "beta_scale": 1.0,
                "sigma_obs_scale": 1.0,
                "n_exponent_alpha": 2.0,
                "n_exponent_beta": 4.0,
            },
        }

    def test_basic_preparation(self):
        """Should produce valid model_args and y_true."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [80.0, 85.0],
                "Album": ["a1", "b1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 82.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )

        assert "artist_idx" in model_args
        assert "X" in model_args
        assert model_args["y"] is None
        assert len(y_true) == 2

    def test_overlap_columns_dropped(self):
        """Overlapping columns should be dropped from test_df."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
                "f1": [999.0],  # overlap
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )
        # f1 should come from features, not test_df
        assert model_args["X"].shape == (1, 1)

    def test_unknown_artist_raises(self):
        """Unknown artists in test data should raise ValueError."""
        test_df = pd.DataFrame(
            {
                "Artist": ["UNKNOWN"],
                "User_Score": [80.0],
                "Album": ["u1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="Unknown artists"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_length_mismatch_raises(self):
        """Length mismatch between test_df and test_features should raise."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="row count mismatch"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_index_mismatch_raises(self):
        """Index mismatch should raise ValueError."""
        test_df = pd.DataFrame(
            {"Artist": ["A"], "User_Score": [80.0], "Album": ["a1"]},
            index=[0],
        )
        test_features = pd.DataFrame(
            {"f1": [1.0], "n_reviews": [10]},
            index=[5],
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="different indices"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_missing_feature_scaler_raises(self):
        """Missing feature_scaler in summary should raise ValueError."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()
        summary.pop("feature_scaler")
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        with pytest.raises(ValueError, match="feature_scaler"):
            _prepare_test_model_args(test_df, test_features, summary, train_df=train_df)

    def test_invalid_n_reviews_filtered(self):
        """Invalid n_reviews rows should be filtered out."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [80.0, 85.0],
                "Album": ["a1", "a2"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, -5],  # one invalid
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )
        assert len(y_true) == 1

    def test_no_train_df(self):
        """Should work when train_df is None."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=None
        )
        assert len(y_true) == 1

    def test_user_ratings_fallback(self):
        """Should fall back to User_Ratings when n_reviews not in features."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
                "User_Ratings": [15],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )
        assert len(y_true) == 1

    def test_no_n_reviews_or_user_ratings_raises(self):
        """Should raise when neither n_reviews nor User_Ratings is available."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        with pytest.raises(ValueError, match="n_reviews or User_Ratings"):
            _prepare_test_model_args(test_df, test_features, summary, train_df=train_df)


# ============================================================================
# Tests: _prepare_disjoint_inputs
# ============================================================================


class TestPrepareDisjointInputs:
    """Tests for _prepare_disjoint_inputs helper."""

    def _make_summary(self):
        return {
            "global_mean_score": 75.0,
            "feature_cols": ["f1"],
            "feature_scaler": {
                "mean": [0.0],
                "std": [1.0],
            },
        }

    def test_basic_preparation(self):
        """Should produce X, prev_score, n_reviews, y_true."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewB"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)

        assert X.shape == (2, 1)
        assert len(y_true) == 2
        # Cold-start: all prev_score should be global mean
        np.testing.assert_allclose(prev_score, [75.0, 75.0])

    def test_overlap_columns_dropped(self):
        """Overlapping columns between test_df and test_features are handled."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA"],
                "User_Score": [80.0],
                "f1": [999.0],  # overlap
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        assert X.shape == (1, 1)

    def test_invalid_n_reviews_filtered(self):
        """Invalid n_reviews should be filtered."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewB"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, -1],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        assert len(y_true) == 1

    def test_length_mismatch_raises(self):
        """Length mismatch should raise ValueError."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewB"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="row count mismatch"):
            _prepare_disjoint_inputs(test_df, test_features, summary)

    def test_index_mismatch_raises(self):
        """Index mismatch should raise ValueError."""
        test_df = pd.DataFrame(
            {"Artist": ["NewA"], "User_Score": [80.0]},
            index=[0],
        )
        test_features = pd.DataFrame(
            {"f1": [1.0], "n_reviews": [10]},
            index=[5],
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="different indices"):
            _prepare_disjoint_inputs(test_df, test_features, summary)

    def test_user_ratings_fallback(self):
        """Should use User_Ratings when n_reviews not present."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA"],
                "User_Score": [80.0],
                "User_Ratings": [15],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        assert len(y_true) == 1
        assert n_reviews[0] == 15

    def test_no_n_reviews_or_user_ratings_raises(self):
        """Should raise when neither column is available."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA"],
                "User_Score": [80.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="n_reviews or User_Ratings"):
            _prepare_disjoint_inputs(test_df, test_features, summary)

    def test_multi_album_artists_use_global_mean(self):
        """Multi-album artists in disjoint split should all use global mean."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewA", "NewB"],
                "User_Score": [80.0, 85.0, 90.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "n_reviews": [10, 20, 30],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        # All prev_score should be global mean = 75.0
        np.testing.assert_allclose(prev_score, [75.0, 75.0, 75.0])
