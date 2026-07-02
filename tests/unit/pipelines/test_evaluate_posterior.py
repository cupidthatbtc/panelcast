"""Tests for posterior predictive evaluation helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import arviz as az
import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.evaluate import (
    _json_safe,
    _prepare_disjoint_inputs,
    _prepare_test_model_args,
    evaluate_models,
)


@pytest.fixture
def mock_summary():
    """Minimal training summary for testing."""
    return {
        "artist_to_idx": {"Artist_A": 0, "Artist_B": 1, "Artist_C": 2},
        "n_artists": 3,
        "max_seq": 5,
        "max_albums": 10,
        "min_albums_filter": 2,
        "global_mean_score": 70.0,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "feature_cols": ["feat_1", "feat_2"],
        },
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "likelihood_df": 4.0,
        "n_ref": None,
    }


@pytest.fixture
def mock_test_data():
    """Create aligned test_df and test_features."""
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B"],
            "User_Score": [75.0, 80.0, 65.0],
            "User_Ratings": [100, 200, 50],
        }
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.5, 2.0, 0.5],
            "feat_2": [3.0, 4.0, 1.0],
            "n_reviews": [100, 200, 50],
        },
        index=test_df.index,
    )
    return test_df, test_features


def test_unknown_artists_raise_error(mock_summary):
    """Primary split should fail fast when unknown artists are present."""
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_D"],
            "User_Score": [75.0, 90.0],
            "User_Ratings": [100, 10],
        }
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.5, 3.0],
            "feat_2": [3.0, 5.0],
            "n_reviews": [100, 10],
        },
        index=test_df.index,
    )
    with pytest.raises(ValueError, match="Unknown artists found in primary split"):
        _prepare_test_model_args(test_df, test_features, mock_summary)


def test_feature_standardization(mock_test_data, mock_summary):
    """Features are standardized using training scaler."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    assert abs(model_args["X"][0, 0] - 1.0) < 1e-5


def test_prev_score_sequential_from_training_history(mock_test_data, mock_summary):
    """Sequential prev_score: first album uses train last, subsequent use preceding actual."""
    test_df, test_features = mock_test_data
    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B"],
            "User_Score": [72.0, 85.0, 60.0],
        }
    )
    model_args, _ = _prepare_test_model_args(
        test_df, test_features, mock_summary, train_df=train_df
    )
    # First test album for Artist_A: prev_score = last training score (85.0)
    assert abs(model_args["prev_score"][0] - 85.0) < 1e-5
    # Second test album for Artist_A: prev_score = first test album's actual (75.0)
    assert abs(model_args["prev_score"][1] - 75.0) < 1e-5
    # Artist_B single album: prev_score = last training score (60.0)
    assert abs(model_args["prev_score"][2] - 60.0) < 1e-5


def test_prev_score_uses_val_when_available(mock_test_data, mock_summary):
    """When val_df is provided, use val scores as prev_score instead of train last."""
    test_df, test_features = mock_test_data
    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B"],
            "User_Score": [72.0, 85.0, 60.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [78.0, 55.0],
        }
    )
    model_args, _ = _prepare_test_model_args(
        test_df, test_features, mock_summary, train_df=train_df, val_df=val_df
    )
    # First test album for Artist_A: prev_score = val score (78.0), not train last (85.0)
    assert abs(model_args["prev_score"][0] - 78.0) < 1e-5
    # Second test album for Artist_A: prev_score = first test album's actual (75.0)
    assert abs(model_args["prev_score"][1] - 75.0) < 1e-5
    # Artist_B: prev_score = val score (55.0), not train last (60.0)
    assert abs(model_args["prev_score"][2] - 55.0) < 1e-5


def test_min_albums_filter_uses_training_history_counts(mock_summary):
    """Dynamic eligibility should be based on train counts, not test fold counts."""
    summary = dict(mock_summary)
    summary["min_albums_filter"] = 2

    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_A", "Artist_B"],
            "User_Score": [70.0, 72.0, 74.0, 65.0],
            "Release_Date_Parsed": pd.to_datetime(
                ["2017-01-01", "2018-01-01", "2019-01-01", "2019-01-01"]
            ),
        }
    )
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [75.0, 66.0],
            "User_Ratings": [100, 80],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "Album": ["A4", "B2"],
        }
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.0, 1.0],
            "feat_2": [2.0, 2.0],
            "n_reviews": [100, 80],
        },
        index=test_df.index,
    )

    model_args, _ = _prepare_test_model_args(test_df, test_features, summary, train_df=train_df)

    np.testing.assert_array_equal(model_args["album_seq"], np.array([4, 1], dtype=np.int32))


def test_prepare_test_model_args_strict_raises_on_horizon_clamp(mock_summary):
    """Strict mode should fail when eval rows exceed training sequence horizon."""
    summary = dict(mock_summary)
    summary["max_seq"] = 2
    summary["min_albums_filter"] = 1

    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B", "Artist_B"],
            "User_Score": [70.0, 72.0, 65.0, 67.0],
            "Release_Date_Parsed": pd.to_datetime(
                ["2018-01-01", "2019-01-01", "2018-01-01", "2019-01-01"]
            ),
        }
    )
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [75.0, 68.0],
            "User_Ratings": [100, 120],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        }
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.0, 1.0],
            "feat_2": [2.0, 2.0],
            "n_reviews": [100, 120],
        },
        index=test_df.index,
    )

    with pytest.raises(ValueError, match="extrapolation beyond training horizon"):
        _prepare_test_model_args(
            test_df,
            test_features,
            summary,
            train_df=train_df,
            strict=True,
        )


def test_prepare_disjoint_inputs_uses_global_mean_prev_score(mock_summary):
    """Cold-start disjoint path must not use held-out labels for prev_score."""
    test_df = pd.DataFrame(
        {
            "Artist": ["New_A", "New_A", "New_B"],
            "User_Score": [90.0, 10.0, 50.0],
            "User_Ratings": [30, 40, 20],
        }
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, 3.0],
            "feat_2": [0.5, 1.5, 2.5],
            "n_reviews": [30, 40, 20],
        },
        index=test_df.index,
    )

    _X, prev_score, _n_reviews, y_true, _g = _prepare_disjoint_inputs(
        test_df, test_features, mock_summary
    )

    assert np.allclose(prev_score, mock_summary["global_mean_score"])
    np.testing.assert_array_equal(y_true, test_df["User_Score"].values.astype(np.float32))


def test_index_alignment_raises_on_mismatch(mock_summary):
    """ValueError raised when test_df and test_features indices differ."""
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [75.0, 80.0],
            "User_Ratings": [100, 200],
        },
        index=[0, 1],
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.5, 2.0],
            "feat_2": [3.0, 4.0],
            "n_reviews": [100, 200],
        },
        index=[10, 11],
    )
    with pytest.raises(ValueError, match="different indices"):
        _prepare_test_model_args(test_df, test_features, mock_summary)


def test_json_safe_replaces_non_finite_with_null():
    """Strict JSON artifacts should convert NaN/inf into null."""
    payload = {
        "nan": float("nan"),
        "pos_inf": np.float64(np.inf),
        "neg_inf": np.float32(-np.inf),
        "nested": [1.0, np.array([2.0, np.nan])],
    }

    safe = _json_safe(payload)

    assert safe["nan"] is None
    assert safe["pos_inf"] is None
    assert safe["neg_inf"] is None
    assert safe["nested"] == [1.0, [2.0, None]]

    serialized = json.dumps(safe, allow_nan=False)
    assert '"nan": null' in serialized


def test_evaluate_models_returns_split_metrics(tmp_path, mock_summary):
    """evaluate_models returns split-aware metrics payload."""
    (tmp_path / "models").mkdir()
    (tmp_path / "data/splits/within_entity_temporal").mkdir(parents=True)
    (tmp_path / "data/features/within_entity_temporal").mkdir(parents=True)

    # Minimal split/features files
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [75.0, 65.0],
            "User_Ratings": [100, 80],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "Album": ["A1", "B1"],
        }
    )
    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [72.0, 60.0],
            "User_Ratings": [90, 70],
            "Release_Date_Parsed": pd.to_datetime(["2018-01-01", "2019-01-01"]),
            "Album": ["A0", "B0"],
        }
    )
    feat_df = pd.DataFrame({"feat_1": [1.0, -0.5], "feat_2": [2.5, 1.5], "n_reviews": [100, 80]})
    train_df.to_parquet(tmp_path / "data/splits/within_entity_temporal/train.parquet")
    test_df.to_parquet(tmp_path / "data/splits/within_entity_temporal/test.parquet")
    feat_df.to_parquet(tmp_path / "data/features/within_entity_temporal/test_features.parquet")

    with open(tmp_path / "models/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(mock_summary, f)

    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))})

    diagnostics = SimpleNamespace(
        passed=True,
        rhat_max=1.0,
        ess_bulk_min=1000,
        divergences=0,
        rhat_threshold=1.01,
        ess_threshold=400,
    )

    ctx = SimpleNamespace(
        seed=42,
        strict=False,
        calibration_intervals=(0.80, 0.95),
        coverage_tolerance=0.03,
        prediction_interval=0.95,
        evaluate_secondary_split=False,
    )

    with (
        patch("panelcast.pipelines.evaluate.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.evaluate.load_model", return_value=fake_idata),
        patch("panelcast.pipelines.evaluate.check_convergence", return_value=diagnostics),
        patch("panelcast.pipelines.evaluate.get_divergence_info"),
        patch(
            "panelcast.pipelines.evaluate._extract_posterior_samples",
            return_value={"user_sigma_obs": np.ones((5,))},
        ),
        patch(
            "panelcast.pipelines.evaluate._run_known_artist_predictive",
            return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
        ),
        patch(
            "panelcast.pipelines.evaluate._compute_info_criteria",
            return_value={
                "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
            },
        ),
        patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
    ):
        result = evaluate_models(ctx)

    assert "metrics" in result
    assert "splits" in result["metrics"]
    assert "within_entity_temporal" in result["metrics"]["splits"]


def test_evaluate_models_strict_fails_when_secondary_artifacts_missing(tmp_path, mock_summary):
    """Strict mode should fail when secondary split evaluation artifacts are missing."""
    (tmp_path / "models").mkdir()
    (tmp_path / "data/splits/within_entity_temporal").mkdir(parents=True)
    (tmp_path / "data/features/within_entity_temporal").mkdir(parents=True)

    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [75.0, 65.0],
            "User_Ratings": [100, 80],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "Album": ["A1", "B1"],
        }
    )
    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [72.0, 60.0],
            "User_Ratings": [90, 70],
            "Release_Date_Parsed": pd.to_datetime(["2018-01-01", "2019-01-01"]),
            "Album": ["A0", "B0"],
        }
    )
    feat_df = pd.DataFrame({"feat_1": [1.0, -0.5], "feat_2": [2.5, 1.5], "n_reviews": [100, 80]})
    train_df.to_parquet(tmp_path / "data/splits/within_entity_temporal/train.parquet")
    test_df.to_parquet(tmp_path / "data/splits/within_entity_temporal/test.parquet")
    feat_df.to_parquet(tmp_path / "data/features/within_entity_temporal/test_features.parquet")

    with open(tmp_path / "models/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(mock_summary, f)

    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))})
    diagnostics = SimpleNamespace(
        passed=True,
        rhat_max=1.0,
        ess_bulk_min=1000,
        divergences=0,
        rhat_threshold=1.01,
        ess_threshold=400,
    )

    ctx = SimpleNamespace(
        seed=42,
        strict=True,
        calibration_intervals=(0.80, 0.95),
        coverage_tolerance=0.03,
        prediction_interval=0.95,
        evaluate_secondary_split=True,
    )

    with (
        patch("panelcast.pipelines.evaluate.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.evaluate.load_model", return_value=fake_idata),
        patch("panelcast.pipelines.evaluate.check_convergence", return_value=diagnostics),
        patch("panelcast.pipelines.evaluate.get_divergence_info"),
        patch(
            "panelcast.pipelines.evaluate._extract_posterior_samples",
            return_value={"user_sigma_obs": np.ones((5,))},
        ),
        patch(
            "panelcast.pipelines.evaluate._run_known_artist_predictive",
            return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
        ),
        patch(
            "panelcast.pipelines.evaluate._compute_info_criteria",
            return_value={
                "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
            },
        ),
        patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
    ):
        with pytest.raises(FileNotFoundError, match="Secondary split evaluation enabled"):
            evaluate_models(ctx)


def test_evaluate_models_strict_fails_on_bad_calibration(mock_summary):
    """Strict mode should fail when calibration exceeds tolerance."""
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))})
    diagnostics = SimpleNamespace(
        passed=True,
        rhat_max=1.0,
        ess_bulk_min=1000,
        divergences=0,
        rhat_threshold=1.01,
        ess_threshold=400,
    )
    ctx = SimpleNamespace(
        seed=42,
        strict=True,
        calibration_intervals=(0.80,),
        coverage_tolerance=0.01,
        prediction_interval=0.95,
        evaluate_secondary_split=False,
    )
    fake_split_metrics = {
        "point_metrics": {
            "rmse": 1.0,
            "mae": 1.0,
            "r2": 0.5,
            "mean_bias": 0.0,
            "n_observations": 2,
        },
        "calibration": {
            "coverages": {"0.80": {"nominal": 0.80, "empirical": 0.60}},
            "coverage_tolerance": 0.01,
            "within_tolerance": False,
        },
        "crps": {"mean_crps": 1.0, "n_obs": 2},
        "prediction_interval": {"level": 0.95, "lower_percentile": 2.5, "upper_percentile": 97.5},
    }

    with (
        patch("panelcast.pipelines.evaluate.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.evaluate.load_model", return_value=fake_idata),
        patch("panelcast.pipelines.evaluate.check_convergence", return_value=diagnostics),
        patch("panelcast.pipelines.evaluate.get_divergence_info"),
        patch("panelcast.pipelines.evaluate.pd.read_parquet", return_value=pd.DataFrame()),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_summary))),
        patch(
            "panelcast.pipelines.evaluate._prepare_test_model_args",
            return_value=(
                {
                    "artist_idx": np.array([0, 1], dtype=np.int32),
                    "album_seq": np.array([1, 1], dtype=np.int32),
                    "prev_score": np.array([70.0, 70.0], dtype=np.float32),
                    "X": np.zeros((2, 2), dtype=np.float32),
                    "y": None,
                    "n_reviews": np.array([50, 60], dtype=np.int32),
                    "n_artists": 2,
                    "max_seq": 5,
                    "n_exponent": 0.0,
                    "learn_n_exponent": False,
                    "n_exponent_prior": "logit-normal",
                    "n_ref": None,
                    "priors": MagicMock(),
                },
                np.array([70.0, 71.0], dtype=np.float32),
            ),
        ),
        patch(
            "panelcast.pipelines.evaluate._run_known_artist_predictive",
            return_value=np.random.default_rng(0).normal(70, 1, size=(5, 2)),
        ),
        patch(
            "panelcast.pipelines.evaluate._evaluate_predictions",
            return_value=(fake_split_metrics, {"x": []}, {"y": []}),
        ),
        patch("panelcast.pipelines.evaluate._compute_info_criteria", return_value={}),
        patch(
            "panelcast.pipelines.evaluate._extract_posterior_samples",
            return_value={"user_sigma_obs": np.ones((5,))},
        ),
    ):
        with pytest.raises(ValueError, match="Calibration coverage outside tolerance"):
            evaluate_models(ctx)


def test_raises_without_manifest():
    """evaluate_models raises ValueError when no manifest found."""
    mock_ctx = MagicMock()
    with patch("panelcast.pipelines.evaluate.load_manifest", return_value=None):
        with pytest.raises(ValueError, match="No trained user_score model"):
            evaluate_models(mock_ctx)


# ============================================================================
# Additional _json_safe Tests
# ============================================================================


class TestJsonSafe:
    """Tests for _json_safe helper function."""

    def test_preserves_int(self):
        """Integers are preserved."""
        assert _json_safe(42) == 42

    def test_preserves_string(self):
        """Strings are preserved."""
        assert _json_safe("hello") == "hello"

    def test_preserves_bool(self):
        """Booleans are preserved."""
        assert _json_safe(True) is True
        assert _json_safe(False) is False

    def test_preserves_none(self):
        """None is preserved."""
        assert _json_safe(None) is None

    def test_converts_nan_to_none(self):
        """NaN is converted to None."""
        assert _json_safe(float("nan")) is None

    def test_converts_inf_to_none(self):
        """Infinity is converted to None."""
        assert _json_safe(float("inf")) is None
        assert _json_safe(float("-inf")) is None

    def test_preserves_finite_float(self):
        """Finite floats are preserved."""
        assert _json_safe(3.14) == 3.14

    def test_converts_numpy_int(self):
        """Numpy integers are converted to Python int."""
        result = _json_safe(np.int64(42))
        assert result == 42

    def test_converts_numpy_float(self):
        """Numpy floats are converted to Python float."""
        result = _json_safe(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_converts_numpy_array_to_list(self):
        """Numpy arrays are converted to lists."""
        result = _json_safe(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_recursive_dict(self):
        """Nested dicts are recursively converted."""
        data = {"a": {"b": float("nan"), "c": 1}}
        result = _json_safe(data)
        assert result == {"a": {"b": None, "c": 1}}

    def test_recursive_list(self):
        """Lists with non-finite values are recursively converted."""
        data = [1.0, float("nan"), [float("inf")]]
        result = _json_safe(data)
        assert result == [1.0, None, [None]]

    def test_tuple_converted_to_list(self):
        """Tuples are converted to lists."""
        result = _json_safe((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set_converted_to_list(self):
        """Sets are converted to lists."""
        result = _json_safe({1})
        assert result == [1]

    def test_nested_numpy_array_with_nan(self):
        """Nested numpy arrays with NaN are handled."""
        data = {"values": np.array([1.0, np.nan, 3.0])}
        result = _json_safe(data)
        assert result == {"values": [1.0, None, 3.0]}


# ============================================================================
# Additional _prepare_disjoint_inputs Tests
# ============================================================================


class TestPrepareDisjointInputs:
    """Tests for _prepare_disjoint_inputs helper."""

    def test_feature_standardization(self, mock_summary):
        """Features are standardized using training scaler."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A"],
                "User_Score": [75.0],
                "User_Ratings": [50],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
                "n_reviews": [50],
            },
            index=test_df.index,
        )
        X, prev_score, n_reviews, y_true, _g = _prepare_disjoint_inputs(
            test_df, test_features, mock_summary
        )
        # X should be standardized: (1.0 - 1.0) / 0.5 = 0.0
        assert abs(X[0, 0] - 0.0) < 1e-5
        # (2.0 - 2.0) / 1.0 = 0.0
        assert abs(X[0, 1] - 0.0) < 1e-5

    def test_returns_correct_n_reviews(self, mock_summary):
        """n_reviews are returned from features."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A", "New_B"],
                "User_Score": [75.0, 80.0],
                "User_Ratings": [50, 100],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0, 1.5],
                "feat_2": [2.0, 2.5],
                "n_reviews": [50, 100],
            },
            index=test_df.index,
        )
        _, _, n_reviews, _, _ = _prepare_disjoint_inputs(test_df, test_features, mock_summary)
        np.testing.assert_array_equal(n_reviews, np.array([50, 100]))


# ============================================================================
# Additional _prepare_test_model_args Tests
# ============================================================================


class TestPrepareTestModelArgs:
    """Additional tests for _prepare_test_model_args."""

    def test_album_seq_clipped_to_max_seq(self, mock_summary):
        """album_seq is clipped to max_seq from training summary."""
        summary = dict(mock_summary)
        summary["max_seq"] = 3
        summary["min_albums_filter"] = 1

        # Artist_A has 5 train albums (exceeds max_seq=3)
        train_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"] * 5 + ["Artist_B"],
                "User_Score": [70.0, 72.0, 74.0, 76.0, 78.0, 65.0],
                "Release_Date_Parsed": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2016-01-01",
                        "2017-01-01",
                        "2018-01-01",
                        "2019-01-01",
                        "2019-01-01",
                    ]
                ),
            }
        )
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A", "Artist_B"],
                "User_Score": [80.0, 68.0],
                "User_Ratings": [100, 80],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0, 0.5],
                "feat_2": [2.0, 1.5],
                "n_reviews": [100, 80],
            },
            index=test_df.index,
        )

        model_args, _ = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, strict=False
        )
        # Album seq should be clipped to max_seq=3
        assert model_args["album_seq"].max() <= summary["max_seq"]
