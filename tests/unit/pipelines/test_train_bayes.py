"""Unit tests for train_bayes pipeline data preparation functions.

Tests cover:
- load_training_data: DataFrame alignment and overlap handling
- prepare_model_data: Artist indexing, album sequences, n_reviews validation
- _apply_max_albums_cap: Most-recent album capping logic

These tests do NOT run MCMC - they focus on pure data preparation logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.errors import ConvergenceError
from panelcast.pipelines.train_bayes import (
    _apply_max_albums_cap,
    _validate_strict_sampling_config,
    load_training_data,
    prepare_model_data,
    train_models,
)


@pytest.fixture
def sample_features_df():
    """Create a sample features DataFrame."""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_reviews": [10, 20, 30, 40, 50],
        },
        index=pd.RangeIndex(5),
    )


@pytest.fixture
def sample_splits_df():
    """Create a sample splits DataFrame with required columns."""
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "B", "B", "B"],
            "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
            "Album": ["a1", "a2", "b1", "b2", "b3"],
        },
        index=pd.RangeIndex(5),
    )


@pytest.fixture
def sample_train_df():
    """Create a sample training DataFrame for prepare_model_data tests."""
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "B", "B", "C"],
            "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0, 65.0],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "n_reviews": [10, 20, 30, 40, 50, 60],
        }
    )


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_length_mismatch_raises_value_error(self, tmp_path, sample_splits_df):
        """Should raise ValueError when DataFrames have different lengths."""
        # Create features with different length
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],  # Only 3 rows vs 5 in splits
                "n_reviews": [10, 20, 30],
            }
        )

        # Save to parquet
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        with pytest.raises(ValueError, match="row count mismatch"):
            load_training_data(features_path, splits_path)

    def test_index_mismatch_raises_value_error(self, tmp_path, sample_splits_df):
        """Should raise ValueError when DataFrames have mismatched indices."""
        # Create features with different indices
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            },
            index=[10, 11, 12, 13, 14],  # Different from splits 0-4
        )

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        with pytest.raises(ValueError, match="different indices"):
            load_training_data(features_path, splits_path)

    def test_overlap_columns_dropped(self, tmp_path, sample_features_df):
        """Should drop overlapping columns from splits before join."""
        # Create splits with a column that overlaps features
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "feature_1": [999.0, 999.0, 999.0, 999.0, 999.0],  # Overlap
                "n_reviews": [100, 200, 300, 400, 500],  # Overlap - different values
            },
            index=pd.RangeIndex(5),
        )

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        sample_features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        model_args, feature_cols, train_df = load_training_data(features_path, splits_path)

        # Feature values should be from features_df, not splits_df
        assert train_df["feature_1"].iloc[0] == 1.0, "feature_1 should come from features"
        assert train_df["n_reviews"].iloc[0] == 10, "n_reviews should come from features"

    def test_n_reviews_excluded_from_predictor_features(
        self, tmp_path, sample_features_df, sample_splits_df
    ):
        """n_reviews is retained for noise scaling but excluded from predictor matrix."""
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        sample_features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        _model_args, feature_cols, train_df = load_training_data(features_path, splits_path)

        assert "n_reviews" not in feature_cols
        assert "n_reviews" in train_df.columns

    def test_only_n_reviews_feature_raises_value_error(self, tmp_path, sample_splits_df):
        """At least one non-n_reviews predictor column is required."""
        features_df = pd.DataFrame({"n_reviews": [10, 20, 30, 40, 50]}, index=pd.RangeIndex(5))

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        with pytest.raises(ValueError, match="No predictor features available"):
            load_training_data(features_path, splits_path)

    def test_nan_features_filled(self, tmp_path, sample_splits_df):
        """Should fill NaN values in feature columns with 0."""
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, np.nan, 3.0, np.nan, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            },
            index=pd.RangeIndex(5),
        )

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        model_args, feature_cols, train_df = load_training_data(features_path, splits_path)

        # NaN values should be filled with 0
        assert not train_df["feature_1"].isna().any(), "NaN values should be filled"
        assert train_df["feature_1"].iloc[1] == 0.0
        assert train_df["feature_1"].iloc[3] == 0.0


class TestPrepareModelData:
    """Tests for prepare_model_data function."""

    def test_artist_index_mapping(self, sample_train_df):
        """Should create sequential artist indices."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        artist_idx = model_args["artist_idx"]

        # Should have unique indices for each artist
        unique_indices = set(artist_idx)
        assert len(unique_indices) == 3, "Should have 3 unique artist indices"
        # Indices should be 0, 1, 2
        assert unique_indices == {0, 1, 2}

    def test_album_seq_computation(self, sample_train_df):
        """Should compute 1-indexed album sequence within artist."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        album_seq = model_args["album_seq"]

        # Artist A has 3 albums -> seq 1, 2, 3
        # Artist B has 2 albums -> seq 1, 2
        # Artist C has 1 album -> seq 1
        # Order: A, A, A, B, B, C
        expected = np.array([1, 2, 3, 1, 2, 1])
        np.testing.assert_array_equal(album_seq, expected)

    def test_min_albums_filter_clamps_seq(self):
        """Artists below min_albums_filter should get album_seq=1."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "C"],  # A:3, B:1, C:1
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=2)

        album_seq = model_args["album_seq"]

        # A has 3 albums >= 2 -> seq 1, 2, 3
        # B has 1 album < 2 -> seq 1 (clamped)
        # C has 1 album < 2 -> seq 1 (clamped)
        expected = np.array([1, 2, 3, 1, 1])
        np.testing.assert_array_equal(album_seq, expected)

    def test_entity_group_pooling_builds_group_args(self):
        """Gate-on: modal per-artist groups with __rest__ bucketing land in model_args."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "C", "D"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                # A and B share Rock (2 entities -> kept); Jazz has one entity
                # and D has no genre -> both bucket to __rest__.
                "primary_genre": ["Rock", "Rock", "Rock", "Jazz", None],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            }
        )
        model_args, _ = prepare_model_data(
            df, ["feature_1"], min_albums_filter=1, entity_group_pooling=True
        )

        assert model_args["group_to_idx"] == {"__rest__": 0, "Rock": 1}
        assert model_args["n_groups"] == 2
        idx = model_args["group_idx_by_artist"]
        assert idx.dtype == np.int32
        np.testing.assert_array_equal(idx, np.array([1, 1, 0, 0], dtype=np.int32))

    def test_entity_group_pooling_missing_column_raises(self):
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "feature_1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        with pytest.raises(ValueError, match="missing"):
            prepare_model_data(df, ["feature_1"], entity_group_pooling=True)

    def test_entity_group_pooling_no_descriptor_column_raises(self):
        from panelcast.config.descriptor import DatasetDescriptor

        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "feature_1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        descriptor = DatasetDescriptor(entity_group_col=None)
        with pytest.raises(ValueError, match="entity_group_col"):
            prepare_model_data(
                df, ["feature_1"], descriptor=descriptor, entity_group_pooling=True
            )

    def test_prev_score_computation(self, sample_train_df):
        """Should compute shifted prev_score within artist."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        prev_score = model_args["prev_score"]
        y = model_args["y"]

        # Global mean used for debut albums (may come from dataset_stats.json
        # or training fallback — test is agnostic to the source).
        global_mean = model_args["global_mean_score"]

        # First album of each artist should have global mean
        # A's first (idx 0): global_mean
        # A's second (idx 1): A's first score = 70.0
        # A's third (idx 2): A's second score = 75.0
        # B's first (idx 3): global_mean
        # B's second (idx 4): B's first score = 85.0
        # C's first (idx 5): global_mean
        assert np.isclose(prev_score[0], global_mean)
        assert np.isclose(prev_score[1], 70.0)
        assert np.isclose(prev_score[2], 75.0)
        assert np.isclose(prev_score[3], global_mean)
        assert np.isclose(prev_score[4], 85.0)
        assert np.isclose(prev_score[5], global_mean)

    def test_n_reviews_validation_filters_invalid(self):
        """Should filter rows with NaN or <= 0 n_reviews."""
        # Keep less than 50% invalid to avoid the "too many invalid" error
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "C"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "n_reviews": [10, np.nan, 30, 40, 0, 60],  # 2 invalid out of 6 (33%)
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # Should keep only valid rows (indices 0, 2, 3, 5)
        assert len(model_args["y"]) == 4
        assert valid_mask.sum() == 4
        # Verify kept rows
        np.testing.assert_array_equal(valid_mask, [True, False, True, True, False, True])

    def test_n_reviews_too_many_invalid_raises(self):
        """Should raise ValueError when >50% n_reviews are invalid."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [np.nan, np.nan, np.nan, 40],  # 75% invalid
            }
        )

        with pytest.raises(ValueError, match="Too many invalid n_reviews"):
            prepare_model_data(df, ["feature_1"], min_albums_filter=1)

    def test_model_args_has_required_keys(self, sample_train_df):
        """Should include all required keys in model_args."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        required_keys = [
            "artist_idx",
            "album_seq",
            "prev_score",
            "X",
            "y",
            "n_reviews",
            "n_artists",
            "artist_album_counts",
        ]

        for key in required_keys:
            assert key in model_args, f"Missing key: {key}"

    def test_feature_matrix_shape(self, sample_train_df):
        """Feature matrix should have correct shape."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        X = model_args["X"]
        assert X.shape == (6, 2), "X should be (n_obs, n_features)"
        assert X.dtype == np.float32

    def test_uses_user_ratings_fallback(self):
        """Should use User_Ratings if n_reviews not in features."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B"],
                "User_Score": [70.0, 75.0, 80.0],
                "User_Ratings": [10, 20, 30],  # Fallback column
                "feature_1": [1.0, 2.0, 3.0],
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # Should use User_Ratings as n_reviews
        np.testing.assert_array_equal(model_args["n_reviews"], [10, 20, 30])

    def test_missing_n_reviews_and_user_ratings_raises(self):
        """Should raise ValueError if neither n_reviews nor User_Ratings present."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
                "feature_1": [1.0, 2.0],
            }
        )

        with pytest.raises(ValueError, match="n_reviews column not found"):
            prepare_model_data(df, ["feature_1"], min_albums_filter=1)

    def test_effective_ceiling_from_train_max(self, sample_train_df):
        """identity: ceiling = train max + 0.5 margin."""
        model_args, _ = prepare_model_data(
            sample_train_df, ["feature_1"], min_albums_filter=1
        )
        assert model_args["effective_ceiling"] == 90.5

    def test_effective_ceiling_clamped_to_bounds(self):
        """A train max near the theoretical bound clamps to the bound."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B"],
                "User_Score": [70.0, 99.8, 80.0],
                "feature_1": [1.0, 2.0, 3.0],
                "n_reviews": [10, 20, 30],
            }
        )
        model_args, _ = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert model_args["effective_ceiling"] == 100.0

    def test_effective_ceiling_none_for_non_identity(self, sample_train_df):
        """The ceiling is score-scale only; transformed targets get None."""
        model_args, _ = prepare_model_data(
            sample_train_df,
            ["feature_1"],
            min_albums_filter=1,
            target_transform="offset_logit",
        )
        assert model_args["effective_ceiling"] is None


class TestApplyMaxAlbumsCap:
    """Tests for _apply_max_albums_cap function."""

    def test_capping_keeps_most_recent(self):
        """Should keep most recent albums when capping."""
        # Artist 0 has 5 albums (seq 1-5), cap at 3 -> keep seq 3, 4, 5 (renumbered to 1, 2, 3)
        model_args = {
            "album_seq": np.array([1, 2, 3, 4, 5, 1, 2]),  # Artist 0: 5, Artist 1: 2
            "artist_idx": np.array([0, 0, 0, 0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([5, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=3, artist_album_counts=artist_album_counts
        )

        # Artist 0: offset = 5-3 = 2
        # new_seq = max(1, original - 2) = [1-2=1, 2-2=1, 3-2=1, 4-2=2, 5-2=3]
        # Artist 1: offset = 2-3 = 0 (no change)
        expected_seq = np.array([1, 1, 1, 2, 3, 1, 2])
        np.testing.assert_array_equal(result["album_seq"], expected_seq)

    def test_max_seq_derived_correctly(self):
        """Should compute max_seq from capped album_seq."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 4, 5, 1, 2, 3, 4]),
            "artist_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
        }
        artist_album_counts = pd.Series([5, 4])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=3, artist_album_counts=artist_album_counts
        )

        assert result["max_seq"] == 3, "max_seq should be max of capped sequences"

    def test_max_albums_one(self):
        """Edge case: max_albums=1 should give all seq=1."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 1, 2]),
            "artist_idx": np.array([0, 0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([3, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=1, artist_album_counts=artist_album_counts
        )

        expected = np.array([1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 1

    def test_artists_with_exactly_max_albums(self):
        """Artists with exactly max_albums should not be modified."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 1, 2, 3]),
            "artist_idx": np.array([0, 0, 0, 1, 1, 1]),
        }
        artist_album_counts = pd.Series([3, 3])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=3, artist_album_counts=artist_album_counts
        )

        # No change needed
        expected = np.array([1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 3

    def test_no_artists_above_cap(self):
        """Should handle case where no artists exceed cap."""
        model_args = {
            "album_seq": np.array([1, 2, 1, 2]),
            "artist_idx": np.array([0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([2, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=5, artist_album_counts=artist_album_counts
        )

        # No change
        expected = np.array([1, 2, 1, 2])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 2

    def test_guards_against_non_positive_cap(self):
        """Should handle max_albums_cap <= 0 by clamping to 1."""
        model_args = {
            "album_seq": np.array([1, 2, 3]),
            "artist_idx": np.array([0, 0, 0]),
        }
        artist_album_counts = pd.Series([3])

        # Zero cap should be treated as 1
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=0, artist_album_counts=artist_album_counts
        )
        assert result["max_seq"] == 1

        # Negative cap should be treated as 1
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=-5, artist_album_counts=artist_album_counts
        )
        assert result["max_seq"] == 1


class TestNRefComputation:
    """Tests verifying n_ref (reference review count) computation from model data.

    n_ref = median(n_reviews) is computed by train_models() and added to model_args
    before calling fit_model(). These tests verify the formula and that
    prepare_model_data() returns the base keys correctly (n_ref is added later
    by train_models, not prepare_model_data).
    """

    def test_n_ref_equals_median_of_n_reviews(self):
        """n_ref should be the median of n_reviews values from model_args."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "C"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            }
        )

        model_args, _valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # n_ref formula: median of n_reviews
        expected_n_ref = float(np.median(model_args["n_reviews"]))
        assert expected_n_ref == 30.0, f"Expected median 30.0, got {expected_n_ref}"

    def test_model_args_keys_include_expected_heteroscedastic_keys(self):
        """prepare_model_data should return base keys but NOT n_ref (added by train_models)."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, 20, 30, 40],
            }
        )

        model_args, _valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # Base keys from prepare_model_data
        expected_base_keys = {
            "artist_idx",
            "album_seq",
            "prev_score",
            "X",
            "y",
            "n_reviews",
            "n_artists",
            "artist_album_counts",
        }
        assert expected_base_keys.issubset(set(model_args.keys())), (
            f"Missing keys: {expected_base_keys - set(model_args.keys())}"
        )

        # n_ref is NOT added by prepare_model_data -- it's added by train_models()
        assert "n_ref" not in model_args, (
            "n_ref should NOT be in model_args from prepare_model_data; "
            "it is added by train_models() after max_albums capping"
        )


class TestStrictSamplingConfigValidation:
    """Tests for strict-mode sampling preflight validation."""

    def test_non_strict_allows_single_chain_low_samples(self):
        """Non-strict mode should not enforce publication convergence constraints."""
        _validate_strict_sampling_config(
            strict=False,
            num_chains=1,
            num_samples=100,
            ess_threshold=400,
        )

    def test_strict_requires_at_least_two_chains(self):
        """Strict mode should fail fast when R-hat cannot be computed."""
        with pytest.raises(ConvergenceError, match="at least 2 chains"):
            _validate_strict_sampling_config(
                strict=True,
                num_chains=1,
                num_samples=1000,
                ess_threshold=400,
            )

    def test_strict_requires_samples_meeting_ess_threshold(self):
        """Strict mode should fail fast when ESS threshold is unattainable."""
        with pytest.raises(ConvergenceError, match="num_samples >= ess_threshold"):
            _validate_strict_sampling_config(
                strict=True,
                num_chains=4,
                num_samples=100,
                ess_threshold=400,
            )

    def test_strict_valid_config_passes(self):
        """A publication-grade strict configuration should pass preflight."""
        _validate_strict_sampling_config(
            strict=True,
            num_chains=4,
            num_samples=1000,
            ess_threshold=400,
        )

    def test_strict_samples_equal_ess_threshold_passes(self):
        """Boundary: num_samples == ess_threshold should pass."""
        _validate_strict_sampling_config(
            strict=True,
            num_chains=2,
            num_samples=400,
            ess_threshold=400,
        )

    def test_strict_samples_one_below_ess_threshold_fails(self):
        """Boundary: num_samples just below ess_threshold should fail."""
        with pytest.raises(ConvergenceError, match="num_samples >= ess_threshold"):
            _validate_strict_sampling_config(
                strict=True,
                num_chains=2,
                num_samples=399,
                ess_threshold=400,
            )

    def test_strict_two_chains_passes(self):
        """Boundary: exactly 2 chains should pass (minimum for R-hat)."""
        _validate_strict_sampling_config(
            strict=True,
            num_chains=2,
            num_samples=1000,
            ess_threshold=400,
        )

    def test_convergence_error_includes_stage(self):
        """ConvergenceError from validation should include stage='train'."""
        with pytest.raises(ConvergenceError) as exc_info:
            _validate_strict_sampling_config(
                strict=True,
                num_chains=1,
                num_samples=1000,
                ess_threshold=400,
            )
        assert exc_info.value.stage == "train"


class TestPrepareModelDataEdgeCases:
    """Additional edge case tests for prepare_model_data."""

    def test_artist_to_idx_in_model_args(self):
        """prepare_model_data should include artist_to_idx mapping."""
        df = pd.DataFrame(
            {
                "Artist": ["Alice", "Alice", "Bob"],
                "User_Score": [70.0, 75.0, 80.0],
                "feature_1": [1.0, 2.0, 3.0],
                "n_reviews": [10, 20, 30],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert "artist_to_idx" in model_args
        assert model_args["artist_to_idx"]["Alice"] == 0  # sorted alphabetically
        assert model_args["artist_to_idx"]["Bob"] == 1

    def test_single_artist(self):
        """Should work with a single artist."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A"],
                "User_Score": [70.0, 75.0, 80.0],
                "feature_1": [1.0, 2.0, 3.0],
                "n_reviews": [10, 20, 30],
            }
        )
        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert model_args["n_artists"] == 1
        np.testing.assert_array_equal(model_args["artist_idx"], [0, 0, 0])
        np.testing.assert_array_equal(model_args["album_seq"], [1, 2, 3])
        assert valid_mask.all()

    def test_n_reviews_cast_to_int32(self):
        """n_reviews should be cast to int32 after validation."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
                "feature_1": [1.0, 2.0],
                "n_reviews": [10.0, 20.0],  # float input
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert model_args["n_reviews"].dtype == np.int32

    def test_feature_matrix_dtype_float32(self):
        """Feature matrix should always be float32."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
                "feature_1": [1, 2],  # int input
                "feature_2": [3.0, 4.0],
                "n_reviews": [10, 20],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1", "feature_2"], min_albums_filter=1)
        assert model_args["X"].dtype == np.float32

    def test_artist_album_counts_full_range(self):
        """artist_album_counts should be reindexed to full artist range."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "C"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, 20, 30, 40],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        counts = model_args["artist_album_counts"]
        # A=2, B=1, C=1 - should have all 3 artists
        assert len(counts) == 3
        assert counts[0] == 2  # A
        assert counts[1] == 1  # B
        assert counts[2] == 1  # C

    def test_valid_mask_all_true_when_no_invalid(self):
        """valid_mask should be all True when all n_reviews are valid."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "feature_1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        _model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert valid_mask.all()
        assert len(valid_mask) == 2

    def test_prev_score_single_album_artist_uses_global_mean(self):
        """Single-album artist should get global mean as prev_score."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "feature_1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        # Global mean may come from dataset_stats.json or training fallback
        global_mean = model_args["global_mean_score"]
        np.testing.assert_allclose(model_args["prev_score"], [global_mean, global_mean])

    def test_min_albums_filter_zero_no_clamping(self):
        """min_albums_filter=0 should never clamp (all artists pass threshold)."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "feature_1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=0)
        # Single albums for each artist, no clamping occurs
        np.testing.assert_array_equal(model_args["album_seq"], [1, 1])

    def test_n_reviews_negative_filtered(self):
        """Negative n_reviews should be filtered out."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, -5, 30, 40],  # One negative
            }
        )
        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert valid_mask.sum() == 3
        assert not valid_mask[1]  # The negative row is filtered
        assert len(model_args["y"]) == 3

    def test_y_is_float32(self):
        """Target y should be float32."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70, 75],  # int input
                "feature_1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert model_args["y"].dtype == np.float32

    def test_artists_sorted_deterministically(self):
        """Artist index assignment should be sorted alphabetically."""
        df = pd.DataFrame(
            {
                "Artist": ["Zebra", "Apple", "Mango"],
                "User_Score": [70.0, 75.0, 80.0],
                "feature_1": [1.0, 2.0, 3.0],
                "n_reviews": [10, 20, 30],
            }
        )
        model_args, _valid = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        # Sorted: Apple=0, Mango=1, Zebra=2
        # Data order: Zebra(2), Apple(0), Mango(1)
        np.testing.assert_array_equal(model_args["artist_idx"], [2, 0, 1])


class TestApplyMaxAlbumsCapEdgeCases:
    """Additional edge case tests for _apply_max_albums_cap."""

    def test_mixed_artists_above_and_below_cap(self):
        """Only artists above the cap should be modified."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 4, 5, 1, 2]),  # Artist 0: 5, Artist 1: 2
            "artist_idx": np.array([0, 0, 0, 0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([5, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=4, artist_album_counts=artist_album_counts
        )

        # Artist 0: offset = 5-4 = 1 -> [max(1,1-1), max(1,2-1), max(1,3-1), max(1,4-1), max(1,5-1)]
        #                             = [1, 1, 2, 3, 4]
        # Artist 1: offset = 0 -> [1, 2] (no change)
        expected = np.array([1, 1, 2, 3, 4, 1, 2])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 4

    def test_single_artist_below_cap(self):
        """Single artist below cap should not be modified."""
        model_args = {
            "album_seq": np.array([1, 2]),
            "artist_idx": np.array([0, 0]),
        }
        artist_album_counts = pd.Series([2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=10, artist_album_counts=artist_album_counts
        )
        np.testing.assert_array_equal(result["album_seq"], [1, 2])
        assert result["max_seq"] == 2

    def test_large_cap_value(self):
        """Very large cap should have no effect."""
        model_args = {
            "album_seq": np.array([1, 2, 3]),
            "artist_idx": np.array([0, 0, 0]),
        }
        artist_album_counts = pd.Series([3])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=1000, artist_album_counts=artist_album_counts
        )
        np.testing.assert_array_equal(result["album_seq"], [1, 2, 3])
        assert result["max_seq"] == 3

    def test_result_preserves_other_keys(self):
        """_apply_max_albums_cap should preserve other model_args keys."""
        model_args = {
            "album_seq": np.array([1, 2, 3]),
            "artist_idx": np.array([0, 0, 0]),
            "y": np.array([70.0, 75.0, 80.0]),
            "X": np.array([[1.0], [2.0], [3.0]]),
        }
        artist_album_counts = pd.Series([3])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=2, artist_album_counts=artist_album_counts
        )
        np.testing.assert_array_equal(result["y"], [70.0, 75.0, 80.0])
        np.testing.assert_array_equal(result["X"], [[1.0], [2.0], [3.0]])

    def test_album_seq_dtype_is_int32(self):
        """Capped album_seq should be int32."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 4]),
            "artist_idx": np.array([0, 0, 0, 0]),
        }
        artist_album_counts = pd.Series([4])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=2, artist_album_counts=artist_album_counts
        )
        assert result["album_seq"].dtype == np.int32

    def test_multiple_artists_all_above_cap(self):
        """All artists above cap should all be capped."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 4, 1, 2, 3, 4, 5]),
            "artist_idx": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]),
        }
        artist_album_counts = pd.Series([4, 5])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=2, artist_album_counts=artist_album_counts
        )
        # Artist 0: offset=4-2=2 -> [max(1,1-2)=1, max(1,2-2)=1, max(1,3-2)=1, max(1,4-2)=2]
        # Artist 1: offset=5-2=3 -> [1,1,1,1,max(1,5-3)=2]
        expected = np.array([1, 1, 1, 2, 1, 1, 1, 1, 2])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 2


class TestLoadTrainingDataEdgeCases:
    """Additional edge case tests for load_training_data."""

    def test_successful_load_returns_three_elements(self, tmp_path):
        """Successful load should return (model_args, feature_cols, train_df) triple."""
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [4.0, 5.0, 6.0],
                "n_reviews": [10, 20, 30],
            },
            index=pd.RangeIndex(3),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B"],
                "User_Score": [70.0, 75.0, 80.0],
            },
            index=pd.RangeIndex(3),
        )
        features_df.to_parquet(tmp_path / "features.parquet")
        splits_df.to_parquet(tmp_path / "splits.parquet")

        model_args, feature_cols, train_df = load_training_data(
            tmp_path / "features.parquet", tmp_path / "splits.parquet", min_albums_filter=1
        )
        assert isinstance(model_args, dict)
        assert isinstance(feature_cols, list)
        assert isinstance(train_df, pd.DataFrame)
        assert "feature_1" in feature_cols
        assert "feature_2" in feature_cols
        assert "n_reviews" not in feature_cols

    def test_no_overlap_columns(self, tmp_path):
        """Should work when there are no overlapping columns."""
        features_df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0],
                "n_reviews": [10, 20],
            },
            index=pd.RangeIndex(2),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
            },
            index=pd.RangeIndex(2),
        )
        features_df.to_parquet(tmp_path / "features.parquet")
        splits_df.to_parquet(tmp_path / "splits.parquet")

        model_args, feature_cols, train_df = load_training_data(
            tmp_path / "features.parquet", tmp_path / "splits.parquet", min_albums_filter=1
        )
        assert "feat_a" in feature_cols
        assert "Artist" in train_df.columns
        assert "User_Score" in train_df.columns

    def test_train_df_filtered_by_valid_mask(self, tmp_path):
        """Returned train_df should be filtered to match valid model data."""
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, np.nan, 30, 40],  # 1 invalid out of 4 = 25%
            },
            index=pd.RangeIndex(4),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
            },
            index=pd.RangeIndex(4),
        )
        features_df.to_parquet(tmp_path / "features.parquet")
        splits_df.to_parquet(tmp_path / "splits.parquet")

        model_args, _feat_cols, train_df = load_training_data(
            tmp_path / "features.parquet", tmp_path / "splits.parquet", min_albums_filter=1
        )
        # train_df should be filtered to exclude the NaN n_reviews row
        assert len(train_df) == 3
        assert len(model_args["y"]) == 3


# --- from unit/pipelines/test_train_bayes_coverage.py ---


def _make_ctx(**overrides):
    """Create a StageContext-like namespace with sensible defaults."""
    defaults = {
        "seed": 42,
        "strict": False,
        "max_albums": 50,
        "min_albums_filter": 2,
        "num_chains": 4,
        "num_samples": 1000,
        "num_warmup": 500,
        "target_accept": 0.9,
        "max_tree_depth": 10,
        "chain_method": "sequential",
        "rhat_threshold": 1.01,
        "ess_threshold": 400,
        "allow_divergences": False,
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_alpha": 2.0,
        "n_exponent_beta": 4.0,
        "n_exponent_prior": "logit-normal",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_train_parquets(tmp_path, n_artists=3, n_albums_per=3, n_features=2):
    """Create feature and split parquet files suitable for train_models."""
    n_rows = n_artists * n_albums_per
    artists = []
    for i in range(n_artists):
        artists.extend([f"artist_{i}"] * n_albums_per)

    splits_df = pd.DataFrame(
        {
            "Artist": artists,
            "User_Score": np.random.default_rng(42).uniform(60, 95, n_rows).astype(np.float32),
        },
        index=pd.RangeIndex(n_rows),
    )

    feature_data = {
        f"feature_{i}": np.random.default_rng(42 + i).standard_normal(n_rows).astype(np.float32)
        for i in range(n_features)
    }
    feature_data["n_reviews"] = np.random.default_rng(99).integers(5, 200, n_rows)
    features_df = pd.DataFrame(feature_data, index=pd.RangeIndex(n_rows))

    features_path = tmp_path / "features.parquet"
    splits_path = tmp_path / "splits.parquet"
    features_df.to_parquet(features_path)
    splits_df.to_parquet(splits_path)
    return features_path, splits_path


def _make_fake_fit_result(divergences=0, runtime=10.0, n_chains=4, n_samples=100):
    """Create a mock FitResult with minimal structure."""
    result = MagicMock()
    result.divergences = divergences
    result.runtime_seconds = runtime
    result.gpu_info = "CPU only"

    # Build a mock posterior that supports dict-style access
    sigma_obs_mock = MagicMock()
    sigma_obs_mock.mean.return_value = 5.0
    sigma_obs_mock.values = np.full((n_chains, n_samples), 5.0)

    posterior = MagicMock()
    posterior.__getitem__ = MagicMock(return_value=sigma_obs_mock)

    idata = MagicMock()
    idata.posterior = posterior
    result.idata = idata

    return result


def _make_fake_diagnostics(passed=True, rhat_max=1.003, ess_bulk_min=2000):
    """Create a mock ConvergenceDiagnostics."""
    diag = MagicMock()
    diag.passed = passed
    diag.rhat_max = rhat_max
    diag.ess_bulk_min = ess_bulk_min
    diag.ess_tail_min = 1800
    diag.divergences = 0
    diag.rhat_threshold = 1.01
    diag.ess_threshold = 400
    return diag


class TestValidateStrictSamplingConfigExtended:
    def test_exact_boundary_two_chains_passes(self):
        """Exactly 2 chains should pass strict validation."""
        _validate_strict_sampling_config(
            strict=True, num_chains=2, num_samples=500, ess_threshold=400
        )

    def test_zero_chains_in_strict_fails(self):
        """Zero chains should fail in strict mode."""
        with pytest.raises(ConvergenceError, match="at least 2 chains"):
            _validate_strict_sampling_config(
                strict=True, num_chains=0, num_samples=1000, ess_threshold=400
            )

    def test_convergence_error_stage_is_train(self):
        """ConvergenceError from strict validation should report stage='train'."""
        with pytest.raises(ConvergenceError) as exc_info:
            _validate_strict_sampling_config(
                strict=True, num_chains=1, num_samples=100, ess_threshold=400
            )
        assert exc_info.value.stage == "train"

    def test_non_strict_allows_any_configuration(self):
        """Non-strict mode should accept any configuration without error."""
        _validate_strict_sampling_config(
            strict=False, num_chains=0, num_samples=0, ess_threshold=10000
        )


class TestTrainModelsEntryPoint:
    def test_train_models_homoscedastic_mode(self, tmp_path):
        """train_models should succeed with homoscedastic mode (n_exponent=0.0)."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.0, learn_n_exponent=False)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        fake_manifest = MagicMock()

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", fake_manifest),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["model_type"] == "user_score"
        assert summary["heteroscedastic_mode"]["mode"] == "homoscedastic"
        assert "mcmc_config" in summary

    def test_train_models_fixed_heteroscedastic_mode(self, tmp_path):
        """train_models should produce fixed heteroscedastic summary when n_exponent != 0."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.5, learn_n_exponent=False)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "fixed"
        assert summary["heteroscedastic_mode"]["n_exponent"] == 0.5

    def test_train_models_learned_heteroscedastic_sigma_obs_mode(self, tmp_path):
        """train_models should produce learned heteroscedastic summary with sigma_obs parameterization."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        # n_exponent=0.0 but learn_n_exponent=True, n_ref will be None when
        # learn_n_exponent is True but n_exponent is 0.0 -- actually n_ref is
        # set when learn_n_exponent is True. Let's just check the learned path.
        ctx = _make_ctx(learn_n_exponent=True, n_exponent=0.0, n_exponent_prior="logit-normal")

        # Build a more detailed fit_result for learned mode
        fit_result = _make_fake_fit_result()
        n_exp_samples = np.full((4, 100), 0.4)
        sigma_obs_samples = np.full((4, 100), 5.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.4

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = np.full((4, 100), 6.0)
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Mock az.hdi and az.summary for n_exponent/sigma_ref
        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.3, 0.5])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "learned"
        assert summary["heteroscedastic_mode"]["parameterization"] == "sigma_ref"
        assert "sigma_ref" in summary["heteroscedastic_mode"]

    def test_train_models_learned_beta_prior_logging(self, tmp_path):
        """train_models should log beta prior mode when n_exponent_prior='beta'."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(
            learn_n_exponent=True,
            n_exponent=0.0,
            n_exponent_prior="beta",
            n_exponent_alpha=3.0,
            n_exponent_beta=5.0,
        )

        fit_result = _make_fake_fit_result()
        n_exp_samples = np.full((4, 100), 0.35)
        sigma_obs_samples = np.full((4, 100), 5.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.35

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = np.full((4, 100), 6.0)
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.2, 0.5])))
        fake_hdi = hdi_result
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [900.0],
                "r_hat": [1.001],
            }
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=fake_hdi),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            # Should not raise
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "learned"


class TestTrainModelsStrictConvergence:
    def test_strict_divergences_raises(self, tmp_path):
        """Strict mode with divergences > 0 and allow_divergences=False should raise."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=True, allow_divergences=False, num_chains=4, num_samples=1000)

        fit_result = _make_fake_fit_result(divergences=5)
        diagnostics = _make_fake_diagnostics(passed=True)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ConvergenceError, match="divergent transitions"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)

    def test_strict_divergences_allowed_does_not_raise(self, tmp_path):
        """Strict mode with allow_divergences=True should not raise on divergences."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=True, allow_divergences=True, num_chains=4, num_samples=1000)

        fit_result = _make_fake_fit_result(divergences=3)
        diagnostics = _make_fake_diagnostics(passed=True)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)
            assert summary["divergences"] == 3

    def test_strict_diagnostics_failed_raises(self, tmp_path):
        """Strict mode with diagnostics.passed=False should raise ConvergenceError."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=True, allow_divergences=True, num_chains=4, num_samples=1000)

        fit_result = _make_fake_fit_result(divergences=0)
        diagnostics = _make_fake_diagnostics(passed=False, rhat_max=1.05, ess_bulk_min=100)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ConvergenceError, match="Convergence failed"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)


class TestTrainModelsHighDivergenceWarning:
    def test_high_divergence_rate_does_not_crash(self, tmp_path):
        """High divergence rate (>10%) should log warning but not crash in non-strict."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=False, num_chains=4, num_samples=100)

        # 50 divergences out of 400 total = 12.5%
        fit_result = _make_fake_fit_result(divergences=50, n_chains=4, n_samples=100)
        diagnostics = _make_fake_diagnostics(passed=False)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)
            assert summary["divergence_rate"] > 0.10


class TestTrainModelsSummaryOutput:
    def test_training_summary_written_to_disk(self, tmp_path):
        """training_summary.json should be written with complete metadata."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx()

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="hash_abc",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        summary_path = tmp_path / "models/training_summary.json"
        assert summary_path.exists()
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        assert saved["data_hash"] == "hash_abc"
        assert "feature_scaler" in saved
        assert "feature_cols" in saved
        assert "n_reviews_stats" in saved
        assert "convergence_thresholds" in saved

    def test_default_feature_paths_used(self, tmp_path):
        """When features_path and splits_path are None, defaults should be used."""
        ctx = _make_ctx()

        # Create the default paths
        default_features = tmp_path / "data/features/train_features.parquet"
        default_splits = tmp_path / "data/splits/within_entity_temporal/train.parquet"

        # Write real parquets at default paths
        n = 9
        features_df = pd.DataFrame(
            {
                "feature_0": np.ones(n, dtype=np.float32),
                "n_reviews": np.full(n, 50, dtype=np.int32),
            },
            index=pd.RangeIndex(n),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "User_Score": np.linspace(70, 90, n, dtype=np.float32),
            },
            index=pd.RangeIndex(n),
        )
        default_features.parent.mkdir(parents=True, exist_ok=True)
        default_splits.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(default_features)
        splits_df.to_parquet(default_splits)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx)
            assert summary["model_type"] == "user_score"


class TestTrainModelsFeatureStandardization:
    def test_features_are_standardized(self, tmp_path):
        """Feature matrix X should be z-score standardized before fitting."""
        features_path, splits_path = _make_train_parquets(tmp_path, n_features=2)
        ctx = _make_ctx()

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        X = captured_model_args["X"]
        # Each column should have mean ~0 and std ~1 (after standardization)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            assert abs(col.mean()) < 0.01, f"Column {col_idx} mean not ~0"
            assert abs(col.std() - 1.0) < 0.1, f"Column {col_idx} std not ~1"

    def test_constant_features_unscaled(self, tmp_path):
        """Constant features (std=0) should remain as-is after standardization."""
        n = 9
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "User_Score": np.linspace(70, 90, n, dtype=np.float32),
            },
            index=pd.RangeIndex(n),
        )
        features_df = pd.DataFrame(
            {
                "feature_const": np.ones(n, dtype=np.float32),  # constant
                "feature_vary": np.arange(n, dtype=np.float32),  # varying
                "n_reviews": np.full(n, 50, dtype=np.int32),
            },
            index=pd.RangeIndex(n),
        )
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        ctx = _make_ctx()

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        X = captured_model_args["X"]
        # Constant column should be (1-1)/1 = 0
        # (X_std_safe = 1.0 when std=0, so (x-mean)/1 = x - mean)
        const_col = X[:, 0]
        assert np.allclose(const_col, 0.0), "Constant feature should be centered to 0"

    def test_nan_in_X_raises_value_error(self, tmp_path):
        """NaN in feature matrix X after fillna should raise ValueError."""
        n = 6
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "C", "C"],
                "User_Score": np.linspace(70, 90, n, dtype=np.float32),
            },
            index=pd.RangeIndex(n),
        )
        # Create features that have NaN AFTER the join but before standardization.
        # Since fillna(0) is applied to feature_cols before prepare_model_data,
        # and then X = train_df[feature_cols].values is built inside prepare_model_data,
        # NaN in X can only happen if something goes wrong. Let's test the
        # validation inside train_models by monkeypatching the X array.
        features_df = pd.DataFrame(
            {
                "feature_1": np.ones(n, dtype=np.float32),
                "n_reviews": np.full(n, 50, dtype=np.int32),
            },
            index=pd.RangeIndex(n),
        )
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        ctx = _make_ctx()

        # Patch load_training_data to return model_args with NaN in X
        def _fake_load(*args, **kwargs):
            model_args = {
                "artist_idx": np.array([0, 0, 1, 1, 2, 2]),
                "album_seq": np.array([1, 2, 1, 2, 1, 2]),
                "prev_score": np.full(6, 75.0),
                "X": np.array([[1.0], [np.nan], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32),
                "y": np.linspace(70, 90, 6, dtype=np.float32),
                "n_reviews": np.full(6, 50, dtype=np.int32),
                "n_artists": 3,
                "artist_album_counts": pd.Series([2, 2, 2]),
                "artist_to_idx": {"A": 0, "B": 1, "C": 2},
                "global_mean_score": 75.0,
                "ar_center": np.float32(75.0),
                "ar_center_value": 75.0,
            }
            feature_cols = ["feature_1"]
            train_df = pd.DataFrame(
                {"Artist": ["A", "A", "B", "B", "C", "C"]},
            )
            return model_args, feature_cols, train_df

        with (
            patch(
                "panelcast.pipelines.train_bayes.load_training_data",
                side_effect=_fake_load,
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ValueError, match="NaN values"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)


class TestTrainModelsNRefComputation:
    def test_n_ref_set_for_learned_exponent(self, tmp_path):
        """n_ref should be set to median of n_reviews when learn_n_exponent=True."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(learn_n_exponent=True)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.3, 0.5])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        fit_result = _make_fake_fit_result()
        n_exp_samples = np.full((4, 100), 0.4)
        sigma_obs_samples = np.full((4, 100), 5.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.4

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = np.full((4, 100), 6.0)
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }
        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert captured_model_args["n_ref"] is not None

    def test_n_ref_none_for_homoscedastic(self, tmp_path):
        """n_ref should be None when n_exponent=0 and learn_n_exponent=False."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.0, learn_n_exponent=False)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert captured_model_args["n_ref"] is None


class TestTrainModelsMCMCConfigPassthrough:
    def test_mcmc_config_from_ctx(self, tmp_path):
        """MCMCConfig should be constructed from ctx attributes."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(
            num_warmup=200,
            num_samples=300,
            num_chains=2,
            seed=123,
            target_accept=0.85,
            max_tree_depth=8,
            chain_method="vectorized",
        )

        captured_config = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_config["config"] = config
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            train_models(ctx, features_path=features_path, splits_path=splits_path)

        config = captured_config["config"]
        assert config.num_warmup == 200
        assert config.num_samples == 300
        assert config.num_chains == 2
        assert config.seed == 123
        assert config.target_accept_prob == 0.85
        assert config.max_tree_depth == 8
        assert config.chain_method == "vectorized"


class TestTrainModelsMinAlbumsFilter:
    def test_min_albums_filter_passthrough(self, tmp_path):
        """min_albums_filter from ctx should be passed to load_training_data."""
        features_path, splits_path = _make_train_parquets(tmp_path, n_artists=3, n_albums_per=3)
        ctx = _make_ctx(min_albums_filter=3)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)
            assert summary["min_albums_filter"] == 3


class TestTrainModelsHeteroscedasticConfig:
    def test_model_args_include_heteroscedastic_keys(self, tmp_path):
        """Model args passed to fit_model should include heteroscedastic configuration."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.5, learn_n_exponent=False)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
        ):
            train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert captured_model_args["n_exponent"] == 0.5
        assert captured_model_args["learn_n_exponent"] is False
        assert captured_model_args["n_exponent_prior"] == "logit-normal"
        # n_ref should be set since n_exponent != 0
        assert captured_model_args["n_ref"] is not None


class TestTrainModelsPriorsPassthrough:
    def test_priors_in_model_args_and_summary(self, tmp_path):
        """PriorConfig should be passed in model_args and serialized in summary."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent_alpha=3.0, n_exponent_beta=6.0)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert "priors" in captured_model_args
        assert summary["priors"]["n_exponent_alpha"] == 3.0
        assert summary["priors"]["n_exponent_beta"] == 6.0


class TestTrainModelsFitFailure:
    def test_fit_model_exception_propagates(self, tmp_path):
        """If fit_model raises, the error should propagate up."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch(
                "panelcast.pipelines.train_bayes.fit_model",
                side_effect=RuntimeError("MCMC crashed"),
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(RuntimeError, match="MCMC crashed"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)


# --- from unit/pipelines/test_train_bayes_new.py ---


def _make_ctx_new(**overrides):
    defaults = {
        "seed": 42,
        "strict": False,
        "max_albums": 50,
        "min_albums_filter": 2,
        "num_chains": 4,
        "num_samples": 1000,
        "num_warmup": 500,
        "target_accept": 0.9,
        "max_tree_depth": 10,
        "chain_method": "sequential",
        "rhat_threshold": 1.01,
        "ess_threshold": 400,
        "allow_divergences": False,
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_alpha": 2.0,
        "n_exponent_beta": 4.0,
        "n_exponent_prior": "logit-normal",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_train_parquets_new(tmp_path, n_artists=3, n_albums_per=3, n_features=2):
    n_rows = n_artists * n_albums_per
    artists = []
    for i in range(n_artists):
        artists.extend([f"artist_{i}"] * n_albums_per)

    splits_df = pd.DataFrame(
        {
            "Artist": artists,
            "User_Score": np.random.default_rng(42).uniform(60, 95, n_rows).astype(np.float32),
        },
        index=pd.RangeIndex(n_rows),
    )

    feature_data = {
        f"feature_{i}": np.random.default_rng(42 + i).standard_normal(n_rows).astype(np.float32)
        for i in range(n_features)
    }
    feature_data["n_reviews"] = np.random.default_rng(99).integers(5, 200, n_rows)
    features_df = pd.DataFrame(feature_data, index=pd.RangeIndex(n_rows))

    features_path = tmp_path / "features.parquet"
    splits_path = tmp_path / "splits.parquet"
    features_df.to_parquet(features_path)
    splits_df.to_parquet(splits_path)
    return features_path, splits_path


def _make_fake_fit_result_new(divergences=0, n_chains=4, n_samples=100):
    result = MagicMock()
    result.divergences = divergences
    result.runtime_seconds = 10.0
    result.gpu_info = "CPU only"

    sigma_obs_mock = MagicMock()
    sigma_obs_mock.mean.return_value = 5.0
    sigma_obs_mock.values = np.full((n_chains, n_samples), 5.0)

    posterior = MagicMock()
    posterior.__getitem__ = MagicMock(return_value=sigma_obs_mock)

    idata = MagicMock()
    idata.posterior = posterior
    result.idata = idata

    return result


def _make_fake_diagnostics_new(passed=True, rhat_max=1.003, ess_bulk_min=2000):
    diag = MagicMock()
    diag.passed = passed
    diag.rhat_max = rhat_max
    diag.ess_bulk_min = ess_bulk_min
    diag.ess_tail_min = 1800
    diag.divergences = 0
    diag.rhat_threshold = 1.01
    diag.ess_threshold = 400
    return diag


class TestTrainModelsLearnedSigmaObsParamterization:
    """Test learned heteroscedastic mode with sigma_obs parameterization (no sigma_ref)."""

    def test_learned_mode_no_sigma_ref(self, tmp_path):
        """When n_ref is None in model_args, sigma_obs parameterization is used."""
        features_path, splits_path = _make_train_parquets_new(tmp_path)
        ctx = _make_ctx_new(learn_n_exponent=True, n_exponent=0.0)

        fit_result = _make_fake_fit_result_new()

        # Build posterior mocks for learned mode
        n_exp_samples = np.full((4, 100), 0.4)
        sigma_obs_samples = np.full((4, 100), 5.0)

        sigma_ref_samples = np.full((4, 100), 6.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.4

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = sigma_ref_samples
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics_new()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.3, 0.5])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        # Patch model_args to have n_ref=None to test sigma_obs parameterization
        original_load = None

        def _fake_load(*args, **kwargs):
            model_args = {
                "artist_idx": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                "album_seq": np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
                "prev_score": np.full(9, 75.0, dtype=np.float32),
                "X": np.random.default_rng(42).standard_normal((9, 2)).astype(np.float32),
                "y": np.linspace(60, 95, 9, dtype=np.float32),
                "n_reviews": np.full(9, 50, dtype=np.int32),
                "n_artists": 3,
                "artist_album_counts": pd.Series([3, 3, 3]),
                "artist_to_idx": {"artist_0": 0, "artist_1": 1, "artist_2": 2},
                "global_mean_score": 75.0,
                "ar_center": np.float32(75.0),
                "ar_center_value": 75.0,
            }
            feature_cols = ["feature_0", "feature_1"]
            train_df = pd.DataFrame(
                {"Artist": ["artist_0"] * 3 + ["artist_1"] * 3 + ["artist_2"] * 3},
            )
            return model_args, feature_cols, train_df

        with (
            patch("panelcast.pipelines.train_bayes.load_training_data", side_effect=_fake_load),
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "learned"
        # When n_ref is set (learn_n_exponent=True => n_ref is computed),
        # parameterization should be sigma_ref
        assert summary["heteroscedastic_mode"]["parameterization"] in ("sigma_ref", "sigma_obs")


class TestTrainModelsInterpretation:
    """Test n_exponent interpretation (closer to cube-root vs square-root)."""

    def test_cube_root_interpretation(self, tmp_path):
        """Mean n_exponent closer to 0.33 should be 'closer to cube-root scaling'."""
        features_path, splits_path = _make_train_parquets_new(tmp_path)
        ctx = _make_ctx_new(learn_n_exponent=True)

        fit_result = _make_fake_fit_result_new()

        # n_exponent_mean = 0.35 => closer to 0.33 than 0.5
        n_exp_samples = np.full((4, 100), 0.35)
        sigma_obs_samples = np.full((4, 100), 5.0)
        sigma_ref_samples = np.full((4, 100), 6.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.35

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = sigma_ref_samples
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics_new()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.25, 0.45])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert "cube-root" in summary["heteroscedastic_mode"]["interpretation"]

    def test_sqrt_interpretation(self, tmp_path):
        """Mean n_exponent closer to 0.5 should be 'closer to square-root scaling'."""
        features_path, splits_path = _make_train_parquets_new(tmp_path)
        ctx = _make_ctx_new(learn_n_exponent=True)

        fit_result = _make_fake_fit_result_new()

        # n_exponent_mean = 0.48 => closer to 0.5 than 0.33
        n_exp_samples = np.full((4, 100), 0.48)
        sigma_obs_samples = np.full((4, 100), 5.0)
        sigma_ref_samples = np.full((4, 100), 6.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.48

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = sigma_ref_samples
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics_new()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.4, 0.56])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert "square-root" in summary["heteroscedastic_mode"]["interpretation"]


class TestPrepareModelDataEdgeCasesNew:
    """Additional edge case tests for prepare_model_data."""

    def test_large_min_albums_filter_clamps_all_to_seq_one(self):
        """When min_albums_filter exceeds all artist counts, all get seq=1."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, 20, 30, 40],
            }
        )
        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=100)
        # All artists have < 100 albums, so all album_seq should be 1
        np.testing.assert_array_equal(model_args["album_seq"], [1, 1, 1, 1])

    def test_n_reviews_exactly_at_fifty_percent_invalid(self):
        """Exactly 50% invalid n_reviews should still succeed (boundary check)."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, np.nan, 30, 40],  # 1 invalid out of 4 = 25%
            }
        )
        # 25% invalid should be OK
        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert len(model_args["y"]) == 3

    def test_multiple_features(self):
        """Multiple feature columns should all be in X."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "f1": [1.0, 2.0],
                "f2": [3.0, 4.0],
                "f3": [5.0, 6.0],
                "n_reviews": [10, 20],
            }
        )
        model_args, _valid = prepare_model_data(df, ["f1", "f2", "f3"], min_albums_filter=1)
        assert model_args["X"].shape == (2, 3)


class TestLoadTrainingDataNew:
    """Additional tests for load_training_data."""

    def test_min_albums_filter_passed_through(self, tmp_path):
        """min_albums_filter parameter should be passed to prepare_model_data."""
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            },
            index=pd.RangeIndex(5),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
            },
            index=pd.RangeIndex(5),
        )
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        # With min_albums_filter=5, both artists (A:3, B:2) are below threshold
        model_args, feature_cols, train_df = load_training_data(
            features_path, splits_path, min_albums_filter=5
        )
        # All album_seq should be 1 since both artists are below threshold
        np.testing.assert_array_equal(model_args["album_seq"], [1, 1, 1, 1, 1])


class TestApplyMaxAlbumsCapNew:
    """Additional tests for _apply_max_albums_cap."""

    def test_float_cap_converted_to_int(self):
        """Float max_albums_cap should be converted to int."""
        model_args = {
            "album_seq": np.array([1, 2, 3]),
            "artist_idx": np.array([0, 0, 0]),
        }
        artist_album_counts = pd.Series([3])
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=2.7, artist_album_counts=artist_album_counts
        )
        # 2.7 -> int(2.7) = 2
        assert result["max_seq"] <= 2

    def test_single_album_artist_always_seq_one(self):
        """Single album artists always have seq=1 regardless of cap."""
        model_args = {
            "album_seq": np.array([1, 1, 1]),
            "artist_idx": np.array([0, 1, 2]),
        }
        artist_album_counts = pd.Series([1, 1, 1])
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=1, artist_album_counts=artist_album_counts
        )
        np.testing.assert_array_equal(result["album_seq"], [1, 1, 1])
        assert result["max_seq"] == 1
