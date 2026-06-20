"""Unit tests for train_bayes pipeline data preparation functions.

Tests cover:
- load_training_data: DataFrame alignment and overlap handling
- prepare_model_data: Artist indexing, album sequences, n_reviews validation
- _apply_max_albums_cap: Most-recent album capping logic

These tests do NOT run MCMC - they focus on pure data preparation logic.
"""

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.errors import ConvergenceError
from panelcast.pipelines.train_bayes import (
    _apply_max_albums_cap,
    _validate_strict_sampling_config,
    load_training_data,
    prepare_model_data,
)

# =============================================================================
# Fixtures
# =============================================================================


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


# =============================================================================
# Tests for load_training_data
# =============================================================================


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


# =============================================================================
# Tests for prepare_model_data
# =============================================================================


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


# =============================================================================
# Tests for _apply_max_albums_cap
# =============================================================================


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


# =============================================================================
# Tests for n_ref computation
# =============================================================================


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
        assert expected_base_keys.issubset(
            set(model_args.keys())
        ), f"Missing keys: {expected_base_keys - set(model_args.keys())}"

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


# =============================================================================
# Additional tests for prepare_model_data
# =============================================================================


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


# =============================================================================
# Additional tests for _apply_max_albums_cap
# =============================================================================


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


# =============================================================================
# Additional tests for load_training_data
# =============================================================================


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
