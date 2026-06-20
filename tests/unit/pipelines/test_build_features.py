"""Unit tests for build_features leakage controls and helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.features.artist import ArtistHistoryBlock
from panelcast.features.base import FeatureContext
from panelcast.features.pipeline import FeaturePipeline
from panelcast.pipelines.build_features import (
    _assign_n_reviews,
    _transform_with_train_history,
    get_default_feature_blocks,
    get_feature_blocks,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create DataFrame with parsed dates for feature tests."""
    df = pd.DataFrame(rows)
    df["Release_Date_Parsed"] = pd.to_datetime(df["Release_Date_Parsed"])
    return df


def test_transform_with_train_history_masks_target_scores_for_known_artist():
    """Held-out target rows must not update artist history features."""
    train_df = _make_df(
        [
            {
                "Artist": "Artist_A",
                "Album": "A1",
                "Release_Date_Parsed": "2019-01-01",
                "User_Score": 70.0,
                "Critic_Score": 60.0,
            },
            {
                "Artist": "Artist_A",
                "Album": "A2",
                "Release_Date_Parsed": "2020-01-01",
                "User_Score": 75.0,
                "Critic_Score": 65.0,
            },
        ]
    )
    target_df = _make_df(
        [
            {
                "Artist": "Artist_A",
                "Album": "A3",
                "Release_Date_Parsed": "2021-01-01",
                "User_Score": 80.0,
                "Critic_Score": 70.0,
            },
            {
                "Artist": "Artist_A",
                "Album": "A4",
                "Release_Date_Parsed": "2022-01-01",
                "User_Score": 85.0,
                "Critic_Score": 72.0,
            },
        ]
    )

    pipeline = FeaturePipeline([ArtistHistoryBlock({})])
    feature_ctx = FeatureContext(config={}, random_state=42)
    pipeline.fit(train_df, feature_ctx)

    target_features = _transform_with_train_history(
        pipeline=pipeline,
        train_df=train_df,
        target_df=target_df,
        feature_ctx=feature_ctx,
    )

    # Both held-out rows should only see the two training albums.
    assert (target_features["user_prior_count"] == 2).all()
    assert (target_features["critic_prior_count"] == 2).all()
    assert target_features["user_prior_mean"].nunique() == 1
    assert target_features["critic_prior_mean"].nunique() == 1


def test_transform_with_train_history_unknown_artist_stays_debut():
    """Unseen artists in held-out data should not leak within-target labels."""
    train_df = _make_df(
        [
            {
                "Artist": "Artist_A",
                "Album": "A1",
                "Release_Date_Parsed": "2019-01-01",
                "User_Score": 70.0,
                "Critic_Score": 60.0,
            }
        ]
    )
    target_df = _make_df(
        [
            {
                "Artist": "Artist_NEW",
                "Album": "N1",
                "Release_Date_Parsed": "2021-01-01",
                "User_Score": 95.0,
                "Critic_Score": 90.0,
            },
            {
                "Artist": "Artist_NEW",
                "Album": "N2",
                "Release_Date_Parsed": "2022-01-01",
                "User_Score": 10.0,
                "Critic_Score": 20.0,
            },
        ]
    )

    pipeline = FeaturePipeline([ArtistHistoryBlock({})])
    feature_ctx = FeatureContext(config={}, random_state=42)
    pipeline.fit(train_df, feature_ctx)

    target_features = _transform_with_train_history(
        pipeline=pipeline,
        train_df=train_df,
        target_df=target_df,
        feature_ctx=feature_ctx,
    )

    assert (target_features["user_prior_count"] == 0).all()
    assert (target_features["critic_prior_count"] == 0).all()
    assert (target_features["is_debut"] == 1).all()


# ============================================================================
# get_feature_blocks Tests
# ============================================================================


class TestGetFeatureBlocks:
    """Tests for get_feature_blocks function."""

    def test_all_enabled_returns_five_blocks(self):
        """All features enabled should return 5 blocks."""
        blocks = get_feature_blocks(enable_genre=True, enable_artist=True, enable_temporal=True)
        assert len(blocks) == 5

    def test_disable_genre_removes_one(self):
        """Disabling genre removes GenreBlock."""
        blocks = get_feature_blocks(enable_genre=False, enable_artist=True, enable_temporal=True)
        block_names = [b.name for b in blocks]
        assert len(blocks) == 4
        assert "genre" not in block_names

    def test_disable_artist_removes_one(self):
        """Disabling artist removes ArtistHistoryBlock."""
        blocks = get_feature_blocks(enable_genre=True, enable_artist=False, enable_temporal=True)
        block_names = [b.name for b in blocks]
        assert len(blocks) == 4
        assert "artist_history" not in block_names

    def test_disable_temporal_removes_one(self):
        """Disabling temporal removes TemporalBlock."""
        blocks = get_feature_blocks(enable_genre=True, enable_artist=True, enable_temporal=False)
        block_names = [b.name for b in blocks]
        assert len(blocks) == 4
        assert "temporal" not in block_names

    def test_all_disabled_returns_core_blocks(self):
        """Disabling all optional features returns only core blocks."""
        blocks = get_feature_blocks(enable_genre=False, enable_artist=False, enable_temporal=False)
        assert len(blocks) == 2  # album_type + collaboration

    def test_album_type_always_present(self):
        """AlbumTypeBlock is always included."""
        for genre in [True, False]:
            for artist in [True, False]:
                for temporal in [True, False]:
                    blocks = get_feature_blocks(
                        enable_genre=genre,
                        enable_artist=artist,
                        enable_temporal=temporal,
                    )
                    block_names = [b.name for b in blocks]
                    assert "album_type" in block_names

    def test_collaboration_always_present(self):
        """CollaborationBlock is always included."""
        blocks = get_feature_blocks(enable_genre=False, enable_artist=False, enable_temporal=False)
        block_names = [b.name for b in blocks]
        assert "collaboration" in block_names

    def test_block_order_temporal_before_album_type(self):
        """Temporal block comes before album type in dependency order."""
        blocks = get_feature_blocks(enable_genre=True, enable_artist=True, enable_temporal=True)
        block_names = [b.name for b in blocks]
        assert block_names.index("temporal") < block_names.index("album_type")

    def test_block_order_collaboration_last(self):
        """Collaboration block is always last."""
        blocks = get_feature_blocks(enable_genre=True, enable_artist=True, enable_temporal=True)
        assert blocks[-1].name == "collaboration"


class TestGetDefaultFeatureBlocks:
    """Tests for get_default_feature_blocks function."""

    def test_returns_all_blocks(self):
        """Default blocks include all feature blocks."""
        blocks = get_default_feature_blocks()
        assert len(blocks) == 5

    def test_matches_all_enabled(self):
        """Default blocks match calling get_feature_blocks with all enabled."""
        default_names = [b.name for b in get_default_feature_blocks()]
        all_enabled_names = [
            b.name
            for b in get_feature_blocks(enable_genre=True, enable_artist=True, enable_temporal=True)
        ]
        assert default_names == all_enabled_names


# ============================================================================
# _assign_n_reviews Tests
# ============================================================================


class TestAssignNReviews:
    """Tests for _assign_n_reviews function."""

    def test_assigns_n_reviews_column(self):
        """Assigns n_reviews column from Series."""
        features = pd.DataFrame({"feat1": [1, 2, 3]}, index=[0, 1, 2])
        n_reviews = pd.Series([10, 20, 30], index=[0, 1, 2], name="n_reviews")
        result = _assign_n_reviews(features, n_reviews, "test")
        assert "n_reviews" in result.columns
        assert result["n_reviews"].tolist() == [10, 20, 30]

    def test_preserves_original_columns(self):
        """Original feature columns are preserved."""
        features = pd.DataFrame({"feat1": [1], "feat2": [2]}, index=[0])
        n_reviews = pd.Series([10], index=[0], name="n_reviews")
        result = _assign_n_reviews(features, n_reviews, "test")
        assert "feat1" in result.columns
        assert "feat2" in result.columns

    def test_does_not_modify_input(self):
        """Input DataFrame is not modified."""
        features = pd.DataFrame({"feat1": [1, 2]}, index=[0, 1])
        n_reviews = pd.Series([10, 20], index=[0, 1])
        _assign_n_reviews(features, n_reviews, "test")
        assert "n_reviews" not in features.columns

    def test_mismatched_index_raises(self):
        """Mismatched indices raise ValueError."""
        features = pd.DataFrame({"feat1": [1, 2]}, index=[0, 1])
        n_reviews = pd.Series([10, 20], index=[5, 6])  # different indices
        with pytest.raises(ValueError, match="null values after reindexing"):
            _assign_n_reviews(features, n_reviews, "test")

    def test_partial_mismatch_raises(self):
        """Partial index overlap raises ValueError."""
        features = pd.DataFrame({"feat1": [1, 2, 3]}, index=[0, 1, 2])
        n_reviews = pd.Series([10, 20], index=[0, 1])  # missing index 2
        with pytest.raises(ValueError, match="1 null values"):
            _assign_n_reviews(features, n_reviews, "test")

    def test_error_message_includes_name(self):
        """Error message includes the split name for debugging."""
        features = pd.DataFrame({"feat1": [1]}, index=[0])
        n_reviews = pd.Series([10], index=[99])
        with pytest.raises(ValueError, match="my_split"):
            _assign_n_reviews(features, n_reviews, "my_split")

    def test_aligned_by_index(self):
        """Values are aligned by index, not position."""
        features = pd.DataFrame({"feat1": [1, 2, 3]}, index=[2, 0, 1])
        n_reviews = pd.Series([100, 200, 300], index=[0, 1, 2])
        result = _assign_n_reviews(features, n_reviews, "test")
        # Index 2 should get n_reviews value 300
        assert result.loc[2, "n_reviews"] == 300
        assert result.loc[0, "n_reviews"] == 100
        assert result.loc[1, "n_reviews"] == 200
