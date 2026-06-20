"""Integration tests for data loading to feature pipeline.

Tests the data -> features integration pathway:
- Load raw CSV -> clean_albums transformation
- Cleaned data -> temporal split
- Split data -> feature pipeline (fit/transform)

These tests verify interface compatibility between modules
that was the source of bugs fixed in Phases 10-12.
"""

import pandas as pd
import pytest

from panelcast.data.split import (
    assert_no_artist_overlap,
    validate_temporal_split,
)
from panelcast.features.album_type import AlbumTypeBlock
from panelcast.features.artist import ArtistHistoryBlock
from panelcast.features.base import FeatureContext
from panelcast.features.pipeline import FeaturePipeline
from panelcast.features.temporal import TemporalBlock
from tests.integration.conftest import generate_synthetic_albums


class TestDataFeaturesIntegration:
    """Integration tests for data loading through feature extraction."""

    def test_load_clean_produces_valid_schema(self, cleaned_albums_df: pd.DataFrame):
        """Test that loading and cleaning produces valid schema.

        Verifies:
        - Required columns exist after cleaning
        - Date parsing creates date_risk column
        - No unexpected NaN in required columns
        """
        required_columns = [
            "Artist",
            "Album",
            "Year",
            "User_Score",
            "Critic_Score",
            "Release_Date_Parsed",
            "date_risk",
            "collab_type",
        ]

        for col in required_columns:
            assert col in cleaned_albums_df.columns, f"Missing required column: {col}"

        # date_risk should have valid values
        valid_risks = {"low", "medium", "high"}
        assert set(cleaned_albums_df["date_risk"].unique()).issubset(valid_risks)

        # Artist and Album should have no NaN
        assert cleaned_albums_df["Artist"].notna().all()
        assert cleaned_albums_df["Album"].notna().all()

    def test_split_preserves_temporal_integrity(self, split_datasets: dict):
        """Test that temporal split maintains chronological ordering.

        For within-artist splits:
        - Each artist should appear in train, val, and test
        - Albums should be ordered chronologically within each artist
        """
        train, val, test = split_datasets["train"], split_datasets["val"], split_datasets["test"]

        # Validate temporal ordering (no train data after test data)
        validate_temporal_split(
            train,
            val,
            test,
            artist_col="Artist",
            date_col="Release_Date_Parsed",
        )

        # Verify artists present in all splits (temporal split property)
        train_artists = set(train["Artist"])
        val_artists = set(val["Artist"])
        test_artists = set(test["Artist"])

        # All artists in test should be in train (temporal split)
        assert test_artists.issubset(train_artists)

    def test_feature_pipeline_fit_transform_shapes(
        self,
        fitted_feature_pipeline: FeaturePipeline,
        split_datasets: dict,
        feature_context: FeatureContext,
    ):
        """Test that pipeline produces consistent feature shapes.

        Verifies:
        - Output DataFrames have same columns for train/val/test
        - Feature dimensions match between splits
        """
        train_features = fitted_feature_pipeline.transform(split_datasets["train"], feature_context)
        val_features = fitted_feature_pipeline.transform(split_datasets["val"], feature_context)
        test_features = fitted_feature_pipeline.transform(split_datasets["test"], feature_context)

        # Same feature columns
        assert set(train_features.data.columns) == set(val_features.data.columns)
        assert set(train_features.data.columns) == set(test_features.data.columns)

        # Expected number of features from 3 blocks:
        # TemporalBlock: 5 features
        # ArtistHistoryBlock: 9 features
        # AlbumTypeBlock: varies by training vocab
        assert len(train_features.data.columns) >= 14  # At minimum

    def test_feature_pipeline_handles_unknown_artists(self, feature_context: FeatureContext):
        """Test that pipeline handles artists not seen during training.

        Creates a new artist in test data that wasn't in training.
        The pipeline should handle gracefully (impute or use defaults).
        """
        # Create training data with known artists
        train_df = generate_synthetic_albums(n_artists=5, albums_per_artist=4, seed=100)

        # Create test data with a NEW artist not in training
        test_df = generate_synthetic_albums(n_artists=1, albums_per_artist=2, seed=200)
        test_df["Artist"] = "Completely_New_Artist"

        # Fit pipeline on train
        blocks = [TemporalBlock(), ArtistHistoryBlock(), AlbumTypeBlock()]
        pipeline = FeaturePipeline(blocks)
        pipeline.fit(train_df, feature_context)

        # Transform test - should not raise error
        test_features = pipeline.transform(test_df, feature_context)

        # Features should be produced (with imputed values for unknown artist)
        assert test_features.data is not None
        assert len(test_features.data) == len(test_df)

        # First album of unknown artist should be debut
        # (subsequent albums have prior history from that artist)
        assert test_features.data["is_debut"].sum() >= 1

        # user_prior_mean should have valid values (imputed, not NaN)
        assert test_features.data["user_prior_mean"].notna().all()

    def test_feature_pipeline_handles_unknown_album_types(self, feature_context: FeatureContext):
        """Test that pipeline handles album types not seen during training.

        Unknown album types should get all-zero one-hot encoding.
        """
        # Create training data with limited album types
        train_df = generate_synthetic_albums(n_artists=5, albums_per_artist=4, seed=300)
        train_df["Album_Type"] = "Album"  # Only "Album" type in training

        # Create test data with unseen album type
        test_df = generate_synthetic_albums(n_artists=2, albums_per_artist=2, seed=301)
        test_df["Album_Type"] = "LiveAlbum"  # Not in training vocabulary

        # Fit and transform
        blocks = [AlbumTypeBlock()]
        pipeline = FeaturePipeline(blocks)
        pipeline.fit(train_df, feature_context)
        test_features = pipeline.transform(test_df, feature_context)

        # Should only have is_album column (the only type in training)
        assert "is_album" in test_features.data.columns

        # Unknown type should produce all zeros for one-hot columns
        one_hot_cols = [c for c in test_features.data.columns if c.startswith("is_")]
        for col in one_hot_cols:
            assert (test_features.data[col] == 0).all()

    def test_artist_history_uses_leave_one_out(self, feature_context: FeatureContext):
        """Test that ArtistHistoryBlock uses leave-one-out correctly.

        Manually verify LOO calculation for an artist with known history.
        """
        # Create data with known scores for one artist
        df = pd.DataFrame(
            {
                "Artist": ["TestArtist"] * 4,
                "Album": ["A", "B", "C", "D"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01"]
                ),
                "Year": [2010, 2011, 2012, 2013],
                "User_Score": [60.0, 70.0, 80.0, 90.0],
                "Critic_Score": [65.0, 75.0, 85.0, 95.0],
                "date_risk": ["low"] * 4,
                "Album_Type": ["Album"] * 4,
            }
        )

        block = ArtistHistoryBlock()
        block.fit(df, feature_context)
        output = block.transform(df, feature_context)

        # Sort by date to get correct order
        result = output.data.loc[df.sort_values("Release_Date_Parsed").index]

        # Album A (debut): is_debut=1, prior_mean should be imputed
        album_a = result.iloc[0]
        assert album_a["is_debut"] == 1

        # Album B (second): prior_mean = 60 (only album A)
        album_b = result.iloc[1]
        assert album_b["is_debut"] == 0
        assert album_b["user_prior_mean"] == pytest.approx(60.0)

        # Album C (third): prior_mean = (60+70)/2 = 65
        album_c = result.iloc[2]
        assert album_c["user_prior_mean"] == pytest.approx(65.0)

        # Album D (fourth): prior_mean = (60+70+80)/3 = 70
        album_d = result.iloc[3]
        assert album_d["user_prior_mean"] == pytest.approx(70.0)

    def test_temporal_features_correct_ordering(self, feature_context: FeatureContext):
        """Test that TemporalBlock computes correct album sequences.

        Verify:
        - album_sequence starts at 1 for each artist
        - career_years is non-negative
        - release_gap_days is 0 for debuts
        """
        # Create data with known release dates
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B"],
                "Album": ["A1", "A2", "A3", "B1", "B2"],
                "Release_Date_Parsed": pd.to_datetime(
                    [
                        "2010-01-01",
                        "2012-01-01",
                        "2015-01-01",  # Artist A
                        "2011-06-01",
                        "2013-06-01",  # Artist B
                    ]
                ),
                "Year": [2010, 2012, 2015, 2011, 2013],
                "date_risk": ["low"] * 5,
            }
        )

        block = TemporalBlock()
        block.fit(df, feature_context)
        output = block.transform(df, feature_context)

        # Check sequences (sorted by date within artist)
        # Artist A albums should have sequence 1, 2, 3
        # Need to find correct indices after sorting
        a1_idx = df[df["Album"] == "A1"].index[0]
        a2_idx = df[df["Album"] == "A2"].index[0]
        a3_idx = df[df["Album"] == "A3"].index[0]
        b1_idx = df[df["Album"] == "B1"].index[0]
        b2_idx = df[df["Album"] == "B2"].index[0]

        assert output.data.loc[a1_idx, "album_sequence"] == 1
        assert output.data.loc[a2_idx, "album_sequence"] == 2
        assert output.data.loc[a3_idx, "album_sequence"] == 3
        assert output.data.loc[b1_idx, "album_sequence"] == 1
        assert output.data.loc[b2_idx, "album_sequence"] == 2

        # Career years should be non-negative
        assert (output.data["career_years"] >= 0).all()

        # First albums (A1, B1) should have release_gap_days = 0
        assert output.data.loc[a1_idx, "release_gap_days"] == 0
        assert output.data.loc[b1_idx, "release_gap_days"] == 0

        # A2 and A3 should have positive release gaps
        assert output.data.loc[a2_idx, "release_gap_days"] > 0
        assert output.data.loc[a3_idx, "release_gap_days"] > 0

    def test_pipeline_no_leakage_after_fit(
        self,
        fitted_feature_pipeline: FeaturePipeline,
        split_datasets: dict,
        feature_context: FeatureContext,
    ):
        """Test that transform doesn't modify fitted state.

        The pipeline's fitted state should remain unchanged
        after multiple transform calls on different data.
        """
        # Get fitted state before transforms
        album_type_block = fitted_feature_pipeline.blocks[2]  # AlbumTypeBlock
        vocab_before = album_type_block._vocabulary_.categories

        # Transform multiple times on different data
        fitted_feature_pipeline.transform(split_datasets["train"], feature_context)
        fitted_feature_pipeline.transform(split_datasets["val"], feature_context)
        fitted_feature_pipeline.transform(split_datasets["test"], feature_context)

        # Create new data with different album types and transform
        new_df = generate_synthetic_albums(n_artists=3, albums_per_artist=2, seed=999)
        new_df["Album_Type"] = "NewType"  # Not in vocabulary
        fitted_feature_pipeline.transform(new_df, feature_context)

        # Vocabulary should be unchanged
        vocab_after = album_type_block._vocabulary_.categories
        assert vocab_before == vocab_after

    def test_feature_output_metadata_complete(
        self,
        fitted_feature_pipeline: FeaturePipeline,
        split_datasets: dict,
        feature_context: FeatureContext,
    ):
        """Test that feature output includes complete metadata.

        Each block's metadata should be preserved in the pipeline output.
        """
        output = fitted_feature_pipeline.transform(split_datasets["train"], feature_context)

        assert "blocks" in output.metadata
        assert (
            len(output.metadata["blocks"]) == 3
        )  # TemporalBlock, ArtistHistoryBlock, AlbumTypeBlock

        # Each block should have recorded its metadata
        for block_meta in output.metadata["blocks"]:
            assert "block" in block_meta

    def test_artist_disjoint_split_no_overlap(self):
        """Test artist-disjoint split has no artist overlap.

        Uses assert_no_artist_overlap to verify the split property.
        """
        from panelcast.data.split import artist_disjoint_split

        df = generate_synthetic_albums(n_artists=20, albums_per_artist=4, seed=500)
        train, val, test = artist_disjoint_split(
            df,
            artist_col="Artist",
            test_size=0.15,
            val_size=0.15,
            random_state=42,
        )

        # Should not raise if splits are disjoint
        assert_no_artist_overlap(train, val, test, artist_col="Artist")

        # Verify no shared artists
        train_artists = set(train["Artist"])
        val_artists = set(val["Artist"])
        test_artists = set(test["Artist"])

        assert len(train_artists & val_artists) == 0
        assert len(train_artists & test_artists) == 0
        assert len(val_artists & test_artists) == 0
