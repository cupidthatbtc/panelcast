"""Tests for ArtistHistoryBlock leave-one-out feature computation.

These tests verify the critical LOO pattern that prevents data leakage:
each album only sees prior albums from the same artist.
"""

import numpy as np
import pandas as pd
import pytest

from panelcast.features.artist import ArtistHistoryBlock, ArtistReputationBlock
from panelcast.features.base import FeatureContext, FeatureOutput
from panelcast.features.errors import NotFittedError

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """Standard feature context."""
    return FeatureContext(config={}, random_state=42)


@pytest.fixture
def single_artist_df():
    """Single artist with 3 albums: scores 70, 80, 90."""
    return pd.DataFrame(
        {
            "Artist": ["TestArtist", "TestArtist", "TestArtist"],
            "Album": ["Album1", "Album2", "Album3"],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "User_Score": [70.0, 80.0, 90.0],
            "Critic_Score": [65.0, 75.0, 85.0],
        }
    )


@pytest.fixture
def multi_artist_df():
    """Two artists with different histories."""
    return pd.DataFrame(
        {
            "Artist": ["ArtistA", "ArtistA", "ArtistA", "ArtistB", "ArtistB"],
            "Album": ["A1", "A2", "A3", "B1", "B2"],
            "Release_Date_Parsed": pd.to_datetime(
                [
                    "2020-01-01",
                    "2021-01-01",
                    "2022-01-01",  # ArtistA
                    "2019-06-01",
                    "2020-06-01",  # ArtistB
                ]
            ),
            "User_Score": [70.0, 80.0, 90.0, 60.0, 50.0],
            "Critic_Score": [75.0, 85.0, 95.0, 55.0, 45.0],
        }
    )


@pytest.fixture
def debut_only_artist_df():
    """Artist with only one album (debut)."""
    return pd.DataFrame(
        {
            "Artist": ["DebutArtist"],
            "Album": ["OnlyAlbum"],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
            "User_Score": [75.0],
            "Critic_Score": [70.0],
        }
    )


# --------------------------------------------------------------------------
# Fit/Transform Enforcement Tests
# --------------------------------------------------------------------------


class TestFitTransformEnforcement:
    """Test that fit/transform pattern is enforced."""

    def test_is_fitted_false_initially(self):
        """Block should not be fitted on instantiation."""
        block = ArtistHistoryBlock()
        assert block.is_fitted is False

    def test_fit_sets_fitted_true(self, single_artist_df, ctx):
        """fit() should set is_fitted to True."""
        block = ArtistHistoryBlock()
        block.fit(single_artist_df, ctx)
        assert block.is_fitted is True

    def test_transform_before_fit_raises_error(self, single_artist_df, ctx):
        """transform() before fit() should raise NotFittedError."""
        block = ArtistHistoryBlock()
        with pytest.raises(NotFittedError):
            block.transform(single_artist_df, ctx)

    def test_fit_stores_global_statistics(self, single_artist_df, ctx):
        """fit() should store global mean/std for imputation."""
        block = ArtistHistoryBlock()
        block.fit(single_artist_df, ctx)

        # Global user mean = (70 + 80 + 90) / 3 = 80
        assert block._global_user_mean_ == pytest.approx(80.0)
        # Global user std
        assert block._global_user_std_ == pytest.approx(np.std([70, 80, 90], ddof=1))
        # Global critic mean = (65 + 75 + 85) / 3 = 75
        assert block._global_critic_mean_ == pytest.approx(75.0)


# --------------------------------------------------------------------------
# LOO Correctness Tests (Critical for leakage prevention)
# --------------------------------------------------------------------------


class TestLOOCorrectness:
    """Test leave-one-out excludes current album from prior statistics."""

    def test_loo_excludes_current_album(self, single_artist_df, ctx):
        """The key LOO test: current album must NOT be in prior mean.

        Artist with scores [70, 80, 90]:
        - Album1: prior_mean = NaN (debut, no prior)
        - Album2: prior_mean = 70 (only sees Album1)
        - Album3: prior_mean = 75 (sees Albums 1 & 2: mean(70, 80) = 75)

        If LOO is wrong (includes current), we'd see:
        - Album2: prior_mean = 75 (WRONG - includes self)
        - Album3: prior_mean = 80 (WRONG - includes self)
        """
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        # Sort by date to match expected order
        df_with_features = single_artist_df.copy()
        df_with_features["user_prior_mean"] = output.data["user_prior_mean"]
        df_sorted = df_with_features.sort_values("Release_Date_Parsed")

        prior_means = df_sorted["user_prior_mean"].tolist()

        # Album1 (debut) should be imputed with global mean (80)
        assert prior_means[0] == pytest.approx(80.0)
        # Album2 should see only Album1: 70
        assert prior_means[1] == pytest.approx(70.0)
        # Album3 should see Albums 1 & 2: (70 + 80) / 2 = 75
        assert prior_means[2] == pytest.approx(75.0)

    def test_loo_second_album_only_sees_first(self, single_artist_df, ctx):
        """Album 2's prior_mean should exactly equal Album 1's score."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # Album1 score is 70, Album2's prior_mean should be 70
        album1_score = df_sorted.iloc[0]["User_Score"]
        album2_prior_mean = result.iloc[1]["user_prior_mean"]
        assert album2_prior_mean == pytest.approx(album1_score)

    def test_loo_third_album_sees_mean_of_first_two(self, single_artist_df, ctx):
        """Album 3's prior_mean should be mean of Album 1 and Album 2 scores."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # Albums 1 & 2 scores: 70, 80 -> mean = 75
        expected_mean = np.mean([70.0, 80.0])
        album3_prior_mean = result.iloc[2]["user_prior_mean"]
        assert album3_prior_mean == pytest.approx(expected_mean)


# --------------------------------------------------------------------------
# Debut Handling Tests
# --------------------------------------------------------------------------


class TestDebutHandling:
    """Test correct handling of debut (first) albums."""

    def test_debut_has_nan_before_imputation(self, single_artist_df, ctx):
        """Internally, debut should have NaN prior count before imputation."""
        # We test this indirectly through is_debut flag
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # First album (Album1) should be marked as debut
        assert result.iloc[0]["is_debut"] == 1
        # Second and third should not be debuts
        assert result.iloc[1]["is_debut"] == 0
        assert result.iloc[2]["is_debut"] == 0

    def test_debut_imputed_with_global_mean(self, single_artist_df, ctx):
        """Debut albums should have prior_mean = global mean from training."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # Global user mean = (70 + 80 + 90) / 3 = 80
        debut_prior_mean = result.iloc[0]["user_prior_mean"]
        assert debut_prior_mean == pytest.approx(80.0)

        # Global critic mean = (65 + 75 + 85) / 3 = 75
        debut_critic_prior_mean = result.iloc[0]["critic_prior_mean"]
        assert debut_critic_prior_mean == pytest.approx(75.0)

    def test_is_debut_flag_correct(self, multi_artist_df, ctx):
        """is_debut should be 1 for first album of each artist, 0 otherwise."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(multi_artist_df, ctx)

        # Merge to check per album
        df_with_features = multi_artist_df.copy()
        df_with_features["is_debut"] = output.data["is_debut"]

        # ArtistA's first album (A1) should be debut
        a1 = df_with_features[df_with_features["Album"] == "A1"]["is_debut"].iloc[0]
        assert a1 == 1

        # ArtistA's subsequent albums should not be debuts
        a2 = df_with_features[df_with_features["Album"] == "A2"]["is_debut"].iloc[0]
        a3 = df_with_features[df_with_features["Album"] == "A3"]["is_debut"].iloc[0]
        assert a2 == 0
        assert a3 == 0

        # ArtistB's first album (B1) should be debut
        b1 = df_with_features[df_with_features["Album"] == "B1"]["is_debut"].iloc[0]
        assert b1 == 1

        # ArtistB's second album (B2) should not be debut
        b2 = df_with_features[df_with_features["Album"] == "B2"]["is_debut"].iloc[0]
        assert b2 == 0


# --------------------------------------------------------------------------
# Trajectory Tests
# --------------------------------------------------------------------------


class TestTrajectory:
    """Test trajectory (slope) computation."""

    def test_trajectory_nan_for_single_prior(self, single_artist_df, ctx):
        """Trajectory requires 2+ prior albums, so should be 0 (imputed) for debut and second album."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # Album1 (debut): no prior -> trajectory = 0 (imputed)
        assert result.iloc[0]["user_trajectory"] == pytest.approx(0.0)
        # Album2: only 1 prior -> trajectory = 0 (imputed from NaN)
        assert result.iloc[1]["user_trajectory"] == pytest.approx(0.0)

    def test_trajectory_positive_for_improving(self, single_artist_df, ctx):
        """Artist improving (70 -> 80 -> 90) should have positive trajectory by Album 3."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # Album3 sees [70, 80] -> positive slope
        trajectory = result.iloc[2]["user_trajectory"]
        assert trajectory > 0

    def test_trajectory_negative_for_declining(self, ctx):
        """Artist declining should have negative trajectory."""
        df = pd.DataFrame(
            {
                "Artist": ["Declining", "Declining", "Declining"],
                "Album": ["D1", "D2", "D3"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
                "User_Score": [90.0, 80.0, 70.0],
                "Critic_Score": [85.0, 75.0, 65.0],
            }
        )

        block = ArtistHistoryBlock()
        output = block.fit_transform(df, ctx)

        df_sorted = df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # D3 sees [90, 80] -> negative slope
        trajectory = result.iloc[2]["user_trajectory"]
        assert trajectory < 0

    def test_trajectory_zero_for_flat(self, ctx):
        """Artist with constant scores should have zero trajectory."""
        df = pd.DataFrame(
            {
                "Artist": ["Flat", "Flat", "Flat"],
                "Album": ["F1", "F2", "F3"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
                "User_Score": [80.0, 80.0, 80.0],
                "Critic_Score": [75.0, 75.0, 75.0],
            }
        )

        block = ArtistHistoryBlock()
        output = block.fit_transform(df, ctx)

        df_sorted = df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # F3 sees [80, 80] -> zero slope
        trajectory = result.iloc[2]["user_trajectory"]
        assert trajectory == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Multi-Artist Tests
# --------------------------------------------------------------------------


class TestMultiArtist:
    """Test that artist histories are independent."""

    def test_multiple_artists_independent_histories(self, multi_artist_df, ctx):
        """Each artist's history should be computed independently."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(multi_artist_df, ctx)

        df_with_features = multi_artist_df.copy()
        df_with_features["user_prior_mean"] = output.data["user_prior_mean"]

        # ArtistA: scores [70, 80, 90]
        # A2 should see 70, A3 should see 75
        a2_prior = df_with_features[df_with_features["Album"] == "A2"]["user_prior_mean"].iloc[0]
        a3_prior = df_with_features[df_with_features["Album"] == "A3"]["user_prior_mean"].iloc[0]
        assert a2_prior == pytest.approx(70.0)
        assert a3_prior == pytest.approx(75.0)

        # ArtistB: scores [60, 50]
        # B2 should see 60 (only B1)
        b2_prior = df_with_features[df_with_features["Album"] == "B2"]["user_prior_mean"].iloc[0]
        assert b2_prior == pytest.approx(60.0)

    def test_artists_dont_leak_to_each_other(self, multi_artist_df, ctx):
        """ArtistB's history should not include ArtistA's scores."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(multi_artist_df, ctx)

        df_with_features = multi_artist_df.copy()
        df_with_features["user_prior_count"] = output.data["user_prior_count"]

        # ArtistB's B2 should have prior_count = 1 (only B1)
        # If leakage occurred, it would include ArtistA's albums
        b2_count = df_with_features[df_with_features["Album"] == "B2"]["user_prior_count"].iloc[0]
        assert b2_count == 1


# --------------------------------------------------------------------------
# Edge Cases
# --------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_album_artist_all_nan_imputed(self, debut_only_artist_df, ctx):
        """Single-album artist should have all prior stats imputed."""
        # Create training data with multiple albums to get meaningful global stats
        train_df = pd.DataFrame(
            {
                "Artist": ["Other", "Other"],
                "Album": ["O1", "O2"],
                "Release_Date_Parsed": pd.to_datetime(["2019-01-01", "2020-01-01"]),
                "User_Score": [60.0, 80.0],
                "Critic_Score": [55.0, 75.0],
            }
        )

        block = ArtistHistoryBlock()
        block.fit(train_df, ctx)

        # Transform the single-album artist
        output = block.transform(debut_only_artist_df, ctx)

        # Global user mean from training = (60 + 80) / 2 = 70
        assert output.data["user_prior_mean"].iloc[0] == pytest.approx(70.0)
        assert output.data["is_debut"].iloc[0] == 1
        assert output.data["user_prior_count"].iloc[0] == 0
        assert output.data["user_trajectory"].iloc[0] == pytest.approx(0.0)

    def test_preserves_original_index(self, single_artist_df, ctx):
        """Output should preserve original DataFrame index order."""
        # Create df with non-sequential index
        df = single_artist_df.copy()
        df.index = [100, 200, 300]

        block = ArtistHistoryBlock()
        output = block.fit_transform(df, ctx)

        assert list(output.data.index) == [100, 200, 300]

    def test_deterministic_with_same_date_albums(self, ctx):
        """Albums on same date should have deterministic ordering (by Album name)."""
        df = pd.DataFrame(
            {
                "Artist": ["SameDay", "SameDay", "SameDay"],
                "Album": ["C_Album", "A_Album", "B_Album"],  # Deliberately unsorted
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
                "User_Score": [70.0, 80.0, 90.0],
                "Critic_Score": [65.0, 75.0, 85.0],
            }
        )

        block = ArtistHistoryBlock()
        output1 = block.fit_transform(df, ctx)
        output2 = block.fit_transform(df, ctx)

        # Results should be identical across runs
        pd.testing.assert_frame_equal(output1.data, output2.data)


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestIntegration:
    """Test integration and output format."""

    def test_user_and_critic_scores_independent(self, single_artist_df, ctx):
        """User and critic histories should be computed separately."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]

        # Album2: user_prior_mean = 70, critic_prior_mean = 65
        assert result.iloc[1]["user_prior_mean"] == pytest.approx(70.0)
        assert result.iloc[1]["critic_prior_mean"] == pytest.approx(65.0)

        # Album3: user_prior_mean = 75, critic_prior_mean = 70
        assert result.iloc[2]["user_prior_mean"] == pytest.approx(75.0)
        assert result.iloc[2]["critic_prior_mean"] == pytest.approx(70.0)

    def test_output_has_all_expected_columns(self, single_artist_df, ctx):
        """Output should have all 9 expected feature columns."""
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)

        expected_cols = [
            "user_prior_mean",
            "user_prior_std",
            "user_prior_count",
            "user_trajectory",
            "critic_prior_mean",
            "critic_prior_std",
            "critic_prior_count",
            "critic_trajectory",
            "is_debut",
        ]

        assert output.feature_names == expected_cols
        assert list(output.data.columns) == expected_cols

    def test_fit_transform_equals_fit_then_transform(self, single_artist_df, ctx):
        """fit_transform should produce same results as fit() then transform()."""
        block1 = ArtistHistoryBlock()
        output1 = block1.fit_transform(single_artist_df, ctx)

        block2 = ArtistHistoryBlock()
        block2.fit(single_artist_df, ctx)
        output2 = block2.transform(single_artist_df, ctx)

        pd.testing.assert_frame_equal(output1.data, output2.data)


# --------------------------------------------------------------------------
# Backwards Compatibility
# --------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """Test backwards compatibility with old name."""

    def test_artist_reputation_block_alias_exists(self):
        """ArtistReputationBlock should be an alias for ArtistHistoryBlock."""
        assert ArtistReputationBlock is ArtistHistoryBlock

    def test_artist_reputation_block_works(self, single_artist_df, ctx):
        """ArtistReputationBlock should work identically to ArtistHistoryBlock."""
        block = ArtistReputationBlock()
        output = block.fit_transform(single_artist_df, ctx)
        assert isinstance(output, FeatureOutput)
        assert len(output.feature_names) == 9


# --------------------------------------------------------------------------
# Additional artist history tests
# --------------------------------------------------------------------------


class TestArtistHistoryBlockAttributes:
    """Tests for block attributes and initialization."""

    def test_name_is_artist_history(self):
        block = ArtistHistoryBlock()
        assert block.name == "artist_history"

    def test_requires_is_empty(self):
        block = ArtistHistoryBlock()
        assert block.requires == []

    def test_required_columns(self):
        block = ArtistHistoryBlock()
        assert "Artist" in block.required_columns
        assert "Release_Date_Parsed" in block.required_columns
        assert "User_Score" in block.required_columns
        assert "Critic_Score" in block.required_columns
        assert "Album" in block.required_columns

    def test_default_params_empty(self):
        block = ArtistHistoryBlock()
        assert block.params == {}


class TestArtistHistoryMissingColumns:
    """Tests for missing column validation."""

    def test_fit_raises_on_missing_columns(self, ctx):
        df = pd.DataFrame({"Artist": ["A"], "Album": ["1"]})
        block = ArtistHistoryBlock()
        with pytest.raises(ValueError, match="missing required columns"):
            block.fit(df, ctx)

    def test_transform_raises_on_missing_columns(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        block.fit(single_artist_df, ctx)
        df_bad = pd.DataFrame({"Artist": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            block.transform(df_bad, ctx)


class TestArtistHistoryGlobalStats:
    """Tests for global statistics computation."""

    def test_global_critic_std_stored(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        block.fit(single_artist_df, ctx)
        assert block._global_critic_std_ == pytest.approx(np.std([65, 75, 85], ddof=1))

    def test_global_stats_with_single_album(self, ctx):
        df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["1"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
                "User_Score": [75.0],
                "Critic_Score": [70.0],
            }
        )
        block = ArtistHistoryBlock()
        block.fit(df, ctx)
        assert block._global_user_mean_ == 75.0
        assert block._global_critic_mean_ == 70.0


class TestArtistHistoryPriorStd:
    """Tests for prior standard deviation computation."""

    def test_prior_std_debut_imputed_with_global(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)
        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]
        # Debut should have global std
        debut_std = result.iloc[0]["user_prior_std"]
        assert debut_std == pytest.approx(block._global_user_std_)

    def test_prior_count_increases(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)
        df_sorted = single_artist_df.sort_values("Release_Date_Parsed")
        result = output.data.loc[df_sorted.index]
        counts = result["user_prior_count"].tolist()
        assert counts == [0, 1, 2]


class TestArtistHistoryMetadata:
    """Tests for output metadata."""

    def test_metadata_contains_global_stats(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)
        assert "global_user_mean" in output.metadata
        assert "global_user_std" in output.metadata
        assert "global_critic_mean" in output.metadata
        assert "global_critic_std" in output.metadata

    def test_metadata_block_name(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)
        assert output.metadata["block"] == "artist_history"


class TestArtistHistoryFitTransformEquivalence:
    """Tests for equivalence of fit_transform vs fit+transform."""

    def test_output_row_count_matches_input(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)
        assert len(output.data) == len(single_artist_df)

    def test_output_has_nine_columns(self, single_artist_df, ctx):
        block = ArtistHistoryBlock()
        output = block.fit_transform(single_artist_df, ctx)
        assert output.data.shape[1] == 9
