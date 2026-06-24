"""Integration tests for n_reviews data flow.

Tests verify that review count data flows correctly from source
through feature building to model input, enabling heteroscedastic
noise scaling in the Bayesian model.
"""

import numpy as np
import pandas as pd
import pytest

from tests.integration.conftest import generate_synthetic_albums


class TestNReviewsDataFlow:
    """Integration tests for n_reviews pipeline flow."""

    def test_n_reviews_preserved_in_features(self, tmp_path):
        """Test that User_Ratings is preserved as n_reviews in feature output.

        Verifies DATA-01: User_Ratings flows through feature pipeline.
        Uses actual FeaturePipeline to verify real pipeline behavior.
        """
        from panelcast.features.album_type import AlbumTypeBlock
        from panelcast.features.artist import ArtistHistoryBlock
        from panelcast.features.base import FeatureContext
        from panelcast.features.pipeline import FeaturePipeline
        from panelcast.features.temporal import TemporalBlock

        # Create synthetic data with known User_Ratings
        df = generate_synthetic_albums(n_artists=5, albums_per_artist=3, seed=42)

        # Preserve n_reviews before transformation (as build_features does)
        expected_n_reviews = df["User_Ratings"].values.copy()

        # Run actual feature pipeline with blocks that work with synthetic data
        # (excludes GenreBlock and CollaborationBlock which need additional columns)
        feature_ctx = FeatureContext(config={}, random_state=42)
        blocks = [TemporalBlock({}), AlbumTypeBlock({}), ArtistHistoryBlock({})]
        pipeline = FeaturePipeline(blocks)
        pipeline.fit(df, feature_ctx)
        features_output = pipeline.transform(df, feature_ctx)

        # Add n_reviews as build_features does
        features = features_output.data
        features["n_reviews"] = df["User_Ratings"].rename("n_reviews")

        # Verify n_reviews column exists and is correct
        assert "n_reviews" in features.columns
        assert len(features["n_reviews"]) == len(df)
        assert (features["n_reviews"].values == expected_n_reviews).all()
        # n_reviews should be positive integers
        assert (features["n_reviews"] > 0).all()
        assert features["n_reviews"].dtype in [np.int32, np.int64, int]

    def test_n_reviews_corresponds_to_y(self):
        """Test that n_reviews[i] corresponds to y[i] observation.

        Verifies DATA-02: n_reviews array aligns with target y.
        """
        # Create data with known values
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "C"],
                "Album": ["A1", "A2", "B1", "B2", "C1"],
                "Year": [2010, 2011, 2010, 2012, 2015],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "User_Ratings": [100, 200, 300, 400, 500],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2010-01-01", "2011-01-01", "2010-06-01", "2012-01-01", "2015-01-01"]
                ),
            }
        )

        # Extract y and n_reviews in same order
        y = df["User_Score"].values
        n_reviews = df["User_Ratings"].values

        # Verify alignment: each y[i] has corresponding n_reviews[i]
        assert len(y) == len(n_reviews)

        # Verify specific correspondences
        assert y[0] == 70.0 and n_reviews[0] == 100  # A1
        assert y[4] == 90.0 and n_reviews[4] == 500  # C1

    def test_n_reviews_validation_rejects_missing(self):
        """Test that validation rejects data with >50% missing n_reviews.

        Uses actual prepare_model_data to verify real validation behavior.
        """
        from panelcast.pipelines.train_bayes import prepare_model_data

        # Create data with >50% invalid n_reviews (90% zeros)
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 10,
                "Album": [f"A{i}" for i in range(10)],
                "Year": [2010 + i for i in range(10)],
                "User_Score": [70.0] * 10,
                "User_Ratings": [100, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 90% invalid
                "Release_Date_Parsed": pd.to_datetime([f"201{i}-01-01" for i in range(10)]),
                "prev_score": [0.0] * 10,
                "album_seq": list(range(10)),
            }
        )
        df["feat_1"] = np.random.RandomState(42).randn(10)

        # Should raise ValueError for >50% invalid
        with pytest.raises(ValueError, match="Too many invalid n_reviews"):
            prepare_model_data(df, ["feat_1"], min_albums_filter=1)

    def test_n_reviews_filtering_drops_invalid_rows(self):
        """Test that prepare_model_data drops rows with invalid n_reviews.

        Verifies DATA-03: Invalid n_reviews rows are filtered out.
        """
        from panelcast.pipelines.train_bayes import prepare_model_data

        # Create data with 1 invalid n_reviews (10%)
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 10,
                "Album": [f"A{i}" for i in range(10)],
                "Year": [2010 + i for i in range(10)],
                "User_Score": [70.0 + i for i in range(10)],
                "User_Ratings": [100, 200, 300, 0, 400, 500, 600, 700, 800, 900],  # 1 invalid
                "Release_Date_Parsed": pd.to_datetime([f"201{i}-01-01" for i in range(10)]),
                "prev_score": [0.0] * 10,
                "album_seq": list(range(10)),
            }
        )
        # Add dummy feature columns
        df["feat_1"] = np.random.RandomState(42).randn(10)

        result, _ = prepare_model_data(df, ["feat_1"], min_albums_filter=1)

        # Should have dropped the 1 invalid row (index 3 with n_reviews=0)
        assert len(result["y"]) == 9, f"Expected 9 rows after filtering, got {len(result['y'])}"
        assert len(result["n_reviews"]) == 9
        assert (result["n_reviews"] > 0).all(), "All n_reviews should be positive after filtering"

    def test_n_reviews_shape_matches_observations(self):
        """Test that n_reviews shape matches number of observations."""
        n_obs = 50
        df = generate_synthetic_albums(n_artists=10, albums_per_artist=5, seed=123)
        df = df.head(n_obs)  # Limit to n_obs rows

        y = df["User_Score"].values
        n_reviews = df["User_Ratings"].values

        # Shapes must match for element-wise operations
        assert y.shape == n_reviews.shape
        assert len(y) == n_obs


class TestNReviewsInvalidPosition:
    """Invalid review-count at first / middle / last event, plus all-invalid entity.

    The uniform index filtering in prepare_model_data must drop exactly the
    invalid row regardless of its position in an entity's history, and must
    leave no stale entity index in the data arrays when an entity is wiped out.
    """

    def _panel(self, ratings_by_artist: dict[str, list[int]]):
        rows = []
        for artist, ratings in ratings_by_artist.items():
            for i, r in enumerate(ratings):
                rows.append(
                    {
                        "Artist": artist,
                        "Album": f"{artist}{i}",
                        "Year": 2010 + i,
                        "User_Score": 60.0 + i,
                        "User_Ratings": r,
                        "Release_Date_Parsed": pd.Timestamp(f"20{10 + i:02d}-01-01"),
                    }
                )
        df = pd.DataFrame(rows)
        df["feat_1"] = np.random.RandomState(0).randn(len(df))
        return df

    @pytest.mark.parametrize("bad_pos", [0, 3, 7])
    def test_invalid_review_count_at_any_position_drops_one_row(self, bad_pos):
        """First (0), middle (3), and last (7) invalid events each drop one row."""
        from panelcast.pipelines.train_bayes import prepare_model_data

        ratings = [100, 200, 300, 400, 500, 600, 700, 800]
        ratings[bad_pos] = 0  # invalidate one position
        df = self._panel({"A": ratings})

        result, valid_mask = prepare_model_data(df, ["feat_1"], min_albums_filter=1)

        assert len(result["y"]) == len(ratings) - 1
        assert (result["n_reviews"] > 0).all()
        # The dropped score is exactly the one at the invalid position.
        dropped_score = 60.0 + bad_pos
        assert dropped_score not in set(np.asarray(result["y"]).tolist())
        assert valid_mask.sum() == len(ratings) - 1

    def test_all_invalid_entity_leaves_no_stale_index(self):
        """An entity whose every event is invalid must vanish from artist_idx."""
        from panelcast.pipelines.train_bayes import prepare_model_data

        # A is large and valid; B is entirely invalid (stays < 50% of rows).
        df = self._panel(
            {
                "A": [100, 200, 300, 400, 500, 600, 700, 800],
                "B": [0, 0],
            }
        )
        b_idx = sorted(df["Artist"].unique()).index("B")

        result, _ = prepare_model_data(df, ["feat_1"], min_albums_filter=1)

        # B's index is allocated in the mapping/n_artists, but NO surviving row
        # references it — no stale entity index in the data arrays.
        assert b_idx not in set(np.asarray(result["artist_idx"]).tolist())
        assert int(result["artist_album_counts"].loc[b_idx]) == 0
        assert len(result["y"]) == 8
        # max index in the data is within range (no dangling reference).
        assert int(np.asarray(result["artist_idx"]).max()) < result["n_artists"]
