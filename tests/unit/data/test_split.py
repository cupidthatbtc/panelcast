"""Unit tests for temporal/disjoint split utilities."""

import pandas as pd
import pytest

from panelcast.data.split import (
    entity_disjoint_split,
    assert_no_artist_overlap,
    validate_temporal_split,
    within_entity_temporal_split,
)

# =============================================================================
# within_entity_temporal_split tests
# =============================================================================


class TestWithinArtistTemporalSplit:
    """Tests for within_entity_temporal_split."""

    def test_requires_date_column(self):
        df = pd.DataFrame({"Artist": ["A", "A", "A"], "Album": ["a1", "a2", "a3"]})
        with pytest.raises(ValueError, match="Missing required date column"):
            within_entity_temporal_split(df, date_col="Release_Date_Parsed")

    def test_places_missing_dates_in_train(self):
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A"],
                "Album": ["a0", "a1", "a2"],
                "Release_Date_Parsed": pd.to_datetime([None, "2020-01-01", "2021-01-01"]),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        assert len(train) == 1
        assert train["Release_Date_Parsed"].isna().all()
        assert val["Release_Date_Parsed"].iloc[0] == pd.Timestamp("2020-01-01")
        assert test["Release_Date_Parsed"].iloc[0] == pd.Timestamp("2021-01-01")

    def test_excludes_artists_with_all_missing_dates(self):
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "B"],
                "Album": ["a0", "a1", "a2", "b0", "b1", "b2"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2019-01-01", "2020-01-01", "2021-01-01", None, None, None]
                ),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        all_artists = set(pd.concat([train["Artist"], val["Artist"], test["Artist"]]))
        assert all_artists == {"A"}

    def test_basic_split_sizes(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 5,
                "Album": [f"a{i}" for i in range(5)],
                "Release_Date_Parsed": pd.date_range("2018", periods=5, freq="YS"),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        assert len(test) == 1
        assert len(val) == 1
        assert len(train) == 3

    def test_test_has_latest_albums(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 4,
                "Album": ["old1", "old2", "recent", "latest"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]
                ),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        assert test.iloc[0]["Album"] == "latest"
        assert val.iloc[0]["Album"] == "recent"

    def test_excludes_artists_below_min_required(self):
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B"],
                "Album": ["a0", "a1", "a2", "b0", "b1"],
                "Release_Date_Parsed": pd.date_range("2018", periods=5, freq="YS")[:5],
            }
        )
        # B has only 2 albums, needs 3 (1+1+1)
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        all_artists = set(pd.concat([train["Artist"], val["Artist"], test["Artist"]]))
        assert "B" not in all_artists

    def test_multiple_artists(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 3 + ["B"] * 4,
                "Album": [f"album_{i}" for i in range(7)],
                "Release_Date_Parsed": pd.date_range("2018", periods=7, freq="YS"),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        total = len(train) + len(val) + len(test)
        assert total == 7

    def test_no_data_leakage(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 5,
                "Album": [f"a{i}" for i in range(5)],
                "Release_Date_Parsed": pd.date_range("2018", periods=5, freq="YS"),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        train_max = train["Release_Date_Parsed"].max()
        test_min = test["Release_Date_Parsed"].min()
        assert train_max <= test_min

    def test_custom_test_val_albums(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 6,
                "Album": [f"a{i}" for i in range(6)],
                "Release_Date_Parsed": pd.date_range("2016", periods=6, freq="YS"),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=2,
            val_albums=2,
            min_train_albums=1,
        )
        assert len(test) == 2
        assert len(val) == 2
        assert len(train) == 2

    def test_no_album_column_fallback(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 3,
                "Release_Date_Parsed": pd.date_range("2018", periods=3, freq="YS"),
            }
        )
        train, val, test = within_entity_temporal_split(
            df,
            test_albums=1,
            val_albums=1,
            min_train_albums=1,
        )
        assert len(train) + len(val) + len(test) == 3


# =============================================================================
# entity_disjoint_split tests
# =============================================================================


class TestArtistDisjointSplit:
    """Tests for entity_disjoint_split."""

    def _make_multi_artist_df(self, n_artists=20, albums_per=3):
        rows = []
        for i in range(n_artists):
            for j in range(albums_per):
                rows.append({"Artist": f"Artist_{i}", "Album": f"album_{i}_{j}", "Score": 70})
        return pd.DataFrame(rows)

    def test_no_artist_overlap(self):
        df = self._make_multi_artist_df(n_artists=20)
        train, val, test = entity_disjoint_split(df, random_state=42)
        train_a = set(train["Artist"])
        val_a = set(val["Artist"])
        test_a = set(test["Artist"])
        assert len(train_a & val_a) == 0
        assert len(train_a & test_a) == 0
        assert len(val_a & test_a) == 0

    def test_all_rows_preserved(self):
        df = self._make_multi_artist_df(n_artists=20)
        train, val, test = entity_disjoint_split(df, random_state=42)
        assert len(train) + len(val) + len(test) == len(df)

    def test_reproducible_with_same_seed(self):
        df = self._make_multi_artist_df(n_artists=20)
        t1, v1, te1 = entity_disjoint_split(df, random_state=42)
        t2, v2, te2 = entity_disjoint_split(df, random_state=42)
        assert set(t1["Artist"]) == set(t2["Artist"])
        assert set(te1["Artist"]) == set(te2["Artist"])

    def test_different_seed_different_split(self):
        df = self._make_multi_artist_df(n_artists=20)
        t1, _, te1 = entity_disjoint_split(df, random_state=42)
        t2, _, te2 = entity_disjoint_split(df, random_state=99)
        # Very unlikely to be identical with different seeds
        assert set(te1["Artist"]) != set(te2["Artist"])

    def test_approximate_proportions(self):
        df = self._make_multi_artist_df(n_artists=100, albums_per=3)
        train, val, test = entity_disjoint_split(df, test_size=0.15, val_size=0.15, random_state=42)
        total = len(df)
        assert len(test) / total > 0.05
        assert len(test) / total < 0.35
        assert len(val) / total > 0.05
        assert len(val) / total < 0.35


# =============================================================================
# assert_no_artist_overlap tests
# =============================================================================


class TestAssertNoArtistOverlap:
    """Tests for assert_no_artist_overlap."""

    def test_no_overlap_passes(self):
        train = pd.DataFrame({"Artist": ["A", "B"]})
        val = pd.DataFrame({"Artist": ["C"]})
        test = pd.DataFrame({"Artist": ["D"]})
        assert_no_artist_overlap(train, val, test)

    def test_train_val_overlap_raises(self):
        train = pd.DataFrame({"Artist": ["A", "B"]})
        val = pd.DataFrame({"Artist": ["B"]})
        test = pd.DataFrame({"Artist": ["C"]})
        with pytest.raises(ValueError, match="Entity overlap"):
            assert_no_artist_overlap(train, val, test)

    def test_train_test_overlap_raises(self):
        train = pd.DataFrame({"Artist": ["A", "B"]})
        val = pd.DataFrame({"Artist": ["C"]})
        test = pd.DataFrame({"Artist": ["A"]})
        with pytest.raises(ValueError, match="Entity overlap"):
            assert_no_artist_overlap(train, val, test)

    def test_val_test_overlap_raises(self):
        train = pd.DataFrame({"Artist": ["A"]})
        val = pd.DataFrame({"Artist": ["B"]})
        test = pd.DataFrame({"Artist": ["B"]})
        with pytest.raises(ValueError, match="Entity overlap"):
            assert_no_artist_overlap(train, val, test)

    def test_empty_splits_pass(self):
        train = pd.DataFrame({"Artist": []})
        val = pd.DataFrame({"Artist": []})
        test = pd.DataFrame({"Artist": []})
        assert_no_artist_overlap(train, val, test)


# =============================================================================
# validate_temporal_split tests
# =============================================================================


class TestValidateTemporalSplit:
    """Tests for validate_temporal_split."""

    def test_raises_when_test_dates_missing(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])})
        test = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": [pd.NaT]})
        with pytest.raises(ValueError, match="missing parsed release dates"):
            validate_temporal_split(train, val, test)

    def test_allows_missing_train_dates(self):
        train = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": [pd.NaT]})
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])}
        )
        validate_temporal_split(train, val, test)

    def test_valid_ordering_passes(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2018-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])}
        )
        validate_temporal_split(train, val, test)

    def test_train_after_test_raises(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2022-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])}
        )
        with pytest.raises(ValueError, match="Temporal violation"):
            validate_temporal_split(train, val, test)

    def test_train_after_val_raises(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2022-01-01"])}
        )
        with pytest.raises(ValueError, match="Temporal violation"):
            validate_temporal_split(train, val, test)

    def test_val_after_test_raises(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2018-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2022-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])}
        )
        with pytest.raises(ValueError, match="Temporal violation"):
            validate_temporal_split(train, val, test)

    def test_same_date_allowed(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])}
        )
        validate_temporal_split(train, val, test)

    def test_non_overlapping_artists_skipped(self):
        train = pd.DataFrame(
            {"Artist": ["A"], "Release_Date_Parsed": pd.to_datetime(["2018-01-01"])}
        )
        val = pd.DataFrame({"Artist": ["B"], "Release_Date_Parsed": pd.to_datetime(["2020-01-01"])})
        test = pd.DataFrame(
            {"Artist": ["C"], "Release_Date_Parsed": pd.to_datetime(["2021-01-01"])}
        )
        validate_temporal_split(train, val, test)
