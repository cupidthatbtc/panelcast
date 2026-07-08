"""Unit tests for cleaning pipeline edge cases."""

import pandas as pd
import pytest

from panelcast.data.cleaning import (
    RAW_TO_CANONICAL,
    CleaningConfig,
    _has_nonempty_text,
    apply_exclusion_filter,
    clean_albums,
    ensure_optional_columns,
    extract_collaboration_features,
    extract_primary_genre,
    flag_unknown_artist,
    parse_release_dates,
    rename_columns,
)
from panelcast.data.lineage import AuditLogger


def _make_raw_df(**overrides) -> pd.DataFrame:
    """Create minimal raw DataFrame for cleaning tests."""
    data = {
        "Artist": ["A"],
        "Album": ["Album 1"],
        "Year": [2020],
        "Release Date": ["January 01, 2020"],
        "Genres": ["Rock"],
        "User Score": [80.0],
        "User Ratings": [100],
        "Tracks": [10],
        "Runtime (min)": [42.0],
        "Avg Track Runtime (min)": [4.2],
        "Album Type": ["LP"],
        "All Artists": ["A"],
        "original_row_id": [0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# =============================================================================
# rename_columns tests
# =============================================================================


class TestRenameColumns:
    """Tests for rename_columns."""

    def test_renames_known_columns(self):
        df = pd.DataFrame({"Release Date": ["x"], "User Score": [1.0]})
        result = rename_columns(df)
        assert "Release_Date" in result.columns
        assert "User_Score" in result.columns

    def test_preserves_unknown_columns(self):
        df = pd.DataFrame({"Artist": ["A"], "CustomCol": [1]})
        result = rename_columns(df)
        assert "Artist" in result.columns
        assert "CustomCol" in result.columns

    def test_all_raw_mappings_applied(self):
        df = pd.DataFrame({col: ["x"] for col in RAW_TO_CANONICAL.keys()})
        result = rename_columns(df)
        for canonical in RAW_TO_CANONICAL.values():
            assert canonical in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"User Score": [1.0]})
        rename_columns(df)
        assert "User Score" in df.columns


# =============================================================================
# ensure_optional_columns tests
# =============================================================================


class TestEnsureOptionalColumns:
    """Tests for ensure_optional_columns."""

    def test_adds_missing_columns(self):
        df = pd.DataFrame({"Artist": ["A"]})
        result = ensure_optional_columns(df)
        assert "Critic_Score" in result.columns
        assert "Descriptors" in result.columns

    def test_preserves_existing_values(self):
        df = pd.DataFrame({"Artist": ["A"], "Critic_Score": [85.0]})
        result = ensure_optional_columns(df)
        assert result["Critic_Score"].iloc[0] == 85.0

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"Artist": ["A"]})
        result = ensure_optional_columns(df)
        assert "Critic_Score" not in df.columns
        assert "Critic_Score" in result.columns

    def test_fabricated_numeric_columns_are_float64(self):
        # Object-dtype pd.NA cannot be coerced by the cleaned schema's float64
        # columns; fabricated numeric optionals must be float64 NaN.
        df = pd.DataFrame({"Artist": ["A"]})
        result = ensure_optional_columns(df)
        for col in ("Critic_Score", "Critic_Reviews", "Avg_Track_Score"):
            assert result[col].dtype == "float64"
            assert result[col].isna().all()

    def test_fabricated_text_columns_stay_object(self):
        df = pd.DataFrame({"Artist": ["A"]})
        result = ensure_optional_columns(df)
        for col in ("Label", "Descriptors", "Album_URL"):
            assert result[col].dtype == "object"


# =============================================================================
# parse_release_dates tests
# =============================================================================


class TestParseReleaseDates:
    """Tests for parse_release_dates."""

    def test_valid_date_low_risk(self):
        df = pd.DataFrame({"Release_Date": ["January 01, 2020"], "Year": [2020.0]})
        result = parse_release_dates(df)
        assert result.loc[0, "date_risk"] == "low"
        assert result.loc[0, "date_imputation_type"] == "none"
        assert result.loc[0, "Release_Date_Parsed"] == pd.Timestamp("2020-01-01")

    def test_missing_date_with_year_medium_risk(self):
        df = pd.DataFrame({"Release_Date": [None], "Year": [2020.0]})
        result = parse_release_dates(df)
        assert result.loc[0, "date_risk"] == "medium"
        assert result.loc[0, "date_imputation_type"] == "jan1"
        assert result.loc[0, "Release_Date_Parsed"] == pd.Timestamp("2020-01-01")

    def test_unparseable_without_year_high_risk(self):
        df = pd.DataFrame({"Release_Date": ["not-a-date"], "Year": [pd.NA]})
        result = parse_release_dates(df)
        assert pd.isna(result.loc[0, "Release_Date_Parsed"])
        assert result.loc[0, "date_risk"] == "high"
        # Tier-3 rows are never imputed; label must say so truthfully.
        assert result.loc[0, "date_imputation_type"] == "unimputed"
        assert bool(result.loc[0, "date_missing"])

    def test_date_missing_false_for_parsed_and_jan1_rows(self):
        df = pd.DataFrame(
            {
                "Release_Date": ["April 10, 2018", None],
                "Year": [2018.0, 2019.0],
            }
        )
        result = parse_release_dates(df)
        assert not result["date_missing"].any()

    def test_flag_future_year(self):
        df = pd.DataFrame({"Release_Date": [None], "Year": [2099.0]})
        result = parse_release_dates(df, max_year=2026)
        assert bool(result.loc[0, "flag_future_year"])

    def test_flag_sparse_era(self):
        df = pd.DataFrame({"Release_Date": [None], "Year": [1940.0]})
        result = parse_release_dates(df, min_year=1950)
        assert bool(result.loc[0, "flag_sparse_era"])

    def test_year_within_range_no_flags(self):
        df = pd.DataFrame({"Release_Date": ["March 15, 2020"], "Year": [2020.0]})
        result = parse_release_dates(df, min_year=1950, max_year=2026)
        assert not bool(result.loc[0, "flag_future_year"])
        assert not bool(result.loc[0, "flag_sparse_era"])

    def test_multiple_rows_mixed_risks(self):
        df = pd.DataFrame(
            {
                "Release_Date": ["January 01, 2020", None, "bad"],
                "Year": [2020.0, 2021.0, pd.NA],
            }
        )
        result = parse_release_dates(df)
        assert result.loc[0, "date_risk"] == "low"
        assert result.loc[1, "date_risk"] == "medium"
        assert result.loc[2, "date_risk"] == "high"


# =============================================================================
# extract_collaboration_features tests
# =============================================================================


class TestExtractCollaborationFeatures:
    """Tests for extract_collaboration_features."""

    def test_solo_artist(self):
        df = pd.DataFrame({"All_Artists": ["Artist A"], "Artist": ["Artist A"]})
        result = extract_collaboration_features(df)
        assert result.loc[0, "num_artists"] == 1
        assert not bool(result.loc[0, "is_collaboration"])
        assert result.loc[0, "collab_type"] == "solo"

    def test_duo(self):
        df = pd.DataFrame({"All_Artists": ["Artist A | Artist B"], "Artist": ["Artist A"]})
        result = extract_collaboration_features(df)
        assert result.loc[0, "num_artists"] == 2
        assert bool(result.loc[0, "is_collaboration"])
        assert result.loc[0, "collab_type"] == "duo"

    def test_small_group(self):
        df = pd.DataFrame({"All_Artists": ["A | B | C"], "Artist": ["A"]})
        result = extract_collaboration_features(df)
        assert result.loc[0, "num_artists"] == 3
        assert result.loc[0, "collab_type"] == "small_group"

    def test_four_artists_small_group(self):
        df = pd.DataFrame({"All_Artists": ["A | B | C | D"], "Artist": ["A"]})
        result = extract_collaboration_features(df)
        assert result.loc[0, "num_artists"] == 4
        assert result.loc[0, "collab_type"] == "small_group"

    def test_ensemble(self):
        df = pd.DataFrame({"All_Artists": ["A | B | C | D | E"], "Artist": ["A"]})
        result = extract_collaboration_features(df)
        assert result.loc[0, "num_artists"] == 5
        assert result.loc[0, "collab_type"] == "ensemble"

    def test_nan_all_artists(self):
        df = pd.DataFrame({"All_Artists": [pd.NA], "Artist": ["A"]})
        result = extract_collaboration_features(df)
        assert result.loc[0, "num_artists"] == 1
        assert result.loc[0, "collab_type"] == "solo"

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"All_Artists": ["A"], "Artist": ["A"]})
        extract_collaboration_features(df)
        assert "num_artists" not in df.columns


# =============================================================================
# extract_primary_genre tests
# =============================================================================


class TestExtractPrimaryGenre:
    """Tests for extract_primary_genre."""

    def test_single_genre(self):
        df = pd.DataFrame({"Genres": ["Rock"]})
        result = extract_primary_genre(df)
        assert result.loc[0, "primary_genre"] == "Rock"

    def test_multiple_genres_takes_first(self):
        df = pd.DataFrame({"Genres": ["Rock, Pop, Jazz"]})
        result = extract_primary_genre(df)
        assert result.loc[0, "primary_genre"] == "Rock"

    def test_nan_genre(self):
        df = pd.DataFrame({"Genres": [pd.NA]})
        result = extract_primary_genre(df)
        assert pd.isna(result.loc[0, "primary_genre"])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"Genres": ["Rock"]})
        extract_primary_genre(df)
        assert "primary_genre" not in df.columns


# =============================================================================
# flag_unknown_artist tests
# =============================================================================


class TestFlagUnknownArtist:
    """Tests for flag_unknown_artist."""

    def test_unknown_artist_flagged(self):
        df = pd.DataFrame({"Artist": ["[unknown artist]"]})
        result = flag_unknown_artist(df)
        assert bool(result.loc[0, "is_unknown_artist"])

    def test_normal_artist_not_flagged(self):
        df = pd.DataFrame({"Artist": ["Radiohead"]})
        result = flag_unknown_artist(df)
        assert not bool(result.loc[0, "is_unknown_artist"])

    def test_similar_but_different_not_flagged(self):
        df = pd.DataFrame({"Artist": ["unknown artist"]})
        result = flag_unknown_artist(df)
        assert not bool(result.loc[0, "is_unknown_artist"])


# =============================================================================
# _has_nonempty_text tests
# =============================================================================


class TestHasNonemptyText:
    """Tests for _has_nonempty_text helper."""

    def test_valid_text(self):
        s = pd.Series(["hello"])
        assert bool(_has_nonempty_text(s).iloc[0])

    def test_empty_string(self):
        s = pd.Series([""])
        assert not bool(_has_nonempty_text(s).iloc[0])

    def test_whitespace_only(self):
        s = pd.Series(["   "])
        assert not bool(_has_nonempty_text(s).iloc[0])

    def test_none_value(self):
        s = pd.Series([None])
        assert not bool(_has_nonempty_text(s).iloc[0])

    def test_na_value(self):
        s = pd.Series([pd.NA])
        assert not bool(_has_nonempty_text(s).iloc[0])

    def test_mixed_values(self):
        s = pd.Series(["good", "", None, "  ", "ok"])
        result = _has_nonempty_text(s)
        assert result.tolist() == [True, False, False, False, True]


# =============================================================================
# apply_exclusion_filter tests
# =============================================================================


class TestApplyExclusionFilter:
    """Tests for apply_exclusion_filter."""

    def test_keeps_matching_rows(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_exclusion_filter(df, df["x"] > 1, reason="test")
        assert len(result) == 2
        assert result["x"].tolist() == [2, 3]

    def test_with_logger(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        df = pd.DataFrame(
            {
                "original_row_id": [0, 1, 2],
                "Artist": ["A", "B", "C"],
                "Album": ["X", "Y", "Z"],
                "x": [1, 2, 3],
            }
        )
        result = apply_exclusion_filter(df, df["x"] > 1, reason="below_threshold", logger=logger)
        assert len(result) == 2
        assert len(logger.exclusions) == 1
        assert logger.exclusions[0].artist == "A"
        assert len(logger.filter_stats) == 1

    def test_no_exclusions(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_exclusion_filter(df, df["x"] > 0, reason="none")
        assert len(result) == 3

    def test_all_excluded(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = apply_exclusion_filter(df, df["x"] > 10, reason="all")
        assert len(result) == 0


# =============================================================================
# clean_albums tests
# =============================================================================


class TestCleanAlbums:
    """Tests for clean_albums."""

    def test_adds_missing_optional_columns(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw, config=CleaningConfig(drop_descriptors=False))
        assert "Critic_Score" in cleaned.columns
        assert "Critic_Reviews" in cleaned.columns

    def test_drops_descriptors_by_default(self):
        raw = _make_raw_df()
        raw["Descriptors"] = ["some descriptor"]
        cleaned = clean_albums(raw)
        assert "Descriptors" not in cleaned.columns

    def test_keeps_descriptors_when_configured(self):
        raw = _make_raw_df()
        raw["Descriptors"] = ["some descriptor"]
        cleaned = clean_albums(raw, config=CleaningConfig(drop_descriptors=False))
        assert "Descriptors" in cleaned.columns

    def test_renames_columns(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw)
        assert "User_Score" in cleaned.columns
        assert "User_Ratings" in cleaned.columns

    def test_adds_collaboration_features(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw)
        assert "num_artists" in cleaned.columns
        assert "is_collaboration" in cleaned.columns
        assert "collab_type" in cleaned.columns

    def test_adds_primary_genre(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw)
        assert "primary_genre" in cleaned.columns

    def test_adds_unknown_artist_flag(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw)
        assert "is_unknown_artist" in cleaned.columns

    def test_adds_date_columns(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw)
        assert "Release_Date_Parsed" in cleaned.columns
        assert "date_risk" in cleaned.columns
        assert "date_imputation_type" in cleaned.columns

    def test_default_config_used_when_none(self):
        raw = _make_raw_df()
        cleaned = clean_albums(raw)
        assert len(cleaned) == 1

    def test_validation_failure_raises_in_strict_mode(self, monkeypatch):
        import panelcast.data.validation as validation_mod

        def _boom(df, lazy=True, descriptor=None):
            raise RuntimeError("schema violated")

        monkeypatch.setattr(validation_mod, "validate_cleaned_dataframe", _boom)
        raw = _make_raw_df()
        with pytest.raises(ValueError, match="strict mode"):
            clean_albums(raw, config=CleaningConfig(strict_validation=True))

    def test_validation_failure_warns_in_permissive_mode(self, monkeypatch):
        import panelcast.data.validation as validation_mod

        def _boom(df, lazy=True, descriptor=None):
            raise RuntimeError("schema violated")

        monkeypatch.setattr(validation_mod, "validate_cleaned_dataframe", _boom)
        raw = _make_raw_df()
        cleaned = clean_albums(raw, config=CleaningConfig(strict_validation=False))
        assert len(cleaned) == 1

    def test_strict_validation_passes_without_critic_columns(self):
        # A contract-valid frame legitimately omitting the optional critic
        # columns must survive post-cleaning validation, including strict mode.
        raw = _make_raw_df()
        assert "Critic Score" not in raw.columns
        assert "Critic Reviews" not in raw.columns
        cleaned = clean_albums(raw, config=CleaningConfig(strict_validation=True))
        assert cleaned["Critic_Score"].dtype == "float64"
        assert cleaned["Critic_Reviews"].dtype == "float64"

    def test_strict_validation_passes_with_tier3_row(self):
        # Tier-3 rows (no parseable date, no year) keep NaN Year by design and
        # must not fail post-cleaning validation.
        raw = pd.concat(
            [
                _make_raw_df(),
                _make_raw_df(
                    Album=["Album 2"],
                    Year=[None],
                    **{"Release Date": ["not-a-date"]},
                ),
            ],
            ignore_index=True,
        )
        cleaned = clean_albums(raw, config=CleaningConfig(strict_validation=True))
        assert cleaned["date_risk"].tolist() == ["low", "high"]
        assert cleaned["Year"].isna().tolist() == [False, True]


# =============================================================================
# CleaningConfig tests
# =============================================================================


class TestCleaningConfig:
    """Tests for CleaningConfig dataclass."""

    def test_defaults(self):
        config = CleaningConfig()
        assert config.min_year == 1950
        assert config.score_min == 0.0
        assert config.score_max == 100.0
        assert config.drop_descriptors is True

    def test_custom_values(self):
        config = CleaningConfig(min_year=2000, score_min=10.0, drop_descriptors=False)
        assert config.min_year == 2000
        assert config.score_min == 10.0
        assert config.drop_descriptors is False
