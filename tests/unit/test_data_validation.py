import pandas as pd
import pandera.pandas as pa
import pytest

from panelcast.data.validation import (
    OPTIONAL_RAW_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    RawAlbumSchema,
    validate_raw_dataframe,
    validate_raw_schema,
)


def _make_valid_raw_df(**overrides) -> pd.DataFrame:
    """Create a minimal valid raw DataFrame with optional overrides."""
    data = {
        "Artist": ["A"],
        "Album": ["Album 1"],
        "Year": [2020.0],
        "Release Date": ["January 01, 2020"],
        "Genres": ["Rock"],
        "User Score": [75.0],
        "User Ratings": [100.0],
        "Tracks": [10.0],
        "Runtime (min)": [40.0],
        "Avg Track Runtime (min)": [4.0],
        "Album Type": ["LP"],
        "All Artists": ["A"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# =============================================================================
# validate_raw_schema (legacy)
# =============================================================================


class TestValidateRawSchema:
    """Tests for legacy validate_raw_schema function."""

    def test_missing_single_column(self):
        df = pd.DataFrame({"Artist": ["a"]})
        with pytest.raises(ValueError, match="Missing required raw columns"):
            validate_raw_schema(df)

    def test_missing_multiple_columns_lists_all(self):
        df = pd.DataFrame({"Artist": ["a"], "Album": ["b"]})
        with pytest.raises(ValueError) as exc_info:
            validate_raw_schema(df)
        msg = str(exc_info.value)
        assert "Year" in msg

    def test_passes_with_all_required_columns(self):
        data = {col: ["test"] for col in REQUIRED_RAW_COLUMNS}
        df = pd.DataFrame(data)
        validate_raw_schema(df)  # Should not raise

    def test_passes_with_extra_columns(self):
        data = {col: ["test"] for col in REQUIRED_RAW_COLUMNS}
        data["ExtraCol"] = ["extra"]
        df = pd.DataFrame(data)
        validate_raw_schema(df)  # Should not raise

    def test_empty_dataframe_with_required_columns(self):
        data = {col: [] for col in REQUIRED_RAW_COLUMNS}
        df = pd.DataFrame(data)
        validate_raw_schema(df)  # Should not raise


# =============================================================================
# REQUIRED_RAW_COLUMNS and OPTIONAL_RAW_COLUMNS
# =============================================================================


class TestColumnConstants:
    """Tests for column constant lists."""

    def test_required_columns_not_empty(self):
        assert len(REQUIRED_RAW_COLUMNS) > 0

    def test_optional_columns_not_empty(self):
        assert len(OPTIONAL_RAW_COLUMNS) > 0

    def test_no_overlap_between_required_and_optional(self):
        overlap = set(REQUIRED_RAW_COLUMNS) & set(OPTIONAL_RAW_COLUMNS)
        assert overlap == set(), f"Overlap found: {overlap}"

    def test_artist_in_required(self):
        assert "Artist" in REQUIRED_RAW_COLUMNS

    def test_album_in_required(self):
        assert "Album" in REQUIRED_RAW_COLUMNS

    def test_user_score_in_required(self):
        assert "User Score" in REQUIRED_RAW_COLUMNS

    def test_critic_score_in_optional(self):
        assert "Critic Score" in OPTIONAL_RAW_COLUMNS

    def test_descriptors_in_optional(self):
        assert "Descriptors" in OPTIONAL_RAW_COLUMNS


# =============================================================================
# validate_raw_dataframe (pandera-based)
# =============================================================================


class TestValidateRawDataframe:
    """Tests for validate_raw_dataframe."""

    def test_allows_missing_optional_columns(self):
        df = _make_valid_raw_df()
        validated = validate_raw_dataframe(df)
        assert len(validated) == 1

    def test_allows_null_album(self):
        df = _make_valid_raw_df(Album=[None])
        validated = validate_raw_dataframe(df)
        assert len(validated) == 1

    def test_coerces_integer_year_to_float(self):
        df = _make_valid_raw_df(Year=[2020])
        validated = validate_raw_dataframe(df)
        assert validated["Year"].dtype == float

    def test_allows_null_year(self):
        df = _make_valid_raw_df(Year=[None])
        validated = validate_raw_dataframe(df)
        assert pd.isna(validated["Year"].iloc[0])

    def test_year_out_of_range_fails(self):
        df = _make_valid_raw_df(Year=[1800.0])
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_year_upper_bound(self):
        df = _make_valid_raw_df(Year=[2030.0])
        validated = validate_raw_dataframe(df)
        assert validated["Year"].iloc[0] == 2030.0

    def test_year_above_upper_bound_fails(self):
        df = _make_valid_raw_df(Year=[2031.0])
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_user_score_in_range(self):
        df = _make_valid_raw_df(**{"User Score": [0.0]})
        validated = validate_raw_dataframe(df)
        assert validated["User Score"].iloc[0] == 0.0

    def test_user_score_upper_boundary(self):
        df = _make_valid_raw_df(**{"User Score": [100.0]})
        validated = validate_raw_dataframe(df)
        assert validated["User Score"].iloc[0] == 100.0

    def test_user_score_out_of_range_fails(self):
        df = _make_valid_raw_df(**{"User Score": [101.0]})
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_negative_user_score_fails(self):
        df = _make_valid_raw_df(**{"User Score": [-1.0]})
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_null_user_score_allowed(self):
        df = _make_valid_raw_df(**{"User Score": [None]})
        validated = validate_raw_dataframe(df)
        assert pd.isna(validated["User Score"].iloc[0])

    def test_negative_user_ratings_fails(self):
        df = _make_valid_raw_df(**{"User Ratings": [-1.0]})
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_zero_user_ratings_allowed(self):
        df = _make_valid_raw_df(**{"User Ratings": [0.0]})
        validated = validate_raw_dataframe(df)
        assert validated["User Ratings"].iloc[0] == 0.0

    def test_negative_tracks_fails(self):
        df = _make_valid_raw_df(Tracks=[-5.0])
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_zero_runtime_allowed(self):
        df = _make_valid_raw_df(**{"Runtime (min)": [0.0]})
        validated = validate_raw_dataframe(df)
        assert validated["Runtime (min)"].iloc[0] == 0.0

    def test_extra_columns_allowed(self):
        df = _make_valid_raw_df()
        df["ExtraColumn"] = ["extra"]
        validated = validate_raw_dataframe(df)
        assert "ExtraColumn" in validated.columns

    def test_with_optional_critic_score(self):
        df = _make_valid_raw_df()
        df["Critic Score"] = [85.0]
        validated = validate_raw_dataframe(df)
        assert validated["Critic Score"].iloc[0] == 85.0

    def test_critic_score_out_of_range_fails(self):
        df = _make_valid_raw_df()
        df["Critic Score"] = [101.0]
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(df, lazy=True)

    def test_lazy_validation_collects_all_errors(self):
        df = _make_valid_raw_df(**{"User Score": [200.0], "Year": [1800.0]})
        with pytest.raises(pa.errors.SchemaErrors) as exc_info:
            validate_raw_dataframe(df, lazy=True)
        # Should collect multiple failures
        assert len(exc_info.value.failure_cases) > 0

    def test_multiple_valid_rows(self):
        df = pd.DataFrame(
            {
                "Artist": ["A", "B", "C"],
                "Album": ["X", "Y", "Z"],
                "Year": [2020.0, 2021.0, 2022.0],
                "Release Date": ["January 01, 2020", "February 15, 2021", "March 10, 2022"],
                "Genres": ["Rock", "Pop", "Jazz"],
                "User Score": [70.0, 80.0, 90.0],
                "User Ratings": [100.0, 200.0, 300.0],
                "Tracks": [10.0, 12.0, 8.0],
                "Runtime (min)": [40.0, 50.0, 35.0],
                "Avg Track Runtime (min)": [4.0, 4.2, 4.4],
                "Album Type": ["LP", "LP", "EP"],
                "All Artists": ["A", "B", "C"],
            }
        )
        validated = validate_raw_dataframe(df)
        assert len(validated) == 3


# =============================================================================
# RawAlbumSchema
# =============================================================================


class TestRawAlbumSchema:
    """Tests for the schema object itself."""

    def test_schema_is_not_strict(self):
        assert RawAlbumSchema.strict is False

    def test_schema_coerces_types(self):
        assert RawAlbumSchema.coerce is True
