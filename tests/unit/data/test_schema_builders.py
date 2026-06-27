"""Frozen-oracle guards for the descriptor-driven Pandera schema builders.

The hand-written AOTY schemas were replaced by ``build_raw_schema`` /
``build_cleaned_schema``. The original literals are frozen here as oracles:

- structural test: builder output on the default descriptor must equal the
  legacy schema column-for-column (dtype, checks, nullable, required);
- behavioral corpus: identical pass/fail outcomes and failure-case counts on
  frames covering every check.
"""

import pandas as pd
import pandera.pandas as pa
import pytest

from panelcast.config.descriptor import DEFAULT_DESCRIPTOR
from panelcast.data.validation import (
    CleanedAlbumSchema,
    RawAlbumSchema,
    build_cleaned_schema,
    build_raw_schema,
    validate_cleaned_dataframe,
    validate_raw_dataframe,
)
from tests.helpers.aero_data import make_aero_descriptor

# ---------------------------------------------------------------------------
# Frozen legacy literals (pre-builder validation.py, verbatim). Do not edit.
# ---------------------------------------------------------------------------

LEGACY_RAW_SCHEMA = pa.DataFrameSchema(
    {
        "Artist": pa.Column(str, nullable=False),
        "Album": pa.Column(str, nullable=True),
        "Year": pa.Column(float, pa.Check.in_range(1900, 2030), nullable=True),
        "Release Date": pa.Column(str, nullable=True),
        "Genres": pa.Column(str, nullable=True),
        "Critic Score": pa.Column(
            float,
            pa.Check.in_range(0, 100),
            nullable=True,
            required=False,
        ),
        "User Score": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        "User Ratings": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Critic Reviews": pa.Column(
            float,
            pa.Check.ge(0),
            nullable=True,
            required=False,
        ),
        "Tracks": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Runtime (min)": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Avg Track Runtime (min)": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Avg Track Score": pa.Column(
            float,
            pa.Check.in_range(0, 100),
            nullable=True,
            required=False,
        ),
        "Label": pa.Column(str, nullable=True, required=False),
        "Descriptors": pa.Column(str, nullable=True, required=False),
        "Album Type": pa.Column(str, nullable=True),
        "Album URL": pa.Column(str, nullable=True, required=False),
        "All Artists": pa.Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)

LEGACY_CLEANED_SCHEMA = pa.DataFrameSchema(
    {
        "Artist": pa.Column(str, nullable=False),
        "Album": pa.Column(str, nullable=False),
        "Year": pa.Column(float, pa.Check.in_range(1900, 2030), nullable=False),
        "User_Score": pa.Column(float, pa.Check.in_range(0, 100), nullable=False),
        "User_Ratings": pa.Column(float, pa.Check.ge(0), nullable=False),
        "Num_Tracks": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Runtime_Min": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Avg_Runtime": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Album_Type": pa.Column(str, nullable=True),
        "Critic_Score": pa.Column(float, pa.Check.in_range(0, 100), nullable=True, required=False),
        "Critic_Reviews": pa.Column(float, pa.Check.ge(0), nullable=True, required=False),
        "primary_genre": pa.Column(str, nullable=True, required=False),
        "num_artists": pa.Column(float, pa.Check.ge(1), nullable=True, required=False),
        "is_collaboration": pa.Column(nullable=True, required=False),
    },
    strict=False,
    coerce=True,
)


def _column_props(column: pa.Column) -> dict:
    return {
        "dtype": str(column.dtype),
        "nullable": column.nullable,
        "required": column.required,
        "checks": [(check.name, check.statistics) for check in column.checks],
    }


def _schema_props(schema: pa.DataFrameSchema) -> dict:
    return {
        "strict": schema.strict,
        "coerce": schema.coerce,
        "columns": {name: _column_props(col) for name, col in schema.columns.items()},
    }


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------


class TestStructuralEquality:
    def test_raw_schema_matches_legacy_literal(self):
        assert _schema_props(build_raw_schema(DEFAULT_DESCRIPTOR)) == _schema_props(
            LEGACY_RAW_SCHEMA
        )

    def test_cleaned_schema_matches_legacy_literal(self):
        assert _schema_props(build_cleaned_schema(DEFAULT_DESCRIPTOR)) == _schema_props(
            LEGACY_CLEANED_SCHEMA
        )

    def test_module_constants_are_default_builds(self):
        assert _schema_props(RawAlbumSchema) == _schema_props(LEGACY_RAW_SCHEMA)
        assert _schema_props(CleanedAlbumSchema) == _schema_props(LEGACY_CLEANED_SCHEMA)

    # NOTE: pandera Column/DataFrameSchema __eq__ is deliberately not used
    # here. Check.__eq__ compares internal state that validate() mutates, so
    # object equality silently depends on whether either schema has validated
    # a frame before (order-dependent under pytest-randomly). Column insertion
    # order also differs from the legacy literal and is presentational only.
    # The _schema_props projection (dtype, nullable, required, check
    # name+statistics, strict, coerce) is the stable structural contract; the
    # behavioral corpus below pins the semantics.


# ---------------------------------------------------------------------------
# Behavioral corpus: identical outcomes legacy vs builder
# ---------------------------------------------------------------------------


def _valid_raw_frame(**overrides) -> pd.DataFrame:
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


def _valid_cleaned_frame(**overrides) -> pd.DataFrame:
    data = {
        "Artist": ["A"],
        "Album": ["Album 1"],
        "Year": [2020.0],
        "User_Score": [75.0],
        "User_Ratings": [100.0],
        "Num_Tracks": [10.0],
        "Runtime_Min": [40.0],
        "Avg_Runtime": [4.0],
        "Album_Type": ["LP"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


RAW_CORPUS = [
    ("valid", _valid_raw_frame()),
    ("null_album", _valid_raw_frame(Album=[None])),
    ("int_year", _valid_raw_frame(Year=[2020])),
    ("null_year", _valid_raw_frame(Year=[None])),
    ("year_too_low", _valid_raw_frame(Year=[1800.0])),
    ("year_too_high", _valid_raw_frame(Year=[2031.0])),
    ("score_high", _valid_raw_frame(**{"User Score": [101.0]})),
    ("score_negative", _valid_raw_frame(**{"User Score": [-1.0]})),
    ("score_null", _valid_raw_frame(**{"User Score": [None]})),
    ("ratings_negative", _valid_raw_frame(**{"User Ratings": [-1.0]})),
    ("tracks_negative", _valid_raw_frame(Tracks=[-5.0])),
    ("null_artist", _valid_raw_frame(Artist=[None])),
    ("missing_required_column", _valid_raw_frame().drop(columns=["Genres"])),
    ("multi_error", _valid_raw_frame(**{"User Score": [200.0], "Year": [1800.0]})),
]

CLEANED_CORPUS = [
    ("valid", _valid_cleaned_frame()),
    ("null_score", _valid_cleaned_frame(User_Score=[None])),
    ("null_year", _valid_cleaned_frame(Year=[None])),
    ("score_high", _valid_cleaned_frame(User_Score=[150.0])),
    ("num_artists_zero", _valid_cleaned_frame(num_artists=[0.0])),
    ("with_optional", _valid_cleaned_frame(Critic_Score=[85.0], primary_genre=["Rock"])),
    ("critic_out_of_range", _valid_cleaned_frame(Critic_Score=[120.0])),
    ("missing_album_type", _valid_cleaned_frame().drop(columns=["Album_Type"])),
]


def _outcome(schema: pa.DataFrameSchema, df: pd.DataFrame):
    try:
        schema.validate(df.copy(), lazy=True)
        return ("pass", None)
    except pa.errors.SchemaErrors as e:
        cases = e.failure_cases
        return (
            "fail",
            sorted(zip(cases["check"].astype(str), cases["column"].astype(str), strict=True)),
        )


class TestBehavioralEquivalence:
    @pytest.mark.parametrize("name,frame", RAW_CORPUS, ids=[n for n, _ in RAW_CORPUS])
    def test_raw_corpus(self, name, frame):
        legacy = _outcome(LEGACY_RAW_SCHEMA, frame)
        built = _outcome(build_raw_schema(DEFAULT_DESCRIPTOR), frame)
        assert built == legacy

    @pytest.mark.parametrize("name,frame", CLEANED_CORPUS, ids=[n for n, _ in CLEANED_CORPUS])
    def test_cleaned_corpus(self, name, frame):
        legacy = _outcome(LEGACY_CLEANED_SCHEMA, frame)
        built = _outcome(build_cleaned_schema(DEFAULT_DESCRIPTOR), frame)
        assert built == legacy


# ---------------------------------------------------------------------------
# Non-AOTY (aero) descriptor schemas
# ---------------------------------------------------------------------------


def _valid_aero_raw(**overrides) -> pd.DataFrame:
    data = {
        "Airframe": ["Falcon-X1"],
        "Flight ID": ["FX1-F01"],
        "Campaign Year": [2021.0],
        "Flight Date": ["2021-03-15"],
        "Perf Score": [7.5],
        "Sensor Samples": [120.0],
        "Test Crew": ["Falcon-X1"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestAeroSchemas:
    def test_raw_schema_columns_follow_descriptor(self):
        schema = build_raw_schema(make_aero_descriptor())
        assert set(schema.columns) == {
            "Airframe",
            "Flight ID",
            "Campaign Year",
            "Flight Date",
            "Perf Score",
            "Sensor Samples",
            "Test Crew",
        }
        assert schema.columns["Airframe"].nullable is False
        score_checks = schema.columns["Perf Score"].checks
        assert score_checks[0].statistics["min_value"] == 0.0
        assert score_checks[0].statistics["max_value"] == 10.0

    def test_valid_aero_frame_passes(self):
        validate_raw_dataframe(_valid_aero_raw(), descriptor=make_aero_descriptor())

    def test_aero_score_out_of_bounds_fails(self):
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw_dataframe(
                _valid_aero_raw(**{"Perf Score": [11.0]}),
                descriptor=make_aero_descriptor(),
            )

    def test_cleaned_schema_uses_canonical_names_and_bounds(self):
        schema = build_cleaned_schema(make_aero_descriptor())
        assert "Perf_Score" in schema.columns
        assert "Num_Tracks" not in schema.columns
        assert "Critic_Score" not in schema.columns
        cleaned = pd.DataFrame(
            {
                "Airframe": ["A"],
                "Flight_ID": ["F1"],
                "Year": [2021.0],
                "Perf_Score": [8.0],
                "Sensor_Samples": [50.0],
            }
        )
        validate_cleaned_dataframe(cleaned, descriptor=make_aero_descriptor())

    def test_cleaned_aero_score_out_of_bounds_fails(self):
        cleaned = pd.DataFrame(
            {
                "Airframe": ["A"],
                "Flight_ID": ["F1"],
                "Year": [2021.0],
                "Perf_Score": [10.5],
                "Sensor_Samples": [50.0],
            }
        )
        with pytest.raises(pa.errors.SchemaErrors):
            validate_cleaned_dataframe(cleaned, descriptor=make_aero_descriptor())
