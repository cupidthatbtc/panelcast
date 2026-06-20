"""Schema validation for raw and processed data.

Schemas are built from a :class:`DatasetDescriptor` at runtime:
:func:`build_raw_schema` and :func:`build_cleaned_schema` derive per-column
dtypes and checks from the descriptor's column roles (entity, event, year,
date, target, observation count, multi-entity). Columns without a role fall
back to AOTY-specific specs (track counts, runtimes, labels) when known, or a
permissive nullable column otherwise.

The module-level ``RawAlbumSchema`` / ``CleanedAlbumSchema`` constants are the
builders applied to the default (AOTY) descriptor, so existing imports keep
working; frozen-oracle tests pin them to the original hand-written literals.
"""

from typing import Callable

import pandas as pd
import pandera.pandas as pa

from panelcast.config.descriptor import DEFAULT_DESCRIPTOR, DatasetDescriptor

# Required baseline columns for user-score modeling.
REQUIRED_RAW_COLUMNS = [
    "Artist",
    "Album",
    "Year",
    "Release Date",
    "Genres",
    "User Score",
    "User Ratings",
    "Tracks",
    "Runtime (min)",
    "Avg Track Runtime (min)",
    "Album Type",
    "All Artists",
]

OPTIONAL_RAW_COLUMNS = [
    "Critic Score",
    "Critic Reviews",
    "Avg Track Score",
    "Descriptors",
    "Label",
    "Album URL",
]

# Sanity range applied to the year column in both schemas.
YEAR_RANGE = (1900, 2030)

# Specs for AOTY columns that have no descriptor role. Raw-name keyed;
# non-role columns absent from this map validate as permissive nullable
# columns. Note: use float for numeric columns because pandas represents
# int+NaN as float.
_EXTRA_RAW_COLUMN_FACTORIES: dict[str, Callable[[bool], pa.Column]] = {
    "Genres": lambda required: pa.Column(str, nullable=True, required=required),
    "Tracks": lambda required: pa.Column(float, pa.Check.ge(0), nullable=True, required=required),
    "Runtime (min)": lambda required: pa.Column(
        float, pa.Check.ge(0), nullable=True, required=required
    ),
    "Avg Track Runtime (min)": lambda required: pa.Column(
        float, pa.Check.ge(0), nullable=True, required=required
    ),
    "Avg Track Score": lambda required: pa.Column(
        float, pa.Check.in_range(0, 100), nullable=True, required=required
    ),
    "Album Type": lambda required: pa.Column(str, nullable=True, required=required),
    "Label": lambda required: pa.Column(str, nullable=True, required=required),
    "Descriptors": lambda required: pa.Column(str, nullable=True, required=required),
    "Album URL": lambda required: pa.Column(str, nullable=True, required=required),
}

# Cleaning-stage canonical counterparts of the non-role AOTY columns, included
# in the cleaned schema only when this descriptor's raw lists produce them.
_EXTRA_CLEANED_COLUMN_FACTORIES: dict[str, Callable[[], pa.Column]] = {
    "Num_Tracks": lambda: pa.Column(float, pa.Check.ge(0), nullable=True),
    "Runtime_Min": lambda: pa.Column(float, pa.Check.ge(0), nullable=True),
    "Avg_Runtime": lambda: pa.Column(float, pa.Check.ge(0), nullable=True),
    "Album_Type": lambda: pa.Column(str, nullable=True),
}


def _as_legacy_bound(value: float) -> float | int:
    """Render integral bounds as ints so check strings match the legacy
    literals exactly (in_range(0, 100), not in_range(0.0, 100.0))."""
    return int(value) if float(value).is_integer() else value


def _target_bounds(descriptor: DatasetDescriptor) -> tuple[float | int, float | int]:
    low, high = descriptor.target_bounds
    return _as_legacy_bound(low), _as_legacy_bound(high)


def _raw_name(descriptor: DatasetDescriptor, canonical: str | None) -> str | None:
    """Invert raw_column_map: canonical column name -> raw export name."""
    if canonical is None:
        return None
    for raw, mapped in descriptor.raw_column_map.items():
        if mapped == canonical:
            return raw
    return canonical


def _raw_role_column(
    descriptor: DatasetDescriptor,
    raw_name: str,
    required: bool,
) -> pa.Column | None:
    """Column spec for a raw column with a descriptor role, else None."""
    low, high = _target_bounds(descriptor)
    if raw_name == _raw_name(descriptor, descriptor.entity_col):
        return pa.Column(str, nullable=False, required=required)
    if raw_name == _raw_name(descriptor, descriptor.event_col):
        # Raw export can contain a small number of missing event names.
        # These are filtered with audit logging in cleaning filters.
        return pa.Column(str, nullable=True, required=required)
    if raw_name == _raw_name(descriptor, descriptor.year_col):
        return pa.Column(float, pa.Check.in_range(*YEAR_RANGE), nullable=True, required=required)
    if raw_name == _raw_name(descriptor, descriptor.date_col):
        return pa.Column(str, nullable=True, required=required)
    if raw_name in (
        _raw_name(descriptor, descriptor.target_col),
        _raw_name(descriptor, descriptor.secondary_target_col),
    ):
        return pa.Column(float, pa.Check.in_range(low, high), nullable=True, required=required)
    if raw_name in (
        _raw_name(descriptor, descriptor.n_obs_col),
        _raw_name(descriptor, descriptor.secondary_n_obs_col),
    ):
        return pa.Column(float, pa.Check.ge(0), nullable=True, required=required)
    if descriptor.multi_entity_col is not None and raw_name == _raw_name(
        descriptor, descriptor.multi_entity_col
    ):
        return pa.Column(str, nullable=True, required=required)
    return None


def build_raw_schema(descriptor: DatasetDescriptor) -> pa.DataFrameSchema:
    """Build the raw-data validation schema for a descriptor."""
    columns: dict[str, pa.Column] = {}
    for raw_name, required in [(c, True) for c in descriptor.required_raw_columns] + [
        (c, False) for c in descriptor.optional_raw_columns
    ]:
        column = _raw_role_column(descriptor, raw_name, required)
        if column is None:
            factory = _EXTRA_RAW_COLUMN_FACTORIES.get(raw_name)
            column = (
                factory(required)
                if factory is not None
                else pa.Column(nullable=True, required=required)
            )
        columns[raw_name] = column
    return pa.DataFrameSchema(
        columns,
        strict=False,  # Allow extra columns (original_row_id added later)
        coerce=True,  # Coerce types where possible
    )


def build_cleaned_schema(descriptor: DatasetDescriptor) -> pa.DataFrameSchema:
    """Build the post-cleaning validation schema for a descriptor."""
    low, high = _target_bounds(descriptor)
    columns: dict[str, pa.Column] = {
        descriptor.entity_col: pa.Column(str, nullable=False),
        descriptor.event_col: pa.Column(str, nullable=False),
        descriptor.year_col: pa.Column(float, pa.Check.in_range(*YEAR_RANGE), nullable=False),
        descriptor.target_col: pa.Column(float, pa.Check.in_range(low, high), nullable=False),
        descriptor.n_obs_col: pa.Column(float, pa.Check.ge(0), nullable=False),
    }
    canonical_required = {
        descriptor.raw_column_map.get(c, c) for c in descriptor.required_raw_columns
    }
    for name, factory in _EXTRA_CLEANED_COLUMN_FACTORIES.items():
        if name in canonical_required:
            columns[name] = factory()
    if descriptor.secondary_target_col is not None:
        columns[descriptor.secondary_target_col] = pa.Column(
            float, pa.Check.in_range(low, high), nullable=True, required=False
        )
        if descriptor.secondary_n_obs_col is not None:
            columns[descriptor.secondary_n_obs_col] = pa.Column(
                float, pa.Check.ge(0), nullable=True, required=False
            )
    # Cleaning-derived columns (fixed canonical names).
    columns["primary_genre"] = pa.Column(str, nullable=True, required=False)
    columns["num_artists"] = pa.Column(float, pa.Check.ge(1), nullable=True, required=False)
    columns["is_collaboration"] = pa.Column(nullable=True, required=False)
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


# Schemas for the default (AOTY) descriptor; kept as module constants so
# existing imports survive. Pinned to the original hand-written literals by
# tests/unit/data/test_schema_builders.py.
RawAlbumSchema = build_raw_schema(DEFAULT_DESCRIPTOR)
CleanedAlbumSchema = build_cleaned_schema(DEFAULT_DESCRIPTOR)


def validate_raw_dataframe(
    df: pd.DataFrame,
    lazy: bool = True,
    descriptor: DatasetDescriptor | None = None,
) -> pd.DataFrame:
    """
    Validate raw DataFrame against the descriptor's raw schema.

    Args:
        df: DataFrame loaded from raw CSV
        lazy: If True, collect all errors before raising. If False, fail fast.
        descriptor: Dataset descriptor (None = AOTY defaults).

    Returns:
        Validated DataFrame (with types coerced)

    Raises:
        pa.errors.SchemaErrors: If validation fails (contains failure_cases attribute)
    """
    schema = RawAlbumSchema if descriptor is None else build_raw_schema(descriptor)
    return schema.validate(df, lazy=lazy)


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Legacy function for backward compatibility.

    Raises ValueError if required columns are missing.
    """
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")


# Post-cleaning canonical column names
REQUIRED_CLEANED_COLUMNS = [
    "Artist",
    "Album",
    "Year",
    "User_Score",
    "User_Ratings",
]


def validate_cleaned_dataframe(
    df: pd.DataFrame,
    lazy: bool = True,
    descriptor: DatasetDescriptor | None = None,
) -> pd.DataFrame:
    """
    Validate cleaned DataFrame against the descriptor's cleaned schema.

    Catches NaN propagation from cleaning steps before model training.

    Args:
        df: DataFrame after cleaning pipeline.
        lazy: If True, collect all errors before raising.
        descriptor: Dataset descriptor (None = AOTY defaults).

    Returns:
        Validated DataFrame.

    Raises:
        pa.errors.SchemaErrors: If validation fails.
    """
    schema = CleanedAlbumSchema if descriptor is None else build_cleaned_schema(descriptor)
    return schema.validate(df, lazy=lazy)
