"""Data cleaning and filtering pipeline.

All domain-specific names (column names, date format, multi-entity separator,
sentinel values, score bounds) come from a :class:`DatasetDescriptor`. Module
constants keep the AOTY literals as defaults so existing call sites behave
byte-identically (default-equals-AOTY contract).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import structlog

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.data.lineage import AuditLogger

# Column name mapping from raw to canonical (AOTY default; descriptor-driven
# callers pass descriptor.raw_column_map instead).
RAW_TO_CANONICAL = {
    "Release Date": "Release_Date",
    "Critic Score": "Critic_Score",
    "User Score": "User_Score",
    "Avg Track Score": "Avg_Track_Score",
    "User Ratings": "User_Ratings",
    "Critic Reviews": "Critic_Reviews",
    "Tracks": "Num_Tracks",
    "Runtime (min)": "Runtime_Min",
    "Avg Track Runtime (min)": "Avg_Runtime",
    "Album URL": "Album_URL",
    "All Artists": "All_Artists",
    "Album Type": "Album_Type",
}

OPTIONAL_CANONICAL_COLUMNS = (
    "Critic_Score",
    "Critic_Reviews",
    "Avg_Track_Score",
    "Label",
    "Descriptors",
    "Album_URL",
)

# Optional columns the cleaned schema types as float64. Fabricated absent
# columns must be float64 NaN, not object pd.NA, which pandera cannot coerce.
OPTIONAL_NUMERIC_CANONICAL_COLUMNS = (
    "Critic_Score",
    "Critic_Reviews",
    "Avg_Track_Score",
)

# AOTY collaboration-size bins: label -> (min_count, max_count); None = open.
DEFAULT_GROUP_SIZE_BINS: dict[str, tuple[int | None, int | None]] = {
    "solo": (1, 1),
    "duo": (2, 2),
    "small_group": (3, 4),
    "ensemble": (5, None),
}


@dataclass
class CleaningConfig:
    """Configuration for cleaning pipeline."""

    min_year: int = 1950
    max_year: int = field(default_factory=lambda: date.today().year)
    score_min: float = 0.0
    score_max: float = 100.0
    drop_descriptors: bool = True  # Per research: 4.2% coverage, severe selection bias
    # If True, post-cleaning schema validation failures raise instead of warn.
    strict_validation: bool = False
    # Domain-specific names; default reproduces AOTY behavior exactly.
    descriptor: DatasetDescriptor = field(default_factory=DatasetDescriptor)


def rename_columns(
    df: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Rename columns from raw format to canonical (underscore-separated)."""
    return df.rename(columns=dict(column_map if column_map is not None else RAW_TO_CANONICAL))


def ensure_optional_columns(
    df: pd.DataFrame,
    optional_columns: Sequence[str] | None = None,
    numeric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Ensure optional canonical columns exist for downstream compatibility."""
    df = df.copy()
    columns = OPTIONAL_CANONICAL_COLUMNS if optional_columns is None else optional_columns
    numeric = OPTIONAL_NUMERIC_CANONICAL_COLUMNS if numeric_columns is None else numeric_columns
    for col in columns:
        if col not in df.columns:
            if col in numeric:
                df[col] = pd.Series(np.nan, dtype="float64", index=df.index)
            else:
                df[col] = pd.NA
    return df


def parse_release_dates(
    df: pd.DataFrame,
    min_year: int = 1950,
    max_year: int | None = None,
    *,
    date_col: str = "Release_Date",
    parsed_date_col: str = "Release_Date_Parsed",
    year_col: str = "Year",
    date_format: str = "%B %d, %Y",
) -> pd.DataFrame:
    """
    Parse the raw date column with three-tier risk classification.

    Creates columns:
    - ``parsed_date_col``: datetime (NaT for tier-3 rows — never imputed)
    - date_risk: 'low' | 'medium' | 'high'
    - date_imputation_type: 'none' | 'jan1' | 'unimputed'
    - date_missing: True where no date could be parsed or derived (tier 3)
    """
    df = df.copy()

    # Parse existing dates (AOTY format: "April 10, 2018")
    df[parsed_date_col] = pd.to_datetime(
        df[date_col],
        format=date_format,
        errors="coerce",
    )

    # Initialize risk columns
    df["date_risk"] = "low"
    df["date_imputation_type"] = "none"

    # Tier 2: Has year but no parseable date
    tier2_mask = df[parsed_date_col].isna() & df[year_col].notna()
    df.loc[tier2_mask, "date_risk"] = "medium"
    df.loc[tier2_mask, "date_imputation_type"] = "jan1"
    df.loc[tier2_mask, parsed_date_col] = pd.to_datetime(
        df.loc[tier2_mask, year_col].astype(int).astype(str) + "-01-01"
    )

    # Tier 3: No parseable date and no year fallback.
    # This includes truly missing release dates and malformed date strings.
    # These rows keep NaT — no imputation happens, and the label says so.
    # date_missing lets downstream features model missingness explicitly
    # instead of silently filling fabricated zeros.
    tier3_mask = df[parsed_date_col].isna() & df[year_col].isna()
    df.loc[tier3_mask, "date_risk"] = "high"
    df.loc[tier3_mask, "date_imputation_type"] = "unimputed"
    df["date_missing"] = tier3_mask

    # Flag edge cases
    if max_year is None:
        max_year = date.today().year
    df["flag_future_year"] = df[year_col] > max_year
    df["flag_sparse_era"] = df[year_col] < min_year

    return df


def _count_entities(value: object, separator: str) -> int:
    """Count separator-delimited entities in a multi-entity cell (NA -> 1)."""
    if pd.isna(value):
        return 1
    text = str(value)
    if separator in text:
        return len(text.split(separator))
    return 1


def _classify_group_size(
    n: int,
    bins: Mapping[str, Sequence[int | None]],
) -> str:
    """Map an entity count onto its labeled size bin (None bound = open)."""
    for label, (low, high) in bins.items():
        if (low is None or n >= low) and (high is None or n <= high):
            return label
    return "unbinned"


def extract_collaboration_features(
    df: pd.DataFrame,
    *,
    multi_entity_col: str = "All_Artists",
    separator: str = " | ",
    group_size_bins: Mapping[str, Sequence[int | None]] | None = None,
) -> pd.DataFrame:
    """
    Extract collaboration features from the multi-entity column.

    Creates columns:
    - num_artists: count of entities
    - is_collaboration: boolean
    - collab_type: size-bin label (AOTY: 'solo' | 'duo' | 'small_group' | 'ensemble')
    """
    df = df.copy()
    bins = DEFAULT_GROUP_SIZE_BINS if group_size_bins is None else group_size_bins

    df["num_artists"] = df[multi_entity_col].apply(_count_entities, separator=separator)
    df["is_collaboration"] = df["num_artists"] > 1
    df["collab_type"] = df["num_artists"].apply(_classify_group_size, bins=bins)

    return df


def extract_primary_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract primary genre (first listed) from Genres column.

    Creates column:
    - primary_genre: first genre in comma-separated list
    """
    df = df.copy()
    df["primary_genre"] = df["Genres"].str.split(", ").str[0]
    return df


def flag_unknown_artist(
    df: pd.DataFrame,
    *,
    entity_col: str = "Artist",
    sentinel: str = "[unknown artist]",
) -> pd.DataFrame:
    """Flag the unknown-entity sentinel for special handling."""
    df = df.copy()
    df["is_unknown_artist"] = df[entity_col] == sentinel
    return df


def apply_exclusion_filter(
    df: pd.DataFrame,
    condition: pd.Series,
    reason: str,
    logger: AuditLogger | None = None,
    value_col: str | None = None,
    *,
    entity_col: str = "Artist",
    event_col: str = "Album",
) -> pd.DataFrame:
    """
    Apply filter and log exclusions.

    Args:
        df: DataFrame to filter
        condition: Boolean Series - True = keep, False = exclude
        reason: Reason string for audit log
        logger: Optional AuditLogger to record exclusions
        value_col: Column name to include in exclusion value
        entity_col: Column recorded as the audit 'artist' field
        event_col: Column recorded as the audit 'album' field

    Returns:
        Filtered DataFrame (rows where condition is True)
    """
    excluded = df[~condition]
    kept = df[condition].copy()

    if logger is not None:
        logger.log_exclusions_bulk(
            excluded,
            reason=reason,
            value_col=value_col,
            entity_col=entity_col,
            event_col=event_col,
        )
        logger.log_filter_stats(
            filter_name=reason,
            rows_before=len(df),
            rows_excluded=len(excluded),
            rows_after=len(kept),
        )

    return kept


def _has_nonempty_text(series: pd.Series) -> pd.Series:
    """Return True where values are non-null and non-empty after stripping."""
    return series.notna() & (series.astype(str).str.strip() != "")


def clean_albums(
    df: pd.DataFrame,
    config: CleaningConfig | None = None,
    logger: AuditLogger | None = None,
) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to raw album data.

    Steps:
    1. Rename columns to canonical format
    2. Parse dates with risk classification
    3. Extract collaboration features (when the descriptor has a multi-entity column)
    4. Extract primary genre (when a Genres column exists)
    5. Flag unknown entity (when the descriptor has a sentinel)
    6. Drop Descriptors column (per research)

    Does NOT apply filtering - use apply_exclusion_filter separately.

    Args:
        df: Raw DataFrame (with original_row_id)
        config: Cleaning configuration
        logger: Optional audit logger

    Returns:
        Cleaned DataFrame with new columns
    """
    config = config or CleaningConfig()
    descriptor = config.descriptor

    # Apply transformations
    df = rename_columns(df, descriptor.raw_column_map)
    df = ensure_optional_columns(
        df,
        tuple(descriptor.raw_column_map.get(col, col) for col in descriptor.optional_raw_columns),
        # Secondary target/n_obs are float64 in the cleaned schema whatever
        # the descriptor calls them; the constant covers the AOTY extras.
        numeric_columns=tuple(
            set(OPTIONAL_NUMERIC_CANONICAL_COLUMNS)
            | {
                c
                for c in (descriptor.secondary_target_col, descriptor.secondary_n_obs_col)
                if c is not None
            }
        ),
    )
    df = parse_release_dates(
        df,
        min_year=config.min_year,
        max_year=config.max_year,
        date_col=descriptor.date_col,
        parsed_date_col=descriptor.parsed_date_col,
        year_col=descriptor.year_col,
        date_format=descriptor.date_format,
    )
    if descriptor.multi_entity_col is not None and descriptor.multi_entity_col in df.columns:
        df = extract_collaboration_features(
            df,
            multi_entity_col=descriptor.multi_entity_col,
            separator=descriptor.multi_entity_separator,
            group_size_bins=descriptor.group_size_bins,
        )
    if "Genres" in df.columns:
        df = extract_primary_genre(df)
    if descriptor.unknown_entity_sentinel is not None:
        df = flag_unknown_artist(
            df,
            entity_col=descriptor.entity_col,
            sentinel=descriptor.unknown_entity_sentinel,
        )

    # Drop Descriptors per research (4.2% coverage, severe selection bias)
    if config.drop_descriptors and "Descriptors" in df.columns:
        df = df.drop(columns=["Descriptors"])
        if logger:
            logger.log.info(
                "column_dropped", column="Descriptors", reason="low_coverage_selection_bias"
            )

    # Post-cleaning schema validation: catch NaN propagation before model training.
    # Under strict_validation, failures abort the pipeline instead of being
    # downgraded to warnings (silent NaN propagation corrupts downstream stages).
    from panelcast.data.validation import validate_cleaned_dataframe

    try:
        validate_cleaned_dataframe(df, lazy=True, descriptor=descriptor)
    except Exception as e:
        if config.strict_validation:
            raise ValueError(
                f"Post-cleaning schema validation failed in strict mode. Original error: {e}"
            ) from e
        structlog.get_logger().warning(
            "post_cleaning_validation_failed",
            error=str(e)[:2000],
            action="proceeding_with_potentially_invalid_data",
            hint="Run with strict validation to fail fast on schema violations.",
        )

    return df


def _n_obs_reason_label(n_obs_col: str, prefix: str | None) -> str:
    """Audit-reason fragment for the observation-count column.

    Strips the model prefix so AOTY reasons keep their legacy form
    ("User_Ratings" with prefix "user" -> "ratings").
    """
    label = n_obs_col.lower()
    if prefix:
        marker = f"{prefix.lower()}_"
        if label.startswith(marker):
            label = label[len(marker) :]
    return label


def filter_for_target_model(
    df: pd.DataFrame,
    descriptor: DatasetDescriptor,
    min_obs: int,
    *,
    target: str = "primary",
    logger: AuditLogger | None = None,
) -> pd.DataFrame:
    """
    Filter dataset for modeling one of the descriptor's targets.

    Requires:
    - Non-empty entity and event identifiers
    - Non-null target within descriptor.target_bounds
    - Observation count >= min_obs

    Args:
        df: Cleaned DataFrame (canonical column names).
        descriptor: Dataset descriptor naming the columns and bounds.
        min_obs: Minimum observation count (ratings/reviews analogue).
        target: "primary" (descriptor.target_col) or "secondary".
        logger: Optional audit logger.

    Returns:
        Filtered DataFrame.
    """
    if target == "primary":
        target_col = descriptor.target_col
        n_obs_col = descriptor.n_obs_col
        prefix = descriptor.model_prefix
    elif target == "secondary":
        if descriptor.secondary_target_col is None:
            raise ValueError(
                "Descriptor has no secondary target; cannot filter with target='secondary'."
            )
        target_col = descriptor.secondary_target_col
        n_obs_col = descriptor.secondary_n_obs_col or ""
        prefix = descriptor.secondary_prefix
    else:
        raise ValueError(f"target must be 'primary' or 'secondary', got {target!r}.")

    entity_col = descriptor.entity_col
    event_col = descriptor.event_col
    low, high = descriptor.target_bounds

    # Filter: retain rows with valid identifiers required downstream.
    df = apply_exclusion_filter(
        df,
        condition=_has_nonempty_text(df[entity_col]) & _has_nonempty_text(df[event_col]),
        reason=f"missing_{entity_col.lower()}_or_{event_col.lower()}_identifier",
        logger=logger,
        entity_col=entity_col,
        event_col=event_col,
    )

    # Filter: has valid target score
    df = apply_exclusion_filter(
        df,
        condition=df[target_col].notna(),
        reason=f"missing_{target_col.lower()}",
        logger=logger,
        value_col=target_col,
        entity_col=entity_col,
        event_col=event_col,
    )

    # Filter: score in valid range
    df = apply_exclusion_filter(
        df,
        condition=(df[target_col] >= low) & (df[target_col] <= high),
        reason=f"invalid_{target_col.lower()}_range",
        logger=logger,
        value_col=target_col,
        entity_col=entity_col,
        event_col=event_col,
    )

    # Filter: minimum observation count
    df = apply_exclusion_filter(
        df,
        condition=df[n_obs_col] >= min_obs,
        reason=f"below_min_{_n_obs_reason_label(n_obs_col, prefix)}_{min_obs}",
        logger=logger,
        value_col=n_obs_col,
        entity_col=entity_col,
        event_col=event_col,
    )

    return df
