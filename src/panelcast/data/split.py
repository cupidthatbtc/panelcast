"""Leak-safe splitting logic for entity event-history prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import GroupShuffleSplit


def within_entity_temporal_split(
    df: pd.DataFrame,
    entity_col: str = "Artist",
    date_col: str = "Release_Date_Parsed",
    test_albums: int = 1,
    val_albums: int = 1,
    min_train_albums: int = 1,
    event_col: str | None = "Album",
    origin_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data holding out the last N events per entity for test/validation.

    This is the PRIMARY evaluation strategy: tests the model's ability to
    predict an entity's next event given its history.

    Args:
        df: Cleaned event DataFrame with entity and date columns
        entity_col: Column name for entity grouping
        date_col: Column name for temporal ordering (Release_Date_Parsed)
        test_albums: Number of most recent events per entity for test set
        val_albums: Number of second-most-recent events per entity for validation
        min_train_albums: Minimum events required in training set per entity
        origin_offset: Rolling-origin backtest offset k: drop each entity's
            last k events entirely (they are the future relative to this
            origin) and hold out the (last-k)-th event as test. 0 reproduces
            the standard split exactly.

    Returns:
        Tuple of (train_df, val_df, test_df)

    Note:
        Entities with fewer than (test_albums + val_albums + min_train_albums
        + origin_offset) events are excluded from all splits — deeper origins
        shrink the eligible entity set.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Artist": ["A"]*4 + ["B"]*3,
        ...     "Album": list(range(7)),
        ...     "Release_Date_Parsed": pd.date_range("2020", periods=7, freq="YS"),
        ... })
        >>> train, val, test = within_entity_temporal_split(df)
        >>> len(train), len(val), len(test)
        (3, 2, 2)
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column for temporal split: '{date_col}'")

    # Sort by entity and date to ensure temporal ordering.
    # Add deterministic tie-breakers to avoid split drift when dates tie.
    # Place missing dates first so unknown chronology is never treated as latest.
    # ASSUMPTION (logged for lineage): events with unknown dates are treated as
    # earliest-in-history, so they always land on the train side and can never
    # reach the held-out test/validation tail.
    n_missing_dates = int(df[date_col].isna().sum())
    if n_missing_dates > 0:
        structlog.get_logger().info(
            "temporal_split_missing_dates",
            n_missing_dates=n_missing_dates,
            assumption="NaT events sorted earliest-in-history (train side)",
            rationale="unknown chronology must never be treated as the latest event",
        )
    if event_col is not None and event_col in df.columns:
        sort_cols = [entity_col, date_col, event_col]
        df_sorted = df.sort_values(sort_cols, na_position="first")
    else:
        df_sorted = (
            df.assign(_row_order=np.arange(len(df)))
            .sort_values([entity_col, date_col, "_row_order"], na_position="first")
            .drop(columns="_row_order")
        )

    if origin_offset < 0:
        raise ValueError(f"origin_offset must be >= 0, got {origin_offset}")

    # Count events per entity
    album_counts = df_sorted.groupby(entity_col).size()
    min_required = test_albums + val_albums + min_train_albums + origin_offset
    valid_artists = album_counts[album_counts >= min_required].index
    df_valid = df_sorted[df_sorted[entity_col].isin(valid_artists)].copy()
    # Exclude entities with fewer dated events than the holdout tail needs:
    # groupby().tail() would otherwise pull NaT rows into val/test, putting
    # events of unknown chronology in the held-out sets. The dropped future
    # events at deeper origins must be dated too, for the same reason.
    n_holdout = test_albums + val_albums + origin_offset
    dated_counts = df_valid.groupby(entity_col)[date_col].transform("count")
    insufficient = dated_counts < max(n_holdout, 1)
    if insufficient.any():
        excluded_entities = df_valid.loc[insufficient, entity_col].unique()
        structlog.get_logger().info(
            "temporal_split_excluded_insufficient_dated_events",
            n_entities=len(excluded_entities),
            required_dated_events=n_holdout,
            rationale="held-out val/test events must have known chronology",
        )
        df_valid = df_valid[~insufficient].copy()

    # Rolling origin: everything after the k-th-from-last event is the
    # future relative to this origin and must not exist anywhere.
    if origin_offset > 0:
        future_df = df_valid.groupby(entity_col).tail(origin_offset)
        df_valid = df_valid.drop(future_df.index)
        structlog.get_logger().info(
            "temporal_split_origin_offset",
            origin_offset=origin_offset,
            n_future_rows_dropped=len(future_df),
            n_entities=df_valid[entity_col].nunique(),
        )

    # Extract last N per entity for test
    test_df = df_valid.groupby(entity_col).tail(test_albums)
    remaining = df_valid.drop(test_df.index)

    # Extract second-to-last N per entity for validation
    val_df = remaining.groupby(entity_col).tail(val_albums)
    train_df = remaining.drop(val_df.index)

    return train_df, val_df, test_df


def entity_disjoint_split(
    df: pd.DataFrame,
    entity_col: str = "Artist",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring no entity appears in multiple splits.

    This is the SECONDARY evaluation strategy: tests the model's ability to
    generalize to unseen entities (cold-start evaluation).

    Uses two-stage GroupShuffleSplit:
    1. Split test set (entity-disjoint)
    2. Split validation from remaining (entity-disjoint)

    Args:
        df: Cleaned event DataFrame
        entity_col: Column name for entity grouping
        test_size: Proportion of data for test set (by entity groups)
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Artist": [f"Artist_{i//3}" for i in range(60)],
        ...     "Album": list(range(60)),
        ...     "Score": [70]*60,
        ... })
        >>> train, val, test = entity_disjoint_split(df, random_state=42)
        >>> # No entity overlap between splits
        >>> train_a = set(train["Artist"])
        >>> test_a = set(test["Artist"])
        >>> len(train_a & test_a)
        0
    """
    groups = df[entity_col].values

    # Stage 1: Separate test set
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    test_df = df.iloc[test_idx].copy()
    train_val_df = df.iloc[train_val_idx]

    # Stage 2: Separate validation from train
    val_proportion = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_proportion,
        random_state=random_state + 1,  # Different seed for second split
    )
    train_val_groups = train_val_df[entity_col].values
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    return train_df, val_df, test_df


def assert_no_artist_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    entity_col: str = "Artist",
) -> None:
    """
    Verify no entity appears in multiple splits (entity-disjoint property).

    Raises:
        ValueError: If any entity overlap is detected

    Note:
        This should ONLY be called for entity-disjoint splits.
        Within-entity temporal splits intentionally have entity overlap.
    """
    train_artists = set(train_df[entity_col])
    val_artists = set(val_df[entity_col])
    test_artists = set(test_df[entity_col])

    overlap_train_val = train_artists & val_artists
    overlap_train_test = train_artists & test_artists
    overlap_val_test = val_artists & test_artists

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Entity overlap detected: "
            f"train-val={len(overlap_train_val)}, "
            f"train-test={len(overlap_train_test)}, "
            f"val-test={len(overlap_val_test)}"
        )


def validate_temporal_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    entity_col: str = "Artist",
    date_col: str = "Release_Date_Parsed",
) -> None:
    """
    Verify temporal ordering is correct for within-entity splits.

    For each entity, checks that:
    - Test events are chronologically after or equal to validation events
    - Validation events are chronologically after or equal to training events

    Note:
        Same-date events are allowed since the split function uses groupby.tail()
        which provides consistent ordering. Only strictly backwards ordering
        (training data after test data) is flagged as a violation.

    Raises:
        ValueError: If temporal ordering is violated (train after test), or if
            any held-out (validation/test) row has a missing parsed date
    """
    # Held-out rows must have known chronology; dropping NaT before the
    # per-entity comparison below would hide unknown-date leakage.
    for split_name, split_df in (("validation", val_df), ("test", test_df)):
        n_missing = int(split_df[date_col].isna().sum())
        if n_missing:
            raise ValueError(
                f"Temporal validation failed: {n_missing} {split_name} row(s) have "
                f"missing parsed release dates (NaT); held-out events must be dated."
            )

    # Get entities that appear in all splits (expected for temporal split)
    train_artists = set(train_df[entity_col])
    test_artists = set(test_df[entity_col])
    val_artists = set(val_df[entity_col])

    # Only validate entities present in both train and test
    common_artists = train_artists & test_artists

    for artist in common_artists:
        train_dates = train_df[train_df[entity_col] == artist][date_col].dropna()
        test_dates = test_df[test_df[entity_col] == artist][date_col].dropna()
        train_max = train_dates.max() if not train_dates.empty else pd.NaT
        test_min = test_dates.min() if not test_dates.empty else pd.NaT

        # Test holdout must have at least one known date per entity for temporal validation.
        if pd.isna(test_min):
            raise ValueError(
                f"Temporal validation failed for {artist}: missing parsed release dates "
                f"(train_max={train_max}, test_min={test_min})."
            )

        # Strict check: training data must not come AFTER test data
        # Same-date events are OK (tail() provides consistent ordering)
        if pd.notna(train_max) and train_max > test_min:
            raise ValueError(
                f"Temporal violation for {artist}: "
                f"train max date {train_max} > test min date {test_min}"
            )

        # Check validation if entity present
        if artist in val_artists:
            val_dates = val_df[val_df[entity_col] == artist][date_col].dropna()
            val_min = val_dates.min() if not val_dates.empty else pd.NaT
            val_max = val_dates.max() if not val_dates.empty else pd.NaT

            if pd.notna(train_max) and pd.notna(val_min) and train_max > val_min:
                raise ValueError(
                    f"Temporal violation for {artist}: train max {train_max} > val min {val_min}"
                )
            if pd.notna(val_max) and val_max > test_min:
                raise ValueError(
                    f"Temporal violation for {artist}: val max {val_max} > test min {test_min}"
                )
