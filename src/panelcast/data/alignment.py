"""Row-alignment helpers for joining split and feature tables.

Split parquets (written by the splits stage) and feature parquets (written by
the features stage) describe the same rows but are persisted independently.
Joining them by positional index silently misaligns rows if either file is
re-sorted or regenerated. These helpers join on the stable ``original_row_id``
key carried from ingest, with loud failures on any key mismatch.
"""

from __future__ import annotations

import pandas as pd
import structlog

log = structlog.get_logger()

ROW_ID_COL = "original_row_id"


def join_splits_with_features(
    split_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    name: str = "train",
) -> pd.DataFrame:
    """Join a split DataFrame with its feature matrix on ``original_row_id``.

    Falls back to the legacy positional (index-equality) join when either
    frame lacks the key column, e.g. feature parquets produced before the
    key was carried through the features stage.

    Args:
        split_df: Split rows (must carry ``original_row_id`` for keyed join).
        features_df: Feature matrix for the same rows. A keyed join consumes
            its ``original_row_id`` column; the joined result keeps the one
            from ``split_df``.
        name: Label used in log/error messages (e.g. "train", "test").

    Returns:
        Joined DataFrame with split columns first, feature columns appended,
        ordered by ``split_df`` row order (keyed join) or positional index
        (legacy fallback). Overlapping columns keep the features version.

    Raises:
        ValueError: On row-count mismatch, duplicate keys, or key-set
            mismatch between the two frames.
    """
    if len(split_df) != len(features_df):
        raise ValueError(
            f"{name}: row count mismatch between split ({len(split_df)}) "
            f"and features ({len(features_df)}). "
            "Re-run the features stage so both artifacts describe the same rows."
        )

    keyed = ROW_ID_COL in split_df.columns and ROW_ID_COL in features_df.columns
    if not keyed:
        log.warning(
            "positional_feature_join",
            split=name,
            reason=f"'{ROW_ID_COL}' missing from split or features frame",
            hint="Re-run the features stage to enable keyed alignment.",
        )
        if not split_df.index.equals(features_df.index):
            raise ValueError(
                f"{name}: split and features have different indices and no "
                f"'{ROW_ID_COL}' key to align on. Re-run the features stage."
            )
        overlap = [c for c in split_df.columns if c in features_df.columns]
        return split_df.drop(columns=overlap).join(features_df, how="left")

    split_keys = pd.Index(split_df[ROW_ID_COL])
    feature_keys = pd.Index(features_df[ROW_ID_COL])
    if split_keys.has_duplicates or feature_keys.has_duplicates:
        raise ValueError(
            f"{name}: duplicate '{ROW_ID_COL}' values detected "
            f"(split: {split_keys.duplicated().sum()}, "
            f"features: {feature_keys.duplicated().sum()}). "
            "Row identity must be unique to align split and feature rows."
        )
    if not split_keys.sort_values().equals(feature_keys.sort_values()):
        n_only_split = len(split_keys.difference(feature_keys))
        n_only_features = len(feature_keys.difference(split_keys))
        raise ValueError(
            f"{name}: '{ROW_ID_COL}' key sets differ between split and features "
            f"({n_only_split} keys only in split, {n_only_features} only in "
            "features). The artifacts were built from different data; re-run "
            "the features stage."
        )

    left = split_df.set_index(ROW_ID_COL, drop=False)
    right = features_df.set_index(ROW_ID_COL, drop=True)
    overlap = [c for c in left.columns if c in right.columns]
    joined = left.drop(columns=overlap).join(right, how="left")
    joined.index = split_df.index
    return joined
