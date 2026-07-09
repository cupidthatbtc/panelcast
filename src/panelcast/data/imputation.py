"""Principled missing-covariate treatment (#158).

Training and evaluation historically imputed missing feature values with raw
0 before standardization, so a missing value in a column whose train mean is
far from zero lands as an extreme pseudo-observation and the model cannot
distinguish "missing" from "measured zero". Gate-on replaces that with
train-median imputation plus ``<col>__missing`` indicator columns; the
recorded medians travel in the training summary's ``feature_scaler`` block so
train, evaluate and predict impute from one shared record. The implicit
assumption is MAR: informative missingness that shifts between train and test
is surfaced by the indicators, not modeled away.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

INDICATOR_SUFFIX = "__missing"


def fit_imputation(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Median-impute the train frame in place; (extended feature_cols, record).

    Medians are recorded for EVERY feature column — a test row can be missing
    in a train-complete column and must still get the train statistic, never
    a statistic of the frame being transformed. Indicator columns are appended
    only for columns with train missingness, so fully-observed blocks keep
    their legacy width.
    """
    collisions = [c for c in feature_cols if c.endswith(INDICATOR_SUFFIX)]
    if collisions:
        raise ValueError(
            f"Feature columns already carry the reserved '{INDICATOR_SUFFIX}' "
            f"suffix: {collisions}. Rename them — generated indicators would collide."
        )
    medians: dict[str, float] = {}
    indicator_cols: list[str] = []
    for col in feature_cols:
        series = df[col]
        median = series.median()
        medians[col] = float(median) if pd.notna(median) else 0.0
        if series.isna().any():
            name = f"{col}{INDICATOR_SUFFIX}"
            df[name] = series.isna().astype(float)
            indicator_cols.append(name)
    df[feature_cols] = df[feature_cols].fillna(medians)
    return feature_cols + indicator_cols, {
        "medians": medians,
        "indicator_cols": indicator_cols,
    }


def apply_imputation(
    df: pd.DataFrame, feature_cols: list[str], imputation: dict[str, Any] | None
) -> pd.DataFrame:
    """Impute a transform-time frame under the recorded train state.

    Mutates ``df`` (fills and adds indicator columns) AND returns it — the
    legacy in-place ``fillna`` contract; no copy is taken.
    ``imputation`` None/empty is the exact legacy path (``fillna(0)``), so
    gate-off outputs stay byte-identical. Gate-on materializes each recorded
    indicator from its base column's NaN mask BEFORE filling, then fills with
    the recorded train medians (0 for a column the record has never seen).
    """
    if not imputation:
        df[feature_cols] = df[feature_cols].fillna(0)
        return df
    indicator_cols = set(imputation.get("indicator_cols", []))
    medians = imputation.get("medians", {})
    for name in indicator_cols:
        base = name[: -len(INDICATOR_SUFFIX)]
        df[name] = df[base].isna().astype(float) if base in df.columns else 0.0
    base_cols = [c for c in feature_cols if c not in indicator_cols]
    df[base_cols] = df[base_cols].fillna({c: medians.get(c, 0) for c in base_cols})
    return df


__all__ = ["INDICATOR_SUFFIX", "apply_imputation", "fit_imputation"]
