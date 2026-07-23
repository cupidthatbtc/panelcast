"""Canonical event chronology shared by splitting, features, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from panelcast.data.alignment import ROW_ID_COL

DATE_MISSING_COL = "date_missing"


def normalize_chronology(
    df: pd.DataFrame,
    *,
    entity_col: str,
    date_col: str,
    event_col: str | None = None,
    reject_missing: bool = False,
) -> pd.DataFrame:
    """Normalize dates and return one deterministic repository-wide event order.

    Missing dates sort first and are explicitly marked. Known dates are parsed
    to UTC-naive timestamps, then tied by entity, normalized event key, and
    immutable row identity. Frames without row identity are accepted only when
    those chronology keys are already unique.
    """
    required = [entity_col, date_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing chronology columns: {missing}")

    out = df.copy()
    if bool(out[entity_col].isna().any()):
        raise ValueError(f"Chronology requires non-missing '{entity_col}' values")
    if pd.api.types.is_datetime64_any_dtype(out[date_col].dtype):
        parsed = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    else:
        try:
            parsed = pd.to_datetime(out[date_col], errors="coerce", format="mixed", utc=True)
        except TypeError:  # pandas < 2.0
            parsed = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    out[date_col] = parsed.dt.tz_convert(None)
    out[DATE_MISSING_COL] = out[date_col].isna().astype("int8")
    if reject_missing and bool(out[DATE_MISSING_COL].any()):
        raise ValueError(
            f"Chronology rejected {int(out[DATE_MISSING_COL].sum())} row(s) with "
            f"missing or invalid '{date_col}'"
        )

    helper_event = "__chronology_event_key"
    helper_row = "__chronology_row_key"
    if event_col is not None and event_col in out.columns:
        out[helper_event] = out[event_col].astype("string").fillna("")
    else:
        out[helper_event] = ""
    chronology_keys = [DATE_MISSING_COL, date_col, entity_col, helper_event]
    if ROW_ID_COL in out.columns:
        if out[ROW_ID_COL].isna().any() or out[ROW_ID_COL].duplicated().any():
            raise ValueError(f"Chronology requires unique, non-missing '{ROW_ID_COL}' values")
        try:
            numeric_row_id = pd.to_numeric(out[ROW_ID_COL], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Chronology requires integer '{ROW_ID_COL}' values") from exc
        if (
            not np.isfinite(numeric_row_id).all()
            or not np.equal(numeric_row_id, np.floor(numeric_row_id)).all()
        ):
            raise ValueError(f"Chronology requires integer '{ROW_ID_COL}' values")
        out[helper_row] = numeric_row_id.astype("int64")
        if out[helper_row].duplicated().any():
            raise ValueError(f"Chronology requires unique '{ROW_ID_COL}' values")
    else:
        duplicate_keys = out.duplicated(chronology_keys, keep=False)
        if bool(duplicate_keys.any()):
            raise ValueError(
                f"Chronology ties require immutable row identity column '{ROW_ID_COL}'"
            )
        out[helper_row] = 0

    ordered = out.sort_values(
        [DATE_MISSING_COL, date_col, entity_col, helper_event, helper_row],
        ascending=[False, True, True, True, True],
        kind="stable",
        na_position="first",
    )
    return ordered.drop(columns=[helper_event, helper_row])
