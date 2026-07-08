"""Per-row and per-slice error decomposition over the identified predictions artifact.

Consumes the predictions.json payload the evaluate stage writes (entity/event
identities plus per-row predictive sd, PIT, and coverage flags — #180) and
turns a headline MAE into the rows and slices that drive it. Read-only over
existing artifacts: no samples, no model, no refit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Observed value in the far tails of its own predictive distribution.
_PIT_TAIL = 0.025

_IDENTITY_KEYS = ("entity", "n_reviews", "train_history")


@dataclass
class ErrorDecomposition:
    rows: pd.DataFrame
    rollups: dict[str, pd.DataFrame] = field(default_factory=dict)


def _require_identified(predictions: dict) -> None:
    missing = [k for k in _IDENTITY_KEYS if k not in predictions]
    if missing:
        raise ValueError(
            f"predictions payload lacks identity fields {missing}; it predates the "
            "identified-predictions schema. Re-run the evaluate stage "
            "(panelcast run --stages evaluate) to regenerate it."
        )


def _rollup(rows: pd.DataFrame, by: str, covered_cols: list[str]) -> pd.DataFrame:
    grouped = rows.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mae": grouped["abs_residual"].mean(),
            "rmse": grouped["residual"].apply(lambda r: float(np.sqrt(np.mean(r**2)))),
            "bias": grouped["residual"].mean(),
            "sq_error_share": grouped["sq_error_share"].sum(),
        }
    )
    for col in covered_cols:
        out[f"coverage_{col.removeprefix('covered_')}"] = grouped[col].mean()
    return out.sort_values("sq_error_share", ascending=False)


def decompose_errors(predictions: dict) -> ErrorDecomposition:
    """Per-row error frame + entity / group / review-count rollups.

    Raises ValueError on payloads that predate the identified schema, so
    callers can degrade with a clear message instead of a KeyError.
    """
    _require_identified(predictions)

    y_true = np.asarray(predictions["y_true"], dtype=float)
    y_pred = np.asarray(predictions["y_pred_mean"], dtype=float)
    residual = y_true - y_pred

    rows = pd.DataFrame(
        {
            "entity": predictions["entity"],
            "y_true": y_true,
            "y_pred_mean": y_pred,
            "residual": residual,
            "abs_residual": np.abs(residual),
        }
    )
    if "event" in predictions:
        rows.insert(1, "event", predictions["event"])
    if "group" in predictions:
        rows["group"] = predictions["group"]
    rows["n_reviews"] = np.asarray(predictions["n_reviews"], dtype=int)
    rows["train_history"] = np.asarray(predictions["train_history"], dtype=int)

    sq = residual**2
    total_sq = float(sq.sum())
    rows["sq_error_share"] = sq / total_sq if total_sq > 0 else 0.0

    if "y_pred_sd" in predictions:
        sd = np.asarray(predictions["y_pred_sd"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rows["std_residual"] = np.where(sd > 0, residual / sd, np.nan)
    if "pit" in predictions:
        pit = np.asarray(predictions["pit"], dtype=float)
        rows["pit"] = pit
        rows["miscalibrated"] = (pit < _PIT_TAIL) | (pit > 1.0 - _PIT_TAIL)

    covered_cols: list[str] = []
    for level, flags in (predictions.get("covered") or {}).items():
        col = f"covered_{level}"
        rows[col] = np.asarray(flags, dtype=bool)
        covered_cols.append(col)

    rows = rows.sort_values("abs_residual", ascending=False).reset_index(drop=True)

    rollups = {"entity": _rollup(rows, "entity", covered_cols)}
    if "group" in rows.columns:
        rollups["group"] = _rollup(rows, "group", covered_cols)
    deciles = pd.qcut(rows["n_reviews"], q=10, duplicates="drop")
    if deciles.nunique() > 1:
        rollups["n_reviews_decile"] = _rollup(
            rows.assign(n_reviews_decile=deciles.astype(str)), "n_reviews_decile", covered_cols
        )

    return ErrorDecomposition(rows=rows, rollups=rollups)
