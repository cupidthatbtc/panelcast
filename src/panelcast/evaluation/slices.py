"""Sliced calibration audit: coverage by subgroup with binomial uncertainty.

Global coverage within tolerance can mask offsetting miscalibration
(over-covered veterans hiding under-covered debuts). This module computes
empirical coverage per slice — genre group, review-count decile, target
tercile, training-history bin — with Wilson 95% CIs, flagging slices whose
nominal level falls outside sampling noise. It is also the single
stratification code path: the history-bin metrics that used to live in
``pipelines/evaluate`` are computed here.

Coverage events within a slice share one posterior, so they are not fully
independent and the Wilson CIs are slightly anti-conservative; flags are
informational, and the payload states the expected false-flag count under
perfect calibration so a lone flag is not over-read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from panelcast.evaluation.calibration import compute_pit_values

# Below this many rows a coverage estimate is noise, not evidence.
MIN_SLICE_N = 20

HISTORY_BINS: tuple[tuple[int, int | None], ...] = ((1, 2), (3, 5), (6, 10), (11, None))

_WILSON_Z = 1.959963984540054  # 95%


@dataclass
class SliceCoverage:
    dimension: str
    label: str
    n: int
    levels: dict[str, dict] = field(default_factory=dict)
    pit_max_abs_dev: float | None = None

    @property
    def flagged(self) -> bool:
        return any(lv["flagged"] for lv in self.levels.values())


def wilson_interval(k: int, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (closed form)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def history_bin_labels(train_history: np.ndarray) -> np.ndarray:
    """Map per-row training-event counts onto the shared history bins."""
    counts = np.asarray(train_history, dtype=np.int64)
    labels = np.full(counts.shape, "0", dtype=object)
    for low, high in HISTORY_BINS:
        upper = np.inf if high is None else high
        mask = (counts >= low) & (counts <= upper)
        labels[mask] = f"{low}+" if high is None else f"{low}-{high}"
    return labels


def coverage_by_slice(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    labels: np.ndarray,
    probs: tuple[float, ...],
    dimension: str,
    min_n: int = MIN_SLICE_N,
) -> list[SliceCoverage]:
    """Per-slice empirical coverage with Wilson CIs for one label vector."""
    y_true = np.asarray(y_true, dtype=float)
    labels = np.asarray(labels, dtype=object)
    if labels.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match y_true {y_true.shape[0]}"
        )

    bounds = {}
    for prob in probs:
        a = (1.0 - prob) / 2.0
        lo = np.percentile(y_samples, 100.0 * a, axis=0)
        hi = np.percentile(y_samples, 100.0 * (1.0 - a), axis=0)
        bounds[prob] = (lo, hi)

    out: list[SliceCoverage] = []
    for label in sorted(set(labels.tolist()), key=str):
        mask = labels == label
        n = int(mask.sum())
        if n < min_n:
            continue
        sc = SliceCoverage(dimension=dimension, label=str(label), n=n)
        for prob in probs:
            lo, hi = bounds[prob]
            covered = int(((y_true >= lo) & (y_true <= hi))[mask].sum())
            ci_lo, ci_hi = wilson_interval(covered, n)
            sc.levels[f"{prob:.2f}"] = {
                "nominal": float(prob),
                "empirical": covered / n,
                "wilson_lo": ci_lo,
                "wilson_hi": ci_hi,
                "mean_interval_width": float(np.mean((hi - lo)[mask])),
                "flagged": not (ci_lo <= prob <= ci_hi),
            }
        pit = compute_pit_values(y_true[mask], y_samples[:, mask])
        sc.pit_max_abs_dev = pit["max_abs_dev_from_uniform"]
        out.append(sc)
    return out


def calibration_by_slice(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    row_ids: pd.DataFrame | None,
    probs: tuple[float, ...],
    min_n: int = MIN_SLICE_N,
) -> dict:
    """The full audit across every slice dimension the row identities support.

    Returns a JSON-ready dict: slices, min-n floor, and the expected number
    of false flags under perfect calibration (5% per slice-level test).
    """
    y_true = np.asarray(y_true, dtype=float)
    dimensions: dict[str, np.ndarray] = {}
    if row_ids is not None and len(row_ids) == len(y_true):
        if "group" in row_ids.columns:
            dimensions["group"] = row_ids["group"].to_numpy(dtype=object)
        if "n_reviews" in row_ids.columns:
            deciles = pd.qcut(row_ids["n_reviews"], q=10, duplicates="drop")
            if deciles.nunique() > 1:
                dimensions["n_reviews_decile"] = deciles.astype(str).to_numpy(dtype=object)
        if "train_history" in row_ids.columns:
            dimensions["train_history"] = history_bin_labels(
                row_ids["train_history"].to_numpy()
            )
    terciles = pd.qcut(pd.Series(y_true), q=3, duplicates="drop")
    if terciles.nunique() > 1:
        dimensions["target_tercile"] = terciles.astype(str).to_numpy(dtype=object)

    slices: list[SliceCoverage] = []
    for dimension, labels in dimensions.items():
        slices += coverage_by_slice(
            y_true, y_samples, labels, probs, dimension=dimension, min_n=min_n
        )

    n_tests = sum(len(s.levels) for s in slices)
    return {
        "min_n": min_n,
        "n_slices": len(slices),
        "n_tests": n_tests,
        "expected_false_flags": round(0.05 * n_tests, 2),
        "note": (
            "Wilson 95% CIs; coverage events share one posterior so CIs are "
            "slightly anti-conservative. Under perfect calibration ~5% of "
            "slice-level tests flag by chance — read clusters, not lone flags."
        ),
        "slices": [
            {
                "dimension": s.dimension,
                "label": s.label,
                "n": s.n,
                "levels": s.levels,
                "pit_max_abs_dev": s.pit_max_abs_dev,
                "flagged": s.flagged,
            }
            for s in slices
        ],
    }


def stratified_history_metrics(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    train_history: np.ndarray,
    interval: float,
) -> list[dict]:
    """Accuracy/coverage/width per history bin (legacy metrics.json block).

    Same output schema as the old ``_stratify_metrics_by_history`` so the
    metrics payload is unchanged; rows with no training history fall outside
    every bin, exactly as before.
    """
    labels = history_bin_labels(train_history)

    lo_q = 100.0 * (1.0 - interval) / 2.0
    hi_q = 100.0 - lo_q
    lo = np.percentile(y_samples, lo_q, axis=0)
    hi = np.percentile(y_samples, hi_q, axis=0)
    pred_mean = y_samples.mean(axis=0)

    rows: list[dict] = []
    for low, high in HISTORY_BINS:
        bin_label = f"{low}+" if high is None else f"{low}-{high}"
        mask = labels == bin_label
        n = int(mask.sum())
        if n == 0:
            continue
        yt = np.asarray(y_true, dtype=float)[mask]
        residuals = yt - pred_mean[mask]
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        rows.append(
            {
                "train_albums_bin": bin_label,
                "n": n,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
                "coverage": float(np.mean((yt >= lo[mask]) & (yt <= hi[mask]))),
                "mean_interval_width": float(np.mean(hi[mask] - lo[mask])),
                "interval": interval,
            }
        )
    return rows
