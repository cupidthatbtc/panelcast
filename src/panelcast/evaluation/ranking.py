"""Decision-oriented ranking metrics: top-K quality and rank calibration.

Scores the model on the ordinal question users actually ask — who tops the
next slate — rather than per-event accuracy. The posterior draws give every
row a full rank distribution, so P(top-K) and expected rank come essentially
free; the rank probabilities users would act on are themselves audited with
a reliability curve.

Metrics are against realized observed scores, not latent quality, and a
single slate is high-variance (precision@10 moves in steps of 0.1) — read
them as descriptive until the rolling-origin backtest supplies multiple
slates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Ties in scores are broken by row order (stable sort), so ranks are
# deterministic for discrete/rounded scores across runs.
TIE_BREAK_RULE = "stable sort: ties broken by row order (entity/date-sorted upstream)"

_CALIBRATION_BINS = 5


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    """Rank 1 = highest value; ties resolved by row order (stable)."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _rank_probability_calibration(
    p_topk: np.ndarray, realized_topk: np.ndarray, n_bins: int = _CALIBRATION_BINS
) -> dict:
    """Reliability curve for P(top-K): binned predicted vs realized frequency."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_topk, edges[1:-1]), 0, n_bins - 1)
    predicted, observed, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        predicted.append(float(p_topk[mask].mean()))
        observed.append(float(realized_topk[mask].mean()))
        counts.append(n)
    return {
        "predicted_probs": predicted,
        "observed_freq": observed,
        "counts": counts,
        "bin_edges": edges.tolist(),
    }


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    entities: np.ndarray | list[str] | None = None,
    ks: tuple[int, ...] = (5, 10, 25),
) -> tuple[dict, pd.DataFrame]:
    """Ranking metrics plus the ranked-slate frame.

    Returns ``(metrics, slate)``: the JSON-ready metrics block and a per-row
    frame (entity, y_true, pred_mean, expected_rank, realized_rank, and one
    p_top{K} column per usable K) sorted by predicted rank. Ks >= n rows are
    reported as null rather than crashing small domains.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_samples = np.asarray(y_samples, dtype=float)
    n = len(y_true)
    if n == 0:
        raise ValueError("empty slate: no rows to rank")
    pred_mean = y_samples.mean(axis=0)

    spearman = stats.spearmanr(pred_mean, y_true)
    kendall = stats.kendalltau(pred_mean, y_true)

    realized_rank = _ranks_desc(y_true)
    predicted_rank = _ranks_desc(pred_mean)

    # Per-draw ranks give each row a rank distribution.
    draw_ranks = np.empty_like(y_samples, dtype=np.int64)
    for d in range(y_samples.shape[0]):
        draw_ranks[d] = _ranks_desc(y_samples[d])
    expected_rank = draw_ranks.mean(axis=0)

    slate = pd.DataFrame(
        {
            "entity": np.asarray(entities, dtype=object) if entities is not None else "",
            "y_true": y_true,
            "pred_mean": pred_mean,
            "predicted_rank": predicted_rank,
            "expected_rank": expected_rank,
            "realized_rank": realized_rank,
        }
    )

    top_k: dict[str, dict | None] = {}
    for k in ks:
        if k >= n:
            top_k[str(k)] = None
            continue
        pred_top = predicted_rank <= k
        real_top = realized_rank <= k
        hits = int((pred_top & real_top).sum())
        p_topk = (draw_ranks <= k).mean(axis=0)
        slate[f"p_top{k}"] = p_topk
        top_k[str(k)] = {
            "precision": hits / k,
            "recall": hits / k,  # |pred top-K| == |realized top-K| == K
            "calibration": _rank_probability_calibration(p_topk, real_top),
        }

    metrics = {
        "n": n,
        "spearman": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_pvalue": float(kendall.pvalue),
        "top_k": top_k,
        "tie_break": TIE_BREAK_RULE,
        "note": (
            "Ranks are against realized observed scores on one slate; "
            "top-K metrics are high-variance until backtesting supplies "
            "multiple slates."
        ),
    }
    return metrics, slate.sort_values("predicted_rank").reset_index(drop=True)
