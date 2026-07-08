"""Conformal calibration wrapper on the Bayesian posterior predictive.

Two layers, both calibrated on the within-entity temporal validation split
(train-only history, test rows never touched):

- Split-conformal interval adjustment (CQR, Romano et al. 2019): conformity
  scores against the predictive interval endpoints give a finite-sample
  widening that guarantees marginal coverage at the nominal level.
- Quantile recalibration (Kuleshov et al. 2018): the empirical distribution
  of calibration PIT values remaps nominal quantile levels, correcting the
  whole predictive CDF rather than one interval.

Caveat, stated wherever these numbers surface: within-entity temporal splits
are not exchangeable in the strict conformal sense — drift between the
validation and test eras weakens the finite-sample guarantee.
"""

from __future__ import annotations

import math

import numpy as np

from panelcast.evaluation.calibration import compute_pit_per_row


def _interval_bounds(samples: np.ndarray, prob: float) -> tuple[np.ndarray, np.ndarray]:
    a = (1.0 - prob) / 2.0
    lo = np.percentile(samples, 100.0 * a, axis=0)
    hi = np.percentile(samples, 100.0 * (1.0 - a), axis=0)
    return lo, hi


def conformal_adjustment(y_cal: np.ndarray, cal_samples: np.ndarray, prob: float) -> float:
    """Finite-sample CQR widening for the equal-tailed interval at ``prob``.

    Conformity scores E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)) are positive
    where the calibration point falls outside its predictive interval. The
    returned adjustment is the ceil((n+1)*prob)/n empirical quantile of the
    scores; adding it to both interval endpoints guarantees marginal coverage
    >= prob under exchangeability. Negative adjustments (over-wide intervals)
    tighten the interval.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    n = len(y_cal)
    if n == 0:
        raise ValueError("empty calibration set")
    lo, hi = _interval_bounds(np.asarray(cal_samples, dtype=float), prob)
    scores = np.maximum(lo - y_cal, y_cal - hi)
    rank = min(math.ceil((n + 1) * prob), n)
    return float(np.sort(scores)[rank - 1])


def conformalized_bounds(
    test_samples: np.ndarray, prob: float, adjustment: float
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed predictive interval widened by the conformal adjustment."""
    lo, hi = _interval_bounds(np.asarray(test_samples, dtype=float), prob)
    return lo - adjustment, hi + adjustment


def recalibrated_levels(pit_cal: np.ndarray, levels: np.ndarray | list[float]) -> np.ndarray:
    """Quantile-recalibration level map from calibration PIT values.

    For a desired CDF level p, the adjusted level p' is the p-th empirical
    quantile of the calibration PITs: by construction the fraction of
    calibration points with PIT <= p' is ~p, so the predictive quantile at p'
    has ~p empirical coverage.
    """
    pit_cal = np.asarray(pit_cal, dtype=float)
    if len(pit_cal) == 0:
        raise ValueError("empty calibration set")
    return np.quantile(pit_cal, np.asarray(levels, dtype=float))


def conformalize(
    y_cal: np.ndarray,
    cal_samples: np.ndarray,
    y_test: np.ndarray,
    test_samples: np.ndarray,
    probs: tuple[float, ...],
) -> dict:
    """Both conformal layers evaluated on the test split, JSON-ready.

    Reports, per nominal level: the CQR adjustment, conformalized empirical
    coverage and mean width, and the recalibrated (Kuleshov) coverage from
    the PIT-remapped quantile levels — next to nothing is refit, so this is
    one predictive pass on the validation split plus array math.
    """
    y_test = np.asarray(y_test, dtype=float)
    test_samples = np.asarray(test_samples, dtype=float)
    pit_cal = compute_pit_per_row(y_cal, cal_samples)

    levels_block: dict[str, dict] = {}
    n_cal = len(pit_cal)
    for prob in probs:
        adjustment = conformal_adjustment(y_cal, cal_samples, prob)
        lo, hi = conformalized_bounds(test_samples, prob, adjustment)
        covered = (y_test >= lo) & (y_test <= hi)
        # ceil((n+1)p) > n means the score quantile clamps at the max and the
        # finite-sample guarantee cannot be met at this level with this n.
        attainable = math.ceil((n_cal + 1) * prob) <= n_cal

        a = (1.0 - prob) / 2.0
        lo_lvl, hi_lvl = recalibrated_levels(pit_cal, [a, 1.0 - a])
        r_lo = np.quantile(test_samples, lo_lvl, axis=0)
        r_hi = np.quantile(test_samples, hi_lvl, axis=0)
        r_covered = (y_test >= r_lo) & (y_test <= r_hi)

        levels_block[f"{prob:.2f}"] = {
            "nominal": float(prob),
            "cqr_adjustment": adjustment,
            "cqr_guarantee_attainable": attainable,
            "cqr_coverage": float(covered.mean()),
            "cqr_mean_width": float(np.mean(hi - lo)),
            "recalibrated_levels": [float(lo_lvl), float(hi_lvl)],
            "recalibrated_coverage": float(r_covered.mean()),
            "recalibrated_mean_width": float(np.mean(r_hi - r_lo)),
        }

    # Full recalibration map at a fixed grid so downstream consumers
    # (predict_next) can remap ANY nominal level without the samples.
    grid = np.round(np.linspace(0.0, 1.0, 101), 2)
    return {
        "n_calibration": int(len(pit_cal)),
        "levels": levels_block,
        "pit_quantile_grid": {
            "levels": grid.tolist(),
            "values": np.quantile(pit_cal, grid).tolist(),
        },
        "note": (
            "Split-conformal guarantee holds under exchangeability; the "
            "within-entity temporal validation/test eras drift, so read the "
            "guarantee as approximate. Calibrated on the validation split "
            "with train-only history (leakage-safe)."
        ),
    }
