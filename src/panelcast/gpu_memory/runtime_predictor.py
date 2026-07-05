"""Fit-runtime prediction from local measurement history (#105, B3).

The unshipped half of #78-B1: sweep and experiment planning kept inheriting
stale timing folklore (the 0.5.0 genre experiment was planned at ~1h from
identity-era numbers and cost ~3.5h under offset_logit). Every fit's
wall-clock now lands in the calibration store; predictions scale from the
measured history and say which source they used.

Runtime is a function of the *model*, not just data size: offset_logit costs
~10x identity at the same draws x n_obs (posterior geometry -> leapfrog steps).
So the estimate is affine and transform-keyed, mirroring the memory refit in
``calibration_store.refit_constants``: a shared startup intercept fit from all
records (so a tiny probe fit reads as evidence about startup, not as a runaway
per-unit rate) plus a per-transform slope from same-transform records.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from panelcast.gpu_memory.calibration_store import load_records

# Cold-start anchors per transform, recorded on the RTX 5090 during the 0.6.0
# cycle (AOTY subset ~5k observations): (total draws, n_obs, seconds). identity
# runs ~10x cheaper than offset_logit at the same draws x n_obs.
_COLD_ANCHORS: dict[str, tuple[tuple[int, int, float], ...]] = {
    "offset_logit": (
        (4 * 2000, 5000, 50 * 60.0),  # diagnostic 4x1000
        (4 * 10000, 5000, 6.3 * 3600.0),  # publication 4x5000
    ),
    "identity": (
        (4 * 2000, 5000, 5 * 60.0),  # diagnostic 4x1000
        (4 * 10000, 5000, 0.63 * 3600.0),  # publication 4x5000
    ),
}
_DEFAULT_TRANSFORM = "offset_logit"
# Points needed to identify a shared startup intercept (mirrors the memory
# refit's min_points); below this, or with no spread in draws x n_obs, the
# estimate collapses to a per-transform rate through the origin.
_MIN_FIXED_POINTS = 5


@dataclass(frozen=True)
class RuntimePrediction:
    seconds: float
    source: str

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


def _total_draws(num_chains: int, num_samples: int, num_warmup: int) -> int:
    return num_chains * (num_samples + num_warmup)


def _fixed_seconds(units: list[float], seconds: list[float]) -> float:
    """Shared startup intercept, least-squares over all usable records.

    An affine fit needs spread in the unit term to separate a fixed cost from a
    per-unit rate; with too few points or a single-unit (degenerate) design we
    can't identify it, so the intercept is 0 and the estimate is a pure
    per-transform rate.
    """
    if len(units) < _MIN_FIXED_POINTS:
        return 0.0
    import numpy as np

    unit_arr = np.asarray(units)
    if float(unit_arr.std()) == 0.0:
        return 0.0
    _, intercept = np.polyfit(unit_arr, np.asarray(seconds), 1)
    return max(float(intercept), 0.0)


def predict_fit_seconds(
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    n_obs: int,
    transform: str | None = None,
    store_path: Path | None = None,
) -> RuntimePrediction:
    """Predicted wall-clock for one fit, from local history when it exists.

    Affine and transform-keyed: ``seconds = FIXED + RATE_t * (draws * n_obs)``,
    where FIXED is a startup cost fit from all records and RATE_t is the median
    residual rate over same-transform records. A transform with no history falls
    back to its own cold-start anchor, never to a mixed all-records rate (fast
    identity fits would otherwise drag an offset_logit estimate low).
    """
    draws = _total_draws(num_chains, num_samples, num_warmup)
    unit = draws * max(n_obs, 1)
    records = [
        r
        for r in load_records(store_path)
        if r.get("wall_clock_seconds") and _record_draws(r) and _record_n_obs(r)
    ]
    units = [float(_record_draws(r) * _record_n_obs(r)) for r in records]
    seconds = [float(r["wall_clock_seconds"]) for r in records]
    fixed = _fixed_seconds(units, seconds)

    matching = [
        r for r in records if transform and r.get("context", {}).get("transform") == transform
    ]
    if matching:
        rate = median(_residual_rate(r, fixed) for r in matching)
        return RuntimePrediction(
            seconds=fixed + rate * unit,
            source=f"local history ({transform}), n={len(matching)}",
        )
    # No same-transform history: scale that model's cold-start anchor.
    label = transform or _DEFAULT_TRANSFORM
    anchors = _COLD_ANCHORS.get(label, _COLD_ANCHORS[_DEFAULT_TRANSFORM])
    anchor_draws, anchor_obs, anchor_seconds = min(anchors, key=lambda a: abs(a[0] - draws))
    return RuntimePrediction(
        seconds=anchor_seconds * (draws / anchor_draws) * (max(n_obs, 1) / anchor_obs),
        source=f"cold-start planning numbers (RTX 5090, {label})",
    )


def _residual_rate(record: dict[str, Any], fixed: float) -> float:
    """Per-unit rate of a record after removing the shared startup cost."""
    unit = _record_draws(record) * _record_n_obs(record)
    return max((float(record["wall_clock_seconds"]) - fixed) / unit, 0.0)


def _record_draws(record: dict[str, Any]) -> int:
    inputs = record.get("estimate_inputs", {})
    try:
        return _total_draws(
            int(inputs["num_chains"]), int(inputs["num_samples"]), int(inputs["num_warmup"])
        )
    except (KeyError, TypeError, ValueError):
        return 0


def _record_n_obs(record: dict[str, Any]) -> int:
    try:
        return int(record.get("estimate_inputs", {}).get("n_observations", 0))
    except (TypeError, ValueError):
        return 0


__all__ = ["RuntimePrediction", "predict_fit_seconds"]
