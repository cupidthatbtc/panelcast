"""Fit-runtime prediction from local measurement history (#105, B3).

The unshipped half of #78-B1: sweep and experiment planning kept inheriting
stale timing folklore (the 0.5.0 genre experiment was planned at ~1h from
identity-era numbers and cost ~3.5h under offset_logit). Every fit's
wall-clock now lands in the calibration store; predictions scale from the
measured history and say which source they used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from panelcast.gpu_memory.calibration_store import load_records

# Cold-start anchors recorded on the RTX 5090 during the 0.6.0 cycle
# (offset_logit defaults, AOTY subset ~5k observations).
_COLD_ANCHORS = (
    # (total draws = chains * (warmup + samples), n_obs, seconds)
    (4 * 2000, 5000, 50 * 60.0),  # diagnostic 4x1000
    (4 * 10000, 5000, 6.3 * 3600.0),  # publication 4x5000
)
_MIN_HISTORY = 3


@dataclass(frozen=True)
class RuntimePrediction:
    seconds: float
    source: str

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


def _total_draws(num_chains: int, num_samples: int, num_warmup: int) -> int:
    return num_chains * (num_samples + num_warmup)


def predict_fit_seconds(
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    n_obs: int,
    transform: str | None = None,
    store_path: Path | None = None,
) -> RuntimePrediction:
    """Predicted wall-clock for one fit, from local history when it exists.

    History model: seconds scale with total draws x n_obs (per-record rates,
    median across the matching-transform subset — falling back to all
    records, then to the cold-start anchors). Deliberately simple; the point
    is measured-over-folklore, not a perfect cost model.
    """
    draws = _total_draws(num_chains, num_samples, num_warmup)
    records = [
        r
        for r in load_records(store_path)
        if r.get("wall_clock_seconds") and _record_draws(r) and r.get("estimate_inputs")
    ]
    matching = [
        r for r in records if transform and r.get("context", {}).get("transform") == transform
    ]
    pool = matching if len(matching) >= _MIN_HISTORY else records
    if len(pool) >= _MIN_HISTORY:
        rates = sorted(
            r["wall_clock_seconds"] / (_record_draws(r) * _record_n_obs(r))
            for r in pool
            if _record_n_obs(r)
        )
        if len(rates) >= _MIN_HISTORY:
            rate = rates[len(rates) // 2]
            label = "local history" if pool is not matching else f"local history ({transform})"
            return RuntimePrediction(
                seconds=rate * draws * max(n_obs, 1),
                source=f"{label}, n={len(rates)}",
            )
    # Cold start: scale from the nearer anchor by draw and size ratios.
    anchor = min(_COLD_ANCHORS, key=lambda a: abs(a[0] - draws))
    anchor_draws, anchor_obs, anchor_seconds = anchor
    seconds = anchor_seconds * (draws / anchor_draws) * (max(n_obs, 1) / anchor_obs)
    return RuntimePrediction(
        seconds=seconds,
        source="cold-start planning numbers (RTX 5090, offset_logit defaults)",
    )


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
