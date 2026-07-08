"""Per-machine GPU-memory self-calibration (#105, B3).

Every fit records {estimate_inputs, actual_peak} via the run-manifest
telemetry (#88); this module accumulates those pairs in a local store and
refits the two estimator constants for THIS machine once enough points exist.
The shipped constants stay the cold-start default — a machine with history
earns tighter numbers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from panelcast.gpu_memory.estimate import (
    COLLECTION_OVERHEAD_FACTOR,
    FIXED_OVERHEAD_GB,
    _count_params,
    estimate_memory_gb,
)

logger = logging.getLogger(__name__)

_STORE_VERSION = 1
_MAX_RECORDS = 200
# Local refits stay conservative: every stored point must be over-covered by
# at least this ratio (mirrors the never-under ladder discipline of #104).
_MIN_LOCAL_ENVELOPE = 1.05


def default_store_path() -> Path:
    return Path.home() / ".panelcast" / "gpu_calibration.json"


@dataclass(frozen=True)
class PerMachineCalibration:
    """Constants refit on this machine's own measurements."""

    collection_overhead_factor: float
    fixed_overhead_gb: float
    n_points: int
    min_ratio: float  # tightest estimate/actual over the fit set, post-envelope


def append_record(
    estimate_inputs: dict[str, Any],
    expected_gb: float,
    actual_peak_gb: float | None,
    wall_clock_seconds: float | None,
    context: dict[str, Any] | None = None,
    path: Path | None = None,
) -> None:
    """Best-effort append; telemetry must never break a fit."""
    path = path or default_store_path()
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "estimate_inputs": estimate_inputs,
        "expected_gb": expected_gb,
        "actual_peak_gb": actual_peak_gb,
        "wall_clock_seconds": wall_clock_seconds,
        "context": context or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _read_records_for_append(path)
    if records is None:
        logger.warning(
            "GPU calibration store %s unreadable; skipping append to avoid "
            "rewriting the store from an empty read",
            path,
        )
        return
    records.append(record)
    payload = {"version": _STORE_VERSION, "records": records[-_MAX_RECORDS:]}
    # Per-process tmp name: concurrent appenders must not share one tmp file.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_records_for_append(path: Path) -> list[dict[str, Any]] | None:
    """Records for the append's read-modify-write; None = present but unreadable.

    A transient read failure (Windows sharing violation during another
    process's os.replace, AV hold) must not be collapsed to an empty list —
    the rewrite would silently destroy the accumulated history. Missing store
    is fine (fresh start); corrupt JSON is real corruption (writes are
    atomic), so rewriting heals it.
    """
    for delay in (0.05, 0.1, 0.2):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except ValueError:
            return []
        except OSError:
            time.sleep(delay)
            continue
        return _records_from_payload(payload)
    return None


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Records list from a parsed store payload; non-dict payloads (a JSON
    array, null, bare string) are corruption too — treat as empty."""
    if not isinstance(payload, dict):
        return []
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Read-only load; any failure falls back to no history (shipped constants)."""
    path = path or default_store_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _records_from_payload(payload)


def _linear_terms(inputs: dict[str, Any]) -> tuple[float, float] | None:
    """(raw_base_gb_without_fixed, collection_unit_gb) mirroring estimate.py.

    The estimator is linear in the two constants:
    total = (1 + jbp) * (raw_base + FIXED + FACTOR * unit), so the local refit
    is a straight least-squares on these terms. Structural gate flags are
    folded into the terms themselves (via the shared _count_params), so
    records fit under different gates stay comparable; records that predate
    the flags being recorded are read as gate-off.
    """
    try:
        n_obs = int(inputs["n_observations"])
        n_features = int(inputs["n_features"])
        n_artists = int(inputs["n_artists"])
        max_seq = int(inputs["max_seq"])
        num_chains = int(inputs["num_chains"])
        num_samples = int(inputs["num_samples"])
        n_params, collected = _count_params(
            n_observations=n_obs,
            n_features=n_features,
            n_artists=n_artists,
            max_seq=max_seq,
            exclude_rw_raw_from_collection=bool(
                inputs.get("exclude_rw_raw_from_collection", False)
            ),
            errors_in_variables=bool(inputs.get("errors_in_variables", False)),
            heteroscedastic_entity_obs=bool(inputs.get("heteroscedastic_entity_obs", False)),
            entity_group_pooling=bool(inputs.get("entity_group_pooling", False)),
            n_groups=int(inputs.get("n_groups", 0) or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None
    gib = 1024**3
    raw_base = (n_params * 4 * 4 + n_obs * n_features * 4) / gib
    unit = collected * num_samples * 4 * num_chains / gib
    return raw_base, unit


def refit_constants(
    records: list[dict[str, Any]],
    min_points: int = 5,
    jit_buffer_percent: float = 0.10,
) -> PerMachineCalibration | None:
    """Least-squares refit of (FACTOR, FIXED) on this machine's measurements.

    Returns None when there are too few usable points or no spread in the
    collection term (a degenerate design can't identify two constants).
    The result is inflated so every local point stays over-covered by
    ``_MIN_LOCAL_ENVELOPE`` — never-under is part of the contract.
    """
    import numpy as np

    bases, units, actuals = [], [], []
    for record in records:
        actual = record.get("actual_peak_gb")
        terms = _linear_terms(record.get("estimate_inputs", {}))
        if not actual or terms is None or actual <= 0:
            continue
        bases.append(terms[0])
        units.append(terms[1])
        actuals.append(float(actual))
    if len(units) < min_points:
        return None
    base_arr = np.asarray(bases)
    unit_arr = np.asarray(units)
    actual_arr = np.asarray(actuals)
    if float(unit_arr.std()) == 0.0:
        return None
    y = actual_arr / (1.0 + jit_buffer_percent) - base_arr
    slope, intercept = np.polyfit(unit_arr, y, 1)
    factor = max(float(slope), 0.1)
    fixed = max(float(intercept), 0.0)

    def _min_ratio(f: float, c: float) -> float:
        est = (1.0 + jit_buffer_percent) * (base_arr + c + f * unit_arr)
        return float((est / actual_arr).min())

    # Inflate minimally until every local point is over-covered. The scale
    # only touches the two constants (not the raw model term), so one pass
    # can land a hair short — iterate.
    min_ratio = _min_ratio(factor, fixed)
    for _ in range(5):
        if min_ratio >= _MIN_LOCAL_ENVELOPE:
            break
        # The 1e-6 overshoot breaks the asymptotic approach from below.
        scale = _MIN_LOCAL_ENVELOPE / min_ratio * (1.0 + 1e-6)
        factor *= scale
        fixed *= scale
        min_ratio = _min_ratio(factor, fixed)
    return PerMachineCalibration(
        collection_overhead_factor=factor,
        fixed_overhead_gb=fixed,
        n_points=len(units),
        min_ratio=min_ratio,
    )


def resolve_calibration(
    path: Path | None = None,
    min_points: int = 5,
) -> tuple[float, float, str]:
    """(factor, fixed, source) — per-machine when history suffices, else shipped."""
    calibration = refit_constants(load_records(path), min_points=min_points)
    if calibration is None:
        return (
            COLLECTION_OVERHEAD_FACTOR,
            FIXED_OVERHEAD_GB,
            "shipped constants (no local calibration history)",
        )
    return (
        calibration.collection_overhead_factor,
        calibration.fixed_overhead_gb,
        f"per-machine calibration ({calibration.n_points} local fits, "
        f"min over-coverage {calibration.min_ratio:.2f}x)",
    )


def estimate_with_calibration(path: Path | None = None, **estimate_kwargs: Any):
    """estimate_memory_gb under this machine's calibration; returns (estimate, source)."""
    factor, fixed, source = resolve_calibration(path)
    estimate = estimate_memory_gb(
        collection_overhead_factor=factor,
        fixed_overhead_gb=fixed,
        **estimate_kwargs,
    )
    return estimate, source


__all__ = [
    "PerMachineCalibration",
    "append_record",
    "estimate_with_calibration",
    "load_records",
    "refit_constants",
    "resolve_calibration",
]
