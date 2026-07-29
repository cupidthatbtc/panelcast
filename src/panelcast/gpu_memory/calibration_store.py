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
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

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
# Slack allowed when re-checking the envelope on the returned constants: the
# scale is solved in closed form, so only float rounding should ever show up.
_ENVELOPE_TOLERANCE = 1e-9


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
        values = {
            name: inputs[name]
            for name in (
                "n_observations",
                "n_features",
                "n_artists",
                "max_seq",
                "num_chains",
                "num_samples",
            )
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
            return None
        n_obs = values["n_observations"]
        n_features = values["n_features"]
        n_artists = values["n_artists"]
        max_seq = values["max_seq"]
        num_chains = values["num_chains"]
        num_samples = values["num_samples"]
        if min(n_obs, n_artists, max_seq, num_chains, num_samples) <= 0 or n_features < 0:
            return None
        chain_method = inputs.get("chain_method", "sequential")
        if chain_method not in ("sequential", "parallel", "vectorized"):
            return None
        gate_names = (
            "exclude_rw_raw_from_collection",
            "errors_in_variables",
            "heteroscedastic_entity_obs",
            "entity_group_pooling",
        )
        if any(name in inputs and not isinstance(inputs[name], bool) for name in gate_names):
            return None
        n_groups = inputs.get("n_groups", 0)
        if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 0:
            return None
        n_params, collected = _count_params(
            n_observations=n_obs,
            n_features=n_features,
            n_artists=n_artists,
            max_seq=max_seq,
            exclude_rw_raw_from_collection=inputs.get("exclude_rw_raw_from_collection", False),
            errors_in_variables=inputs.get("errors_in_variables", False),
            heteroscedastic_entity_obs=inputs.get("heteroscedastic_entity_obs", False),
            entity_group_pooling=inputs.get("entity_group_pooling", False),
            n_groups=n_groups,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    gib = 1024**3
    live_chains = num_chains if chain_method == "vectorized" else 1
    raw_base = (n_params * 4 * 4 * live_chains + n_obs * n_features * 4) / gib
    unit = collected * num_samples * 4 * num_chains / gib
    if not (_is_finite(raw_base) and _is_finite(unit)) or raw_base < 0.0 or unit < 0.0:
        return None
    return raw_base, unit


def _is_finite(value: Any) -> TypeGuard[float]:
    """A real, finite number — bools and NaN/inf telemetry are not measurements."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _usable_points(records: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    """(bases, units, actuals) for records that are real, positive measurements."""
    bases: list[float] = []
    units: list[float] = []
    actuals: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        actual = record.get("actual_peak_gb")
        if not _is_finite(actual) or float(actual) <= 0.0:
            continue
        inputs = record.get("estimate_inputs", {})
        terms = _linear_terms(inputs) if isinstance(inputs, dict) else None
        if terms is None:
            continue
        bases.append(terms[0])
        units.append(terms[1])
        actuals.append(float(actual))
    return bases, units, actuals


def _envelope_scale(
    base_arr: Any,
    unit_arr: Any,
    actual_arr: Any,
    factor: float,
    fixed: float,
    jit_buffer_percent: float,
) -> float | None:
    """Smallest s >= 1 making every point over-covered, or None if none exists.

    Scaling both constants by s gives, per point,
    ``(1+jbp) * (base + s*(fixed + factor*unit)) >= envelope * actual``,
    which is linear in s — so the binding constraint solves directly instead
    of being chased by a fixed number of shrinking correction passes. A point
    whose scalable term is zero cannot be lifted at all, and no amount of
    iteration would have found that out.
    """
    import numpy as np

    required = _MIN_LOCAL_ENVELOPE * actual_arr / (1.0 + jit_buffer_percent) - base_arr
    scalable = fixed + factor * unit_arr
    binding = required > 0.0
    if bool(np.any(binding & (scalable <= 0.0))):
        return None
    ratios = np.where(binding, required / np.where(scalable > 0.0, scalable, 1.0), 0.0)
    scale = float(np.max(ratios))
    if not math.isfinite(scale):
        return None
    if scale <= 1.0:
        return 1.0
    # A hair of overshoot so the binding point cannot land under the envelope
    # by one ulp; the caller's re-check is what actually decides.
    return scale * (1.0 + 1e-12)


def _min_envelope_ratio(
    base_arr: Any,
    unit_arr: Any,
    actual_arr: Any,
    factor: float,
    fixed: float,
    jit_buffer_percent: float,
) -> float:
    """Tightest estimate/actual over the fit set under the given constants."""
    estimate = (1.0 + jit_buffer_percent) * (base_arr + fixed + factor * unit_arr)
    return float((estimate / actual_arr).min())


def refit_constants(
    records: list[dict[str, Any]],
    min_points: int = 5,
    jit_buffer_percent: float = 0.10,
) -> PerMachineCalibration | None:
    """Least-squares refit of (FACTOR, FIXED) on this machine's measurements.

    Returns None — meaning "keep the shipped constants" — whenever the result
    cannot be shown to be safe: too few usable points, no spread in the
    collection term (a degenerate design can't identify two constants),
    non-finite fitted coefficients, or an envelope that the final constants
    fail to satisfy. Never-under is the contract, so it is verified on the
    numbers actually returned rather than assumed from the fitting procedure.
    """
    import numpy as np

    if not _is_finite(jit_buffer_percent) or jit_buffer_percent < 0.0:
        return None

    bases, units, actuals = _usable_points(records)
    if len(units) < min_points:
        return None
    base_arr = np.asarray(bases, dtype=float)
    unit_arr = np.asarray(units, dtype=float)
    actual_arr = np.asarray(actuals, dtype=float)
    if not float(unit_arr.std()) > 0.0:
        return None

    y = actual_arr / (1.0 + jit_buffer_percent) - base_arr
    try:
        slope, intercept = np.polyfit(unit_arr, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not (_is_finite(float(slope)) and _is_finite(float(intercept))):
        return None
    factor = max(float(slope), 0.1)
    fixed = 0.0 if abs(float(intercept)) < 1e-12 else max(float(intercept), 0.0)

    scale = _envelope_scale(
        base_arr, unit_arr, actual_arr, factor, fixed, jit_buffer_percent
    )
    if scale is None:
        return None
    factor *= scale
    fixed *= scale
    if not (_is_finite(factor) and _is_finite(fixed)):
        return None

    min_ratio = _min_envelope_ratio(
        base_arr, unit_arr, actual_arr, factor, fixed, jit_buffer_percent
    )
    if not _is_finite(min_ratio) or min_ratio < _MIN_LOCAL_ENVELOPE * (1.0 - _ENVELOPE_TOLERANCE):
        logger.warning(
            "local GPU calibration failed its over-coverage check "
            "(min ratio %.4f < %.4f); keeping the shipped constants",
            min_ratio,
            _MIN_LOCAL_ENVELOPE,
        )
        return None
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
