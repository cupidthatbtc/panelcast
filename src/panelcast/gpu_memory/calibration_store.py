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
from collections import Counter
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

_STORE_VERSION = 2
_MAX_RECORDS = 200
# Local refits stay conservative: every stored point must be over-covered by
# at least this ratio (mirrors the never-under ladder discipline of #104).
_MIN_LOCAL_ENVELOPE = 1.05
# Slack allowed when re-checking the envelope on the returned constants: the
# scale is solved in closed form, so only float rounding should ever show up.
_ENVELOPE_TOLERANCE = 1e-9
_PEAK_PROVENANCE_VERSION = 1
_TRUSTED_ATTRIBUTION = "fit_interval_new_process_peak"
# Quarantine reasons double as log text and counter keys, so a producer
# attribution only survives into them if it is one this vocabulary knows —
# the store is a plain on-disk file and anything else is untrusted text.
_UNTRUSTED_ATTRIBUTIONS = frozenset(
    {
        "overlapping_fit_interval",
        "missing_start_snapshot",
        "allocator_identity_changed",
        "stale_process_peak",
    }
)
# Record ids are free-form on disk; the diagnostics line names a few of them.
_MAX_LABEL_CHARS = 64


def default_store_path() -> Path:
    return Path.home() / ".panelcast" / "gpu_calibration.json"


@dataclass(frozen=True)
class PerMachineCalibration:
    """Constants refit on this machine's own measurements."""

    collection_overhead_factor: float
    fixed_overhead_gb: float
    n_points: int
    min_ratio: float
    envelope_inflation: float = 1.0
    quarantined_points: int = 0
    invalid_points: int = 0
    quarantine_reasons: tuple[tuple[str, int], ...] = ()
    quarantined_records: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CalibrationDiagnostics:
    total_records: int
    trusted_points: int
    quarantined_points: int
    invalid_points: int
    quarantine_reasons: tuple[tuple[str, int], ...]
    quarantined_records: tuple[tuple[str, str], ...]
    envelope_inflation: float | None = None


@dataclass(frozen=True)
class _CalibrationPoint:
    base: float
    unit: float
    actual: float


def append_record(
    estimate_inputs: dict[str, Any],
    expected_gb: float,
    actual_peak_gb: float | None,
    wall_clock_seconds: float | None,
    context: dict[str, Any] | None = None,
    path: Path | None = None,
    peak_provenance: dict[str, Any] | None = None,
) -> None:
    """Best-effort append; telemetry must never break a fit."""
    path = path or default_store_path()
    record = {
        "record_id": f"{os.getpid()}-{time.time_ns()}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "estimate_inputs": estimate_inputs,
        "expected_gb": expected_gb,
        "actual_peak_gb": actual_peak_gb,
        "wall_clock_seconds": wall_clock_seconds,
        "context": context or {},
        "peak_provenance": peak_provenance,
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
        n_entities = values["n_artists"]
        max_seq = values["max_seq"]
        num_chains = values["num_chains"]
        num_samples = values["num_samples"]
        if min(n_obs, n_entities, max_seq, num_chains, num_samples) <= 0 or n_features < 0:
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
            n_artists=n_entities,
            max_seq=max_seq,
            exclude_rw_raw_from_collection=inputs.get("exclude_rw_raw_from_collection", False),
            errors_in_variables=inputs.get("errors_in_variables", False),
            heteroscedastic_entity_obs=inputs.get("heteroscedastic_entity_obs", False),
            entity_group_pooling=inputs.get("entity_group_pooling", False),
            n_groups=n_groups,
        )
        gib = 1024**3
        live_chains = num_chains if chain_method == "vectorized" else 1
        raw_base_bytes = n_params * 4 * 4 * live_chains + n_obs * n_features * 4
        collection_bytes = collected * num_samples * 4
        # The estimator multiplies collection_bytes by a float before dividing;
        # prove that conversion is representable too, not only the later quotient.
        float(collection_bytes)
        raw_base = raw_base_bytes / gib
        unit = collection_bytes * num_chains / gib
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (_is_finite(raw_base) and _is_finite(unit)):
        return None
    return float(raw_base), float(unit)


def _is_finite(value: Any) -> TypeGuard[int | float]:
    """A real, finite number — bools and NaN/inf telemetry are not measurements."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _snapshot_is_valid(snapshot: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return False
    integer_fields = (
        "process_id",
        "thread_id",
        "device_id",
        "process_index",
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_limit",
        "bytes_reserved",
        "num_allocs",
    )
    for name in integer_fields:
        value = snapshot.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    return (
        snapshot["process_id"] > 0
        and snapshot.get("platform") == "gpu"
        and isinstance(snapshot.get("device_kind"), str)
        and bool(snapshot["device_kind"])
        and isinstance(snapshot.get("recorded_at"), str)
        and bool(snapshot["recorded_at"])
    )


def _provenance_quarantine_reason(record: dict[str, Any], actual: float) -> str | None:
    provenance = record.get("peak_provenance")
    if provenance is None:
        return "missing_provenance"
    if not isinstance(provenance, dict):
        return "malformed_provenance"
    if (
        provenance.get("version") != _PEAK_PROVENANCE_VERSION
        or provenance.get("source") != "jax_allocator"
        or provenance.get("scope") != "process"
    ):
        return "unsupported_provenance"
    if provenance.get("trusted_for_calibration") is not True:
        attribution = provenance.get("attribution")
        if isinstance(attribution, str) and attribution in _UNTRUSTED_ATTRIBUTIONS:
            return attribution
        return "untrusted_provenance"
    if provenance.get("attribution") != _TRUSTED_ATTRIBUTION:
        return "invalid_attribution"

    before = provenance.get("before")
    after = provenance.get("after")
    if not _snapshot_is_valid(before):
        return "malformed_provenance"
    if not _snapshot_is_valid(after):
        return "malformed_provenance"
    identity_fields = ("process_id", "device_id", "process_index", "platform", "device_kind")
    if any(before[name] != after[name] for name in identity_fields):
        return "allocator_identity_changed"
    if after["peak_bytes_in_use"] <= before["peak_bytes_in_use"]:
        return "stale_process_peak"
    measured_gb = after["peak_bytes_in_use"] / (1024**3)
    if abs(measured_gb - actual) > 0.001:
        return "peak_provenance_mismatch"
    return None


def _record_label(record: dict[str, Any], index: int) -> str:
    for name in ("record_id", "timestamp"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return value[:_MAX_LABEL_CHARS]
    return f"record[{index}]"


def _partition_points(
    records: list[dict[str, Any]],
) -> tuple[list[_CalibrationPoint], CalibrationDiagnostics]:
    """Trusted fit-owned points, and why the rest were left out.

    Provenance is the only admission test: a peak this fit provably owns is a
    real measurement of this machine, however large. Dropping the large ones
    would leave an envelope verified against a sample censored of exactly the
    workloads that need the most memory.
    """
    points: list[_CalibrationPoint] = []
    quarantine_reasons: Counter[str] = Counter()
    quarantined_records: list[tuple[str, str]] = []
    invalid_points = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_points += 1
            continue
        actual = record.get("actual_peak_gb")
        inputs = record.get("estimate_inputs", {})
        terms = _linear_terms(inputs) if isinstance(inputs, dict) else None
        if not _is_finite(actual) or float(actual) <= 0.0 or terms is None:
            invalid_points += 1
            continue
        actual_value = float(actual)
        provenance_reason = _provenance_quarantine_reason(record, actual_value)
        if provenance_reason is not None:
            quarantine_reasons[provenance_reason] += 1
            quarantined_records.append((_record_label(record, index), provenance_reason))
            continue

        base, unit = terms
        points.append(_CalibrationPoint(base, unit, actual_value))

    reasons = tuple(sorted(quarantine_reasons.items()))
    return points, CalibrationDiagnostics(
        total_records=len(records),
        trusted_points=len(points),
        quarantined_points=sum(quarantine_reasons.values()),
        invalid_points=invalid_points,
        quarantine_reasons=reasons,
        quarantined_records=tuple(quarantined_records),
    )


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


def _refit_with_diagnostics(
    records: list[dict[str, Any]],
    min_points: int,
    jit_buffer_percent: float,
) -> tuple[PerMachineCalibration | None, CalibrationDiagnostics]:
    import numpy as np

    if not _is_finite(jit_buffer_percent) or jit_buffer_percent < 0.0:
        return None, CalibrationDiagnostics(len(records), 0, 0, len(records), (), ())

    points, diagnostics = _partition_points(records)
    if diagnostics.quarantined_points:
        logger.warning(
            "quarantined %d GPU calibration record(s): %s",
            diagnostics.quarantined_points,
            ", ".join(f"{reason}={count}" for reason, count in diagnostics.quarantine_reasons),
        )
    if len(points) < min_points:
        return None, diagnostics

    base_arr = np.asarray([point.base for point in points], dtype=float)
    unit_arr = np.asarray([point.unit for point in points], dtype=float)
    actual_arr = np.asarray([point.actual for point in points], dtype=float)
    if not float(unit_arr.std()) > 0.0:
        return None, diagnostics

    y = actual_arr / (1.0 + jit_buffer_percent) - base_arr
    try:
        slope, intercept = np.polyfit(unit_arr, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None, diagnostics
    if not (_is_finite(float(slope)) and _is_finite(float(intercept))):
        return None, diagnostics
    factor = max(float(slope), 0.1)
    fixed = max(float(intercept), 0.0)

    scale = _envelope_scale(base_arr, unit_arr, actual_arr, factor, fixed, jit_buffer_percent)
    if scale is None:
        return None, diagnostics
    if scale > 2.0:
        logger.warning("local GPU calibration envelope inflation: %.3fx", scale)
    else:
        logger.info("local GPU calibration envelope inflation: %.3fx", scale)
    factor *= scale
    fixed *= scale
    if not (_is_finite(factor) and _is_finite(fixed)):
        return None, diagnostics

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
        return None, diagnostics

    final_diagnostics = CalibrationDiagnostics(
        total_records=diagnostics.total_records,
        trusted_points=diagnostics.trusted_points,
        quarantined_points=diagnostics.quarantined_points,
        invalid_points=diagnostics.invalid_points,
        quarantine_reasons=diagnostics.quarantine_reasons,
        quarantined_records=diagnostics.quarantined_records,
        envelope_inflation=scale,
    )
    return PerMachineCalibration(
        collection_overhead_factor=factor,
        fixed_overhead_gb=fixed,
        n_points=len(points),
        min_ratio=min_ratio,
        envelope_inflation=scale,
        quarantined_points=diagnostics.quarantined_points,
        invalid_points=diagnostics.invalid_points,
        quarantine_reasons=diagnostics.quarantine_reasons,
        quarantined_records=diagnostics.quarantined_records,
    ), final_diagnostics


def refit_constants(
    records: list[dict[str, Any]],
    min_points: int = 5,
    jit_buffer_percent: float = 0.10,
) -> PerMachineCalibration | None:
    """Refit constants only from attributable, bounded-influence measurements."""
    calibration, _ = _refit_with_diagnostics(records, min_points, jit_buffer_percent)
    return calibration


def _diagnostic_summary(diagnostics: CalibrationDiagnostics) -> str:
    details = [
        f"{diagnostics.trusted_points} trusted",
        f"{diagnostics.quarantined_points} quarantined",
        f"{diagnostics.invalid_points} invalid",
    ]
    if diagnostics.quarantine_reasons:
        details.append(
            "reasons "
            + ", ".join(f"{reason}={count}" for reason, count in diagnostics.quarantine_reasons)
        )
    if diagnostics.quarantined_records:
        shown = ", ".join(
            f"{label}:{reason}" for label, reason in diagnostics.quarantined_records[:3]
        )
        remaining = len(diagnostics.quarantined_records) - 3
        details.append(f"excluded {shown}" + (f", +{remaining} more" if remaining > 0 else ""))
    return ", ".join(details)


def resolve_calibration(
    path: Path | None = None,
    min_points: int = 5,
) -> tuple[float, float, str]:
    """(factor, fixed, source) — per-machine when history suffices, else shipped."""
    records = load_records(path)
    calibration, diagnostics = _refit_with_diagnostics(records, min_points, 0.10)
    summary = _diagnostic_summary(diagnostics)
    if calibration is None:
        return (
            COLLECTION_OVERHEAD_FACTOR,
            FIXED_OVERHEAD_GB,
            f"shipped constants (local calibration rejected; {summary})",
        )
    return (
        calibration.collection_overhead_factor,
        calibration.fixed_overhead_gb,
        f"per-machine calibration ({calibration.n_points} local fits, {summary}, "
        f"envelope inflation {calibration.envelope_inflation:.3f}x, "
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
    "CalibrationDiagnostics",
    "PerMachineCalibration",
    "append_record",
    "estimate_with_calibration",
    "load_records",
    "refit_constants",
    "resolve_calibration",
]
