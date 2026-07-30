"""JAX device memory statistics wrapper.

Provides typed access to JAX's jax.Device.memory_stats() API for accurate
peak GPU memory measurement during MCMC runs.

Example:
    >>> stats = get_jax_memory_stats()
    >>> print(f"Peak memory: {stats.peak_gb:.2f} GB")
    Peak memory: 4.25 GB
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import jax

logger = logging.getLogger(__name__)

_FIT_INTERVAL_LOCK = threading.Lock()
_ACTIVE_FIT_INTERVALS = 0
_FIT_OVERLAP_EPOCH = 0


@dataclass(frozen=True)
class JaxMemoryStats:
    """JAX device memory statistics.

    Wraps jax.Device.memory_stats() return values with type safety
    and convenient property accessors for GB values.

    Attributes:
        bytes_in_use: Current bytes being used on the device.
        peak_bytes_in_use: Maximum bytes used since device initialization.
        bytes_limit: Maximum available bytes on the device.
        bytes_reserved: Pre-allocated bytes from system.
    """

    bytes_in_use: int
    peak_bytes_in_use: int
    bytes_limit: int
    bytes_reserved: int

    @property
    def peak_gb(self) -> float:
        """Peak memory usage in GB (1024^3 bytes)."""
        return self.peak_bytes_in_use / (1024**3)

    @property
    def limit_gb(self) -> float:
        """Memory limit in GB (1024^3 bytes)."""
        return self.bytes_limit / (1024**3)

    @property
    def in_use_gb(self) -> float:
        """Current memory usage in GB (1024^3 bytes)."""
        return self.bytes_in_use / (1024**3)

    @property
    def reserved_gb(self) -> float:
        """Reserved memory in GB (1024^3 bytes)."""
        return self.bytes_reserved / (1024**3)


@dataclass(frozen=True)
class JaxAllocatorSnapshot:
    """Process-local JAX allocator state at one fit boundary."""

    recorded_at: str
    process_id: int
    thread_id: int
    device_id: int
    process_index: int
    platform: str
    device_kind: str
    bytes_in_use: int
    peak_bytes_in_use: int
    bytes_limit: int
    bytes_reserved: int
    num_allocs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitPeakTrackingToken:
    before: JaxAllocatorSnapshot | None
    overlap_epoch: int
    exclusive_at_start: bool


def capture_jax_allocator_snapshot(device_index: int = 0) -> JaxAllocatorSnapshot | None:
    """Capture provenance for the process-local GPU allocator, if available."""
    try:
        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if device_index < 0 or device_index >= len(devices):
            return None
        device = devices[device_index]
        stats = device.memory_stats()
        if not stats or "peak_bytes_in_use" not in stats:
            return None

        def _stat(name: str) -> int:
            value = stats.get(name, 0)
            if isinstance(value, bool):
                raise ValueError(name)
            parsed = int(value)
            if parsed < 0:
                raise ValueError(name)
            return parsed

        return JaxAllocatorSnapshot(
            recorded_at=datetime.now(UTC).isoformat(),
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
            device_id=int(getattr(device, "id", device_index)),
            process_index=int(getattr(device, "process_index", 0)),
            platform=str(device.platform),
            device_kind=str(getattr(device, "device_kind", "unknown")),
            bytes_in_use=_stat("bytes_in_use"),
            peak_bytes_in_use=_stat("peak_bytes_in_use"),
            bytes_limit=_stat("bytes_limit"),
            bytes_reserved=_stat("bytes_reserved"),
            num_allocs=_stat("num_allocs"),
        )
    except Exception as exc:  # pragma: no cover - backend-specific telemetry failures
        logger.debug("JAX allocator snapshot unavailable: %s", type(exc).__name__, exc_info=True)
        return None


def attribute_fit_peak(
    before: JaxAllocatorSnapshot | None,
    after: JaxAllocatorSnapshot | None,
    *,
    interval_exclusive: bool = True,
) -> tuple[int | None, dict[str, Any] | None]:
    """Return the observed peak and whether this fit established that process peak."""
    if after is None:
        return None, None

    attribution = "fit_interval_new_process_peak"
    trusted = True
    if not interval_exclusive:
        attribution = "overlapping_fit_interval"
        trusted = False
    elif before is None:
        attribution = "missing_start_snapshot"
        trusted = False
    elif (
        before.process_id != after.process_id
        or before.device_id != after.device_id
        or before.process_index != after.process_index
        or before.platform != after.platform
        or before.device_kind != after.device_kind
    ):
        attribution = "allocator_identity_changed"
        trusted = False
    elif after.peak_bytes_in_use <= before.peak_bytes_in_use:
        attribution = "stale_process_peak"
        trusted = False

    return after.peak_bytes_in_use, {
        "version": 1,
        "source": "jax_allocator",
        "scope": "process",
        "trusted_for_calibration": trusted,
        "attribution": attribution,
        "before": before.to_dict() if before is not None else None,
        "after": after.to_dict(),
    }


def begin_fit_peak_tracking() -> FitPeakTrackingToken:
    global _ACTIVE_FIT_INTERVALS, _FIT_OVERLAP_EPOCH

    with _FIT_INTERVAL_LOCK:
        exclusive = _ACTIVE_FIT_INTERVALS == 0
        if not exclusive:
            _FIT_OVERLAP_EPOCH += 1
        _ACTIVE_FIT_INTERVALS += 1
        overlap_epoch = _FIT_OVERLAP_EPOCH
    return FitPeakTrackingToken(
        before=capture_jax_allocator_snapshot(),
        overlap_epoch=overlap_epoch,
        exclusive_at_start=exclusive,
    )


def end_fit_peak_tracking(
    token: FitPeakTrackingToken,
) -> tuple[int | None, dict[str, Any] | None]:
    global _ACTIVE_FIT_INTERVALS

    after = capture_jax_allocator_snapshot()
    with _FIT_INTERVAL_LOCK:
        interval_exclusive = token.exclusive_at_start and token.overlap_epoch == _FIT_OVERLAP_EPOCH
        _ACTIVE_FIT_INTERVALS = max(0, _ACTIVE_FIT_INTERVALS - 1)
    return attribute_fit_peak(
        token.before,
        after,
        interval_exclusive=interval_exclusive,
    )


def get_jax_memory_stats(device_index: int = 0) -> JaxMemoryStats:
    """Get JAX memory statistics for specified GPU device.

    Queries jax.Device.memory_stats() for the specified GPU and returns
    a typed JaxMemoryStats dataclass with peak memory usage.

    Args:
        device_index: GPU device index (default 0 for first GPU).

    Returns:
        JaxMemoryStats with current and peak memory usage.

    Raises:
        RuntimeError: If no GPU devices available or device_index out of range.

    Example:
        >>> stats = get_jax_memory_stats()
        >>> print(f"Peak: {stats.peak_gb:.2f} GB of {stats.limit_gb:.2f} GB")
        Peak: 4.25 GB of 24.00 GB
    """
    if device_index < 0:
        raise RuntimeError("GPU index must be non-negative")

    try:
        devices = jax.devices("gpu")
    except RuntimeError as e:
        raise RuntimeError(f"No GPU devices available for JAX: {e}") from e

    if not devices:
        raise RuntimeError("No GPU devices available for JAX")

    if device_index >= len(devices):
        raise RuntimeError(
            f"GPU index {device_index} out of range. Available GPUs: 0-{len(devices) - 1}"
        )

    device = devices[device_index]
    stats = device.memory_stats()

    # Handle case where memory_stats() returns None (e.g., on some platforms)
    if stats is None:
        raise RuntimeError(
            f"Device {device_index} does not support memory_stats(). "
            "This may occur on non-CUDA backends."
        )

    # Check for missing expected keys and log debug info
    # peak_bytes_in_use is critical for preflight decisions - must be present
    if "peak_bytes_in_use" not in stats:
        raise KeyError(
            f"Critical key 'peak_bytes_in_use' missing from JAX memory stats. "
            f"Available keys: {sorted(stats.keys())}"
        )

    optional_keys = {"bytes_in_use", "bytes_limit", "bytes_reserved"}
    missing_optional = optional_keys - set(stats.keys())
    if missing_optional:
        logger.debug(
            "Missing optional JAX memory stat keys: %s",
            sorted(missing_optional),
        )

    return JaxMemoryStats(
        bytes_in_use=stats.get("bytes_in_use", 0),
        peak_bytes_in_use=stats["peak_bytes_in_use"],
        bytes_limit=stats.get("bytes_limit", 0),
        bytes_reserved=stats.get("bytes_reserved", 0),
    )
