"""GPU-memory admission control for concurrent select arms (#167).

The estimator (`estimate_with_calibration`) prices one fit; this module rations
those prices against measured free VRAM so several arm subprocesses can share
one device without over-committing it. Reservations are held from admission to
release, so a child that has not yet materialized its JAX pool still counts;
as pools materialize, measured free shrinks while the reservation persists —
later admissions get strictly MORE conservative, never less (the estimator's
never-under contract, extended to concurrency).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

_GIB = 1024**3


def nvml_free_bytes(device_index: int = 0) -> int | None:
    """Measured free VRAM in bytes, or None when NVML/GPU is unavailable."""
    from panelcast.gpu_memory.query import GpuMemoryError, query_gpu_memory

    try:
        return int(query_gpu_memory(device_index).free_bytes)
    except GpuMemoryError:
        return None


class GpuAdmission:
    """Reserve estimated fit footprints against measured free VRAM.

    ``try_admit`` grants when outstanding reservations plus the candidate fit
    stay under measured-free × headroom; ``admit`` spin-waits until granted.
    Without a measurable GPU (no NVML — CPU-fit domains), admission degrades
    to one-at-a-time: the estimator has nothing trustworthy to ration.
    """

    def __init__(
        self,
        headroom: float = 0.8,
        free_bytes_fn: Callable[[], int | None] | None = None,
    ):
        if not 0.0 < headroom <= 1.0:
            raise ValueError(f"headroom must be in (0, 1], got {headroom}")
        self._headroom = headroom
        self._free_bytes_fn = free_bytes_fn or nvml_free_bytes
        self._lock = threading.Lock()
        self._reserved_gb = 0.0

    @property
    def reserved_gb(self) -> float:
        with self._lock:
            return self._reserved_gb

    def try_admit(self, estimate_gb: float) -> bool:
        with self._lock:
            free = self._free_bytes_fn()
            if free is None:
                if self._reserved_gb == 0.0:
                    self._reserved_gb += estimate_gb
                    return True
                return False
            budget_gb = free / _GIB * self._headroom
            # The first reservation always admits: an arm priced above the whole
            # headroom budget must still run (alone), never spin admit() forever.
            if self._reserved_gb == 0.0 or self._reserved_gb + estimate_gb <= budget_gb:
                self._reserved_gb += estimate_gb
                return True
            return False

    def admit(self, estimate_gb: float, poll_seconds: float = 5.0) -> None:
        """Block until the reservation is granted (a release frees budget)."""
        while not self.try_admit(estimate_gb):
            time.sleep(poll_seconds)

    def release(self, estimate_gb: float) -> None:
        with self._lock:
            self._reserved_gb = max(0.0, self._reserved_gb - estimate_gb)


__all__ = ["GpuAdmission", "nvml_free_bytes"]
