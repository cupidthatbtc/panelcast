"""Shared GPU peak-provenance fixtures.

Built from the producer's own dataclass so the store tests cannot drift from
the shape ``measure.attribute_fit_peak`` actually writes.
"""

from __future__ import annotations

from typing import Any

from panelcast.gpu_memory.measure import JaxAllocatorSnapshot

TRUSTED_ATTRIBUTION = "fit_interval_new_process_peak"


def allocator_snapshot(
    peak: int,
    *,
    device_id: int = 0,
    bytes_limit: int | None = None,
) -> JaxAllocatorSnapshot:
    """One fit-boundary snapshot whose process peak is ``peak`` bytes."""
    return JaxAllocatorSnapshot(
        recorded_at="2026-07-30T00:00:00+00:00",
        process_id=1234,
        thread_id=5678,
        device_id=device_id,
        process_index=0,
        platform="gpu",
        device_kind="test GPU",
        bytes_in_use=peak,
        peak_bytes_in_use=peak,
        bytes_limit=max(peak * 2, 1) if bytes_limit is None else bytes_limit,
        bytes_reserved=peak,
        num_allocs=1,
    )


def peak_provenance(
    actual_gb: float,
    *,
    attribution: str = TRUSTED_ATTRIBUTION,
) -> dict[str, Any]:
    """Store-shaped provenance for a fit that measured ``actual_gb``.

    Only the trusted attribution gets a lower ``before`` peak; every other
    attribution describes a fit that did not raise the process peak.
    """
    peak = round(actual_gb * 1024**3)
    limit = max(peak * 2, 1)
    trusted = attribution == TRUSTED_ATTRIBUTION
    before = max(peak - 1, 0) if trusted else peak
    return {
        "version": 1,
        "source": "jax_allocator",
        "scope": "process",
        "trusted_for_calibration": trusted,
        "attribution": attribution,
        "before": allocator_snapshot(before, bytes_limit=limit).to_dict(),
        "after": allocator_snapshot(peak, bytes_limit=limit).to_dict(),
    }
