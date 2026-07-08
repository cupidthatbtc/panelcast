"""Preflight check logic for GPU memory verification.

Connects memory estimation to GPU queries to produce actionable
pass/fail/warning results before committing to long-running MCMC jobs.
"""

from __future__ import annotations

from panelcast.gpu_memory import (
    MemoryEstimate,
    estimate_memory_gb,
    query_gpu_memory,
)
from panelcast.pipelines.errors import GpuMemoryError
from panelcast.preflight import PreflightResult, PreflightStatus


def run_preflight_check(
    n_observations: int,
    n_features: int,
    n_artists: int,
    max_seq: int,
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    headroom_target: float = 0.20,
    jit_buffer_percent: float = 0.10,
    exclude_rw_raw_from_collection: bool = False,
    errors_in_variables: bool = False,
    heteroscedastic_entity_obs: bool = False,
    entity_group_pooling: bool = False,
    n_groups: int = 0,
) -> PreflightResult:
    """Run preflight memory check.

    Estimates GPU memory requirements and compares against available VRAM
    to determine if the MCMC run is likely to succeed.

    Args:
        n_observations: Number of observations in dataset.
        n_features: Number of features in model.
        n_artists: Number of unique artists (for hierarchical effects).
        max_seq: Maximum sequence length for time-varying effects.
        num_chains: Number of parallel MCMC chains.
        num_samples: Number of samples per chain (post-warmup).
        num_warmup: Number of warmup samples per chain.
        headroom_target: Target headroom as fraction (default 0.20 = 20%).
            PASS requires at least this much headroom.
        jit_buffer_percent: Allocator-slack buffer as fraction (default 0.10).
        exclude_rw_raw_from_collection: Whether the run excludes rw_raw from
            in-sampler collection (changes the dominant memory term).
        errors_in_variables: EIV gate (adds an n_obs-sized latent term).
        heteroscedastic_entity_obs: Entity-overdispersion gate (adds an
            n_artists-sized latent term).
        entity_group_pooling: Group-pooling gate (adds an n_groups-sized
            latent term).
        n_groups: Group count for the entity_group_pooling term.

    Returns:
        PreflightResult with status, estimate, and suggestions.

    Example:
        >>> result = run_preflight_check(
        ...     n_observations=1000, n_features=20, n_artists=50,
        ...     max_seq=10, num_chains=4, num_samples=1000, num_warmup=1000,
        ... )
        >>> if result.status == PreflightStatus.FAIL:
        ...     raise SystemExit(result.exit_code)
    """
    gate_flags = {
        "exclude_rw_raw_from_collection": exclude_rw_raw_from_collection,
        "errors_in_variables": errors_in_variables,
        "heteroscedastic_entity_obs": heteroscedastic_entity_obs,
        "entity_group_pooling": entity_group_pooling,
        "n_groups": n_groups,
    }

    # Step 1: Get memory estimate
    estimate = estimate_memory_gb(
        n_observations=n_observations,
        n_features=n_features,
        n_artists=n_artists,
        max_seq=max_seq,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        jit_buffer_percent=jit_buffer_percent,
        **gate_flags,
    )

    # Step 2: Try to query GPU memory
    try:
        gpu_info = query_gpu_memory()
    except GpuMemoryError as e:
        # Cannot query GPU - return CANNOT_CHECK status
        return PreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            estimate=estimate,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            message=f"Cannot query GPU memory: {e}",
            suggestions=("Consider running with --device cpu",),
        )

    # Step 3: Calculate headroom and determine status
    available_gb = gpu_info.free_gb
    total_gpu_gb = gpu_info.total_gb

    if available_gb > 0:
        headroom_percent = (available_gb - estimate.total_gb) / available_gb * 100
    else:
        headroom_percent = -100.0  # No available memory

    # Step 4: Determine status based on thresholds
    if estimate.total_gb > available_gb:
        status = PreflightStatus.FAIL
    elif estimate.total_gb <= available_gb * (1 - headroom_target):
        # Sufficient headroom (>= 20% by default)
        status = PreflightStatus.PASS
    else:
        # Fits but low headroom
        status = PreflightStatus.WARNING

    # Step 5: Generate suggestions for FAIL or WARNING
    suggestions = _generate_suggestions(
        status=status,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        n_observations=n_observations,
        n_features=n_features,
        n_artists=n_artists,
        max_seq=max_seq,
        jit_buffer_percent=jit_buffer_percent,
        **gate_flags,
    )

    # Step 6: Generate message
    message = _generate_message(
        status=status,
        estimate=estimate,
        available_gb=available_gb,
        headroom_percent=headroom_percent,
    )

    return PreflightResult(
        status=status,
        estimate=estimate,
        available_gb=available_gb,
        total_gpu_gb=total_gpu_gb,
        headroom_percent=headroom_percent,
        message=message,
        suggestions=suggestions,
        device_name=gpu_info.device_name,
    )


def _generate_suggestions(
    status: PreflightStatus,
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    n_observations: int,
    n_features: int,
    n_artists: int,
    max_seq: int,
    jit_buffer_percent: float,
    exclude_rw_raw_from_collection: bool = False,
    errors_in_variables: bool = False,
    heteroscedastic_entity_obs: bool = False,
    entity_group_pooling: bool = False,
    n_groups: int = 0,
) -> tuple[str, ...]:
    """Generate configuration adjustment suggestions for FAIL/WARNING."""
    if status == PreflightStatus.PASS:
        return ()

    suggestions: list[str] = []

    # Suggest reducing chains first (most effective). The halved-chains
    # estimate must use the run's own gate flags — without the exclusion flag
    # it prices a structurally different (~25x bigger) model.
    if num_chains > 1:
        reduced_chains = max(1, num_chains // 2)
        new_estimate = estimate_memory_gb(
            n_observations=n_observations,
            n_features=n_features,
            n_artists=n_artists,
            max_seq=max_seq,
            num_chains=reduced_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
            jit_buffer_percent=jit_buffer_percent,
            exclude_rw_raw_from_collection=exclude_rw_raw_from_collection,
            errors_in_variables=errors_in_variables,
            heteroscedastic_entity_obs=heteroscedastic_entity_obs,
            entity_group_pooling=entity_group_pooling,
            n_groups=n_groups,
        )
        suggestions.append(
            f"Try --num-chains {reduced_chains} (estimated: {new_estimate.total_gb:.1f} GB)"
        )

    # Suggest reducing samples
    if num_samples > 500:
        suggestions.append(f"Try --num-samples {num_samples // 2}")

    # Suggest the in-sampler exclusion (removes the dominant memory term;
    # warmup reduction is no longer suggested — warmup draws are not stored)
    if not exclude_rw_raw_from_collection:
        suggestions.append(
            "Try --exclude-rw-raw-from-collection (removes the dominant memory term)"
        )

    # Always suggest full preflight on fail
    if status == PreflightStatus.FAIL:
        suggestions.append("Run --preflight-full for accurate measurement")

    return tuple(suggestions)


def _generate_message(
    status: PreflightStatus,
    estimate: MemoryEstimate,
    available_gb: float,
    headroom_percent: float,
) -> str:
    """Generate human-readable status message."""
    match status:
        case PreflightStatus.PASS:
            return (
                f"Memory check passed: {estimate.total_gb:.1f} GB estimated, "
                f"{available_gb:.1f} GB available ({headroom_percent:.0f}% headroom)"
            )
        case PreflightStatus.WARNING:
            return (
                f"Memory check warning: {estimate.total_gb:.1f} GB estimated, "
                f"{available_gb:.1f} GB available (low headroom: {headroom_percent:.0f}%)"
            )
        case PreflightStatus.FAIL:
            return (
                f"Memory check failed: {estimate.total_gb:.1f} GB estimated "
                f"exceeds {available_gb:.1f} GB available"
            )
        case PreflightStatus.CANNOT_CHECK:
            # This case is handled separately in run_preflight_check
            return "Cannot check GPU memory"
        case _:
            return f"Unknown preflight status: {status}"
