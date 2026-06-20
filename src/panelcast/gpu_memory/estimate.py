"""GPU memory estimation for MCMC runs.

Provides conservative memory estimates for NumPyro/JAX MCMC inference.
The formula intentionally overestimates (within ~2x of measured peaks) to
avoid OOM surprises during long-running sampling jobs.

The memory model was recalibrated against a ladder of measured peaks on the
real dataset (scripts/experiment_preflight_validation.py, RTX 5090,
2026-06-10; results in outputs/experiments/preflight_validation.json):

- Warmup iterations are NOT stored: rungs with warmup 50 vs 250 at equal
  sample counts measured byte-identical peaks. The old formula counted
  warmup draws as stored samples — its largest error source.
- Collected draws accumulate across sequential chains and are duplicated
  once more by the end-of-run concatenation: measured bytes per kept draw
  were 2.0-2.9x the raw site size. COLLECTION_OVERHEAD_FACTOR = 3.0 keeps
  the estimate conservative.
- Fixed JIT/runtime overhead measured ~0.17-0.26 GiB (ladder fit
  intercept); modeled as FIXED_OVERHEAD_GB plus a 10% proportional buffer
  (the old 40% multiplicative buffer stacked error on error).

Validated ratios (estimate / measured) across the eight ladder rungs:
1.1x - 2.0x, never under.
"""

from __future__ import annotations

from dataclasses import dataclass

# Measured bytes-per-collected-draw run 2.0-2.9x the raw site size
# (per-chain collection buffers live until the end of the run, plus one
# concatenated copy across chains). 3.0 keeps the estimate conservative.
COLLECTION_OVERHEAD_FACTOR = 3.0

# Fixed JIT compilation / CUDA workspace overhead (ladder fit intercept
# measured 0.17-0.26 GiB on the real model).
FIXED_OVERHEAD_GB = 0.25


@dataclass(frozen=True)
class MemoryEstimate:
    """Memory estimation breakdown for MCMC run.

    This dataclass holds the components of GPU memory estimation,
    allowing inspection of where memory is expected to be used.

    Attributes:
        base_model_gb: Memory for parameters, gradients, feature matrix and
            fixed JIT/runtime overhead.
        per_chain_gb: Memory per chain for collected post-warmup samples
            (incl. collection overhead).
        jit_buffer_gb: Proportional buffer for allocator slack.
        num_chains: Number of MCMC chains.
    """

    base_model_gb: float
    per_chain_gb: float
    jit_buffer_gb: float
    num_chains: int

    @property
    def total_gb(self) -> float:
        """Total estimated memory in GB.

        Returns:
            Sum of base model, chain memory, and JIT buffer.
        """
        return self.base_model_gb + self.chain_memory_gb + self.jit_buffer_gb

    @property
    def chain_memory_gb(self) -> float:
        """Memory for all chains combined in GB.

        Returns:
            Per-chain memory multiplied by number of chains.
        """
        return self.per_chain_gb * self.num_chains


def estimate_memory_gb(
    n_observations: int,
    n_features: int,
    n_artists: int,
    max_seq: int,
    num_chains: int,
    num_samples: int,
    num_warmup: int,
    jit_buffer_percent: float = 0.10,
    exclude_rw_raw_from_collection: bool = False,
) -> MemoryEstimate:
    """Estimate GPU memory for MCMC run.

    This is a CONSERVATIVE estimate - designed to overestimate (within ~2x)
    rather than underestimate to avoid OOM surprises during long runs.

    The formula accounts for:
    1. Base model memory: parameters, gradients/momentum, feature matrix,
       and fixed JIT/runtime overhead.
    2. Per-chain collected samples: post-warmup draws only (warmup is never
       stored), times the measured collection-overhead factor.
    3. Proportional buffer for allocator slack.

    Args:
        n_observations: Number of observations in dataset.
        n_features: Number of features in model.
        n_artists: Number of unique artists (for hierarchical effects).
        max_seq: Maximum sequence length for time-varying effects.
        num_chains: Number of MCMC chains (sequential chains still
            accumulate their collected draws on device).
        num_samples: Number of samples per chain (post-warmup).
        num_warmup: Warmup iterations per chain. Kept for API compatibility
            and run provenance; warmup draws are NOT stored and do not
            contribute to the estimate (measured: identical peaks at warmup
            50 vs 250).
        jit_buffer_percent: Proportional allocator-slack buffer
            (default 0.10).
        exclude_rw_raw_from_collection: Whether the run excludes the rw_raw
            tensor from in-sampler collection (--exclude-rw-raw-from-
            collection). Removes the dominant n_artists*(max_seq-1) term
            from collected-draw size (~96% cut at production settings).

    Returns:
        MemoryEstimate with breakdown of memory components.

    Example:
        >>> estimate = estimate_memory_gb(
        ...     n_observations=1000,
        ...     n_features=20,
        ...     n_artists=50,
        ...     max_seq=10,
        ...     num_chains=4,
        ...     num_samples=1000,
        ...     num_warmup=1000,
        ... )
        >>> print(f"Estimated: {estimate.total_gb:.2f} GB")
    """
    bytes_per_float = 4  # float32
    gib = 1024**3

    # Model parameters (approximate for hierarchical time-varying model)
    # - init_artist_effect: n_artists
    # - rw_raw: n_artists * (max_seq - 1)
    # - beta: n_features
    # - hyperpriors: ~10 scalars (sigma_obs, sigma_artist, etc.)
    rw_raw_params = n_artists * max(0, max_seq - 1)
    n_params = n_artists + rw_raw_params + n_features + 10

    # Sites actually collected per draw (the in-sampler exclusion removes
    # rw_raw from storage; gradients/momentum still cover all parameters).
    collected_params = n_params - rw_raw_params if exclude_rw_raw_from_collection else n_params

    # Base model memory: parameters + gradients/momentum/state (~3x params)
    # + feature matrix X + fixed JIT/runtime overhead.
    base_bytes = (
        n_params * bytes_per_float
        + n_params * bytes_per_float * 3
        + n_observations * n_features * bytes_per_float
    )
    base_gb = base_bytes / gib + FIXED_OVERHEAD_GB

    # Per-chain collected samples: post-warmup draws only. Sequential chains
    # accumulate on device, and the end-of-run concatenation duplicates the
    # storage once more — covered by COLLECTION_OVERHEAD_FACTOR.
    per_chain_bytes = collected_params * num_samples * bytes_per_float * COLLECTION_OVERHEAD_FACTOR
    per_chain_gb = per_chain_bytes / gib

    # Proportional allocator-slack buffer on the subtotal.
    subtotal = base_gb + per_chain_gb * num_chains
    jit_buffer_gb = subtotal * jit_buffer_percent

    return MemoryEstimate(
        base_model_gb=base_gb,
        per_chain_gb=per_chain_gb,
        jit_buffer_gb=jit_buffer_gb,
        num_chains=num_chains,
    )
