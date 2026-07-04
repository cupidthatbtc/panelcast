"""GPU memory estimation for MCMC runs.

Provides conservative memory estimates for NumPyro/JAX MCMC inference.
The formula intentionally overestimates (within ~2x of measured peaks) to
avoid OOM surprises during long-running sampling jobs.

The memory model was recalibrated against a ladder of measured peaks on the
real dataset (scripts/experiment_preflight_validation.py, RTX 5090 Laptop,
2026-07-04; results in .audit/preflight_validation/):

- Warmup iterations are NOT stored: rungs with warmup 50 vs 250 at equal
  sample counts measured byte-identical peaks.
- Collected draws accumulate across sequential chains and are duplicated once
  more by the end-of-run concatenation. FACTOR is set so the estimated
  per-draw slope stays at or above the measured high-end slope, so
  extrapolation to production (4x1000) and publication (4x5000) scale stays
  over — never under — measured.
- The fixed JIT/CUDA-context floor measured ~0.06 GiB (cal_10, cal_50 and
  2x50 all clustered there); FIXED_OVERHEAD_GB models it, plus a 10%
  proportional buffer.

Estimate/measured across the eight ladder rungs: 1.1x - 1.9x, never under;
the production-scale rungs (2x250, 2x500) land 1.1x - 1.3x. The superseded
constants (FIXED 0.25 / FACTOR 3.0, from a 2026-06-10 ladder) over-projected
small runs ~5x and the 4x5000 publication run at 96.9 GB against a
measurement-consistent ~9 GiB.
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-draw collection factor. Set so the estimated per-draw slope stays >= the
# measured high-end slope (2x250->2x500), keeping 4x1000/4x5000 extrapolation
# over measured (per-chain buffers live to end-of-run, plus one concat copy).
COLLECTION_OVERHEAD_FACTOR = 3.4

# Fixed JIT / CUDA-context floor (~0.06 GiB measured: cal_10, cal_50, 2x50).
FIXED_OVERHEAD_GB = 0.06


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
