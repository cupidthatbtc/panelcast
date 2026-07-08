"""MCMC fitting orchestration with GPU acceleration.

This module provides the infrastructure for fitting NumPyro models using MCMC
with GPU acceleration via JAX. Key features:
- NUTS kernel with configurable chain_method (sequential default for stability)
- Automatic GPU detection and logging
- Divergence tracking (logged but not failing - Phase 7 handles thresholds)
- ArviZ InferenceData conversion with observed/constant data groups
"""

import gc
import hashlib
import json
import logging
import pickle
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import arviz as az
import jax
import numpy as np
import xarray as xr
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_feasible, init_to_median, init_to_uniform

__all__ = [
    "MCMCConfig",
    "FitResult",
    "fit_model",
    "get_gpu_info",
    "resolve_progress_bar",
]

logger = logging.getLogger(__name__)


def resolve_progress_bar(progress_bar: bool | None) -> bool:
    """Resolve a tri-state progress-bar setting to a concrete bool.

    None means auto: show the bar only when stderr is a TTY — tqdm's
    carriage returns make piped/redirected logs unreadable.
    """
    if progress_bar is None:
        return sys.stderr.isatty()
    return bool(progress_bar)


@dataclass(frozen=True)
class MCMCConfig:
    """MCMC configuration for reproducibility.

    All parameters are frozen to ensure immutability during model fitting.

    Attributes:
        num_warmup: Number of warmup (burn-in) iterations per chain.
            Default 1000 is standard for publication.
        num_samples: Number of post-warmup samples per chain.
            Default 1000 provides 4000 total samples with 4 chains.
        num_chains: Number of parallel chains.
            Default 4 is standard for Rhat convergence assessment.
        chain_method: How to parallelize chains.
            "sequential" runs chains one at a time (default, most stable).
            "vectorized" runs chains on single GPU (faster but uses more memory).
            "parallel" uses pmap across multiple devices.
        seed: Random seed for reproducibility.
            Default 0 for consistent results.
        max_tree_depth: Maximum tree depth for NUTS.
            Default 10 (numpyro default).
        target_accept_prob: Target acceptance probability for adaptation.
            Default 0.90 improves adaptation for challenging posteriors.
        init_strategy: NUTS initialization strategy: "uniform", "median", or
            "feasible". Default "uniform" is NumPyro's default and keeps every
            published trajectory byte-identical; changing it re-baselines any
            init-sensitive fixture, so flip the default only in a versioned
            re-baseline.
        checkpoint_every_draws: Sample in blocks of this many post-warmup draws,
            persisting sampler state and collected draws after each block so an
            interrupted fit resumes from the last block instead of restarting
            (requires ``checkpoint_dir`` on fit_model). None (default) keeps the
            single-shot path byte-identical. Continuing through
            ``post_warmup_state`` is the same Markov chain — parity with the
            single-shot draws is test-guarded.
    """

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    chain_method: str = "sequential"
    seed: int = 0
    max_tree_depth: int = 10
    target_accept_prob: float = 0.90
    init_strategy: str = "uniform"
    checkpoint_every_draws: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FitResult:
    """Result container for MCMC fitting.

    This is not frozen because it contains large mutable objects (MCMC, InferenceData).

    Attributes:
        mcmc: NumPyro MCMC object with samples and extra fields. Under
            checkpointed sampling it is the FINAL block's runner (its
            get_samples covers only that block; use idata for the full
            posterior), and None when a resume found every block already done.
        idata: ArviZ InferenceData with posterior, observed_data, constant_data groups.
        divergences: Total count of divergent transitions across all chains.
        runtime_seconds: Total wall-clock time for fitting.
        gpu_info: String describing GPU used (or "CPU only").
        peak_gpu_memory_bytes: Process-wide peak GPU allocation observed right
            after sampling (None on CPU or when the backend exposes no stats).
            Free telemetry used to validate preflight memory projections.
        tree_depth_saturation: Fraction of post-warmup transitions whose tree
            completed max_tree_depth doublings (an efficiency signal, not a
            correctness one).
        resumed_from_checkpoint: True when part of this fit's draws were
            restored from a checkpoint — its wall clock covers only the blocks
            this process ran, so it must not become a runtime-calibration
            datapoint.
    """

    mcmc: MCMC | None
    idata: az.InferenceData
    divergences: int
    runtime_seconds: float
    gpu_info: str
    peak_gpu_memory_bytes: int | None = None
    tree_depth_saturation: float | None = None
    resumed_from_checkpoint: bool = False


def measure_peak_gpu_bytes() -> int | None:
    """Peak GPU bytes allocated by this process, if a GPU backend is active.

    Reads the same counter the preflight mini-runs use
    (``device.memory_stats()["peak_bytes_in_use"]``), so projections and
    real-run telemetry are directly comparable. The counter is cumulative for
    the process; with several fits per process it reports the largest so far.
    """
    try:
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpu_devices:
            return None
        stats = gpu_devices[0].memory_stats()
        if not stats or "peak_bytes_in_use" not in stats:
            return None
        return int(stats["peak_bytes_in_use"])
    except Exception as e:  # pragma: no cover - defensive against backend quirks
        logger.debug("peak GPU bytes unavailable: %s", type(e).__name__, exc_info=True)
        return None


def get_gpu_info() -> str:
    """Get GPU device info for reproducibility logging.

    Attempts to get detailed GPU information via nvidia-smi.
    Falls back to JAX device info if nvidia-smi is unavailable.

    Returns:
        String describing GPU (e.g., "NVIDIA GeForce RTX 5090, 32GB") or "CPU only".

    Example:
        >>> info = get_gpu_info()
        >>> print(info)  # "NVIDIA GeForce RTX 5090, 32768 MiB" or "CPU only"
    """
    # Check JAX devices first
    devices = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devices)

    if not has_gpu:
        return "CPU only"

    # Try nvidia-smi for detailed info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Format: "NVIDIA GeForce RTX 5090, 32768"
            gpu_infos = []
            for line in lines:
                parts = line.split(", ")
                if len(parts) == 2:
                    name, memory_mb = parts
                    gpu_infos.append(f"{name.strip()}, {memory_mb.strip()} MiB")
                else:
                    gpu_infos.append(line.strip())
            return "; ".join(gpu_infos)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fall back to JAX device kind
    gpu_devices = [d.device_kind for d in devices if d.platform == "gpu"]
    return ", ".join(gpu_devices) if gpu_devices else "GPU (unknown type)"


def _resolve_init_strategy(name: str) -> Callable:
    strategies = {
        "uniform": init_to_uniform,
        "median": init_to_median,
        "feasible": init_to_feasible,
    }
    if name not in strategies:
        raise ValueError(f"Invalid init_strategy: '{name}'. Must be one of {sorted(strategies)}.")
    return strategies[name]


def _log_tree_depth_saturation(extra_fields: dict, max_tree_depth: int) -> float | None:
    """Fraction of transitions that completed max_tree_depth doublings.

    A full tree at depth d takes 2^d - 1 leapfrog steps; counting num_steps >=
    that undercounts trees that turned inside the final doubling, so this is a
    conservative saturation measure.
    """
    num_steps = extra_fields.get("num_steps")
    if num_steps is None:
        return None
    saturation = float(np.mean(np.asarray(num_steps) >= 2**max_tree_depth - 1))
    logger.info(
        f"Tree-depth saturation: {saturation:.1%} of transitions "
        f"hit max_tree_depth={max_tree_depth}"
    )
    if saturation > 0.05:
        logger.warning(
            f"{saturation:.1%} of transitions saturate "
            f"max_tree_depth={max_tree_depth}; consider raising it."
        )
    return saturation


def _block_sizes(num_samples: int, block: int) -> list[int]:
    full, rem = divmod(num_samples, block)
    return [block] * full + ([rem] if rem else [])


def _checkpoint_identity(config: MCMCConfig, run_args: dict) -> dict:
    """What must match for a checkpoint to belong to this fit."""
    import numpyro

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(run_args["y"])).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(run_args["X"])).tobytes())
    return {
        "config": config.to_dict(),
        "data_hash": digest.hexdigest()[:16],
        "numpyro_version": numpyro.__version__,
        "jax_version": jax.__version__,
    }


def _load_blocks(checkpoint_dir: Path, n_blocks: int) -> tuple[dict, dict]:
    """(grouped samples, chain-major-flat extra fields) concatenated across blocks."""
    samples: dict[str, list] = {}
    extra: dict[str, list] = {}
    for i in range(n_blocks):
        with np.load(checkpoint_dir / f"block_{i:04d}.npz") as payload:
            for key in payload.files:
                kind, name = key.split(".", 1)
                (samples if kind == "s" else extra).setdefault(name, []).append(payload[key])
    samples_grouped = {k: np.concatenate(v, axis=1) for k, v in samples.items()}
    extra_flat = {}
    for k, v in extra.items():
        grouped = np.concatenate(v, axis=1)
        extra_flat[k] = grouped.reshape(-1, *grouped.shape[2:])
    return samples_grouped, extra_flat


def _run_blocked(
    kernel: NUTS,
    config: MCMCConfig,
    rng_key,
    run_args: dict,
    extra_fields: tuple[str, ...],
    progress_bar: bool,
    checkpoint_dir: Path,
) -> tuple[MCMC | None, dict, dict, bool]:
    """Sample in blocks via ``post_warmup_state``, checkpointing after each.

    Continuing through post_warmup_state is the same Markov chain, so blocked
    draws equal the single-shot draws for the same seed (parity-tested). A
    matching cursor resumes at the first missing block; any config/data/version
    mismatch refuses loudly rather than silently mixing two fits.
    Returns (last block's MCMC or None, grouped samples, flat extras, resumed).
    """
    assert config.checkpoint_every_draws is not None
    sizes = _block_sizes(config.num_samples, config.checkpoint_every_draws)
    cursor_path = checkpoint_dir / "cursor.json"
    state_path = checkpoint_dir / "state.pkl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity = _checkpoint_identity(config, run_args)

    start_block = 0
    state = None
    if cursor_path.exists():
        cursor = json.loads(cursor_path.read_text(encoding="utf-8"))
        if cursor.get("identity") != identity:
            raise ValueError(
                f"checkpoint at {checkpoint_dir} belongs to a different fit "
                "(config, data, or numpyro/jax version changed); delete it to start over"
            )
        start_block = int(cursor.get("blocks_done", 0))
        if start_block > 0:
            with state_path.open("rb") as fh:
                state = pickle.load(fh)
            logger.info("Resuming from checkpoint: %d/%d blocks done", start_block, len(sizes))

    mcmc: MCMC | None = None
    for i in range(start_block, len(sizes)):
        mcmc = MCMC(
            kernel,
            num_warmup=config.num_warmup if i == 0 else 0,
            num_samples=sizes[i],
            num_chains=config.num_chains,
            chain_method=config.chain_method,
            progress_bar=progress_bar,
        )
        if i == 0:
            mcmc.run(rng_key, extra_fields=extra_fields, **run_args)
        else:
            mcmc.post_warmup_state = state
            mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=extra_fields, **run_args)
        state = jax.device_get(mcmc.last_state)
        block_payload = {
            f"s.{k}": np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()
        }
        block_payload.update(
            {
                f"e.{k}": np.asarray(v)
                for k, v in mcmc.get_extra_fields(group_by_chain=True).items()
            }
        )
        np.savez(checkpoint_dir / f"block_{i:04d}.npz", **block_payload)
        with state_path.open("wb") as fh:
            pickle.dump(state, fh)
        cursor_path.write_text(
            json.dumps({"identity": identity, "blocks_done": i + 1, "block_sizes": sizes}),
            encoding="utf-8",
        )
        logger.info("Checkpoint block %d/%d written", i + 1, len(sizes))

    samples, extra_flat = _load_blocks(checkpoint_dir, len(sizes))
    return mcmc, samples, extra_flat, start_block > 0


def _run_sampling(
    kernel: NUTS,
    config: MCMCConfig,
    rng_key,
    run_args: dict,
    extra_field_names: tuple[str, ...],
    progress_bar: bool,
    checkpoint_dir: Path | None,
) -> tuple[MCMC | None, dict, dict, bool]:
    """(mcmc, grouped samples, extra fields, resumed) — blocked or single-shot."""
    if config.checkpoint_every_draws:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_every_draws requires a checkpoint_dir")
        return _run_blocked(
            kernel, config, rng_key, run_args, extra_field_names, progress_bar,
            Path(checkpoint_dir),
        )
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method=config.chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, extra_fields=extra_field_names, **run_args)
    return mcmc, mcmc.get_samples(group_by_chain=True), mcmc.get_extra_fields(), False


def fit_model(
    model: Callable,
    model_args: dict,
    config: MCMCConfig | None = None,
    progress_bar: bool = True,
    exclude_from_idata: tuple[str, ...] | None = None,
    exclude_from_collection: tuple[str, ...] | None = None,
    checkpoint_dir: Path | None = None,
) -> FitResult:
    """Fit NumPyro model via MCMC with GPU acceleration.

    Runs NUTS sampling with the specified configuration, converts results to
    ArviZ InferenceData, and logs basic statistics.

    Parameters
    ----------
    model : Callable
        NumPyro model function to fit (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Must include:
        - artist_idx: Integer array mapping observations to artists
        - album_seq: Integer array with album sequence numbers
        - prev_score: Float array with previous album scores
        - X: Feature matrix
        - y: Target scores
        - n_artists: Number of unique artists
        - max_seq: Maximum album sequence number
        Optional (for heteroscedastic models):
        - n_reviews: Array of per-observation review counts
        - n_exponent: Fixed exponent for noise scaling
        - learn_n_exponent: Whether to sample exponent from prior
    config : MCMCConfig, optional
        MCMC configuration. If None, uses default MCMCConfig().
    progress_bar : bool, default True
        Whether to display NumPyro's progress bar during sampling.
    exclude_from_idata : tuple of str, optional
        Sample site names to exclude from InferenceData. These sites are filtered
        out after calling mcmc.get_samples() to prevent large auxiliary tensors
        (e.g., "rw_raw") from consuming memory during InferenceData
        construction. If None, all sampled sites are included.
    exclude_from_collection : tuple of str, optional
        Sample site names to exclude from NumPyro's collected states DURING
        sampling (``~z.<site>`` extra_fields). Unlike exclude_from_idata,
        this prevents the sampler from ever storing the site's draws on
        device, cutting peak GPU memory (~96% for the rw_raw tensor at
        production settings). The posterior for all other sites is unchanged
        (guarded by parity tests). Sites excluded here never reach
        get_samples(), so the post-hoc exclude_from_idata filter for the
        same site becomes a no-op fallback.
    checkpoint_dir : Path, optional
        Where checkpointed sampling persists per-block draws, the sampler
        state, and its cursor. Required when
        ``config.checkpoint_every_draws`` is set; ignored otherwise.

    Returns
    -------
    FitResult
        Container with MCMC object, InferenceData, divergence count,
        runtime, and GPU info.

    Example
    -------
    >>> from panelcast.models.bayes import user_score_model, fit_model, MCMCConfig
    >>> config = MCMCConfig(num_warmup=100, num_samples=100)
    >>> result = fit_model(user_score_model, model_args, config=config)
    >>> print(f"Divergences: {result.divergences}")
    >>> print(f"Runtime: {result.runtime_seconds:.1f}s")

    Notes
    -----
    Divergences are logged but do not cause failure. Phase 7 handles
    diagnostic thresholds and convergence assessment.
    """
    if config is None:
        config = MCMCConfig()

    # Get GPU info before fitting
    gpu_info = get_gpu_info()
    logger.info(f"GPU info: {gpu_info}")
    logger.info(f"JAX default backend: {jax.default_backend()}")

    logger.info(f"NUTS init strategy: {config.init_strategy}")

    kernel = NUTS(
        model,
        max_tree_depth=config.max_tree_depth,
        target_accept_prob=config.target_accept_prob,
        init_strategy=_resolve_init_strategy(config.init_strategy),
    )

    # Generate random key (using modern JAX API)
    rng_key = random.key(config.seed)

    # Run MCMC with timing
    logger.info(
        f"Starting MCMC: {config.num_chains} chains, "
        f"{config.num_warmup} warmup, {config.num_samples} samples"
    )
    start_time = time.perf_counter()

    # Separate metadata-only keys from actual model parameters
    _metadata_keys = {"n_ref_method"}
    run_args = {k: v for k, v in model_args.items() if k not in _metadata_keys}

    extra_field_names: list[str] = ["diverging", "num_steps"]
    if exclude_from_collection:
        # NumPyro's "~z.<site>" syntax removes the site from the collected
        # states during sampling (see numpyro.infer.mcmc collect_fields /
        # remove_sites handling) — the in-sampler memory win.
        for site in exclude_from_collection:
            extra_field_names.append(f"~z.{site}")
            logger.info("Excluding '%s' from in-sampler collection (~z.)", site)

    mcmc, samples, extra_fields, resumed_from_checkpoint = _run_sampling(
        kernel, config, rng_key, run_args, tuple(extra_field_names), progress_bar, checkpoint_dir
    )

    runtime_seconds = time.perf_counter() - start_time

    # Measure peak GPU allocation right after sampling, before any
    # InferenceData construction allocates on top of it.
    peak_gpu_memory_bytes = measure_peak_gpu_bytes()
    if peak_gpu_memory_bytes is not None:
        logger.info(f"Peak GPU memory: {peak_gpu_memory_bytes / (1024**3):.2f} GiB")

    # Count divergences
    divergences = int(np.asarray(extra_fields["diverging"]).sum())

    logger.info(f"MCMC completed in {runtime_seconds:.1f}s")
    logger.info(f"Divergences: {divergences}")
    if divergences > 0:
        logger.warning(
            f"Found {divergences} divergent transitions. "
            "Consider increasing target_accept_prob or checking model specification."
        )

    tree_depth_saturation = _log_tree_depth_saturation(extra_fields, config.max_tree_depth)

    # Release memory from MCMC warmup before processing samples
    gc.collect()

    # Filter out excluded sample sites (post-filtering to avoid OOM on large tensors)
    if exclude_from_idata:
        excluded = [k for k in samples if k in exclude_from_idata]
        if excluded:
            # Log excluded sites with estimated memory sizes
            for site in excluded:
                size_mb = samples[site].nbytes / (1024 * 1024)
                logger.info("Excluded '%s' from InferenceData (%.1f MB)", site, size_mb)
        samples = {k: v for k, v in samples.items() if k not in exclude_from_idata}

    if not samples:
        raise ValueError("exclude_from_idata removed all sample sites; cannot build InferenceData.")

    # Build InferenceData manually with samples
    first_var = next(iter(samples.values()))
    n_chains, n_draws = first_var.shape[:2]

    # Convert to xarray Dataset
    posterior_dict = {}
    for var_name, var_data in samples.items():
        # Convert JAX arrays to numpy
        var_data = np.asarray(var_data)
        # Shape is (chains, draws, *var_shape)
        var_shape = var_data.shape[2:]
        if len(var_shape) == 0:
            # Scalar parameter
            dims = ["chain", "draw"]
        elif len(var_shape) == 1:
            # 1D parameter (e.g., beta, artist effects)
            dims = ["chain", "draw", f"{var_name}_dim_0"]
        else:
            # Multi-dimensional parameter
            dims = ["chain", "draw"] + [f"{var_name}_dim_{i}" for i in range(len(var_shape))]

        posterior_dict[var_name] = xr.DataArray(
            data=var_data,
            dims=dims,
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )

    posterior_ds = xr.Dataset(posterior_dict)

    # Sample stats (divergences, etc.) from the flat chain-major extra fields
    sample_stats_dict = {}
    for field_name, field_data in extra_fields.items():
        # Convert JAX arrays to numpy
        field_data = np.asarray(field_data)
        # extra_fields are flat (n_samples_total,) - reshape to (chains, draws, ...)
        # for ArviZ compatibility
        if field_data.ndim == 1:
            reshaped = field_data.reshape(n_chains, n_draws)
            dims = ["chain", "draw"]
            extra_dims = []
        else:
            reshaped = field_data.reshape(n_chains, n_draws, *field_data.shape[1:])
            # Add dimension names for any extra dimensions
            extra_dims = [f"{field_name}_dim_{i}" for i in range(len(reshaped.shape) - 2)]
            dims = ["chain", "draw"] + extra_dims

        all_coords = {"chain": range(n_chains), "draw": range(n_draws)}
        for i, dim_name in enumerate(extra_dims):
            all_coords[dim_name] = range(reshaped.shape[i + 2])

        sample_stats_dict[field_name] = xr.DataArray(
            data=reshaped,
            dims=dims,
            coords=all_coords,
        )

    sample_stats_ds = xr.Dataset(sample_stats_dict)

    # Create InferenceData
    idata = az.InferenceData(posterior=posterior_ds, sample_stats=sample_stats_ds)

    # Prepare data groups for InferenceData with explicit dimensions
    # CRITICAL: Raw numpy arrays cause ArviZ to misinterpret shapes as (chains, draws)
    # which triggers OOM when it tries to allocate per-chain statistics for 41k "chains"
    for required_key in ("y", "X"):
        if required_key not in model_args:
            raise ValueError(
                f"model_args missing required key '{required_key}'. "
                f"Available keys: {list(model_args.keys())}"
            )
    n_obs = len(model_args["y"])
    n_features = model_args["X"].shape[1]

    observed_data_ds = xr.Dataset(
        {
            "y": xr.DataArray(
                np.asarray(model_args["y"]),
                dims=["obs"],
                coords={"obs": range(n_obs)},
            )
        }
    )

    constant_data_ds = xr.Dataset(
        {
            "X": xr.DataArray(
                np.asarray(model_args["X"]),
                dims=["obs", "feature"],
                coords={"obs": range(n_obs), "feature": range(n_features)},
            ),
            "artist_idx": xr.DataArray(
                np.asarray(model_args["artist_idx"]),
                dims=["obs"],
                coords={"obs": range(n_obs)},
            ),
            "album_seq": xr.DataArray(
                np.asarray(model_args["album_seq"]),
                dims=["obs"],
                coords={"obs": range(n_obs)},
            ),
            "prev_score": xr.DataArray(
                np.asarray(model_args["prev_score"]),
                dims=["obs"],
                coords={"obs": range(n_obs)},
            ),
        }
    )

    # Include n_reviews for heteroscedastic models (if present)
    if "n_reviews" in model_args:
        constant_data_ds["n_reviews"] = xr.DataArray(
            np.asarray(model_args["n_reviews"]),
            dims=["obs"],
            coords={"obs": range(n_obs)},
        )

    # Include n_ref and n_ref_method for sigma-ref reparameterization reproducibility
    if "n_ref" in model_args and model_args["n_ref"] is not None:
        constant_data_ds["n_ref"] = xr.DataArray(
            np.float64(model_args["n_ref"]),
        )
        constant_data_ds["n_ref_method"] = xr.DataArray(
            np.array(model_args.get("n_ref_method", "median"), dtype="U"),
        )

    # Add groups only if they don't already exist
    existing_groups = set(idata.groups())
    if "observed_data" not in existing_groups:
        idata.add_groups(observed_data=observed_data_ds)
    if "constant_data" not in existing_groups:
        idata.add_groups(constant_data=constant_data_ds)

    logger.info(f"InferenceData groups: {list(idata.groups())}")

    return FitResult(
        mcmc=mcmc,
        idata=idata,
        divergences=divergences,
        runtime_seconds=runtime_seconds,
        gpu_info=gpu_info,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        tree_depth_saturation=tree_depth_saturation,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )
