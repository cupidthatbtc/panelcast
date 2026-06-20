"""Full preflight check with mini-MCMC memory measurement.

Runs a mini-MCMC in a subprocess to measure actual peak GPU memory,
providing ~95% accuracy compared to formula-based estimation (~70-80%).

The subprocess approach guarantees a clean CUDA context for accurate
measurement, as CUDA contexts persist within a process.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from panelcast.gpu_memory import query_gpu_memory
from panelcast.pipelines.errors import GpuMemoryError
from panelcast.preflight import ExtrapolationResult, FullPreflightResult, PreflightStatus

if TYPE_CHECKING:
    from panelcast.preflight.calibrate import CalibrationResult

logger = logging.getLogger(__name__)


def _derive_dimensions_from_model_args(model_args: dict) -> tuple[int, int, int, int]:
    """Derive cache-key dimensions from model_args.

    This is the canonical source of dimension extraction for cache hash
    computation, used by both full_check and calibrate modules to ensure
    consistent cache keys.

    Args:
        model_args: Dictionary of model arguments. Expected keys:
            - y: Array-like of observations
            - X: 2D array-like of features
            - n_artists: Number of unique artists
            - max_seq: Maximum album sequence number

    Returns:
        Tuple of (n_observations, n_artists, n_features, max_seq).
    """
    # n_observations from y length
    y = model_args.get("y")
    n_observations = len(y) if y is not None else 0

    # n_artists directly from model_args
    n_artists = model_args.get("n_artists", 0)

    # n_features from X shape (handle array-like objects)
    X = model_args.get("X")
    if X is None:
        n_features = 0
    elif hasattr(X, "shape"):
        if len(X.shape) == 1:
            raise ValueError(
                f"X must be a 2D array, got 1D array with shape {X.shape}. "
                "Reshape to (n_samples, n_features) or use X[:, np.newaxis] for single feature."
            )
        n_features = X.shape[1]
    else:
        n_features = len(X[0]) if X else 0

    # max_seq directly from model_args
    max_seq = model_args.get("max_seq", 0)

    return n_observations, n_artists, n_features, max_seq


def derive_model_signature(model_args: dict) -> dict:
    """Minimal gate signature derivable from model_args alone.

    Captures the structural flags present in the serialized model arguments
    (heteroscedastic gates, likelihood df). Callers with access to the full
    run configuration should extend this with descriptor hash and model
    gates (latent_process, target_transform) before hashing.
    """
    signature: dict = {}
    for key in ("n_exponent", "learn_n_exponent", "likelihood_df"):
        value = model_args.get(key)
        if isinstance(value, (bool, int, float, str)):
            signature[key] = value
    return signature


def calculate_headroom_percent(available_gb: float, peak_gb: float) -> float:
    """Calculate headroom as percentage of available memory.

    Args:
        available_gb: Available GPU memory in GB.
        peak_gb: Peak memory usage in GB.

    Returns:
        Headroom percentage (0-100), or -100 if no memory available.
    """
    if available_gb > 0:
        return ((available_gb - peak_gb) / available_gb) * 100
    return -100.0


def serialize_model_args(model_args: dict) -> Path:
    """Serialize model arguments to temporary JSON file.

    Converts JAX arrays to Python lists for JSON compatibility.
    Caller is responsible for cleaning up the temp file.

    Args:
        model_args: Dictionary of model arguments, may contain JAX arrays.

    Returns:
        Path to temporary JSON file containing serialized arguments.

    Example:
        >>> args_path = serialize_model_args({"X": jnp.ones(10), "n": 5})
        >>> try:
        ...     # Use args_path
        ...     pass
        ... finally:
        ...     args_path.unlink()  # Cleanup
    """
    serializable = {}
    for key, value in model_args.items():
        if hasattr(value, "tolist"):
            # Convert JAX/NumPy arrays to Python lists
            serializable[key] = value.tolist()
        else:
            # Scalars (n_artists, max_seq) can be serialized directly
            serializable[key] = value

    # Create temp file (caller responsible for cleanup)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="mini_run_args_")
    with os.fdopen(fd, "w") as f:
        json.dump(serializable, f)

    return Path(path)


def _run_mini_mcmc_subprocess(
    args_path: Path,
    timeout_seconds: int,
    num_warmup: int = 10,
    num_samples: int = 1,
    num_chains: int = 1,
    prefix: str = "user",
    exclude_collection: tuple[str, ...] = (),
) -> dict:
    """Run mini-MCMC in subprocess and return measured memory.

    Spawns a subprocess with XLA preallocation disabled for accurate
    memory measurement.

    Args:
        args_path: Path to JSON file with model arguments.
        timeout_seconds: Maximum time to wait for subprocess.
        num_warmup: Number of warmup iterations (default 10).
        num_samples: Number of post-warmup samples (default 1).
        num_chains: Number of sequential chains (default 1).
        prefix: Posterior-site prefix / score type (default "user").
        exclude_collection: Site names excluded from in-sampler collection
            (mirrors the production fit's memory gate).

    Returns:
        Dictionary with keys:
        - success: bool - Whether mini-run completed
        - peak_memory_bytes: int - Peak GPU memory in bytes
        - runtime_seconds: float - Mini-run execution time
        - error: str (only if success=False)
    """
    # Set up environment with preallocation disabled for accurate measurement
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

    command = [
        sys.executable,
        "-m",
        "panelcast.preflight.mini_run",
        str(args_path),
        "--num-warmup",
        str(num_warmup),
        "--num-samples",
        str(num_samples),
        "--num-chains",
        str(num_chains),
        "--prefix",
        prefix,
    ]
    if exclude_collection:
        command.extend(["--exclude-collection", ",".join(exclude_collection)])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        if result.returncode != 0:
            # Subprocess failed
            error_msg = result.stderr.strip() or "Unknown subprocess error"
            return {
                "success": False,
                "error": error_msg,
                "peak_memory_bytes": 0,
                "runtime_seconds": 0.0,
            }

        # Parse JSON output from subprocess
        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Mini-run timeout exceeded ({timeout_seconds}s). "
            "Model may be too large for quick measurement.",
            "peak_memory_bytes": 0,
            "runtime_seconds": float(timeout_seconds),
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse mini-run output: {e}",
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }


def run_full_preflight_check(
    model_args: dict,
    headroom_target: float = 0.20,
    timeout_seconds: int = 120,
) -> FullPreflightResult:
    """Run full preflight check with mini-MCMC measurement.

    Measures actual peak GPU memory by running a mini-MCMC (1 chain,
    10 warmup, 1 sample) in a subprocess, then compares against
    available VRAM.

    Args:
        model_args: Arguments to pass to model (will be serialized to JSON).
            Should contain: artist_idx, album_seq, prev_score, X, y,
            n_artists, max_seq.
        headroom_target: Target headroom as fraction (default 0.20 = 20%).
            PASS requires at least this much headroom.
        timeout_seconds: Maximum time for mini-run (default 120s).

    Returns:
        FullPreflightResult with status, measured peak memory, and suggestions.

    Example:
        >>> result = run_full_preflight_check(model_args)
        >>> if result.status == PreflightStatus.FAIL:
        ...     raise SystemExit(result.exit_code)
    """
    # Step 1: Query available GPU memory via NVML
    try:
        gpu_info = query_gpu_memory()
    except GpuMemoryError as e:
        return FullPreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            measured_peak_gb=0.0,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            mini_run_seconds=0.0,
            message=f"Cannot query GPU: {e}",
            suggestions=("Use --preflight for estimation without GPU query",),
        )

    # Step 2: Serialize model args to temp file
    args_path = serialize_model_args(model_args)

    try:
        # Step 3: Run mini-MCMC subprocess
        result = _run_mini_mcmc_subprocess(args_path, timeout_seconds)
    finally:
        # Always cleanup temp file
        args_path.unlink(missing_ok=True)

    # Step 4: Handle subprocess failure
    if not result.get("success", False):
        return FullPreflightResult(
            status=PreflightStatus.CANNOT_CHECK,
            measured_peak_gb=0.0,
            available_gb=gpu_info.free_gb,
            total_gpu_gb=gpu_info.total_gb,
            headroom_percent=0.0,
            mini_run_seconds=result.get("runtime_seconds", 0.0),
            message=f"Mini-run failed: {result.get('error', 'Unknown error')}",
            suggestions=(
                "Use --preflight for formula-based estimation",
                "Check GPU driver status with nvidia-smi",
            ),
            device_name=gpu_info.device_name,
        )

    # Step 5: Calculate headroom
    peak_bytes = result["peak_memory_bytes"]
    peak_gb = peak_bytes / (1024**3)
    available_gb = gpu_info.free_gb

    headroom_percent = calculate_headroom_percent(available_gb, peak_gb)

    # Step 6: Determine status
    if peak_gb > available_gb:
        status = PreflightStatus.FAIL
    elif peak_gb <= available_gb * (1 - headroom_target):
        # Sufficient headroom
        status = PreflightStatus.PASS
    else:
        # Fits but low headroom
        status = PreflightStatus.WARNING

    # Step 7: Generate message and suggestions
    message = _generate_message(status, peak_gb, available_gb, headroom_percent)
    suggestions = _generate_suggestions(status, peak_gb, available_gb)

    return FullPreflightResult(
        status=status,
        measured_peak_gb=peak_gb,
        available_gb=available_gb,
        total_gpu_gb=gpu_info.total_gb,
        headroom_percent=headroom_percent,
        mini_run_seconds=result.get("runtime_seconds", 0.0),
        message=message,
        suggestions=suggestions,
        device_name=gpu_info.device_name,
    )


def _generate_message(
    status: PreflightStatus,
    peak_gb: float,
    available_gb: float,
    headroom_percent: float,
) -> str:
    """Generate human-readable status message."""
    match status:
        case PreflightStatus.PASS:
            return (
                f"Full preflight passed: {peak_gb:.2f} GB measured peak, "
                f"{available_gb:.1f} GB available ({headroom_percent:.0f}% headroom)"
            )
        case PreflightStatus.WARNING:
            return (
                f"Full preflight warning: {peak_gb:.2f} GB measured peak, "
                f"{available_gb:.1f} GB available (low headroom: {headroom_percent:.0f}%)"
            )
        case PreflightStatus.FAIL:
            return (
                f"Full preflight failed: {peak_gb:.2f} GB measured peak "
                f"exceeds {available_gb:.1f} GB available"
            )
        case PreflightStatus.CANNOT_CHECK:
            return "Cannot complete full preflight check"
        case _:
            return f"Unknown full preflight status: {status}"


def _generate_suggestions(
    status: PreflightStatus,
    peak_gb: float,
    available_gb: float,
) -> tuple[str, ...]:
    """Generate configuration adjustment suggestions."""
    if status == PreflightStatus.PASS:
        return ()

    suggestions: list[str] = []

    if status == PreflightStatus.FAIL:
        deficit_gb = peak_gb - available_gb
        suggestions.append(f"Need {deficit_gb:.1f} GB more GPU memory")
        suggestions.append("Try reducing --num-chains (most effective)")
        suggestions.append("Try reducing data subset or feature count")

    elif status == PreflightStatus.WARNING:
        suggestions.append("Memory is tight; consider reducing --num-chains")
        suggestions.append("Close other GPU-using applications")

    return tuple(suggestions)


def _generate_extrapolation_message(
    status: PreflightStatus,
    projected_gb: float,
    available_gb: float,
    headroom_percent: float,
    target_samples: int,
) -> str:
    """Generate human-readable status message for extrapolation result."""
    match status:
        case PreflightStatus.PASS:
            return (
                f"Extrapolation passed: {projected_gb:.1f} GB projected "
                f"for {target_samples:,} samples, "
                f"{available_gb:.1f} GB available ({headroom_percent:.0f}% headroom)"
            )
        case PreflightStatus.WARNING:
            return (
                f"Extrapolation warning: {projected_gb:.1f} GB projected "
                f"for {target_samples:,} samples, "
                f"{available_gb:.1f} GB available (low headroom: {headroom_percent:.0f}%)"
            )
        case PreflightStatus.FAIL:
            return (
                f"Extrapolation failed: {projected_gb:.1f} GB projected "
                f"for {target_samples:,} samples exceeds {available_gb:.1f} GB available"
            )
        case PreflightStatus.CANNOT_CHECK:
            return "Cannot complete extrapolation check"
        case _:
            return f"Unknown extrapolation status: {status}"


def _generate_extrapolation_suggestions(
    status: PreflightStatus,
    projected_gb: float,
    available_gb: float,
) -> tuple[str, ...]:
    """Generate suggestions for extrapolation result."""
    if status == PreflightStatus.PASS:
        return ()

    suggestions: list[str] = []

    if status == PreflightStatus.FAIL:
        deficit_gb = projected_gb - available_gb
        suggestions.append(f"Projected memory exceeds available by {deficit_gb:.1f} GB")
        suggestions.append("Try reducing --num-samples or --num-warmup")
        suggestions.append("Try reducing --num-chains (most effective)")
        suggestions.append("Try reducing data subset or feature count")

    elif status == PreflightStatus.WARNING:
        suggestions.append("Projected memory is close to available; may OOM during run")
        suggestions.append("Consider reducing --num-samples or --num-chains")
        suggestions.append("Close other GPU-using applications")

    return tuple(suggestions)


def _create_dummy_calibration(config_hash: str = "") -> CalibrationResult:
    """Create a minimal CalibrationResult for error cases.

    Args:
        config_hash: Optional config hash for cache key lookup.

    Returns:
        A CalibrationResult with zeroed values suitable for error cases.
    """
    from panelcast.preflight.calibrate import CalibrationResult

    return CalibrationResult(
        fixed_overhead_gb=0.0,
        per_sample_gb=0.0,
        calibration_points=((0, 0.0), (0, 0.0)),
        config_hash=config_hash,
        calibration_time=0.0,
    )


def run_extrapolated_preflight_check(
    model_args: dict,
    target_samples: int,
    n_observations: int,
    n_artists: int,
    n_features: int,
    max_seq: int,
    headroom_target: float = 0.20,
    timeout_seconds: int = 120,
    *,
    recalibrate: bool = False,
    model_signature: dict | None = None,
    exclude_collection: tuple[str, ...] = (),
    num_chains: int = 1,
) -> ExtrapolationResult:
    """Run preflight check with calibration and extrapolation to target samples.

    Performs two-point calibration (10 and 50 samples, at the production
    chain count) to measure memory scaling, then extrapolates to the target
    POST-WARMUP sample count per chain. Warmup iterations are not stored by
    the sampler and must not be folded into target_samples (measured:
    identical peaks at warmup 50 vs 250). Uses caching to avoid re-running
    calibration for the same model configuration.

    Note:
        The explicit dimension parameters (n_observations, n_artists, n_features,
        max_seq) are kept for backward compatibility but are NOT used for cache
        key computation. The cache key is derived directly from model_args to
        ensure consistency with the calibration storage in run_calibration().

    Args:
        model_args: Arguments to pass to model (will be serialized to JSON).
            Should contain: artist_idx, album_seq, prev_score, X, y,
            n_artists, max_seq.
        target_samples: POST-WARMUP samples per chain to project memory for
            (num_samples from the CLI; warmup draws are never stored).
        n_observations: Number of observations in the dataset (deprecated for
            cache key, derived from model_args instead).
        n_artists: Number of unique artists (deprecated for cache key, derived
            from model_args instead).
        n_features: Number of features in the feature matrix (deprecated for
            cache key, derived from model_args instead).
        max_seq: Maximum album sequence number (deprecated for cache key,
            derived from model_args instead).
        headroom_target: Target headroom as fraction (default 0.20 = 20%).
            PASS requires at least this much headroom.
        timeout_seconds: Maximum time for each mini-run (default 120s).
        recalibrate: If True, bypass cache and run fresh calibration.
        model_signature: Gate-relevant flags for the calibration cache key
            (descriptor hash, latent_process, target_transform, ...). When
            None, a minimal signature is derived from model_args.
        exclude_collection: Site names excluded from in-sampler collection in
            the calibration mini-runs (must mirror the production fit's
            memory gate; also folded into the derived cache signature).
        num_chains: Chain count of the production run. Calibration runs at
            this count — sequential chains accumulate collected draws, so a
            1-chain calibration under-projects multi-chain runs.

    Returns:
        ExtrapolationResult with status based on projected memory vs available.

    Example:
        >>> result = run_extrapolated_preflight_check(
        ...     model_args, target_samples=2000,
        ...     n_observations=1000, n_artists=100, n_features=20, max_seq=10
        ... )
        >>> if result.status == PreflightStatus.FAIL:
        ...     raise SystemExit(result.exit_code)
    """
    from panelcast.preflight.cache import (
        compute_config_hash,
        load_calibration_cache,
        save_calibration_cache,
    )
    from panelcast.preflight.calibrate import (
        CALIBRATION_SAMPLES,
        CalibrationError,
        run_calibration,
    )

    # Step 1: Query available GPU memory via NVML
    try:
        gpu_info = query_gpu_memory()
    except GpuMemoryError as e:
        return ExtrapolationResult(
            status=PreflightStatus.CANNOT_CHECK,
            measured_peak_gb=0.0,
            projected_gb=0.0,
            target_samples=target_samples,
            calibration_samples=sum(CALIBRATION_SAMPLES),
            uncertainty_percent=10.0,
            available_gb=0.0,
            total_gpu_gb=0.0,
            headroom_percent=0.0,
            calibration=_create_dummy_calibration(),
            from_cache=False,
            message=f"Cannot query GPU: {e}",
            suggestions=("Use --preflight for estimation without GPU query",),
        )

    # Step 2: Derive dimensions from model_args for consistent cache hash
    # (matches what run_calibration uses when storing to cache)
    derived_obs, derived_artists, derived_features, derived_seq = (
        _derive_dimensions_from_model_args(model_args)
    )

    # Log if explicit parameters differ from derived (diagnostic for callers)
    if (n_observations, n_artists, n_features, max_seq) != (
        derived_obs,
        derived_artists,
        derived_features,
        derived_seq,
    ):
        logger.debug(
            "Explicit dimensions (%d, %d, %d, %d) differ from derived (%d, %d, %d, %d); "
            "using derived for cache key",
            n_observations,
            n_artists,
            n_features,
            max_seq,
            derived_obs,
            derived_artists,
            derived_features,
            derived_seq,
        )

    if model_signature is None:
        model_signature = derive_model_signature(model_args)
        if exclude_collection:
            model_signature["exclude_from_collection"] = sorted(exclude_collection)
    # The chain count changes what the calibration measures; key the cache
    # on it regardless of where the signature came from (must match the
    # signature run_calibration stores under).
    model_signature = {**model_signature, "num_chains": num_chains}
    config_hash = compute_config_hash(
        derived_obs,
        derived_artists,
        derived_features,
        derived_seq,
        model_signature=model_signature,
    )

    # Step 3: Try loading from cache (unless recalibrate=True)
    calibration: CalibrationResult | None = None
    from_cache = False

    if not recalibrate:
        calibration = load_calibration_cache(config_hash)
        if calibration is not None:
            from_cache = True

    # Step 4: If cache miss or recalibrate, run fresh calibration
    if calibration is None:
        try:
            calibration = run_calibration(
                model_args,
                timeout_seconds=timeout_seconds,
                model_signature=model_signature,
                exclude_collection=exclude_collection,
                num_chains=num_chains,
            )
        except CalibrationError as e:
            return ExtrapolationResult(
                status=PreflightStatus.CANNOT_CHECK,
                measured_peak_gb=0.0,
                projected_gb=0.0,
                target_samples=target_samples,
                calibration_samples=sum(CALIBRATION_SAMPLES),
                uncertainty_percent=10.0,
                available_gb=gpu_info.free_gb,
                total_gpu_gb=gpu_info.total_gb,
                headroom_percent=0.0,
                calibration=_create_dummy_calibration(config_hash),
                from_cache=False,
                message=f"Calibration failed: {e}",
                suggestions=(
                    "Model may not fit in GPU even at 10 samples",
                    "Use --preflight for formula-based estimation",
                    "Try reducing model complexity",
                ),
                device_name=gpu_info.device_name,
            )

        # Step 5: Save to cache after fresh calibration
        save_calibration_cache(calibration)

    # Step 6: Extrapolate to target samples
    projected_gb = calibration.extrapolate(target_samples)

    # Calculate measured peak (max of calibration points)
    measured_peak_gb = max(
        calibration.calibration_points[0][1],
        calibration.calibration_points[1][1],
    )

    # Step 7: Determine status based on projected_gb vs available_gb
    available_gb = gpu_info.free_gb
    headroom_percent = calculate_headroom_percent(available_gb, projected_gb)

    if projected_gb > available_gb:
        status = PreflightStatus.FAIL
    elif projected_gb <= available_gb * (1 - headroom_target):
        # Sufficient headroom
        status = PreflightStatus.PASS
    else:
        # Fits but low headroom
        status = PreflightStatus.WARNING

    # Step 8: Generate message and suggestions
    message = _generate_extrapolation_message(
        status, projected_gb, available_gb, headroom_percent, target_samples
    )
    suggestions = _generate_extrapolation_suggestions(status, projected_gb, available_gb)

    return ExtrapolationResult(
        status=status,
        measured_peak_gb=measured_peak_gb,
        projected_gb=projected_gb,
        target_samples=target_samples,
        calibration_samples=sum(CALIBRATION_SAMPLES),
        uncertainty_percent=10.0,
        available_gb=available_gb,
        total_gpu_gb=gpu_info.total_gb,
        headroom_percent=headroom_percent,
        calibration=calibration,
        from_cache=from_cache,
        message=message,
        suggestions=suggestions,
        device_name=gpu_info.device_name,
    )
