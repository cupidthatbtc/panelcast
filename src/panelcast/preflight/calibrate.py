"""Memory calibration for extrapolation to arbitrary sample counts.

Provides two-point linear calibration to separate fixed JIT overhead from
per-sample memory cost, enabling accurate memory extrapolation for any
target sample count.

The calibration runs mini-MCMC at two sample counts (10 and 50) to fit a
linear model: memory = fixed_overhead + per_sample * samples

Example:
    >>> from panelcast.preflight.calibrate import run_calibration, CalibrationResult
    >>> result = run_calibration(model_args)
    >>> projected_gb = result.extrapolate(2000)  # Project to 2000 samples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CalibrationError",
    "CalibrationResult",
    "CALIBRATION_SAMPLES",
    "calculate_calibration",
    "run_calibration",
]

# Two calibration points per CONTEXT.md decision
CALIBRATION_SAMPLES: tuple[int, int] = (10, 50)


class CalibrationError(Exception):
    """Error during calibration process.

    Raised when:
    - Calibration gives negative fixed overhead (invalid linear fit)
    - Mini-MCMC fails during calibration
    - Timeout during calibration runs
    """


@dataclass(frozen=True)
class CalibrationResult:
    """Result of two-point memory calibration.

    Provides extrapolation capability for predicting memory usage at
    any target sample count based on calibration measurements.

    Attributes:
        fixed_overhead_gb: Y-intercept representing JIT/model setup cost in GB.
        per_sample_gb: Slope representing per-sample storage cost in GB.
        calibration_points: Tuple of (samples, gb) measurement pairs.
        config_hash: Hash of model config for cache key lookup.
        calibration_time: Total calibration time in seconds.
    """

    fixed_overhead_gb: float
    per_sample_gb: float
    calibration_points: tuple[tuple[int, float], tuple[int, float]]
    config_hash: str
    calibration_time: float

    def extrapolate(self, target_samples: int) -> float:
        """Project memory for target sample count.

        Args:
            target_samples: Number of samples to project memory for.

        Returns:
            Projected memory in GB using linear extrapolation:
            fixed_overhead_gb + per_sample_gb * target_samples
        """
        return self.fixed_overhead_gb + self.per_sample_gb * target_samples


def calculate_calibration(
    point1: tuple[int, float], point2: tuple[int, float]
) -> tuple[float, float]:
    """Calculate slope and intercept from two calibration points.

    Uses standard linear regression from two points to separate
    fixed overhead (intercept) from per-sample cost (slope).

    Args:
        point1: First calibration point as (samples, gb).
        point2: Second calibration point as (samples, gb).

    Returns:
        Tuple of (fixed_overhead_gb, per_sample_gb).

    Raises:
        CalibrationError: If the calculated fixed overhead is negative,
            which indicates an invalid linear fit (possibly due to
            non-linear memory behavior or measurement noise).
    """
    samples1, gb1 = point1
    samples2, gb2 = point2

    if samples1 == samples2:
        raise CalibrationError(
            f"Cannot calibrate with identical sample counts (both points have {samples1} samples)"
        )

    # slope = (y2 - y1) / (x2 - x1)
    per_sample_gb = (gb2 - gb1) / (samples2 - samples1)

    # intercept = y1 - slope * x1
    fixed_overhead_gb = gb1 - per_sample_gb * samples1

    if fixed_overhead_gb < 0:
        raise CalibrationError(
            f"Calibration failed: negative fixed overhead ({fixed_overhead_gb:.2f} GB). "
            "This may indicate non-linear memory behavior or measurement variance. "
            "Try running calibration again or use --preflight for formula-based estimation."
        )

    return fixed_overhead_gb, per_sample_gb


def run_calibration(
    model_args: dict[str, Any],
    timeout_seconds: int = 120,
    model_signature: dict[str, Any] | None = None,
    exclude_collection: tuple[str, ...] = (),
    num_chains: int = 1,
) -> CalibrationResult:
    """Run two-point calibration for memory extrapolation.

    Runs mini-MCMC at two sample counts (10 and 50) to establish a linear
    relationship between sample count and memory usage. This separates
    the fixed JIT compilation overhead from per-sample storage cost.

    Calibration must run at the TARGET chain count: sequential chains
    accumulate collected draws on device, so a single-chain calibration
    under-projects multi-chain runs (measured -31% for 2 chains; see
    outputs/experiments/preflight_validation.json).

    Args:
        model_args: Dictionary of model arguments (will be serialized to JSON).
            Should contain: artist_idx, album_seq, prev_score, X, y,
            n_artists, max_seq, and optionally n_reviews, n_exponent.
        timeout_seconds: Maximum time for each mini-run (default 120s).
        model_signature: Gate-relevant flags included in the cache key so a
            calibration never serves a structurally different model. When
            None, a minimal signature is derived from model_args.
        exclude_collection: Site names excluded from in-sampler collection in
            the mini-runs (mirrors the production fit's memory gate so the
            calibration measures the structure it projects for).
        num_chains: Chain count of the production run being projected
            (calibration mini-runs use the same count).

    Returns:
        CalibrationResult with fixed_overhead_gb, per_sample_gb, and
        extrapolate() method.

    Raises:
        CalibrationError: If calibration fails (e.g., mini-run fails,
            negative fixed overhead).
    """
    # Import at function level to avoid circular imports
    import time

    from panelcast.preflight.cache import compute_config_hash
    from panelcast.preflight.full_check import (
        _derive_dimensions_from_model_args,
        _run_mini_mcmc_subprocess,
        derive_model_signature,
        serialize_model_args,
    )

    if model_signature is None:
        model_signature = derive_model_signature(model_args)
        if exclude_collection:
            model_signature["exclude_from_collection"] = sorted(exclude_collection)
    # The chain count changes what the calibration measures; key the cache
    # on it regardless of where the signature came from.
    model_signature = {**model_signature, "num_chains": num_chains}

    start_time = time.perf_counter()

    # Serialize model args once (reused for both runs)
    args_path = serialize_model_args(model_args)

    try:
        points: list[tuple[int, float]] = []

        for num_samples in CALIBRATION_SAMPLES:
            result = _run_mini_mcmc_subprocess(
                args_path,
                timeout_seconds=timeout_seconds,
                num_warmup=10,
                num_samples=num_samples,
                num_chains=num_chains,
                exclude_collection=exclude_collection,
            )

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                raise CalibrationError(f"Calibration failed at {num_samples} samples: {error_msg}")

            peak_gb = result["peak_memory_bytes"] / (1024**3)
            points.append((num_samples, peak_gb))

        # Calculate linear fit
        fixed, per_sample = calculate_calibration(points[0], points[1])

        # Compute config hash for caching using shared dimension derivation
        n_obs, n_artists, n_features, max_seq = _derive_dimensions_from_model_args(model_args)
        config_hash = compute_config_hash(
            n_obs, n_artists, n_features, max_seq, model_signature=model_signature
        )

        total_time = time.perf_counter() - start_time

        return CalibrationResult(
            fixed_overhead_gb=fixed,
            per_sample_gb=per_sample,
            calibration_points=(tuple(points[0]), tuple(points[1])),
            config_hash=config_hash,
            calibration_time=total_time,
        )

    finally:
        # Always cleanup temp file
        args_path.unlink(missing_ok=True)
