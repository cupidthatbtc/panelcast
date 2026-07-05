"""Calibration cache for storing and retrieving calibration results.

Caches calibration results keyed by model configuration hash to avoid
re-running calibration when the same model configuration is used.

Cache location: ~/.cache/panelcast/calibration/

Example:
    >>> from panelcast.preflight.cache import (
    ...     compute_config_hash, load_calibration_cache, save_calibration_cache
    ... )
    >>> hash = compute_config_hash(1000, 100, 20, 10)
    >>> cached = load_calibration_cache(hash)
    >>> if cached is None:
    ...     result = run_calibration(...)  # Run fresh calibration
    ...     save_calibration_cache(result)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panelcast.preflight.calibrate import CalibrationResult

__all__ = [
    "CACHE_DIR",
    "compute_config_hash",
    "load_calibration_cache",
    "save_calibration_cache",
]

logger = logging.getLogger(__name__)

# Standard XDG cache location
CACHE_DIR: Path = Path.home() / ".cache" / "panelcast" / "calibration"


def compute_config_hash(
    n_observations: int,
    n_artists: int,
    n_features: int,
    max_seq: int,
    model_signature: dict | None = None,
) -> str:
    """Compute hash of model configuration for cache key.

    The cache key is based on factors that affect memory scaling:
    - n_observations: Number of data points (affects data array sizes)
    - n_artists: Number of unique artists (affects artist effect arrays)
    - n_features: Number of features (affects X matrix size)
    - max_seq: Maximum album sequence (affects time-varying effects)
    - the NumPyro version (collection internals change between releases)
    - model_signature: structural gates that change what the sampler
      allocates (descriptor hash, latent process, target transform,
      heteroscedastic flags, ...). A calibration measured under one model
      structure must never serve projections for a different one.

    Args:
        n_observations: Number of observations in the dataset.
        n_artists: Number of unique artists.
        n_features: Number of features in the feature matrix.
        max_seq: Maximum album sequence number.
        model_signature: Optional mapping of gate-relevant flags to include
            in the key. Keys/values must be JSON-serializable.

    Returns:
        16-character hex string hash of the configuration.
    """
    import numpyro

    config = {
        "n_observations": n_observations,
        "n_artists": n_artists,
        "n_features": n_features,
        "max_seq": max_seq,
        "numpyro_version": numpyro.__version__,
    }
    if model_signature:
        config["model_signature"] = dict(sorted(model_signature.items()))
    config_json = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def load_calibration_cache(config_hash: str) -> CalibrationResult | None:
    """Load cached calibration result if valid.

    Args:
        config_hash: 16-character hash from compute_config_hash().

    Returns:
        CalibrationResult if cache hit and valid, None otherwise.
    """
    # Import at function level to avoid circular imports
    from panelcast.preflight.calibrate import CalibrationResult

    cache_file = CACHE_DIR / f"{config_hash}.json"

    if not cache_file.exists():
        logger.debug(f"Cache miss: {cache_file} does not exist")
        return None

    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)

        # Validate required keys
        required_keys = [
            "fixed_overhead_gb",
            "per_sample_gb",
            "calibration_points",
            "config_hash",
            "calibration_time",
        ]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            logger.warning(f"Cache invalid: missing keys {missing_keys}")
            return None

        # Convert calibration_points from list to tuple of tuples
        points = data["calibration_points"]
        data["calibration_points"] = (tuple(points[0]), tuple(points[1]))

        logger.debug(f"Cache hit: loaded from {cache_file}")
        return CalibrationResult(**data)

    except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
        logger.warning(f"Cache load error: {e}")
        return None


def save_calibration_cache(result: CalibrationResult) -> None:
    """Save calibration result to cache.

    Uses atomic write pattern (write to .tmp then rename) to prevent
    partial writes on crash or interrupt.

    Args:
        result: CalibrationResult to cache.
    """
    # Create cache directory if needed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE_DIR / f"{result.config_hash}.json"
    temp_file = cache_file.with_suffix(".tmp")

    try:
        # Convert to dict for JSON serialization
        data = asdict(result)

        # Convert tuple of tuples to list of lists for JSON
        data["calibration_points"] = [
            list(data["calibration_points"][0]),
            list(data["calibration_points"][1]),
        ]

        # Write to temp file
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Atomic replace — overwrites an existing cache on every platform.
        # (Path.rename raises FileExistsError over an existing dest on Windows,
        # which the OSError handler below would silently swallow, discarding a
        # fresh --recalibrate result and leaving the stale cache in place.)
        temp_file.replace(cache_file)
        logger.debug(f"Cache saved: {cache_file}")

    except OSError as e:
        logger.warning(f"Failed to save cache: {e}")
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
