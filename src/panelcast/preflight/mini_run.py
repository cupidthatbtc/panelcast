"""Mini-MCMC run for GPU memory measurement.

This module is designed to be run as a subprocess entry point for measuring
actual peak GPU memory usage during MCMC. It runs a minimal MCMC with
configurable warmup and sample counts and reports peak memory via JSON to stdout.

Usage:
    python -m panelcast.preflight.mini_run /path/to/model_args.json
    python -m panelcast.preflight.mini_run /path/to/model_args.json --num-warmup 10 --num-samples 50

The model_args.json file should contain:
    - artist_idx: List of integers mapping observations to artists
    - album_seq: List of integers with album sequence numbers
    - prev_score: List of floats with previous album scores
    - X: 2D list of floats (feature matrix)
    - y: List of floats (target scores)
    - n_artists: Integer count of unique artists
    - max_seq: Integer maximum album sequence number
    - n_reviews: Optional list of floats (per-observation review counts)
    - n_exponent: Optional float (heteroscedastic exponent)
    - learn_n_exponent: Optional bool (whether to sample exponent)

Output (stdout, JSON):
    {
        "success": true,
        "exit_code": 0,
        "peak_memory_bytes": 4567890123,
        "runtime_seconds": 45.2
    }

On error:
    {
        "success": false,
        "exit_code": 1,
        "error": "Error message here",
        "peak_memory_bytes": 0,
        "runtime_seconds": 0.0
    }

Note: All JSON output goes to stdout. Logs/warnings go to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["run_and_measure"]

logger = logging.getLogger(__name__)


def run_and_measure(
    model_args_path: Path,
    num_warmup: int = 10,
    num_samples: int = 1,
    num_chains: int = 1,
    prefix: str = "user",
    exclude_collection: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run mini-MCMC and measure peak GPU memory.

    Loads model arguments from JSON, runs MCMC with specified warmup and
    sample counts (default: 1 chain, 10 warmup, 1 sample), and returns
    peak memory usage from JAX device stats.

    Args:
        model_args_path: Path to JSON file with model arguments.
        num_warmup: Number of warmup iterations (default 10).
        num_samples: Number of post-warmup samples (default 1).
        num_chains: Number of sequential chains (default 1).
        prefix: Posterior-site prefix / score type (default "user").
        exclude_collection: Site names excluded from in-sampler collection
            (``~z.<site>``), mirroring the production fit's memory gate so
            calibration measures the same structure it projects for.

    Returns:
        Dictionary with:
            - success: True if measurement completed
            - peak_memory_bytes: Peak GPU memory in bytes
            - runtime_seconds: Wall-clock time for mini-run
    """
    # Import JAX/NumPyro at function level for subprocess isolation
    import jax.numpy as jnp
    from jax import random
    from numpyro.infer import MCMC, NUTS

    from panelcast.gpu_memory.measure import get_jax_memory_stats
    from panelcast.models.bayes.model import make_score_model

    # Load model args from JSON
    logger.info(f"Loading model args from {model_args_path}")
    with open(model_args_path, encoding="utf-8") as f:
        args_json = json.load(f)

    # Validate required keys exist before building model_args
    required_keys = ["artist_idx", "album_seq", "prev_score", "X", "y", "n_artists", "max_seq"]
    missing_keys = [k for k in required_keys if k not in args_json]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in model args JSON: {missing_keys}. "
            f"Present keys: {list(args_json.keys())}"
        )

    # Convert lists to JAX arrays where needed
    model_args: dict[str, Any] = {
        # Integer arrays
        "artist_idx": jnp.array(args_json["artist_idx"], dtype=jnp.int32),
        "album_seq": jnp.array(args_json["album_seq"], dtype=jnp.int32),
        # Float arrays
        "prev_score": jnp.array(args_json["prev_score"], dtype=jnp.float32),
        "X": jnp.array(args_json["X"], dtype=jnp.float32),
        "y": jnp.array(args_json["y"], dtype=jnp.float32),
        # Scalar integers
        "n_artists": args_json["n_artists"],
        "max_seq": args_json["max_seq"],
    }

    # Optional heteroscedastic parameters
    if "n_reviews" in args_json:
        model_args["n_reviews"] = jnp.array(args_json["n_reviews"], dtype=jnp.float32)
    if "n_exponent" in args_json:
        model_args["n_exponent"] = args_json["n_exponent"]
    if "learn_n_exponent" in args_json:
        model_args["learn_n_exponent"] = args_json["learn_n_exponent"]

    # Create NUTS kernel for the requested score-model prefix ("user" default
    # is consistent with CLI default behavior, most common use case)
    model = make_score_model(prefix)
    kernel = NUTS(model)

    # Configure MCMC: mini-run parameters for memory measurement
    # configurable warmup (captures JIT overhead), samples and chain count
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=False,
    )

    logger.info("Starting mini-MCMC run")
    start_time = time.perf_counter()

    # Run MCMC (optionally excluding sites from in-sampler collection)
    rng_key = random.key(0)
    extra_fields = tuple(f"~z.{site}" for site in exclude_collection)
    mcmc.run(rng_key, extra_fields=extra_fields, **model_args)

    runtime = time.perf_counter() - start_time
    logger.info(f"Mini-MCMC completed in {runtime:.2f}s")

    # Get peak memory from JAX device stats
    stats = get_jax_memory_stats()
    peak_bytes = stats.peak_bytes_in_use

    logger.info(f"Peak memory: {stats.peak_gb:.2f} GB")

    return {
        "success": True,
        "exit_code": 0,
        "peak_memory_bytes": peak_bytes,
        "runtime_seconds": runtime,
    }


def _parse_args(args: list[str]) -> tuple[Path, int, int, int, str, tuple[str, ...]]:
    """Parse command line arguments.

    Args:
        args: List of command line arguments (excluding script name).

    Returns:
        Tuple of (model_args_path, num_warmup, num_samples, num_chains,
        prefix, exclude_collection).

    Raises:
        ValueError: If arguments are invalid.
    """
    if not args:
        raise ValueError(
            "Usage: python -m panelcast.preflight.mini_run <model_args.json> "
            "[--num-warmup N] [--num-samples N] [--num-chains N] [--prefix NAME] "
            "[--exclude-collection site1,site2]"
        )

    model_args_path = Path(args[0])
    num_warmup = 10
    num_samples = 1
    num_chains = 1
    prefix = "user"
    exclude_collection: tuple[str, ...] = ()

    int_flags = {
        "--num-warmup": "num_warmup",
        "--num-samples": "num_samples",
        "--num-chains": "num_chains",
    }
    values: dict[str, int] = {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
    }

    # Parse optional arguments
    i = 1
    while i < len(args):
        if args[i] in int_flags:
            name = int_flags[args[i]]
            if i + 1 >= len(args):
                raise ValueError(f"{args[i]} requires an integer value")
            try:
                values[name] = int(args[i + 1])
            except ValueError:
                raise ValueError(
                    f"{args[i]} requires an integer value, got: {args[i + 1]}"
                ) from None
            i += 2
        elif args[i] == "--prefix":
            if i + 1 >= len(args):
                raise ValueError("--prefix requires a value")
            prefix = args[i + 1]
            i += 2
        elif args[i] == "--exclude-collection":
            if i + 1 >= len(args):
                raise ValueError("--exclude-collection requires a value")
            exclude_collection = tuple(
                site.strip() for site in args[i + 1].split(",") if site.strip()
            )
            i += 2
        else:
            raise ValueError(f"Unknown argument: {args[i]}")

    num_warmup = values["num_warmup"]
    num_samples = values["num_samples"]
    num_chains = values["num_chains"]

    # Validate positive values
    if num_warmup <= 0:
        raise ValueError(f"--num-warmup must be positive, got: {num_warmup}")
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got: {num_samples}")
    if num_chains <= 0:
        raise ValueError(f"--num-chains must be positive, got: {num_chains}")

    return model_args_path, num_warmup, num_samples, num_chains, prefix, exclude_collection


if __name__ == "__main__":
    # Configure logging to stderr only (stdout reserved for JSON output)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        (
            model_args_path,
            num_warmup,
            num_samples,
            num_chains,
            prefix,
            exclude_collection,
        ) = _parse_args(sys.argv[1:])
    except ValueError as e:
        result = {
            "success": False,
            "exit_code": 1,
            "error": str(e),
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }
        print(json.dumps(result))
        sys.exit(1)

    try:
        result = run_and_measure(
            model_args_path, num_warmup, num_samples, num_chains, prefix, exclude_collection
        )
        print(json.dumps(result))
    except Exception as e:
        logger.exception("Mini-run failed")
        result = {
            "success": False,
            "exit_code": 1,
            "error": str(e),
            "peak_memory_bytes": 0,
            "runtime_seconds": 0.0,
        }
        print(json.dumps(result))
        sys.exit(1)
