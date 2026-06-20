#!/usr/bin/env python3
"""GPU MCMC benchmarking script with ESS per second calculation.

This script benchmarks NumPyro MCMC performance on GPU using synthetic data.
It measures wall-clock time, effective sample size (ESS), and computes the
ESS/second efficiency metric.

Usage:
    python scripts/benchmark_gpu.py --config quick
    python scripts/benchmark_gpu.py --config standard --output reports/gpu_benchmark_results.json
    python scripts/benchmark_gpu.py --config publication --runs 3

Benchmark configurations:
    - quick: 200 warmup, 200 samples, 2 chains (for testing)
    - standard: 500 warmup, 500 samples, 4 chains (production baseline)
    - publication: 1000 warmup, 1000 samples, 4 chains (full publication quality)

Output JSON includes:
    - gpu_info: GPU device details from nvidia-smi
    - jax_backend: JAX default backend (should be "gpu")
    - config_name: Which benchmark config was used
    - runtime_seconds: Wall-clock time for MCMC
    - min_ess_bulk: Minimum bulk ESS across all parameters
    - min_ess_tail: Minimum tail ESS across all parameters
    - ess_per_second: Efficiency metric (min_ess_bulk / runtime)
    - divergences: Number of divergent transitions
    - timestamp: ISO format timestamp
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp

from panelcast.models.bayes.fit import MCMCConfig, fit_model, get_gpu_info
from panelcast.models.bayes.model import make_score_model

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    num_warmup: int
    num_samples: int
    num_chains: int
    description: str


# Predefined benchmark configurations
BENCHMARK_CONFIGS = {
    "quick": BenchmarkConfig(
        name="quick",
        num_warmup=200,
        num_samples=200,
        num_chains=2,
        description="Quick sanity check (200 warmup, 200 samples, 2 chains)",
    ),
    "standard": BenchmarkConfig(
        name="standard",
        num_warmup=500,
        num_samples=500,
        num_chains=4,
        description="Standard production benchmark (500 warmup, 500 samples, 4 chains)",
    ),
    "publication": BenchmarkConfig(
        name="publication",
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        description="Full publication quality (1000 warmup, 1000 samples, 4 chains)",
    ),
}


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_synthetic_data(
    n_obs: int = 1000,
    n_artists: int = 100,
    n_features: int = 10,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic data for benchmarking.

    Creates data that mimics the structure expected by the score model:
    - artist_idx: Random integers [0, n_artists)
    - album_seq: Random integers [1, 5] representing album sequence
    - prev_score: Random normal scores (mean 70, std 15)
    - X: Random normal feature matrix
    - y: Random normal target scores (mean 70, std 15)

    Args:
        n_obs: Number of observations
        n_artists: Number of unique artists
        n_features: Number of features in X
        seed: Random seed for reproducibility

    Returns:
        Dictionary with all model inputs
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)

    # Artist indices: random assignment
    artist_idx = jax.random.randint(keys[0], shape=(n_obs,), minval=0, maxval=n_artists)

    # Album sequence: random integers 1-5
    album_seq = jax.random.randint(keys[1], shape=(n_obs,), minval=1, maxval=6)

    # Previous score: random normal (mean 70, std 15), 0 for debuts
    # For simplicity, we'll just generate random scores
    prev_score = jax.random.normal(keys[2], shape=(n_obs,)) * 15 + 70

    # Feature matrix: random normal
    X = jax.random.normal(keys[3], shape=(n_obs, n_features))

    # Target scores: random normal (mean 70, std 15)
    y = jax.random.normal(keys[4], shape=(n_obs,)) * 15 + 70

    # Convert to appropriate types
    return {
        "artist_idx": artist_idx.astype(jnp.int32),
        "album_seq": album_seq.astype(jnp.int32),
        "prev_score": prev_score.astype(jnp.float32),
        "X": X.astype(jnp.float32),
        "y": y.astype(jnp.float32),
        "n_artists": n_artists,
        "max_seq": 5,  # Fixed for synthetic data
    }


# =============================================================================
# Benchmarking
# =============================================================================


def run_single_benchmark(
    model_args: dict,
    config: BenchmarkConfig,
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Run a single benchmark and return results.

    Args:
        model_args: Dictionary of model arguments (synthetic data)
        config: Benchmark configuration
        progress_bar: Whether to show progress bar

    Returns:
        Dictionary with benchmark results
    """
    # Create model
    model = make_score_model("user")

    # Create MCMC config
    mcmc_config = MCMCConfig(
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method="vectorized",
        seed=42,
    )

    # Run MCMC with timing
    start_time = time.perf_counter()
    result = fit_model(model, model_args, config=mcmc_config, progress_bar=progress_bar)
    runtime = time.perf_counter() - start_time

    # Calculate ESS
    ess_bulk = az.ess(result.idata, method="bulk")
    ess_tail = az.ess(result.idata, method="tail")

    # Get minimum ESS across all parameters
    min_ess_bulk = float(ess_bulk.to_array().min())
    min_ess_tail = float(ess_tail.to_array().min())

    # ESS per second (key efficiency metric)
    ess_per_second = min_ess_bulk / runtime

    return {
        "runtime_seconds": runtime,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "ess_per_second": ess_per_second,
        "divergences": result.divergences,
    }


def run_benchmark(
    config_name: str = "quick",
    num_runs: int = 1,
    n_obs: int = 1000,
    n_artists: int = 100,
    n_features: int = 10,
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Run benchmark with specified configuration.

    Args:
        config_name: Name of benchmark config (quick/standard/publication)
        num_runs: Number of runs to average
        n_obs: Number of observations in synthetic data
        n_artists: Number of artists in synthetic data
        n_features: Number of features in synthetic data
        progress_bar: Whether to show progress bar

    Returns:
        Dictionary with aggregated benchmark results
    """
    if config_name not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(BENCHMARK_CONFIGS.keys())}"
        )

    config = BENCHMARK_CONFIGS[config_name]

    print("=" * 70)
    print("GPU MCMC Benchmark")
    print("=" * 70)

    # Get GPU info upfront
    gpu_info = get_gpu_info()
    jax_backend = jax.default_backend()

    print(f"GPU info: {gpu_info}")
    print(f"JAX backend: {jax_backend}")
    print(f"Config: {config.description}")
    print(f"Synthetic data: {n_obs} obs, {n_artists} artists, {n_features} features")
    print(f"Runs: {num_runs}")
    print("-" * 70)

    # Generate synthetic data (same for all runs)
    model_args = generate_synthetic_data(
        n_obs=n_obs,
        n_artists=n_artists,
        n_features=n_features,
        seed=42,
    )

    # Run benchmarks
    results = []
    for i in range(num_runs):
        if num_runs > 1:
            print(f"\nRun {i+1}/{num_runs}...")

        run_result = run_single_benchmark(model_args, config, progress_bar=progress_bar)
        results.append(run_result)

        print(f"  Runtime: {run_result['runtime_seconds']:.2f}s")
        print(f"  ESS/second: {run_result['ess_per_second']:.1f}")
        print(f"  Divergences: {run_result['divergences']}")

    # Aggregate results
    if num_runs == 1:
        # Single run - use directly
        aggregated = {
            "runtime_seconds": results[0]["runtime_seconds"],
            "min_ess_bulk": results[0]["min_ess_bulk"],
            "min_ess_tail": results[0]["min_ess_tail"],
            "ess_per_second": results[0]["ess_per_second"],
            "divergences": results[0]["divergences"],
        }
    else:
        # Multiple runs - compute averages
        aggregated = {
            "runtime_seconds": sum(r["runtime_seconds"] for r in results) / num_runs,
            "min_ess_bulk": sum(r["min_ess_bulk"] for r in results) / num_runs,
            "min_ess_tail": sum(r["min_ess_tail"] for r in results) / num_runs,
            "ess_per_second": sum(r["ess_per_second"] for r in results) / num_runs,
            "divergences": sum(r["divergences"] for r in results),  # Total divergences
        }

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"GPU: {gpu_info}")
    print(f"JAX backend: {jax_backend}")
    print(f"Config: {config_name}")
    print(f"Average runtime: {aggregated['runtime_seconds']:.2f}s")
    print(f"Average ESS/second: {aggregated['ess_per_second']:.1f}")
    print(f"Total divergences: {aggregated['divergences']}")

    # Build final results dictionary
    final_results = {
        "gpu_info": gpu_info,
        "jax_backend": jax_backend,
        "config_name": config_name,
        "config_description": config.description,
        "synthetic_data": {
            "n_obs": n_obs,
            "n_artists": n_artists,
            "n_features": n_features,
        },
        "num_runs": num_runs,
        "runtime_seconds": round(aggregated["runtime_seconds"], 3),
        "min_ess_bulk": round(aggregated["min_ess_bulk"], 1),
        "min_ess_tail": round(aggregated["min_ess_tail"], 1),
        "ess_per_second": round(aggregated["ess_per_second"], 2),
        "divergences": aggregated["divergences"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add per-run details if multiple runs
    if num_runs > 1:
        final_results["runs"] = [
            {
                "run": i + 1,
                "runtime_seconds": round(r["runtime_seconds"], 3),
                "min_ess_bulk": round(r["min_ess_bulk"], 1),
                "min_ess_tail": round(r["min_ess_tail"], 1),
                "ess_per_second": round(r["ess_per_second"], 2),
                "divergences": r["divergences"],
            }
            for i, r in enumerate(results)
        ]

    return final_results


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save benchmark results to JSON file.

    Args:
        results: Benchmark results dictionary
        output_path: Path to output JSON file
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="GPU MCMC benchmarking with ESS per second calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick sanity check
    python scripts/benchmark_gpu.py --config quick

    # Standard benchmark, save results
    python scripts/benchmark_gpu.py --config standard --output reports/gpu_benchmark_results.json

    # Publication quality with 3 runs for averaging
    python scripts/benchmark_gpu.py --config publication --runs 3 \
        --output reports/gpu_benchmark_pub.json

    # Custom synthetic data size
    python scripts/benchmark_gpu.py --config quick --n-obs 5000 --n-artists 200

Benchmark configs:
    quick:       200 warmup, 200 samples, 2 chains (fast testing)
    standard:    500 warmup, 500 samples, 4 chains (production baseline)
    publication: 1000 warmup, 1000 samples, 4 chains (full quality)
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=list(BENCHMARK_CONFIGS.keys()),
        default="quick",
        help="Benchmark configuration (default: quick)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/gpu_benchmark_results.json",
        help="Output JSON path (default: reports/gpu_benchmark_results.json)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for averaging (default: 1)",
    )

    parser.add_argument(
        "--n-obs",
        type=int,
        default=1000,
        help="Number of observations in synthetic data (default: 1000)",
    )

    parser.add_argument(
        "--n-artists",
        type=int,
        default=100,
        help="Number of artists in synthetic data (default: 100)",
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=10,
        help="Number of features in synthetic data (default: 10)",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    try:
        # Run benchmark
        results = run_benchmark(
            config_name=args.config,
            num_runs=args.runs,
            n_obs=args.n_obs,
            n_artists=args.n_artists,
            n_features=args.n_features,
            progress_bar=not args.no_progress,
        )

        # Save results
        output_path = Path(args.output)
        save_results(results, output_path)

        # Exit code based on GPU detection
        if results["jax_backend"] != "gpu":
            print("\nWARNING: Running on CPU, not GPU!")
            return 1

        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
