#!/usr/bin/env python
"""GPU verification script for WSL2 RTX 5090 setup.

This script validates the WSL2 CUDA environment for JAX/NumPyro GPU acceleration.
Run this after completing WSL2 CUDA setup to verify everything works correctly.

Usage:
    python scripts/verify_gpu.py [--help] [--jax]

Exit codes:
    0 - All checks passed
    1 - One or more checks failed

Flags:
    --jax   Only run JAX-specific checks (skip NumPyro MCMC and fit.py checks)

Checks performed (all, no flags):
    1. JAX GPU detection - verifies jax.devices() includes GPU
    2. JAX default backend - verifies 'gpu' is default
    3. Matrix multiply smoke test - 1000x1000 matrix multiplication
    4. NumPyro MCMC smoke test - simple model with 2 chains
    5. GPU info from existing infrastructure - calls get_gpu_info()

Checks performed (--jax flag):
    1. JAX GPU detection - verifies jax.devices() includes GPU
    2. JAX default backend - verifies 'gpu' is default
    3. Matrix multiply smoke test - 1000x1000 matrix multiplication
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple


class CheckResult(NamedTuple):
    """Result of a verification check."""

    name: str
    passed: bool
    message: str


def check_jax_gpu_detection() -> CheckResult:
    """Check if JAX can detect GPU devices.

    Returns:
        CheckResult with pass/fail status and device information.
    """
    try:
        import jax

        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]

        if gpu_devices:
            device_info = ", ".join(str(d) for d in gpu_devices)
            return CheckResult(
                name="JAX GPU Detection",
                passed=True,
                message=f"GPU detected: {device_info}",
            )
        else:
            device_info = ", ".join(str(d) for d in devices)
            return CheckResult(
                name="JAX GPU Detection",
                passed=False,
                message=f"No GPU detected. Available devices: {device_info}",
            )
    except ImportError as e:
        return CheckResult(
            name="JAX GPU Detection",
            passed=False,
            message=f"JAX import failed: {e}",
        )
    except Exception as e:
        return CheckResult(
            name="JAX GPU Detection",
            passed=False,
            message=f"Error checking devices: {e}",
        )


def check_jax_default_backend() -> CheckResult:
    """Check if JAX default backend is GPU.

    Returns:
        CheckResult with pass/fail status and backend information.
    """
    try:
        import jax

        backend = jax.default_backend()

        if backend == "gpu":
            return CheckResult(
                name="JAX Default Backend",
                passed=True,
                message=f"Default backend: {backend}",
            )
        else:
            return CheckResult(
                name="JAX Default Backend",
                passed=False,
                message=f"Default backend is '{backend}', expected 'gpu'",
            )
    except ImportError as e:
        return CheckResult(
            name="JAX Default Backend",
            passed=False,
            message=f"JAX import failed: {e}",
        )
    except Exception as e:
        return CheckResult(
            name="JAX Default Backend",
            passed=False,
            message=f"Error checking backend: {e}",
        )


def check_matrix_multiply() -> CheckResult:
    """Run a matrix multiplication smoke test.

    Creates two 1000x1000 matrices and multiplies them on GPU.

    Returns:
        CheckResult with pass/fail status and timing information.
    """
    try:
        import jax.numpy as jnp

        # Create matrices
        size = 1000
        x = jnp.ones((size, size))

        # Time the multiplication
        start = time.perf_counter()
        y = x @ x
        y.block_until_ready()  # Ensure GPU computation completes
        elapsed = time.perf_counter() - start

        # Verify result
        expected_shape = (size, size)
        if y.shape == expected_shape:
            return CheckResult(
                name="Matrix Multiply Test",
                passed=True,
                message=f"Result shape: {y.shape}, time: {elapsed*1000:.1f}ms",
            )
        else:
            return CheckResult(
                name="Matrix Multiply Test",
                passed=False,
                message=f"Unexpected shape: {y.shape}, expected {expected_shape}",
            )
    except ImportError as e:
        return CheckResult(
            name="Matrix Multiply Test",
            passed=False,
            message=f"JAX import failed: {e}",
        )
    except Exception as e:
        return CheckResult(
            name="Matrix Multiply Test",
            passed=False,
            message=f"Error during matrix multiply: {e}",
        )


def check_numpyro_mcmc() -> CheckResult:
    """Run a NumPyro MCMC smoke test.

    Fits a simple Normal model with 100 observations using 2 chains,
    100 warmup iterations, and 100 samples.

    Returns:
        CheckResult with pass/fail status and sample information.
    """
    try:
        import jax
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        # Simple model: mu ~ Normal(0,1), sigma ~ HalfNormal(1), obs ~ Normal(mu, sigma)
        def simple_model(y):
            mu = numpyro.sample("mu", dist.Normal(0, 1))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        # Generate fake observations
        key = jax.random.key(0)
        y = jax.random.normal(key, shape=(100,))

        # Run MCMC
        kernel = NUTS(simple_model)
        mcmc = MCMC(
            kernel,
            num_warmup=100,
            num_samples=100,
            num_chains=2,
            chain_method="vectorized",
            progress_bar=False,
        )

        start = time.perf_counter()
        mcmc.run(jax.random.key(1), y=y)
        elapsed = time.perf_counter() - start

        # Verify samples
        samples = mcmc.get_samples()
        mu_shape = samples["mu"].shape
        sigma_shape = samples["sigma"].shape

        expected_samples = 200  # 2 chains * 100 samples
        if mu_shape[0] == expected_samples and sigma_shape[0] == expected_samples:
            return CheckResult(
                name="NumPyro MCMC Test",
                passed=True,
                message=f"Samples: mu={mu_shape}, sigma={sigma_shape}, time: {elapsed:.1f}s",
            )
        else:
            return CheckResult(
                name="NumPyro MCMC Test",
                passed=False,
                message=f"Unexpected sample shapes: mu={mu_shape}, sigma={sigma_shape}",
            )
    except ImportError as e:
        return CheckResult(
            name="NumPyro MCMC Test",
            passed=False,
            message=f"Import failed: {e}",
        )
    except Exception as e:
        return CheckResult(
            name="NumPyro MCMC Test",
            passed=False,
            message=f"Error during MCMC: {e}",
        )


def check_get_gpu_info() -> CheckResult:
    """Call existing get_gpu_info() from fit.py.

    Returns:
        CheckResult with GPU information from the existing infrastructure.
    """
    try:
        from panelcast.models.bayes.fit import get_gpu_info

        gpu_info = get_gpu_info()

        if gpu_info and gpu_info != "CPU only":
            return CheckResult(
                name="GPU Info (fit.py)",
                passed=True,
                message=f"GPU: {gpu_info}",
            )
        else:
            return CheckResult(
                name="GPU Info (fit.py)",
                passed=False,
                message=f"No GPU info: {gpu_info}",
            )
    except ImportError as e:
        return CheckResult(
            name="GPU Info (fit.py)",
            passed=False,
            message=f"Import failed: {e}",
        )
    except Exception as e:
        return CheckResult(
            name="GPU Info (fit.py)",
            passed=False,
            message=f"Error getting GPU info: {e}",
        )


def run_verification(jax_only: bool = False) -> list[CheckResult]:
    """Run verification checks.

    Args:
        jax_only: If True, only run JAX-specific checks (skip NumPyro MCMC and fit.py).

    Returns:
        List of CheckResult objects for all checks.
    """
    results = []
    check_num = 0

    # 1. JAX GPU detection
    check_num += 1
    print("=" * 60)
    print(f"{check_num}. Checking JAX GPU Detection")
    print("=" * 60)
    result = check_jax_gpu_detection()
    results.append(result)
    print(f"   {result.message}")
    print()

    # 2. JAX default backend
    check_num += 1
    print("=" * 60)
    print(f"{check_num}. Checking JAX Default Backend")
    print("=" * 60)
    result = check_jax_default_backend()
    results.append(result)
    print(f"   {result.message}")
    print()

    # 3. Matrix multiply test
    check_num += 1
    print("=" * 60)
    print(f"{check_num}. Running Matrix Multiply Test")
    print("=" * 60)
    result = check_matrix_multiply()
    results.append(result)
    print(f"   {result.message}")
    print()

    if jax_only:
        print("(--jax flag: skipping NumPyro MCMC and fit.py checks)")
        print()
        return results

    # 4. NumPyro MCMC test
    check_num += 1
    print("=" * 60)
    print(f"{check_num}. Running NumPyro MCMC Test")
    print("=" * 60)
    result = check_numpyro_mcmc()
    results.append(result)
    print(f"   {result.message}")
    print()

    # 5. GPU info from fit.py
    check_num += 1
    print("=" * 60)
    print(f"{check_num}. Checking GPU Info (fit.py)")
    print("=" * 60)
    result = check_get_gpu_info()
    results.append(result)
    print(f"   {result.message}")
    print()

    return results


def print_summary(results: list[CheckResult]) -> bool:
    """Print a summary of all check results.

    Args:
        results: List of CheckResult objects.

    Returns:
        True if all checks passed, False otherwise.
    """
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        symbol = "[+]" if result.passed else "[-]"
        print(f"  {symbol} {result.name}: {status}")
        if not result.passed:
            all_passed = False

    print()
    if all_passed:
        print("SUCCESS: All GPU verification checks passed!")
    else:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"FAILED: {passed}/{total} checks passed")
        print()
        print("Troubleshooting tips:")
        print("  - Verify NVIDIA driver 570+ is installed on Windows")
        print("  - Run 'nvidia-smi' in WSL2 to check GPU passthrough")
        print("  - Ensure cuda-toolkit-12-x is installed (NOT cuda-drivers)")
        print("  - Check JAX version: pip show jax (need >= 0.4.38)")
        print("  - See docs/GPU_SETUP.md for full setup instructions")

    return all_passed


def main() -> int:
    """Main entry point for GPU verification.

    Returns:
        Exit code: 0 if all checks passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Verify WSL2 CUDA setup for JAX/NumPyro GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/verify_gpu.py          Run all verification checks
    python scripts/verify_gpu.py --jax    Run JAX-only checks (faster)
    python scripts/verify_gpu.py --help   Show this help message

Exit codes:
    0  All checks passed - GPU is working correctly
    1  One or more checks failed - see output for troubleshooting
        """,
    )
    parser.add_argument(
        "--jax",
        action="store_true",
        help="Only check JAX GPU (skip NumPyro MCMC and fit.py checks)",
    )
    args = parser.parse_args()

    print()
    print("GPU Verification Script for WSL2 RTX 5090")
    print("=========================================")
    if args.jax:
        print("Mode: JAX-only checks")
    else:
        print("Mode: Full verification")
    print()

    results = run_verification(jax_only=args.jax)
    all_passed = print_summary(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
