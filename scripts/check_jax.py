#!/usr/bin/env python3
"""Check JAX installation and GPU availability."""

import sys


def check_jax():
    try:
        import jax

        print(f"JAX version: {jax.__version__}")
        print(f"JAX default backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")

        import jaxlib

        print(f"jaxlib version: {jaxlib.__version__}")

        # Check for CUDA
        has_gpu = any(d.platform == "gpu" for d in jax.devices())
        if has_gpu:
            print("SUCCESS: GPU is available!")
        else:
            print("WARNING: No GPU detected, running on CPU")

        return 0 if has_gpu else 1

    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:  # Broad catch: JAX init can fail many ways (device, driver, etc.)
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(check_jax())
