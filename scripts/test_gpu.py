#!/usr/bin/env python3
"""Quick GPU test."""

import jax

print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
