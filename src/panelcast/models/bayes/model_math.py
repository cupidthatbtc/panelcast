"""Pure numeric helpers shared by the score model and target transforms.

Lives below both model.py and transforms.py in the import graph so the
transform registry can use soft_clip without a circular import.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Sharpness for softplus-based boundary clipping.  Higher values make the
# transition sharper (closer to hard clip).  At sharpness=5, the function
# is effectively identity for x in [1, 99] and smoothly saturates outside.
_CLIP_SHARPNESS = 5.0


def soft_clip(
    x: jnp.ndarray,
    low: float = 0.0,
    high: float = 100.0,
    sharpness: float = _CLIP_SHARPNESS,
) -> jnp.ndarray:
    """Differentiable soft clipping of predictions to (low, high).

    Uses a double-softplus that acts as identity in the interior and
    smoothly saturates only near the bounds.  Unlike tanh-based clipping
    this preserves the scale of values well within [low, high].

    Args:
        x: Input array (unbounded).
        low: Lower bound (default 0).
        high: Upper bound (default 100).
        sharpness: Controls transition steepness near bounds.
            Higher => sharper (closer to hard clip).  Default 5.0.

    Returns:
        Array of same shape, values in (low, high).
    """
    # softplus(s*z)/s ≈ z for z >> 0, ≈ 0 for z << 0
    x = low + jax.nn.softplus(sharpness * (x - low)) / sharpness
    x = high - jax.nn.softplus(sharpness * (high - x)) / sharpness
    return x
