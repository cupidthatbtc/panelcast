"""Random seed management for reproducibility.

This module provides utilities for setting random seeds consistently across
numpy and Python's random module, ensuring reproducible results for
non-MCMC operations like data splitting and feature preprocessing.

Note: JAX PRNG is handled separately via jax.random.key() in fit.py.
This module provides a helper function for JAX key generation but does
NOT set any global JAX random state.
"""

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


def set_seeds(seed: int) -> None:
    """Set random seeds for Python and NumPy random number generators.

    This ensures reproducibility for operations that use random.random()
    or np.random functions, such as data shuffling, train/test splits,
    and feature sampling.

    Note: JAX uses explicit PRNG keys and does not have global state.
    Use get_rng_key(seed) to create a JAX PRNGKey for MCMC operations.

    Args:
        seed: Integer seed value. Should be non-negative.

    Example:
        >>> set_seeds(42)
        >>> import numpy as np
        >>> np.random.rand()  # Will be same each time with seed 42
        0.3745401188473625
    """
    # Set Python's built-in random module seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)


def get_rng_key(seed: int) -> "jax.Array":
    """Create a JAX PRNGKey from an integer seed.

    Uses the modern jax.random.key() API (not the deprecated PRNGKey).
    This provides a consistent entry point for JAX random state
    used in MCMC sampling.

    Args:
        seed: Integer seed value for the PRNG key.

    Returns:
        jax.Array: A JAX PRNGKey that can be used with jax.random functions.

    Example:
        >>> key = get_rng_key(42)
        >>> # Use key for MCMC: fit_model(..., rng_key=key)

    Note:
        Import is deferred to avoid loading JAX when not needed.
    """
    import jax.random

    return jax.random.key(seed)
