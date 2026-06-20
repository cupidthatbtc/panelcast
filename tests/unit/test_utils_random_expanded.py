"""Expanded tests for random seed utilities."""

import jax.random
import numpy as np
import pytest

from panelcast.utils.random import get_rng_key, set_seeds


class TestSetSeedsExpanded:
    """Extended tests for set_seeds."""

    def test_deterministic_seed(self):
        """Should produce deterministic results."""
        set_seeds(seed=42)
        val1 = np.random.rand()
        set_seeds(seed=42)
        val2 = np.random.rand()
        assert val1 == val2

    def test_custom_seed(self):
        set_seeds(seed=12345)
        val1 = np.random.rand()
        set_seeds(seed=12345)
        val2 = np.random.rand()
        assert val1 == val2

    def test_different_seeds_different_results(self):
        set_seeds(seed=1)
        val1 = np.random.rand()
        set_seeds(seed=2)
        val2 = np.random.rand()
        assert val1 != val2

    def test_seed_zero(self):
        set_seeds(seed=0)
        val = np.random.rand()
        assert 0.0 <= val <= 1.0

    def test_large_seed(self):
        set_seeds(seed=2**31 - 1)
        val = np.random.rand()
        assert 0.0 <= val <= 1.0


class TestGetRngKeyExpanded:
    """Extended tests for get_rng_key."""

    def test_returns_jax_array(self):
        key = get_rng_key(42)
        assert hasattr(key, "shape")

    def test_deterministic(self):
        key1 = get_rng_key(42)
        key2 = get_rng_key(42)
        np.testing.assert_array_equal(jax.random.key_data(key1), jax.random.key_data(key2))

    def test_different_seeds_different_keys(self):
        key1 = get_rng_key(1)
        key2 = get_rng_key(2)
        assert not np.array_equal(jax.random.key_data(key1), jax.random.key_data(key2))

    def test_seed_zero(self):
        key = get_rng_key(0)
        assert hasattr(key, "shape")
