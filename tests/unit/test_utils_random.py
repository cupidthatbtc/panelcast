"""Tests for random seed management utilities."""

import random

import numpy as np

from panelcast.utils.random import get_rng_key, set_seeds


class TestSetSeeds:
    """Tests for set_seeds function."""

    def test_set_seeds_numpy_reproducible(self):
        """set_seeds produces reproducible numpy random numbers."""
        set_seeds(42)
        result1 = np.random.rand(5).tolist()

        set_seeds(42)
        result2 = np.random.rand(5).tolist()

        assert result1 == result2

    def test_set_seeds_python_reproducible(self):
        """set_seeds produces reproducible python random numbers."""
        set_seeds(42)
        result1 = [random.random() for _ in range(5)]

        set_seeds(42)
        result2 = [random.random() for _ in range(5)]

        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds produce different random sequences."""
        set_seeds(42)
        result1 = np.random.rand(5).tolist()

        set_seeds(123)
        result2 = np.random.rand(5).tolist()

        assert result1 != result2

    def test_set_seeds_returns_none(self):
        """set_seeds returns None."""
        result = set_seeds(42)
        assert result is None

    def test_set_seeds_accepts_zero(self):
        """set_seeds accepts zero as valid seed."""
        set_seeds(0)
        result1 = np.random.rand()

        set_seeds(0)
        result2 = np.random.rand()

        assert result1 == result2

    def test_set_seeds_accepts_large_seed(self):
        """set_seeds accepts large integer seed."""
        set_seeds(2**31 - 1)
        result1 = np.random.rand()

        set_seeds(2**31 - 1)
        result2 = np.random.rand()

        assert result1 == result2

    def test_numpy_rand_determinism(self):
        """Verify numpy random sequences are deterministic."""
        set_seeds(42)
        seq1 = np.random.randint(0, 1000, size=10).tolist()

        set_seeds(42)
        seq2 = np.random.randint(0, 1000, size=10).tolist()

        assert seq1 == seq2

    def test_python_random_determinism(self):
        """Verify python random sequences are deterministic."""
        set_seeds(42)
        seq1 = [random.randint(0, 1000) for _ in range(10)]

        set_seeds(42)
        seq2 = [random.randint(0, 1000) for _ in range(10)]

        assert seq1 == seq2

    def test_numpy_choice_determinism(self):
        """Verify numpy choice is deterministic after set_seeds."""
        set_seeds(42)
        choices1 = np.random.choice(100, size=10, replace=False).tolist()

        set_seeds(42)
        choices2 = np.random.choice(100, size=10, replace=False).tolist()

        assert choices1 == choices2

    def test_python_shuffle_determinism(self):
        """Verify python shuffle is deterministic after set_seeds."""
        set_seeds(42)
        list1 = list(range(10))
        random.shuffle(list1)

        set_seeds(42)
        list2 = list(range(10))
        random.shuffle(list2)

        assert list1 == list2


class TestGetRngKey:
    """Tests for get_rng_key function."""

    def test_get_rng_key_returns_valid_key(self):
        """get_rng_key returns a valid JAX key."""
        key = get_rng_key(42)

        # Should be a JAX array
        import jax

        assert isinstance(key, jax.Array)

    def test_get_rng_key_same_seed_same_key(self):
        """Same seed produces equivalent JAX keys."""
        key1 = get_rng_key(42)
        key2 = get_rng_key(42)

        import jax.numpy as jnp

        # Keys should be equal
        assert jnp.array_equal(key1, key2)

    def test_get_rng_key_different_seeds(self):
        """Different seeds produce different JAX keys."""
        key1 = get_rng_key(42)
        key2 = get_rng_key(123)

        import jax.numpy as jnp

        # Keys should be different
        assert not jnp.array_equal(key1, key2)

    def test_get_rng_key_accepts_zero(self):
        """get_rng_key accepts zero as valid seed."""
        key = get_rng_key(0)

        import jax

        assert isinstance(key, jax.Array)

    def test_get_rng_key_usable_for_random(self):
        """get_rng_key produces key usable with jax.random functions."""
        key = get_rng_key(42)

        import jax.random

        # Should be able to use key for random operations
        sample = jax.random.normal(key)

        # Result should be a scalar
        assert sample.shape == ()

    def test_get_rng_key_deterministic_samples(self):
        """Same key produces same random samples."""
        import jax.random

        key1 = get_rng_key(42)
        sample1 = jax.random.normal(key1)

        key2 = get_rng_key(42)
        sample2 = jax.random.normal(key2)

        assert float(sample1) == float(sample2)

    def test_get_rng_key_can_be_split(self):
        """get_rng_key produces key that can be split."""
        import jax.random

        key = get_rng_key(42)
        key1, key2 = jax.random.split(key)

        # Both should be valid keys
        sample1 = jax.random.normal(key1)
        sample2 = jax.random.normal(key2)

        # Should produce different values
        assert float(sample1) != float(sample2)
