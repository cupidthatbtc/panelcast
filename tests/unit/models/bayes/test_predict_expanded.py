"""Expanded tests for models/bayes/predict.py: PredictionResult dataclass."""

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.predict import PredictionResult


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_y_accessible(self):
        y = jnp.ones((10, 5))
        result = PredictionResult(y=y)
        assert result.y.shape == (10, 5)

    def test_mu_default_none(self):
        y = jnp.ones((10, 5))
        result = PredictionResult(y=y)
        assert result.mu is None

    def test_mu_provided(self):
        y = jnp.ones((10, 5))
        mu = jnp.zeros((10, 5))
        result = PredictionResult(y=y, mu=mu)
        assert result.mu is not None
        assert result.mu.shape == (10, 5)

    def test_y_is_jax_array(self):
        y = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = PredictionResult(y=y)
        assert hasattr(result.y, "device")  # JAX arrays have device attribute

    def test_mutable(self):
        """PredictionResult is not frozen (contains mutable JAX arrays)."""
        y = jnp.ones((10, 5))
        result = PredictionResult(y=y)
        result.mu = jnp.zeros((10, 5))
        assert result.mu is not None

    def test_numpy_y(self):
        """Can also hold numpy arrays (for flexibility)."""
        y = np.ones((10, 5))
        result = PredictionResult(y=y)
        assert result.y.shape == (10, 5)

    def test_different_shapes(self):
        y = jnp.ones((100, 50))
        result = PredictionResult(y=y)
        assert result.y.shape[0] == 100
        assert result.y.shape[1] == 50

    def test_single_sample(self):
        y = jnp.ones((1, 10))
        result = PredictionResult(y=y)
        assert result.y.shape == (1, 10)
