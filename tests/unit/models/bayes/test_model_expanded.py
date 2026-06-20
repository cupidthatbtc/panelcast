"""Expanded tests for models/bayes/model.py: compute_sigma_scaled, make_score_model."""

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.model import (
    compute_sigma_scaled,
    critic_score_model,
    make_score_model,
    user_score_model,
)
from panelcast.models.bayes.priors import PriorConfig


class TestComputeSigmaScaledExpanded:
    """Expanded edge-case and boundary tests for compute_sigma_scaled."""

    def test_homoscedastic_mode(self):
        """exponent=0 should return sigma_obs for all n_reviews."""
        result = compute_sigma_scaled(1.5, jnp.array([1.0, 10.0, 1000.0]), 0.0)
        np.testing.assert_allclose(result, 1.5, atol=0.01)

    def test_sqrt_scaling(self):
        """exponent=0.5 should give 1/sqrt(n) scaling."""
        result = compute_sigma_scaled(1.0, jnp.array([100.0]), 0.5)
        np.testing.assert_allclose(result, 0.1, atol=0.01)

    def test_single_review_penalty(self):
        """n_reviews=1 with exponent>0 should apply 2x penalty."""
        result = compute_sigma_scaled(1.0, jnp.array([1.0]), 0.5)
        assert float(result[0]) == pytest.approx(2.0)

    def test_custom_single_review_multiplier(self):
        result = compute_sigma_scaled(1.0, jnp.array([1.0]), 0.5, single_review_multiplier=3.0)
        assert float(result[0]) == pytest.approx(3.0)

    def test_min_sigma_floor(self):
        """Very large n should be floored at min_sigma."""
        result = compute_sigma_scaled(1.0, jnp.array([1e10]), 1.0, min_sigma=0.01)
        assert float(result[0]) == pytest.approx(0.01, abs=1e-6)

    def test_custom_min_sigma(self):
        result = compute_sigma_scaled(1.0, jnp.array([1e10]), 1.0, min_sigma=0.1)
        assert float(result[0]) == pytest.approx(0.1, abs=1e-5)

    def test_large_sigma_obs(self):
        result = compute_sigma_scaled(100.0, jnp.array([1000.0]), 0.5)
        expected = 100.0 / np.sqrt(1000.0)
        assert float(result[0]) == pytest.approx(expected, rel=0.01)

    def test_scalar_n_reviews(self):
        """Should work with scalar n_reviews."""
        result = compute_sigma_scaled(1.0, jnp.array([50.0]), 0.5)
        assert result.shape == (1,)

    def test_many_observations(self):
        n = jnp.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        result = compute_sigma_scaled(1.0, n, 0.5)
        assert result.shape == (5,)
        # Should be monotonically decreasing (more reviews = less noise)
        for i in range(len(result) - 1):
            assert float(result[i]) > float(result[i + 1])

    def test_fractional_exponent(self):
        result = compute_sigma_scaled(1.0, jnp.array([100.0]), 0.33)
        expected = 1.0 / (100.0**0.33)
        assert float(result[0]) == pytest.approx(expected, rel=0.05)


class TestMakeScoreModelExpanded:
    """Expanded tests for make_score_model factory."""

    def test_user_model_callable(self):
        model = make_score_model("user")
        assert callable(model)

    def test_critic_model_callable(self):
        model = make_score_model("critic")
        assert callable(model)

    def test_any_identifier_prefix_accepted(self):
        # Portability contract: the descriptor supplies the domain prefix,
        # so any identifier is a valid posterior-site prefix.
        assert callable(make_score_model("perf"))

    def test_non_identifier_prefix_raises(self):
        with pytest.raises(ValueError, match="identifier"):
            make_score_model("not an identifier")

    def test_non_string_prefix_raises(self):
        with pytest.raises(ValueError, match="identifier"):
            make_score_model(123)

    def test_user_score_model_is_callable(self):
        assert callable(user_score_model)

    def test_critic_score_model_is_callable(self):
        assert callable(critic_score_model)

    def test_make_score_model_returns_callable(self):
        """make_score_model should return a callable model."""
        model = make_score_model("user")
        assert callable(model)

    def test_make_score_model_critic(self):
        model = make_score_model("critic")
        assert callable(model)
