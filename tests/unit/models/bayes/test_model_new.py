"""New coverage tests for models/bayes/model.py.

Targets uncovered code paths (97% -> higher):
- make_score_model: invalid score_type raises ValueError
- compute_sigma_scaled: edge cases for homoscedastic mode (exponent=0),
  large n_reviews, min_sigma floor
"""

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.model import (
    compute_sigma_scaled,
    make_score_model,
)


class TestMakeScoreModelValidation:
    """Cover invalid score_type validation."""

    def test_non_identifier_score_type_raises(self):
        """score_type must be usable as a posterior-site prefix."""
        with pytest.raises(ValueError, match="score_type must be"):
            make_score_model("not-an-identifier!")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="score_type must be"):
            make_score_model("")

    def test_custom_prefix_valid(self):
        """Descriptor-driven domains supply their own prefix."""
        model = make_score_model("perf")
        assert callable(model)

    def test_user_type_valid(self):
        model = make_score_model("user")
        assert callable(model)

    def test_critic_type_valid(self):
        model = make_score_model("critic")
        assert callable(model)


class TestComputeSigmaScaledHomoscedastic:
    """Cover exponent=0 (homoscedastic) path."""

    def test_exponent_zero_returns_sigma_obs(self):
        """With exponent=0, sigma_scaled should equal sigma_obs."""
        sigma_obs = 5.0
        n_reviews = jnp.array([1.0, 10.0, 100.0, 1000.0])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=0.0)
        # exponent=0 means sigma_obs / n^0 = sigma_obs
        np.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_single_review_no_penalty_homoscedastic(self):
        """exponent=0 means no single-review penalty."""
        sigma_obs = 5.0
        n_reviews = jnp.array([1.0])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=0.0)
        # No penalty when exponent=0 (apply_penalty requires exponent > 0)
        np.testing.assert_allclose(result, 5.0, atol=1e-5)


class TestComputeSigmaScaledMinFloor:
    """Cover min_sigma floor."""

    def test_very_large_n_reviews_hits_min_floor(self):
        """Very large n_reviews with high exponent should hit min_sigma floor."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1e10])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=1.0)
        # sigma = 1.0 / 1e10 = 1e-10 < 0.01 -> floored to 0.01
        assert float(result[0]) == pytest.approx(0.01)

    def test_custom_min_sigma(self):
        """Custom min_sigma floor should be respected."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1e10])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=1.0, min_sigma=0.1)
        assert float(result[0]) == pytest.approx(0.1)


class TestComputeSigmaScaledSingleReview:
    """Cover single-review penalty application."""

    def test_single_review_with_nonzero_exponent(self):
        """n_reviews=1 with exponent>0 should apply single-review penalty."""
        sigma_obs = 5.0
        n_reviews = jnp.array([1.0])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=0.5)
        # Should apply multiplier: sigma_obs * 2.0 = 10.0
        assert float(result[0]) == pytest.approx(10.0)

    def test_single_review_custom_multiplier(self):
        """Custom single_review_multiplier should be used."""
        sigma_obs = 5.0
        n_reviews = jnp.array([1.0])
        result = compute_sigma_scaled(
            sigma_obs, n_reviews, exponent=0.5, single_review_multiplier=3.0
        )
        assert float(result[0]) == pytest.approx(15.0)


class TestComputeSigmaScaledNReviewsClamping:
    """Cover n_reviews clamping to minimum of 1.0."""

    def test_zero_n_reviews_clamped(self):
        """n_reviews=0 should be clamped to 1.0."""
        sigma_obs = 5.0
        n_reviews = jnp.array([0.0])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=0.5)
        # Clamped to 1.0, then single-review penalty applies (exponent > 0)
        assert float(result[0]) == pytest.approx(10.0)

    def test_negative_n_reviews_clamped(self):
        """Negative n_reviews should be clamped to 1.0."""
        sigma_obs = 5.0
        n_reviews = jnp.array([-5.0])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=0.5)
        assert float(result[0]) == pytest.approx(10.0)


class TestComputeSigmaScaledBroadcasting:
    """Cover array broadcasting behavior."""

    def test_multiple_reviews_array(self):
        """Multiple n_reviews values should produce array output."""
        sigma_obs = 10.0
        n_reviews = jnp.array([4.0, 100.0, 10000.0])
        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent=0.5)
        assert result.shape == (3,)
        # sigma = 10 / sqrt(4) = 5.0, 10 / sqrt(100) = 1.0, 10 / sqrt(10000) = 0.1
        np.testing.assert_allclose(result, [5.0, 1.0, 0.1], atol=1e-3)
