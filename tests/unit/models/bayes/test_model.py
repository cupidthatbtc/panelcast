"""Unit tests for model.py model structure and validation.

Tests cover:
- compute_sigma_scaled: edge cases (negative/zero n_reviews, JAX arrays)
- make_score_model factory: validation and returned model structure
- Model parameter validation: required args, invalid priors

These tests focus on structural/validation tests, NOT MCMC execution
(actual MCMC tests are in test_heteroscedastic.py and test_model_fit.py).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS

from panelcast.models.bayes.model import (
    compute_sigma_scaled,
    critic_score_model,
    make_score_model,
    user_score_model,
)
from panelcast.models.bayes.priors import PriorConfig


class TestComputeSigmaScaledEdgeCases:
    """Additional edge case tests for compute_sigma_scaled.

    Core functionality tests are in test_heteroscedastic.py.
    These tests focus on edge cases and JAX compatibility.
    """

    def test_negative_n_reviews_clamped(self):
        """Negative n_reviews should be clamped to 1.0."""
        sigma_obs = 1.0
        n_reviews = jnp.array([-5.0, -100.0, -1.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # All negative values should be treated as n=1 (single review penalty)
        # n=1 with exponent > 0 -> sigma_obs * single_review_multiplier = 2.0
        assert np.allclose(result, 2.0)

    def test_zero_n_reviews_clamped(self):
        """Zero n_reviews should be clamped to 1.0."""
        sigma_obs = 1.0
        n_reviews = jnp.array([0.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # Zero treated as n=1 -> single review penalty = 2.0
        assert np.isclose(result[0], 2.0)

    def test_jax_array_types(self):
        """Should work correctly with JAX array types."""
        sigma_obs = jnp.array(1.0)
        n_reviews = jnp.array([10.0, 100.0, 1000.0])
        exponent = jnp.array(0.5)

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # Should return JAX array
        assert hasattr(result, "device")  # JAX arrays have device attribute
        expected = jnp.array([1 / np.sqrt(10), 1 / np.sqrt(100), 1 / np.sqrt(1000)])
        # Apply min_sigma floor
        expected = jnp.maximum(expected, 0.01)
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-5)

    def test_very_small_n_reviews(self):
        """Very small positive n_reviews should still work."""
        sigma_obs = 1.0
        n_reviews = jnp.array([0.001, 0.1, 0.5])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # Values < 1 are clamped to 1.0
        # So all should get single-review penalty
        assert np.allclose(result, 2.0)

    def test_mixed_valid_invalid(self):
        """Should handle mix of valid and invalid n_reviews."""
        sigma_obs = 1.0
        n_reviews = jnp.array([100.0, -5.0, 0.0, 25.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # Valid: 100 -> 1/10 = 0.1, 25 -> 1/5 = 0.2
        # Invalid: clamped to 1 -> 2.0 (single review penalty)
        expected = [0.1, 2.0, 2.0, 0.2]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_sigma_obs(self):
        """Should work with scalar sigma_obs."""
        sigma_obs = 2.0
        n_reviews = jnp.array([4.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 2.0 / sqrt(4) = 2.0 / 2 = 1.0
        assert np.isclose(result[0], 1.0)


class TestMakeScoreModel:
    """Tests for make_score_model factory function."""

    def test_raises_for_invalid_score_type(self):
        """Should raise ValueError for non-identifier score_type."""
        with pytest.raises(ValueError, match="score_type must be"):
            make_score_model("")

        with pytest.raises(ValueError, match="score_type must be"):
            make_score_model("not a prefix!")

    def test_accepts_custom_domain_prefix(self):
        """Any identifier works as a prefix (descriptor-driven domains)."""
        model = make_score_model("perf")
        assert callable(model)

    def test_returns_callable_for_user(self):
        """Should return callable for 'user' score type."""
        model = make_score_model("user")
        assert callable(model)

    def test_returns_callable_for_critic(self):
        """Should return callable for 'critic' score type."""
        model = make_score_model("critic")
        assert callable(model)

    def test_returned_model_has_docstring(self):
        """Returned model should have expected docstring."""
        model = make_score_model("user")

        assert model.__doc__ is not None
        assert "user" in model.__doc__
        assert "hierarchical" in model.__doc__.lower()

    def test_user_model_docstring_has_correct_prefix(self):
        """User model docstring should mention user_ prefix."""
        model = make_score_model("user")

        assert "user_" in model.__doc__

    def test_critic_model_docstring_has_correct_prefix(self):
        """Critic model docstring should mention critic_ prefix."""
        model = make_score_model("critic")

        assert "critic_" in model.__doc__


class TestModelParameterValidation:
    """Tests for model parameter validation."""

    @pytest.fixture
    def minimal_data(self):
        """Create minimal valid model data."""
        n_obs = 10
        n_artists = 3
        n_features = 2

        return {
            "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
            "album_seq": jnp.array([1, 2, 3, 1, 2, 1, 2, 3, 1, 2]),
            "prev_score": jnp.zeros(n_obs),
            "X": random.normal(random.PRNGKey(0), (n_obs, n_features)),
            "y": random.normal(random.PRNGKey(1), (n_obs,)) * 10 + 70,
            "n_artists": n_artists,
            "max_seq": 3,
        }

    def test_raises_when_n_artists_none(self, minimal_data):
        """Should raise ValueError when n_artists is None."""
        data = {**minimal_data, "n_artists": None}

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=1,
            num_samples=1,
            num_chains=1,
        )

        with pytest.raises(ValueError, match="n_artists must be provided"):
            mcmc.run(random.PRNGKey(0), **data)

    def test_raises_when_max_seq_none(self, minimal_data):
        """Should raise ValueError when max_seq is None."""
        data = {**minimal_data, "max_seq": None}

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=1,
            num_samples=1,
            num_chains=1,
        )

        with pytest.raises(ValueError, match="max_seq must be provided"):
            mcmc.run(random.PRNGKey(0), **data)

    def test_raises_for_invalid_n_exponent_prior(self, minimal_data):
        """Should raise ValueError for invalid n_exponent_prior."""
        data = {
            **minimal_data,
            "n_reviews": jnp.ones(len(minimal_data["y"])) * 100,
        }

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=1,
            num_samples=1,
            num_chains=1,
        )

        with pytest.raises(ValueError, match="Invalid n_exponent_prior"):
            mcmc.run(
                random.PRNGKey(0),
                learn_n_exponent=True,
                n_exponent_prior="invalid-prior-type",
                **data,
            )


class TestExportedModels:
    """Tests for user_score_model and critic_score_model exports."""

    def test_user_score_model_exists_and_callable(self):
        """user_score_model should exist and be callable."""
        assert user_score_model is not None
        assert callable(user_score_model)

    def test_critic_score_model_exists_and_callable(self):
        """critic_score_model should exist and be callable."""
        assert critic_score_model is not None
        assert callable(critic_score_model)

    def test_user_and_critic_are_different_models(self):
        """user_score_model and critic_score_model should be distinct."""
        # They should be different function objects
        assert user_score_model is not critic_score_model

        # Their docstrings should differ
        assert "user_" in user_score_model.__doc__
        assert "critic_" in critic_score_model.__doc__

    def test_models_have_expected_docstrings(self):
        """Both models should have non-empty docstrings."""
        assert user_score_model.__doc__ is not None
        assert len(user_score_model.__doc__) > 100

        assert critic_score_model.__doc__ is not None
        assert len(critic_score_model.__doc__) > 100


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestModelStructure:
    """Tests for verifying model produces expected sample sites."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for model tests."""
        n_obs = 20
        n_artists = 5
        n_features = 2

        return {
            "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
            "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
            "prev_score": jnp.zeros(n_obs),
            "X": random.normal(random.PRNGKey(0), (n_obs, n_features)),
            "y": random.normal(random.PRNGKey(1), (n_obs,)) * 5 + 70,
            "n_artists": n_artists,
            "max_seq": (n_obs // n_artists),
        }

    def test_user_model_sample_sites(self, sample_data):
        """User model should produce expected sample sites with user_ prefix."""
        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=5,
            num_samples=5,
            num_chains=1,
        )
        mcmc.run(random.PRNGKey(0), **sample_data)

        samples = mcmc.get_samples()

        # All sites should have user_ prefix
        expected_sites = [
            "user_mu_artist",
            "user_sigma_artist",
            "user_sigma_rw",
            "user_rho",
            "user_init_artist_effect",
            "user_beta",
            "user_sigma_obs",
        ]

        for site in expected_sites:
            assert site in samples, f"Missing sample site: {site}"

    def test_critic_model_sample_sites(self, sample_data):
        """Critic model should produce expected sample sites with critic_ prefix."""
        mcmc = MCMC(
            NUTS(critic_score_model),
            num_warmup=5,
            num_samples=5,
            num_chains=1,
        )
        mcmc.run(random.PRNGKey(0), **sample_data)

        samples = mcmc.get_samples()

        # All sites should have critic_ prefix
        expected_sites = [
            "critic_mu_artist",
            "critic_sigma_artist",
            "critic_sigma_rw",
            "critic_rho",
            "critic_init_artist_effect",
            "critic_beta",
            "critic_sigma_obs",
        ]

        for site in expected_sites:
            assert site in samples, f"Missing sample site: {site}"


class TestSigmaRwPriorType:
    """Tests for LogNormal vs HalfNormal sigma_rw prior."""

    @pytest.fixture
    def minimal_args(self):
        n_obs, n_features, n_artists = 20, 2, 3
        return {
            "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
            "album_seq": jnp.ones(n_obs, dtype=jnp.int32),
            "prev_score": jnp.zeros(n_obs),
            "X": jnp.zeros((n_obs, n_features)),
            "y": None,
            "n_artists": n_artists,
            "max_seq": 1,
        }

    def test_lognormal_no_boundary_pileup(self, minimal_args):
        """LogNormal prior should have < 3% of prior mass below 0.01."""
        from numpyro.handlers import seed, trace

        priors = PriorConfig(sigma_rw_prior_type="lognormal")
        args = {**minimal_args, "priors": priors}

        values = []
        for i in range(500):
            tr = trace(seed(user_score_model, rng_seed=i)).get_trace(**args)
            values.append(float(tr["user_sigma_rw"]["value"]))

        values = np.array(values)
        frac_below = float(np.mean(values < 0.01))
        assert frac_below < 0.03, (
            f"LogNormal prior has {frac_below:.1%} mass below 0.01 (expected < 3%)"
        )

    def test_halfnormal_still_works(self, minimal_args):
        """HalfNormal prior should still function when selected."""
        from numpyro.handlers import seed, trace

        priors = PriorConfig(sigma_rw_prior_type="halfnormal")
        args = {**minimal_args, "priors": priors}
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
        val = float(tr["user_sigma_rw"]["value"])
        assert val >= 0, "sigma_rw should be non-negative"

    def test_invalid_type_raises(self, minimal_args):
        """Invalid sigma_rw_prior_type should raise ValueError."""
        from numpyro.handlers import seed, trace

        priors = PriorConfig(sigma_rw_prior_type="invalid")
        args = {**minimal_args, "priors": priors}
        with pytest.raises(ValueError, match="Invalid sigma_rw_prior_type"):
            trace(seed(user_score_model, rng_seed=0)).get_trace(**args)


# --- from unit/models/bayes/test_model_expanded.py ---


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


# --- from unit/models/bayes/test_model_new.py ---


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
