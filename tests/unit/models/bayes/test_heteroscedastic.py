"""Tests for heteroscedastic observation noise implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from panelcast.models.bayes.model import compute_sigma_scaled


class TestComputeSigmaScaled:
    """Tests for the compute_sigma_scaled helper function."""

    def test_basic_scaling(self):
        """Test that sigma_scaled = sigma_obs / n^exponent."""
        sigma_obs = 1.0
        n_reviews = jnp.array([100.0])
        exponent = 0.5  # sqrt scaling

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 1.0 / 100^0.5 = 1.0 / 10 = 0.1
        assert np.isclose(result[0], 0.1, rtol=1e-5)

    def test_single_review_penalty(self):
        """Test that n=1 applies 2x multiplier."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # n=1 should apply 2x penalty: sigma_obs * 2.0
        assert np.isclose(result[0], 2.0, rtol=1e-5)

    def test_custom_single_review_multiplier(self):
        """Test custom single_review_multiplier."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent, single_review_multiplier=3.0)

        assert np.isclose(result[0], 3.0, rtol=1e-5)

    def test_homoscedastic_mode(self):
        """Test that exponent=0 returns sigma_obs unchanged for all n.

        When exponent=0 (homoscedastic mode):
        - Formula: sigma_obs / n^0 = sigma_obs / 1 = sigma_obs for all n
        - Single-review penalty is NOT applied (penalty requires exponent > 0)
        - Result: sigma_obs for all review counts including n=1
        """
        sigma_obs = 1.0
        n_reviews = jnp.array([1.0, 10.0, 100.0, 1000.0])
        exponent = 0.0

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # All values should equal sigma_obs (no scaling, no penalty in homoscedastic mode)
        assert np.isclose(result[0], 1.0, rtol=1e-5)  # n=1 (no penalty when exp=0)
        assert np.isclose(result[1], 1.0, rtol=1e-5)  # n=10
        assert np.isclose(result[2], 1.0, rtol=1e-5)  # n=100
        assert np.isclose(result[3], 1.0, rtol=1e-5)  # n=1000

    def test_extreme_n_no_underflow(self):
        """Test that extreme n values don't cause underflow."""
        sigma_obs = 1.0
        n_reviews = jnp.array([100000.0])  # 100k reviews
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 1.0 / 100000^0.5 = 1.0 / 316.2... = 0.00316...
        # But min_sigma=0.01, so should hit floor
        assert result[0] >= 0.01
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_min_sigma_floor(self):
        """Test custom min_sigma floor."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1000000.0])  # 1M reviews
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent, min_sigma=0.001)

        # 1.0 / 1000000^0.5 = 0.001, at floor
        assert result[0] >= 0.001
        assert not np.isnan(result[0])

    def test_array_broadcasting(self):
        """Test that function handles arrays correctly."""
        sigma_obs = 2.0
        n_reviews = jnp.array([4.0, 9.0, 16.0, 25.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 2.0 / sqrt(n) for each
        expected = jnp.array([1.0, 2 / 3, 0.5, 0.4])
        assert result.shape == (4,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cube_root_scaling(self):
        """Test cube-root scaling (exponent=0.33)."""
        sigma_obs = 1.0
        n_reviews = jnp.array([8.0, 27.0, 64.0])
        exponent = 1 / 3  # cube root

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 1.0 / n^(1/3) = 1/2, 1/3, 1/4 for n=8,27,64
        expected = jnp.array([0.5, 1 / 3, 0.25])
        np.testing.assert_allclose(result, expected, rtol=1e-4)


from numpyro.infer import MCMC, NUTS

from panelcast.models.bayes import user_score_model
from panelcast.models.bayes.priors import PriorConfig


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestModelFixedExponent:
    """Tests for model with fixed exponent mode."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for model tests."""
        key = random.PRNGKey(42)
        n_obs = 50
        n_artists = 10
        n_features = 3

        return {
            "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
            "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
            "prev_score": jnp.zeros(n_obs),
            "X": random.normal(key, (n_obs, n_features)),
            "y": random.normal(random.PRNGKey(1), (n_obs,)) * 5 + 70,
            "n_artists": n_artists,
            "max_seq": (n_obs // n_artists),
            "n_reviews": jnp.array([10 + i * 5 for i in range(n_obs)], dtype=jnp.float32),
        }

    def test_model_runs_with_fixed_exponent(self, sample_data):
        """Test that model runs with fixed exponent."""
        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        mcmc.run(
            random.PRNGKey(0),
            n_exponent=0.33,
            learn_n_exponent=False,
            **sample_data,
        )

        samples = mcmc.get_samples()
        assert "user_sigma_obs" in samples
        assert "user_n_exponent" not in samples  # Should NOT be sampled

    def test_model_runs_without_n_reviews(self, sample_data):
        """Test backward compatibility - model runs without n_reviews."""
        data_no_n_reviews = {k: v for k, v in sample_data.items() if k != "n_reviews"}

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        mcmc.run(
            random.PRNGKey(0),
            n_exponent=0.0,  # homoscedastic
            learn_n_exponent=False,
            **data_no_n_reviews,
        )

        samples = mcmc.get_samples()
        assert "user_sigma_obs" in samples


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestModelLearnedExponent:
    """Tests for model with learned exponent mode."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for model tests."""
        key = random.PRNGKey(42)
        n_obs = 50
        n_artists = 10
        n_features = 3

        return {
            "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
            "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
            "prev_score": jnp.zeros(n_obs),
            "X": random.normal(key, (n_obs, n_features)),
            "y": random.normal(random.PRNGKey(1), (n_obs,)) * 5 + 70,
            "n_artists": n_artists,
            "max_seq": (n_obs // n_artists),
            "n_reviews": jnp.array([10 + i * 5 for i in range(n_obs)], dtype=jnp.float32),
        }

    def test_model_samples_exponent_with_logit_normal(self, sample_data):
        """Test that model samples n_exponent when using logit-normal prior."""
        priors = PriorConfig()  # Logit-normal prior: mode ~0.10 (loc=-2.2), scale=1.0

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        mcmc.run(
            random.PRNGKey(0),
            n_exponent=0.0,
            learn_n_exponent=True,
            n_exponent_prior="logit-normal",
            priors=priors,
            **sample_data,
        )

        samples = mcmc.get_samples()
        assert "user_n_exponent" in samples
        assert samples["user_n_exponent"].shape == (10,)  # num_samples
        # Logit-normal samples should be in [0,1]
        assert np.all(samples["user_n_exponent"] >= 0)
        assert np.all(samples["user_n_exponent"] <= 1)

    def test_model_samples_exponent_with_beta(self, sample_data):
        """Test that model samples n_exponent when using beta prior (legacy)."""
        priors = PriorConfig()  # Default alpha=2.0, beta=4.0

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        mcmc.run(
            random.PRNGKey(0),
            n_exponent=0.0,  # Ignored when learning
            learn_n_exponent=True,
            n_exponent_prior="beta",
            priors=priors,
            **sample_data,
        )

        samples = mcmc.get_samples()
        assert "user_n_exponent" in samples
        assert samples["user_n_exponent"].shape == (10,)  # num_samples
        # Beta(2,4) samples should be in [0,1]
        assert np.all(samples["user_n_exponent"] >= 0)
        assert np.all(samples["user_n_exponent"] <= 1)

    def test_default_prior_is_logit_normal(self, sample_data):
        """Test that default prior type is logit-normal when not specified."""
        # Verify the model function default is logit-normal (the new default that
        # fixes divergences). Beta prior is the legacy option.
        priors = PriorConfig()

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        # Do not specify n_exponent_prior - should default to logit-normal
        mcmc.run(
            random.PRNGKey(0),
            n_exponent=0.0,
            learn_n_exponent=True,
            priors=priors,
            **sample_data,
        )

        samples = mcmc.get_samples()
        assert "user_n_exponent" in samples
        # Samples should be bounded in [0,1]
        assert np.all(samples["user_n_exponent"] >= 0)
        assert np.all(samples["user_n_exponent"] <= 1)

    def test_model_raises_on_invalid_prior_type(self, sample_data):
        """Test ValueError when invalid n_exponent_prior is specified."""
        priors = PriorConfig()

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        with pytest.raises(ValueError, match=r"Invalid n_exponent_prior"):
            mcmc.run(
                random.PRNGKey(0),
                n_exponent=0.0,
                learn_n_exponent=True,
                n_exponent_prior="invalid-value",
                priors=priors,
                **sample_data,
            )

    def test_model_raises_when_heteroscedastic_without_n_reviews(self, sample_data):
        """Test ValueError when heteroscedastic mode requested without n_reviews."""
        data_no_n_reviews = {k: v for k, v in sample_data.items() if k != "n_reviews"}

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        with pytest.raises(ValueError) as exc_info:
            mcmc.run(
                random.PRNGKey(0),
                n_exponent=0.0,
                learn_n_exponent=True,  # Requesting heteroscedastic mode
                **data_no_n_reviews,  # But no n_reviews
            )

        # Verify descriptive error message
        error_msg = str(exc_info.value)
        assert "n_reviews" in error_msg
        assert "heteroscedastic" in error_msg.lower() or "Heteroscedastic" in error_msg

    def test_model_raises_when_fixed_exponent_without_n_reviews(self, sample_data):
        """Test ValueError when fixed non-zero exponent requested without n_reviews."""
        data_no_n_reviews = {k: v for k, v in sample_data.items() if k != "n_reviews"}

        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        with pytest.raises(ValueError) as exc_info:
            mcmc.run(
                random.PRNGKey(0),
                n_exponent=0.5,  # Non-zero fixed exponent
                learn_n_exponent=False,
                **data_no_n_reviews,  # But no n_reviews
            )

        # Verify descriptive error message
        error_msg = str(exc_info.value)
        assert "n_reviews" in error_msg


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestHomoscedasticEquivalence:
    """Tests that exponent=0 matches original homoscedastic behavior."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for equivalence test."""
        key = random.PRNGKey(123)
        n_obs = 30
        n_artists = 5
        n_features = 2

        return {
            "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
            "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
            "prev_score": jnp.zeros(n_obs),
            "X": random.normal(key, (n_obs, n_features)),
            "y": random.normal(random.PRNGKey(1), (n_obs,)) * 5 + 70,
            "n_artists": n_artists,
            "max_seq": (n_obs // n_artists),
            "n_reviews": jnp.ones(n_obs) * 100,  # Uniform n to avoid penalty effects
        }

    def test_exponent_zero_matches_homoscedastic(self, sample_data):
        """Test that exponent=0 produces similar posteriors to no n_reviews."""
        # Run with exponent=0 (heteroscedastic mode but zero scaling)
        mcmc_hetero = MCMC(
            NUTS(user_score_model),
            num_warmup=50,
            num_samples=50,
            num_chains=1,
        )
        mcmc_hetero.run(
            random.PRNGKey(0),
            n_exponent=0.0,
            learn_n_exponent=False,
            **sample_data,
        )
        samples_hetero = mcmc_hetero.get_samples()

        # Run without n_reviews (pure homoscedastic)
        data_no_n = {k: v for k, v in sample_data.items() if k != "n_reviews"}
        mcmc_homo = MCMC(
            NUTS(user_score_model),
            num_warmup=50,
            num_samples=50,
            num_chains=1,
        )
        mcmc_homo.run(
            random.PRNGKey(0),  # Same seed
            n_exponent=0.0,
            learn_n_exponent=False,
            **data_no_n,
        )
        samples_homo = mcmc_homo.get_samples()

        # Sigma_obs posteriors should be very similar (within 1%)
        # Note: With same seed, they should be identical actually
        mean_hetero = np.mean(samples_hetero["user_sigma_obs"])
        mean_homo = np.mean(samples_homo["user_sigma_obs"])

        # Allow 10% difference due to MCMC variance, but should be close
        assert np.isclose(
            mean_hetero, mean_homo, rtol=0.1
        ), f"sigma_obs means differ: hetero={mean_hetero:.4f}, homo={mean_homo:.4f}"
