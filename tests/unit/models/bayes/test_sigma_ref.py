"""Tests for sigma_ref reparameterization algebraic correctness.

Tests verify:
1. Correct sample/deterministic sites exist when sigma_ref mode is active
2. Algebraic identity sigma_obs = sigma_ref * n_ref^n_exp holds exactly
   in prior predictive samples (no MCMC needed -- fast, deterministic)
3. Identity holds in real MCMC posterior samples (short integration tests)
4. Homoscedastic mode is unaffected (backward compatibility)
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive

from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.priors import PriorConfig

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------


def _make_sample_data() -> dict:
    """Create sample data dict for model tests."""
    key = random.PRNGKey(42)
    n_obs = 50
    n_artists = 10
    n_features = 3

    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
        "prev_score": jnp.zeros(n_obs),
        "X": random.normal(key, (n_obs, n_features)),
        "n_artists": n_artists,
        "max_seq": (n_obs // n_artists),
        "n_reviews": jnp.array([10 + i * 5 for i in range(n_obs)], dtype=jnp.float32),
    }


# ============================================================================
# Class 1: Site existence tests
# ============================================================================


class TestSigmaRefSiteExistence:
    """Tests that the correct sample/deterministic sites exist."""

    @pytest.fixture
    def sample_data(self):
        return _make_sample_data()

    def test_sigma_ref_site_exists_with_n_ref(self, sample_data):
        """sigma_ref and sigma_obs (deterministic) should appear when n_ref is set."""
        predictive = Predictive(
            user_score_model,
            num_samples=10,
            exclude_deterministic=False,
        )
        samples = predictive(
            random.PRNGKey(0),
            **sample_data,
            y=None,
            n_ref=50.0,
            learn_n_exponent=True,
        )
        assert "user_sigma_ref" in samples
        assert "user_sigma_obs" in samples

    def test_sigma_ref_not_present_when_n_ref_none(self, sample_data):
        """sigma_ref should NOT appear when n_ref is None (homoscedastic)."""
        predictive = Predictive(
            user_score_model,
            num_samples=10,
            exclude_deterministic=False,
        )
        samples = predictive(
            random.PRNGKey(0),
            **sample_data,
            y=None,
            n_ref=None,
            n_exponent=0.0,
            learn_n_exponent=False,
        )
        assert "user_sigma_ref" not in samples
        assert "user_sigma_obs" in samples

    def test_sigma_ref_not_present_when_homoscedastic_with_n_ref_none(self, sample_data):
        """Explicitly passing n_ref=None with homoscedastic settings takes original path."""
        # Remove n_reviews to be fully homoscedastic
        data = {k: v for k, v in sample_data.items() if k != "n_reviews"}
        predictive = Predictive(
            user_score_model,
            num_samples=10,
            exclude_deterministic=False,
        )
        samples = predictive(
            random.PRNGKey(0),
            **data,
            y=None,
            n_ref=None,
            n_exponent=0.0,
            learn_n_exponent=False,
        )
        assert "user_sigma_ref" not in samples
        assert "user_sigma_obs" in samples


# ============================================================================
# Class 2: Algebraic identity (prior predictive -- fast, no MCMC)
# ============================================================================


class TestSigmaRefAlgebraicIdentity:
    """Core mathematical verification using prior predictive.

    The algebraic identity being tested:
        sigma_obs = sigma_ref * n_ref^n_exp

    This must hold exactly (within float tolerance) for every sample.
    """

    @pytest.fixture
    def sample_data(self):
        return _make_sample_data()

    def test_algebraic_identity_prior_predictive(self, sample_data):
        """Identity sigma_obs = sigma_ref * n_ref^n_exp holds for 1000 samples."""
        n_ref = 50.0
        predictive = Predictive(
            user_score_model,
            num_samples=1000,
            exclude_deterministic=False,
        )
        samples = predictive(
            random.PRNGKey(0),
            **sample_data,
            y=None,
            n_ref=n_ref,
            learn_n_exponent=True,
        )

        sigma_ref = samples["user_sigma_ref"]
        sigma_obs = samples["user_sigma_obs"]
        n_exp = samples["user_n_exponent"]

        expected_sigma_obs = sigma_ref * jnp.power(n_ref, n_exp)
        np.testing.assert_allclose(
            sigma_obs,
            expected_sigma_obs,
            rtol=1e-5,
            err_msg="Algebraic identity sigma_obs = sigma_ref * n_ref^n_exp violated",
        )

    def test_algebraic_identity_with_fixed_exponent(self, sample_data):
        """Identity holds with fixed n_exponent=0.33."""
        n_ref = 50.0
        fixed_exp = 0.33
        predictive = Predictive(
            user_score_model,
            num_samples=1000,
            exclude_deterministic=False,
        )
        samples = predictive(
            random.PRNGKey(0),
            **sample_data,
            y=None,
            n_ref=n_ref,
            n_exponent=fixed_exp,
            learn_n_exponent=False,
        )

        sigma_ref = samples["user_sigma_ref"]
        sigma_obs = samples["user_sigma_obs"]

        expected_sigma_obs = sigma_ref * jnp.power(n_ref, fixed_exp)
        np.testing.assert_allclose(
            sigma_obs,
            expected_sigma_obs,
            rtol=1e-5,
            err_msg="Algebraic identity violated with fixed exponent",
        )

    @pytest.mark.parametrize("n_ref", [5.0, 50.0, 500.0, 5000.0])
    def test_algebraic_identity_various_n_ref_values(self, sample_data, n_ref):
        """Identity holds across a range of n_ref values."""
        predictive = Predictive(
            user_score_model,
            num_samples=100,
            exclude_deterministic=False,
        )
        samples = predictive(
            random.PRNGKey(0),
            **sample_data,
            y=None,
            n_ref=n_ref,
            learn_n_exponent=True,
        )

        sigma_ref = samples["user_sigma_ref"]
        sigma_obs = samples["user_sigma_obs"]
        n_exp = samples["user_n_exponent"]

        expected_sigma_obs = sigma_ref * jnp.power(n_ref, n_exp)
        np.testing.assert_allclose(
            sigma_obs,
            expected_sigma_obs,
            rtol=1e-5,
            err_msg=f"Algebraic identity violated for n_ref={n_ref}",
        )


# ============================================================================
# Class 3: MCMC integration tests (real posterior samples)
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestSigmaRefMCMCIntegration:
    """Short MCMC tests confirming identity holds in real posterior samples."""

    @pytest.fixture
    def sample_data(self):
        data = _make_sample_data()
        # MCMC tests need observed y
        data["y"] = random.normal(random.PRNGKey(1), (50,)) * 5 + 70
        return data

    def test_sigma_ref_identity_in_mcmc_samples(self, sample_data):
        """Algebraic identity holds in real MCMC posterior samples."""
        n_ref = 50.0
        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=20,
            num_samples=20,
            num_chains=1,
        )
        mcmc.run(
            random.PRNGKey(0),
            **sample_data,
            n_ref=n_ref,
            learn_n_exponent=True,
        )
        samples = mcmc.get_samples()

        sigma_ref = samples["user_sigma_ref"]
        sigma_obs = samples["user_sigma_obs"]
        n_exp = samples["user_n_exponent"]

        expected_sigma_obs = sigma_ref * jnp.power(n_ref, n_exp)
        np.testing.assert_allclose(
            sigma_obs,
            expected_sigma_obs,
            rtol=1e-5,
            err_msg="Algebraic identity violated in MCMC posterior samples",
        )

    def test_sigma_ref_mcmc_with_fixed_exponent(self, sample_data):
        """Identity holds in MCMC with fixed exponent."""
        n_ref = 50.0
        fixed_exp = 0.33
        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=20,
            num_samples=20,
            num_chains=1,
        )
        mcmc.run(
            random.PRNGKey(0),
            **sample_data,
            n_ref=n_ref,
            n_exponent=fixed_exp,
            learn_n_exponent=False,
        )
        samples = mcmc.get_samples()

        sigma_ref = samples["user_sigma_ref"]
        sigma_obs = samples["user_sigma_obs"]

        expected_sigma_obs = sigma_ref * jnp.power(n_ref, fixed_exp)
        np.testing.assert_allclose(
            sigma_obs,
            expected_sigma_obs,
            rtol=1e-5,
            err_msg="Algebraic identity violated in MCMC with fixed exponent",
        )


# ============================================================================
# Class 4: Backward compatibility
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestSigmaRefBackwardCompat:
    """Backward compatibility: homoscedastic mode unchanged."""

    @pytest.fixture
    def sample_data(self):
        data = _make_sample_data()
        data["y"] = random.normal(random.PRNGKey(1), (50,)) * 5 + 70
        return data

    def test_homoscedastic_unchanged_without_n_ref(self, sample_data):
        """Homoscedastic mode (n_ref=None) preserves original sigma_obs sampling."""
        # Remove n_reviews for pure homoscedastic
        data = {k: v for k, v in sample_data.items() if k != "n_reviews"}
        mcmc = MCMC(
            NUTS(user_score_model),
            num_warmup=20,
            num_samples=20,
            num_chains=1,
        )
        mcmc.run(
            random.PRNGKey(0),
            **data,
            n_ref=None,
            n_exponent=0.0,
            learn_n_exponent=False,
        )
        samples = mcmc.get_samples()

        assert "user_sigma_obs" in samples, "sigma_obs should be sampled in homoscedastic mode"
        assert (
            "user_sigma_ref" not in samples
        ), "sigma_ref should NOT be present in homoscedastic mode"
