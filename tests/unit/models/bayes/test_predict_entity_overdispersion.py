"""Cold-start predictions under the entity-overdispersion gate.

Gate-on fits scale each observation's sigma by exp(tau_entity * z_entity)
during training; predict_new_entity must marginalize that factor over its
prior for unseen entities or intervals come out too narrow.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.predict import predict_new_entity

N = 4000
TAU = 0.5


@pytest.fixture
def base_samples():
    """Posterior collapsed to constants so predictive spread is purely noise."""
    return {
        "user_mu_artist": jnp.full(N, 60.0),
        "user_sigma_artist": jnp.zeros(N),
        "user_beta": jnp.zeros((N, 2)),
        "user_rho": jnp.zeros(N),
        "user_sigma_obs": jnp.full(N, 5.0),
    }


class TestEntityOverdispersionColdStart:
    def test_gate_on_widens_intervals(self, base_samples):
        """tau_entity in the posterior must widen the predictive noise."""
        off = predict_new_entity(base_samples, X_new=jnp.zeros(2), prev_score=60.0, seed=7)
        on_samples = dict(base_samples, user_tau_entity=jnp.full(N, TAU))
        on = predict_new_entity(on_samples, X_new=jnp.zeros(2), prev_score=60.0, seed=7)

        noise_off = np.asarray(off["y"]) - np.asarray(off["mu"])
        noise_on = np.asarray(on["y"]) - np.asarray(on["mu"])
        # Var multiplier is E[exp(2 tau z)] = exp(2 tau^2), SD ratio ~ 1.28 at tau=0.5
        assert np.std(noise_on) > np.std(noise_off) * 1.1

    def test_sigma_scaled_carries_inflation(self, base_samples):
        on_samples = dict(base_samples, user_tau_entity=jnp.full(N, TAU))
        result = predict_new_entity(on_samples, X_new=jnp.zeros(2), prev_score=60.0, seed=7)

        sigma = np.asarray(result["sigma_scaled"])
        assert sigma.std() > 0  # no longer the constant sigma_obs
        # log inflation is tau * z with z ~ N(0,1): mean log sigma stays at log(5)
        assert abs(np.mean(np.log(sigma)) - np.log(5.0)) < 0.05
        assert abs(np.std(np.log(sigma)) - TAU) < 0.05

    def test_one_z_shared_across_events(self, base_samples):
        """One z per posterior draw: identical inflation for the entity's events."""
        het_off = dict(base_samples, user_n_exponent=jnp.full(N, 0.5))
        het_on = dict(het_off, user_tau_entity=jnp.full(N, TAU))
        kwargs = dict(
            X_new=jnp.zeros((3, 2)),
            prev_score=jnp.full(3, 60.0),
            n_reviews_new=jnp.array([10.0, 100.0, 1000.0]),
            seed=7,
        )
        off = predict_new_entity(het_off, **kwargs)
        on = predict_new_entity(het_on, **kwargs)

        ratio = np.asarray(on["sigma_scaled"]) / np.asarray(off["sigma_scaled"])
        assert ratio.shape == (N, 3)
        assert np.allclose(ratio, ratio[:, :1])
        assert np.std(np.log(ratio[:, 0])) > 0.3  # varies across draws

    def test_deterministic_under_fixed_key(self, base_samples):
        on_samples = dict(base_samples, user_tau_entity=jnp.full(N, TAU))
        a = predict_new_entity(on_samples, X_new=jnp.zeros(2), prev_score=60.0, seed=13)
        b = predict_new_entity(on_samples, X_new=jnp.zeros(2), prev_score=60.0, seed=13)
        assert np.array_equal(np.asarray(a["y"]), np.asarray(b["y"]))
        assert np.array_equal(np.asarray(a["sigma_scaled"]), np.asarray(b["sigma_scaled"]))

    def test_subsampling_handles_tau_entity(self, base_samples):
        on_samples = dict(base_samples, user_tau_entity=jnp.full(N, TAU))
        result = predict_new_entity(
            on_samples, X_new=jnp.zeros(2), prev_score=60.0, seed=7, n_predictions=100
        )
        assert np.asarray(result["y"]).shape == (100,)
        assert np.asarray(result["sigma_scaled"]).shape == (100,)
