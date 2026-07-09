"""Ancestral rollout (#157): h=1 reconciliation, compounding, AR feedback."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.rollout import predict_horizon
from panelcast.models.bayes.transforms import get_transform


def _posterior(
    n_samples=200,
    n_artists=3,
    n_features=2,
    sigma_rw=0.0,
    rho=0.5,
    sigma_obs=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    mu_artist = np.full(n_samples, 70.0)
    sigma_artist = np.full(n_samples, 1.0)
    z = rng.normal(size=(n_samples, n_artists))
    return {
        "user_mu_artist": jnp.asarray(mu_artist),
        "user_sigma_artist": jnp.asarray(sigma_artist),
        "user_init_artist_effect_decentered": jnp.asarray(z),
        "user_init_artist_effect": jnp.asarray(mu_artist[:, None] + z),
        "user_sigma_rw": jnp.full((n_samples,), sigma_rw),
        "user_rho": jnp.full((n_samples,), rho),
        "user_beta": jnp.asarray(rng.normal(scale=0.1, size=(n_samples, n_features))),
        "user_sigma_obs": jnp.full((n_samples,), sigma_obs),
    }


def _rollout(posterior, X_future, **kw):
    defaults = dict(
        artist_idx=jnp.arange(X_future.shape[1], dtype=jnp.int32),
        n_train_events=jnp.full((X_future.shape[1],), 3),
        y_last=jnp.full((X_future.shape[1],), 72.0),
        X_future=X_future,
        prefix="user_",
        seed=11,
        ar_center=70.0,
        likelihood_family="normal",
    )
    defaults.update(kw)
    return predict_horizon(posterior, **defaults)


class TestH1Reconciliation:
    def test_mu_matches_analytic_when_latent_is_frozen(self):
        """sigma_rw=0: the h=1 location equals the one-step linear predictor."""
        post = _posterior(sigma_rw=0.0)
        X = np.asarray(np.random.default_rng(1).normal(size=(1, 2, 2)), dtype=np.float32)
        out = _rollout(post, jnp.asarray(X), artist_idx=jnp.asarray([0, 2]))
        transform = get_transform("identity", (0.0, 100.0), 0.5)
        eff = np.asarray(post["user_init_artist_effect"])[:, [0, 2]]
        lin = np.einsum("sf,ef->se", np.asarray(post["user_beta"]), X[0])
        ar = np.asarray(post["user_rho"])[:, None] * (72.0 - 70.0)
        expected = np.asarray(transform.transform_mu(jnp.asarray(eff + lin + ar)))
        np.testing.assert_allclose(np.asarray(out["mu"])[:, 0, :], expected, rtol=1e-5)

    def test_h1_draws_reconcile_with_predictive_one_step(self):
        """The issue's sanity anchor: h=1 rollout matches the model's own
        Predictive one-step path (fresh rw_raw, observed lag) in distribution."""
        S, T = 4000, 4
        post = _posterior(n_samples=S, n_artists=1, sigma_rw=0.3, rho=0.5, seed=3)
        X = np.asarray(np.random.default_rng(2).normal(size=(1, 1, 2)), dtype=np.float32)

        out = _rollout(
            post,
            jnp.asarray(X),
            artist_idx=jnp.asarray([0]),
            n_train_events=jnp.asarray([T]),
            likelihood_family="studentt",
            likelihood_df=4.0,
        )
        roll_draws = np.asarray(out["y"])[:, 0, 0]

        predictive = Predictive(make_score_model("user"), posterior_samples=post, batch_ndims=1)
        preds = predictive(
            random.key(5),
            artist_idx=np.array([0], dtype=np.int32),
            album_seq=np.array([T + 1], dtype=np.int32),
            prev_score=np.array([72.0], dtype=np.float32),
            X=X[0].astype(np.float32),
            y=None,
            n_artists=1,
            max_seq=T + 1,
            priors=PriorConfig(),
            likelihood_df=4.0,
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        ref_draws = np.asarray(preds["user_y"])[:, 0]

        assert abs(roll_draws.mean() - ref_draws.mean()) < 0.15
        assert 0.9 < roll_draws.std() / ref_draws.std() < 1.1


class TestLatentCompounding:
    def test_rw_variance_grows_with_horizon(self):
        post = _posterior(n_samples=4000, n_artists=1, sigma_rw=1.0, rho=0.0, sigma_obs=0.01)
        X = jnp.zeros((6, 1, 2), dtype=jnp.float32)
        out = _rollout(post, X, artist_idx=jnp.asarray([0]), n_train_events=jnp.asarray([1]))
        var_by_h = np.asarray(out["mu"])[:, :, 0].var(axis=0)
        assert np.all(np.diff(var_by_h) > 0)
        assert var_by_h[5] > 4.0

    def test_ar1_deviation_is_damped(self):
        post = _posterior(n_samples=4000, n_artists=1, sigma_rw=1.0, rho=0.0, sigma_obs=0.01)
        post["user_phi"] = jnp.zeros((4000,))
        X = jnp.zeros((6, 1, 2), dtype=jnp.float32)
        out = _rollout(
            post,
            X,
            artist_idx=jnp.asarray([0]),
            n_train_events=jnp.asarray([1]),
            latent_process="ar1",
        )
        var_by_h = np.asarray(out["mu"])[:, :, 0].var(axis=0)
        assert var_by_h[5] < 2.0

    def test_static_entity_stays_frozen(self):
        post = _posterior(n_samples=2000, n_artists=1, sigma_rw=1.0, rho=0.0, sigma_obs=0.1)
        X = jnp.zeros((6, 1, 2), dtype=jnp.float32)
        out = _rollout(
            post,
            X,
            artist_idx=jnp.asarray([0]),
            n_train_events=jnp.asarray([5]),
            dynamic_mask=jnp.asarray([False]),
        )
        mu = np.asarray(out["mu"])[:, :, 0]
        # Frozen latent + rho=0: each draw's location is constant across h
        # (posterior spread of the init effect remains, but never grows).
        np.testing.assert_allclose(mu.max(axis=1), mu.min(axis=1), rtol=1e-6)


class TestARFeedback:
    def test_lag_feedback_follows_the_recursion(self):
        """rho pulls the sampled lag toward the center step by step."""
        S = 500
        post = _posterior(n_samples=S, n_artists=1, sigma_rw=0.0, rho=0.8, sigma_obs=1e-4, seed=7)
        post["user_beta"] = jnp.zeros((S, 2))
        H = 4
        X = jnp.zeros((H, 1, 2), dtype=jnp.float32)
        out = _rollout(
            post, X, artist_idx=jnp.asarray([0]), y_last=jnp.asarray([90.0]), ar_center=70.0
        )
        transform = get_transform("identity", (0.0, 100.0), 0.5)
        eff = np.asarray(post["user_init_artist_effect"])[:, 0]
        m = np.full(S, 90.0)
        for h in range(H):
            mu_expected = np.asarray(transform.transform_mu(jnp.asarray(eff + 0.8 * (m - 70.0))))
            got = np.asarray(out["mu"])[:, h, 0]
            np.testing.assert_allclose(got, mu_expected, atol=5e-3)
            m = mu_expected


class TestValidation:
    def test_heteroscedastic_requires_n_reviews(self):
        post = _posterior(n_samples=50, n_artists=1)
        post["user_n_exponent"] = jnp.full((50,), 0.5)
        X = jnp.zeros((2, 1, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="n_reviews_future"):
            _rollout(post, X, artist_idx=jnp.asarray([0]))

    def test_unknown_latent_process_rejected(self):
        post = _posterior(n_samples=50, n_artists=1)
        X = jnp.zeros((1, 1, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="latent_process"):
            _rollout(post, X, artist_idx=jnp.asarray([0]), latent_process="brownian")
