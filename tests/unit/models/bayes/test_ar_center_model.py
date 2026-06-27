"""Model- and predict-level tests for AR(1) centering.

The defining property: when prev_score == ar_center for every observation
(all debuts under global centering), the likelihood is INDEPENDENT of rho.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density

from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.predict import predict_new_entity


def _debut_model_args(center: float) -> dict:
    """All-debut data: every prev_score equals the centering value."""
    n_obs, n_artists, n_features = 12, 4, 2
    rng = np.random.default_rng(3)
    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.ones(n_obs, dtype=jnp.int32),
        "prev_score": jnp.full(n_obs, center, dtype=jnp.float32),
        "X": jnp.asarray(rng.normal(size=(n_obs, n_features)), dtype=jnp.float32),
        "y": jnp.asarray(rng.normal(70.0, 5.0, n_obs), dtype=jnp.float32),
        "n_artists": n_artists,
        "max_seq": 1,
        "ar_center": center,
    }


def _latents(args: dict) -> dict:
    """One prior draw of every latent site (excluding the observed y)."""
    tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
    return {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _logden(args: dict, params: dict) -> float:
    density, _ = log_density(user_score_model, (), args, params)
    return float(density)


class TestDebutArTermIsZero:
    def test_likelihood_independent_of_rho_when_centered(self):
        """prev_score == ar_center everywhere => rho drops out of mu, so the
        joint density changes only through rho's own prior."""
        args = _debut_model_args(center=75.0)
        params = _latents(args)

        def loglik_only(rho_value: float) -> float:
            p = dict(params)
            p["user_rho"] = jnp.asarray(rho_value)
            # Cancel rho's prior contribution by evaluating it separately:
            # difference of joint densities at two rho values equals the
            # difference of prior log-probs iff the likelihood ignores rho.
            return _logden(args, p)

        from numpyro import distributions as dist

        rho_prior = dist.TruncatedNormal(loc=0.0, scale=0.3, low=-0.99, high=0.99)
        d1, d2 = loglik_only(0.8), loglik_only(-0.8)
        prior1 = float(rho_prior.log_prob(jnp.asarray(0.8)))
        prior2 = float(rho_prior.log_prob(jnp.asarray(-0.8)))
        assert d1 - d2 == pytest.approx(prior1 - prior2, abs=1e-3)

    def test_likelihood_depends_on_rho_when_uncentered(self):
        """Same data with ar_center=0.0 (legacy): rho moves the likelihood."""
        args = _debut_model_args(center=75.0)
        args["ar_center"] = 0.0
        params = _latents(args)

        p1 = dict(params)
        p1["user_rho"] = jnp.asarray(0.8)
        p2 = dict(params)
        p2["user_rho"] = jnp.asarray(-0.8)

        from numpyro import distributions as dist

        rho_prior = dist.TruncatedNormal(loc=0.0, scale=0.3, low=-0.99, high=0.99)
        joint_diff = _logden(args, p1) - _logden(args, p2)
        prior_diff = float(rho_prior.log_prob(jnp.asarray(0.8))) - float(
            rho_prior.log_prob(jnp.asarray(-0.8))
        )
        assert abs(joint_diff - prior_diff) > 1.0

    def test_default_ar_center_zero_matches_legacy(self):
        """Omitting ar_center must equal passing 0.0 (rollback guarantee)."""
        args = _debut_model_args(center=75.0)
        args["ar_center"] = 0.0
        params = _latents(args)
        baseline = _logden(args, params)

        args_no_kwarg = {k: v for k, v in args.items() if k != "ar_center"}
        assert _logden(args_no_kwarg, params) == pytest.approx(baseline, rel=1e-6)


class TestPredictNewEntityArCenter:
    def _samples(self, rho: float, n: int = 500) -> dict:
        return {
            "user_mu_artist": jnp.zeros(n),
            "user_sigma_artist": jnp.zeros(n),
            "user_beta": jnp.zeros((n, 2)),
            "user_rho": jnp.full(n, rho),
            "user_sigma_obs": jnp.full(n, 1e-6),
        }

    def test_prev_equal_center_zeroes_ar_term(self):
        """rho large, prev_score == center: prediction ignores rho entirely."""
        from panelcast.models.bayes.transforms import get_transform

        result = predict_new_entity(
            self._samples(rho=0.9),
            X_new=jnp.zeros(2),
            prev_score=75.0,
            prefix="user_",
            seed=2,
            ar_center=75.0,
        )
        mu = np.asarray(result["mu"])
        # mu_artist=0, beta=0, AR term 0 => mu == soft_clip(0).
        expected = float(get_transform("identity", (0.0, 100.0)).transform_mu(jnp.asarray(0.0)))
        assert np.allclose(mu, expected, atol=1e-3)

        # Contrast: the uncentered form puts rho * 75 into the mean instead.
        legacy = predict_new_entity(
            self._samples(rho=0.9),
            X_new=jnp.zeros(2),
            prev_score=75.0,
            prefix="user_",
            seed=2,
            ar_center=0.0,
        )
        assert np.allclose(np.asarray(legacy["mu"]), 0.9 * 75.0, atol=0.1)

    def test_center_zero_reproduces_legacy(self):
        legacy = predict_new_entity(
            self._samples(rho=0.5),
            X_new=jnp.zeros(2),
            prev_score=60.0,
            prefix="user_",
            seed=2,
        )
        explicit = predict_new_entity(
            self._samples(rho=0.5),
            X_new=jnp.zeros(2),
            prev_score=60.0,
            prefix="user_",
            seed=2,
            ar_center=0.0,
        )
        assert np.allclose(np.asarray(legacy["y"]), np.asarray(explicit["y"]))
