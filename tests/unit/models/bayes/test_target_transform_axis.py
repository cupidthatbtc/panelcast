"""Parametrized target-transform axis for model and predict tests.

Every test runs under both "identity" and "offset_logit" so the transform
seam stays exercised in CI without MCMC. Cross-transform invariant: a
posterior collapsed to constants must produce the same score-scale
prediction regardless of the scale the model trained on.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer.util import initialize_model

from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.predict import predict_new_entity
from panelcast.models.bayes.priors import priors_for_transform
from panelcast.models.bayes.transforms import get_transform

TRANSFORMS = ["identity", "offset_logit"]
BOUNDS = (0.0, 100.0)

# Degenerate-posterior locations on each model scale. 60.0 on the raw
# scale; 0.5 on the logit scale (~62 raw). Noise scales are O(target).
MODEL_SCALE = {
    "identity": {"mu": 60.0, "sigma": 5.0},
    "offset_logit": {"mu": 0.5, "sigma": 0.4},
}


def _model_args(transform_name: str, observed: bool = True) -> dict:
    """Minimal model args with y/prev_score on the model's training scale."""
    n_obs, n_artists, n_features = 20, 4, 2
    rng = np.random.default_rng(0)
    t = get_transform(transform_name, BOUNDS)
    y_raw = rng.normal(70.0, 8.0, n_obs).clip(1.0, 99.0).astype(np.float32)
    prev_raw = rng.normal(70.0, 8.0, n_obs).clip(1.0, 99.0).astype(np.float32)
    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
        "prev_score": jnp.asarray(t.forward(jnp.asarray(prev_raw))),
        "X": jnp.asarray(rng.normal(size=(n_obs, n_features)), dtype=jnp.float32),
        "y": jnp.asarray(t.forward(jnp.asarray(y_raw))) if observed else None,
        "n_artists": n_artists,
        "max_seq": (n_obs + n_artists - 1) // n_artists,
        "priors": priors_for_transform(transform_name),
        "target_bounds": BOUNDS,
    }


@pytest.mark.parametrize("transform_name", TRANSFORMS)
class TestModelTransformAxis:
    """The model must expose the same structure under either transform."""

    def test_latent_sites_identical_across_transforms(self, transform_name):
        args = _model_args(transform_name)
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
        for site in [
            "user_mu_artist",
            "user_sigma_artist",
            "user_sigma_rw",
            "user_rho",
            "user_beta",
            "user_sigma_obs",
            "user_y",
        ]:
            assert site in tr, f"Missing site {site} under {transform_name}"

    def test_initial_log_density_finite(self, transform_name):
        """Observed y on the training scale must yield a finite density."""
        args = _model_args(transform_name)
        model_info = initialize_model(jrandom.PRNGKey(0), user_score_model, model_kwargs=args)
        potential = model_info.potential_fn(model_info.param_info.z)
        assert np.isfinite(float(potential))

    def test_prior_predictive_draws_back_transform_into_bounds(self, transform_name):
        """Prior-predictive obs draws live on the model scale; the inverse
        transform must land inside the bounds extension."""
        args = _model_args(transform_name, observed=False)
        t = get_transform(transform_name, BOUNDS, offset=0.5)
        draws = []
        for i in range(50):
            tr = trace(seed(user_score_model, rng_seed=i)).get_trace(**args)
            draws.append(np.asarray(tr["user_y"]["value"]))
        y_raw = np.asarray(t.inverse(jnp.asarray(np.stack(draws))))
        assert np.all(np.isfinite(y_raw))
        if transform_name == "offset_logit":
            # sigmoid output cannot leave [low - offset, high + offset]
            assert np.all(y_raw >= BOUNDS[0] - 0.5)
            assert np.all(y_raw <= BOUNDS[1] + 0.5)


def _degenerate_posterior(mu: float, sigma_obs: float, n: int = 2000, rho: float = 0.0) -> dict:
    """Posterior collapsed to constants so the predictive path is deterministic
    up to observation noise (sigma_artist=0 removes new-artist variance)."""
    return {
        "user_mu_artist": jnp.full(n, mu),
        "user_sigma_artist": jnp.zeros(n),
        "user_beta": jnp.zeros((n, 2)),
        "user_rho": jnp.full(n, rho),
        "user_sigma_obs": jnp.full(n, sigma_obs),
    }


@pytest.mark.parametrize("transform_name", TRANSFORMS)
class TestPredictNewEntityTransformAxis:
    """predict_new_entity must return score-scale outputs for any transform."""

    def test_mu_matches_back_transformed_location(self, transform_name):
        ps = MODEL_SCALE[transform_name]
        samples = _degenerate_posterior(ps["mu"], ps["sigma"])
        result = predict_new_entity(
            samples,
            X_new=jnp.zeros(2),
            prev_score=0.0,
            prefix="user_",
            seed=3,
            target_bounds=BOUNDS,
            target_transform=transform_name,
        )
        mu = np.asarray(result["mu"])
        t = get_transform(transform_name, BOUNDS, offset=0.5)
        expected = float(t.inverse(t.transform_mu(jnp.asarray(ps["mu"]))))
        assert np.all(np.isfinite(mu))
        assert np.allclose(mu, expected, atol=1e-3)

    def test_draws_finite_and_centred_on_mu(self, transform_name):
        ps = MODEL_SCALE[transform_name]
        samples = _degenerate_posterior(ps["mu"], ps["sigma"])
        result = predict_new_entity(
            samples,
            X_new=jnp.zeros(2),
            prev_score=0.0,
            prefix="user_",
            seed=7,
            target_bounds=BOUNDS,
            target_transform=transform_name,
        )
        y = np.asarray(result["y"])
        mu = float(np.asarray(result["mu"])[0])
        assert np.all(np.isfinite(y))
        assert abs(float(np.median(y)) - mu) < 2.0
        if transform_name == "offset_logit":
            # Back-transformed draws cannot leave the bounds extension.
            assert np.all(y >= BOUNDS[0] - 0.5)
            assert np.all(y <= BOUNDS[1] + 0.5)

    def test_prev_score_consumed_on_model_scale(self, transform_name):
        """rho=1, mu_artist=0, tiny noise: the prediction reduces to the
        back-transformed prev_score, so both transforms must yield ~80."""
        t = get_transform(transform_name, BOUNDS, offset=0.5)
        prev_model = float(t.forward(jnp.asarray(80.0)))
        samples = _degenerate_posterior(0.0, 1e-6, rho=1.0)
        result = predict_new_entity(
            samples,
            X_new=jnp.zeros(2),
            prev_score=prev_model,
            prefix="user_",
            seed=5,
            target_bounds=BOUNDS,
            target_transform=transform_name,
        )
        mu = np.asarray(result["mu"])
        assert np.allclose(mu, 80.0, atol=0.1)
