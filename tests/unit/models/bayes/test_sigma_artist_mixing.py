"""Tests for the sigma_artist mixing gates (zerosum effects, lognormal prior).

Both gates default to legacy behavior; they exist to attack the published
sigma_artist ESS deficit (561 < 800) via geometry, and are adopted only if
they win the cheap fixture-run ESS comparison.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, trace

from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.priors import PriorConfig


def _model_args(priors: PriorConfig) -> dict:
    n_obs, n_artists, n_features = 20, 4, 2
    rng = np.random.default_rng(0)
    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
        "prev_score": jnp.zeros(n_obs),
        "X": jnp.asarray(rng.normal(size=(n_obs, n_features)), dtype=jnp.float32),
        "y": jnp.asarray(rng.normal(70.0, 5.0, n_obs), dtype=jnp.float32),
        "n_artists": n_artists,
        "max_seq": 5,
        "priors": priors,
    }


class TestSigmaArtistPriorType:
    def test_default_is_halfnormal(self):
        assert PriorConfig().sigma_artist_prior_type == "halfnormal"

    def test_lognormal_no_boundary_pileup(self):
        """LogNormal prior should have < 3% of prior mass below 0.01."""
        priors = PriorConfig(sigma_artist_prior_type="lognormal")
        args = _model_args(priors)
        values = []
        for i in range(300):
            tr = trace(seed(user_score_model, rng_seed=i)).get_trace(**args)
            values.append(float(tr["user_sigma_artist"]["value"]))
        frac_below = float(np.mean(np.asarray(values) < 0.01))
        assert frac_below < 0.03

    def test_halfnormal_still_works(self):
        args = _model_args(PriorConfig(sigma_artist_prior_type="halfnormal"))
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
        assert float(tr["user_sigma_artist"]["value"]) >= 0

    def test_invalid_type_raises(self):
        args = _model_args(PriorConfig(sigma_artist_prior_type="bogus"))
        with pytest.raises(ValueError, match="sigma_artist_prior_type"):
            trace(seed(user_score_model, rng_seed=0)).get_trace(**args)


class TestArtistEffectParam:
    def test_default_is_noncentered(self):
        assert PriorConfig().artist_effect_param == "noncentered"

    def test_zerosum_deviations_sum_to_zero(self):
        args = _model_args(PriorConfig(artist_effect_param="zerosum"))
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
        assert "user_artist_effect_z" in tr
        z = np.asarray(tr["user_artist_effect_z"]["value"])
        assert z.shape == (args["n_artists"],)
        assert float(np.sum(z)) == pytest.approx(0.0, abs=1e-5)
        # The exported effect site survives as a deterministic.
        assert "user_init_artist_effect" in tr
        mu = float(tr["user_mu_artist"]["value"])
        sigma = float(tr["user_sigma_artist"]["value"])
        effects = np.asarray(tr["user_init_artist_effect"]["value"])
        np.testing.assert_allclose(effects, mu + sigma * z, rtol=1e-5)

    def test_noncentered_keeps_legacy_sites(self):
        args = _model_args(PriorConfig(artist_effect_param="noncentered"))
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
        assert "user_artist_effect_z" not in tr
        # The exported model is reparameterized: the decentered site carries
        # the sampled values and the original name is reconstructed.
        assert any(k.startswith("user_init_artist_effect") for k in tr)

    def test_invalid_param_raises(self):
        args = _model_args(PriorConfig(artist_effect_param="bogus"))
        with pytest.raises(ValueError, match="artist_effect_param"):
            trace(seed(user_score_model, rng_seed=0)).get_trace(**args)
