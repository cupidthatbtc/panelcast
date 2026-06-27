"""Long-horizon random-walk variance: the mechanism behind propagate_rw_horizon.

The model's seq-index clip (model.py) is generic in max_seq, so passing a larger
max_seq at prediction grows the re-sampled rw_raw trajectory and the deep-horizon
predictive accumulates the full innovations. The pipeline gate just chooses
whether to clamp at max_seq_train or pass album_seq.max(); this proves the model
side widens deep intervals while leaving within-horizon draws distributionally
unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from panelcast.models.bayes.model import user_score_model

N_DRAWS = 400
N_ARTISTS = 3


def _posterior() -> dict:
    """Fixed scalar posterior with a large sigma_rw so the RW dominates; the
    high-cardinality latents (init effect, rw_raw) are left to resample."""
    return {
        "user_mu_artist": jnp.zeros(N_DRAWS),
        "user_sigma_artist": jnp.full(N_DRAWS, 0.01),
        "user_sigma_rw": jnp.full(N_DRAWS, 2.0),
        "user_rho": jnp.zeros(N_DRAWS),
        "user_beta": jnp.zeros((N_DRAWS, 1)),
        "user_sigma_obs": jnp.full(N_DRAWS, 0.5),
    }


def _predict_y(album_seq: int, max_seq: int) -> np.ndarray:
    pred = Predictive(user_score_model, posterior_samples=_posterior(), batch_ndims=1)
    out = pred(
        random.key(0),
        artist_idx=jnp.zeros(1, dtype=jnp.int32),
        album_seq=jnp.array([album_seq], dtype=jnp.int32),
        prev_score=jnp.zeros(1),
        X=jnp.zeros((1, 1)),
        y=None,
        n_artists=N_ARTISTS,
        max_seq=max_seq,
        ar_center=0.0,
    )
    return np.asarray(out["user_y"]).reshape(-1)


@pytest.mark.slow
class TestRwHorizonVariance:
    def test_deep_horizon_widens_when_propagated(self):
        h = 12
        # Clamped: max_seq=2 freezes the trajectory at seq_idx=1.
        var_clamp = float(np.var(_predict_y(album_seq=h, max_seq=2)))
        # Propagated: max_seq=h accumulates h-1 innovations.
        var_prop = float(np.var(_predict_y(album_seq=h, max_seq=h)))
        assert var_prop > 3.0 * var_clamp

    def test_variance_grows_with_horizon(self):
        # var ~ (seq_idx) * sigma_rw^2; deeper horizon => strictly wider.
        var_shallow = float(np.var(_predict_y(album_seq=3, max_seq=12)))
        var_deep = float(np.var(_predict_y(album_seq=11, max_seq=12)))
        assert var_deep > var_shallow

    def test_within_horizon_distribution_unchanged(self):
        # album_seq=2 sits within the training horizon; growing max_seq must not
        # change its predictive spread beyond Monte-Carlo noise.
        var_small = float(np.var(_predict_y(album_seq=2, max_seq=2)))
        var_large = float(np.var(_predict_y(album_seq=2, max_seq=12)))
        assert var_small == pytest.approx(var_large, rel=0.35)
