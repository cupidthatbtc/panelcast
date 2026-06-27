"""Tests for Student-t observation noise in cold-start predictions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.predict import predict_new_entity


@pytest.fixture
def degenerate_samples():
    """Posterior collapsed to constants so predictive spread is purely noise.

    sigma_artist=0 removes new-artist-effect variance; mu stays well inside
    the (0, 100) soft-clip interior so clipping does not distort the spread.
    """
    n = 4000
    return {
        "user_mu_artist": jnp.full(n, 60.0),
        "user_sigma_artist": jnp.zeros(n),
        "user_beta": jnp.zeros((n, 2)),
        "user_rho": jnp.zeros(n),
        "user_sigma_obs": jnp.full(n, 5.0),
    }


class TestStudentTColdStart:
    def test_predictive_sd_matches_student_t(self, degenerate_samples):
        """Empirical SD of y - mu must be sigma * sqrt(df/(df-2)), not sigma."""
        df = 4.0
        result = predict_new_entity(
            degenerate_samples,
            X_new=jnp.zeros(2),
            prev_score=60.0,
            prefix="user_",
            seed=7,
            likelihood_df=df,
        )
        noise = np.asarray(result["y"]) - np.asarray(result["mu"])
        expected_sd = 5.0 * np.sqrt(df / (df - 2.0))
        # 4000 draws of t(4): tolerate heavy-tail sampling error.
        assert abs(np.std(noise) - expected_sd) / expected_sd < 0.15
        # Clearly wider than Normal noise would be.
        assert np.std(noise) > 5.0 * 1.15

    def test_df_at_least_100_uses_normal(self, degenerate_samples):
        """df >= 100 is the Normal limit: SD ~= sigma."""
        result = predict_new_entity(
            degenerate_samples,
            X_new=jnp.zeros(2),
            prev_score=60.0,
            prefix="user_",
            seed=7,
            likelihood_df=200.0,
        )
        noise = np.asarray(result["y"]) - np.asarray(result["mu"])
        assert abs(np.std(noise) - 5.0) / 5.0 < 0.05

    def test_default_df_is_student_t(self, degenerate_samples):
        """Default likelihood_df=4.0 must produce t noise (matches the model)."""
        result = predict_new_entity(
            degenerate_samples,
            X_new=jnp.zeros(2),
            prev_score=60.0,
            prefix="user_",
            seed=11,
        )
        noise = np.asarray(result["y"]) - np.asarray(result["mu"])
        assert np.std(noise) > 5.0 * 1.15
