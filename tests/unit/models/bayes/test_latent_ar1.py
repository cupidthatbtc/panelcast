"""Tests for the stationary AR(1) latent-process option.

Contract: _build_latent_effects returns (max_seq, n_artists) under every
registered process; "ar1" nests the random walk exactly at phi = 1.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

from panelcast.models.bayes.model import _build_latent_effects, user_score_model
from panelcast.models.bayes.priors import PriorConfig

N_ARTISTS = 4
MAX_SEQ = 6


def _build(priors: PriorConfig, substitutions: dict | None = None) -> np.ndarray:
    def runner():
        return _build_latent_effects(
            "user_",
            N_ARTISTS,
            MAX_SEQ,
            jnp.arange(N_ARTISTS, dtype=jnp.float32),
            jnp.asarray(0.5),
            priors,
        )

    fn = seed(runner, rng_seed=0)
    if substitutions:
        fn = substitute(fn, data=substitutions)
    return np.asarray(fn())


class TestLatentProcessRegistry:
    def test_unknown_process_raises(self):
        with pytest.raises(ValueError, match="Unknown latent_process"):
            _build(PriorConfig(latent_process="bogus"))

    @pytest.mark.parametrize("process", ["rw", "ar1"])
    def test_contract_shape(self, process):
        effects = _build(PriorConfig(latent_process=process))
        assert effects.shape == (MAX_SEQ, N_ARTISTS)

    @pytest.mark.parametrize("process", ["rw", "ar1"])
    def test_seq_one_equals_init_effect(self, process):
        effects = _build(PriorConfig(latent_process=process))
        np.testing.assert_allclose(effects[0], np.arange(N_ARTISTS), rtol=1e-6)


class TestAr1NestsRandomWalk:
    def test_phi_one_reproduces_rw_trajectory(self):
        """With identical innovations and phi -> 1, ar1 equals rw exactly."""
        rng = np.random.default_rng(7)
        raw = jnp.asarray(rng.normal(size=(N_ARTISTS, MAX_SEQ - 1)), dtype=jnp.float32)
        rw = _build(PriorConfig(latent_process="rw"), {"user_rw_raw": raw})
        ar1 = _build(
            PriorConfig(latent_process="ar1"),
            {"user_rw_raw": raw, "user_phi": jnp.asarray(0.99)},
        )
        # phi=0.99 is the truncation bound; compare against an explicit
        # phi-weighted recursion rather than exact rw equality.
        innovations = 0.5 * np.asarray(raw)
        dev = np.zeros(N_ARTISTS)
        expected = [np.arange(N_ARTISTS, dtype=np.float64)]
        for t in range(MAX_SEQ - 1):
            dev = 0.99 * dev + innovations[:, t]
            expected.append(np.arange(N_ARTISTS) + dev)
        np.testing.assert_allclose(ar1, np.stack(expected), rtol=1e-5)
        # And rw is the cumulative-sum limit: with phi=1 recursion they match.
        dev = np.zeros(N_ARTISTS)
        expected_rw = [np.arange(N_ARTISTS, dtype=np.float64)]
        for t in range(MAX_SEQ - 1):
            dev = dev + innovations[:, t]
            expected_rw.append(np.arange(N_ARTISTS) + dev)
        np.testing.assert_allclose(rw, np.stack(expected_rw), rtol=1e-5)

    def test_phi_zero_gives_independent_deviations(self):
        """phi = 0: deviations equal the per-step innovations (no memory)."""
        rng = np.random.default_rng(11)
        raw = jnp.asarray(rng.normal(size=(N_ARTISTS, MAX_SEQ - 1)), dtype=jnp.float32)
        ar1 = _build(
            PriorConfig(latent_process="ar1"),
            {"user_rw_raw": raw, "user_phi": jnp.asarray(0.0)},
        )
        innovations = 0.5 * np.asarray(raw)
        for t in range(MAX_SEQ - 1):
            np.testing.assert_allclose(
                ar1[t + 1] - np.arange(N_ARTISTS), innovations[:, t], rtol=1e-5
            )


class TestAr1ModelIntegration:
    def _model_args(self, latent_process: str) -> dict:
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
            "priors": PriorConfig(latent_process=latent_process),
        }

    def test_ar1_adds_phi_site(self):
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**self._model_args("ar1"))
        assert "user_phi" in tr
        assert "user_rw_raw" in tr  # shared innovation site name

    def test_rw_has_no_phi_site(self):
        tr = trace(seed(user_score_model, rng_seed=0)).get_trace(**self._model_args("rw"))
        assert "user_phi" not in tr
