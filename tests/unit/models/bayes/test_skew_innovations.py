"""Asymmetric random-walk innovations (#233): parity, sites, consistency.

"normal" must be bit-identical to the legacy path (no new sites);
"skew_normal" draws innovations from a learned-alpha skew-normal via
standardized_skew_innovation — the single source of truth shared with the
horizon rollout.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro import handlers

from panelcast.models.bayes.model import make_score_model, standardized_skew_innovation
from panelcast.models.bayes.priors import (
    PriorConfig,
    entity_skew_site_names,
    is_skew_rw_innovation,
    rw_latent_sites,
)

_N_OBS, _N_FEAT, _N_ART = 12, 2, 4


def _model_args(priors: PriorConfig, max_seq: int = 3) -> dict:
    rng = np.random.default_rng(0)
    return {
        "artist_idx": jnp.array([i % _N_ART for i in range(_N_OBS)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // _N_ART) + 1 for i in range(_N_OBS)], dtype=jnp.int32),
        "prev_score": jnp.full(_N_OBS, 70.0),
        "X": jnp.asarray(rng.standard_normal((_N_OBS, _N_FEAT)), dtype=jnp.float32),
        "y": jnp.asarray(rng.normal(70.0, 8.0, _N_OBS), dtype=jnp.float32),
        "n_artists": _N_ART,
        "max_seq": max_seq,
        "priors": priors,
        "target_bounds": (0.0, 100.0),
        "likelihood_df": 4.0,
        "ar_center": 70.0,
    }


def _seeded_trace(args: dict) -> dict:
    model = make_score_model("user")
    with handlers.seed(rng_seed=0):
        return handlers.trace(model).get_trace(**args)


def test_skew_random_walk_gate_is_exact():
    assert is_skew_rw_innovation("skew_normal")
    assert not is_skew_rw_innovation(None)
    assert not is_skew_rw_innovation(" SKEW_NORMAL ")


def test_random_walk_site_names_cover_both_prefix_conventions():
    expected_skew = ("user_rw_raw", "user_rw_raw_abs")
    assert rw_latent_sites(
        "user", "skew_normal", has_trajectory=True
    ).present() == expected_skew
    assert rw_latent_sites(
        "user_", "skew_normal", has_trajectory=True
    ).present() == expected_skew
    assert rw_latent_sites("user_", "normal", has_trajectory=True).present() == (
        "user_rw_raw",
    )
    assert rw_latent_sites("user", "normal", has_trajectory=False).present() == ()


def test_entity_skew_site_names_follow_the_resolved_prior_type():
    assert entity_skew_site_names("user_", "skew_normal") == (
        "user_entity_skew_abs",
        "user_entity_skew_sym",
    )
    assert entity_skew_site_names("user", "normal") == ()


class TestNormalParity:
    def test_normal_is_bit_identical_to_legacy(self):
        base = _seeded_trace(_model_args(PriorConfig()))
        explicit = _seeded_trace(_model_args(PriorConfig(rw_innovation_type="normal")))
        assert set(base) == set(explicit)
        for site, record in base.items():
            np.testing.assert_array_equal(
                np.asarray(record["value"]),
                np.asarray(explicit[site]["value"]),
                err_msg=f"normal-path draw changed at site '{site}'",
            )

    def test_normal_has_no_skew_sites(self):
        trace = _seeded_trace(_model_args(PriorConfig()))
        assert not any("rw_skew" in site or "rw_raw_abs" in site for site in trace)


class TestSkewInnovations:
    def test_adds_exactly_the_new_sites(self):
        off = _seeded_trace(_model_args(PriorConfig()))
        on = _seeded_trace(_model_args(PriorConfig(rw_innovation_type="skew_normal")))
        assert set(on) - set(off) == {"user_rw_skew_alpha", "user_rw_raw_abs"}
        assert set(off) - set(on) == set()

    def test_unknown_value_rejected(self):
        args = _model_args(PriorConfig(rw_innovation_type="nope"))
        with pytest.raises(ValueError, match="rw_innovation_type"):
            _seeded_trace(args)

    def test_single_step_domain_skips_the_gate(self):
        # max_seq=1 builds no trajectory, so the gate must not create sites.
        trace = _seeded_trace(
            _model_args(PriorConfig(rw_innovation_type="skew_normal"), max_seq=1)
        )
        assert not any("rw_skew" in site for site in trace)

    def test_training_innovations_use_the_shared_helper(self):
        """The likelihood location equals the helper-rebuilt trajectory —
        the single-source-of-truth property the rollout relies on. With
        beta and rho pinned to zero (and the AR term centered away), each
        observation's location is exactly init + cumsum(helper innovations)
        at its sequence position, mid-range so the soft-clip is ~identity."""
        priors = PriorConfig(rw_innovation_type="skew_normal", mu_artist_loc=50.0)
        args = _model_args(priors)
        model = make_score_model("user")
        pinned = {
            "user_beta": jnp.zeros(_N_FEAT),
            "user_rho": jnp.asarray(0.0),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        rw_raw = np.asarray(trace["user_rw_raw"]["value"])
        rw_abs = np.asarray(trace["user_rw_raw_abs"]["value"])
        alpha = np.asarray(trace["user_rw_skew_alpha"]["value"])
        sigma_rw = np.asarray(trace["user_sigma_rw"]["value"])
        init = np.asarray(trace["user_init_artist_effect"]["value"])
        innovations = sigma_rw * np.asarray(
            standardized_skew_innovation(jnp.asarray(rw_abs), jnp.asarray(rw_raw), alpha)
        )
        trajectory = np.cumsum(innovations, axis=1)  # (n_artists, max_seq-1)
        artist_idx = np.asarray(args["artist_idx"])
        album_seq = np.asarray(args["album_seq"])
        expected = np.where(
            album_seq == 1,
            init[artist_idx],
            init[artist_idx] + trajectory[artist_idx, np.maximum(album_seq - 2, 0)],
        )
        loc = np.asarray(trace["user_y"]["fn"].loc)
        np.testing.assert_allclose(loc, expected, atol=1e-3)  # soft-clip slack


class TestSharedHelper:
    def test_alpha_zero_is_the_symmetric_part(self):
        z_abs = jnp.asarray([2.0, 3.0])
        z_sym = jnp.asarray([0.5, -1.0])
        out = np.asarray(standardized_skew_innovation(z_abs, z_sym, jnp.asarray(0.0)))
        np.testing.assert_allclose(out, [0.5, -1.0], rtol=1e-6)

    def test_moments_and_skewness(self):
        key = random.key(0)
        k1, k2 = random.split(key)
        n = 200_000
        z_abs = jnp.abs(random.normal(k1, (n,)))
        z_sym = random.normal(k2, (n,))
        draws = np.asarray(standardized_skew_innovation(z_abs, z_sym, jnp.asarray(3.0)))
        assert abs(draws.mean()) < 0.01
        assert abs(draws.std() - 1.0) < 0.01
        centered = draws - draws.mean()
        skewness = float((centered**3).mean() / (centered**2).mean() ** 1.5)
        assert 0.55 < skewness < 0.8  # SkewNormal(alpha=3) skewness ~ 0.667

    def _rollout(self, posterior, n_entities: int):
        from panelcast.models.bayes.rollout import predict_horizon

        return predict_horizon(
            posterior,
            artist_idx=jnp.arange(n_entities),
            # One training event: no accumulated pre-horizon deviation, so
            # the h=1 draw is exactly one innovation.
            n_train_events=jnp.full(n_entities, 1),
            y_last=jnp.full(n_entities, 50.0),
            X_future=jnp.zeros((1, n_entities, _N_FEAT), dtype=jnp.float32),
            seed=0,
            target_bounds=(0.0, 100.0),
            ar_center=50.0,
        )

    def test_rollout_compounds_the_skewed_innovation(self):
        """With rw_skew_alpha in the posterior, the h=1 rollout draw is one
        skewed innovation: alpha = -3 must produce left-skewed deviations
        where the gate-off posterior stays symmetric."""
        n_samples, n_entities = 4000, 2
        base = {
            "user_init_artist_effect": jnp.full((n_samples, n_entities), 50.0),
            "user_sigma_rw": jnp.full((n_samples,), 1.0),
            "user_rho": jnp.zeros((n_samples,)),
            "user_beta": jnp.zeros((n_samples, _N_FEAT)),
            "user_sigma_obs": jnp.full((n_samples,), 0.01),
        }
        skewed = self._rollout(
            {**base, "user_rw_skew_alpha": jnp.full((n_samples,), -3.0)}, n_entities
        )
        symmetric = self._rollout(base, n_entities)

        def sample_skew(y: np.ndarray) -> float:
            centered = y - y.mean()
            return float((centered**3).mean() / (centered**2).mean() ** 1.5)

        skew_on = sample_skew(np.asarray(skewed["y"][:, 0, 0]))
        skew_off = sample_skew(np.asarray(symmetric["y"][:, 0, 0]))
        assert skew_on < -0.3  # SkewNormal(alpha=-3) skewness ~ -0.667
        assert abs(skew_off) < 0.2
