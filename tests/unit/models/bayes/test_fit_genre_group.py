"""Genre/group pooling gate (#41): parity, sites, shift, and recovery.

Gate-off must be bit-identical to the legacy path (no new sites, group args
ignored), verified by seeded forward traces. Deliberately NO off-vs-on
shared-site parity test: the gate inserts sites mid-sequence (they feed the
init-effect loc), which legitimately reshuffles downstream RNG when on.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.predict import predict_new_entity
from panelcast.models.bayes.priors import PriorConfig
from panelcast.pipelines.train_bayes import _build_entity_groups

_N_OBS, _N_FEAT, _N_ART = 12, 2, 4
_GROUP_IDX = np.array([0, 0, 1, 1], dtype=np.int32)  # two groups over four artists


def _model_args(priors: PriorConfig, with_groups: bool = False) -> dict:
    rng = np.random.default_rng(0)
    args = {
        "artist_idx": jnp.array([i % _N_ART for i in range(_N_OBS)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // _N_ART) + 1 for i in range(_N_OBS)], dtype=jnp.int32),
        "prev_score": jnp.full(_N_OBS, 70.0),
        "X": jnp.asarray(rng.standard_normal((_N_OBS, _N_FEAT)), dtype=jnp.float32),
        "y": jnp.asarray(rng.normal(70.0, 8.0, _N_OBS), dtype=jnp.float32),
        "n_artists": _N_ART,
        "max_seq": 3,
        "priors": priors,
        "target_bounds": (0.0, 100.0),
        "likelihood_df": 4.0,
        "ar_center": 70.0,
    }
    if with_groups:
        args["group_idx_by_artist"] = jnp.asarray(_GROUP_IDX)
        args["n_groups"] = 2
    return args


def _seeded_trace(args: dict) -> dict:
    model = make_score_model("user")
    with handlers.seed(rng_seed=0):
        return handlers.trace(model).get_trace(**args)


class TestGateOffParity:
    def test_gate_off_ignores_group_args_bit_identical(self):
        base = _seeded_trace(_model_args(PriorConfig()))
        with_args = _seeded_trace(_model_args(PriorConfig(), with_groups=True))

        assert set(base) == set(with_args)
        for site, record in base.items():
            np.testing.assert_array_equal(
                np.asarray(record["value"]),
                np.asarray(with_args[site]["value"]),
                err_msg=f"gate-off draw changed at site '{site}'",
            )

    def test_gate_off_has_no_group_sites(self):
        trace = _seeded_trace(_model_args(PriorConfig(), with_groups=True))
        assert not any("group" in site for site in trace)


class TestGateOn:
    def test_adds_exactly_the_new_sites(self):
        off = _seeded_trace(_model_args(PriorConfig()))
        on = _seeded_trace(
            _model_args(PriorConfig(entity_group_pooling=True), with_groups=True)
        )
        assert set(on) - set(off) == {
            "user_sigma_group",
            "user_group_offset_z",
            "user_group_offset",
        }

    def test_missing_group_args_raises(self):
        args = _model_args(PriorConfig(entity_group_pooling=True))
        with pytest.raises(ValueError, match="entity_group_pooling=True requires"):
            _seeded_trace(args)

    def test_group_offsets_sum_to_zero(self):
        trace = _seeded_trace(
            _model_args(PriorConfig(entity_group_pooling=True), with_groups=True)
        )
        offset = np.asarray(trace["user_group_offset"]["value"])
        assert offset.shape == (2,)
        assert abs(offset.sum()) < 1e-5

    def test_group_offset_shifts_init_effect_loc(self):
        """With decentered noise pinned to zero, init effects sit exactly at
        mu_artist + group_offset[group(artist)]."""
        args = _model_args(PriorConfig(entity_group_pooling=True), with_groups=True)
        model = make_score_model("user")
        pinned = {
            "user_mu_artist": jnp.asarray(0.5),
            "user_sigma_group": jnp.asarray(2.0),
            "user_group_offset_z": jnp.asarray([1.0, -1.0]),
            "user_init_artist_effect_decentered": jnp.zeros(_N_ART),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)

        expected_offset = 2.0 * np.array([1.0, -1.0])
        np.testing.assert_allclose(
            np.asarray(trace["user_group_offset"]["value"]), expected_offset, rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(trace["user_init_artist_effect"]["value"]),
            0.5 + expected_offset[_GROUP_IDX],
            rtol=1e-5,
        )


class TestBuildEntityGroups:
    def test_modal_reduction_and_rest_bucket(self):
        df = pd.DataFrame(
            {
                "Artist": ["a", "a", "a", "b", "b", "c", "d", "e"],
                "primary_genre": ["Rock", "Rock", "Pop", "Rock", None, "Jazz", "Pop", None],
            }
        )
        artists = sorted(df["Artist"].unique())
        idx, group_to_idx = _build_entity_groups(df, artists, "Artist", "primary_genre")

        # Jazz has one entity -> __rest__; e has no genre -> __rest__.
        # Rock holds a+b, Pop holds d (a's mode is Rock)... Pop only holds d -> __rest__.
        assert group_to_idx["__rest__"] == 0
        assert "Rock" in group_to_idx
        assert "Jazz" not in group_to_idx
        assert "Pop" not in group_to_idx
        by_artist = dict(zip(artists, idx))
        assert by_artist["a"] == group_to_idx["Rock"]
        assert by_artist["b"] == group_to_idx["Rock"]
        assert by_artist["c"] == 0
        assert by_artist["d"] == 0
        assert by_artist["e"] == 0

    def test_deterministic_tie_break(self):
        df = pd.DataFrame(
            {
                "Artist": ["a", "a", "b", "b"],
                "primary_genre": ["Pop", "Rock", "Pop", "Rock"],
            }
        )
        idx1, m1 = _build_entity_groups(df, ["a", "b"], "Artist", "primary_genre")
        idx2, m2 = _build_entity_groups(df, ["a", "b"], "Artist", "primary_genre")
        np.testing.assert_array_equal(idx1, idx2)
        assert m1 == m2


class TestPredictNewEntityGroup:
    @staticmethod
    def _posterior(n_samples: int = 50) -> dict:
        # mu sits mid-range so the identity transform's soft-clip is inert.
        return {
            "user_mu_artist": jnp.full(n_samples, 70.0),
            "user_sigma_artist": jnp.full(n_samples, 1e-6),
            "user_beta": jnp.zeros((n_samples, _N_FEAT)),
            "user_rho": jnp.zeros(n_samples),
            "user_sigma_obs": jnp.full(n_samples, 1e-6),
            "user_group_offset": jnp.asarray(
                np.tile(np.array([5.0, -5.0]), (n_samples, 1))
            ),
        }

    def test_seen_group_shifts_mu(self):
        pred = predict_new_entity(
            self._posterior(),
            jnp.zeros((2, _N_FEAT)),
            prev_score=jnp.zeros(2),
            prefix="user_",
            group_idx_new=jnp.array([0, 1]),
        )
        mu = np.asarray(pred["mu"])
        assert np.allclose(mu[:, 0].mean(), 75.0, atol=0.1)
        assert np.allclose(mu[:, 1].mean(), 65.0, atol=0.1)

    def test_unseen_group_falls_back_to_population(self):
        pred = predict_new_entity(
            self._posterior(),
            jnp.zeros((2, _N_FEAT)),
            prev_score=jnp.zeros(2),
            prefix="user_",
            group_idx_new=jnp.array([-1, -1]),
        )
        assert np.allclose(np.asarray(pred["mu"]).mean(axis=0), 70.0, atol=0.1)

    def test_default_ignores_group_offset(self):
        """No group_idx_new (default -1 scalar) -> zero shift even when the
        posterior carries group_offset."""
        pred = predict_new_entity(
            self._posterior(),
            jnp.zeros((1, _N_FEAT)),
            prev_score=jnp.zeros(1),
            prefix="user_",
        )
        assert np.allclose(np.asarray(pred["mu"]).mean(), 70.0, atol=0.1)


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_two_group_synthetic_fit_recovers_separation():
    """Gate-on NUTS on data with a real group gap: the fitted group offsets
    separate in the right direction. Data lives on a 0-centered scale so
    mu_artist's N(0, 1) prior can carry the level (the AR term is zeroed by
    prev_score == ar_center == 0)."""
    rng = np.random.default_rng(7)
    n_art, per_artist = 12, 6
    n_obs = n_art * per_artist
    group_idx = np.array([0] * (n_art // 2) + [1] * (n_art // 2), dtype=np.int32)
    artist_idx = np.repeat(np.arange(n_art), per_artist).astype(np.int32)
    album_seq = np.tile(np.arange(1, per_artist + 1), n_art).astype(np.int32)
    level = np.where(group_idx == 0, 2.5, -2.5)
    y = rng.normal(level[artist_idx], 1.0).astype(np.float32)

    mcmc = MCMC(
        NUTS(make_score_model("user")),
        num_warmup=300,
        num_samples=300,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(0),
        artist_idx=jnp.asarray(artist_idx),
        album_seq=jnp.asarray(album_seq),
        prev_score=jnp.zeros(n_obs),
        X=jnp.zeros((n_obs, 1), dtype=jnp.float32),
        y=jnp.asarray(y),
        n_artists=n_art,
        max_seq=per_artist,
        priors=PriorConfig(entity_group_pooling=True),
        target_bounds=(-100.0, 100.0),
        likelihood_df=4.0,
        ar_center=0.0,
        group_idx_by_artist=jnp.asarray(group_idx),
        n_groups=2,
    )
    offset = np.asarray(mcmc.get_samples()["user_group_offset"])
    assert offset.shape[1] == 2
    assert offset[:, 0].mean() > offset[:, 1].mean() + 1.0
