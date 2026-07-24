"""Period-effects gate (#269): parity, sites, constraints, unseen periods.

Gate-off must be bit-identical to the legacy path (no new sites, period args
ignored), verified by seeded forward traces. Gate-on inserts sites
mid-sequence, which legitimately reshuffles downstream RNG when on.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

_N_OBS, _N_FEAT, _N_ART, _N_PER = 12, 2, 4, 3
# Three periods cycling over the observations.
_PERIOD_IDX = np.array([i % _N_PER for i in range(_N_OBS)], dtype=np.int32)


def _model_args(priors: PriorConfig, with_periods: bool = False) -> dict:
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
    if with_periods:
        args["period_idx"] = jnp.asarray(_PERIOD_IDX)
        args["n_periods"] = _N_PER
    return args


def _seeded_trace(args: dict) -> dict:
    model = make_score_model("user")
    with handlers.seed(rng_seed=0):
        return handlers.trace(model).get_trace(**args)


class TestGateOffParity:
    def test_gate_off_ignores_period_args_bit_identical(self):
        base = _seeded_trace(_model_args(PriorConfig()))
        with_args = _seeded_trace(_model_args(PriorConfig(), with_periods=True))

        assert set(base) == set(with_args)
        for site, record in base.items():
            np.testing.assert_array_equal(
                np.asarray(record["value"]),
                np.asarray(with_args[site]["value"]),
                err_msg=f"gate-off draw changed at site '{site}'",
            )

    def test_gate_off_has_no_period_sites(self):
        trace = _seeded_trace(_model_args(PriorConfig(), with_periods=True))
        assert not any("period" in site for site in trace)


class TestGateOn:
    def test_adds_exactly_the_new_sites(self):
        off = _seeded_trace(_model_args(PriorConfig()))
        on = _seeded_trace(
            _model_args(PriorConfig(period_effects=True), with_periods=True)
        )
        assert set(on) - set(off) == {
            "user_sigma_period",
            "user_period_offset_z",
            "user_period_offset",
        }

    def test_missing_period_args_raises(self):
        args = _model_args(PriorConfig(period_effects=True))
        with pytest.raises(ValueError, match="period_effects=True requires"):
            _seeded_trace(args)

    def test_zero_sum_offsets_sum_to_zero(self):
        trace = _seeded_trace(
            _model_args(PriorConfig(period_effects=True), with_periods=True)
        )
        offset = np.asarray(trace["user_period_offset"]["value"])
        assert offset.shape == (_N_PER,)
        assert abs(offset.sum()) < 1e-5

    @pytest.mark.parametrize(
        ("constraint", "pinned"),
        [("pin_first", 0), ("pin_last", _N_PER - 1)],
    )
    def test_pin_constraints_zero_the_pinned_period(self, constraint, pinned):
        trace = _seeded_trace(
            _model_args(
                PriorConfig(period_effects=True, period_constraint=constraint),
                with_periods=True,
            )
        )
        offset = np.asarray(trace["user_period_offset"]["value"])
        assert offset.shape == (_N_PER,)
        assert offset[pinned] == 0.0
        # The free periods are real draws, not degenerate zeros.
        assert np.any(offset != 0.0)

    def test_unknown_constraint_raises(self):
        args = _model_args(
            PriorConfig(period_effects=True, period_constraint="nope"),
            with_periods=True,
        )
        with pytest.raises(ValueError, match="period_constraint"):
            _seeded_trace(args)

    def test_offset_shifts_mu_and_unseen_period_is_zero(self):
        """With everything else pinned flat, mu differs between observations
        exactly by the period offsets, and period_idx=-1 contributes zero."""
        # mu_artist_loc mid-range keeps mu_raw in the soft-clip's ~identity
        # interior, so period offsets survive the transform unattenuated.
        priors = PriorConfig(period_effects=True, mu_artist_loc=70.0)
        args = _model_args(priors, with_periods=True)
        # Observations 0/1/2 -> periods 0/1/2, observation 3 unseen (-1).
        # Pin the constrained z draw for a known offset vector.
        args["period_idx"] = jnp.asarray(
            np.array([0, 1, 2, -1] + [0] * (_N_OBS - 4), dtype=np.int32)
        )
        model = make_score_model("user")
        pinned = {
            "user_sigma_period": jnp.asarray(2.0),
            "user_period_offset_z": jnp.asarray([1.0, 0.0, -1.0]),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        offset = np.asarray(trace["user_period_offset"]["value"])
        np.testing.assert_allclose(offset, [2.0, 0.0, -2.0], rtol=1e-6)
        mu = np.asarray(trace["user_y"]["fn"].loc)
        base = _model_args(priors, with_periods=True)
        base["period_idx"] = jnp.asarray(np.full(_N_OBS, -1, dtype=np.int32))
        with handlers.seed(rng_seed=0):
            base_trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**base)
        base_mu = np.asarray(base_trace["user_y"]["fn"].loc)
        # Same seed + same pins: the likelihood location differs only by each
        # observation's period offset (soft-clip is ~identity mid-range), and
        # the unseen period contributes exactly zero.
        np.testing.assert_allclose(mu[0] - base_mu[0], 2.0, atol=1e-3)
        np.testing.assert_allclose(mu[1] - base_mu[1], 0.0, atol=1e-3)
        np.testing.assert_allclose(mu[2] - base_mu[2], -2.0, atol=1e-3)
        np.testing.assert_allclose(mu[3] - base_mu[3], 0.0, atol=1e-3)
