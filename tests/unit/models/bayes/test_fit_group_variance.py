"""Per-group entity-effect variances (#271): parity, sites, scaling.

"shared" must be bit-identical to the legacy pooling path (no new sites);
"per_group" gives each entity group its own sigma_artist via log-scale
partial pooling around the shared draw.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

_N_OBS, _N_FEAT, _N_ART = 12, 2, 4
_GROUP_IDX = np.array([0, 0, 1, 1], dtype=np.int32)  # two groups over four artists


def _model_args(priors: PriorConfig, with_groups: bool = True) -> dict:
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


class TestSharedParity:
    def test_shared_is_bit_identical_to_pooling_only(self):
        pooling = PriorConfig(entity_group_pooling=True)
        explicit = PriorConfig(entity_group_pooling=True, group_variance="shared")
        base = _seeded_trace(_model_args(pooling))
        same = _seeded_trace(_model_args(explicit))
        assert set(base) == set(same)
        for site, record in base.items():
            np.testing.assert_array_equal(
                np.asarray(record["value"]),
                np.asarray(same[site]["value"]),
                err_msg=f"shared draw changed at site '{site}'",
            )

    def test_shared_has_no_variance_sites(self):
        trace = _seeded_trace(_model_args(PriorConfig(entity_group_pooling=True)))
        assert not any("group_sigma" in site or "sigma_artist_group" in site for site in trace)

    def test_shared_site_set_is_the_pinned_pooling_roster(self):
        # Hard-pin the pooling-only roster so a shared-path divergence from
        # the pre-#271 model cannot slip through the field-equal parity test.
        trace = _seeded_trace(_model_args(PriorConfig(entity_group_pooling=True)))
        assert set(trace) == {
            "user_artists",
            "user_beta",
            "user_entity_log_scale",
            "user_entity_obs_raw",
            "user_group_offset",
            "user_group_offset_z",
            "user_init_artist_effect",
            "user_init_artist_effect_decentered",
            "user_mu_artist",
            "user_obs",
            "user_rho",
            "user_rw_raw",
            "user_sigma_artist",
            "user_sigma_group",
            "user_sigma_obs",
            "user_sigma_rw",
            "user_tau_entity",
            "user_y",
        }


class TestPerGroup:
    def test_adds_exactly_the_new_sites(self):
        off = _seeded_trace(_model_args(PriorConfig(entity_group_pooling=True)))
        on = _seeded_trace(
            _model_args(
                PriorConfig(entity_group_pooling=True, group_variance="per_group")
            )
        )
        assert set(on) - set(off) == {
            "user_tau_group_sigma",
            "user_group_sigma_z",
            "user_sigma_artist_group",
        }

    def test_requires_entity_group_pooling(self):
        args = _model_args(PriorConfig(group_variance="per_group"), with_groups=True)
        with pytest.raises(ValueError, match="requires entity_group_pooling"):
            _seeded_trace(args)

    def test_unknown_group_variance_raises(self):
        args = _model_args(
            PriorConfig(entity_group_pooling=True, group_variance="nope")
        )
        with pytest.raises(ValueError, match="group_variance"):
            _seeded_trace(args)

    def test_pinned_draws_scale_each_group_around_the_shared_sigma(self):
        """sigma_g = sigma_artist * exp(tau * z_g), and each entity's init
        effect uses its own group's sigma."""
        priors = PriorConfig(entity_group_pooling=True, group_variance="per_group")
        args = _model_args(priors)
        model = make_score_model("user")
        pinned = {
            "user_mu_artist": jnp.asarray(0.0),
            "user_sigma_artist": jnp.asarray(4.0),
            "user_sigma_group": jnp.asarray(0.0),
            "user_group_offset_z": jnp.asarray([0.0, 0.0]),
            "user_tau_group_sigma": jnp.asarray(0.5),
            "user_group_sigma_z": jnp.asarray([1.0, -1.0]),
            "user_init_artist_effect_decentered": jnp.ones(_N_ART),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        sigma_group = np.asarray(trace["user_sigma_artist_group"]["value"])
        np.testing.assert_allclose(
            sigma_group, 4.0 * np.exp(0.5 * np.array([1.0, -1.0])), rtol=1e-6
        )
        # Decentered z pinned to 1 => init effect equals that entity's sigma.
        init = np.asarray(trace["user_init_artist_effect"]["value"])
        np.testing.assert_allclose(init, sigma_group[_GROUP_IDX], rtol=1e-5)

    def test_zerosum_param_with_per_group_runs_but_loses_exact_zero_sum(self):
        # Pin the interaction: with a per-entity sigma vector, sigma * z no
        # longer sums to zero, so zerosum identification is only approximate.
        priors = PriorConfig(
            entity_group_pooling=True,
            group_variance="per_group",
            artist_effect_param="zerosum",
        )
        args = _model_args(priors)
        model = make_score_model("user")
        pinned = {
            "user_mu_artist": jnp.asarray(0.0),
            "user_sigma_artist": jnp.asarray(4.0),
            "user_sigma_group": jnp.asarray(0.0),
            "user_group_offset_z": jnp.asarray([0.0, 0.0]),
            "user_tau_group_sigma": jnp.asarray(0.5),
            "user_group_sigma_z": jnp.asarray([1.0, -1.0]),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        init = np.asarray(trace["user_init_artist_effect"]["value"])
        z = np.asarray(trace["user_artist_effect_z"]["value"])
        np.testing.assert_allclose(z.sum(), 0.0, atol=1e-5)
        assert abs(init.sum()) > 1e-6  # sigma-weighted deviations: not zero-sum

    def test_tau_zero_recovers_shared(self):
        priors = PriorConfig(entity_group_pooling=True, group_variance="per_group")
        args = _model_args(priors)
        model = make_score_model("user")
        pinned = {
            "user_sigma_artist": jnp.asarray(4.0),
            "user_tau_group_sigma": jnp.asarray(0.0),
            "user_group_sigma_z": jnp.asarray([2.0, -3.0]),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        np.testing.assert_allclose(
            np.asarray(trace["user_sigma_artist_group"]["value"]), [4.0, 4.0], rtol=1e-6
        )
