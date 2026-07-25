"""Boundary censoring (#234): parity, censored mass, atoms, conventions."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
import pytest
from jax import random
from numpyro import handlers

from panelcast.models.bayes.likelihoods import BoundCensoredDistribution
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

_N_OBS, _N_FEAT, _N_ART = 12, 2, 4


def _model_args(priors: PriorConfig, y=None) -> dict:
    rng = np.random.default_rng(0)
    if y is None:
        y = rng.normal(70.0, 8.0, _N_OBS)
    return {
        "artist_idx": jnp.array([i % _N_ART for i in range(_N_OBS)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // _N_ART) + 1 for i in range(_N_OBS)], dtype=jnp.int32),
        "prev_score": jnp.full(_N_OBS, 70.0),
        "X": jnp.asarray(rng.standard_normal((_N_OBS, _N_FEAT)), dtype=jnp.float32),
        "y": jnp.asarray(y, dtype=jnp.float32),
        "n_artists": _N_ART,
        "max_seq": 3,
        "priors": priors,
        "target_bounds": (0.0, 100.0),
        "likelihood_df": 4.0,
        "ar_center": 70.0,
    }


def _seeded_trace(args: dict) -> dict:
    model = make_score_model("user")
    with handlers.seed(rng_seed=0):
        return handlers.trace(model).get_trace(**args)


import numpyro.distributions as dist  # noqa: E402


class TestDistribution:
    def _censored(self, mu=50.0, sigma=10.0):
        base = dist.Normal(mu, sigma)
        cdf = lambda v: jss.norm.cdf(v, loc=mu, scale=sigma)  # noqa: E731
        return BoundCensoredDistribution(base, cdf, 0.0, 100.0), base

    def test_interior_matches_the_base_density(self):
        censored, base = self._censored()
        v = jnp.asarray([30.0, 50.0, 70.0])
        np.testing.assert_allclose(
            np.asarray(censored.log_prob(v)), np.asarray(base.log_prob(v)), rtol=1e-6
        )

    def test_bounds_contribute_cdf_mass(self):
        near_high, _ = self._censored(mu=95.0, sigma=10.0)
        upper = float(near_high.log_prob(jnp.asarray(100.0)))
        expected_upper = np.log(1.0 - jss.norm.cdf(100.0, loc=95.0, scale=10.0))
        np.testing.assert_allclose(upper, expected_upper, rtol=1e-5)
        near_low, _ = self._censored(mu=5.0, sigma=10.0)
        lower = float(near_low.log_prob(jnp.asarray(0.0)))
        expected_lower = np.log(jss.norm.cdf(0.0, loc=5.0, scale=10.0))
        np.testing.assert_allclose(lower, expected_lower, rtol=1e-4)

    def test_negligible_mass_hits_the_numerical_floor(self):
        # A bound 9.5 sigma out has mass below _TINY; the floor keeps the
        # log finite instead of -inf.
        censored, _ = self._censored(mu=95.0, sigma=10.0)
        assert np.isfinite(float(censored.log_prob(jnp.asarray(0.0))))

    def test_samples_clip_and_carry_boundary_atoms(self):
        censored, _ = self._censored(mu=95.0, sigma=10.0)
        draws = np.asarray(censored.sample(random.key(0), (20_000,)))
        assert draws.min() >= 0.0 and draws.max() <= 100.0
        # P(Y >= 100 | mu=95, sigma=10) ~ 0.31: a real atom, not a fluke.
        assert (draws == 100.0).mean() > 0.2

    def test_empirical_pit_at_the_bound_is_the_interval_upper_edge(self):
        """The pinned convention: clipped replicated draws count as <= the
        bound, so empirical PIT at a censored bound sits at 1."""
        censored, _ = self._censored(mu=95.0, sigma=10.0)
        draws = np.asarray(censored.sample(random.key(0), (20_000,)))
        pit_at_bound = (draws <= 100.0).mean()
        assert pit_at_bound == 1.0


class TestModelGate:
    def test_gate_off_is_bit_identical(self):
        base = _seeded_trace(_model_args(PriorConfig()))
        explicit = _seeded_trace(_model_args(PriorConfig(censor_at_bounds=False)))
        assert set(base) == set(explicit)
        for site, record in base.items():
            np.testing.assert_array_equal(
                np.asarray(record["value"]),
                np.asarray(explicit[site]["value"]),
                err_msg=f"gate-off draw changed at site '{site}'",
            )

    def _pinned_trace(self, priors: PriorConfig, y):
        model = make_score_model("user")
        pinned = {
            "user_mu_artist": jnp.asarray(95.0),
            "user_init_artist_effect_decentered": jnp.zeros(_N_ART),
            "user_rw_raw": jnp.zeros((_N_ART, 2)),
            "user_beta": jnp.zeros(_N_FEAT),
            "user_rho": jnp.asarray(0.0),
            "user_sigma_obs": jnp.asarray(5.0),
        }
        with handlers.seed(rng_seed=0):
            return handlers.trace(handlers.substitute(model, pinned)).get_trace(
                **_model_args(priors, y=y)
            )

    def test_gate_on_adds_no_sites_and_scores_boundary_mass(self):
        y = np.full(_N_OBS, 90.0)
        y[0] = 100.0  # a boundary observation, one sigma above mu=95
        off_priors = PriorConfig(heteroscedastic_entity_obs=False)
        on_priors = PriorConfig(heteroscedastic_entity_obs=False, censor_at_bounds=True)
        off = self._pinned_trace(off_priors, y)
        on = self._pinned_trace(on_priors, y)
        assert set(on) == set(off)  # censoring is a likelihood change, not a site
        lp_on = np.asarray(on["user_y"]["fn"].log_prob(on["user_y"]["value"]))
        lp_off = np.asarray(off["user_y"]["fn"].log_prob(off["user_y"]["value"]))
        # With mu pinned at 95 and sigma 5, the boundary point's censored
        # tail mass P(Y >= 100) beats the density at exactly 100; interior
        # points are untouched.
        assert lp_on[0] > lp_off[0]
        np.testing.assert_allclose(lp_on[1:], lp_off[1:], rtol=1e-5)

    @pytest.mark.parametrize("family", ["skew_studentt", "beta"])
    def test_unsupported_families_reject(self, family):
        priors = PriorConfig(
            censor_at_bounds=True,
            likelihood_family=family,
            target_transform="identity",
        )
        with pytest.raises(ValueError, match="censor_at_bounds"):
            _seeded_trace(_model_args(priors))

    def test_offset_logit_thresholds_are_forward_mapped(self):
        from panelcast.models.bayes.likelihoods import _censor_thresholds
        from panelcast.models.bayes.transforms import get_transform

        priors = PriorConfig(censor_at_bounds=True, target_transform="offset_logit")
        low, high = _censor_thresholds(priors, (0.0, 100.0))
        transform = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.5)
        np.testing.assert_allclose(float(low), float(transform.forward(jnp.asarray(0.0))))
        np.testing.assert_allclose(float(high), float(transform.forward(jnp.asarray(100.0))))
        assert np.isfinite(float(low)) and np.isfinite(float(high))


class TestConfigPlumbing:
    def test_censor_with_discretize_rejected(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="mutually"):
            PipelineConfig(
                censor_at_bounds=True,
                discretize_observation=True,
                target_transform="identity",
            )

    def test_command_string_records_the_gate(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(censor_at_bounds=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        assert "--censor-at-bounds" in orch._build_command_string()
