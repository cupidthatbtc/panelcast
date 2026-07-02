"""Registry-level contracts for the plug-and-play likelihood families.

These guard the *mechanism* (every spec round-trips, the static Literal stays in
sync with the runtime registry, discretization is integer-valued and is rejected
where it cannot work) rather than any one family's statistics — those live in
``test_likelihood_families.py`` and ``test_likelihood_parity.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from panelcast.config.gates import LikelihoodFamily
from panelcast.models.bayes.likelihoods import REGISTRY, DequantizedDistribution
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

N_OBS, N_FEAT, N_ART = 40, 3, 5
DISCRETIZABLE = tuple(f for f, s in REGISTRY.items() if s.supports_discretization)


def _model_args(family: str, **prior_kw) -> dict:
    rng = np.random.default_rng(0)
    artist_idx = jnp.array([i % N_ART for i in range(N_OBS)], dtype=jnp.int32)
    album_seq = jnp.array([(i // N_ART) + 1 for i in range(N_OBS)], dtype=jnp.int32)
    if family == "beta_ceiling":
        # Normally derived from the training split by prepare_model_data.
        prior_kw.setdefault("effective_ceiling", 95.0)
    return dict(
        artist_idx=artist_idx,
        album_seq=album_seq,
        prev_score=jnp.full(N_OBS, 70.0),
        X=jnp.asarray(rng.standard_normal((N_OBS, N_FEAT)), dtype=jnp.float32),
        n_artists=N_ART,
        max_seq=int(album_seq.max()),
        # beta_binomial needs it; the other families are homoscedastic here and ignore it.
        n_reviews=jnp.full(N_OBS, 50, dtype=jnp.int32),
        priors=PriorConfig(likelihood_family=family, **prior_kw),
        target_bounds=(0.0, 100.0),
        likelihood_df=4.0,
        ar_center=70.0,
    )


class TestRegistryContract:
    def test_literal_matches_registry(self):
        assert set(LikelihoodFamily.__args__) == set(REGISTRY)

    def test_spec_names_match_keys(self):
        for key, spec in REGISTRY.items():
            assert spec.name == key

    def test_discretizable_specs_expose_cdf(self):
        for family, spec in REGISTRY.items():
            if spec.supports_discretization:
                assert spec.cdf is not None, f"{family} discretizable but has no cdf"

    @pytest.mark.parametrize("family", sorted(REGISTRY))
    def test_required_sites_present_in_trace(self, family):
        model = make_score_model("user")
        tr = trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **_model_args(family))
        assert "user_y" in tr
        for site in REGISTRY[family].required_sites:
            assert f"user_{site}" in tr, f"{family}: missing required site user_{site}"

    @pytest.mark.parametrize("family", sorted(REGISTRY))
    def test_predictive_round_trips(self, family):
        pred = Predictive(make_score_model("user"), num_samples=20)
        draws = np.asarray(pred(random.PRNGKey(1), y=None, **_model_args(family))["user_y"])
        assert draws.shape == (20, N_OBS)
        assert np.isfinite(draws).all()


class TestDiscretization:
    @pytest.mark.parametrize("family", DISCRETIZABLE)
    def test_discretized_predictive_is_integer(self, family):
        pred = Predictive(make_score_model("user"), num_samples=20)
        draws = np.asarray(
            pred(
                random.PRNGKey(2),
                y=None,
                **_model_args(family, discretize_observation=True),
            )["user_y"]
        )
        assert np.array_equal(draws, np.round(draws)), f"{family}: non-integer reps"

    @pytest.mark.parametrize("family", DISCRETIZABLE)
    def test_discretized_log_density_is_finite(self, family):
        # Include the boundary integers 0 and 100, where the old interval-CDF
        # log-difference underflowed.
        draws = np.round(np.clip(np.random.default_rng(5).normal(70, 8, N_OBS), 0, 100))
        draws[0], draws[1] = 0.0, 100.0
        y_int = jnp.asarray(draws, dtype=jnp.float32)
        tr = trace(seed(make_score_model("user"), random.PRNGKey(3))).get_trace(
            y=y_int, **_model_args(family, discretize_observation=True)
        )
        site = tr["user_y"]
        lp = np.asarray(site["fn"].log_prob(site["value"]))
        assert np.isfinite(lp).all(), f"{family}: non-finite dequantized log_prob"

    def test_dequantized_log_prob_has_finite_gradient(self):
        # The crux the interval-CDF path failed: its tail log-difference gave a
        # flat/NaN gradient and walled the sampler. normal and studentt were the
        # two underflow-prone CDFs.
        import jax

        y_extreme = jnp.array([0.0, 100.0, 250.0, -120.0])

        def make_logp(base_ctor):
            def logp(mu, sigma):
                d = DequantizedDistribution(base_ctor(mu, sigma))
                return jnp.sum(d.log_prob(y_extreme))

            return logp

        base_ctors = {
            "normal": lambda mu, sigma: dist.Normal(mu, sigma),
            "studentt": lambda mu, sigma: dist.StudentT(4.0, mu, sigma),
        }
        for name, ctor in base_ctors.items():
            g_mu, g_sigma = jax.grad(make_logp(ctor), argnums=(0, 1))(70.0, 8.0)
            assert jnp.isfinite(g_mu), f"{name}: non-finite d log_prob / d mu at tail"
            assert jnp.isfinite(g_sigma), f"{name}: non-finite d log_prob / d sigma at tail"

    @pytest.mark.parametrize(
        "family", ("beta", "skew_studentt", "beta_binomial", "beta_ceiling")
    )
    def test_discretization_rejected_for_unsupported(self, family):
        with pytest.raises(ValueError, match="discretiz"):
            trace(seed(make_score_model("user"), random.PRNGKey(0))).get_trace(
                y=None, **_model_args(family, discretize_observation=True)
            )

    def test_non_discretizable_have_no_cdf(self):
        for family in ("beta", "skew_studentt", "beta_binomial", "beta_ceiling"):
            assert REGISTRY[family].cdf is None


class TestBetaCeiling:
    def test_missing_effective_ceiling_raises(self):
        args = _model_args("beta_ceiling")
        args["priors"] = PriorConfig(likelihood_family="beta_ceiling")
        with pytest.raises(ValueError, match="effective_ceiling"):
            trace(seed(make_score_model("user"), random.PRNGKey(0))).get_trace(y=None, **args)

    def test_ceiling_above_bounds_rejected(self):
        with pytest.raises(ValueError, match="low < ceiling"):
            trace(seed(make_score_model("user"), random.PRNGKey(0))).get_trace(
                y=None, **_model_args("beta_ceiling", effective_ceiling=150.0)
            )

    def test_ceiling_persisted_as_posterior_site(self):
        tr = trace(seed(make_score_model("user"), random.PRNGKey(0))).get_trace(
            y=None, **_model_args("beta_ceiling")
        )
        assert float(tr["user_effective_ceiling"]["value"]) == 95.0

    def test_replicated_draws_respect_ceiling(self):
        pred = Predictive(make_score_model("user"), num_samples=50)
        draws = np.asarray(
            pred(random.PRNGKey(1), y=None, **_model_args("beta_ceiling"))["user_y"]
        )
        assert draws.max() <= 95.0
        assert draws.min() >= 0.0
