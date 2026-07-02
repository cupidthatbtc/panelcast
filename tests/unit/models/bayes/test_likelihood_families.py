"""Tests for the skew/bounded likelihood families and prediction dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from panelcast.models.bayes.model import SinhArcsinhTransform, make_score_model
from panelcast.models.bayes.predict import predict_new_entity
from panelcast.models.bayes.priors import PriorConfig

N_OBS, N_FEATURES, N_ARTISTS = 50, 3, 5


def _model_args(family: str) -> dict:
    rng = np.random.default_rng(0)
    artist_idx = jnp.array([i % N_ARTISTS for i in range(N_OBS)], dtype=jnp.int32)
    album_seq = jnp.array([(i // N_ARTISTS) + 1 for i in range(N_OBS)], dtype=jnp.int32)
    return dict(
        artist_idx=artist_idx,
        album_seq=album_seq,
        prev_score=jnp.full(N_OBS, 70.0),
        X=jnp.asarray(rng.standard_normal((N_OBS, N_FEATURES)), dtype=jnp.float32),
        n_artists=N_ARTISTS,
        max_seq=int(album_seq.max()),
        # beta_binomial needs it; the other families are homoscedastic here and ignore it.
        n_reviews=jnp.full(N_OBS, 50, dtype=jnp.int32),
        priors=PriorConfig(likelihood_family=family),
        target_bounds=(0.0, 100.0),
        likelihood_df=4.0,
        ar_center=70.0,
    )


class TestSinhArcsinhTransform:
    def test_identity_at_zero_skew(self):
        x = jnp.linspace(-3, 3, 11)
        assert jnp.allclose(SinhArcsinhTransform(0.0, 1.0)(x), x, atol=1e-5)

    def test_roundtrip(self):
        x = jnp.linspace(-3, 3, 11)
        t = SinhArcsinhTransform(-0.6, 1.0)
        assert jnp.allclose(t._inverse(t(x)), x, atol=1e-4)

    def test_log_det_matches_numeric_gradient(self):
        t = SinhArcsinhTransform(0.4, 1.0)
        x0 = 0.3
        y0 = float(t(jnp.array(x0)))
        analytic = float(t.log_abs_det_jacobian(jnp.array(x0), jnp.array(y0)))
        h = 1e-4
        numeric = float(t(jnp.array(x0 + h)) - t(jnp.array(x0 - h))) / (2 * h)
        assert np.isclose(np.exp(analytic), abs(numeric), rtol=1e-2)

    def test_negative_skew_tilts_left(self):
        # A symmetric base mapped with negative skew should pull mass to the left
        # (mean of transformed standard normals < 0).
        z = jnp.asarray(np.random.default_rng(1).standard_normal(20000))
        skewed = SinhArcsinhTransform(-0.8, 1.0)(z)
        assert float(jnp.mean(skewed)) < 0.0


class TestLikelihoodSites:
    @pytest.mark.parametrize(
        "family,extra",
        [
            ("studentt", None),
            ("normal", None),
            ("skew_studentt", "user_skewness"),
            ("beta", "user_phi"),
            ("skew_normal", "user_skewness"),
            ("split_normal", "user_split_log_ratio"),
            ("mixture", "user_mix_sep"),
            ("beta_binomial", "user_bb_phi"),
        ],
    )
    def test_obs_site_and_family_param(self, family, extra):
        model = make_score_model("user")
        args = _model_args(family)
        tr = trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **args)
        assert "user_y" in tr
        if extra is not None:
            assert extra in tr, f"{family} missing global site {extra}"

    def test_beta_predictive_respects_bounds(self):
        model = make_score_model("user")
        args = _model_args("beta")
        pred = Predictive(model, num_samples=100)
        draws = np.asarray(pred(random.PRNGKey(0), y=None, **args)["user_y"])
        assert draws.min() >= 0.0 and draws.max() <= 100.0

    def test_studentt_is_default(self):
        assert PriorConfig().likelihood_family == "studentt"


class TestPredictNewEntityDispatch:
    def _posterior(self, n=40, extra=None):
        rng = np.random.default_rng(2)
        post = {
            "user_mu_artist": jnp.asarray(rng.normal(0, 1, n)),
            "user_sigma_artist": jnp.asarray(np.abs(rng.normal(0.5, 0.1, n))),
            "user_beta": jnp.asarray(rng.normal(0, 1, (n, N_FEATURES))),
            "user_rho": jnp.asarray(rng.normal(0, 0.2, n)),
            "user_sigma_obs": jnp.asarray(np.abs(rng.normal(8, 1, n))),
        }
        if extra:
            post.update(extra)
        return post

    def test_beta_prediction_bounded(self):
        post = self._posterior(extra={"user_phi": jnp.asarray(np.abs(np.random.default_rng(3).normal(20, 2, 40)))})
        out = predict_new_entity(
            post,
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="beta",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        y = np.asarray(out["y"])
        assert y.min() >= 0.0 and y.max() <= 100.0

    def test_skew_prediction_runs(self):
        post = self._posterior(extra={"user_skewness": jnp.asarray(np.random.default_rng(4).normal(-0.5, 0.1, 40))})
        out = predict_new_entity(
            post,
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="skew_studentt",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        assert np.asarray(out["y"]).shape[0] == 40

    def test_beta_without_phi_raises(self):
        with pytest.raises(ValueError, match="phi"):
            predict_new_entity(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="beta",
            )

    def test_skew_normal_prediction_runs(self):
        post = self._posterior(
            extra={"user_skewness": jnp.asarray(np.random.default_rng(4).normal(-0.5, 0.1, 40))}
        )
        out = predict_new_entity(
            post,
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="skew_normal",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        assert np.asarray(out["y"]).shape[0] == 40

    def test_split_normal_prediction_runs(self):
        post = self._posterior(
            extra={"user_split_log_ratio": jnp.asarray(np.random.default_rng(4).normal(-0.3, 0.1, 40))}
        )
        out = predict_new_entity(
            post,
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="split_normal",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        assert np.asarray(out["y"]).shape[0] == 40

    def test_split_normal_without_site_raises(self):
        with pytest.raises(ValueError, match="split_log_ratio"):
            predict_new_entity(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="split_normal",
            )

    def _mixture_posterior(self):
        rng = np.random.default_rng(7)
        return self._posterior(
            extra={
                "user_mix_sep": jnp.asarray(rng.lognormal(0.0, 0.75, 40)),
                "user_mix_weight": jnp.asarray(rng.beta(2.0, 2.0, 40)),
                "user_mix_log_scale_ratio": jnp.asarray(rng.normal(0.0, 0.5, 40)),
            }
        )

    def test_mixture_prediction_runs(self):
        out = predict_new_entity(
            self._mixture_posterior(),
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="mixture",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        assert np.asarray(out["y"]).shape[0] == 40

    def test_mixture_without_site_raises(self):
        with pytest.raises(ValueError, match="mix_sep"):
            predict_new_entity(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="mixture",
            )

    def test_discretized_prediction_is_integer(self):
        out = predict_new_entity(
            self._posterior(),
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="studentt",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
            discretize_observation=True,
        )
        y = np.asarray(out["y"])
        assert np.array_equal(y, np.round(y))

    def _bb_posterior(self):
        return self._posterior(
            extra={"user_bb_phi": jnp.asarray(np.abs(np.random.default_rng(3).normal(20, 2, 40)))}
        )

    def test_beta_binomial_prediction_bounded(self):
        out = predict_new_entity(
            self._bb_posterior(),
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            n_reviews_new=jnp.asarray(50),
            likelihood_family="beta_binomial",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        y = np.asarray(out["y"])
        assert y.shape[0] == 40
        assert y.min() >= 0.0 and y.max() <= 100.0

    def test_beta_binomial_without_phi_raises(self):
        with pytest.raises(ValueError, match="bb_phi"):
            predict_new_entity(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                n_reviews_new=jnp.asarray(50),
                likelihood_family="beta_binomial",
            )

    def test_beta_binomial_without_n_reviews_raises(self):
        with pytest.raises(ValueError, match="n_reviews"):
            predict_new_entity(
                self._bb_posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="beta_binomial",
            )

    def _beta_ceiling_posterior(self):
        rng = np.random.default_rng(3)
        return self._posterior(
            extra={
                "user_phi": jnp.asarray(np.abs(rng.normal(20, 2, 40))),
                "user_effective_ceiling": jnp.full(40, 95.0),
            }
        )

    def test_beta_ceiling_prediction_respects_ceiling(self):
        out = predict_new_entity(
            self._beta_ceiling_posterior(),
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            likelihood_family="beta_ceiling",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        y = np.asarray(out["y"])
        assert y.shape[0] == 40
        assert y.min() >= 0.0 and y.max() <= 95.0

    def test_beta_ceiling_without_ceiling_site_raises(self):
        post = self._posterior(
            extra={"user_phi": jnp.asarray(np.abs(np.random.default_rng(3).normal(20, 2, 40)))}
        )
        with pytest.raises(ValueError, match="effective_ceiling"):
            predict_new_entity(
                post,
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="beta_ceiling",
            )

    def test_beta_binomial_cold_start_caps_huge_n_reviews(self):
        # A mega-reviewed event (n far above the cap) must still draw bounded scores
        # rather than blow total_count into the float32-jagged regime.
        out = predict_new_entity(
            self._bb_posterior(),
            X_new=jnp.zeros(N_FEATURES),
            prev_score=70.0,
            n_reviews_new=jnp.asarray(50_000),
            likelihood_family="beta_binomial",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
        )
        y = np.asarray(out["y"])
        assert y.shape[0] == 40
        assert y.min() >= 0.0 and y.max() <= 100.0


class TestBetaBinomialScore:
    """Generative contract of the score-scale Beta-Binomial wrapper."""

    def _dist(self, n_reviews, p=0.7, phi=30.0, low=0.0, span=100.0):
        from panelcast.models.bayes.likelihoods import BetaBinomialScore

        return BetaBinomialScore(p * phi, (1.0 - p) * phi, n_reviews, low, span)

    def test_samples_stay_on_score_scale(self):
        draws = np.asarray(self._dist(200).sample(random.PRNGKey(0), (20000,)))
        assert draws.min() >= 0.0 and draws.max() <= 100.0

    def test_mean_matches_mu(self):
        draws = np.asarray(self._dist(200, p=0.7).sample(random.PRNGKey(0), (40000,)))
        assert abs(float(draws.mean()) - 70.0) < 1.0

    def test_variance_shrinks_with_more_reviews(self):
        def var(n):
            return float(np.var(np.asarray(self._dist(n).sample(random.PRNGKey(1), (40000,)))))

        assert var(500) < var(5)

    def test_log_prob_matches_betabinomial_count(self):
        import numpyro.distributions as dist

        low, span, n, p, phi = 0.0, 100.0, 50, 0.6, 25.0
        total = int(round(span * n))
        bb = dist.BetaBinomial(p * phi, (1.0 - p) * phi, total_count=total)
        counts = jnp.array([0.0, 1500.0, 3000.0, float(total)])
        scores = low + counts / n
        d = self._dist(n, p=p, phi=phi)
        assert jnp.allclose(d.log_prob(scores), bb.log_prob(counts), atol=1e-4)


class TestBetaBinomialCap:
    """The effective-rater cap that keeps Beta-Binomial out of the jagged regime."""

    def test_cap_floors_above_passes_below_and_min_one(self):
        from panelcast.models.bayes.likelihoods import _cap_n_reviews

        out = np.asarray(
            _cap_n_reviews(jnp.array([0.0, 0.4, 1.0, 50.0, 100.0, 23000.0]), 100.0)
        )
        assert out.tolist() == [1.0, 1.0, 1.0, 50.0, 100.0, 100.0]

    def test_huge_n_reviews_capped_in_trace(self):
        from panelcast.models.bayes.priors import DEFAULT_BETABINOM_MAX_N

        model = make_score_model("user")
        cap = int(DEFAULT_BETABINOM_MAX_N)
        big = {**_model_args("beta_binomial"), "n_reviews": jnp.full(N_OBS, 50_000, jnp.int32)}
        small = {**_model_args("beta_binomial"), "n_reviews": jnp.full(N_OBS, cap, jnp.int32)}
        tc_big = np.asarray(
            trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **big)["user_y"]["fn"]._bb.total_count
        )
        tc_small = np.asarray(
            trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **small)["user_y"]["fn"]._bb.total_count
        )
        # span = 100 (target_bounds), so the capped total_count is round(100 * cap).
        assert np.all(tc_big == round(100.0 * DEFAULT_BETABINOM_MAX_N))
        assert np.array_equal(tc_big, tc_small)


class TestNormalMixture2:
    """Generative contract of the two-component Normal mixture."""

    def _dist(self, w=0.3, loc0=40.0, scale0=12.0, loc1=80.0, scale1=4.0):
        from panelcast.models.bayes.likelihoods import NormalMixture2

        return NormalMixture2(loc0, scale0, loc1, scale1, w)

    def test_log_prob_is_logaddexp_of_components(self):
        import jax.scipy.stats as jss

        v = jnp.linspace(0.0, 120.0, 25)
        lp0 = jnp.log(0.3) + jss.norm.logpdf(v, loc=40.0, scale=12.0)
        lp1 = jnp.log(0.7) + jss.norm.logpdf(v, loc=80.0, scale=4.0)
        assert jnp.allclose(self._dist(w=0.3).log_prob(v), jnp.logaddexp(lp0, lp1), atol=1e-5)

    def test_cdf_monotone_in_unit_interval(self):
        c = self._dist().cdf(jnp.linspace(-50.0, 200.0, 200))
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
        assert bool(jnp.all(jnp.diff(c) >= -1e-6))

    def test_weight_shifts_the_mean(self):
        lo = np.asarray(self._dist(w=0.9).sample(random.PRNGKey(0), (40000,)))
        hi = np.asarray(self._dist(w=0.1).sample(random.PRNGKey(0), (40000,)))
        assert lo.mean() < hi.mean()

    def test_sample_recovers_two_components(self):
        # A well-separated 0.5/0.5 mixture is clearly bimodal: both centers heavily
        # populated, the valley between them nearly empty.
        d = self._dist(w=0.5, loc0=30.0, scale0=3.0, loc1=90.0, scale1=3.0)
        draws = np.asarray(d.sample(random.PRNGKey(1), (40000,)))
        near_lo = float(np.mean(np.abs(draws - 30.0) < 6.0))
        near_hi = float(np.mean(np.abs(draws - 90.0) < 6.0))
        near_mid = float(np.mean(np.abs(draws - 60.0) < 6.0))
        assert near_lo > 0.3 and near_hi > 0.3 and near_mid < 0.02
