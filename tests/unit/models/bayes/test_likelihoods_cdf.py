"""CDF methods, RoundedDistribution, and interval_cdf discretization path.

Targets the dormant fallback paths and per-family CDF functions in
panelcast/models/bayes/likelihoods.py that are not exercised by the main
test suite.  Missing lines covered here:
  121-129  SplitNormal.cdf
  248-262  RoundedDistribution.__init__ / .sample / .log_prob
  304-336  _studentt_cdf, _normal_cdf, _skewnormal_cdf, _splitnormal_cdf, _mixture_cdf
  349      _emit_obs interval_cdf branch
  458      _beta_sample_obs target_transform guard
  482, 488 _beta_binomial_sample_obs target_transform / n_reviews guards
  543-547  _normal_predict_draws discretize branch
  556      _skew_studentt_predict_draws missing-site guard
  572      _skew_normal_predict_draws missing-site guard
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.handlers import seed, trace

from panelcast.models.bayes.likelihoods import (
    NormalMixture2,
    RoundedDistribution,
    SplitNormal,
    _mixture_cdf,
    _normal_cdf,
    _skewnormal_cdf,
    _splitnormal_cdf,
    _studentt_cdf,
)
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

N_OBS, N_FEAT, N_ART = 30, 2, 4


def _model_args(family: str, **prior_kw) -> dict:
    rng = np.random.default_rng(0)
    artist_idx = jnp.array([i % N_ART for i in range(N_OBS)], dtype=jnp.int32)
    album_seq = jnp.array([(i // N_ART) + 1 for i in range(N_OBS)], dtype=jnp.int32)
    return dict(
        artist_idx=artist_idx,
        album_seq=album_seq,
        prev_score=jnp.full(N_OBS, 70.0),
        X=jnp.asarray(rng.standard_normal((N_OBS, N_FEAT)), dtype=jnp.float32),
        n_artists=N_ART,
        max_seq=int(album_seq.max()),
        n_reviews=jnp.full(N_OBS, 50, dtype=jnp.int32),
        priors=PriorConfig(likelihood_family=family, **prior_kw),
        target_bounds=(0.0, 100.0),
        likelihood_df=4.0,
        ar_center=70.0,
    )


class TestSplitNormalCdf:
    """SplitNormal.cdf — lines 121-129."""

    def _dist(self, loc=70.0, sl=8.0, sr=5.0):
        return SplitNormal(loc, sl, sr)

    def test_cdf_at_loc_is_scale_ratio(self):
        d = self._dist(loc=0.0, sl=3.0, sr=7.0)
        # F(loc) = 2*sL/(sL+sR)*0.5 = sL/(sL+sR)
        expected = 3.0 / 10.0
        assert float(jnp.abs(d.cdf(jnp.array(0.0)) - expected)) < 1e-5

    def test_cdf_left_branch(self):
        d = self._dist()
        c = d.cdf(jnp.array(50.0))
        assert float(c) > 0.0
        assert float(c) < float(d.cdf(jnp.array(70.0)))

    def test_cdf_right_branch(self):
        d = self._dist()
        c_right = d.cdf(jnp.array(85.0))
        c_loc = d.cdf(jnp.array(70.0))
        assert float(c_right) > float(c_loc)

    def test_cdf_monotone(self):
        d = self._dist()
        xs = jnp.linspace(30.0, 110.0, 50)
        cs = d.cdf(xs)
        assert bool(jnp.all(jnp.diff(cs) >= -1e-6))
        assert float(cs.min()) >= 0.0
        assert float(cs.max()) <= 1.0

    def test_cdf_symmetric_when_equal_scales(self):
        d = SplitNormal(0.0, 5.0, 5.0)
        import jax.scipy.stats as jss
        v = jnp.linspace(-20.0, 20.0, 30)
        assert jnp.allclose(d.cdf(v), jss.norm.cdf(v, loc=0.0, scale=5.0), atol=1e-5)


class TestRoundedDistribution:
    """RoundedDistribution — lines 248-262."""

    def _make(self, mu=70.0, sigma=8.0):
        import jax.scipy.stats as jss
        base = dist.Normal(mu, sigma)
        # Use jax.scipy.stats.norm.cdf directly; jnp has no .erf
        cdf_fn = lambda v: jss.norm.cdf(v, loc=mu, scale=sigma)
        return RoundedDistribution(base, cdf_fn)

    def test_init_inherits_batch_shape(self):
        # lines 248-254
        rd = self._make()
        assert rd.batch_shape == ()
        assert rd.event_shape == ()

    def test_sample_returns_integers(self):
        # line 257
        rd = self._make()
        draws = np.asarray(rd.sample(random.PRNGKey(0), (1000,)))
        assert np.array_equal(draws, np.round(draws))

    def test_log_prob_finite_for_integers(self):
        # lines 259-262
        rd = self._make()
        vals = jnp.array([60.0, 70.0, 80.0, 90.0])
        lp = rd.log_prob(vals)
        assert bool(jnp.all(jnp.isfinite(lp)))

    def test_log_prob_uses_interval_mass(self):
        # log(F(k+0.5) - F(k-0.5)) should be finite and reasonable at the mode.
        mu, sigma = 70.0, 15.0
        rd = self._make(mu, sigma)
        lp = float(rd.log_prob(jnp.array(70.0)))
        assert np.isfinite(lp)
        assert lp > -5.0

    def test_log_prob_clamps_tiny_tails(self):
        # At extreme values the mass underflows; we should still get a finite (not -inf)
        # log-prob thanks to the _TINY floor.
        rd = self._make(mu=70.0, sigma=8.0)
        lp = float(rd.log_prob(jnp.array(1000.0)))
        assert np.isfinite(lp)

    def test_batch_base(self):
        # batch_shape propagates from base
        base = dist.Normal(jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0]))
        rd = RoundedDistribution(base, lambda v: dist.Normal(0.0, 1.0).cdf(v))
        assert rd.batch_shape == (2,)


class TestFamilyCdfs:
    """Stand-alone CDF helper functions — lines 304-336."""

    def test_studentt_cdf_at_loc(self):
        # P(T <= mu) = 0.5 by symmetry
        v = jnp.array(70.0)
        c = _studentt_cdf(v, mu=70.0, sigma=8.0, df=4.0, params={})
        assert float(jnp.abs(c - 0.5)) < 1e-5

    def test_studentt_cdf_monotone(self):
        xs = jnp.linspace(40.0, 100.0, 30)
        cs = _studentt_cdf(xs, mu=70.0, sigma=8.0, df=4.0, params={})
        assert bool(jnp.all(jnp.diff(cs) >= 0.0))

    def test_studentt_cdf_left_is_small(self):
        c = _studentt_cdf(jnp.array(30.0), mu=70.0, sigma=8.0, df=4.0, params={})
        assert float(c) < 0.05

    def test_normal_cdf_at_loc(self):
        c = _normal_cdf(jnp.array(70.0), mu=70.0, sigma=8.0, df=4.0, params={})
        assert float(jnp.abs(c - 0.5)) < 1e-5

    def test_normal_cdf_monotone(self):
        xs = jnp.linspace(30.0, 110.0, 40)
        cs = _normal_cdf(xs, mu=70.0, sigma=8.0, df=4.0, params={})
        assert bool(jnp.all(jnp.diff(cs) >= 0.0))

    def test_skewnormal_cdf_zero_skew_matches_normal(self):
        import jax.scipy.stats as jss
        xs = jnp.linspace(30.0, 110.0, 25)
        params = {"skewness": jnp.array(0.0), "tailweight": 1.0}
        cs = _skewnormal_cdf(xs, mu=70.0, sigma=8.0, df=4.0, params=params)
        expected = jss.norm.cdf(xs, loc=70.0, scale=8.0)
        assert jnp.allclose(cs, expected, atol=1e-4)

    def test_skewnormal_cdf_nonzero_skew(self):
        xs = jnp.linspace(40.0, 100.0, 20)
        params = {"skewness": jnp.array(-0.5), "tailweight": 1.2}
        cs = _skewnormal_cdf(xs, mu=70.0, sigma=8.0, df=4.0, params=params)
        assert float(cs.min()) >= 0.0
        assert float(cs.max()) <= 1.0

    def test_splitnormal_cdf_uses_split_normal(self):
        xs = jnp.linspace(40.0, 100.0, 20)
        log_ratio = jnp.array(0.5)
        params = {"split_log_ratio": log_ratio}
        cs = _splitnormal_cdf(xs, mu=70.0, sigma=8.0, df=4.0, params=params)
        assert bool(jnp.all(jnp.diff(cs) >= -1e-6))

    def test_mixture_cdf_is_mixture_of_normals(self):
        xs = jnp.linspace(30.0, 110.0, 25)
        params = {
            "mix_sep": jnp.array(1.0),
            "mix_weight": jnp.array(0.4),
            "mix_log_scale_ratio": jnp.array(0.0),
        }
        cs = _mixture_cdf(xs, mu=70.0, sigma=8.0, df=4.0, params=params)
        assert float(cs.min()) >= 0.0
        assert float(cs.max()) <= 1.0
        assert bool(jnp.all(jnp.diff(cs) >= -1e-6))

    def test_mixture_cdf_matches_normal_mixture_object(self):
        mu, sigma = 70.0, 8.0
        params = {
            "mix_sep": jnp.array(1.5),
            "mix_weight": jnp.array(0.3),
            "mix_log_scale_ratio": jnp.array(0.2),
        }
        xs = jnp.linspace(50.0, 90.0, 10)
        w = params["mix_weight"]
        delta = params["mix_sep"]
        log_sr = params["mix_log_scale_ratio"]
        loc0 = mu - (1.0 - w) * delta * sigma
        loc1 = mu + w * delta * sigma
        scale0 = sigma * jnp.exp(0.5 * log_sr)
        scale1 = sigma * jnp.exp(-0.5 * log_sr)
        expected = NormalMixture2(loc0, scale0, loc1, scale1, w).cdf(xs)
        cs = _mixture_cdf(xs, mu=mu, sigma=sigma, df=4.0, params=params)
        assert jnp.allclose(cs, expected, atol=1e-5)


class TestIntervalCdfBranch:
    """The dormant interval_cdf discretization path — line 349."""

    def test_emit_obs_interval_cdf_mode_uses_rounded_dist(self, monkeypatch):
        import panelcast.models.bayes.likelihoods as lik_mod
        monkeypatch.setattr(lik_mod, "_DISCRETIZE_MODE", "interval_cdf")

        model = make_score_model("user")
        args = _model_args("normal", discretize_observation=True)
        tr = trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **args)
        site = tr["user_y"]
        # With interval_cdf the fn is RoundedDistribution, not DequantizedDistribution.
        assert isinstance(site["fn"], lik_mod.RoundedDistribution)

    def test_interval_cdf_log_prob_finite(self, monkeypatch):
        import panelcast.models.bayes.likelihoods as lik_mod
        monkeypatch.setattr(lik_mod, "_DISCRETIZE_MODE", "interval_cdf")

        model = make_score_model("user")
        draws = np.round(np.clip(np.random.default_rng(3).normal(70, 8, N_OBS), 0, 100))
        y_int = jnp.asarray(draws, dtype=jnp.float32)
        args = _model_args("normal", discretize_observation=True)
        tr = trace(seed(model, random.PRNGKey(1))).get_trace(y=y_int, **args)
        site = tr["user_y"]
        lp = np.asarray(site["fn"].log_prob(site["value"]))
        assert np.isfinite(lp).all()


class TestBetaSampleObsGuards:
    """Guards in _beta_sample_obs and _beta_binomial_sample_obs — lines 458, 482, 488."""

    def test_beta_bad_target_transform_raises(self):
        # line 458
        model = make_score_model("user")
        args = _model_args("beta", target_transform="log")
        with pytest.raises(ValueError, match="target_transform"):
            trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **args)

    def test_beta_binomial_bad_target_transform_raises(self):
        # line 482
        model = make_score_model("user")
        args = _model_args("beta_binomial", target_transform="log")
        with pytest.raises(ValueError, match="target_transform"):
            trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **args)

    def test_beta_binomial_no_n_reviews_raises(self):
        # line 488
        model = make_score_model("user")
        args = _model_args("beta_binomial")
        args["n_reviews"] = None
        with pytest.raises(ValueError, match="n_reviews"):
            trace(seed(model, random.PRNGKey(0))).get_trace(y=None, **args)


class TestNormalPredictDrawsDiscretize:
    """_normal_predict_draws discretize branch — lines 543-547."""

    def test_normal_predict_discretize_returns_integers(self):
        from panelcast.models.bayes.predict import predict_new_entity
        rng = np.random.default_rng(9)
        post = {
            "user_mu_artist": jnp.asarray(rng.normal(0, 1, 40)),
            "user_sigma_artist": jnp.asarray(np.abs(rng.normal(0.5, 0.1, 40))),
            "user_beta": jnp.asarray(rng.normal(0, 1, (40, N_FEAT))),
            "user_rho": jnp.asarray(rng.normal(0, 0.2, 40)),
            "user_sigma_obs": jnp.asarray(np.abs(rng.normal(8, 1, 40))),
        }
        out = predict_new_entity(
            post,
            X_new=jnp.zeros(N_FEAT),
            prev_score=70.0,
            likelihood_family="normal",
            target_bounds=(0.0, 100.0),
            ar_center=70.0,
            discretize_observation=True,
        )
        y = np.asarray(out["y"])
        assert np.array_equal(y, np.round(y))


class TestSkewStudenttPredictMissingSite:
    """_skew_studentt_predict_draws missing site — line 556."""

    def test_skew_studentt_predict_missing_skewness_raises(self):
        from panelcast.models.bayes.predict import predict_new_entity
        rng = np.random.default_rng(10)
        post = {
            "user_mu_artist": jnp.asarray(rng.normal(0, 1, 20)),
            "user_sigma_artist": jnp.asarray(np.abs(rng.normal(0.5, 0.1, 20))),
            "user_beta": jnp.asarray(rng.normal(0, 1, (20, N_FEAT))),
            "user_rho": jnp.asarray(rng.normal(0, 0.2, 20)),
            "user_sigma_obs": jnp.asarray(np.abs(rng.normal(8, 1, 20))),
        }
        with pytest.raises(ValueError, match="skewness"):
            predict_new_entity(
                post,
                X_new=jnp.zeros(N_FEAT),
                prev_score=70.0,
                likelihood_family="skew_studentt",
                target_bounds=(0.0, 100.0),
                ar_center=70.0,
            )


class TestSkewNormalPredictMissingSite:
    """_skew_normal_predict_draws missing site — line 572."""

    def test_skew_normal_predict_missing_skewness_raises(self):
        from panelcast.models.bayes.predict import predict_new_entity
        rng = np.random.default_rng(11)
        post = {
            "user_mu_artist": jnp.asarray(rng.normal(0, 1, 20)),
            "user_sigma_artist": jnp.asarray(np.abs(rng.normal(0.5, 0.1, 20))),
            "user_beta": jnp.asarray(rng.normal(0, 1, (20, N_FEAT))),
            "user_rho": jnp.asarray(rng.normal(0, 0.2, 20)),
            "user_sigma_obs": jnp.asarray(np.abs(rng.normal(8, 1, 20))),
        }
        with pytest.raises(ValueError, match="skewness"):
            predict_new_entity(
                post,
                X_new=jnp.zeros(N_FEAT),
                prev_score=70.0,
                likelihood_family="skew_normal",
                target_bounds=(0.0, 100.0),
                ar_center=70.0,
            )
