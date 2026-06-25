"""Tests for the skew/bounded likelihood families and prediction dispatch."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from panelcast.models.bayes.model import SinhArcsinhTransform, make_score_model
from panelcast.models.bayes.predict import predict_new_artist
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


class TestPredictNewArtistDispatch:
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
        out = predict_new_artist(
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
        out = predict_new_artist(
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
            predict_new_artist(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="beta",
            )

    def test_skew_normal_prediction_runs(self):
        post = self._posterior(
            extra={"user_skewness": jnp.asarray(np.random.default_rng(4).normal(-0.5, 0.1, 40))}
        )
        out = predict_new_artist(
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
        out = predict_new_artist(
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
            predict_new_artist(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="split_normal",
            )

    def test_discretized_prediction_is_integer(self):
        out = predict_new_artist(
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
        out = predict_new_artist(
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
            predict_new_artist(
                self._posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                n_reviews_new=jnp.asarray(50),
                likelihood_family="beta_binomial",
            )

    def test_beta_binomial_without_n_reviews_raises(self):
        with pytest.raises(ValueError, match="n_reviews"):
            predict_new_artist(
                self._bb_posterior(),
                X_new=jnp.zeros(N_FEATURES),
                prev_score=70.0,
                likelihood_family="beta_binomial",
            )


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
