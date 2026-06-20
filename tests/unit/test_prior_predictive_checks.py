"""Tests for the score-scale prior-predictive plausibility checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from panelcast.evaluation.prior_predictive import run_prior_predictive
from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.priors import PriorConfig, priors_for_transform
from panelcast.models.bayes.transforms import get_transform


def _model_args(priors: PriorConfig, n_obs: int = 40) -> dict:
    rng = np.random.default_rng(0)
    n_artists = 5
    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
        "prev_score": jnp.zeros(n_obs),
        "X": jnp.asarray(rng.normal(size=(n_obs, 2)), dtype=jnp.float32),
        "y": None,
        "n_artists": n_artists,
        "max_seq": (n_obs + n_artists - 1) // n_artists,
        "priors": priors,
    }


def _well_located_model(X, y=None, **kwargs):
    """Toy model whose prior predictive is plausible for (0, 100) scores:
    level near 75, spread ~8, mild negative skew from the ceiling."""
    mu = numpyro.sample("toy_mu", dist.Normal(75.0, 2.0))
    sd = numpyro.sample("toy_sd", dist.HalfNormal(10.0))
    with numpyro.plate("toy_obs", X.shape[0]):
        numpyro.sample("toy_y", dist.TruncatedNormal(mu, sd + 1.0, low=0.0, high=92.0), obs=y)


class TestPlausibilityChecks:
    def test_well_located_prior_passes(self):
        result = run_prior_predictive(
            _well_located_model,
            {"X": jnp.zeros((50, 2)), "y": None},
            n_samples=300,
            seed=1,
        )
        assert result.checks_passed, result.informational_flags
        assert result.checks is not None
        assert result.checks["mean"]["passed"]
        assert result.checks["sd"]["passed"]
        assert result.checks["skewness"]["passed"]
        assert result.informational_flags == []

    def test_huge_beta_scale_fails(self):
        """beta_scale=50 explodes prior-predictive spread: flags raised."""
        priors = PriorConfig(beta_scale=50.0)
        result = run_prior_predictive(
            user_score_model,
            _model_args(priors),
            n_samples=200,
            seed=2,
        )
        assert not result.checks_passed
        assert result.informational_flags is not None
        assert any("sd" in f or "mean" in f for f in result.informational_flags)

    def test_ranges_scale_with_bounds(self):
        """For (0, 10) bounds the gates shrink proportionately."""
        result = run_prior_predictive(
            _well_located_model,
            {"X": jnp.zeros((50, 2)), "y": None},
            n_samples=100,
            seed=3,
            score_bounds=(0.0, 10.0),
        )
        assert result.checks is not None
        assert result.checks["mean"]["low"] == pytest.approx(6.0)
        assert result.checks["mean"]["high"] == pytest.approx(9.0)
        assert result.checks["sd"]["low"] == pytest.approx(0.5)
        assert result.checks["sd"]["high"] == pytest.approx(2.0)

    def test_summary_includes_skewness(self):
        result = run_prior_predictive(
            _well_located_model,
            {"X": jnp.zeros((30, 2)), "y": None},
            n_samples=100,
            seed=4,
        )
        assert "skewness" in result.summary
        assert np.isfinite(result.summary["skewness"])


class TestTransformBackConversion:
    def test_logit_draws_reported_on_score_scale(self):
        """Under offset_logit the raw draws live on the logit scale; the
        result must report score-scale statistics and bounds fractions."""
        priors = priors_for_transform("offset_logit")
        transform = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        result = run_prior_predictive(
            user_score_model,
            _model_args(priors),
            n_samples=200,
            seed=5,
            transform=transform,
        )
        # Back-transformed draws cannot leave the bounds extension, so the
        # legacy in-bounds fraction is meaningful (near 1) instead of the
        # near-0 it would be on the logit scale.
        assert result.fraction_in_bounds > 0.95
        assert -0.5 <= result.summary["min"] <= result.summary["max"] <= 100.5

    def test_without_transform_logit_draws_stay_model_scale(self):
        """Control: omitting the transform reports logit-scale values, which
        fail the (0, 100) bounds check almost everywhere."""
        priors = priors_for_transform("offset_logit")
        result = run_prior_predictive(
            user_score_model,
            _model_args(priors),
            n_samples=100,
            seed=6,
        )
        assert result.summary["mean"] < 60.0
        assert result.fraction_in_bounds < 0.9
