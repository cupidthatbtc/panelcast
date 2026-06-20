"""Expanded tests for prior predictive module."""

import numpy as np
import pytest

from panelcast.evaluation.prior_predictive import (
    PriorPredictiveResult,
    generate_prior_justification_text,
)
from panelcast.models.bayes.priors import PriorConfig


class TestPriorPredictiveResultDataclass:
    """Tests for PriorPredictiveResult dataclass."""

    def test_y_samples_shape(self):
        y = np.random.default_rng(42).normal(50, 10, (50, 30))
        result = PriorPredictiveResult(
            y_samples=y,
            summary={
                "mean": 50.0,
                "sd": 10.0,
                "q2.5": 30.0,
                "q97.5": 70.0,
                "min": 10.0,
                "max": 90.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.98,
            n_samples=50,
            n_obs_original=30,
            max_obs=2000,
            sampled_indices=None,
            seed=42,
        )
        assert result.y_samples.shape == (50, 30)

    def test_reasonable_true(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=0,
        )
        assert result.reasonable is True

    def test_reasonable_false(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=False,
            bounds=(0, 100),
            fraction_in_bounds=0.2,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=0,
        )
        assert result.reasonable is False

    def test_bounds_tuple(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=0,
        )
        assert result.bounds == (0, 100)
        assert result.bounds[0] == 0
        assert result.bounds[1] == 100

    def test_summary_keys(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={
                "mean": 50.0,
                "sd": 15.0,
                "q2.5": 20.0,
                "q97.5": 80.0,
                "min": 5.0,
                "max": 95.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.95,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=42,
        )
        assert "mean" in result.summary
        assert "sd" in result.summary
        assert "q2.5" in result.summary
        assert "q97.5" in result.summary

    def test_with_sampled_indices(self):
        indices = np.array([0, 5, 10, 15, 20])
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
            n_samples=10,
            n_obs_original=100,
            max_obs=5,
            sampled_indices=indices,
            seed=42,
        )
        assert result.sampled_indices is not None
        assert len(result.sampled_indices) == 5


class TestGeneratePriorJustificationText:
    """Extended tests for generate_prior_justification_text."""

    def test_returns_string(self):
        text = generate_prior_justification_text(PriorConfig())
        assert isinstance(text, str)

    def test_contains_all_prior_names(self):
        text = generate_prior_justification_text(PriorConfig())
        for name in ["mu_artist", "sigma_artist", "sigma_rw", "rho", "beta", "sigma_obs"]:
            assert name in text

    def test_custom_priors_reflected(self):
        config = PriorConfig(mu_artist_loc=70.0, mu_artist_scale=5.0)
        text = generate_prior_justification_text(config)
        assert "70.0" in text
        assert "5.0" in text

    def test_no_ppc_section_without_result(self):
        text = generate_prior_justification_text(PriorConfig())
        assert "Prior Predictive Check" not in text

    def test_no_sensitivity_section_without_data(self):
        text = generate_prior_justification_text(PriorConfig())
        assert "Sensitivity" not in text

    def test_length_reasonable(self):
        text = generate_prior_justification_text(PriorConfig())
        # Should be a meaningful length, not empty
        assert len(text) > 100
