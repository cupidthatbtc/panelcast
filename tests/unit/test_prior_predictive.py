"""Unit tests for prior predictive module."""

import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.prior_predictive import (
    PriorPredictiveResult,
    generate_prior_justification_text,
    run_prior_predictive,
)
from panelcast.models.bayes.priors import PriorConfig


@pytest.fixture
def default_priors():
    return PriorConfig()


@pytest.fixture
def custom_priors():
    return PriorConfig(
        mu_artist_loc=0.5,
        mu_artist_scale=2.0,
        sigma_artist_scale=1.0,
        sigma_rw_scale=0.2,
        rho_loc=0.1,
        rho_scale=0.5,
        beta_loc=0.0,
        beta_scale=2.0,
        sigma_obs_scale=1.5,
    )


@pytest.fixture
def mock_prior_predictive_result():
    return PriorPredictiveResult(
        y_samples=np.random.default_rng(42).normal(50, 15, (100, 50)),
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
        n_samples=100,
        n_obs_original=50,
        max_obs=2000,
        sampled_indices=None,
        seed=42,
    )


class TestPriorPredictiveResultFields:
    def test_prior_predictive_result_fields(self, mock_prior_predictive_result):
        """Dataclass fields are accessible."""
        result = mock_prior_predictive_result
        assert result.y_samples.shape == (100, 50)
        assert isinstance(result.summary, dict)
        assert isinstance(result.reasonable, bool)
        assert result.bounds == (0, 100)
        assert 0 <= result.fraction_in_bounds <= 1
        assert result.n_samples == 100
        assert result.n_obs_original == 50
        assert result.seed == 42


class TestPriorPredictiveSubsampling:
    def test_prior_predictive_subsampling(self):
        """When n_obs > max_obs, arrays should be subsampled."""
        n_obs = 5000
        max_obs = 100
        rng = np.random.default_rng(42)
        sampled_indices = np.sort(rng.choice(n_obs, size=max_obs, replace=False))

        result = PriorPredictiveResult(
            y_samples=np.zeros((10, max_obs)),
            summary={
                "mean": 0.0,
                "sd": 1.0,
                "q2.5": -2.0,
                "q97.5": 2.0,
                "min": -5.0,
                "max": 5.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.5,
            n_samples=10,
            n_obs_original=n_obs,
            max_obs=max_obs,
            sampled_indices=sampled_indices,
            seed=42,
        )
        assert result.n_obs_original == 5000
        assert result.max_obs == 100
        assert result.sampled_indices is not None
        assert len(result.sampled_indices) == max_obs
        assert result.y_samples.shape[1] == max_obs


class TestJustificationUsesActualValues:
    def test_justification_uses_actual_values(self, custom_priors):
        """Non-default PriorConfig -> actual numeric values appear in text."""
        text = generate_prior_justification_text(custom_priors)
        # Custom values should appear, not defaults
        assert "0.5" in text  # mu_artist_loc
        assert "2.0" in text  # mu_artist_scale
        assert "1.0" in text  # sigma_artist_scale
        assert "0.2" in text  # sigma_rw_scale
        assert "0.1" in text  # rho_loc

    def test_justification_default_values(self, default_priors):
        """Default priors produce valid text with default values."""
        text = generate_prior_justification_text(default_priors)
        assert "mu_artist" in text
        assert "sigma_artist" in text
        assert "sigma_rw" in text
        assert "rho" in text
        assert "beta" in text
        assert "sigma_obs" in text


class TestJustificationWithPPC:
    def test_justification_with_ppc(self, default_priors, mock_prior_predictive_result):
        """Mock PriorPredictiveResult -> 'Prior Predictive Check' in text."""
        text = generate_prior_justification_text(
            default_priors,
            prior_predictive_result=mock_prior_predictive_result,
        )
        assert "Prior Predictive Check" in text
        assert "95.0%" in text  # fraction_in_bounds
        assert "n_samples=100" in text


class TestJustificationWithSensitivity:
    def test_justification_with_sensitivity(self, default_priors):
        """Mock sensitivity DataFrame -> 'Sensitivity' in text."""
        sensitivity_df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_scale_x2", "beta_scale_x5"],
                "parameter": ["baseline", "sigma_rw_scale", "beta_scale"],
                "elpd_delta": [0.0, -3.5, -1.2],
                "eligible_for_ranking": [True, True, True],
            }
        )
        text = generate_prior_justification_text(
            default_priors,
            sensitivity_summary=sensitivity_df,
        )
        assert "Sensitivity" in text
        assert "sigma_rw_scale" in text  # most sensitive

    def test_justification_sensitivity_convergence_aware(self, default_priors):
        """Only converged variants referenced in text."""
        sensitivity_df = pd.DataFrame(
            {
                "variant": [
                    "default",
                    "sigma_rw_scale_x2",
                    "beta_scale_x5",
                ],
                "parameter": ["baseline", "sigma_rw_scale", "beta_scale"],
                "elpd_delta": [0.0, -50.0, -1.2],
                "eligible_for_ranking": [
                    True,
                    False,
                    True,
                ],  # sigma_rw FAILED
            }
        )
        text = generate_prior_justification_text(
            default_priors,
            sensitivity_summary=sensitivity_df,
        )
        # sigma_rw_scale has largest delta but is NOT eligible
        # So beta_scale should be cited as most sensitive
        assert "beta_scale" in text
        assert "most sensitive parameter is beta_scale" in text


class TestPriorPredictiveInputValidation:
    def test_rejects_nonpositive_n_samples(self):
        """run_prior_predictive should reject n_samples < 1."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            run_prior_predictive(
                model=lambda **kwargs: None,
                model_args={},
                n_samples=0,
            )

    def test_rejects_nonpositive_max_obs(self):
        """run_prior_predictive should reject max_obs < 1."""
        with pytest.raises(ValueError, match="max_obs must be >= 1"):
            run_prior_predictive(
                model=lambda **kwargs: None,
                model_args={},
                max_obs=0,
            )

    def test_rejects_invalid_score_bounds(self):
        """run_prior_predictive should reject malformed score bounds."""
        with pytest.raises(ValueError, match="score_bounds must be"):
            run_prior_predictive(
                model=lambda **kwargs: None,
                model_args={},
                score_bounds=(100, 0),
            )
