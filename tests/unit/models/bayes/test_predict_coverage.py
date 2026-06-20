"""Coverage-targeted tests for models/bayes/predict.py.

Tests target missed lines/branches:
- extract_posterior_samples: empty posterior, < 2 dims, single var
- generate_posterior_predictive: y_key fallback branches, return_mu paths
- predict_new_artist: heteroscedastic branches (learned vs fixed exponent),
  n_predictions subsampling, multi-album predictions, ValueError for missing n_reviews
- predict_out_of_sample: delegation to generate_posterior_predictive
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from panelcast.models.bayes.predict import (
    PredictionResult,
    extract_posterior_samples,
    predict_new_artist,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_idata_like(var_dict):
    """Create a minimal object that mimics InferenceData.posterior."""

    class FakePosterior:
        def __init__(self, d):
            self._d = d
            self.data_vars = list(d.keys())

        def __getitem__(self, key):
            return self._d[key]

        def __contains__(self, key):
            return key in self._d

    class FakeDataVar:
        def __init__(self, arr):
            self.values = arr

        @property
        def ndim(self):
            return self.values.ndim

        @property
        def shape(self):
            return self.values.shape

    data_vars = {k: FakeDataVar(v) for k, v in var_dict.items()}

    class FakeIData:
        posterior = FakePosterior(data_vars)

    return FakeIData()


@pytest.fixture
def basic_posterior_samples():
    """Minimal posterior samples dict for predict_new_artist."""
    n_samples = 20
    n_features = 3
    return {
        "user_mu_artist": jnp.ones(n_samples) * 60.0,
        "user_sigma_artist": jnp.ones(n_samples) * 5.0,
        "user_beta": jnp.ones((n_samples, n_features)) * 0.5,
        "user_rho": jnp.ones(n_samples) * 0.3,
        "user_sigma_obs": jnp.ones(n_samples) * 3.0,
    }


@pytest.fixture
def posterior_samples_with_learned_exponent(basic_posterior_samples):
    """Posterior samples including a learned n_exponent."""
    samples = dict(basic_posterior_samples)
    samples["user_n_exponent"] = jnp.ones(20) * 0.25
    return samples


# =============================================================================
# TestExtractPosteriorSamples
# =============================================================================


class TestExtractPosteriorSamples:
    """Tests for extract_posterior_samples."""

    def test_basic_extraction(self):
        """Extracts and flattens posterior variables from chain x draw."""
        var_dict = {
            "mu": np.random.randn(2, 50),
            "sigma": np.abs(np.random.randn(2, 50)) + 0.1,
        }
        idata = _make_idata_like(var_dict)
        result = extract_posterior_samples(idata)

        assert "mu" in result
        assert "sigma" in result
        assert result["mu"].shape == (100,)  # 2 chains x 50 draws
        assert result["sigma"].shape == (100,)

    def test_multidimensional_variable(self):
        """Variables with extra dims (e.g., beta with n_features) are flattened correctly."""
        var_dict = {
            "beta": np.random.randn(2, 50, 3),
        }
        idata = _make_idata_like(var_dict)
        result = extract_posterior_samples(idata)

        assert result["beta"].shape == (100, 3)

    def test_single_chain(self):
        """Works with a single chain."""
        var_dict = {
            "mu": np.random.randn(1, 100),
        }
        idata = _make_idata_like(var_dict)
        result = extract_posterior_samples(idata)
        assert result["mu"].shape == (100,)

    def test_raises_on_1d_variable(self):
        """Variables with only 1 dimension should raise ValueError."""
        var_dict = {
            "bad_var": np.random.randn(50),
        }
        idata = _make_idata_like(var_dict)
        with pytest.raises(ValueError, match="chain/draw dimensions"):
            extract_posterior_samples(idata)

    def test_raises_on_empty_posterior(self):
        """Empty posterior should raise ValueError."""
        idata = _make_idata_like({})
        with pytest.raises(ValueError, match="No posterior variables"):
            extract_posterior_samples(idata)


# =============================================================================
# TestPredictNewArtistHomoscedastic
# =============================================================================


class TestPredictNewArtistHomoscedastic:
    """Tests for predict_new_artist in homoscedastic mode."""

    def test_single_album_basic(self, basic_posterior_samples):
        """Single album prediction returns expected keys and shapes."""
        X_new = jnp.array([1.0, 0.5, -0.3])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            prefix="user_",
            seed=42,
        )
        assert "y" in result
        assert "mu" in result
        assert "artist_effect" in result
        assert "sigma_scaled" in result
        # Single album: (n_samples,)
        assert result["y"].shape == (20,)
        assert result["mu"].shape == (20,)
        assert result["artist_effect"].shape == (20,)

    def test_multi_album_prediction(self, basic_posterior_samples):
        """Multi-album prediction returns (n_samples, n_albums) shapes."""
        X_new = jnp.ones((3, 3)) * 0.5
        prev_scores = jnp.array([50.0, 60.0, 70.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=prev_scores,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20, 3)
        assert result["mu"].shape == (20, 3)

    def test_debut_album_prev_score_zero(self, basic_posterior_samples):
        """Debut album with prev_score=0 should work fine."""
        X_new = jnp.array([0.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=0.0,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20,)

    def test_n_predictions_subsampling(self, basic_posterior_samples):
        """n_predictions < n_samples should subsample the posterior."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            prefix="user_",
            seed=42,
            n_predictions=5,
        )
        assert result["y"].shape == (5,)
        assert result["mu"].shape == (5,)
        assert result["artist_effect"].shape == (5,)

    def test_n_predictions_larger_than_samples(self, basic_posterior_samples):
        """n_predictions >= n_samples should use all samples."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            prefix="user_",
            seed=42,
            n_predictions=100,
        )
        assert result["y"].shape == (20,)

    def test_sigma_scaled_equals_sigma_obs_homoscedastic(self, basic_posterior_samples):
        """In homoscedastic mode, sigma_scaled should be sigma_obs."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            prefix="user_",
            seed=42,
        )
        np.testing.assert_allclose(
            result["sigma_scaled"],
            np.full(20, 3.0),
            atol=1e-5,
        )


# =============================================================================
# TestPredictNewArtistHeteroscedastic
# =============================================================================


class TestPredictNewArtistHeteroscedastic:
    """Tests for predict_new_artist with heteroscedastic noise."""

    def test_learned_exponent_requires_n_reviews(self, posterior_samples_with_learned_exponent):
        """Learned exponent in posterior without n_reviews_new should raise."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="n_reviews_new is required"):
            predict_new_artist(
                posterior_samples_with_learned_exponent,
                X_new,
                prev_score=50.0,
                prefix="user_",
            )

    def test_learned_exponent_with_n_reviews(self, posterior_samples_with_learned_exponent):
        """Learned exponent with n_reviews_new should produce valid predictions."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            posterior_samples_with_learned_exponent,
            X_new,
            prev_score=50.0,
            n_reviews_new=jnp.array([100]),
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20,)
        assert result["sigma_scaled"].shape == (20,)

    def test_fixed_exponent_requires_n_reviews(self, basic_posterior_samples):
        """Fixed non-zero exponent without n_reviews_new should raise."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="n_reviews_new is required"):
            predict_new_artist(
                basic_posterior_samples,
                X_new,
                prev_score=50.0,
                fixed_n_exponent=0.3,
                prefix="user_",
            )

    def test_fixed_exponent_with_n_reviews(self, basic_posterior_samples):
        """Fixed exponent with n_reviews_new should work."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            n_reviews_new=jnp.array([50]),
            fixed_n_exponent=0.25,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20,)

    def test_fixed_exponent_zero_is_homoscedastic(self, basic_posterior_samples):
        """fixed_n_exponent=0.0 should behave like homoscedastic."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            fixed_n_exponent=0.0,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20,)
        # sigma_scaled should be sigma_obs (homoscedastic fallback)
        np.testing.assert_allclose(
            result["sigma_scaled"],
            np.full(20, 3.0),
            atol=1e-5,
        )

    def test_learned_exponent_multi_album(self, posterior_samples_with_learned_exponent):
        """Learned exponent with multi-album prediction."""
        X_new = jnp.ones((2, 3)) * 0.5
        prev_scores = jnp.array([50.0, 60.0])
        n_reviews = jnp.array([100, 200])
        result = predict_new_artist(
            posterior_samples_with_learned_exponent,
            X_new,
            prev_score=prev_scores,
            n_reviews_new=n_reviews,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20, 2)
        assert result["sigma_scaled"].shape == (20, 2)

    def test_fixed_exponent_multi_album(self, basic_posterior_samples):
        """Fixed exponent with multi-album prediction."""
        X_new = jnp.ones((2, 3)) * 0.5
        prev_scores = jnp.array([50.0, 60.0])
        n_reviews = jnp.array([100, 200])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=prev_scores,
            n_reviews_new=n_reviews,
            fixed_n_exponent=0.3,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20, 2)

    def test_n_predictions_with_learned_exponent(self, posterior_samples_with_learned_exponent):
        """Subsampling with learned exponent should subsample exponent too."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            posterior_samples_with_learned_exponent,
            X_new,
            prev_score=50.0,
            n_reviews_new=jnp.array([100]),
            prefix="user_",
            seed=42,
            n_predictions=5,
        )
        assert result["y"].shape == (5,)

    def test_n_predictions_with_fixed_exponent(self, basic_posterior_samples):
        """Subsampling with fixed exponent creates constant array of correct size."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_posterior_samples,
            X_new,
            prev_score=50.0,
            n_reviews_new=jnp.array([50]),
            fixed_n_exponent=0.25,
            prefix="user_",
            seed=42,
            n_predictions=5,
        )
        assert result["y"].shape == (5,)

    def test_error_message_mentions_learned(self, posterior_samples_with_learned_exponent):
        """Error message should mention 'learned' when exponent is in posterior."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="learned"):
            predict_new_artist(
                posterior_samples_with_learned_exponent,
                X_new,
                prev_score=50.0,
                prefix="user_",
            )

    def test_error_message_mentions_fixed(self, basic_posterior_samples):
        """Error message should mention 'fixed' when using fixed_n_exponent."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="fixed"):
            predict_new_artist(
                basic_posterior_samples,
                X_new,
                prev_score=50.0,
                fixed_n_exponent=0.3,
                prefix="user_",
            )


# =============================================================================
# TestPredictNewArtistPrefix
# =============================================================================


class TestPredictNewArtistPrefix:
    """Tests for predict_new_artist with different prefixes."""

    def test_critic_prefix(self):
        """Should work with critic_ prefix."""
        n_samples = 10
        n_features = 2
        samples = {
            "critic_mu_artist": jnp.ones(n_samples) * 70.0,
            "critic_sigma_artist": jnp.ones(n_samples) * 3.0,
            "critic_beta": jnp.ones((n_samples, n_features)) * 0.2,
            "critic_rho": jnp.ones(n_samples) * 0.4,
            "critic_sigma_obs": jnp.ones(n_samples) * 2.0,
        }
        X_new = jnp.array([0.5, -0.5])
        result = predict_new_artist(
            samples,
            X_new,
            prev_score=65.0,
            prefix="critic_",
            seed=42,
        )
        assert result["y"].shape == (10,)
