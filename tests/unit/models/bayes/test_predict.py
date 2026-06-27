"""Coverage-targeted tests for models/bayes/predict.py.

Tests target missed lines/branches:
- extract_posterior_samples: empty posterior, < 2 dims, single var
- generate_posterior_predictive: y_key fallback branches, return_mu paths
- predict_new_entity: heteroscedastic branches (learned vs fixed exponent),
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
    generate_posterior_predictive,
    predict_new_entity,
    predict_out_of_sample,
)


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
    """Minimal posterior samples dict for predict_new_entity."""
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


class TestPredictNewEntityHomoscedastic:
    """Tests for predict_new_entity in homoscedastic mode."""

    def test_single_album_basic(self, basic_posterior_samples):
        """Single album prediction returns expected keys and shapes."""
        X_new = jnp.array([1.0, 0.5, -0.3])
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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


class TestPredictNewEntityHeteroscedastic:
    """Tests for predict_new_entity with heteroscedastic noise."""

    def test_learned_exponent_requires_n_reviews(self, posterior_samples_with_learned_exponent):
        """Learned exponent in posterior without n_reviews_new should raise."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="n_reviews_new is required"):
            predict_new_entity(
                posterior_samples_with_learned_exponent,
                X_new,
                prev_score=50.0,
                prefix="user_",
            )

    def test_learned_exponent_with_n_reviews(self, posterior_samples_with_learned_exponent):
        """Learned exponent with n_reviews_new should produce valid predictions."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_entity(
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
            predict_new_entity(
                basic_posterior_samples,
                X_new,
                prev_score=50.0,
                fixed_n_exponent=0.3,
                prefix="user_",
            )

    def test_fixed_exponent_with_n_reviews(self, basic_posterior_samples):
        """Fixed exponent with n_reviews_new should work."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
        result = predict_new_entity(
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
            predict_new_entity(
                posterior_samples_with_learned_exponent,
                X_new,
                prev_score=50.0,
                prefix="user_",
            )

    def test_error_message_mentions_fixed(self, basic_posterior_samples):
        """Error message should mention 'fixed' when using fixed_n_exponent."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="fixed"):
            predict_new_entity(
                basic_posterior_samples,
                X_new,
                prev_score=50.0,
                fixed_n_exponent=0.3,
                prefix="user_",
            )


class TestPredictNewEntityPrefix:
    """Tests for predict_new_entity with different prefixes."""

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
        result = predict_new_entity(
            samples,
            X_new,
            prev_score=65.0,
            prefix="critic_",
            seed=42,
        )
        assert result["y"].shape == (10,)


# --- from unit/models/bayes/test_predict_expanded.py ---


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_y_accessible(self):
        y = jnp.ones((10, 5))
        result = PredictionResult(y=y)
        assert result.y.shape == (10, 5)

    def test_mu_default_none(self):
        y = jnp.ones((10, 5))
        result = PredictionResult(y=y)
        assert result.mu is None

    def test_mu_provided(self):
        y = jnp.ones((10, 5))
        mu = jnp.zeros((10, 5))
        result = PredictionResult(y=y, mu=mu)
        assert result.mu is not None
        assert result.mu.shape == (10, 5)

    def test_y_is_jax_array(self):
        y = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = PredictionResult(y=y)
        assert hasattr(result.y, "device")  # JAX arrays have device attribute

    def test_mutable(self):
        """PredictionResult is not frozen (contains mutable JAX arrays)."""
        y = jnp.ones((10, 5))
        result = PredictionResult(y=y)
        result.mu = jnp.zeros((10, 5))
        assert result.mu is not None

    def test_numpy_y(self):
        """Can also hold numpy arrays (for flexibility)."""
        y = np.ones((10, 5))
        result = PredictionResult(y=y)
        assert result.y.shape == (10, 5)

    def test_different_shapes(self):
        y = jnp.ones((100, 50))
        result = PredictionResult(y=y)
        assert result.y.shape[0] == 100
        assert result.y.shape[1] == 50

    def test_single_sample(self):
        y = jnp.ones((1, 10))
        result = PredictionResult(y=y)
        assert result.y.shape == (1, 10)


# --- from unit/models/bayes/test_predict_new.py ---


def _make_idata_like_new(var_dict):
    """Minimal InferenceData-like object."""

    class FakeDataVar:
        def __init__(self, arr):
            self.values = arr

        @property
        def ndim(self):
            return self.values.ndim

        @property
        def shape(self):
            return self.values.shape

    class FakePosterior:
        def __init__(self, d):
            self._d = d
            self.data_vars = list(d.keys())

        def __getitem__(self, key):
            return self._d[key]

    data_vars = {k: FakeDataVar(v) for k, v in var_dict.items()}

    class FakeIData:
        posterior = FakePosterior(data_vars)

    return FakeIData()


def _make_mock_mcmc(predictions_dict, samples_dict=None):
    """Create a mock MCMC object that returns given predictions."""
    mock_mcmc = MagicMock()
    if samples_dict is None:
        samples_dict = {"user_beta": jnp.ones((10, 3))}
    mock_mcmc.get_samples.return_value = samples_dict
    return mock_mcmc, predictions_dict


class TestGeneratePosteriorPredictive:
    """Tests for generate_posterior_predictive with mocked MCMC."""

    def test_basic_execution(self, monkeypatch):
        """Basic execution with y_key ending in '_y'."""
        n_samples, n_obs = 10, 5
        y_pred = jnp.ones((n_samples, n_obs)) * 50.0

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"user_beta": jnp.ones((n_samples, 3))}

        # Mock Predictive
        class FakePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"user_y": y_pred}

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", FakePredictive)

        model_args = {
            "artist_idx": np.arange(n_obs),
            "y": np.ones(n_obs) * 60,
        }

        result = generate_posterior_predictive(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            model_args=model_args,
            seed=1,
        )

        assert isinstance(result, PredictionResult)
        assert result.y.shape == (n_samples, n_obs)
        assert result.mu is None  # return_mu=False by default

    def test_return_mu_true(self, monkeypatch):
        """return_mu=True extracts mu predictions."""
        n_samples, n_obs = 10, 5
        y_pred = jnp.ones((n_samples, n_obs)) * 50.0
        mu_pred = jnp.ones((n_samples, n_obs)) * 48.0

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"user_beta": jnp.ones((n_samples, 3))}

        class FakePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"user_y": y_pred, "user_mu": mu_pred}

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", FakePredictive)

        model_args = {"y": np.ones(n_obs) * 60}

        result = generate_posterior_predictive(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            model_args=model_args,
            seed=1,
            return_mu=True,
        )

        assert result.mu is not None
        assert result.mu.shape == (n_samples, n_obs)

    def test_return_mu_false_ignores_mu_key(self, monkeypatch):
        """return_mu=False should ignore mu key even if present."""
        n_samples, n_obs = 10, 5

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"beta": jnp.ones((n_samples, 3))}

        class FakePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                return {
                    "user_y": jnp.ones((n_samples, n_obs)) * 50.0,
                    "user_mu": jnp.ones((n_samples, n_obs)) * 48.0,
                }

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", FakePredictive)

        result = generate_posterior_predictive(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            model_args={"y": np.ones(n_obs)},
            seed=1,
            return_mu=False,
        )

        assert result.mu is None

    def test_y_key_fallback_to_plain_y(self, monkeypatch):
        """When no key ends in '_y', fall back to plain 'y'."""
        n_samples, n_obs = 10, 5

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"beta": jnp.ones((n_samples, 3))}

        class FakePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                return {
                    "y": jnp.ones((n_samples, n_obs)) * 50.0,
                    "mu": jnp.ones((n_samples, n_obs)),
                }

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", FakePredictive)

        result = generate_posterior_predictive(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            model_args={"y": np.ones(n_obs)},
            seed=1,
        )

        assert result.y.shape == (n_samples, n_obs)

    def test_y_key_fallback_to_first_key(self, monkeypatch):
        """When no '_y' or 'y' key, fall back to first key."""
        n_samples, n_obs = 10, 5

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"beta": jnp.ones((n_samples, 3))}

        class FakePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"predictions": jnp.ones((n_samples, n_obs)) * 50.0}

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", FakePredictive)

        result = generate_posterior_predictive(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            model_args={"y": np.ones(n_obs)},
            seed=1,
        )

        assert result.y.shape == (n_samples, n_obs)

    def test_sets_y_to_none(self, monkeypatch):
        """Model args should have y=None for prediction."""
        captured = {}

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"beta": jnp.ones((5, 3))}

        class CapturePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                captured.update(kwargs)
                return {"obs_y": jnp.ones((5, 3)) * 50.0}

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", CapturePredictive)

        generate_posterior_predictive(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            model_args={"y": np.ones(3) * 60, "X": np.ones((3, 2))},
            seed=1,
        )

        assert captured["y"] is None
        assert captured["X"] is not None


class TestPredictOutOfSample:
    """Tests for predict_out_of_sample delegation."""

    def test_delegates_to_generate_posterior_predictive(self, monkeypatch):
        """predict_out_of_sample should call generate_posterior_predictive."""
        n_samples, n_obs = 10, 5

        mock_mcmc = MagicMock()
        mock_mcmc.get_samples.return_value = {"user_beta": jnp.ones((n_samples, 3))}

        class FakePredictive:
            def __init__(self, model, posterior_samples, batch_ndims):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"user_y": jnp.ones((n_samples, n_obs)) * 50.0}

        import panelcast.models.bayes.predict as pred_mod

        monkeypatch.setattr(pred_mod, "Predictive", FakePredictive)

        new_model_args = {
            "artist_idx": np.arange(n_obs),
            "y": None,
        }

        result = predict_out_of_sample(
            model=lambda **kw: None,
            mcmc=mock_mcmc,
            new_model_args=new_model_args,
            seed=2,
        )

        assert isinstance(result, PredictionResult)
        assert result.y.shape == (n_samples, n_obs)
        assert result.mu is None  # predict_out_of_sample doesn't return mu


@pytest.fixture
def basic_samples():
    """Minimal posterior samples."""
    n = 20
    return {
        "user_mu_artist": jnp.ones(n) * 60.0,
        "user_sigma_artist": jnp.ones(n) * 5.0,
        "user_beta": jnp.ones((n, 3)) * 0.5,
        "user_rho": jnp.ones(n) * 0.3,
        "user_sigma_obs": jnp.ones(n) * 3.0,
    }


class TestPredictNewEntityFixedNone:
    """Cover fixed_n_exponent=None explicitly."""

    def test_none_fixed_exponent_is_homoscedastic(self, basic_samples):
        """fixed_n_exponent=None should be homoscedastic."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_entity(
            basic_samples,
            X_new,
            prev_score=50.0,
            fixed_n_exponent=None,
            prefix="user_",
            seed=42,
        )
        assert result["y"].shape == (20,)
        # Sigma should be sigma_obs
        np.testing.assert_allclose(result["sigma_scaled"], 3.0, atol=1e-5)


class TestPredictNewEntityMultiAlbumHeteroscedastic:
    """Cover multi-album with n_reviews and fixed exponent."""

    def test_multi_album_fixed_exponent_shapes(self, basic_samples):
        """Multi-album prediction with fixed exponent returns correct shapes."""
        X_new = jnp.ones((4, 3)) * 0.5
        prev_scores = jnp.array([50.0, 60.0, 70.0, 80.0])
        n_reviews = jnp.array([10, 50, 100, 500])

        result = predict_new_entity(
            basic_samples,
            X_new,
            prev_score=prev_scores,
            n_reviews_new=n_reviews,
            fixed_n_exponent=0.4,
            prefix="user_",
            seed=42,
        )

        assert result["y"].shape == (20, 4)
        assert result["mu"].shape == (20, 4)
        assert result["sigma_scaled"].shape == (20, 4)

    def test_multi_album_learned_exponent_subsampled(self):
        """Multi-album with learned exponent and n_predictions."""
        n = 30
        samples = {
            "user_mu_artist": jnp.ones(n) * 60.0,
            "user_sigma_artist": jnp.ones(n) * 5.0,
            "user_beta": jnp.ones((n, 2)) * 0.5,
            "user_rho": jnp.ones(n) * 0.3,
            "user_sigma_obs": jnp.ones(n) * 3.0,
            "user_n_exponent": jnp.ones(n) * 0.25,
        }
        X_new = jnp.ones((2, 2)) * 0.5
        prev_scores = jnp.array([50.0, 60.0])
        n_reviews = jnp.array([100, 200])

        result = predict_new_entity(
            samples,
            X_new,
            prev_score=prev_scores,
            n_reviews_new=n_reviews,
            prefix="user_",
            seed=42,
            n_predictions=10,
        )

        assert result["y"].shape == (10, 2)
        assert result["mu"].shape == (10, 2)
        assert result["sigma_scaled"].shape == (10, 2)


class TestPredictNewEntityNReviewsNone:
    """Cover n_reviews_new=None with no exponent (homoscedastic)."""

    def test_homoscedastic_no_n_reviews(self, basic_samples):
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_entity(
            basic_samples,
            X_new,
            prev_score=0.0,
            n_reviews_new=None,
            prefix="user_",
            seed=42,
        )
        # Should work fine without n_reviews
        assert result["y"].shape == (20,)


class TestPredictNewEntityMultiAlbumPrevScore:
    """Cover multi-album prev_score array handling."""

    def test_multi_album_with_array_prev_score(self, basic_samples):
        """Multi-album prediction with prev_score array."""
        X_new = jnp.ones((3, 3)) * 0.5
        prev_scores = jnp.array([0.0, 50.0, 70.0])

        result = predict_new_entity(
            basic_samples,
            X_new,
            prev_score=prev_scores,
            prefix="user_",
            seed=42,
        )

        assert result["y"].shape == (20, 3)
        assert result["mu"].shape == (20, 3)


class TestExtractPosteriorSamplesAdditional:
    """Additional tests for extract_posterior_samples."""

    def test_returns_jax_arrays(self):
        """Output should be jnp arrays."""
        var_dict = {
            "mu": np.random.randn(2, 50),
        }
        idata = _make_idata_like_new(var_dict)
        result = extract_posterior_samples(idata)
        # Should be a JAX array
        assert hasattr(result["mu"], "device") or hasattr(result["mu"], "devices")

    def test_many_variables(self):
        """Handles many variables correctly."""
        var_dict = {f"var_{i}": np.random.randn(2, 50) for i in range(10)}
        idata = _make_idata_like_new(var_dict)
        result = extract_posterior_samples(idata)
        assert len(result) == 10
        for key in result:
            assert result[key].shape == (100,)

    def test_3d_variable(self):
        """3D variable (chain, draw, n_artists, max_seq-1)."""
        var_dict = {
            "rw_raw": np.random.randn(2, 50, 5, 3),
        }
        idata = _make_idata_like_new(var_dict)
        result = extract_posterior_samples(idata)
        assert result["rw_raw"].shape == (100, 5, 3)
