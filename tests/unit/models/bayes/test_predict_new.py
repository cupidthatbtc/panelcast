"""New coverage tests for models/bayes/predict.py.

Targets uncovered code paths:
- generate_posterior_predictive: full execution with mocked MCMC,
  y_key fallback to "y", return_mu=True with mu_key present,
  y_key fallback to first prediction key when no _y or "y" key
- predict_out_of_sample: delegation behavior
- predict_new_artist: sigma_scaled shape validation error path,
  multi-album with n_reviews_new (heteroscedastic, non-single),
  fixed_n_exponent=None is treated as homoscedastic
- extract_posterior_samples: returns jnp arrays
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.predict import (
    PredictionResult,
    extract_posterior_samples,
    generate_posterior_predictive,
    predict_new_artist,
    predict_out_of_sample,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_idata_like(var_dict):
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


# ===========================================================================
# generate_posterior_predictive
# ===========================================================================


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


# ===========================================================================
# predict_out_of_sample
# ===========================================================================


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


# ===========================================================================
# predict_new_artist: additional coverage
# ===========================================================================


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


class TestPredictNewArtistFixedNone:
    """Cover fixed_n_exponent=None explicitly."""

    def test_none_fixed_exponent_is_homoscedastic(self, basic_samples):
        """fixed_n_exponent=None should be homoscedastic."""
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
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


class TestPredictNewArtistMultiAlbumHeteroscedastic:
    """Cover multi-album with n_reviews and fixed exponent."""

    def test_multi_album_fixed_exponent_shapes(self, basic_samples):
        """Multi-album prediction with fixed exponent returns correct shapes."""
        X_new = jnp.ones((4, 3)) * 0.5
        prev_scores = jnp.array([50.0, 60.0, 70.0, 80.0])
        n_reviews = jnp.array([10, 50, 100, 500])

        result = predict_new_artist(
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

        result = predict_new_artist(
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


class TestPredictNewArtistNReviewsNone:
    """Cover n_reviews_new=None with no exponent (homoscedastic)."""

    def test_homoscedastic_no_n_reviews(self, basic_samples):
        X_new = jnp.array([1.0, 0.0, 0.0])
        result = predict_new_artist(
            basic_samples,
            X_new,
            prev_score=0.0,
            n_reviews_new=None,
            prefix="user_",
            seed=42,
        )
        # Should work fine without n_reviews
        assert result["y"].shape == (20,)


class TestPredictNewArtistMultiAlbumPrevScore:
    """Cover multi-album prev_score array handling."""

    def test_multi_album_with_array_prev_score(self, basic_samples):
        """Multi-album prediction with prev_score array."""
        X_new = jnp.ones((3, 3)) * 0.5
        prev_scores = jnp.array([0.0, 50.0, 70.0])

        result = predict_new_artist(
            basic_samples,
            X_new,
            prev_score=prev_scores,
            prefix="user_",
            seed=42,
        )

        assert result["y"].shape == (20, 3)
        assert result["mu"].shape == (20, 3)


# ===========================================================================
# extract_posterior_samples: additional coverage
# ===========================================================================


class TestExtractPosteriorSamplesAdditional:
    """Additional tests for extract_posterior_samples."""

    def test_returns_jax_arrays(self):
        """Output should be jnp arrays."""
        var_dict = {
            "mu": np.random.randn(2, 50),
        }
        idata = _make_idata_like(var_dict)
        result = extract_posterior_samples(idata)
        # Should be a JAX array
        assert hasattr(result["mu"], "device") or hasattr(result["mu"], "devices")

    def test_many_variables(self):
        """Handles many variables correctly."""
        var_dict = {f"var_{i}": np.random.randn(2, 50) for i in range(10)}
        idata = _make_idata_like(var_dict)
        result = extract_posterior_samples(idata)
        assert len(result) == 10
        for key in result:
            assert result[key].shape == (100,)

    def test_3d_variable(self):
        """3D variable (chain, draw, n_artists, max_seq-1)."""
        var_dict = {
            "rw_raw": np.random.randn(2, 50, 5, 3),
        }
        idata = _make_idata_like(var_dict)
        result = extract_posterior_samples(idata)
        assert result["rw_raw"].shape == (100, 5, 3)
