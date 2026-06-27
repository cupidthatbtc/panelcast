"""Tests for heteroscedastic prediction functionality.

Verifies that:
1. predict_new_entity accepts n_reviews_new parameter
2. Prediction intervals differ by review count (heteroscedastic behavior)
3. ValueError raised when n_reviews_new missing for learned exponent model
4. sigma_scaled included in prediction output
5. Backward compatibility: homoscedastic models work without n_reviews_new
"""

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.predict import predict_new_entity


class TestPredictHeteroscedastic:
    """Tests for heteroscedastic prediction support."""

    @pytest.fixture
    def homoscedastic_posterior_samples(self):
        """Create mock posterior samples for homoscedastic model (no n_exponent)."""
        n_samples = 100
        n_features = 3
        rng = np.random.default_rng(42)
        return {
            "user_mu_artist": jnp.array(rng.normal(70, 2, n_samples)),
            "user_sigma_artist": jnp.array(rng.exponential(5, n_samples)),
            "user_beta": jnp.array(rng.normal(0, 1, (n_samples, n_features))),
            "user_rho": jnp.array(rng.uniform(0.1, 0.5, n_samples)),
            "user_sigma_obs": jnp.array(rng.exponential(5, n_samples)),
        }

    @pytest.fixture
    def heteroscedastic_posterior_samples(self, homoscedastic_posterior_samples):
        """Create mock posterior samples with learned n_exponent."""
        samples = homoscedastic_posterior_samples.copy()
        n_samples = samples["user_mu_artist"].shape[0]
        rng = np.random.default_rng(42)
        # Add n_exponent samples (Beta(2,4) prior has mean ~0.33)
        samples["user_n_exponent"] = jnp.array(rng.beta(2, 4, n_samples))
        return samples

    def test_homoscedastic_backward_compatible(self, homoscedastic_posterior_samples):
        """Test that homoscedastic models work without n_reviews_new."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0

        # Should work without n_reviews_new for homoscedastic model
        result = predict_new_entity(
            homoscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=None,  # Not provided
            prefix="user_",
        )

        assert "y" in result
        assert "mu" in result
        assert "sigma_scaled" in result
        # sigma_scaled should be scalar-like for single album
        assert result["sigma_scaled"].shape == result["y"].shape

    def test_heteroscedastic_requires_n_reviews(self, heteroscedastic_posterior_samples):
        """Test that learned exponent model raises error without n_reviews_new."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0

        with pytest.raises(ValueError, match="n_reviews_new is required"):
            predict_new_entity(
                heteroscedastic_posterior_samples,
                X_new,
                prev_score,
                n_reviews_new=None,  # Missing!
                prefix="user_",
            )

    def test_heteroscedastic_prediction_single_album(self, heteroscedastic_posterior_samples):
        """Test heteroscedastic prediction for single album."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0
        n_reviews = jnp.array([100])

        result = predict_new_entity(
            heteroscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            prefix="user_",
        )

        assert "y" in result
        assert "mu" in result
        assert "sigma_scaled" in result
        # Single album output
        assert result["y"].ndim == 1
        assert result["sigma_scaled"].ndim == 1

    def test_heteroscedastic_prediction_multiple_albums(self, heteroscedastic_posterior_samples):
        """Test heteroscedastic prediction for multiple albums."""
        n_albums = 5
        n_features = 3
        X_new = jnp.array(np.random.randn(n_albums, n_features))
        prev_score = jnp.zeros(n_albums)
        n_reviews = jnp.array([10, 50, 100, 500, 1000])

        result = predict_new_entity(
            heteroscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            prefix="user_",
        )

        n_samples = heteroscedastic_posterior_samples["user_mu_artist"].shape[0]
        assert result["y"].shape == (n_samples, n_albums)
        assert result["sigma_scaled"].shape == (n_samples, n_albums)

    def test_sigma_scaled_differs_by_review_count(self, heteroscedastic_posterior_samples):
        """Test that sigma_scaled is smaller for albums with more reviews."""
        X_new = jnp.array([[0.5, -0.2, 0.1], [0.5, -0.2, 0.1]])  # Same features
        prev_score = jnp.zeros(2)
        # Different review counts: 10 vs 1000
        n_reviews = jnp.array([10, 1000])

        result = predict_new_entity(
            heteroscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            prefix="user_",
        )

        # sigma_scaled for low reviews (n=10) should be higher than high reviews (n=1000)
        sigma_low_reviews = result["sigma_scaled"][:, 0].mean()  # n=10
        sigma_high_reviews = result["sigma_scaled"][:, 1].mean()  # n=1000

        assert sigma_low_reviews > sigma_high_reviews, (
            f"Expected sigma at n=10 ({sigma_low_reviews:.4f}) > "
            f"sigma at n=1000 ({sigma_high_reviews:.4f})"
        )

    def test_prediction_intervals_wider_for_low_reviews(self, heteroscedastic_posterior_samples):
        """Test that prediction intervals are wider for albums with fewer reviews."""
        # Same features, different review counts
        X_new = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        prev_score = jnp.zeros(2)
        n_reviews = jnp.array([10, 1000])

        result = predict_new_entity(
            heteroscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            prefix="user_",
            seed=42,
        )

        # Compute prediction interval widths (97.5 - 2.5 percentile)
        y_pred = result["y"]
        width_low_reviews = np.percentile(y_pred[:, 0], 97.5) - np.percentile(y_pred[:, 0], 2.5)
        width_high_reviews = np.percentile(y_pred[:, 1], 97.5) - np.percentile(y_pred[:, 1], 2.5)

        assert width_low_reviews > width_high_reviews, (
            f"Expected wider interval for n=10 ({width_low_reviews:.2f}) "
            f"than n=1000 ({width_high_reviews:.2f})"
        )

    def test_n_predictions_subsampling_with_n_exponent(self, heteroscedastic_posterior_samples):
        """Test that n_predictions subsampling works with heteroscedastic model."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0
        n_reviews = jnp.array([100])
        n_predictions = 50

        result = predict_new_entity(
            heteroscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            prefix="user_",
            n_predictions=n_predictions,
        )

        # Should have exactly n_predictions samples
        assert result["y"].shape[0] == n_predictions
        assert result["sigma_scaled"].shape[0] == n_predictions

    def test_fixed_exponent_prediction(self, homoscedastic_posterior_samples):
        """Test prediction with fixed (non-learned) exponent."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0
        n_reviews = jnp.array([100])

        result = predict_new_entity(
            homoscedastic_posterior_samples,  # No n_exponent in posterior
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            fixed_n_exponent=0.5,  # Use fixed exponent
            prefix="user_",
        )

        assert "y" in result
        assert "mu" in result
        assert "sigma_scaled" in result
        # Single album output
        assert result["y"].ndim == 1
        assert result["sigma_scaled"].ndim == 1

    def test_fixed_exponent_requires_n_reviews(self, homoscedastic_posterior_samples):
        """Test that fixed non-zero exponent requires n_reviews_new."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0

        with pytest.raises(ValueError, match="n_reviews_new is required"):
            predict_new_entity(
                homoscedastic_posterior_samples,
                X_new,
                prev_score,
                n_reviews_new=None,  # Missing!
                fixed_n_exponent=0.5,  # Fixed non-zero exponent
                prefix="user_",
            )

    def test_fixed_exponent_sigma_differs_by_review_count(self, homoscedastic_posterior_samples):
        """Test that sigma_scaled differs by review count with fixed exponent."""
        X_new = jnp.array([[0.5, -0.2, 0.1], [0.5, -0.2, 0.1]])  # Same features
        prev_score = jnp.zeros(2)
        n_reviews = jnp.array([10, 1000])

        result = predict_new_entity(
            homoscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            fixed_n_exponent=0.5,  # sqrt scaling
            prefix="user_",
        )

        # sigma_scaled for low reviews (n=10) should be higher than high reviews (n=1000)
        sigma_low_reviews = result["sigma_scaled"][:, 0].mean()  # n=10
        sigma_high_reviews = result["sigma_scaled"][:, 1].mean()  # n=1000

        assert sigma_low_reviews > sigma_high_reviews, (
            f"Expected sigma at n=10 ({sigma_low_reviews:.4f}) > "
            f"sigma at n=1000 ({sigma_high_reviews:.4f})"
        )

    def test_fixed_exponent_zero_treated_as_homoscedastic(self, homoscedastic_posterior_samples):
        """Test that fixed_n_exponent=0 is treated as homoscedastic."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0

        # fixed_n_exponent=0 should NOT require n_reviews_new
        result = predict_new_entity(
            homoscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=None,
            fixed_n_exponent=0.0,  # Zero exponent = homoscedastic
            prefix="user_",
        )

        assert "y" in result
        assert "sigma_scaled" in result

    def test_fixed_exponent_with_subsampling(self, homoscedastic_posterior_samples):
        """Test that n_predictions subsampling works with fixed exponent."""
        X_new = jnp.array([0.5, -0.2, 0.1])
        prev_score = 0.0
        n_reviews = jnp.array([100])
        n_predictions = 50

        result = predict_new_entity(
            homoscedastic_posterior_samples,
            X_new,
            prev_score,
            n_reviews_new=n_reviews,
            fixed_n_exponent=0.33,
            prefix="user_",
            n_predictions=n_predictions,
        )

        # Should have exactly n_predictions samples
        assert result["y"].shape[0] == n_predictions
        assert result["sigma_scaled"].shape[0] == n_predictions


class TestGetTracePlotVars:
    """Tests for dynamic trace plot variable selection."""

    @pytest.fixture
    def mock_idata_homoscedastic(self):
        """Create mock InferenceData without n_exponent (homoscedastic)."""
        import xarray as xr

        # Create minimal posterior structure
        posterior = xr.Dataset(
            {
                "user_mu_artist": xr.DataArray(np.random.randn(2, 100), dims=["chain", "draw"]),
                "user_sigma_artist": xr.DataArray(
                    np.abs(np.random.randn(2, 100)), dims=["chain", "draw"]
                ),
                "user_sigma_rw": xr.DataArray(
                    np.abs(np.random.randn(2, 100)), dims=["chain", "draw"]
                ),
                "user_sigma_obs": xr.DataArray(
                    np.abs(np.random.randn(2, 100)), dims=["chain", "draw"]
                ),
                "user_rho": xr.DataArray(
                    np.random.uniform(-0.5, 0.5, (2, 100)), dims=["chain", "draw"]
                ),
            }
        )
        import arviz as az

        return az.InferenceData(posterior=posterior)

    @pytest.fixture
    def mock_idata_heteroscedastic(self, mock_idata_homoscedastic):
        """Create mock InferenceData with n_exponent (heteroscedastic learned)."""
        import arviz as az
        import xarray as xr

        # Add n_exponent to posterior
        posterior = mock_idata_homoscedastic.posterior.copy()
        posterior["user_n_exponent"] = xr.DataArray(
            np.random.beta(2, 4, (2, 100)), dims=["chain", "draw"]
        )
        return az.InferenceData(posterior=posterior)

    def test_homoscedastic_vars_no_n_exponent(self, mock_idata_homoscedastic):
        """Test that n_exponent not included for homoscedastic model."""
        from panelcast.reporting.figures import get_trace_plot_vars

        var_names = get_trace_plot_vars(mock_idata_homoscedastic, prefix="user_")

        assert "user_n_exponent" not in var_names
        assert "user_sigma_obs" in var_names
        assert "user_rho" in var_names

    def test_heteroscedastic_vars_includes_n_exponent(self, mock_idata_heteroscedastic):
        """Test that n_exponent included for heteroscedastic learned model."""
        from panelcast.reporting.figures import get_trace_plot_vars

        var_names = get_trace_plot_vars(mock_idata_heteroscedastic, prefix="user_")

        assert "user_n_exponent" in var_names
        assert "user_sigma_obs" in var_names

    def test_include_hyperpriors_flag(self, mock_idata_homoscedastic):
        """Test include_hyperpriors=False excludes population-level params."""
        from panelcast.reporting.figures import get_trace_plot_vars

        var_names = get_trace_plot_vars(
            mock_idata_homoscedastic, prefix="user_", include_hyperpriors=False
        )

        assert "user_mu_artist" not in var_names
        assert "user_sigma_artist" not in var_names
        assert "user_sigma_obs" in var_names  # Still included
        assert "user_rho" in var_names  # Still included
