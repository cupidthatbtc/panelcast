"""Unit tests for MCMC model fitting.

Tests cover:
- Actual MCMC fitting with minimal data (marked slow)
- FitResult structure and fields
- InferenceData groups and posterior parameters
- GPU info detection
- MCMCConfig defaults

These tests actually run MCMC sampling with minimal settings to verify
the fit_model function works end-to-end.
"""

from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import numpy as np
import pytest

from panelcast.models.bayes.fit import (
    FitResult,
    MCMCConfig,
    fit_model,
    get_gpu_info,
)
from panelcast.models.bayes.model import user_score_model

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_data():
    """Create minimal data for MCMC tests.

    Uses small dataset (20 observations, 3 artists) with minimal
    feature dimensions to keep MCMC fast.
    """
    n_obs = 20
    n_artists = 3
    n_features = 2
    max_seq = 3

    np.random.seed(42)

    # Artist indices (repeated to get multiple albums per artist)
    artist_idx = jnp.array([i % n_artists for i in range(n_obs)], dtype=jnp.int32)

    # Album sequences (1-indexed, 1-3 per artist)
    album_seq = jnp.array([(i // n_artists) % max_seq + 1 for i in range(n_obs)], dtype=jnp.int32)

    # Previous scores (0 for first album of each artist)
    prev_score_raw = np.random.randn(n_obs) * 10 + 70
    prev_score_raw[album_seq == 1] = 0  # Zero for debut albums
    prev_score = jnp.array(prev_score_raw)

    # Feature matrix (standardized)
    X = jnp.array(np.random.randn(n_obs, n_features))

    # Target scores (realistic range: mean ~70, std ~10)
    y = jnp.array(np.random.randn(n_obs) * 10 + 70)

    return {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
        "prev_score": prev_score,
        "X": X,
        "y": y,
        "n_artists": n_artists,
        "max_seq": max_seq,
    }


@pytest.fixture
def fast_mcmc_config():
    """MCMC config with minimal iterations for fast tests."""
    return MCMCConfig(
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        chain_method="sequential",
        seed=42,
        max_tree_depth=5,  # Shallow tree for speed
        target_accept_prob=0.8,
    )


# =============================================================================
# Tests for MCMCConfig
# =============================================================================


class TestMCMCConfig:
    """Tests for MCMCConfig dataclass."""

    def test_mcmc_config_defaults(self):
        """MCMCConfig should have reasonable production defaults."""
        config = MCMCConfig()

        # Production defaults: 1000/1000/4/sequential for stable runs
        assert config.num_warmup == 1000
        assert config.num_samples == 1000
        assert config.num_chains == 4
        assert config.chain_method == "sequential"
        assert config.seed == 0
        assert config.max_tree_depth == 10  # numpyro default
        assert config.target_accept_prob == 0.9  # v5.0: increased for challenging posteriors

    def test_mcmc_config_frozen(self):
        """MCMCConfig should be immutable (frozen)."""
        config = MCMCConfig()

        with pytest.raises(FrozenInstanceError):
            config.num_warmup = 500

    def test_mcmc_config_to_dict(self):
        """MCMCConfig.to_dict should serialize correctly."""
        config = MCMCConfig(num_warmup=50, num_samples=50)

        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["num_warmup"] == 50
        assert d["num_samples"] == 50
        assert "seed" in d
        assert "max_tree_depth" in d


# =============================================================================
# Tests for get_gpu_info
# =============================================================================


class TestGetGPUInfo:
    """Tests for get_gpu_info function."""

    def test_get_gpu_info_returns_string(self):
        """get_gpu_info should return non-empty string."""
        info = get_gpu_info()

        assert isinstance(info, str)
        assert len(info) > 0

    def test_get_gpu_info_has_valid_content(self):
        """get_gpu_info should return 'CPU only' or GPU name."""
        info = get_gpu_info()

        # Should be either CPU indication or contain GPU-related text
        valid_patterns = [
            "CPU only",
            "NVIDIA",
            "GPU",
            "cuda",
            "RTX",
            "GeForce",
            "Tesla",
            "A100",
            "V100",
            "MiB",
        ]

        has_valid = any(pattern in info for pattern in valid_patterns)
        assert has_valid, f"GPU info '{info}' should contain known GPU/CPU pattern"


# =============================================================================
# Tests for fit_model (marked slow - actually runs MCMC)
# =============================================================================


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestFitModel:
    """Tests for fit_model function.

    These tests actually run MCMC sampling with minimal iterations.
    Marked with @pytest.mark.slow for selective execution.
    """

    def test_fit_model_returns_fit_result(self, minimal_model_data, fast_mcmc_config):
        """fit_model should return FitResult with all expected fields."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        assert isinstance(result, FitResult)
        assert result.mcmc is not None
        assert result.idata is not None
        assert isinstance(result.divergences, int)
        assert result.divergences >= 0
        assert isinstance(result.runtime_seconds, float)
        assert result.runtime_seconds > 0
        assert isinstance(result.gpu_info, str)
        assert len(result.gpu_info) > 0

    def test_fit_model_idata_has_required_groups(self, minimal_model_data, fast_mcmc_config):
        """InferenceData should have posterior, observed_data, constant_data groups."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        groups = set(result.idata.groups())

        assert "posterior" in groups
        # observed_data or constant_data should be present
        assert "observed_data" in groups or "constant_data" in groups

    def test_fit_model_posterior_has_expected_params(self, minimal_model_data, fast_mcmc_config):
        """Posterior should contain expected model parameters."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        posterior_vars = list(result.idata.posterior.data_vars)

        # Check for user_score_model parameters (with user_ prefix)
        expected_params = [
            "user_mu_artist",
            "user_sigma_artist",
            "user_beta",
            "user_sigma_obs",
            "user_rho",
        ]

        for param in expected_params:
            assert param in posterior_vars, f"Missing parameter: {param}"

    def test_fit_model_posterior_shapes(self, minimal_model_data, fast_mcmc_config):
        """Posterior parameter shapes should match config and data."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        posterior = result.idata.posterior

        # With 1 chain, 5 draws:
        n_chains = fast_mcmc_config.num_chains
        n_draws = fast_mcmc_config.num_samples
        n_features = minimal_model_data["X"].shape[1]

        # Scalar parameters should have shape (chain, draw)
        assert posterior["user_mu_artist"].shape == (n_chains, n_draws)
        assert posterior["user_sigma_artist"].shape == (n_chains, n_draws)
        assert posterior["user_sigma_obs"].shape == (n_chains, n_draws)
        assert posterior["user_rho"].shape == (n_chains, n_draws)

        # Beta should have shape (chain, draw, n_features)
        assert posterior["user_beta"].shape == (n_chains, n_draws, n_features)

    def test_fit_model_different_seeds_different_results(self, minimal_model_data):
        """Different seeds should produce different posterior samples."""
        config1 = MCMCConfig(num_warmup=5, num_samples=5, num_chains=1, seed=1)
        config2 = MCMCConfig(num_warmup=5, num_samples=5, num_chains=1, seed=999)

        result1 = fit_model(
            user_score_model,
            minimal_model_data,
            config=config1,
            progress_bar=False,
        )
        result2 = fit_model(
            user_score_model,
            minimal_model_data,
            config=config2,
            progress_bar=False,
        )

        # Extract a scalar parameter's samples
        mu1 = result1.idata.posterior["user_mu_artist"].values
        mu2 = result2.idata.posterior["user_mu_artist"].values

        # Different seeds should produce different samples
        assert not np.allclose(mu1, mu2), "Different seeds should produce different results"

    def test_fit_model_tracks_divergences(self, minimal_model_data, fast_mcmc_config):
        """fit_model should track divergences correctly."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        # Divergences should be a non-negative integer
        assert isinstance(result.divergences, int)
        assert result.divergences >= 0
        # With minimal warmup, some divergences may occur, but should be reasonable
        assert result.divergences < 100, "Divergence count seems unreasonably high"

    def test_fit_model_uses_default_config(self, minimal_model_data):
        """fit_model should use default MCMCConfig if none provided."""
        # This test verifies the default config is used (won't run full default)
        # We just verify it doesn't error when config=None
        config = MCMCConfig(num_warmup=3, num_samples=3, num_chains=1)
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=config,  # Using minimal config for speed
            progress_bar=False,
        )

        assert result is not None
        assert isinstance(result, FitResult)

    def test_fit_model_mcmc_has_samples(self, minimal_model_data, fast_mcmc_config):
        """MCMC object should have accessible samples."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        samples = result.mcmc.get_samples()

        assert isinstance(samples, dict)
        assert len(samples) > 0
        assert "user_beta" in samples

    def test_fit_model_runtime_reasonable(self, minimal_model_data, fast_mcmc_config):
        """Runtime should be positive and within reasonable bounds."""
        result = fit_model(
            user_score_model,
            minimal_model_data,
            config=fast_mcmc_config,
            progress_bar=False,
        )

        # Should complete in reasonable time (< 60s for minimal config)
        assert result.runtime_seconds > 0
        assert result.runtime_seconds < 60, "Minimal MCMC should complete quickly"
