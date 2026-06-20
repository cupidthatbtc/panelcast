"""Expanded tests for models/bayes/priors.py: PriorConfig, get_default_priors."""

from dataclasses import FrozenInstanceError

import pytest

from panelcast.models.bayes.priors import PriorConfig, get_default_priors


class TestPriorConfigExpanded:
    """Expanded PriorConfig tests."""

    def test_all_defaults(self):
        cfg = PriorConfig()
        assert cfg.mu_artist_loc == 0.0
        assert cfg.mu_artist_scale == 1.0
        assert cfg.sigma_artist_scale == 0.5
        assert cfg.sigma_rw_scale == 0.1
        assert cfg.rho_loc == 0.0
        assert cfg.rho_scale == 0.3
        assert cfg.beta_loc == 0.0
        assert cfg.beta_scale == 1.0
        assert cfg.sigma_obs_scale == 1.0
        assert cfg.sigma_ref_scale == 1.0
        assert cfg.n_exponent_alpha == 2.0
        assert cfg.n_exponent_beta == 4.0
        assert cfg.n_exponent_loc == -2.2
        assert cfg.n_exponent_scale == 1.0

    def test_frozen(self):
        cfg = PriorConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.mu_artist_loc = 5.0

    def test_custom_values(self):
        cfg = PriorConfig(
            mu_artist_loc=1.0,
            mu_artist_scale=2.0,
            sigma_artist_scale=0.3,
            sigma_rw_scale=0.05,
        )
        assert cfg.mu_artist_loc == 1.0
        assert cfg.mu_artist_scale == 2.0
        assert cfg.sigma_artist_scale == 0.3
        assert cfg.sigma_rw_scale == 0.05

    def test_equality(self):
        a = PriorConfig(mu_artist_loc=0.5)
        b = PriorConfig(mu_artist_loc=0.5)
        assert a == b

    def test_inequality(self):
        a = PriorConfig(mu_artist_loc=0.0)
        b = PriorConfig(mu_artist_loc=1.0)
        assert a != b

    def test_hashable(self):
        cfg = PriorConfig()
        s = {cfg}
        assert len(s) == 1

    def test_sigma_ref_scale(self):
        cfg = PriorConfig(sigma_ref_scale=2.0)
        assert cfg.sigma_ref_scale == 2.0

    def test_n_exponent_logit_normal(self):
        cfg = PriorConfig(n_exponent_loc=-1.0, n_exponent_scale=0.5)
        assert cfg.n_exponent_loc == -1.0
        assert cfg.n_exponent_scale == 0.5

    def test_negative_scale(self):
        """PriorConfig doesn't validate, so negative scale is accepted."""
        cfg = PriorConfig(sigma_artist_scale=-1.0)
        assert cfg.sigma_artist_scale == -1.0

    def test_zero_scale(self):
        cfg = PriorConfig(sigma_obs_scale=0.0)
        assert cfg.sigma_obs_scale == 0.0


class TestGetDefaultPriors:
    """Tests for get_default_priors."""

    def test_returns_prior_config(self):
        result = get_default_priors()
        assert isinstance(result, PriorConfig)

    def test_is_default(self):
        result = get_default_priors()
        assert result == PriorConfig()

    def test_new_instance(self):
        a = get_default_priors()
        b = get_default_priors()
        assert a == b

    def test_weakly_informative(self):
        """Default priors should be weakly informative."""
        cfg = get_default_priors()
        # Scales should be positive and not too small/large
        assert 0 < cfg.sigma_artist_scale <= 5.0
        assert 0 < cfg.sigma_rw_scale <= 5.0
        assert 0 < cfg.rho_scale <= 5.0
        assert 0 < cfg.beta_scale <= 5.0
        assert 0 < cfg.sigma_obs_scale <= 5.0
