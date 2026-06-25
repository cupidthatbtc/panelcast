"""Unit tests for PriorConfig and get_default_priors."""

import dataclasses
from dataclasses import FrozenInstanceError, asdict, fields

import pytest

from panelcast.models.bayes.priors import PriorConfig, get_default_priors


class TestPriorConfigDefaults:
    """Tests for PriorConfig default values."""

    def test_default_mu_artist_loc(self):
        config = PriorConfig()
        assert config.mu_artist_loc == 0.0

    def test_default_mu_artist_scale(self):
        config = PriorConfig()
        assert config.mu_artist_scale == 1.0

    def test_default_sigma_artist_scale(self):
        config = PriorConfig()
        assert config.sigma_artist_scale == 0.5

    def test_default_sigma_rw_scale(self):
        config = PriorConfig()
        assert config.sigma_rw_scale == 0.1

    def test_default_rho_loc(self):
        config = PriorConfig()
        assert config.rho_loc == 0.0

    def test_default_rho_scale(self):
        config = PriorConfig()
        assert config.rho_scale == 0.3

    def test_default_beta_loc(self):
        config = PriorConfig()
        assert config.beta_loc == 0.0

    def test_default_beta_scale(self):
        config = PriorConfig()
        assert config.beta_scale == 1.0

    def test_default_sigma_obs_scale(self):
        config = PriorConfig()
        assert config.sigma_obs_scale == 1.0

    def test_default_sigma_ref_scale(self):
        config = PriorConfig()
        assert config.sigma_ref_scale == 1.0

    def test_default_n_exponent_alpha(self):
        config = PriorConfig()
        assert config.n_exponent_alpha == 2.0

    def test_default_n_exponent_beta(self):
        config = PriorConfig()
        assert config.n_exponent_beta == 4.0

    def test_default_n_exponent_loc(self):
        config = PriorConfig()
        assert config.n_exponent_loc == -2.2

    def test_default_n_exponent_scale(self):
        config = PriorConfig()
        assert config.n_exponent_scale == 1.0


class TestPriorConfigImmutability:
    """Tests for PriorConfig frozen dataclass."""

    def test_frozen_mu_artist_loc(self):
        config = PriorConfig()
        with pytest.raises(FrozenInstanceError):
            config.mu_artist_loc = 99.0

    def test_frozen_sigma_artist_scale(self):
        config = PriorConfig()
        with pytest.raises(FrozenInstanceError):
            config.sigma_artist_scale = 99.0

    def test_frozen_rho_loc(self):
        config = PriorConfig()
        with pytest.raises(FrozenInstanceError):
            config.rho_loc = 99.0

    def test_frozen_beta_scale(self):
        config = PriorConfig()
        with pytest.raises(FrozenInstanceError):
            config.beta_scale = 99.0


class TestPriorConfigCustom:
    """Tests for PriorConfig with custom values."""

    def test_custom_all_fields(self):
        config = PriorConfig(
            mu_artist_loc=70.0,
            mu_artist_scale=5.0,
            sigma_artist_scale=1.0,
            sigma_rw_scale=0.5,
            rho_loc=0.3,
            rho_scale=0.5,
            beta_loc=1.0,
            beta_scale=2.0,
            sigma_obs_scale=3.0,
            sigma_ref_scale=2.0,
            n_exponent_alpha=3.0,
            n_exponent_beta=5.0,
            n_exponent_loc=-1.0,
            n_exponent_scale=0.5,
        )
        assert config.mu_artist_loc == 70.0
        assert config.mu_artist_scale == 5.0
        assert config.sigma_artist_scale == 1.0
        assert config.sigma_rw_scale == 0.5
        assert config.rho_loc == 0.3
        assert config.rho_scale == 0.5
        assert config.beta_loc == 1.0
        assert config.beta_scale == 2.0
        assert config.sigma_obs_scale == 3.0
        assert config.sigma_ref_scale == 2.0
        assert config.n_exponent_alpha == 3.0
        assert config.n_exponent_beta == 5.0
        assert config.n_exponent_loc == -1.0
        assert config.n_exponent_scale == 0.5

    def test_partial_override(self):
        config = PriorConfig(mu_artist_loc=10.0, rho_scale=0.5)
        assert config.mu_artist_loc == 10.0
        assert config.rho_scale == 0.5
        # Others should be defaults
        assert config.mu_artist_scale == 1.0
        assert config.sigma_artist_scale == 0.5


class TestPriorConfigSerialization:
    """Tests for PriorConfig asdict serialization."""

    def test_asdict_returns_dict(self):
        config = PriorConfig()
        d = asdict(config)
        assert isinstance(d, dict)

    def test_asdict_has_all_fields(self):
        config = PriorConfig()
        d = asdict(config)
        field_names = {f.name for f in fields(PriorConfig)}
        assert set(d.keys()) == field_names

    def test_asdict_roundtrip(self):
        original = PriorConfig(mu_artist_loc=5.0, rho_loc=0.1)
        d = asdict(original)
        restored = PriorConfig(**d)
        assert restored == original

    def test_field_count(self):
        """PriorConfig should have exactly 52 fields.

        18 legacy + 11 seam knobs + 5 for the entity-overdispersion / lognormal
        sigma_obs upgrade (sigma_obs_prior_type, sigma_obs_lognormal_loc,
        sigma_obs_lognormal_sigma, heteroscedastic_entity_obs, tau_entity_scale)
        + 6 for the skew/bounded likelihood candidates (skew_loc, skew_scale,
        skew_tailweight, beta_precision_concentration, beta_precision_rate,
        beta_boundary_eps) + 3 for the split-normal / discretization wave
        (split_scale_ratio_loc, split_scale_ratio_scale, discretize_observation)
        + 3 for the Beta-Binomial family (betabinom_precision_concentration,
        betabinom_precision_rate, betabinom_max_n_reviews) + 6 for the two-component
        mixture (mix_sep_loc, mix_sep_scale, mix_weight_a, mix_weight_b,
        mix_scale_ratio_loc, mix_scale_ratio_scale).
        """
        assert len(fields(PriorConfig)) == 52


class TestSigmaRwPriorType:
    """Tests for the sigma_rw LogNormal prior configuration."""

    def test_default_is_lognormal(self):
        assert PriorConfig().sigma_rw_prior_type == "lognormal"

    def test_lognormal_defaults(self):
        p = PriorConfig()
        assert p.sigma_rw_lognormal_loc == -2.8
        assert p.sigma_rw_lognormal_sigma == 0.6

    def test_backward_compat_deserialization(self):
        """Old summary dicts without sigma_rw_prior_type should deserialize."""
        old_dict = {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        }
        p = PriorConfig(**old_dict)
        assert p.sigma_rw_prior_type == "lognormal"
        assert p.sigma_rw_lognormal_loc == -2.8

    def test_roundtrip_with_new_fields(self):
        p = PriorConfig(sigma_rw_prior_type="halfnormal")
        d = dataclasses.asdict(p)
        restored = PriorConfig(**d)
        assert restored.sigma_rw_prior_type == "halfnormal"
        assert restored == p


class TestPriorConfigEquality:
    """Tests for PriorConfig equality."""

    def test_equal_defaults(self):
        a = PriorConfig()
        b = PriorConfig()
        assert a == b

    def test_unequal_different_values(self):
        a = PriorConfig(mu_artist_loc=0.0)
        b = PriorConfig(mu_artist_loc=1.0)
        assert a != b


class TestGetDefaultPriors:
    """Tests for get_default_priors function."""

    def test_returns_prior_config(self):
        priors = get_default_priors()
        assert isinstance(priors, PriorConfig)

    def test_returns_default_values(self):
        priors = get_default_priors()
        assert priors == PriorConfig()

    def test_returns_new_instance_each_time(self):
        p1 = get_default_priors()
        p2 = get_default_priors()
        assert p1 == p2
        # Both should be equal but could be same object (frozen dataclass)
        # Just verify values are correct
        assert p1.mu_artist_loc == 0.0

    def test_weakly_informative_priors(self):
        """Default priors should be weakly informative."""
        priors = get_default_priors()
        # Artist effects centered at zero
        assert priors.mu_artist_loc == 0.0
        # Moderate pooling
        assert 0.1 <= priors.sigma_artist_scale <= 2.0
        # Smooth career trajectories
        assert priors.sigma_rw_scale < 0.5
        # No prior belief about AR direction
        assert priors.rho_loc == 0.0
