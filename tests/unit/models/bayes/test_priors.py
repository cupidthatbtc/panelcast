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
        """PriorConfig should have exactly 61 fields.

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
        mix_scale_ratio_loc, mix_scale_ratio_scale) + 2 for the model-v2 gates
        (errors_in_variables, propagate_rw_horizon) + 2 for the genre-pooling
        gate (entity_group_pooling, sigma_group_scale) + 1 for the
        beta_ceiling support bound (effective_ceiling) + 4 for the
        regularized-horseshoe beta prior (beta_prior_type, hs_global_scale,
        hs_slab_scale, hs_slab_df).
        """
        assert len(fields(PriorConfig)) == 61


class TestModelV2Gates:
    """Errors-in-variables and long-horizon RW propagation gates (model-v2)."""

    def test_defaults_off(self):
        p = PriorConfig()
        assert p.errors_in_variables is False
        assert p.propagate_rw_horizon is False

    def test_roundtrip(self):
        p = PriorConfig(errors_in_variables=True, propagate_rw_horizon=True)
        restored = PriorConfig(**asdict(p))
        assert restored == p
        assert restored.errors_in_variables is True
        assert restored.propagate_rw_horizon is True

    def test_backward_compat_without_v2_keys(self):
        """A priors dict predating model-v2 deserializes with both gates off."""
        d = asdict(PriorConfig())
        d.pop("errors_in_variables")
        d.pop("propagate_rw_horizon")
        p = PriorConfig(**d)
        assert p.errors_in_variables is False
        assert p.propagate_rw_horizon is False


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


# --- from unit/models/bayes/test_priors_expanded.py ---


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


class TestGetDefaultPriors_expanded:
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
