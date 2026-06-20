"""Tests for cache key consistency between full_check and calibrate."""

import numpy as np
import pytest

from panelcast.preflight.cache import compute_config_hash
from panelcast.preflight.full_check import _derive_dimensions_from_model_args


class TestCacheKeyConsistency:
    """Test that cache keys are computed consistently."""

    def test_derive_dimensions_from_model_args(self):
        """Test dimension derivation from model_args."""
        model_args = {
            "y": np.zeros(100),
            "X": np.zeros((100, 5)),
            "n_artists": 20,
            "max_seq": 10,
        }
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        assert n_obs == 100
        assert n_art == 20
        assert n_feat == 5
        assert max_s == 10

    def test_cache_hash_consistency(self):
        """Test that derived dimensions produce same hash as explicit."""
        model_args = {
            "y": np.zeros(100),
            "X": np.zeros((100, 5)),
            "n_artists": 20,
            "max_seq": 10,
        }

        # Derive dimensions
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        # Hash from derived should match hash from explicit
        derived_hash = compute_config_hash(n_obs, n_art, n_feat, max_s)
        explicit_hash = compute_config_hash(100, 20, 5, 10)

        assert derived_hash == explicit_hash

    def test_empty_model_args_defaults(self):
        """Test dimension derivation with missing keys."""
        model_args = {}
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        assert n_obs == 0
        assert n_art == 0
        assert n_feat == 0
        assert max_s == 0

    def test_jax_array_handling(self):
        """Test dimension derivation with JAX arrays (if available)."""
        pytest.importorskip("jax")
        import jax.numpy as jnp

        model_args = {
            "y": jnp.zeros(50),
            "X": jnp.zeros((50, 3)),
            "n_artists": 10,
            "max_seq": 5,
        }
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        assert n_obs == 50
        assert n_art == 10
        assert n_feat == 3
        assert max_s == 5

    def test_1d_array_x_handling(self):
        """Test that 1D X array raises ValueError."""
        model_args = {
            "y": np.zeros(10),
            "X": np.zeros(10),  # 1D array - should be rejected
            "n_artists": 5,
            "max_seq": 3,
        }
        with pytest.raises(ValueError, match="X must be a 2D array"):
            _derive_dimensions_from_model_args(model_args)

    def test_list_x_handling(self):
        """Test dimension derivation with nested list X."""
        model_args = {
            "y": [0] * 25,
            "X": [[0, 0, 0, 0] for _ in range(25)],  # List of lists
            "n_artists": 8,
            "max_seq": 4,
        }
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        assert n_obs == 25
        assert n_art == 8
        assert n_feat == 4
        assert max_s == 4

    def test_none_y_handling(self):
        """Test dimension derivation with None y."""
        model_args = {
            "y": None,
            "X": np.zeros((10, 3)),
            "n_artists": 5,
            "max_seq": 2,
        }
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        assert n_obs == 0
        assert n_art == 5
        assert n_feat == 3
        assert max_s == 2

    def test_none_x_handling(self):
        """Test dimension derivation with None X."""
        model_args = {
            "y": np.zeros(10),
            "X": None,
            "n_artists": 5,
            "max_seq": 2,
        }
        n_obs, n_art, n_feat, max_s = _derive_dimensions_from_model_args(model_args)

        assert n_obs == 10
        assert n_art == 5
        assert n_feat == 0
        assert max_s == 2


class TestCacheKeyModelSignature:
    """The cache key must change with model structure, not just dimensions."""

    def test_model_signature_changes_hash(self):
        base = compute_config_hash(100, 20, 5, 10)
        gated = compute_config_hash(100, 20, 5, 10, model_signature={"latent_process": "ar1"})
        assert base != gated

    def test_different_signatures_differ(self):
        rw = compute_config_hash(100, 20, 5, 10, model_signature={"latent_process": "rw"})
        ar1 = compute_config_hash(100, 20, 5, 10, model_signature={"latent_process": "ar1"})
        assert rw != ar1

    def test_signature_key_order_irrelevant(self):
        a = compute_config_hash(100, 20, 5, 10, model_signature={"a": 1, "b": 2})
        b = compute_config_hash(100, 20, 5, 10, model_signature={"b": 2, "a": 1})
        assert a == b

    def test_descriptor_hash_in_signature_changes_key(self):
        aoty = compute_config_hash(100, 20, 5, 10, model_signature={"descriptor_hash": "aaa"})
        aero = compute_config_hash(100, 20, 5, 10, model_signature={"descriptor_hash": "bbb"})
        assert aoty != aero

    def test_numpyro_version_is_part_of_key(self, monkeypatch):
        import numpyro

        before = compute_config_hash(100, 20, 5, 10)
        monkeypatch.setattr(numpyro, "__version__", "0.0.0-test")
        after = compute_config_hash(100, 20, 5, 10)
        assert before != after

    def test_derive_model_signature_extracts_gates(self):
        from panelcast.preflight.full_check import derive_model_signature

        signature = derive_model_signature(
            {
                "n_exponent": 0.5,
                "learn_n_exponent": True,
                "likelihood_df": 4.0,
                "X": np.zeros((5, 2)),  # non-scalar keys are ignored
            }
        )
        assert signature == {
            "n_exponent": 0.5,
            "learn_n_exponent": True,
            "likelihood_df": 4.0,
        }
