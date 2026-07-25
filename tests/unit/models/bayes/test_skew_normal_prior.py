"""Skew-normal latent population prior (#232): parity, sites, shape.

"normal" must be bit-identical to the legacy path (no new sites);
"skew_normal" draws the initial entity effects from a learned-alpha
skew-normal population, standardized so mu_artist / sigma_artist keep
their meanings.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import PriorConfig

_N_OBS, _N_FEAT, _N_ART = 12, 2, 4


def _model_args(priors: PriorConfig, n_artists: int = _N_ART) -> dict:
    rng = np.random.default_rng(0)
    return {
        "artist_idx": jnp.array([i % _N_ART for i in range(_N_OBS)], dtype=jnp.int32),
        "album_seq": jnp.array([(i // _N_ART) + 1 for i in range(_N_OBS)], dtype=jnp.int32),
        "prev_score": jnp.full(_N_OBS, 70.0),
        "X": jnp.asarray(rng.standard_normal((_N_OBS, _N_FEAT)), dtype=jnp.float32),
        "y": jnp.asarray(rng.normal(70.0, 8.0, _N_OBS), dtype=jnp.float32),
        "n_artists": n_artists,
        "max_seq": 3,
        "priors": priors,
        "target_bounds": (0.0, 100.0),
        "likelihood_df": 4.0,
        "ar_center": 70.0,
    }


def _seeded_trace(args: dict) -> dict:
    model = make_score_model("user")
    with handlers.seed(rng_seed=0):
        return handlers.trace(model).get_trace(**args)


class TestNormalParity:
    def test_normal_is_bit_identical_to_legacy(self):
        base = _seeded_trace(_model_args(PriorConfig()))
        explicit = _seeded_trace(_model_args(PriorConfig(entity_effect_prior_type="normal")))
        assert set(base) == set(explicit)
        for site, record in base.items():
            np.testing.assert_array_equal(
                np.asarray(record["value"]),
                np.asarray(explicit[site]["value"]),
                err_msg=f"normal-path draw changed at site '{site}'",
            )

    def test_normal_has_no_skew_sites(self):
        trace = _seeded_trace(_model_args(PriorConfig()))
        assert not any("skew" in site for site in trace)


class TestSkewNormal:
    def test_adds_exactly_the_new_sites(self):
        off = _seeded_trace(_model_args(PriorConfig()))
        on = _seeded_trace(
            _model_args(PriorConfig(entity_effect_prior_type="skew_normal"))
        )
        added = set(on) - set(off)
        removed = set(off) - set(on)
        assert added == {
            "user_entity_skew_alpha",
            "user_entity_skew_abs",
            "user_entity_skew_sym",
        }
        # The reparam decentered site and its plate disappear: init effects
        # become a deterministic of the skew construction.
        assert removed == {"user_init_artist_effect_decentered", "user_artists"}

    def test_zerosum_param_rejected(self):
        args = _model_args(
            PriorConfig(
                entity_effect_prior_type="skew_normal", artist_effect_param="zerosum"
            )
        )
        with pytest.raises(ValueError, match="noncentered"):
            _seeded_trace(args)

    def test_unknown_prior_type_rejected(self):
        args = _model_args(PriorConfig(entity_effect_prior_type="nope"))
        with pytest.raises(ValueError, match="entity_effect_prior_type"):
            _seeded_trace(args)

    def test_alpha_zero_recovers_the_symmetric_draw(self):
        """delta=0 => init = mu + sigma * z_sym exactly (unit-SD symmetric)."""
        args = _model_args(PriorConfig(entity_effect_prior_type="skew_normal"))
        model = make_score_model("user")
        pinned = {
            "user_mu_artist": jnp.asarray(50.0),
            "user_sigma_artist": jnp.asarray(4.0),
            "user_entity_skew_alpha": jnp.asarray(0.0),
            "user_entity_skew_abs": jnp.asarray([9.0, 9.0, 9.0, 9.0]),  # must not matter
            "user_entity_skew_sym": jnp.asarray([1.0, -1.0, 0.5, 0.0]),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        init = np.asarray(trace["user_init_artist_effect"]["value"])
        np.testing.assert_allclose(init, 50.0 + 4.0 * np.array([1.0, -1.0, 0.5, 0.0]), rtol=1e-5)

    def test_population_is_standardized_and_skewed(self):
        """With alpha pinned high, the standardized effects have positive
        sample skewness, near-zero mean, and near-unit SD."""
        n = 400
        args = _model_args(PriorConfig(entity_effect_prior_type="skew_normal"), n_artists=n)
        model = make_score_model("user")
        pinned = {
            "user_mu_artist": jnp.asarray(0.0),
            "user_sigma_artist": jnp.asarray(1.0),
            "user_entity_skew_alpha": jnp.asarray(3.0),
        }
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(handlers.substitute(model, pinned)).get_trace(**args)
        init = np.asarray(trace["user_init_artist_effect"]["value"])
        assert init.shape == (n,)
        assert abs(init.mean()) < 0.15
        assert 0.85 < init.std() < 1.15
        centered = init - init.mean()
        skewness = float((centered**3).mean() / (centered**2).mean() ** 1.5)
        # SkewNormal(alpha=3) has skewness ~0.67; sampling noise at n=400
        # keeps it well above 0.3.
        assert skewness > 0.3


class TestConfigPlumbing:
    def test_invalid_value_rejected(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="entity_effect_prior_type"):
            PipelineConfig(entity_effect_prior_type="nope")

    def test_skew_with_zerosum_rejected(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="noncentered"):
            PipelineConfig(
                entity_effect_prior_type="skew_normal", artist_effect_param="zerosum"
            )

    def test_invalid_scale_rejected(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="entity_skew_alpha_scale"):
            PipelineConfig(entity_skew_alpha_scale=0.0)

    def test_command_string_records_non_default(self, tmp_path):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(entity_effect_prior_type="skew_normal")
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        assert "--entity-effect-prior-type skew_normal" in orch._build_command_string()