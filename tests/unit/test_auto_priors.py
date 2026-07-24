"""priors: auto — data-derived lognormal locations (#267)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from panelcast.model_preflight import cross_entity_mean_sd, within_entity_step_sd
from panelcast.models.bayes.priors import PriorConfig
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.train_bayes import apply_auto_prior_locs, maybe_apply_auto_priors


def _panel():
    # Three entities, four observations each, distinct levels and wiggle.
    y = np.array([0.0, 0.2, 0.1, 0.3, 5.0, 5.4, 5.2, 5.6, 10.0, 9.7, 9.9, 9.6])
    idx = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return y, idx


class TestDerivation:
    def test_locs_follow_the_documented_convention(self):
        y, idx = _panel()
        priors, record = apply_auto_prior_locs(PriorConfig(), y, idx)
        rw_moment = within_entity_step_sd(y, idx)
        art_moment = cross_entity_mean_sd(y, idx)
        # sigma_artist sits AT its governing moment; sigma_rw half an e-fold
        # below its upper-bound moment.
        assert priors.sigma_artist_lognormal_loc == pytest.approx(math.log(art_moment))
        assert priors.sigma_rw_lognormal_loc == pytest.approx(math.log(rw_moment) - 0.5)
        assert priors.sigma_rw_lognormal_sigma == 0.8
        assert priors.sigma_artist_lognormal_sigma == 0.6
        assert priors.sigma_artist_prior_type == "lognormal"
        assert record["within_entity_step_sd"] == pytest.approx(rw_moment)
        assert record["cross_entity_mean_sd"] == pytest.approx(art_moment)

    def test_degenerate_moments_raise_actionably(self):
        y = np.zeros(6)
        idx = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="degenerate"):
            apply_auto_prior_locs(PriorConfig(), y, idx)

    def test_other_prior_fields_untouched(self):
        y, idx = _panel()
        base = PriorConfig()
        priors, _ = apply_auto_prior_locs(base, y, idx)
        assert priors.rho_scale == base.rho_scale
        assert priors.likelihood_family == base.likelihood_family


class TestSharedHook:
    def test_hook_derives_when_gate_on(self):
        y, idx = _panel()
        model_args = {"y": y, "artist_idx": idx}
        config = PipelineConfig(auto_priors=True)
        priors = maybe_apply_auto_priors(config, PriorConfig(), model_args)
        assert priors.sigma_artist_prior_type == "lognormal"
        assert priors.sigma_rw_lognormal_loc == pytest.approx(
            math.log(within_entity_step_sd(y, idx)) - 0.5
        )

    def test_hook_passes_through_when_gate_off(self):
        y, idx = _panel()
        base = PriorConfig()
        assert maybe_apply_auto_priors(PipelineConfig(), base, {"y": y, "artist_idx": idx}) is base


class TestConfigPlumbing:
    def test_auto_with_explicit_locs_conflicts(self):
        with pytest.raises(ValueError, match="auto_priors"):
            PipelineConfig(auto_priors=True, sigma_rw_lognormal_loc=-5.0)

    def test_auto_alone_is_valid(self):
        assert PipelineConfig(auto_priors=True).auto_priors is True

    def test_descriptor_declares_auto(self, tmp_path):
        dataset = tmp_path / "d.yaml"
        dataset.write_text("name: autod\nauto_priors: true\n", encoding="utf-8")
        config = PipelineConfig(dataset=str(dataset))
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.auto_priors is True

    def test_explicit_config_off_beats_descriptor_on(self, tmp_path):
        dataset = tmp_path / "d.yaml"
        dataset.write_text("name: autod\nauto_priors: true\n", encoding="utf-8")
        config = PipelineConfig(dataset=str(dataset), auto_priors=False)
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.auto_priors is False

    def test_default_resolves_off(self, tmp_path):
        config = PipelineConfig()
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.auto_priors is False

    def test_descriptor_auto_conflicts_with_explicit_locs_at_resolution(self, tmp_path):
        # The conflict must also fire when auto arrives via the descriptor,
        # i.e. inside resolve_model_facts' re-validation, not just at
        # PipelineConfig construction.
        dataset = tmp_path / "d.yaml"
        dataset.write_text("name: autod\nauto_priors: true\n", encoding="utf-8")
        config = PipelineConfig(dataset=str(dataset), sigma_rw_lognormal_loc=-5.0)
        with pytest.raises(ValueError, match="auto_priors"):
            PipelineOrchestrator(config, output_base=tmp_path)

    def test_descriptor_hash_stable_when_unset(self):
        from panelcast.config.descriptor import DatasetDescriptor

        assert (
            DatasetDescriptor().descriptor_hash()
            == "a9e3e20540b1dcb5d6253bd342cff6fd73ed823597428f4e94abd51f8b67b8ec"
        )
        assert (
            DatasetDescriptor(auto_priors=True).descriptor_hash()
            != DatasetDescriptor().descriptor_hash()
        )
