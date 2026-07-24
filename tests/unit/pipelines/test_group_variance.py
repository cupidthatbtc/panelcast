"""Per-group variance config plumbing (#271)."""

from __future__ import annotations

import pytest

from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator


class TestConfigPlumbing:
    def test_invalid_value_rejected(self):
        with pytest.raises(ValueError, match="group_variance"):
            PipelineConfig(group_variance="nope")

    def test_per_group_with_pooling_off_rejected(self):
        with pytest.raises(ValueError, match="entity_group_pooling"):
            PipelineConfig(group_variance="per_group", entity_group_pooling=False)

    def test_per_group_with_pooling_on_valid(self):
        config = PipelineConfig(group_variance="per_group", entity_group_pooling=True)
        assert config.group_variance == "per_group"

    def test_invalid_scale_rejected(self):
        with pytest.raises(ValueError, match="tau_group_sigma_scale"):
            PipelineConfig(tau_group_sigma_scale=-1.0)

    def test_command_string_records_non_default(self, tmp_path):
        config = PipelineConfig(group_variance="per_group", entity_group_pooling=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        assert "--group-variance per_group" in orch._build_command_string()
