"""Tests for DatasetDescriptor plumbing through stages and orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.errors import PipelineError
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import (
    StageContext,
    build_pipeline_stages,
    make_stage_data,
    make_stage_splits,
)


def _aero_descriptor() -> DatasetDescriptor:
    return DatasetDescriptor(
        name="aero",
        entity_col="Airframe",
        target_col="Perf_Score",
        target_bounds=(0.0, 10.0),
        model_prefix="perf",
        n_obs_col="Sensor_Samples",
        secondary_target_col=None,
        secondary_prefix=None,
        secondary_n_obs_col=None,
        processed_name_template="perf_minobs_{min_ratings}",
        raw_path_env="AERO_DATASET_PATH",
        raw_path_default="data/raw/flights.csv",
    )


class TestDefaultStagePaths:
    """Default (no descriptor) must reproduce today's exact paths."""

    def test_data_stage_outputs_unchanged(self):
        stage = make_stage_data()
        assert [str(p).replace("\\", "/") for p in stage.output_paths] == [
            "data/processed/cleaned_all.parquet",
            "data/processed/user_score_minratings_5.parquet",
            "data/processed/user_score_minratings_10.parquet",
            "data/processed/user_score_minratings_25.parquet",
            "data/processed/critic_score.parquet",
        ]

    def test_splits_stage_input_unchanged(self):
        stage = make_stage_splits(min_ratings=10)
        assert [str(p).replace("\\", "/") for p in stage.input_paths] == [
            "data/processed/user_score_minratings_10.parquet"
        ]

    def test_build_pipeline_stages_default_names(self):
        names = [s.name for s in build_pipeline_stages()]
        assert names == ["data", "splits", "features", "train", "evaluate", "predict", "report"]


class TestDescriptorDrivenPaths:
    def test_data_stage_outputs_from_descriptor(self):
        stage = make_stage_data(descriptor=_aero_descriptor())
        outputs = [str(p).replace("\\", "/") for p in stage.output_paths]
        assert "data/processed/perf_minobs_10.parquet" in outputs
        # Secondary model disabled -> no critic dataset.
        assert not any("critic" in p for p in outputs)

    def test_data_stage_input_from_descriptor_env(self, monkeypatch):
        monkeypatch.delenv("AERO_DATASET_PATH", raising=False)
        stage = make_stage_data(descriptor=_aero_descriptor())
        assert str(stage.input_paths[0]).replace("\\", "/") == "data/raw/flights.csv"
        monkeypatch.setenv("AERO_DATASET_PATH", "elsewhere/flights.csv")
        stage = make_stage_data(descriptor=_aero_descriptor())
        assert str(stage.input_paths[0]).replace("\\", "/") == "elsewhere/flights.csv"

    def test_descriptor_yaml_in_input_paths(self, tmp_path):
        yaml_path = tmp_path / "aero.yaml"
        yaml_path.write_text("name: aero\n", encoding="utf-8")
        stage = make_stage_data(descriptor=_aero_descriptor(), descriptor_path=yaml_path)
        assert yaml_path in stage.input_paths

    def test_splits_input_from_descriptor(self):
        stage = make_stage_splits(min_ratings=5, descriptor=_aero_descriptor())
        assert [str(p).replace("\\", "/") for p in stage.input_paths] == [
            "data/processed/perf_minobs_5.parquet"
        ]


class TestStageContextDescriptor:
    def test_default_descriptor_is_aoty(self):
        ctx = StageContext(
            run_dir=Path("outputs/test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.descriptor == DatasetDescriptor()


class TestResumeDescriptorGuard:
    def _orchestrator_with_manifest(self, flags: dict) -> PipelineOrchestrator:
        orch = PipelineOrchestrator(PipelineConfig())
        manifest = MagicMock()
        manifest.flags = flags
        orch.manifest = manifest
        return orch

    def test_resume_roundtrip_restores_dataset(self):
        recorded = DatasetDescriptor().descriptor_hash()
        orch = self._orchestrator_with_manifest(
            {"dataset": None, "dataset_descriptor_hash": recorded}
        )
        orch._restore_config_from_manifest()
        assert orch.config.dataset is None
        assert orch.descriptor == DatasetDescriptor()

    def test_resume_descriptor_hash_mismatch_raises(self):
        orch = self._orchestrator_with_manifest(
            {"dataset": None, "dataset_descriptor_hash": "deadbeef" * 8}
        )
        with pytest.raises(PipelineError, match="descriptor changed"):
            orch._restore_config_from_manifest()

    def test_legacy_manifest_without_hash_warns_and_defaults(self):
        orch = self._orchestrator_with_manifest({})
        orch._restore_config_from_manifest()
        assert orch.descriptor == DatasetDescriptor()

    def test_fresh_manifest_records_descriptor_hash(self, tmp_path):
        config = PipelineConfig(dry_run=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch._setup_run()
        assert orch.manifest is not None
        assert orch.manifest.flags["dataset"] is None
        assert (
            orch.manifest.flags["dataset_descriptor_hash"] == DatasetDescriptor().descriptor_hash()
        )


class TestBetaBinomialGate:
    """The orchestrator rejects beta_binomial on a non-aggregation descriptor."""

    def _yaml(self, tmp_path, agg: bool) -> Path:
        p = tmp_path / "ds.yaml"
        p.write_text(
            f"name: ds\nn_obs_is_aggregation_count: {str(agg).lower()}\n", encoding="utf-8"
        )
        return p

    def test_gate_raises_for_non_aggregation_descriptor(self, tmp_path):
        config = PipelineConfig(
            likelihood_family="beta_binomial", dataset=str(self._yaml(tmp_path, agg=False))
        )
        with pytest.raises(ValueError, match="aggregation"):
            PipelineOrchestrator(config)

    def test_gate_allows_aggregation_descriptor(self, tmp_path):
        config = PipelineConfig(
            likelihood_family="beta_binomial", dataset=str(self._yaml(tmp_path, agg=True))
        )
        orch = PipelineOrchestrator(config)
        assert orch.descriptor.n_obs_is_aggregation_count is True

    def test_gate_ignores_other_families(self, tmp_path):
        config = PipelineConfig(
            likelihood_family="studentt", dataset=str(self._yaml(tmp_path, agg=False))
        )
        orch = PipelineOrchestrator(config)
        assert orch.descriptor.n_obs_is_aggregation_count is False
