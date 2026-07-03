"""Tests for staleness stamps on shared data artifacts."""

import json
from pathlib import Path

import pytest

from panelcast.pipelines.errors import StaleArtifactError
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
)
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import PipelineStage
from panelcast.pipelines.stamps import (
    DATA_STAGE_ROOTS,
    read_stamp,
    stamp_path,
    verify_stamps,
    write_stamp,
)


def _manifest(run_id: str = "run-A") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at="2026-07-02T00:00:00",
        command="test",
        flags={},
        seed=42,
        git=GitStateModel(commit="abc", branch="main", dirty=False, untracked_count=0),
        environment=EnvironmentInfo(
            python_version="3.11",
            jax_version="0.0",
            numpyro_version=None,
            arviz_version=None,
            platform="test",
            pixi_lock_hash=None,
        ),
        input_hashes={},
        stage_hashes={},
        stages_completed=[],
        stages_skipped=[],
        outputs={},
        success=False,
    )


def _orchestrator(tmp_path: Path) -> PipelineOrchestrator:
    orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path / "outputs")
    orch.manifest = _manifest()
    orch.run_dir = tmp_path / "outputs" / "run-A"
    return orch


def _noop_stage(name: str) -> PipelineStage:
    return PipelineStage(name=name, description="test", run_fn=lambda ctx: None)


class TestStampIO:
    def test_write_read_round_trip(self, tmp_path):
        root = tmp_path / "data" / "processed"
        payload = write_stamp(root, "data", "hash123", "run-A")
        assert read_stamp(root) == payload
        assert payload["stage"] == "data"
        assert payload["input_hash"] == "hash123"
        assert payload["run_id"] == "run-A"
        assert payload["written_at"]

    def test_read_missing_returns_none(self, tmp_path):
        assert read_stamp(tmp_path / "nowhere") is None

    def test_read_corrupt_returns_none(self, tmp_path):
        stamp_path(tmp_path).write_text("{not json", encoding="utf-8")
        assert read_stamp(tmp_path) is None


class TestVerifyStamps:
    def test_matching_stamp_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        recorded = write_stamp(DATA_STAGE_ROOTS["features"], "features", "h1", "run-A")
        verify_stamps({"features": recorded}, "train")

    def test_foreign_run_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        recorded = write_stamp(DATA_STAGE_ROOTS["features"], "features", "h1", "run-A")
        write_stamp(DATA_STAGE_ROOTS["features"], "features", "h2", "run-B")
        with pytest.raises(StaleArtifactError) as exc_info:
            verify_stamps({"features": recorded}, "evaluate")
        message = str(exc_info.value)
        assert "data/features" in message
        assert "run-B" in message
        assert "run-A" in message
        assert exc_info.value.exit_code == 7

    def test_changed_inputs_same_run_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        recorded = write_stamp(DATA_STAGE_ROOTS["splits"], "splits", "h1", "run-A")
        write_stamp(DATA_STAGE_ROOTS["splits"], "splits", "h2", "run-A")
        with pytest.raises(StaleArtifactError):
            verify_stamps({"splits": recorded}, "train")

    def test_missing_current_stamp_tolerated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        verify_stamps({"features": {"run_id": "run-A", "input_hash": "h1"}}, "train")

    def test_unknown_stage_and_empty_payload_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        verify_stamps({"mystery": {"run_id": "x"}, "features": {}}, "train")


class TestOrchestratorIntegration:
    def test_data_stage_writes_stamp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = _orchestrator(tmp_path)
        orch._execute_stage(_noop_stage("data"))
        stamp = read_stamp(DATA_STAGE_ROOTS["data"])
        assert stamp is not None
        assert stamp["run_id"] == "run-A"
        assert stamp["input_hash"] == orch.manifest.stage_hashes["data"]
        assert orch.manifest.data_stamps["data"] == stamp

    def test_consumer_observes_existing_stamps(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        foreign = write_stamp(DATA_STAGE_ROOTS["features"], "features", "h1", "run-Z")
        orch = _orchestrator(tmp_path)
        orch._execute_stage(_noop_stage("train"))
        assert orch.manifest.data_stamps["features"] == foreign

    def test_consumer_fails_on_regenerated_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = _orchestrator(tmp_path)
        recorded = write_stamp(DATA_STAGE_ROOTS["features"], "features", "h1", "run-A")
        orch.manifest.data_stamps["features"] = recorded
        write_stamp(DATA_STAGE_ROOTS["features"], "features", "h2", "run-B")
        with pytest.raises(StaleArtifactError):
            orch._execute_stage(_noop_stage("evaluate"))

    def test_mid_run_regeneration_detected_between_consumers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_stamp(DATA_STAGE_ROOTS["features"], "features", "h1", "run-Z")
        orch = _orchestrator(tmp_path)
        orch._execute_stage(_noop_stage("train"))
        write_stamp(DATA_STAGE_ROOTS["features"], "features", "h2", "run-B")
        with pytest.raises(StaleArtifactError):
            orch._execute_stage(_noop_stage("evaluate"))

    def test_same_world_passes_through_consumers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for name, root in DATA_STAGE_ROOTS.items():
            write_stamp(root, name, "h1", "run-Z")
        orch = _orchestrator(tmp_path)
        orch._execute_stage(_noop_stage("train"))
        orch._execute_stage(_noop_stage("evaluate"))
        assert "train" in orch.manifest.stages_completed
        assert "evaluate" in orch.manifest.stages_completed


class TestManifestField:
    def test_data_stamps_round_trip(self):
        manifest = _manifest()
        manifest.data_stamps["features"] = {"run_id": "run-A", "input_hash": "h1"}
        restored = RunManifest.model_validate_json(manifest.model_dump_json())
        assert restored.data_stamps == manifest.data_stamps

    def test_old_manifest_without_field_loads(self):
        payload = json.loads(_manifest().model_dump_json())
        payload.pop("data_stamps")
        restored = RunManifest.model_validate(payload)
        assert restored.data_stamps == {}
