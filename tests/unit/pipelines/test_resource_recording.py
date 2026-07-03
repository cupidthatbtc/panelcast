"""Tests for expected-vs-actual resource recording (#78)."""

from types import SimpleNamespace

import numpy as np

from panelcast.models.bayes.fit import MCMCConfig
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
)
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import PipelineStage
from panelcast.pipelines.train_bayes import _build_resource_usage
from panelcast.pipelines.training_summary import TrainingSummary


def _model_args(n_obs=100, n_features=5, n_artists=20, max_seq=8):
    return {
        "y": np.zeros(n_obs),
        "X": np.zeros((n_obs, n_features)),
        "n_artists": n_artists,
        "max_seq": max_seq,
    }


def _fit_result(peak_bytes, runtime=12.5):
    return SimpleNamespace(peak_gpu_memory_bytes=peak_bytes, runtime_seconds=runtime)


class TestBuildResourceUsage:
    def test_gpu_run_records_ratio(self):
        config = MCMCConfig(num_warmup=10, num_samples=10, num_chains=2, seed=0)
        usage = _build_resource_usage(
            _model_args(), config, _fit_result(peak_bytes=2 * 1024**3), False
        )
        assert usage["expected_gb"] > 0
        assert usage["actual_peak_gb"] == 2.0
        assert usage["ratio"] == round(2.0 / usage["expected_gb"], 3)
        assert usage["wall_clock_seconds"] == 12.5

    def test_cpu_run_records_none_peak(self):
        config = MCMCConfig(num_warmup=10, num_samples=10, num_chains=2, seed=0)
        usage = _build_resource_usage(_model_args(), config, _fit_result(None), False)
        assert usage["actual_peak_gb"] is None
        assert usage["ratio"] is None
        assert usage["expected_gb"] > 0

    def test_exclusion_flag_lowers_expected(self):
        config = MCMCConfig(num_warmup=10, num_samples=500, num_chains=4, seed=0)
        args = _model_args(n_artists=500, max_seq=40)
        with_rw = _build_resource_usage(args, config, _fit_result(None), False)
        without_rw = _build_resource_usage(args, config, _fit_result(None), True)
        assert without_rw["expected_gb"] < with_rw["expected_gb"]


class TestSummaryField:
    def test_resource_usage_round_trips(self):
        usage = {
            "expected_gb": 1.5,
            "actual_peak_gb": 2.0,
            "ratio": 1.333,
            "wall_clock_seconds": 60.0,
        }
        summary = TrainingSummary(resource_usage=usage)
        assert summary.to_json_dict()["resource_usage"] == usage

    def test_legacy_summary_without_field(self):
        summary = TrainingSummary(model_type="user_score")
        assert "resource_usage" not in summary.to_json_dict()


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="run-A",
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


class TestOrchestratorRecording:
    def _orchestrator(self, tmp_path):
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path / "outputs")
        orch.manifest = _manifest()
        orch.run_dir = tmp_path / "outputs" / "run-A"
        return orch

    def test_stage_duration_recorded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = self._orchestrator(tmp_path)
        stage = PipelineStage(name="report", description="t", run_fn=lambda ctx: None)
        orch._execute_stage(stage)
        assert "report" in orch.manifest.stage_durations
        assert orch.manifest.stage_durations["report"] >= 0.0

    def test_resource_usage_captured_from_run_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        orch = self._orchestrator(tmp_path)
        usage = {"expected_gb": 1.0, "actual_peak_gb": 2.0, "ratio": 2.0}
        stage = PipelineStage(
            name="report",
            description="t",
            run_fn=lambda ctx: {"resource_usage": usage},
        )
        orch._execute_stage(stage)
        assert orch.manifest.resources["report"] == usage

    def test_manifest_fields_round_trip(self):
        manifest = _manifest()
        manifest.stage_durations["train"] = 12.5
        manifest.resources["train"] = {"expected_gb": 1.0}
        restored = RunManifest.model_validate_json(manifest.model_dump_json())
        assert restored.stage_durations == manifest.stage_durations
        assert restored.resources == manifest.resources
