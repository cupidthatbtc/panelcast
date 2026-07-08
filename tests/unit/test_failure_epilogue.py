"""Failure epilogue + `runs why` (#163): actionable failure UX."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.pipelines.errors import (
    ConvergenceError,
    EnvironmentError,
    GpuMemoryError,
    PipelineError,
    failure_hint,
)

runner = CliRunner()


class TestFailureHint:
    def test_typed_errors_have_hints(self):
        assert "--allow-divergences" in failure_hint(ConvergenceError("x", stage="train"))
        assert "pixi install" in failure_hint(EnvironmentError("x", stage="setup"))
        assert "--exclude-rw-raw-from-collection" in failure_hint(
            GpuMemoryError("x", stage="train")
        )

    def test_unknown_error_has_none(self):
        assert failure_hint(RuntimeError("x")) is None
        assert failure_hint(PipelineError("x", stage="train")) is None


class TestRecentEvents:
    def test_processor_captures_and_rolls(self):
        from panelcast.utils.logging import _RECENT_EVENTS, _capture_recent, recent_events

        _RECENT_EVENTS.clear()
        for i in range(25):
            _capture_recent(None, "info", {"event": f"e{i}", "n": i})
        _capture_recent(None, "debug", {"event": "quiet"})
        events = recent_events()
        assert len(events) == 20  # ring rolled
        assert events[-1]["event"] == "e24"
        assert all(e["event"] != "quiet" for e in events)  # debug not captured


class TestHandleFailureWritesForensics:
    def _orchestrator(self, tmp_path, monkeypatch):
        from panelcast.pipelines.manifest import EnvironmentInfo
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
        from panelcast.utils.git_state import GitState

        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_git_state",
            lambda: GitState(commit="abc", branch="t", dirty=False, untracked_count=0),
        )
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_environment",
            lambda: EnvironmentInfo(
                python_version="3.14",
                jax_version="0.8.2",
                numpyro_version=None,
                arviz_version=None,
                platform="Linux",
                pixi_lock_hash=None,
            ),
        )
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path / "outputs")
        orch._setup_run()
        return orch

    def test_failure_json_survives_the_move(self, tmp_path, monkeypatch):
        orch = self._orchestrator(tmp_path, monkeypatch)
        run_id = orch.run_dir.name
        orch._handle_failure(ConvergenceError("rhat 1.2", stage="train"), "train")
        failed_dir = tmp_path / "outputs" / "failed" / run_id
        payload = json.loads((failed_dir / "failure.json").read_text(encoding="utf-8"))
        assert payload["stage"] == "train"
        assert payload["exception_type"] == "ConvergenceError"
        assert "--allow-divergences" in payload["hint"]
        assert payload["resume_command"] == f"panelcast run --resume {run_id}"

    def test_epilogue_prints_resume_and_hint(self, tmp_path, monkeypatch, capsys):
        orch = self._orchestrator(tmp_path, monkeypatch)
        run_id = orch.run_dir.name
        orch._handle_failure(ConvergenceError("rhat 1.2", stage="train"), "train")
        err = capsys.readouterr().err
        assert f"panelcast run --resume {run_id}" in err
        assert "panelcast runs why" in err


def _failed_run(base: Path, run_id: str, with_failure_json: bool = True) -> Path:
    from panelcast.pipelines.manifest import (
        EnvironmentInfo,
        GitStateModel,
        RunManifest,
        save_run_manifest,
    )

    run_dir = base / "failed" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=run_id,
        created_at="2026-07-08T00:00:00Z",
        command="panelcast run",
        flags={},
        seed=42,
        git=GitStateModel(commit="a" * 40, branch="main", dirty=False, untracked_count=0),
        environment=EnvironmentInfo(
            python_version="3.14",
            jax_version="0.8.2",
            numpyro_version=None,
            arviz_version=None,
            platform="Linux",
            pixi_lock_hash=None,
        ),
        input_hashes={},
        stage_hashes={},
        stages_completed=["data", "splits"],
        stages_skipped=[],
        outputs={},
        success=False,
        error="boom",
    )
    save_run_manifest(manifest, run_dir)
    if with_failure_json:
        (run_dir / "failure.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "stage": "train",
                    "exception_type": "GpuMemoryError",
                    "message": "OOM at 23.1 GB",
                    "traceback_tail": ["  raise GpuMemoryError(...)"],
                    "stages_completed": ["data", "splits"],
                    "hint": "free VRAM",
                    "resume_command": f"panelcast run --resume {run_id}",
                    "recent_events": [{"level": "info", "event": "fitting_model"}],
                }
            ),
            encoding="utf-8",
        )
    return run_dir


class TestRunsWhy:
    def test_pretty_prints_failure_json(self, tmp_path):
        _failed_run(tmp_path / "outputs", "run_x")
        result = runner.invoke(
            app, ["runs", "why", "run_x", "--output-base", str(tmp_path / "outputs")]
        )
        assert result.exit_code == 0, result.output
        assert "GpuMemoryError: OOM at 23.1 GB" in result.output
        assert "free VRAM" in result.output
        assert "fitting_model" in result.output

    def test_defaults_to_most_recent_failed(self, tmp_path):
        _failed_run(tmp_path / "outputs", "2026-07-08_a")
        _failed_run(tmp_path / "outputs", "2026-07-08_b")
        result = runner.invoke(app, ["runs", "why", "--output-base", str(tmp_path / "outputs")])
        assert result.exit_code == 0, result.output
        assert "2026-07-08_b" in result.output

    def test_pre_feature_run_falls_back_to_manifest(self, tmp_path):
        _failed_run(tmp_path / "outputs", "run_old", with_failure_json=False)
        result = runner.invoke(
            app, ["runs", "why", "run_old", "--output-base", str(tmp_path / "outputs")]
        )
        assert result.exit_code == 0, result.output
        assert "no failure.json recorded" in result.output
        assert "boom" in result.output

    def test_no_failed_runs_is_clean(self, tmp_path):
        (tmp_path / "outputs").mkdir()
        result = runner.invoke(app, ["runs", "why", "--output-base", str(tmp_path / "outputs")])
        assert result.exit_code == 0
        assert "no failed runs" in result.output
