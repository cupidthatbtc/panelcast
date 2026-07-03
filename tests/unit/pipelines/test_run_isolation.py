"""Run-isolation suite for the run-scoped output layout (#81).

Drives the real orchestrator with lightweight fake stages whose run_fns
write marker files through ctx.paths, asserting the isolation contract:
back-to-back runs leave each other's product dirs untouched, latest.json
tracks the most recent successful run, resume writes into the original
run dir, and skip-existing still works across runs for the flat data
stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panelcast.paths import ArtifactPaths, resolve_latest
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import PipelineStage


@pytest.fixture
def mock_env():
    with (
        patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
        patch(
            "panelcast.pipelines.orchestrator.verify_environment",
            return_value=MagicMock(is_reproducible=True, pixi_lock_hash="abc123", warnings=[]),
        ),
    ):
        yield


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch, mock_env):
    """Hermetic cwd with a raw input file; returns the output base."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw" / "raw.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path / "outputs"


def _fake_stages(
    paths: ArtifactPaths,
    executed: list[str],
    fail: dict[str, bool] | None = None,
) -> list[PipelineStage]:
    """Data stage on the flat layout, train/evaluate through ctx.paths."""
    fail = fail or {}

    def _run(name: str, root_attr: str, filename: str):
        def run_fn(ctx):
            if fail.get(name):
                raise RuntimeError(f"{name} failed")
            executed.append(name)
            root = getattr(ctx.paths, root_attr)
            root.mkdir(parents=True, exist_ok=True)
            (root / filename).write_text(ctx.manifest.run_id, encoding="utf-8")

        return run_fn

    return [
        PipelineStage(
            name="data",
            description="fake data stage",
            run_fn=_run("data", "processed", "marker.parquet"),
            input_paths=[Path("data/raw/raw.csv")],
            output_paths=[Path("data/processed/marker.parquet")],
        ),
        PipelineStage(
            name="train",
            description="fake train stage",
            run_fn=_run("train", "models", "model.txt"),
            input_paths=[Path("data/processed/marker.parquet")],
            output_paths=[paths.models / "model.txt"],
            depends_on=["data"],
        ),
        PipelineStage(
            name="evaluate",
            description="fake evaluate stage",
            run_fn=_run("evaluate", "evaluation", "metrics.json"),
            input_paths=[paths.models / "model.txt"],
            output_paths=[paths.evaluation / "metrics.json"],
            depends_on=["train"],
        ),
    ]


def _run_pipeline(
    output_base: Path,
    run_id: str,
    executed: list[str] | None = None,
    fail: dict[str, bool] | None = None,
    **config_kwargs,
) -> int:
    """Run the orchestrator with fake stages declared against its paths."""
    executed = executed if executed is not None else []
    config = PipelineConfig(enforce_lockfile=False, **config_kwargs)
    orchestrator = PipelineOrchestrator(config, output_base=output_base)

    def fake_order(stages=None, min_ratings=10, descriptor=None, descriptor_path=None, paths=None):
        return _fake_stages(paths or ArtifactPaths.flat(), executed, fail=fail)

    with (
        patch("panelcast.pipelines.orchestrator.get_execution_order", side_effect=fake_order),
        patch("panelcast.pipelines.orchestrator.generate_run_id", return_value=run_id),
    ):
        return orchestrator.run()


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every product file under root by relative path and exact content.

    pipeline.log.json is excluded: the root logger is global, so a later
    run's early records can reach the previous run's still-attached file
    handler before that run reconfigures logging. Logs are not products.
    """
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "pipeline.log.json"
    }


class TestRunIsolation:
    def test_back_to_back_runs_leave_each_other_untouched(self, isolated_outputs):
        out = isolated_outputs

        assert _run_pipeline(out, run_id="runA") == 0
        run_a_snapshot = _snapshot(out / "runA")
        assert (out / "runA" / "models" / "model.txt").read_text(encoding="utf-8") == "runA"
        assert (out / "runA" / "evaluation" / "metrics.json").read_text(encoding="utf-8") == "runA"

        assert _run_pipeline(out, run_id="runB") == 0

        assert _snapshot(out / "runA") == run_a_snapshot
        assert (out / "runB" / "models" / "model.txt").read_text(encoding="utf-8") == "runB"
        assert (out / "runB" / "evaluation" / "metrics.json").read_text(encoding="utf-8") == "runB"

    def test_latest_json_tracks_most_recent_successful_run(self, isolated_outputs):
        out = isolated_outputs

        assert _run_pipeline(out, run_id="runA") == 0
        pointer = json.loads((out / "latest.json").read_text(encoding="utf-8"))
        assert pointer == {"run_id": "runA", "run_dir": "runA"}

        assert _run_pipeline(out, run_id="runB") == 0
        pointer = json.loads((out / "latest.json").read_text(encoding="utf-8"))
        assert pointer == {"run_id": "runB", "run_dir": "runB"}
        assert resolve_latest(out) == out / "runB"

        # A failed run must not move the pointer.
        assert _run_pipeline(out, run_id="runC", fail={"train": True}) != 0
        assert resolve_latest(out) == out / "runB"

    def test_resume_writes_into_original_run_dir(self, isolated_outputs):
        out = isolated_outputs

        # First attempt fails at evaluate: data + train complete, run moves to failed/.
        assert _run_pipeline(out, run_id="runA", fail={"evaluate": True}) != 0
        assert (out / "failed" / "runA").exists()

        executed: list[str] = []
        assert _run_pipeline(out, run_id="unused", executed=executed, resume="runA") == 0

        # Completed stages are not re-run; evaluate lands in the ORIGINAL run dir.
        assert executed == ["evaluate"]
        assert (out / "runA" / "evaluation" / "metrics.json").read_text(encoding="utf-8") == "runA"
        assert not (out / "failed" / "runA").exists()
        assert not (out / "unused").exists()
        assert resolve_latest(out) == out / "runA"

    def test_skip_existing_skips_unchanged_flat_data_stage(self, isolated_outputs):
        out = isolated_outputs

        assert _run_pipeline(out, run_id="runA") == 0

        executed: list[str] = []
        orchestrator_exit = _run_pipeline(out, run_id="runB", executed=executed, skip_existing=True)
        assert orchestrator_exit == 0

        # The flat data stage is skipped (inputs unchanged, outputs shared);
        # the run-scoped products are rebuilt into the new run dir.
        assert executed == ["train", "evaluate"]
        manifest = json.loads((out / "runB" / "manifest.json").read_text(encoding="utf-8"))
        assert "data" in manifest["stages_skipped"]
        assert (out / "runB" / "models" / "model.txt").read_text(encoding="utf-8") == "runB"


class TestLatestPointerFailure:
    def test_pointer_write_failure_is_nonfatal(self, tmp_path, monkeypatch):
        """A failed latest.json replace warns instead of failing the run."""
        import panelcast.pipelines.orchestrator as orch_mod

        out = tmp_path / "outputs"
        orch = PipelineOrchestrator(PipelineConfig(), output_base=out)
        orch.run_dir = out / "runX"
        orch.run_dir.mkdir(parents=True)

        def boom(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(orch_mod.os, "replace", boom)
        orch._write_latest_pointer()
        assert not (out / "latest.json").exists()
