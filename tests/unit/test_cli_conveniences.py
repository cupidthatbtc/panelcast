"""CLI convenience coverage: shell completion, stage --dry-run, runs list,
and the demo --dataset alias."""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

from typer.testing import CliRunner

from panelcast.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_pipeline_mocks(monkeypatch, exit_code: int = 0):
    captured: dict[str, object] = {}

    def fake_config(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
    monkeypatch.setattr(
        "panelcast.pipelines.orchestrator.run_pipeline", lambda config: exit_code
    )
    return captured


class TestShellCompletion:
    """add_completion=True exposes typer's completion installers."""

    def test_completion_options_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--install-completion" in output
        assert "--show-completion" in output


class TestStageDryRun:
    """panelcast stage <name> --dry-run mirrors run's dry-run plumbing."""

    def test_stage_dry_run_default_false(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data"])
        assert result.exit_code == 0
        assert captured["kwargs"]["dry_run"] is False

    def test_stage_data_dry_run(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data", "--dry-run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["dry_run"] is True
        assert captured["kwargs"]["stages"] == ["data"]

    def test_stage_train_dry_run(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "train", "--dry-run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["dry_run"] is True

    def test_stage_sensitivity_dry_run(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "sensitivity", "--dry-run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["dry_run"] is True

    def test_stage_dry_run_in_help(self):
        result = runner.invoke(app, ["stage", "predict", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in strip_ansi(result.output)


def _write_manifest(run_dir, run_id: str, success: bool, stages: list[str]) -> None:
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "created_at": f"2026-07-01T0{len(stages)}:00:00",
        "success": success,
        "stages_completed": stages,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestRunsList:
    """panelcast runs list summarizes run manifests under outputs/."""

    def test_missing_output_dir(self, tmp_path):
        result = runner.invoke(
            app, ["runs", "list", "--output-dir", str(tmp_path / "nope")]
        )
        assert result.exit_code == 0
        assert "does not exist" in strip_ansi(result.output)

    def test_empty_output_dir(self, tmp_path):
        result = runner.invoke(app, ["runs", "list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No runs found" in strip_ansi(result.output)

    def test_lists_runs_with_status_and_stage_counts(self, tmp_path):
        _write_manifest(
            tmp_path / "2026-06-30_1200", "2026-06-30_1200", True, ["data", "splits"]
        )
        _write_manifest(tmp_path / "2026-07-01_0900", "2026-07-01_0900", False, ["data"])
        result = runner.invoke(app, ["runs", "list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "2026-06-30_1200" in output
        assert "2026-07-01_0900" in output
        assert "ok" in output
        assert "incomplete" in output
        line_old = next(li for li in output.splitlines() if "2026-06-30_1200" in li)
        assert " 2 " in line_old or line_old.rstrip().endswith("2")

    def test_marks_latest_and_skips_non_runs(self, tmp_path):
        _write_manifest(tmp_path / "2026-06-30_1200", "2026-06-30_1200", True, ["data"])
        _write_manifest(tmp_path / "2026-07-01_0900", "2026-07-01_0900", True, ["data"])
        # Non-run directories: stage outputs (no manifest) and the failed bucket.
        (tmp_path / "evaluation").mkdir()
        _write_manifest(tmp_path / "failed" / "2026-05-01_0000", "x", False, [])
        os.symlink(
            tmp_path / "2026-07-01_0900", tmp_path / "latest", target_is_directory=True
        )
        result = runner.invoke(app, ["runs", "list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "evaluation" not in output
        assert "2026-05-01_0000" not in output
        marked = next(li for li in output.splitlines() if li.startswith("*"))
        assert "2026-07-01_0900" in marked
        assert "-> 2026-07-01_0900" in output


class TestDemoDatasetAlias:
    """demo accepts --dataset as an alias for --descriptor."""

    def test_dataset_alias_parses_to_descriptor(self):
        result = runner.invoke(app, ["demo", "--dataset", "does/not/exist.yaml"])
        assert result.exit_code == 1
        output = strip_ansi(result.output)
        assert "demo descriptor not found" in output
        assert "does/not/exist.yaml" in output

    def test_dataset_alias_happy_path(self, monkeypatch):
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.run_pipeline", lambda config: 0
        )
        result = runner.invoke(
            app, ["demo", "--dataset", "examples/aerospace/descriptor.yaml"]
        )
        assert result.exit_code == 0
        assert "Demo complete" in strip_ansi(result.output)

    def test_descriptor_still_works(self):
        result = runner.invoke(app, ["demo", "--descriptor", "also/missing.yaml"])
        assert result.exit_code == 1
        assert "demo descriptor not found" in strip_ansi(result.output)

    def test_alias_documented_in_help(self):
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--descriptor" in output
        assert "--dataset" in output
