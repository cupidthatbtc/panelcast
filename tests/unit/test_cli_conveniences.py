"""CLI convenience coverage: shell completion, stage --dry-run, runs list,
runs history, run --tag, and the demo --dataset alias."""

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

    def test_corrupt_manifest_gets_distinct_status(self, tmp_path):
        _write_manifest(tmp_path / "2026-06-30_1200", "2026-06-30_1200", True, ["data"])
        bad = tmp_path / "2026-07-01_0900"
        bad.mkdir()
        (bad / "manifest.json").write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["runs", "list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        bad_line = next(li for li in output.splitlines() if "2026-07-01_0900" in li)
        assert "corrupt" in bad_line
        assert "incomplete" not in bad_line

    def test_unreadable_output_dir_scan_degrades_to_empty(self, tmp_path, monkeypatch):
        from pathlib import Path

        def boom(self):
            raise OSError("scan failed")

        monkeypatch.setattr(Path, "iterdir", boom)
        result = runner.invoke(app, ["runs", "list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No runs found" in strip_ansi(result.output)


def _write_history_run(
    base,
    run_id: str,
    *,
    stamp_hash: str | None = "feat-aaa",
    mae: float = 5.0,
    cov80: float = 0.79,
    elpd: float = -3.2,
    version: str | None = "0.7.0",
    tag: str | None = None,
    success: bool = True,
    dry_run: bool = False,
) -> None:
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "created_at": "2026-07-01T09:00:00",
        "success": success,
        "stages_completed": ["train", "evaluate"],
        "flags": {"num_chains": 4, "num_samples": 1000, "dry_run": dry_run},
        "duration_seconds": 3600.0,
    }
    if version is not None:
        manifest["version"] = version
    if tag is not None:
        manifest["tag"] = tag
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metrics = {
        "point_metrics": {"mae": mae, "rmse": mae + 2.0, "r2": 0.5},
        "calibration": {
            "coverages": {
                "0.80": {"empirical": cov80},
                "0.95": {"empirical": 0.95},
            },
            "coverage_tolerance": 0.03,
            "wis": 4.0,
        },
        "crps": {"mean_crps": 3.9},
        "info_criteria": {"heldout_elpd": {"elpd_per_obs": elpd}},
        "feature_stamp": (
            {
                "stage": "features",
                "input_hash": stamp_hash,
                "run_id": "feat-src-run",
                "written_at": "2026-07-01T00:00:00+00:00",
            }
            if stamp_hash is not None
            else None
        ),
    }
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


class TestRunsHistory:
    """panelcast runs history groups runs by feature stamp and flags drift."""

    def test_missing_output_dir(self, tmp_path):
        result = runner.invoke(
            app, ["runs", "history", "--output-dir", str(tmp_path / "nope")]
        )
        assert result.exit_code == 0
        assert "does not exist" in strip_ansi(result.output)

    def test_no_evaluated_runs(self, tmp_path):
        _write_manifest(tmp_path / "2026-06-30_1200", "2026-06-30_1200", True, ["data"])
        result = runner.invoke(app, ["runs", "history", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No evaluated runs found" in strip_ansi(result.output)

    def test_groups_by_feature_stamp_with_epoch_breaks(self, tmp_path):
        _write_history_run(tmp_path, "run1", stamp_hash="feat-aaa")
        _write_history_run(tmp_path, "run2", stamp_hash="feat-aaa")
        _write_history_run(tmp_path, "run3", stamp_hash="feat-bbb")
        result = runner.invoke(app, ["runs", "history", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "epoch 1: feature stamp feat-aaa" in output
        assert "epoch 2: feature stamp feat-bbb" in output
        lines = output.splitlines()
        epoch2_at = next(i for i, li in enumerate(lines) if "feat-bbb" in li)
        assert next(i for i, li in enumerate(lines) if "run2" in li) < epoch2_at
        assert next(i for i, li in enumerate(lines) if "run3" in li) > epoch2_at

    def test_flags_drift_within_epoch_only(self, tmp_path):
        # Epoch aaa: run1 is the best-MAE reference; run2 drifts on every axis.
        _write_history_run(tmp_path, "run1", mae=5.0, cov80=0.79, elpd=-3.2)
        _write_history_run(tmp_path, "run2", mae=6.0, cov80=0.70, elpd=-4.0)
        # Epoch bbb: much worse in absolute terms, but alone -> never flagged.
        _write_history_run(tmp_path, "run3", stamp_hash="feat-bbb", mae=9.0, cov80=0.60)
        result = runner.invoke(app, ["runs", "history", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        line1 = next(li for li in output.splitlines() if "run1" in li)
        line2 = next(li for li in output.splitlines() if "run2" in li)
        line3 = next(li for li in output.splitlines() if "run3" in li)
        assert "*" not in line1
        assert "6.000*" in line2
        assert "0.700*" in line2
        assert "0.950*" not in line2  # only the drifted coverage level is starred
        assert "-4.000*" in line2
        assert "*" not in line3
        assert "drift within epoch" in output

    def test_tolerates_corrupt_dry_run_and_failed(self, tmp_path):
        _write_history_run(tmp_path, "good1")
        _write_history_run(tmp_path, "dry1", dry_run=True)
        _write_history_run(tmp_path, "incomplete1", success=False)
        bad = tmp_path / "corrupt1"
        (bad / "evaluation").mkdir(parents=True)
        (bad / "manifest.json").write_text("{not json", encoding="utf-8")
        (bad / "evaluation" / "metrics.json").write_text("{}", encoding="utf-8")
        _write_history_run(tmp_path / "failed", "failedrun")
        result = runner.invoke(app, ["runs", "history", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "good1" in output
        for absent in ("dry1", "incomplete1", "corrupt1", "failedrun"):
            assert absent not in output

    def test_legacy_run_without_version_renders_question_mark(self, tmp_path):
        _write_history_run(tmp_path, "legacy1", version=None)
        _write_history_run(tmp_path, "tagged2", version="0.7.0", tag="exp-a")
        result = runner.invoke(app, ["runs", "history", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        legacy_line = next(li for li in output.splitlines() if "legacy1" in li)
        tagged_line = next(li for li in output.splitlines() if "tagged2" in li)
        assert " ? " in legacy_line
        assert "0.7.0" in tagged_line
        assert "exp-a" in tagged_line

    def test_json_output_shape(self, tmp_path):
        _write_history_run(tmp_path, "run1", mae=5.0, tag="base")
        _write_history_run(tmp_path, "run2", mae=6.0, cov80=0.70, elpd=-4.0)
        _write_history_run(tmp_path, "run3", stamp_hash="feat-bbb")
        result = runner.invoke(
            app, ["runs", "history", "--output-dir", str(tmp_path), "--json"]
        )
        assert result.exit_code == 0
        # Unrelated import-time log lines can precede the payload under pytest.
        lines = strip_ansi(result.output).splitlines()
        start = next(i for i, li in enumerate(lines) if li == "[")
        payload = json.loads("\n".join(lines[start:]))
        assert [g["feature_stamp"]["input_hash"] for g in payload] == [
            "feat-aaa",
            "feat-bbb",
        ]
        runs = {r["run_id"]: r for g in payload for r in g["runs"]}
        assert runs["run1"]["tag"] == "base"
        assert runs["run1"]["version"] == "0.7.0"
        assert runs["run1"]["metrics"]["mae"] == 5.0
        assert runs["run1"]["num_chains"] == 4
        assert runs["run1"]["drift"] == []
        assert set(runs["run2"]["drift"]) == {"mae", "elpd_per_obs", "coverage@0.80"}
        assert runs["run3"]["drift"] == []

    def test_null_metrics_tolerated(self, tmp_path):
        _write_history_run(tmp_path, "run1")
        metrics_path = tmp_path / "run1" / "evaluation" / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["info_criteria"] = {"status": "unavailable"}
        del payload["crps"]
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(app, ["runs", "history", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        line = next(li for li in strip_ansi(result.output).splitlines() if "run1" in li)
        assert "?" in line


class TestRunTagOption:
    """panelcast run --tag lands on PipelineConfig for the manifest."""

    def test_tag_passed_to_config(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--tag", "exp-1", "--dry-run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["tag"] == "exp-1"

    def test_tag_defaults_to_none(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--dry-run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["tag"] is None


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
