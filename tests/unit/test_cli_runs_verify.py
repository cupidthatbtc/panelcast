"""`panelcast runs verify` (#169): the manifest as a checkable integrity contract."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    save_run_manifest,
)
from panelcast.utils.hashing import sha256_path

runner = CliRunner()


def _write_run(tmp_path: Path, run_id: str = "run_a", tamper: str | None = None) -> Path:
    output_base = tmp_path / "outputs"
    run_dir = output_base / run_id
    artifact = run_dir / "evaluation" / "metrics.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")

    raw = tmp_path / "raw.csv"
    raw.write_text("a,b\n1,2\n", encoding="utf-8")

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
        input_hashes={str(raw): sha256_path(raw)},
        stage_hashes={},
        stages_completed=["evaluate"],
        stages_skipped=[],
        outputs={"evaluate:metrics": str(artifact)},
        output_hashes={"evaluate:metrics": sha256_path(artifact)},
        success=True,
    )
    save_run_manifest(manifest, run_dir)

    if tamper == "output":
        artifact.write_text(json.dumps({"mae": 1.0}), encoding="utf-8")
    elif tamper == "delete":
        artifact.unlink()
    elif tamper == "input":
        raw.write_text("a,b\n9,9\n", encoding="utf-8")
    return output_base


class TestRunsVerify:
    def test_untouched_run_passes(self, tmp_path):
        base = _write_run(tmp_path)
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
        assert "OK       evaluate:metrics" in result.output

    def test_modified_output_fails(self, tmp_path):
        base = _write_run(tmp_path, tamper="output")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "MODIFIED evaluate:metrics" in result.output

    def test_deleted_output_fails(self, tmp_path):
        base = _write_run(tmp_path, tamper="delete")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "MISSING  evaluate:metrics" in result.output

    def test_changed_raw_input_fails(self, tmp_path):
        base = _write_run(tmp_path, tamper="input")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "raw data changed" in result.output

    def test_pre_090_manifest_reports_not_recorded(self, tmp_path):
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        del payload["output_hashes"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 0
        assert "no hashes recorded" in result.output

    def test_failed_runs_are_resolvable(self, tmp_path):
        base = _write_run(tmp_path)
        (base / "failed").mkdir()
        (base / "run_a").rename(base / "failed" / "run_a")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 0, result.output

    def test_unknown_run_id_is_a_usage_error(self, tmp_path):
        base = _write_run(tmp_path)
        result = runner.invoke(app, ["runs", "verify", "nope", "--output-base", str(base)])
        assert result.exit_code != 0
