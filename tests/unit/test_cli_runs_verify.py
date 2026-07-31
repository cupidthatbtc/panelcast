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
        assert "OK           evaluate:metrics" in result.output

    def test_modified_output_fails(self, tmp_path):
        base = _write_run(tmp_path, tamper="output")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "MODIFIED     evaluate:metrics" in result.output

    def test_deleted_output_fails(self, tmp_path):
        base = _write_run(tmp_path, tamper="delete")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "MISSING      evaluate:metrics" in result.output

    def test_changed_raw_input_fails(self, tmp_path):
        base = _write_run(tmp_path, tamper="input")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "raw data changed" in result.output

    def test_a_manifest_with_no_hashes_is_unverifiable_not_excused(self, tmp_path):
        # Shape cannot tell a pre-0.9.0 run from a modern one someone emptied
        # the hash map on, so the note explains and the verdict still refuses —
        # the skip path treats the same manifest the same way.
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        del payload["output_hashes"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "no hashes recorded" in result.output
        assert "UNVERIFIABLE evaluate:metrics (recorded output has no hash)" in result.output

    def test_a_run_that_recorded_nothing_still_passes(self, tmp_path):
        # Nothing recorded is not the same as recorded-and-unprovable.
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["outputs"] = {}
        payload["output_hashes"] = {}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 0, result.output
        assert "no hashes recorded" not in result.output

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

    def test_every_failure_line_says_why(self, tmp_path):
        # A bare status told an operator which key failed but not what about
        # it; the reason follows the status now, in a column wide enough for
        # the longest verdict.
        base = _write_run(tmp_path, tamper="delete")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert "MISSING      evaluate:metrics (recorded output is missing)" in result.output

    def test_a_hash_with_no_recorded_path_is_unverifiable(self, tmp_path):
        # Previously reported as a bare MISSING, which reads as "the artifact
        # is gone" when the truth is the manifest never said where it was.
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["outputs"] = {}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "UNVERIFIABLE evaluate:metrics (hashed output has no recorded path)" in result.output

    def test_a_recorded_path_with_no_hash_is_unverifiable(self, tmp_path):
        # Previously skipped in silence: the CLI iterated `output_hashes`, so a
        # key present only in `outputs` was never looked at.
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["outputs"]["evaluate:extra"] = str(base / "run_a" / "evaluation" / "extra.json")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])
        assert result.exit_code == 1
        assert "UNVERIFIABLE evaluate:extra (recorded output has no hash)" in result.output

    def test_an_output_outside_the_run_and_the_artifact_roots_is_refused(self, tmp_path):
        # Containment the CLI did not have: a tampered manifest could aim the
        # re-hash at any readable path and have a match reported as OK. The
        # roots are the run dir and the artifact roots, not the whole tree, so
        # this holds wherever the command is run from.
        outside = tmp_path / "outside" / "metrics.json"
        outside.parent.mkdir(parents=True)
        outside.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["outputs"]["evaluate:metrics"] = str(outside)
        payload["output_hashes"]["evaluate:metrics"] = sha256_path(outside)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert "UNBOUND      evaluate:metrics (recorded output path escapes the run roots)" in (
            result.output
        )

    def test_the_output_base_decides_which_copy_is_verified(self, tmp_path, monkeypatch):
        # End to end for the re-rooting order: a stale copy sits where the
        # manifest recorded it, the real run has been archived under a
        # different base, and the two differ — so a green result also says
        # *which* bytes were hashed, not merely which path was formed.
        import shutil

        base = _write_run(tmp_path)
        archive = tmp_path / "archive" / "outputs"
        archive.parent.mkdir()
        shutil.copytree(base, archive)
        stale = base / "run_a" / "evaluation" / "metrics.json"
        stale.write_text(json.dumps({"mae": 9.9}), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(archive)])

        assert result.exit_code == 0, result.output
        assert "OK           evaluate:metrics" in result.output

    def test_another_runs_copy_of_the_same_artifact_is_refused(self, tmp_path):
        # The substitution containment exists for: identical bytes, so the
        # hash matches, but the artifact belongs to a different run.
        base = _write_run(tmp_path)
        _write_run(tmp_path, run_id="run_b")
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["outputs"]["evaluate:metrics"] = str(
            base / "run_b" / "evaluation" / "metrics.json"
        )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert "UNBOUND      evaluate:metrics" in result.output
