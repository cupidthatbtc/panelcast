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


def _write_run(
    tmp_path: Path,
    run_id: str = "run_a",
    tamper: str | None = None,
    *,
    run_owned_input: bool = False,
) -> Path:
    output_base = tmp_path / "outputs"
    run_dir = output_base / run_id
    artifact = run_dir / "evaluation" / "metrics.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")

    raw = tmp_path / "raw.csv"
    raw.write_text("a,b\n1,2\n", encoding="utf-8")

    input_hashes = {str(raw): sha256_path(raw)}
    if run_owned_input:
        # What a run that failed at `evaluate` records: the train stage's
        # run-scoped products, hashed before the stage body ran.
        for name in ("manifest.json", "training_summary.json"):
            product = run_dir / "models" / name
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_text(json.dumps({"stage": "train"}), encoding="utf-8")
            input_hashes[str(product)] = sha256_path(product)

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
        input_hashes=input_hashes,
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


def _quarantine(output_base: Path, run_id: str = "run_a") -> Path:
    """Move a run under `outputs/failed/`, leaving its manifest as written."""
    failed = output_base / "failed"
    failed.mkdir(exist_ok=True)
    (output_base / run_id).rename(failed / run_id)
    return failed / run_id


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

    def test_a_manifest_recording_nothing_still_passes(self, tmp_path):
        # Nothing recorded is not the same as recorded-and-unprovable. Named
        # for the manifest: the artifact is still on disk here, so this is
        # also the shape of a manifest whose records were erased, which
        # `runs verify` cannot tell apart — see the completeness note in
        # docs/CLI.md.
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

    def test_failed_runs_verify_their_own_products_where_they_moved(self, tmp_path):
        # The #420 regression, end to end: a run that failed at `evaluate` has
        # already recorded the train stage's run-scoped products as *inputs*,
        # so quarantine moves them out from under their recorded paths. Before
        # the re-rooting they came back MISSING on every such run — a false
        # integrity alarm on exactly the run someone is debugging.
        base = _write_run(tmp_path, run_owned_input=True)
        moved = _quarantine(base)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 0, result.output
        assert "MISSING  input" not in result.output
        recorded = base / "run_a" / "models" / "manifest.json"
        checked = moved / "models" / "manifest.json"
        # The line names where it looked, not only what the manifest said.
        assert f"OK       input {checked} (recorded as {recorded})" in result.output

    def test_a_shared_input_is_checked_where_the_manifest_recorded_it(self, tmp_path):
        # The other half of run-ownership: the raw data root did not move with
        # the run, so nothing may map it — and a change to it is still caught.
        base = _write_run(tmp_path, run_owned_input=True)
        _quarantine(base)
        raw = tmp_path / "raw.csv"
        raw.write_text("a,b\n9,9\n", encoding="utf-8")

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert f"MODIFIED input {raw} (raw data changed since this run)" in result.output

    def test_a_deleted_run_owned_input_still_reads_as_missing(self, tmp_path):
        # Mapping unconditionally reports where the artifact *should* be; it
        # must not turn a deletion into a pass by pointing somewhere it can
        # find something else.
        base = _write_run(tmp_path, run_owned_input=True)
        moved = _quarantine(base)
        (moved / "models" / "manifest.json").unlink()

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert f"MISSING  input {moved / 'models' / 'manifest.json'} (recorded as" in result.output

    def test_the_evaluate_stage_really_declares_inputs_inside_the_run(self, tmp_path):
        # The premise the fixture stands on, asserted rather than assumed: if
        # the run-scoped layout stopped putting `evaluate`'s inputs under the
        # run directory, the cases above would keep passing while testing a
        # manifest shape production no longer writes.
        from panelcast.paths import ArtifactPaths
        from panelcast.pipelines.stages import make_stage_evaluate

        run_dir = tmp_path / "outputs" / "run_a"
        stage = make_stage_evaluate(paths=ArtifactPaths.for_run(run_dir))

        run_owned = [p for p in stage.input_paths if run_dir in p.parents]

        assert [p.name for p in run_owned] == ["manifest.json", "training_summary.json"]

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


class TestTheInputMappingIsBounded:
    """What re-rooting an input may not do, since nothing judges it afterwards."""

    def test_a_tail_that_climbs_out_of_the_run_is_not_mapped(self, tmp_path):
        # `verify_output_records` maps first and refuses the result by
        # containment; the input pass only stats and hashes, so the guard has
        # to sit in the mapping. Left at the recorded spelling, which is where
        # it was checked before any re-rooting existed.
        from panelcast.pipelines.output_integrity import reroot_contained, reroot_under

        moved = tmp_path / "outputs" / "failed" / "run_a"
        escaping = Path("outputs") / "run_a" / ".." / ".." / "secret.txt"

        # The premise: the pair still matches, so it is containment declining
        # the result rather than the mapping never firing.
        assert reroot_under(escaping, moved) != escaping
        assert reroot_contained(escaping, moved) == escaping

        # ...and a tail that stays inside still maps, or the guard would be
        # refusing everything and the assertion above would prove nothing.
        inside = Path("outputs") / "run_a" / "models" / "manifest.json"
        assert reroot_contained(inside, moved) == moved / "models" / "manifest.json"

    def test_a_path_the_run_does_not_own_is_returned_untouched(self, tmp_path):
        from panelcast.pipelines.output_integrity import reroot_contained

        moved = tmp_path / "outputs" / "failed" / "run_a"
        shared = Path("data") / "raw" / "albums.csv"

        assert reroot_contained(shared, moved) == shared


class TestReproduceOnAQuarantinedRun:
    """`runs reproduce` reads the same recorded inputs, and aborts on them (#420)."""

    def _reproduce(self, tmp_path, monkeypatch, *, delete: bool = False):
        from panelcast.pipelines import orchestrator

        base = _write_run(tmp_path, run_owned_input=True)
        moved = _quarantine(base)
        if delete:
            (moved / "models" / "manifest.json").unlink()
        monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: 0)
        return runner.invoke(app, ["runs", "reproduce", "run_a", "--output-base", str(base)])

    def test_a_moved_input_does_not_abort_the_reproduction(self, tmp_path, monkeypatch):
        # The hard-abort half of #420: unmapped, a run that failed at
        # `evaluate` could never be reproduced, because the very products it
        # failed while reading were reported gone.
        result = self._reproduce(tmp_path, monkeypatch)

        assert "ABORT" not in result.output, result.output
        assert result.exit_code == 0, result.output

    def test_an_input_that_really_is_gone_still_aborts(self, tmp_path, monkeypatch):
        # Tolerance, not permissiveness: the gate exists to refuse a
        # reproduction whose inputs no longer stand behind it.
        result = self._reproduce(tmp_path, monkeypatch, delete=True)

        assert result.exit_code == 1
        assert "ABORT: recorded input missing" in result.output
