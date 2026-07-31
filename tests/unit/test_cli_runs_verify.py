"""`panelcast runs verify` (#169): the manifest as a checkable integrity contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
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
        assert "contents changed since this run" in result.output

    def test_a_recorded_spelling_that_only_normalises_is_not_called_a_move(
        self, tmp_path, monkeypatch
    ):
        # `Path("./raw.csv") == Path("raw.csv")` but the strings differ, so
        # comparing them would announce a re-rooting that never happened. The
        # unowned branch — nothing was mapped, so nothing can have moved, and
        # the line is the manifest's own string.
        base = _write_run(tmp_path)
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["input_hashes"] = {"./raw.csv": sha256_path(tmp_path / "raw.csv")}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 0, result.output
        assert "OK       input ./raw.csv" in result.output
        assert "recorded as" not in result.output

    def test_an_unreadable_input_is_a_verdict_not_a_traceback(self, tmp_path, monkeypatch):
        # The output pass catches the same triple at its hash site. Raising
        # here would abandon the remaining inputs, the stamps and the lockfile
        # over one unreadable file, and the pass reads more paths than it did.
        base = _write_run(tmp_path)

        def refuse(path, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("panelcast.utils.hashing.sha256_path", refuse)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        # The summary only prints once the whole chain has run, so it is what
        # separates a verdict from a traceback that took the stamps and the
        # lockfile with it. Counted rather than totalled: the output pass binds
        # `sha256_path` at import and so is not stubbed here, and a change to
        # that should not fail this test for an unrelated reason.
        assert result.output.count("unreadable: permission denied") == 1
        assert "FAILED:" in result.output

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
        assert f"MODIFIED input {raw} (contents changed since this run)" in result.output

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

    def test_an_edited_run_owned_input_still_reads_as_modified(self, tmp_path):
        # The half that pins re-rooting to the *hash* rather than to existence:
        # the mapping changes which bytes get read, so the digest has to be
        # taken at the moved location and still be the recorded one.
        base = _write_run(tmp_path, run_owned_input=True)
        moved = _quarantine(base)
        edited = moved / "models" / "manifest.json"
        edited.write_text(json.dumps({"stage": "tampered"}), encoding="utf-8")

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert f"MODIFIED input {edited} (recorded as" in result.output

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

        assert sorted(p.name for p in run_owned) == ["manifest.json", "training_summary.json"]

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


class TestWhatTheRunOwns:
    """Where the ownership answer stops, since nothing judges it afterwards."""

    def test_a_tail_that_climbs_out_of_the_run_is_not_owned(self, tmp_path):
        # `verify_output_records` maps first and refuses the result by
        # containment; the input callers only stat and hash, so the guard has
        # to sit in the mapping.
        from panelcast.pipelines.output_integrity import reroot_under, run_owned_path

        moved = tmp_path / "outputs" / "failed" / "run_a"
        escaping = Path("outputs") / "run_a" / ".." / ".." / "secret.txt"

        # The premise: the pair still matches, so it is containment declining
        # the result rather than the mapping never firing.
        assert reroot_under(escaping, moved) != escaping
        assert run_owned_path(escaping, moved) is None

        # ...and a tail that stays inside still maps, or the guard would be
        # refusing everything and the assertion above would prove nothing.
        inside = Path("outputs") / "run_a" / "models" / "manifest.json"
        assert run_owned_path(inside, moved) == moved / "models" / "manifest.json"

    def test_a_mapped_location_that_will_not_resolve_is_not_owned(self, tmp_path, monkeypatch):
        # Fail closed: with the mapped location unreadable there is no evidence
        # it is inside the run, so the caller falls back to the recorded path
        # rather than reading one it could not confirm.
        from panelcast.pipelines.output_integrity import run_owned_path

        run_dir = tmp_path / "outputs" / "run_a"
        mapped = run_dir / "models" / "manifest.json"
        real = Path.resolve

        def refuse(self, strict=False):
            if self == mapped:
                raise OSError("symlink loop")
            return real(self, strict)

        monkeypatch.setattr(Path, "resolve", refuse)
        recorded = Path("outputs") / "run_a" / "models" / "manifest.json"

        assert run_owned_path(recorded, run_dir) is None

    def test_a_path_the_run_does_not_own_has_no_location(self, tmp_path):
        from panelcast.pipelines.output_integrity import run_owned_path

        moved = tmp_path / "outputs" / "failed" / "run_a"

        assert run_owned_path(Path("data") / "raw" / "albums.csv", moved) is None

    def test_an_active_runs_own_product_is_owned_though_nothing_moved(self, tmp_path):
        # For an active run the mapping moves nothing — but it still rewrites
        # the spelling onto `run_dir`, as the assertion below shows, so a
        # changed path cannot answer ownership in either direction. Containment
        # can, which is what lets `runs reproduce` rely on it.
        from panelcast.pipelines.output_integrity import run_owned_path

        active = tmp_path / "outputs" / "run_a"
        recorded = Path("outputs") / "run_a" / "models" / "manifest.json"

        assert run_owned_path(recorded, active) == active / "models" / "manifest.json"

    def test_a_flat_layout_product_is_not_owned(self, tmp_path):
        # The limit of a containment-scoped answer, pinned rather than implied.
        # Under the flat layout a stage's model inputs are recorded relative to
        # the project root, where nothing distinguishes them from data the run
        # did not produce — so `runs reproduce` still gates on them, and
        # pruning or overwriting `models/` still aborts an earlier run.
        from panelcast.pipelines.output_integrity import run_owned_path

        run_dir = tmp_path / "outputs" / "run_a"

        assert run_owned_path(Path("models") / "manifest.json", run_dir) is None


class TestWhatUnownedStillReads:
    """Unowned is not unread: where such a path is checked, and how it reads."""

    def test_an_unresolvable_recorded_spelling_names_what_was_checked(
        self, tmp_path, monkeypatch
    ):
        # The label's error path. With resolution failing there is no evidence
        # the two spellings name one file, so naming the recorded one alone
        # would put the pre-#420 false alarm back on the screen through the
        # branch nothing exercises.
        base = _write_run(tmp_path, run_owned_input=True)
        moved = _quarantine(base)
        recorded = base / "run_a" / "models" / "manifest.json"
        real = Path.resolve

        def refuse(self, strict=False):
            if self == recorded:
                raise OSError("symlink loop")
            return real(self, strict)

        monkeypatch.setattr(Path, "resolve", refuse)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 0, result.output
        assert f"OK       input {moved / 'models' / 'manifest.json'}\n" in result.output

    def test_an_active_run_verified_by_an_absolute_base_claims_no_move(
        self, tmp_path, monkeypatch
    ):
        # The rewritten spelling reaches the report, so a label decided on
        # spelling would announce a relocation for an artifact sitting exactly
        # where it was written.
        base = _write_run(tmp_path, run_owned_input=True)
        product = base / "run_a" / "models" / "manifest.json"
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = Path("outputs") / "run_a" / "models" / "manifest.json"
        payload["input_hashes"] = {str(recorded): sha256_path(product)}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 0, result.output
        assert f"OK       input {recorded}" in result.output
        assert "recorded as" not in result.output

    def _symlinked_product_run(self, tmp_path, target: str = "shared"):
        """A run whose `models/` points outside it, recorded as input and output."""
        run_dir = _write_run(tmp_path) / "run_a"
        shared = tmp_path / target
        shared.mkdir()
        product = shared / "manifest.json"
        product.write_text("{}", encoding="utf-8")
        try:
            (run_dir / "models").symlink_to(shared, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable, giving up the parity claim: {exc}")
        recorded = run_dir / "models" / "manifest.json"
        manifest_path = run_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["outputs"]["train:models"] = str(recorded)
        payload["output_hashes"]["train:models"] = sha256_path(product)
        payload["input_hashes"] = {str(recorded): sha256_path(product)}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return run_dir, recorded

    def test_a_symlinked_out_product_is_unowned_but_still_read(self, tmp_path):
        # Resolved containment, deliberately — but stated for what it does.
        # Ownership is symmetric with the output side; the *consequence* is
        # not, because an unowned path is checked where the manifest recorded
        # it, and on an active run that reads straight through the link. So
        # the same file is OK as an input and UNBOUND as an output, and the
        # run fails on the output. Both verdicts through the CLI, so the claim
        # compares two things of the same kind against the roots that ship.
        from panelcast.pipelines.output_integrity import run_owned_path

        run_dir, recorded = self._symlinked_product_run(tmp_path)

        assert run_owned_path(recorded, run_dir) is None

        result = runner.invoke(
            app, ["runs", "verify", "run_a", "--output-base", str(run_dir.parent)]
        )

        assert result.exit_code == 1
        assert "UNBOUND      train:models (recorded output path escapes the run roots)" in (
            result.output
        )
        assert f"OK       input {recorded}" in result.output

    def test_what_the_symlinked_out_product_gives_up_is_the_re_rooting(self, tmp_path):
        # The cost, asserted rather than described: quarantine and the
        # recorded location is gone, so the input the active run verified now
        # reads MISSING. That is #420's own failure surviving for this layout
        # — on a run whose outputs are UNBOUND regardless, which is why the
        # trade is worth the callers agreeing about ownership.
        run_dir, recorded = self._symlinked_product_run(tmp_path)
        base = run_dir.parent
        _quarantine(base)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert f"MISSING  input {recorded}" in result.output

    def test_a_symlink_into_an_artifact_root_is_where_the_two_sides_part(
        self, tmp_path, monkeypatch
    ):
        # The other target, and the one the parity argument does not cover.
        # Ownership asks about the run directory alone; output verification
        # also accepts the artifact roots, so a link into the project-level
        # `models/` verifies as an output while the quarantined input reads
        # MISSING — on the same file, on a run that otherwise passes. Asserted
        # rather than argued away, since the case the other tests pick (a
        # target outside every root) is the one where the sides agree.
        monkeypatch.chdir(tmp_path)
        run_dir, recorded = self._symlinked_product_run(tmp_path, target="models")
        base = run_dir.parent
        _quarantine(base)

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1, result.output
        assert "OK           train:models" in result.output
        assert f"MISSING  input {recorded}" in result.output

    def test_an_escaping_recorded_input_is_checked_where_it_was_recorded(self, tmp_path):
        # What unowned means, stated so the guard is not read as a refusal to
        # read: an active run's escaping recorded input is still stat'd and
        # hashed at the location the manifest names, exactly as it was before
        # any re-rooting existed. The guard bounds where the *mapping* may aim.
        base = _write_run(tmp_path)
        outside = tmp_path / "secret.txt"
        outside.write_text("real", encoding="utf-8")
        recorded_at = base / "run_a" / ".." / ".." / "secret.txt"
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["input_hashes"] = {str(recorded_at): sha256_path(outside)}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 0, result.output
        assert f"OK       input {recorded_at}" in result.output

    def test_the_input_pass_does_not_follow_the_mapping_out_of_the_run(self, tmp_path):
        # End to end, and discriminating: a file whose bytes match the recorded
        # hash sits exactly where the unguarded mapping would land, so
        # following it would report this run clean. Quarantine is what makes
        # the two places differ — the mapped `../../` climbs to `<base>`, the
        # recorded one to `<tmp>` — so the verdict cannot be a coincidence.
        from panelcast.pipelines.output_integrity import reroot_under

        base = _write_run(tmp_path)
        decoy = base / "secret.txt"
        decoy.write_text("real", encoding="utf-8")
        recorded_at = base / "run_a" / ".." / ".." / "secret.txt"
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["input_hashes"] = {str(recorded_at): sha256_path(decoy)}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        moved = _quarantine(base)

        # The premise, since the case is only discriminating while it holds:
        # the pairing rule fires on this absolute spelling, and unguarded it
        # lands on the decoy rather than merely somewhere else.
        assert reroot_under(recorded_at, moved).resolve() == decoy.resolve()

        result = runner.invoke(app, ["runs", "verify", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert f"MISSING  input {recorded_at}" in result.output
        assert "recorded as" not in result.output


class TestReproduceOnAQuarantinedRun:
    """The pre-flight gate is about raw data, not the run's own products (#420)."""

    def _reproduce(
        self,
        tmp_path,
        monkeypatch,
        *,
        prune: bool = False,
        drift: bool = False,
        vanish: bool = False,
        resolved_config: bool = False,
    ):
        from panelcast.pipelines import orchestrator

        base = _write_run(tmp_path, run_owned_input=True)
        moved = _quarantine(base)
        if resolved_config:
            # `seed` is the discriminator: a second mapped field reproduce does
            # not clear, so its arrival proves this tier was used rather than
            # a partial file being rejected for the manifest fallback.
            (moved / "resolved_config.yaml").write_text(
                "seed: 4242\nskip_existing: true\n", encoding="utf-8"
            )
        if prune:
            shutil.rmtree(moved / "models")
        if drift:
            (tmp_path / "raw.csv").write_text("a,b\n9,9\n", encoding="utf-8")
        if vanish:
            (tmp_path / "raw.csv").unlink()
        # Captured rather than discarded: the skip below rests on what this
        # call is made with, so a stub that threw it away would leave the
        # premise untested.
        self.launched = []
        monkeypatch.setattr(
            orchestrator,
            "run_pipeline",
            lambda config, **kw: self.launched.append((config, kw)) or 0,
        )
        return runner.invoke(app, ["runs", "reproduce", "run_a", "--output-base", str(base)])

    def test_a_quarantined_runs_own_products_do_not_abort_it(self, tmp_path, monkeypatch):
        # The hard-abort half of #420: read at their recorded paths, the very
        # products the run failed while reading were reported gone, so a run
        # could not be reproduced precisely because it had failed.
        result = self._reproduce(tmp_path, monkeypatch)

        assert "ABORT" not in result.output, result.output
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("resolved_config", [False, True])
    def test_the_reproduction_regenerates_what_the_gate_stopped_checking(
        self, tmp_path, monkeypatch, resolved_config
    ):
        # The premise the skip rests on, asserted where it is decided rather
        # than quoted from the docstring: a reproduction that resumed or
        # skipped could reuse a run-owned input the gate no longer proves is
        # there, turning an early abort into a stage-level crash midway.
        #
        # Both config tiers, because they are not equally interesting. In the
        # pre-0.9.0 fallback `resume is None` is nearly free — no flag records
        # one. `resolved_config.yaml` *can* carry `skip_existing`, so that is
        # the tier where the premise could be inherited rather than set.
        result = self._reproduce(tmp_path, monkeypatch, resolved_config=resolved_config)

        assert result.exit_code == 0, result.output
        (config, kwargs) = self.launched[0]
        if resolved_config:
            # ...and this tier was the one that answered, or the assertions
            # below would hold because the file went unread.
            assert config.seed == 4242
        assert config.resume is None
        assert config.skip_existing is False
        # Both argument channels: a keyword taking precedence over the config
        # would leave the assertions above reading defaults nothing acts on.
        # Named rather than exhaustive, so an unrelated keyword added to the
        # call does not read as the premise breaking.
        assert not {"resume", "skip_existing"} & set(kwargs)

    def test_the_planted_flag_is_one_a_resolved_config_can_carry(self, tmp_path):
        # The other half of the True case's premise: `skip_existing` is a
        # mapped key, so it is inheritable at all and clearing it is a thing
        # `runs reproduce` has to do rather than get for free. `resume` is not
        # mapped, which is why the case plants this flag and not that one.
        from panelcast.config.pipeline_yaml import PIPELINE_YAML_MAPPING, load_resolved_config

        resolved = tmp_path / "resolved_config.yaml"
        resolved.write_text("skip_existing: true\n", encoding="utf-8")

        assert load_resolved_config(resolved)["skip_existing"] is True
        assert "resume" not in PIPELINE_YAML_MAPPING

    def test_pruning_the_failed_runs_models_does_not_abort_it_either(
        self, tmp_path, monkeypatch
    ):
        # One step further out, and the reason the gate skips run-owned inputs
        # rather than merely re-rooting them: cleaning up after a failed
        # training run would otherwise make it permanently unreproducible.
        result = self._reproduce(tmp_path, monkeypatch, prune=True)

        assert "ABORT" not in result.output, result.output
        assert result.exit_code == 0, result.output

    def test_raw_data_that_drifted_still_aborts(self, tmp_path, monkeypatch):
        # Tolerance, not permissiveness. What the gate is for is untouched:
        # data changing underneath the comparison invalidates it.
        result = self._reproduce(tmp_path, monkeypatch, drift=True)

        assert result.exit_code == 1
        assert "ABORT: raw input changed since the run" in result.output

    def test_raw_data_that_vanished_still_aborts(self, tmp_path, monkeypatch):
        # The other half of the same gate, and the one whose message the
        # run-owned skip took over: what is left of it is external data only.
        result = self._reproduce(tmp_path, monkeypatch, vanish=True)

        assert result.exit_code == 1
        assert f"ABORT: recorded input missing: {tmp_path / 'raw.csv'}" in result.output

    def test_a_product_the_run_directory_does_not_hold_is_still_gated(
        self, tmp_path, monkeypatch
    ):
        # The documented limit, at the gate rather than at the helper. Under a
        # layout that keeps products outside the run directory — flat here,
        # symlinked-out equivalently — containment cannot tell them from data
        # the run did not produce, so the skip does not reach them and the
        # ordinary act of running again over `models/` still stops an earlier
        # run's reproduction.
        from panelcast.pipelines import orchestrator

        monkeypatch.chdir(tmp_path)
        base = _write_run(tmp_path, run_owned_input=True)
        product = tmp_path / "models" / "manifest.json"
        product.parent.mkdir()
        product.write_text("{}", encoding="utf-8")
        manifest_path = base / "run_a" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = Path("models") / "manifest.json"
        payload["input_hashes"] = {str(recorded): sha256_path(product)}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        _quarantine(base)
        monkeypatch.setattr(orchestrator, "run_pipeline", lambda config, **kw: 0)
        product.write_text('{"run": "a later one"}', encoding="utf-8")

        result = runner.invoke(app, ["runs", "reproduce", "run_a", "--output-base", str(base)])

        assert result.exit_code == 1
        assert f"ABORT: raw input changed since the run: {recorded}" in result.output

    def test_raw_data_that_cannot_be_read_aborts_legibly(self, tmp_path, monkeypatch):
        # The gate's whole job is turning a late failure into an early legible
        # one, so it must not be the thing that raises. `runs verify` says the
        # same about the same file one function away.
        def refuse(path, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("panelcast.utils.hashing.sha256_path", refuse)
        result = self._reproduce(tmp_path, monkeypatch)

        assert result.exit_code == 1
        assert "ABORT: recorded input unreadable" in result.output
        assert "permission denied" in result.output
