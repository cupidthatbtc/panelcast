"""The skip path and `runs verify` agree about what proves a manifest (#385).

Both re-hash the outputs a manifest recorded, and before #385 they did it
twice. These feed the *same* manifest to both and assert the same verdict, so a
change to one that the other does not get shows up here rather than as drift
nobody notices until an artifact is trusted by one caller and refused by the
other.

Where they legitimately differ — `runs verify` re-roots a quarantined run,
skip-existing only follows the active pointer — the difference is asserted
explicitly rather than left implicit.
"""

from __future__ import annotations

import json
from dataclasses import fields
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
from panelcast.pipelines.stages import PipelineStage
from panelcast.utils.hashing import sha256_path

runner = CliRunner()

STAGE = "evaluate"


class Fixture:
    """One run directory, one recorded output, and both callers over it."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.output_base = tmp_path / "outputs"
        self.run_dir = self.output_base / "run_a"
        self.artifact = self.run_dir / "evaluation" / "metrics.json"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        self.key = f"{STAGE}:{self.artifact.as_posix()}"
        self.source = tmp_path / "raw.csv"
        self.source.write_text("a,b\n1,2\n", encoding="utf-8")

        self.stage = PipelineStage(
            name=STAGE,
            description="fake",
            run_fn=None,
            input_paths=[self.source],
            output_paths=[self.artifact],
        )
        self.outputs = {self.key: str(self.artifact)}
        self.output_hashes = {self.key: sha256_path(self.artifact)}

    def _manifest(self) -> RunManifest:
        return RunManifest(
            run_id="run_a",
            created_at="2026-07-31T00:00:00Z",
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
                fingerprint="0" * 16,
            ),
            input_hashes={},
            stage_hashes={STAGE: self.stage.compute_input_hash()},
            stages_completed=[STAGE],
            stages_skipped=[],
            outputs=dict(self.outputs),
            output_hashes=dict(self.output_hashes),
            success=True,
        )

    def skip_accepts(self, allowed_roots: list[Path] | None = None) -> bool:
        """Whether the incremental skip path trusts the recorded outputs.

        The default is the run dir, so the parity cases isolate the per-key
        rules; `skip_accepts_on_production_roots` runs the same manifest
        through the roots the orchestrator passes. The stage's own
        `_default_roots()` fallback is not reachable from here — that path is
        covered directly in `test_skip_output_verification.py`.
        """
        roots = [self.run_dir] if allowed_roots is None else allowed_roots
        return self.stage.skip_decision(self._manifest(), allowed_roots=roots).skip

    def skip_accepts_on_production_roots(self) -> bool:
        """The same, through the roots the orchestrator actually passes."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            PipelineConfig(dry_run=True), output_base=self.output_base
        )
        return self.stage.skip_decision(
            self._manifest(),
            allowed_roots=orchestrator._output_verification_roots(self.run_dir),
        ).skip

    def verify_accepts(self) -> bool:
        """Whether `runs verify` reports the run as matching its manifest."""
        save_run_manifest(self._manifest(), self._verify_dir())
        result = runner.invoke(
            app, ["runs", "verify", "run_a", "--output-base", str(self.output_base)]
        )
        return result.exit_code == 0

    def _verify_dir(self) -> Path:
        """Wherever the run lives now — `runs verify` resolves either."""
        failed = self.output_base / "failed" / "run_a"
        return failed if failed.exists() else self.run_dir

    def quarantine(self) -> None:
        """Move the run under `outputs/failed/`, leaving the manifest as-is."""
        (self.output_base / "failed").mkdir(parents=True, exist_ok=True)
        self.run_dir.rename(self.output_base / "failed" / "run_a")


@pytest.fixture
def fx(tmp_path, monkeypatch):
    # Pinned so a repo-local `--basetemp` cannot quietly bring `tmp_path`
    # inside a relative artifact root and flip the containment cases.
    monkeypatch.chdir(tmp_path)
    return Fixture(tmp_path)


class TestBothCallersAgree:
    """Same manifest in, same integrity verdict out."""

    def test_valid_manifest_is_accepted_by_both(self, fx):
        assert fx.skip_accepts() is True
        assert fx.verify_accepts() is True
        # Not only through the stubbed roots: the roots the orchestrator
        # actually passes have to accept the artifact too, or the parity cases
        # below are measuring a configuration production never runs.
        assert fx.skip_accepts_on_production_roots() is True

    def test_an_escaping_path_is_refused_on_the_production_roots(self, fx):
        # A *dynamic* key, so nothing but containment is behind it — pointing a
        # declared key elsewhere is refused by the binding first, which is why
        # an earlier version of this test never reached containment at all.
        outside = fx.tmp_path / "outside.json"
        outside.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        key = f"{STAGE}:dataset_hash"
        fx.outputs[key] = str(outside)
        fx.output_hashes[key] = sha256_path(outside)

        assert fx.skip_accepts_on_production_roots() is False
        assert fx.skip_accepts() is False
        # ...and the class contract: same manifest in, same verdict out. This
        # is the one production-roots refusal, so it is the one place the CLI
        # side could have diverged unnoticed.
        assert fx.verify_accepts() is False

    def test_a_modified_output_is_refused_by_both(self, fx):
        fx.artifact.write_text(json.dumps({"mae": 1.0}), encoding="utf-8")
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_a_substituted_output_is_refused_by_both(self, fx):
        # Same size, same shape, different bytes: existence proves nothing.
        fx.artifact.write_text(json.dumps({"mae": 9.9}), encoding="utf-8")
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_a_missing_output_is_refused_by_both(self, fx):
        fx.artifact.unlink()
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_a_path_without_a_hash_is_unverifiable_to_both(self, fx):
        fx.output_hashes.clear()
        fx.output_hashes["evaluate:other"] = "0" * 64
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_a_hash_without_a_path_is_unverifiable_to_both(self, fx):
        # The key-map half that used to be silently skipped on the skip path
        # and reported as a bare MISSING by the CLI.
        fx.outputs.clear()
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_a_half_recorded_key_blocks_an_otherwise_valid_manifest(self, fx):
        # The union, on both sides: everything the stage declares verifies, but
        # one extra key is hashed and not recorded. Neither caller may treat
        # "I have no path for this" as "nothing to check".
        fx.output_hashes["evaluate:dataset_hash"] = "0" * 64
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

        del fx.output_hashes["evaluate:dataset_hash"]
        fx.outputs["evaluate:dataset_hash"] = str(fx.run_dir / "dataset_hash.txt")
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_an_empty_hash_is_unverifiable_to_both(self, fx):
        fx.output_hashes[fx.key] = ""
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_an_empty_hash_map_is_unverifiable_to_both(self, fx):
        # The whole-map limit of the same rule: shape alone cannot separate a
        # pre-0.9.0 run from a modern manifest someone emptied, so neither
        # caller may excuse it.
        fx.output_hashes.clear()
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_an_empty_manifest_is_accepted_by_both(self, fx):
        # ...and nothing recorded is not the same as recorded-and-unprovable.
        # Named for the manifest, not the run: the fixture's artifact is still
        # on disk, so this is a run whose records are gone, which is
        # indistinguishable from one that produced nothing — see
        # `test_an_erased_dynamic_record_is_invisible_to_both`.
        fx.outputs.clear()
        fx.output_hashes.clear()
        fx.stage.output_paths.clear()
        assert fx.skip_accepts() is True
        assert fx.verify_accepts() is True

    def test_a_stage_declaring_nothing_still_sees_a_hash_only_key(self, fx):
        # The lenient direction: with no declared outputs and nothing in
        # `outputs`, the skip path used to shortcut to True and never look at
        # a key the manifest hashed. It is the caller that decides whether to
        # *reuse* artifacts, so it must not be the softer of the two.
        fx.outputs.clear()
        fx.stage.output_paths.clear()
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_an_escaping_path_is_refused_by_both(self, fx):
        outside = fx.tmp_path / "outside.json"
        outside.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        fx.outputs[fx.key] = str(outside)
        fx.output_hashes[fx.key] = sha256_path(outside)
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False

    def test_a_directory_output_is_hashed_by_both(self, fx):
        model = fx.run_dir / "models" / "trace"
        model.mkdir(parents=True)
        (model / "trace.nc").write_bytes(b"posterior")
        key = f"{STAGE}:{model.as_posix()}"
        fx.stage.output_paths.append(model)
        fx.outputs[key] = str(model)
        fx.output_hashes[key] = sha256_path(model)
        assert fx.skip_accepts() is True
        assert fx.verify_accepts() is True

        (model / "smuggled.bin").write_bytes(b"extra")
        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is False


class TestSeverity:
    """`untrusted` is the last caller-dependent value, so pin both branches."""

    def _verdict(self, fx, key, *, declared):
        from panelcast.pipelines.output_integrity import verify_output_records

        (verdict,) = verify_output_records(
            {key: fx.outputs[key]},
            {key: fx.output_hashes[key]},
            roots=[fx.run_dir],
            declared={key: Path(fx.outputs[key])} if declared else None,
        )
        return verdict

    @pytest.mark.parametrize("declared", [True, False])
    def test_a_missing_output_is_untrusted_only_when_declared(self, fx, declared):
        fx.artifact.unlink()
        assert self._verdict(fx, fx.key, declared=declared).untrusted is declared

    @pytest.mark.parametrize("declared", [True, False])
    def test_an_unreadable_output_is_untrusted_only_when_declared(
        self, fx, declared, monkeypatch
    ):
        # The round-20 change: an unreadable *dynamic* output is unproven, not
        # corrupt, for the same reason a missing one is.
        def refuse(path, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("panelcast.pipelines.output_integrity.sha256_path", refuse)

        verdict = self._verdict(fx, fx.key, declared=declared)

        assert verdict.reason.startswith("recorded output unreadable")
        assert verdict.untrusted is declared

    def test_a_redirect_is_untrusted_whether_or_not_it_exists(self, fx):
        from panelcast.pipelines.output_integrity import UNBOUND, verify_output_records

        decoy = fx.run_dir / "evaluation" / "decoy.json"
        decoy.write_text(json.dumps({"mae": 9.9}), encoding="utf-8")

        (verdict,) = verify_output_records(
            {fx.key: str(decoy)},
            {fx.key: sha256_path(decoy)},
            roots=[fx.run_dir],
            declared={fx.key: fx.artifact},
        )

        assert verdict.label == UNBOUND
        assert verdict.untrusted is True

        # The half the name is actually about: the binding is checked *before*
        # existence, so a redirect at a path that is not there is still a
        # redirect. Move that check below `path.exists()` and this becomes
        # `MISSING`, with `untrusted` coincidentally still True.
        decoy.unlink()
        (verdict,) = verify_output_records(
            {fx.key: str(decoy)},
            {fx.key: fx.output_hashes[fx.key]},
            roots=[fx.run_dir],
            declared={fx.key: fx.artifact},
        )

        assert verdict.label == UNBOUND
        assert verdict.reason == "recorded output path disagrees with its manifest key"

    @pytest.mark.parametrize("declared", [True, False])
    def test_an_unlocatable_path_is_untrusted_either_way(self, fx, declared, monkeypatch):
        # `UNBOUND` is a fact about the manifest's *claim* — it named a path
        # that cannot be tied to this run — so unlike `MISSING` it does not
        # depend on whether a declared path stands behind the key.
        from panelcast.pipelines.output_integrity import UNBOUND

        real = Path.resolve

        def refuse(self, strict=False):
            if self == fx.artifact:
                raise OSError("symlink loop")
            return real(self)

        monkeypatch.setattr(Path, "resolve", refuse)
        verdict = self._verdict(fx, fx.key, declared=declared)

        assert verdict.label == UNBOUND
        assert verdict.reason == "recorded output path is unreadable"
        assert verdict.untrusted is True

    def test_a_workspace_that_cannot_resolve_the_declared_path_is_not_tampering(self, fx):
        from panelcast.pipelines.output_integrity import UNVERIFIABLE, verify_output_records

        class Unresolvable(type(fx.artifact)):
            def resolve(self, strict=False):
                raise OSError("symlink loop")

        (verdict,) = verify_output_records(
            {fx.key: str(fx.artifact)},
            {fx.key: fx.output_hashes[fx.key]},
            roots=[fx.run_dir],
            declared={fx.key: Unresolvable(fx.artifact)},
        )

        # The stage's own configuration, not the manifest's claim — so the
        # operator gets "cannot prove this", not the severity reserved for
        # substitution and root escape.
        assert verdict.label == UNVERIFIABLE
        assert verdict.untrusted is False
        assert "declared path" in verdict.reason


class TestContainment:
    """The root check has to survive a workspace an operator has bent."""

    def test_one_unresolvable_root_does_not_condemn_every_output(self, tmp_path, monkeypatch):
        # Asserted through `_resolved_roots`, which is where the tolerance
        # lives now that resolution is hoisted out of the per-key check.
        # Pointing this at `_is_contained` would pass without exercising
        # anything: that function no longer resolves, so a root that cannot be
        # resolved never raises there.
        from panelcast.pipelines.output_integrity import _is_contained, _resolved_roots

        good = tmp_path / "models"
        good.mkdir()
        artifact = good / "trace.nc"
        artifact.write_bytes(b"posterior")
        bad = tmp_path / "bad"
        real_resolve = Path.resolve

        def resolve(self, strict=False):
            if self == bad:
                raise OSError("symlink loop")
            return real_resolve(self)

        resolved = artifact.resolve()
        good_resolved = good.resolve()
        monkeypatch.setattr(Path, "resolve", resolve)

        # A bad root ahead of the good one must be dropped, not fatal: the
        # roots are a shared workspace, and one bent entry out of eight cannot
        # turn every output in every run into an apparent escape.
        roots = _resolved_roots([bad, good])

        assert roots == (good_resolved,)
        assert _is_contained(resolved, roots) is True

    def test_no_resolvable_root_contains_nothing(self, tmp_path, monkeypatch):
        # The other end of dropping bad roots: tolerance must not become
        # permissiveness. With every root gone the result is empty, and empty
        # has to mean "not contained" — the refusal both callers already
        # handle — rather than an unguarded `any()` over nothing being read as
        # success somewhere downstream.
        from panelcast.pipelines.output_integrity import _is_contained, _resolved_roots

        artifact = tmp_path / "trace.nc"
        artifact.write_bytes(b"posterior")
        resolved = artifact.resolve()
        monkeypatch.setattr(Path, "resolve", lambda self, strict=False: (_ for _ in ()).throw(
            OSError("symlink loop")
        ))

        roots = _resolved_roots([tmp_path / "a", tmp_path / "b"])

        assert roots == ()
        assert _is_contained(resolved, roots) is False

    @pytest.mark.parametrize("layout", ["flat", "run"])
    def test_the_root_enumeration_is_the_dataclass(self, tmp_path, layout):
        # Set equality, not a length check: the flat layout shares roots
        # between products, so `roots()` deduplicates and counting alone would
        # let an added field hide behind a collision. The run-scoped layout has
        # no collisions by construction, so it catches every added field.
        from panelcast.paths import ArtifactPaths

        paths = ArtifactPaths.flat() if layout == "flat" else ArtifactPaths.for_run(tmp_path)
        every = {
            value
            for f in fields(ArtifactPaths)
            if isinstance(value := getattr(paths, f.name), Path)
        }

        assert set(paths.roots()) == every

    def test_both_callers_have_exactly_the_shared_roots_and_one_run(self, tmp_path):
        # Equality, not containment. Every root defect this change went through
        # — the working tree, the output base, the current run's own products —
        # was an *extra* root, which `⊆` cannot see. Both sides are now the
        # same sentence: the artifact roots, plus the one run being verified.
        from panelcast.cli.runs_cmd import _output_roots
        from panelcast.paths import ArtifactPaths
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        run_dir = tmp_path / "run_a"
        expected = {*ArtifactPaths.flat().roots(), run_dir}
        orchestrator = PipelineOrchestrator(PipelineConfig(dry_run=True), output_base=tmp_path)

        assert set(orchestrator._output_verification_roots(run_dir)) == expected
        assert set(_output_roots(run_dir)) == expected

    def test_a_dynamic_key_may_not_reach_a_sibling_run(self, fx, tmp_path):
        # A dynamic key has no declared binding, so containment is the only
        # thing behind it. If the skip path's roots covered the output *base*
        # rather than the one run it is reading, a rewritten manifest could
        # point it at any sibling run and be believed — and the parity cases
        # above cannot see it, because they only reach containment in the
        # accepting direction and only for declared keys.
        sibling = fx.output_base / "run_b" / "evaluation" / "metrics.json"
        sibling.parent.mkdir(parents=True)
        sibling.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        key = f"{STAGE}:dataset_hash"
        fx.outputs[key] = str(sibling)
        fx.output_hashes[key] = sha256_path(sibling)

        assert fx.skip_accepts(allowed_roots=[fx.run_dir]) is False
        assert fx.verify_accepts() is False
        # The base as a root is what would admit it — the very substitution
        # `_output_verification_roots` names the run for.
        assert fx.skip_accepts(allowed_roots=[fx.output_base]) is True

    def test_the_orchestrator_names_only_the_run_it_verifies(self, tmp_path):
        # Neither the base that holds every run, nor the run-scoped products
        # the *current* run is writing into as it goes — a rewritten previous
        # manifest must not be able to point a dynamic key at either.
        from panelcast.paths import ArtifactPaths
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        base = tmp_path / "outputs"
        orchestrator = PipelineOrchestrator(PipelineConfig(dry_run=True), output_base=base)
        orchestrator.run_dir = base / "current"
        orchestrator._resolved_paths = ArtifactPaths.for_run(orchestrator.run_dir)

        roots = set(orchestrator._output_verification_roots(base / "prev"))

        assert base / "prev" in roots
        assert base not in roots
        assert not any(str(r).startswith(str(base / "current")) for r in roots)

    def test_the_run_root_comes_from_where_the_manifest_was_read(self, tmp_path):
        # Not from a field of the manifest: deriving it from `run_id` would let
        # the document under verification choose which run it is checked
        # against, which is the whole class of thing containment exists for.
        import inspect

        from panelcast.pipelines.orchestrator import PipelineOrchestrator

        signature = inspect.signature(PipelineOrchestrator._output_verification_roots)
        (name, parameter), *_ = list(signature.parameters.items())[1:]

        assert name == "previous_run"
        assert parameter.annotation == "Path"
        assert parameter.default is inspect.Parameter.empty

    def test_the_default_roots_are_never_consulted_without_a_manifest(self, fx, monkeypatch):
        # The orchestrator passes `None` only where there is no previous run,
        # and `skip_decision` returns before roots are read there. Asserted by
        # making the fallback fatal: reaching it at all fails this.
        def explode(self):
            raise AssertionError("_default_roots was consulted")

        monkeypatch.setattr(PipelineStage, "_default_roots", explode)

        assert fx.stage.skip_decision(None).skip is False

        # The other half — that the manifest and the roots are present or
        # absent together, so the `None` above never reaches a call that would
        # read it — needs a real pipeline, so it lives beside the orchestrator
        # harness as
        # `test_a_manifest_never_arrives_without_the_run_that_named_its_roots`.

    def test_the_contained_path_is_the_one_that_gets_hashed(self, tmp_path):
        from panelcast.pipelines.output_integrity import _is_contained

        root = tmp_path / "models"
        root.mkdir()
        real = root / "real.nc"
        real.write_bytes(b"posterior")
        link = root / "alias.nc"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        # Returning the resolved form is what keeps the read from walking the
        # same symlink a second time and landing somewhere else. Asserted on
        # the verdict, not on `_is_contained` alone: now that the caller
        # resolves, passing it an already-resolved path proves nothing.
        from panelcast.pipelines.output_integrity import OK, verify_output_records

        assert _is_contained(link.resolve(), [root]) is True

        key = "train:trace"
        (verdict,) = verify_output_records(
            {key: str(link)}, {key: sha256_path(link)}, roots=[root]
        )

        assert verdict.label == OK
        assert verdict.path == real.resolve()

    def test_an_unlocatable_dynamic_path_is_not_reported_as_an_escape(self, fx, monkeypatch):
        # One `None` for two facts would say a path the tool never located had
        # escaped the roots — a false statement about a path that may be
        # sitting squarely inside them. A declared key gets the accurate reason
        # from the binding; this is the same fact for a dynamic one.
        from panelcast.pipelines.output_integrity import UNBOUND, verify_output_records

        key = f"{STAGE}:dataset_hash"
        target = fx.run_dir / "dataset_hash.txt"
        real = Path.resolve

        def refuse(self, strict=False):
            if self == target:
                raise OSError("symlink loop")
            return real(self)

        monkeypatch.setattr(Path, "resolve", refuse)
        (verdict,) = verify_output_records(
            {key: str(target)}, {key: "0" * 64}, roots=[fx.run_dir]
        )

        assert verdict.label == UNBOUND
        assert verdict.reason == "recorded output path is unreadable"


class TestTheDeclaredCallerDifference:
    """What the two are meant to disagree on, asserted rather than implied."""

    def test_an_erased_declared_record_is_noticed_only_by_the_skip_path(self, fx):
        # A key absent from *both* maps is absent from the union, so there is
        # no verdict to return. The skip path still notices, because the stage
        # declares the output and can see the key is gone; `runs verify` has no
        # stage. That half is a divergence, and #439 closes it.
        fx.outputs.clear()
        fx.output_hashes.clear()

        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is True

    def test_an_erased_dynamic_record_is_invisible_to_both(self, fx):
        # The other half, and the one that matters: with no declared output
        # behind it, an erased key is indistinguishable from one that never
        # existed — *both* callers accept, and the skip path is the one that
        # then reuses artifacts on that basis. Not a divergence but a shared
        # blind spot, and not closable by #439 either: a `declared_outputs`
        # field is part of the same document and can be shortened with it.
        fx.stage.output_paths.clear()
        fx.outputs.clear()
        fx.output_hashes.clear()

        assert fx.skip_accepts() is True
        assert fx.verify_accepts() is True

    def test_a_redirect_inside_the_run_dir_is_refused_only_by_the_skip_path(self, fx):
        # Containment has nothing to say here — both files are in the run
        # directory — so the declared binding is the only thing that refuses
        # it, and only the caller holding stage objects has one. `runs verify`
        # reports it clean, which is a real gap rather than a design choice:
        # closing it needs the manifest to record which outputs were declared
        # (#439), since key shape does not carry that.
        decoy = fx.run_dir / "evaluation" / "decoy.json"
        decoy.write_text(json.dumps({"mae": 9.9}), encoding="utf-8")
        fx.outputs[fx.key] = str(decoy)
        fx.output_hashes[fx.key] = sha256_path(decoy)

        assert fx.skip_accepts() is False
        assert fx.verify_accepts() is True

    def test_a_quarantined_run_verifies_but_does_not_skip(self, fx):
        fx.quarantine()

        # `runs verify` re-roots onto the failed location and finds everything.
        assert fx.verify_accepts() is True
        # The skip path follows the active `latest` pointer, so from its side
        # the run's outputs are simply gone — and it must not go looking under
        # `failed/` for them, since a quarantined run is not a source of truth.
        assert fx.skip_accepts() is False

    def test_a_deleted_artifact_on_a_moved_run_reads_as_missing(self, fx):
        # The re-root maps a run-owned path whether or not the target survived
        # the move: reporting the pre-move location instead would turn a plain
        # deletion into an apparent escape from the run root.
        from panelcast.pipelines.output_integrity import MISSING, verify_output_records

        fx.quarantine()
        moved = fx.output_base / "failed" / "run_a"
        (moved / "evaluation" / "metrics.json").unlink()

        verdicts = list(
            verify_output_records(
                fx.outputs, fx.output_hashes, roots=(moved,), reroot=moved
            )
        )

        assert [v.label for v in verdicts] == [MISSING]
        assert fx.verify_accepts() is False

    def test_a_same_named_directory_elsewhere_is_not_run_owned(self, fx):
        # The membership test is `<output base>/<run id>/…`, not "contains a
        # component with this name" — otherwise a manifest describing a
        # different workspace launders its paths into this run directory and
        # verifies clean.
        from panelcast.pipelines.output_integrity import reroot_under

        fx.quarantine()
        moved = fx.output_base / "failed" / "run_a"
        foreign = fx.tmp_path / "elsewhere" / "run_a" / "evaluation" / "metrics.json"

        assert reroot_under(foreign, moved) == foreign
        # ...while the run's own pre-move spelling still maps.
        recorded = fx.output_base / "run_a" / "evaluation" / "metrics.json"
        assert reroot_under(recorded, moved) == moved / "evaluation" / "metrics.json"

    def test_a_relocated_output_base_still_maps(self, tmp_path):
        # The base is matched by name, so the spelling a manifest recorded —
        # usually the relative default — still maps onto a run directory that
        # is absolute, moved, or both. Comparing them as paths would resolve
        # the recorded one against whatever directory the command runs in.
        from panelcast.pipelines.output_integrity import reroot_under

        recorded = Path("outputs") / "gone" / "evaluation" / "metrics.json"
        moved = tmp_path / "archive" / "outputs" / "failed" / "gone"
        active = tmp_path / "archive" / "outputs" / "gone"

        assert reroot_under(recorded, moved) == moved / "evaluation" / "metrics.json"
        assert reroot_under(recorded, active) == active / "evaluation" / "metrics.json"

    def test_the_quarantine_directory_is_not_mistaken_for_the_base(self, tmp_path):
        # An active run's grandparent is the workspace, not an output base, so
        # accepting it would launder `<workspace>/<id>/…` back in.
        from panelcast.pipelines.output_integrity import reroot_under

        active = tmp_path / "outputs" / "run_a"
        assert reroot_under(tmp_path / "run_a" / "x.json", active) == tmp_path / "run_a" / "x.json"

    def test_a_foreign_workspace_manifest_does_not_verify_clean(self, fx):
        # End to end through `runs verify`, which is where a laundered path
        # would turn "this manifest is not about this workspace" into exit 0.
        # The foreign artifact is real and hashes correctly, so refusal cannot
        # come from its absence — ownership is what decides it.
        foreign = fx.tmp_path / "elsewhere" / "run_a" / "evaluation" / "metrics.json"
        foreign.parent.mkdir(parents=True)
        foreign.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        fx.outputs[fx.key] = str(foreign)
        fx.output_hashes[fx.key] = sha256_path(foreign)
        fx.quarantine()

        assert foreign.exists()
        assert fx.verify_accepts() is False

    def test_a_same_named_base_elsewhere_is_indistinguishable_from_relocation(self, fx):
        # The residue, stated rather than implied: another checkout of the same
        # project spells its base `outputs` too, so with a relative recorded
        # base relocation and impersonation are the same string. The mapping
        # still only aims into this run's directory, so what it can produce is
        # bounded by containment and the recorded hash — it passes here only
        # because the run id, the run-relative path and the bytes all agree.
        from panelcast.pipelines.output_integrity import reroot_under

        fx.quarantine()
        moved = fx.output_base / "failed" / "run_a"
        sibling = fx.tmp_path / "other" / "outputs" / "run_a" / "evaluation" / "metrics.json"
        sibling.parent.mkdir(parents=True)
        sibling.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")

        # Existing, not merely absent: since ownership is decided before the
        # path is consulted, a live sibling maps the same way a stale spelling
        # does, and the residue is the same size either way.
        assert reroot_under(sibling, moved) == moved / "evaluation" / "metrics.json"
        # ...and a base spelled differently is refused, which is the half the
        # ownership check can actually decide.
        elsewhere = fx.tmp_path / "other" / "elsewhere" / "run_a" / "evaluation" / "metrics.json"
        assert reroot_under(elsewhere, moved) == elsewhere

        # The verdict itself, not just the mapping: for a reproducible pipeline
        # the hashes agree across checkouts, so this is the expected state
        # rather than a corner case. `runs verify` says the bytes match; it
        # does not say the manifest was written about this workspace.
        fx.outputs[fx.key] = str(sibling)
        assert fx.verify_accepts() is True

    def test_a_run_id_repeated_in_the_output_base_finds_the_right_occurrence(self, tmp_path):
        # Locating `<base>/<id>` as a unit, right to left: scanning for the id
        # alone would match inside the base's own path and compare the wrong
        # pair, declaring a healthy run's output foreign.
        from panelcast.pipelines.output_integrity import reroot_under

        moved = tmp_path / "run_a" / "outputs" / "failed" / "run_a"
        recorded = Path("run_a") / "outputs" / "run_a" / "evaluation" / "metrics.json"

        assert reroot_under(recorded, moved) == moved / "evaluation" / "metrics.json"

    def test_an_output_base_of_dot_still_reroots(self, tmp_path, monkeypatch):
        # `--output-base .` leaves the base with no name to match, so the run
        # id stands alone rather than the mapping bailing out.
        from panelcast.pipelines.output_integrity import reroot_under

        monkeypatch.chdir(tmp_path)
        moved = Path("failed") / "gone"
        recorded = Path("gone") / "evaluation" / "metrics.json"

        assert reroot_under(recorded, moved) == moved / "evaluation" / "metrics.json"
        # ...anchored at the first component, not scanned for: without a base
        # name to pair it with, matching the id anywhere would be exactly the
        # bare-name laundering the pairing exists to refuse.
        foreign = Path("/somewhere/else") / "gone" / "evaluation" / "metrics.json"
        assert reroot_under(foreign, moved) == foreign

    def test_the_output_base_decides_which_copy_is_verified(self, tmp_path, monkeypatch):
        # Not the working directory: a stale copy sitting under cwd must not
        # be the one checked (and then refused as outside the roots) while the
        # real artifact under `--output-base` goes unread.
        from panelcast.pipelines.output_integrity import reroot_under

        monkeypatch.chdir(tmp_path)
        stale = tmp_path / "outputs" / "run_a" / "evaluation" / "metrics.json"
        stale.parent.mkdir(parents=True)
        stale.write_text("{}", encoding="utf-8")
        archived = tmp_path / "archive" / "outputs" / "run_a"

        recorded = Path("outputs") / "run_a" / "evaluation" / "metrics.json"

        assert stale.exists()
        assert reroot_under(recorded, archived) == archived / "evaluation" / "metrics.json"

    def test_the_reroot_mapping_is_generic_over_what_it_moves(self, fx):
        # Not output-specific on purpose: run-owned *inputs* need the same
        # mapping (#420), which this module does not verify but must not block.
        from panelcast.pipelines.output_integrity import reroot_under

        fx.quarantine()
        moved = fx.output_base / "failed" / "run_a"
        recorded_input = fx.run_dir / "models" / "manifest.json"
        (moved / "models").mkdir(parents=True, exist_ok=True)
        (moved / "models" / "manifest.json").write_text("{}", encoding="utf-8")

        assert reroot_under(recorded_input, moved) == moved / "models" / "manifest.json"

    def test_the_quarantine_name_has_one_definition(self):
        # `reroot_under` recognizes a moved run by matching `<base>/failed/<id>`,
        # so it is a *reader* of the name. This pins the reader to the layout's
        # definition; it cannot pin the writers, which still use literals. What
        # it rules out is this module drifting from the layout and reporting
        # every intact quarantined run as tampered rather than moved.
        #
        # Static, not `is`: CPython interns identifier-like string constants, so
        # a restored local `QUARANTINE_DIR = "failed"` would be the *same
        # object* as the layout's and an identity assertion would pass against
        # the exact defect it exists to forbid. `from`-import is a snapshot
        # binding anyway, so "one definition" is not observable at runtime —
        # it is a property of the source, and the source is what to read.
        import ast

        from panelcast import paths
        from panelcast.pipelines import output_integrity

        tree = ast.parse(Path(output_integrity.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        assert "QUARANTINE_DIR" in imported
        assert "QUARANTINE_DIR" not in assigned
        assert paths.QUARANTINE_DIR in paths._RESERVED_RUN_IDS
