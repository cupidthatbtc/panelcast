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

        ``allowed_roots`` defaults to the run dir so the parity cases isolate
        the per-key rules; pass None explicitly to exercise the stage's own
        root derivation instead.
        """
        roots = [self.run_dir] if allowed_roots is None else allowed_roots
        return self.stage.skip_decision(self._manifest(), allowed_roots=roots).skip

    def skip_accepts_on_default_roots(self) -> bool:
        """The same, but through `_default_roots()` rather than a stub."""
        return self.stage.skip_decision(self._manifest()).skip

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
        # Not only through the stubbed roots: the stage's own derivation has to
        # accept its own artifact too, or the parity cases below are measuring
        # a configuration production never runs.
        assert fx.skip_accepts_on_default_roots() is True

    def test_an_escaping_path_is_refused_on_the_stages_own_roots(self, fx):
        outside = fx.tmp_path / "outside.json"
        outside.write_text(json.dumps({"mae": 5.3}), encoding="utf-8")
        fx.outputs[fx.key] = str(outside)
        fx.output_hashes[fx.key] = sha256_path(outside)
        assert fx.skip_accepts_on_default_roots() is False

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

    def test_a_run_that_recorded_nothing_is_accepted_by_both(self, fx):
        # ...and nothing recorded is not the same as recorded-and-unprovable.
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


class TestContainment:
    """The root check has to survive a workspace an operator has bent."""

    def test_one_unresolvable_root_does_not_condemn_every_output(self, tmp_path, monkeypatch):
        from panelcast.pipelines.output_integrity import contained_path

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

        monkeypatch.setattr(Path, "resolve", resolve)

        # A bad root ahead of the good one must be skipped, not fatal: the
        # roots are a shared workspace, and one bent entry out of eight cannot
        # turn every output in every run into an apparent escape.
        assert contained_path(artifact, [bad, good]) is not None

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

    def test_both_callers_cover_the_shared_root_definition(self, tmp_path):
        # *Which* roots is part of what a caller accepts as proof, so it is the
        # last place the two could drift. Containment is `⊆`, not `=`, on
        # purpose: each adds what only it knows about — the orchestrator its
        # output base, `runs verify` the resolved run dir — but neither may
        # narrow the shared set.
        from panelcast.cli.runs_cmd import _output_roots
        from panelcast.paths import ArtifactPaths
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        declared = set(ArtifactPaths.flat().roots())
        orchestrator = PipelineOrchestrator(PipelineConfig(dry_run=True), output_base=tmp_path)

        assert declared <= set(orchestrator._output_verification_roots("run_a"))
        assert declared <= set(_output_roots(tmp_path / "run_a"))

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

        roots = set(orchestrator._output_verification_roots("prev"))

        assert base / "prev" in roots
        assert base not in roots
        assert not any(str(r).startswith(str(base / "current")) for r in roots)

    def test_the_stages_default_roots_are_deliberately_narrower(self, fx):
        # Not part of the shared definition and not a gap: with no caller-named
        # roots a stage vouches only for the parents of the outputs it declares
        # itself. `runs verify` has no stages, which is why it needs the
        # workspace enumeration instead.
        assert fx.stage._default_roots() == (fx.artifact.parent,)

    def test_the_contained_path_is_the_one_that_gets_hashed(self, tmp_path):
        from panelcast.pipelines.output_integrity import contained_path

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
        # same symlink a second time and landing somewhere else.
        assert contained_path(link, [root]) == real.resolve()


class TestTheDeclaredCallerDifference:
    """Re-rooting is the one behaviour the two are meant to disagree on."""

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
