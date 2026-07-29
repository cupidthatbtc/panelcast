"""--skip-existing re-hashes recorded outputs before trusting them (#367).

Existence is not integrity: these are the mutations that used to survive a
skip and get consumed downstream as if nothing had happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
)
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.stages import PipelineStage
from panelcast.utils.hashing import sha256_path


def _manifest(stage_hashes, outputs, output_hashes) -> RunManifest:
    return RunManifest(
        run_id="prev",
        created_at="2026-07-26T00:00:00Z",
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
        stage_hashes=stage_hashes,
        stages_completed=["data"],
        stages_skipped=[],
        outputs=outputs,
        output_hashes=output_hashes,
        success=True,
    )


def _write_parquet(path: Path, frame: pd.DataFrame | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    frame.to_parquet(path)
    return path


def _write_json(path: Path, payload=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"mae": 5.3}), encoding="utf-8")
    return path


def _write_model_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "trace.nc").write_bytes(b"\x89HDF\r\n\x1a\n" + b"posterior" * 64)
    (root / "meta.json").write_text('{"chains": 4}', encoding="utf-8")
    return root


class _Fixture:
    """A stage with one input and a set of recorded outputs, ready to skip."""

    def __init__(self, tmp_path: Path, outputs: dict[str, Path]):
        self.tmp_path = tmp_path
        source = tmp_path / "raw.csv"
        source.write_text("a,b\n1,2\n", encoding="utf-8")
        self.stage = PipelineStage(
            name="data",
            description="fake",
            run_fn=None,
            input_paths=[source],
            output_paths=list(outputs.values()),
        )
        self.outputs = {f"data:{p.as_posix()}": p for p in outputs.values()}
        self.manifest = _manifest(
            stage_hashes={"data": self.stage.compute_input_hash()},
            outputs={k: str(p) for k, p in self.outputs.items()},
            output_hashes={k: sha256_path(p) for k, p in self.outputs.items()},
        )

    def decision(self, roots=None):
        allowed = [self.tmp_path] if roots is None else roots
        return self.stage.skip_decision(self.manifest, allowed_roots=allowed)


@pytest.fixture
def parquet_fixture(tmp_path):
    return _Fixture(tmp_path, {"table": _write_parquet(tmp_path / "processed" / "d.parquet")})


@pytest.fixture
def json_fixture(tmp_path):
    return _Fixture(tmp_path, {"metrics": _write_json(tmp_path / "evaluation" / "m.json")})


@pytest.fixture
def model_dir_fixture(tmp_path):
    return _Fixture(tmp_path, {"model": _write_model_dir(tmp_path / "models" / "trace")})


class TestUnchangedOutputsStillSkip:
    def test_parquet(self, parquet_fixture):
        assert parquet_fixture.decision().skip

    def test_json(self, json_fixture):
        assert json_fixture.decision().skip

    def test_directory(self, model_dir_fixture):
        assert model_dir_fixture.decision().skip

    def test_mixed_file_and_directory(self, tmp_path):
        fx = _Fixture(
            tmp_path,
            {
                "table": _write_parquet(tmp_path / "processed" / "d.parquet"),
                "metrics": _write_json(tmp_path / "evaluation" / "m.json"),
                "model": _write_model_dir(tmp_path / "models" / "trace"),
            },
        )
        assert fx.decision().skip

    def test_touching_without_editing_still_skips(self, parquet_fixture):
        # mtime is not the contract; bytes are.
        path = next(iter(parquet_fixture.outputs.values()))
        path.touch()
        assert parquet_fixture.decision().skip


class TestMutatedOutputsBlockSkip:
    def _path(self, fx):
        return next(iter(fx.outputs.values()))

    def test_truncated_parquet(self, parquet_fixture):
        path = self._path(parquet_fixture)
        path.write_bytes(path.read_bytes()[:-16])
        decision = parquet_fixture.decision()
        assert not decision.skip
        assert decision.outputs_untrusted

    def test_emptied_parquet(self, parquet_fixture):
        self._path(parquet_fixture).write_bytes(b"")
        assert not parquet_fixture.decision().skip

    def test_substituted_parquet(self, parquet_fixture, tmp_path):
        # A different table swapped in under the same name — still a valid
        # Parquet file, which is exactly why existence proves nothing.
        other = _write_parquet(
            tmp_path / "other.parquet",
            pd.DataFrame({"a": [9, 9, 9], "b": ["q", "q", "q"]}),
        )
        path = self._path(parquet_fixture)
        path.write_bytes(other.read_bytes())
        assert not parquet_fixture.decision().skip

    def test_single_byte_flip(self, parquet_fixture):
        path = self._path(parquet_fixture)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x01
        path.write_bytes(bytes(data))
        assert not parquet_fixture.decision().skip

    def test_edited_json(self, json_fixture):
        self._path(json_fixture).write_text('{"mae": 0.0}', encoding="utf-8")
        assert not json_fixture.decision().skip

    def test_reformatted_json_is_still_a_change(self, json_fixture):
        self._path(json_fixture).write_text('{"mae":   5.3}', encoding="utf-8")
        assert not json_fixture.decision().skip

    def test_deleted_output(self, parquet_fixture):
        self._path(parquet_fixture).unlink()
        decision = parquet_fixture.decision()
        assert not decision.skip
        assert decision.outputs_untrusted

    def test_directory_file_modified(self, model_dir_fixture):
        root = self._path(model_dir_fixture)
        (root / "meta.json").write_text('{"chains": 1}', encoding="utf-8")
        assert not model_dir_fixture.decision().skip

    def test_directory_file_removed(self, model_dir_fixture):
        root = self._path(model_dir_fixture)
        (root / "meta.json").unlink()
        assert not model_dir_fixture.decision().skip

    def test_directory_file_added(self, model_dir_fixture):
        root = self._path(model_dir_fixture)
        (root / "extra.bin").write_bytes(b"smuggled")
        assert not model_dir_fixture.decision().skip

    def test_directory_replaced_by_file(self, model_dir_fixture):
        import shutil

        root = self._path(model_dir_fixture)
        shutil.rmtree(root)
        root.write_bytes(b"not a directory")
        assert not model_dir_fixture.decision().skip

    def test_only_one_of_several_outputs_mutated(self, tmp_path):
        fx = _Fixture(
            tmp_path,
            {
                "table": _write_parquet(tmp_path / "processed" / "d.parquet"),
                "metrics": _write_json(tmp_path / "evaluation" / "m.json"),
            },
        )
        (tmp_path / "evaluation" / "m.json").write_text("{}", encoding="utf-8")
        assert not fx.decision().skip


class TestUnverifiableManifestsFailClosed:
    def test_legacy_manifest_without_output_hashes(self, parquet_fixture):
        parquet_fixture.manifest.output_hashes = {}
        decision = parquet_fixture.decision()
        assert not decision.skip
        assert not decision.outputs_untrusted
        assert decision.outputs_unverifiable
        assert "0.9.0" in decision.reason

    def test_one_recorded_output_missing_its_hash(self, tmp_path):
        fx = _Fixture(
            tmp_path,
            {
                "table": _write_parquet(tmp_path / "processed" / "d.parquet"),
                "metrics": _write_json(tmp_path / "evaluation" / "m.json"),
            },
        )
        key = next(iter(fx.manifest.output_hashes))
        del fx.manifest.output_hashes[key]
        decision = fx.decision()
        assert not decision.skip
        assert decision.outputs_unverifiable
        assert not decision.outputs_untrusted

    def test_stage_with_outputs_but_no_recorded_entries(self, parquet_fixture):
        parquet_fixture.manifest.outputs = {}
        decision = parquet_fixture.decision()
        assert not decision.skip
        assert decision.outputs_unverifiable
        assert not decision.outputs_untrusted

    def test_declared_output_never_recorded(self, tmp_path):
        fx = _Fixture(tmp_path, {"table": _write_parquet(tmp_path / "processed" / "d.parquet")})
        extra = _write_json(tmp_path / "evaluation" / "m.json")
        fx.stage.output_paths.append(extra)
        decision = fx.decision()
        assert not decision.skip
        assert decision.outputs_unverifiable
        assert not decision.outputs_untrusted

    def test_malformed_hash_value(self, parquet_fixture):
        key = next(iter(parquet_fixture.manifest.output_hashes))
        parquet_fixture.manifest.output_hashes[key] = "not-a-digest"
        assert not parquet_fixture.decision().skip

    def test_empty_hash_value(self, parquet_fixture):
        key = next(iter(parquet_fixture.manifest.output_hashes))
        parquet_fixture.manifest.output_hashes[key] = ""
        assert not parquet_fixture.decision().skip

    def test_stage_with_no_outputs_at_all_can_still_skip(self, tmp_path):
        source = tmp_path / "raw.csv"
        source.write_text("a\n1\n", encoding="utf-8")
        stage = PipelineStage(
            name="report", description="fake", run_fn=None, input_paths=[source]
        )
        manifest = _manifest({"report": stage.compute_input_hash()}, {}, {})
        assert stage.skip_decision(manifest, allowed_roots=[tmp_path]).skip


class TestDynamicallyRecordedOutputs:
    """run_fn result paths are recorded under their own keys, not their paths."""

    def _fixture(self, tmp_path):
        fx = _Fixture(tmp_path, {"table": _write_parquet(tmp_path / "processed" / "d.parquet")})
        extra = _write_model_dir(tmp_path / "models" / "trace")
        fx.manifest.outputs["data:model_path"] = str(extra)
        fx.manifest.output_hashes["data:model_path"] = sha256_path(extra)
        return fx, extra

    def test_unchanged_dynamic_output_still_skips(self, tmp_path):
        fx, _ = self._fixture(tmp_path)
        assert fx.decision().skip

    def test_mutated_dynamic_output_blocks_skip(self, tmp_path):
        fx, extra = self._fixture(tmp_path)
        (extra / "trace.nc").write_bytes(b"tampered")
        decision = fx.decision()
        assert not decision.skip
        assert decision.key == "data:model_path"

    def test_deleted_dynamic_output_blocks_skip(self, tmp_path):
        import shutil

        fx, extra = self._fixture(tmp_path)
        shutil.rmtree(extra)
        assert not fx.decision().skip


class TestRecordedPathContainment:
    @pytest.mark.parametrize("escape", ["../outside.parquet", "/etc/hostname"])
    def test_path_outside_roots_is_refused(self, tmp_path, escape):
        fx = _Fixture(tmp_path, {"table": _write_parquet(tmp_path / "processed" / "d.parquet")})
        key = next(iter(fx.manifest.outputs))
        fx.manifest.outputs[key] = escape
        decision = fx.decision(roots=[tmp_path / "processed"])
        assert not decision.skip
        assert decision.outputs_untrusted

    def test_static_output_cannot_be_substituted_with_another_file_in_root(self, tmp_path):
        declared = _write_parquet(tmp_path / "processed" / "declared.parquet")
        substitute = tmp_path / "processed" / "substitute.parquet"
        substitute.write_bytes(declared.read_bytes())
        fx = _Fixture(tmp_path, {"table": declared})
        key = next(iter(fx.manifest.outputs))
        fx.manifest.outputs[key] = str(substitute)
        fx.manifest.output_hashes[key] = sha256_path(substitute)

        decision = fx.decision(roots=[tmp_path / "processed"])
        assert not decision.skip
        assert decision.outputs_untrusted
        assert "manifest key" in decision.reason

    def test_symlink_escaping_roots_is_refused(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        real = _write_parquet(outside / "d.parquet")
        workspace = tmp_path / "workspace"
        (workspace / "processed").mkdir(parents=True)
        link = workspace / "processed" / "d.parquet"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        fx = _Fixture(tmp_path, {"table": link})
        assert not fx.decision(roots=[workspace]).skip

    def test_path_inside_roots_is_accepted(self, tmp_path):
        fx = _Fixture(tmp_path, {"table": _write_parquet(tmp_path / "processed" / "d.parquet")})
        assert fx.decision(roots=[tmp_path / "processed"]).skip

    def test_empty_root_allowlist_fails_closed(self, tmp_path):
        fx = _Fixture(tmp_path, {"table": _write_parquet(tmp_path / "processed" / "d.parquet")})

        decision = fx.decision(roots=[])

        assert not decision.skip
        assert decision.outputs_untrusted
        assert "escapes" in decision.reason

    def test_orchestrator_allows_every_declared_artifact_root(self, tmp_path):
        from panelcast.paths import ArtifactPaths

        orchestrator = PipelineOrchestrator(PipelineConfig(dry_run=True), output_base=tmp_path)
        roots = set(orchestrator._output_verification_roots())
        paths = ArtifactPaths.flat()

        assert {
            paths.processed,
            paths.splits,
            paths.features,
            paths.models,
            paths.evaluation,
            paths.predictions,
            paths.reports,
        } <= roots


class TestSkipDecisionReasons:
    def test_force_never_skips(self, parquet_fixture):
        assert not parquet_fixture.stage.skip_decision(parquet_fixture.manifest, force=True).skip

    def test_no_manifest_never_skips(self, parquet_fixture):
        assert not parquet_fixture.stage.skip_decision(None).skip

    def test_changed_input_is_not_an_output_problem(self, parquet_fixture):
        parquet_fixture.stage.input_paths[0].write_text("a,b\n9,9\n", encoding="utf-8")
        decision = parquet_fixture.decision()
        assert not decision.skip
        assert not decision.outputs_untrusted

    def test_new_run_scoped_output_is_not_reported_as_corruption(self, tmp_path):
        source = tmp_path / "features.parquet"
        source.write_bytes(b"features")
        previous = tmp_path / "outputs" / "previous" / "models" / "trace.nc"
        previous.parent.mkdir(parents=True)
        previous.write_bytes(b"posterior")
        current = tmp_path / "outputs" / "current" / "models" / "trace.nc"
        stage = PipelineStage(
            name="train",
            description="fake",
            run_fn=None,
            input_paths=[source],
            output_paths=[current],
        )
        key = f"train:{previous.as_posix()}"
        manifest = _manifest(
            stage_hashes={"train": stage.compute_input_hash()},
            outputs={key: str(previous)},
            output_hashes={key: sha256_path(previous)},
        )

        decision = stage.skip_decision(manifest, allowed_roots=[tmp_path])

        assert not decision.skip
        assert not decision.outputs_untrusted
        assert decision.reason == "output not produced by the previous run"

    def test_missing_recorded_output_is_reported_as_corruption(self, tmp_path):
        source = tmp_path / "features.parquet"
        source.write_bytes(b"features")
        missing = tmp_path / "outputs" / "previous" / "models" / "trace.nc"
        stage = PipelineStage(
            name="train",
            description="fake",
            run_fn=None,
            input_paths=[source],
            output_paths=[missing],
        )
        key = f"train:{missing.as_posix()}"
        manifest = _manifest(
            stage_hashes={"train": stage.compute_input_hash()},
            outputs={key: str(missing)},
            output_hashes={key: "0" * 64},
        )

        decision = stage.skip_decision(manifest, allowed_roots=[tmp_path])

        assert not decision.skip
        assert decision.outputs_untrusted
        assert decision.key == key

    def test_should_skip_matches_decision(self, parquet_fixture):
        roots = [parquet_fixture.tmp_path]
        assert parquet_fixture.stage.should_skip(
            parquet_fixture.manifest, allowed_roots=roots
        ) is parquet_fixture.stage.skip_decision(
            parquet_fixture.manifest, allowed_roots=roots
        ).skip


# ---------------------------------------------------------------------------
# End to end through the orchestrator, with the fake-stage harness used by the
# run-isolation suite: a corrupted shared artifact must force a rerun.
# ---------------------------------------------------------------------------


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


def _fake_stages(paths, executed):
    from panelcast.paths import ArtifactPaths

    paths = paths or ArtifactPaths.flat()

    def _run(name, root_attr, filename):
        def run_fn(ctx):
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
    ]


def _run_pipeline(output_base, run_id, executed, **config_kwargs):
    config = PipelineConfig(enforce_lockfile=False, **config_kwargs)
    orchestrator = PipelineOrchestrator(config, output_base=output_base)

    def fake_order(stages=None, min_ratings=10, descriptor=None, descriptor_path=None, paths=None):
        return _fake_stages(paths, executed)

    with (
        patch("panelcast.pipelines.orchestrator.get_execution_order", side_effect=fake_order),
        patch("panelcast.pipelines.orchestrator.generate_run_id", return_value=run_id),
    ):
        return orchestrator.run()


class TestOrchestratorSkipVerification:
    @pytest.fixture
    def workspace(self, tmp_path, monkeypatch, mock_env):
        monkeypatch.chdir(tmp_path)
        raw = tmp_path / "data" / "raw" / "raw.csv"
        raw.parent.mkdir(parents=True)
        raw.write_text("a,b\n1,2\n", encoding="utf-8")
        return tmp_path / "outputs"

    def test_unchanged_shared_output_is_skipped(self, workspace):
        assert _run_pipeline(workspace, "runA", []) == 0
        executed: list[str] = []
        assert _run_pipeline(workspace, "runB", executed, skip_existing=True) == 0
        assert "data" not in executed
        manifest = json.loads((workspace / "runB" / "manifest.json").read_text(encoding="utf-8"))
        assert "data" in manifest["stages_skipped"]

    def test_consecutive_runs_keep_skipping_verified_shared_output(self, workspace):
        assert _run_pipeline(workspace, "runA", []) == 0
        second: list[str] = []
        assert _run_pipeline(workspace, "runB", second, skip_existing=True) == 0
        third: list[str] = []
        assert _run_pipeline(workspace, "runC", third, skip_existing=True) == 0

        assert "data" not in second
        assert "data" not in third
        manifest = json.loads((workspace / "runB" / "manifest.json").read_text(encoding="utf-8"))
        assert "data" in manifest["stage_hashes"]
        assert any(key.startswith("data:") for key in manifest["outputs"])
        assert any(key.startswith("data:") for key in manifest["output_hashes"])

    def test_corrupted_shared_output_forces_rerun(self, workspace, tmp_path):
        assert _run_pipeline(workspace, "runA", []) == 0
        marker = tmp_path / "data" / "processed" / "marker.parquet"
        marker.write_text("corrupted", encoding="utf-8")

        executed: list[str] = []
        assert _run_pipeline(workspace, "runB", executed, skip_existing=True) == 0
        assert "data" in executed
        manifest = json.loads((workspace / "runB" / "manifest.json").read_text(encoding="utf-8"))
        assert "data" not in manifest["stages_skipped"]
        assert marker.read_text(encoding="utf-8") == "runB"

    def test_deleted_shared_output_forces_rerun(self, workspace, tmp_path):
        assert _run_pipeline(workspace, "runA", []) == 0
        (tmp_path / "data" / "processed" / "marker.parquet").unlink()

        executed: list[str] = []
        assert _run_pipeline(workspace, "runB", executed, skip_existing=True) == 0
        assert "data" in executed

    def test_legacy_manifest_forces_rerun(self, workspace):
        assert _run_pipeline(workspace, "runA", []) == 0
        manifest_path = workspace / "runA" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["output_hashes"] = {}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        executed: list[str] = []
        assert _run_pipeline(workspace, "runB", executed, skip_existing=True) == 0
        assert "data" in executed
