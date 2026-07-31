"""Durability of the records a later process reads back (#424).

The run manifest, the data-root stamps and the `latest` pointer are all
rewritten in place while a run is going, and all three are read by something
that cannot tell a truncated file from a short one. These tests pin the one
property that makes that safe: a failed write leaves the previous file, whole.
"""

import json
from pathlib import Path

import pytest

from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    load_run_manifest,
    save_run_manifest,
)
from panelcast.pipelines.stamps import read_stamp, write_stamp
from panelcast.utils.atomic import atomic_write_text


def _manifest(**overrides) -> RunManifest:
    fields = dict(
        run_id="2026-01-19_143052",
        created_at="2026-01-19T14:30:52Z",
        command="panelcast run --seed 42",
        flags={"seed": 42},
        seed=42,
        git=GitStateModel(
            commit="abc123def456789012345678901234567890abcd",
            branch="main",
            dirty=False,
            untracked_count=0,
        ),
        environment=EnvironmentInfo(
            python_version="3.11.5",
            jax_version="0.4.26",
            numpyro_version="0.15.0",
            arviz_version="0.18.0",
            platform="Linux 6.6",
            pixi_lock_hash=None,
        ),
        input_hashes={},
        stage_hashes={"data": "hash_data"},
        stages_completed=["data"],
        stages_skipped=[],
        outputs={},
        success=False,
    )
    fields.update(overrides)
    return RunManifest(**fields)


def _fail_fsync(monkeypatch) -> None:
    """Make persistence fail after the temporary is open and written."""

    def boom(_fd):
        raise OSError("no space left on device")

    monkeypatch.setattr("panelcast.utils.atomic.os.fsync", boom)


class TestAtomicWriteText:
    def test_replaces_the_file_whole(self, tmp_path: Path):
        target = tmp_path / "record.json"
        atomic_write_text(target, "first")
        atomic_write_text(target, "second")

        assert target.read_text(encoding="utf-8") == "second"
        assert [p.name for p in tmp_path.iterdir()] == ["record.json"]

    def test_a_failed_write_leaves_the_previous_content_and_no_debris(
        self, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "record.json"
        atomic_write_text(target, "first")
        _fail_fsync(monkeypatch)

        with pytest.raises(OSError):
            atomic_write_text(target, "second")

        assert target.read_text(encoding="utf-8") == "first"
        assert [p.name for p in tmp_path.iterdir()] == ["record.json"]

    def test_an_unencodable_value_never_creates_a_file(self, tmp_path: Path):
        target = tmp_path / "record.txt"

        with pytest.raises(UnicodeEncodeError):
            atomic_write_text(target, "\ud800", encoding="utf-8")

        assert not list(tmp_path.iterdir())


class TestManifestWritesAreAtomic:
    def test_persistence_failure_keeps_the_previous_manifest(self, tmp_path: Path, monkeypatch):
        manifest = _manifest()
        save_run_manifest(manifest, tmp_path)
        manifest.stages_completed.append("splits")
        _fail_fsync(monkeypatch)

        with pytest.raises(OSError):
            save_run_manifest(manifest, tmp_path)

        assert load_run_manifest(tmp_path / "manifest.json").stages_completed == ["data"]
        assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]

    def test_serialization_failure_keeps_the_previous_manifest(self, tmp_path: Path, monkeypatch):
        manifest = _manifest()
        save_run_manifest(manifest, tmp_path)

        def boom(*_args, **_kwargs):
            raise ValueError("unserializable")

        monkeypatch.setattr(RunManifest, "model_dump_json", boom)
        with pytest.raises(ValueError):
            save_run_manifest(manifest, tmp_path)

        assert load_run_manifest(tmp_path / "manifest.json").run_id == manifest.run_id

    def test_a_completed_write_replaces_the_manifest_whole(self, tmp_path: Path):
        manifest = _manifest()
        save_run_manifest(manifest, tmp_path)
        manifest.stages_completed.append("splits")
        path = save_run_manifest(manifest, tmp_path)

        assert load_run_manifest(path).stages_completed == ["data", "splits"]


class TestStampWritesAreAtomic:
    def test_a_failed_stamp_write_keeps_the_recorded_stamp(self, tmp_path: Path, monkeypatch):
        root = tmp_path / "features"
        write_stamp(root, "features", "hash-one", "run-one")
        _fail_fsync(monkeypatch)

        with pytest.raises(OSError):
            write_stamp(root, "features", "hash-two", "run-two")

        stamp = read_stamp(root)
        assert stamp is not None
        assert stamp["run_id"] == "run-one"
        assert [p.name for p in root.iterdir()] == [".stamp.json"]


class TestLatestPointerIsAtomic:
    def test_a_failed_pointer_write_keeps_the_previous_target(self, tmp_path: Path, monkeypatch):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        orchestrator = PipelineOrchestrator(PipelineConfig(dry_run=True), output_base=tmp_path)
        orchestrator.run_dir = tmp_path / "run-one"
        orchestrator.run_dir.mkdir()
        orchestrator._write_latest_pointer()

        orchestrator.run_dir = tmp_path / "run-two"
        orchestrator.run_dir.mkdir()
        _fail_fsync(monkeypatch)
        orchestrator._write_latest_pointer()  # warns, never raises

        pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
        assert pointer["run_dir"] == "run-one"
        assert not list(tmp_path.glob("latest.json.tmp*"))
