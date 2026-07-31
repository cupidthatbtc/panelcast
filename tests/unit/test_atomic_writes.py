"""Durability of the records a later process reads back (#424).

The run manifest, the data-root stamps and the `latest` pointer are all
rewritten in place while a run is going, and all three are read by something
that cannot tell a truncated file from a short one. These tests pin the one
property that makes that safe: a failed write leaves the previous file, whole.
"""

import json
import os
import stat
from contextlib import contextmanager
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
from panelcast.utils import atomic as atomic_module
from panelcast.utils.atomic import TMP_MARKER, atomic_write, atomic_write_text


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


def _fail_the_commit(monkeypatch) -> None:
    """Let the payload reach the temporary, then fail before it is published.

    Patches the ``atomic_write`` name inside its own module — the one every
    text writer resolves at call time — rather than reaching into the real
    ``os`` module, so exactly one commit fails and nothing else in the process
    is affected.
    """
    real = atomic_module.atomic_write

    @contextmanager
    def guarded(path):
        with real(path) as handle:
            yield handle
            raise OSError("no space left on device")

    monkeypatch.setattr(atomic_module, "atomic_write", guarded)


def _temps(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if TMP_MARKER in p.name)


class TestAtomicWrite:
    def test_the_handle_replaces_the_target_only_on_success(self, tmp_path: Path):
        target = tmp_path / "payload.bin"
        target.write_bytes(b"original")

        with atomic_write(target) as handle:
            handle.write(b"replacement")

        assert target.read_bytes() == b"replacement"
        assert not _temps(tmp_path)

    def test_an_exception_in_the_body_leaves_the_previous_bytes(self, tmp_path: Path):
        target = tmp_path / "payload.bin"
        target.write_bytes(b"original")

        with pytest.raises(ValueError):
            with atomic_write(target) as handle:
                handle.write(b"partial")
                raise ValueError("mid-write")

        assert target.read_bytes() == b"original"
        assert not _temps(tmp_path)

    def test_a_cleanup_failure_never_replaces_the_original_error(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "payload.bin"
        target.write_bytes(b"original")

        def refuse(_self, missing_ok: bool = False):
            raise OSError("read-only directory")

        monkeypatch.setattr(Path, "unlink", refuse)

        # The write failure is the diagnostic; failing to tidy up must not
        # stand in front of it.
        with pytest.raises(ValueError, match="mid-write"):
            with atomic_write(target) as handle:
                handle.write(b"partial")
                raise ValueError("mid-write")

        assert target.read_bytes() == b"original"

    @pytest.mark.parametrize("target_exists", [False, True])
    def test_two_writers_on_one_target_do_not_share_a_temporary(
        self, tmp_path: Path, target_exists: bool
    ):
        # Both cases, because the two create paths differ: a fresh target asks
        # for the umask default, an existing one for the bits it already has.
        target = tmp_path / "payload.bin"
        if target_exists:
            target.write_bytes(b"original")

        with atomic_write(target) as first, atomic_write(target) as second:
            first.write(b"a")
            second.write(b"bb")
            assert len({p.name for p in _temps(tmp_path)}) == 2

    def test_the_directory_entry_is_synced_after_the_rename(self, tmp_path: Path, monkeypatch):
        synced: list[Path] = []
        monkeypatch.setattr(atomic_module, "fsync_dir", synced.append)

        atomic_write_text(tmp_path / "payload.txt", "x")

        assert synced == [tmp_path]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_an_existing_target_keeps_its_permissions(self, tmp_path: Path):
        target = tmp_path / "manifest.json"
        atomic_write_text(target, "{}")
        target.chmod(0o600)

        atomic_write_text(target, '{"rewritten": true}')

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_the_temporary_is_never_wider_than_the_file_it_replaces(self, tmp_path: Path):
        target = tmp_path / "manifest.json"
        atomic_write_text(target, "{}")
        target.chmod(0o600)

        seen: list[int] = []
        with atomic_write(target) as handle:
            # Whatever a reader could open between here and the rename — the
            # window that holds the payload while it is written and fsynced.
            seen = [stat.S_IMODE(p.stat().st_mode) for p in _temps(tmp_path)]
            handle.write(b"{}")

        assert seen == [0o600]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_inherited_bits_are_not_narrowed_by_the_umask(self, tmp_path: Path):
        target = tmp_path / "manifest.json"
        atomic_write_text(target, "{}")
        target.chmod(0o664)

        previous = os.umask(0o077)  # would strip the group bits from a bare create
        try:
            atomic_write_text(target, '{"rewritten": true}')
        finally:
            os.umask(previous)

        assert stat.S_IMODE(target.stat().st_mode) == 0o664

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_a_missing_target_is_the_only_stat_failure_treated_as_absent(self, tmp_path: Path):
        loop = tmp_path / "manifest.json"
        loop.symlink_to(loop)  # stat raises ELOOP, not FileNotFoundError

        with pytest.raises(OSError):
            atomic_write_text(loop, "{}")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_a_brand_new_file_gets_the_umask_default(self, tmp_path: Path):
        reference = tmp_path / "reference.json"
        reference.write_text("{}", encoding="utf-8")  # what write_text would have given
        target = tmp_path / "manifest.json"

        atomic_write_text(target, "{}")

        assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


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
        _fail_the_commit(monkeypatch)

        with pytest.raises(OSError):
            atomic_write_text(target, "second")

        assert target.read_text(encoding="utf-8") == "first"
        assert [p.name for p in tmp_path.iterdir()] == ["record.json"]

    def test_an_unencodable_value_never_creates_a_file(self, tmp_path: Path):
        target = tmp_path / "record.txt"

        with pytest.raises(UnicodeEncodeError):
            atomic_write_text(target, "\ud800", encoding="utf-8")

        assert not list(tmp_path.iterdir())

    def test_newlines_are_written_exactly_as_given(self, tmp_path: Path):
        target = tmp_path / "record.json"
        atomic_write_text(target, '{\n  "a": 1\n}')

        assert target.read_bytes() == b'{\n  "a": 1\n}'

    def test_debris_from_a_killed_writer_does_not_stop_the_next_write(self, tmp_path: Path):
        # Reclaiming what a killed process left is #445; what this pins is
        # that it never stands in the way of the write that follows.
        (tmp_path / f"record.json{TMP_MARKER}999-deadbeef").write_text("junk", encoding="utf-8")

        atomic_write_text(tmp_path / "record.json", "fresh")

        assert (tmp_path / "record.json").read_text(encoding="utf-8") == "fresh"


class TestManifestWritesAreAtomic:
    def test_persistence_failure_keeps_the_previous_manifest(self, tmp_path: Path, monkeypatch):
        manifest = _manifest()
        save_run_manifest(manifest, tmp_path)
        manifest.stages_completed.append("splits")
        _fail_the_commit(monkeypatch)

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
        _fail_the_commit(monkeypatch)

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
        _fail_the_commit(monkeypatch)
        orchestrator._write_latest_pointer()  # warns, never raises

        pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
        assert pointer["run_dir"] == "run-one"
        assert not _temps(tmp_path)


class TestResumeReadsWhatSurvived:
    """The crash-resume path #424 exists for, driven end to end."""

    def test_a_quarantined_run_resumes_past_a_killed_attempts_debris(self, tmp_path: Path):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        failed_dir = tmp_path / "failed" / "run-one"
        failed_dir.mkdir(parents=True)
        save_run_manifest(_manifest(run_id="run-one"), failed_dir)
        (failed_dir / f"manifest.json{TMP_MARKER}999-deadbeef").write_text(
            "half a manifest", encoding="utf-8"
        )

        orchestrator = PipelineOrchestrator(PipelineConfig(resume="run-one"), output_base=tmp_path)
        orchestrator._setup_resume()

        assert orchestrator.manifest is not None
        assert orchestrator.manifest.stages_completed == ["data"]
