"""Transactional checkpoint store (#366): fault injection at every write boundary.

A checkpoint only helps if a crash can never leave a cursor pointing at
artifacts that disagree with it. These tests drive the store with a
deterministic fake chain — each block's draws are a pure function of the state
it starts from — so an interrupted run can be resumed and compared against an
uninterrupted one exactly, at every point a write can fail.
"""

from __future__ import annotations

import json
import os
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import panelcast.models.bayes.checkpoint as ckpt
from panelcast.models.bayes.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointError,
    CheckpointStore,
    atomic_write,
)

BLOCK_SIZES = [4, 4, 3]
NUM_CHAINS = 2
IDENTITY = {"config": {"seed": 0, "num_samples": 11}, "data_hash": "abc123"}


class Boom(RuntimeError):
    """Stands in for the kill signal a checkpoint exists to survive."""


def _block_arrays(state, draws: int, num_chains: int = NUM_CHAINS):
    """One block of a deterministic 'chain' continuing from ``state``."""
    start = 0 if state is None else int(state["draws_done"])
    index = np.arange(start, start + draws, dtype=float)
    mu = np.stack([index + 1000.0 * chain for chain in range(num_chains)])
    beta = np.stack(
        [np.outer(index + 1000.0 * chain, [1.0, 2.0, 3.0]) for chain in range(num_chains)]
    )
    diverging = np.stack([(index % 3 == 0).astype(float) for _ in range(num_chains)])
    return {"mu": mu, "beta": beta}, {"diverging": diverging}, {"draws_done": start + draws}


def _store(directory: Path, **overrides) -> CheckpointStore:
    kwargs = {
        "identity": IDENTITY,
        "block_sizes": BLOCK_SIZES,
        "num_chains": NUM_CHAINS,
    }
    kwargs.update(overrides)
    return CheckpointStore(directory, **kwargs)


def _run(directory: Path, **overrides) -> tuple[dict, dict]:
    """Run (or resume) the fake chain to completion and return the posterior."""
    store = _store(directory, **overrides)
    start, state = store.resume()
    for index in range(start, len(BLOCK_SIZES)):
        samples, extras, state = _block_arrays(state, BLOCK_SIZES[index])
        store.append(index, samples, extras, state)
    return store.load_blocks()


def _assert_same(a: tuple[dict, dict], b: tuple[dict, dict]) -> None:
    for left, right in zip(a, b, strict=True):
        assert set(left) == set(right)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=f"site {name} differs")


def _tmp_files(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if ".tmp-" in p.name]


def _cursor(directory: Path) -> dict:
    return json.loads((directory / "cursor.json").read_text(encoding="utf-8"))


@contextmanager
def _fail_before_write(monkeypatch, target: str, occurrence: int = 1):
    """Crash on entering the atomic write for ``target`` — nothing hits disk."""
    real = ckpt.atomic_write
    seen = {"n": 0}

    @contextmanager
    def guarded(path):
        if Path(path).name.startswith(target):
            seen["n"] += 1
            if seen["n"] == occurrence:
                raise Boom(f"crash before writing {path}")
        with real(path) as handle:
            yield handle

    monkeypatch.setattr(ckpt, "atomic_write", guarded)
    yield
    monkeypatch.setattr(ckpt, "atomic_write", real)


@contextmanager
def _fail_before_replace(monkeypatch, target: str, occurrence: int = 1):
    """Crash after the temp file is complete but before it becomes ``target``."""
    real = os.replace
    seen = {"n": 0}

    def guarded(src, dst, *args, **kwargs):
        if Path(dst).name.startswith(target):
            seen["n"] += 1
            if seen["n"] == occurrence:
                raise Boom(f"crash before replacing {dst}")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", guarded)
    yield
    monkeypatch.setattr(os, "replace", real)


# Every point a block-1 write can fail: (injector, target file, which write).
# The cursor is written once per committed block, so block 1's commit is the
# second one.
WRITE_BOUNDARIES = [
    (_fail_before_write, "block_0001.npz", 1),
    (_fail_before_replace, "block_0001.npz", 1),
    (_fail_before_write, "state_0001.pkl", 1),
    (_fail_before_replace, "state_0001.pkl", 1),
    (_fail_before_write, "cursor.json", 2),
    (_fail_before_replace, "cursor.json", 2),
]


class TestAtomicWrite:
    def test_replaces_the_target_only_on_success(self, tmp_path):
        target = tmp_path / "payload.bin"
        target.write_bytes(b"original")

        with atomic_write(target) as handle:
            handle.write(b"replacement")
        assert target.read_bytes() == b"replacement"
        assert not _tmp_files(tmp_path)

    def test_failure_leaves_the_previous_file_and_no_debris(self, tmp_path):
        target = tmp_path / "payload.bin"
        target.write_bytes(b"original")

        with pytest.raises(Boom):
            with atomic_write(target) as handle:
                handle.write(b"partial")
                raise Boom("mid-write")

        assert target.read_bytes() == b"original"
        assert not _tmp_files(tmp_path)

    def test_failed_first_write_leaves_no_target(self, tmp_path):
        target = tmp_path / "payload.bin"
        with pytest.raises(Boom):
            with atomic_write(target) as handle:
                handle.write(b"partial")
                raise Boom("mid-write")
        assert not target.exists()
        assert not _tmp_files(tmp_path)

    def test_concurrent_writers_do_not_share_a_temp_name(self, tmp_path):
        target = tmp_path / "payload.bin"
        with atomic_write(target) as first, atomic_write(target) as second:
            first.write(b"a")
            second.write(b"bb")
            names = {p.name for p in _tmp_files(tmp_path)}
            assert len(names) == 2


class TestUninterruptedRun:
    def test_writes_immutable_per_block_artifacts(self, tmp_path):
        _run(tmp_path)
        for index in range(len(BLOCK_SIZES)):
            assert (tmp_path / f"block_{index:04d}.npz").exists()
            assert (tmp_path / f"state_{index:04d}.pkl").exists()
        assert not (tmp_path / "state.pkl").exists()
        assert not _tmp_files(tmp_path)

    def test_cursor_records_every_artifact_by_hash(self, tmp_path):
        _run(tmp_path)
        cursor = _cursor(tmp_path)
        assert cursor["format"] == CHECKPOINT_FORMAT
        assert cursor["identity"] == IDENTITY
        assert cursor["block_sizes"] == BLOCK_SIZES
        assert cursor["blocks_done"] == len(BLOCK_SIZES)
        for index, record in enumerate(cursor["blocks"]):
            assert record["index"] == index
            assert record["draws"] == BLOCK_SIZES[index]
            assert record["num_chains"] == NUM_CHAINS
            assert record["block_file"] == f"block_{index:04d}.npz"
            assert record["state_file"] == f"state_{index:04d}.pkl"
            assert set(record["sample_sites"]) == {"mu", "beta"}
            assert set(record["extra_sites"]) == {"diverging"}
            for path, digest in (
                (record["block_file"], record["block_sha256"]),
                (record["state_file"], record["state_sha256"]),
            ):
                assert ckpt._sha256_file(tmp_path / path) == digest

    def test_posterior_is_the_whole_chain_in_order(self, tmp_path):
        samples, extras = _run(tmp_path)
        total = sum(BLOCK_SIZES)
        assert samples["mu"].shape == (NUM_CHAINS, total)
        assert samples["beta"].shape == (NUM_CHAINS, total, 3)
        np.testing.assert_array_equal(samples["mu"][0], np.arange(total, dtype=float))
        np.testing.assert_array_equal(samples["mu"][1], np.arange(total, dtype=float) + 1000.0)
        assert extras["diverging"].shape == (NUM_CHAINS * total,)

    def test_completed_checkpoint_resumes_without_running_a_block(self, tmp_path):
        first = _run(tmp_path)
        store = _store(tmp_path)
        start, _ = store.resume()
        assert start == len(BLOCK_SIZES)
        _assert_same(first, store.load_blocks())


class TestFaultInjection:
    """Every write boundary, interrupted; the resumed chain must be identical."""

    @pytest.fixture
    def reference(self, tmp_path_factory):
        return _run(tmp_path_factory.mktemp("uninterrupted"))

    @pytest.mark.parametrize(("injector", "target", "occurrence"), WRITE_BOUNDARIES)
    def test_crash_at_a_write_boundary_resumes_identically(
        self, tmp_path, monkeypatch, reference, injector, target, occurrence
    ):
        with injector(monkeypatch, target, occurrence), pytest.raises(Boom):
            _run(tmp_path)

        # Only the first block ever committed, whatever hit disk afterwards.
        assert _cursor(tmp_path)["blocks_done"] == 1
        assert not _tmp_files(tmp_path)

        _assert_same(reference, _run(tmp_path))

    @pytest.mark.parametrize(("injector", "target", "occurrence"), WRITE_BOUNDARIES)
    def test_resume_never_starts_from_a_state_past_the_cursor(
        self, tmp_path, monkeypatch, injector, target, occurrence
    ):
        with injector(monkeypatch, target, occurrence), pytest.raises(Boom):
            _run(tmp_path)

        start, state = _store(tmp_path).resume()
        assert start == 1
        # Block 0 ran 4 draws; a state carrying 8 would be the corruption the
        # old single mutable state.pkl allowed.
        assert state == {"draws_done": BLOCK_SIZES[0]}

    def test_orphan_artifacts_past_the_cursor_are_ignored_then_overwritten(
        self, tmp_path, monkeypatch, tmp_path_factory
    ):
        reference = _run(tmp_path_factory.mktemp("uninterrupted"))
        with _fail_before_write(monkeypatch, "cursor.json", 2), pytest.raises(Boom):
            _run(tmp_path)

        # Block 1's draws and state are on disk but uncommitted.
        assert (tmp_path / "block_0001.npz").exists()
        assert (tmp_path / "state_0001.pkl").exists()
        assert _cursor(tmp_path)["blocks_done"] == 1
        assert len(_cursor(tmp_path)["blocks"]) == 1

        _assert_same(reference, _run(tmp_path))
        assert _cursor(tmp_path)["blocks_done"] == len(BLOCK_SIZES)

    def test_crash_right_after_a_commit_resumes_at_the_next_block(self, tmp_path, tmp_path_factory):
        reference = _run(tmp_path_factory.mktemp("uninterrupted"))
        # The process dies in the window between one cursor rename and the
        # next block's first write.
        store = _store(tmp_path)
        _, state = store.resume()
        samples, extras, state = _block_arrays(state, BLOCK_SIZES[0])
        store.append(0, samples, extras, state)
        samples, extras, state = _block_arrays(state, BLOCK_SIZES[1])
        store.append(1, samples, extras, state)
        del store

        start, resumed_state = _store(tmp_path).resume()
        assert start == 2
        assert resumed_state == {"draws_done": BLOCK_SIZES[0] + BLOCK_SIZES[1]}
        _assert_same(reference, _run(tmp_path))

    def test_every_prefix_of_the_run_resumes_identically(self, tmp_path, tmp_path_factory):
        reference = _run(tmp_path_factory.mktemp("uninterrupted"))
        for stop_after in range(len(BLOCK_SIZES)):
            directory = tmp_path / f"stop{stop_after}"
            store = _store(directory)
            start, state = store.resume()
            for index in range(start, stop_after + 1):
                samples, extras, state = _block_arrays(state, BLOCK_SIZES[index])
                store.append(index, samples, extras, state)
            _assert_same(reference, _run(directory))


class TestLoadValidation:
    def _completed(self, tmp_path) -> Path:
        _run(tmp_path)
        return tmp_path

    def _rewrite_cursor(self, directory: Path, mutate) -> None:
        cursor = _cursor(directory)
        mutate(cursor)
        (directory / "cursor.json").write_text(json.dumps(cursor), encoding="utf-8")

    def test_missing_cursor_starts_from_scratch(self, tmp_path):
        start, state = _store(tmp_path).resume()
        assert (start, state) == (0, None)

    def test_unreadable_cursor_refuses(self, tmp_path):
        self._completed(tmp_path)
        (tmp_path / "cursor.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(CheckpointError, match="unreadable"):
            _store(tmp_path).resume()

    def test_legacy_format_refuses_instead_of_guessing(self, tmp_path):
        # The pre-#366 layout: no format key, one mutable state.pkl. It cannot
        # be proven to match its cursor, so it must not be resumed.
        (tmp_path).mkdir(parents=True, exist_ok=True)
        (tmp_path / "cursor.json").write_text(
            json.dumps({"identity": IDENTITY, "blocks_done": 1, "block_sizes": BLOCK_SIZES}),
            encoding="utf-8",
        )
        (tmp_path / "state.pkl").write_bytes(pickle.dumps({"draws_done": 4}))
        with pytest.raises(CheckpointError, match="format"):
            _store(tmp_path).resume()

    def test_identity_mismatch_refuses(self, tmp_path):
        self._completed(tmp_path)
        other = _store(tmp_path, identity={**IDENTITY, "data_hash": "different"})
        with pytest.raises(CheckpointError, match="different fit"):
            other.resume()

    def test_block_layout_mismatch_refuses(self, tmp_path):
        self._completed(tmp_path)
        with pytest.raises(CheckpointError, match="block sizes"):
            _store(tmp_path, block_sizes=[5, 6]).resume()

    def test_chain_count_mismatch_refuses(self, tmp_path):
        self._completed(tmp_path)
        with pytest.raises(CheckpointError, match="chains"):
            _store(tmp_path, num_chains=4).resume()

    def test_blocks_done_disagreeing_with_the_record_list_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c.update(blocks_done=2))
        with pytest.raises(CheckpointError, match="block records"):
            _store(tmp_path).resume()

    def test_blocks_done_past_the_layout_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c.update(blocks_done=99))
        with pytest.raises(CheckpointError, match="blocks done"):
            _store(tmp_path).resume()

    def test_non_integer_blocks_done_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c.update(blocks_done="two"))
        with pytest.raises(CheckpointError, match="blocks_done"):
            _store(tmp_path).resume()

    def test_tampered_block_file_refuses(self, tmp_path):
        self._completed(tmp_path)
        path = tmp_path / "block_0001.npz"
        path.write_bytes(path.read_bytes() + b"\x00")
        with pytest.raises(CheckpointError, match="does not match the hash"):
            _store(tmp_path).resume()

    def test_tampered_state_file_refuses(self, tmp_path):
        self._completed(tmp_path)
        path = tmp_path / "state_0002.pkl"
        path.write_bytes(pickle.dumps({"draws_done": 999}))
        with pytest.raises(CheckpointError, match="does not match the hash"):
            _store(tmp_path).resume()

    def test_missing_block_file_refuses(self, tmp_path):
        self._completed(tmp_path)
        (tmp_path / "block_0000.npz").unlink()
        with pytest.raises(CheckpointError, match="missing artifact"):
            _store(tmp_path).resume()

    def test_missing_state_file_refuses(self, tmp_path):
        self._completed(tmp_path)
        (tmp_path / "state_0001.pkl").unlink()
        with pytest.raises(CheckpointError, match="missing artifact"):
            _store(tmp_path).resume()

    def test_record_naming_a_path_outside_the_directory_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].update(block_file="../escape.npz"))
        with pytest.raises(CheckpointError, match="plain file name"):
            _store(tmp_path).resume()

    def test_record_renaming_an_artifact_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].update(block_file="block_0002.npz"))
        with pytest.raises(CheckpointError, match="unexpected draws file"):
            _store(tmp_path).resume()

    def test_record_with_a_bad_digest_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].update(block_sha256="nope"))
        with pytest.raises(CheckpointError, match="sha256 digest"):
            _store(tmp_path).resume()

    def test_record_missing_a_field_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].pop("state_sha256"))
        with pytest.raises(CheckpointError, match="missing"):
            _store(tmp_path).resume()

    def test_record_with_the_wrong_draw_count_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].update(draws=7))
        with pytest.raises(CheckpointError, match="draws"):
            _store(tmp_path).resume()

    def test_record_with_the_wrong_chain_count_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].update(num_chains=3))
        with pytest.raises(CheckpointError, match="chains"):
            _store(tmp_path).resume()

    def test_record_out_of_position_refuses(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(tmp_path, lambda c: c["blocks"][0].update(index=1))
        with pytest.raises(CheckpointError, match="claims index"):
            _store(tmp_path).resume()

    def test_blocks_with_different_sites_refuse(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(
            tmp_path,
            lambda c: c["blocks"][1]["sample_sites"].pop("beta"),
        )
        with pytest.raises(CheckpointError, match="sites"):
            _store(tmp_path).resume()

    def test_blocks_with_different_site_shapes_refuse(self, tmp_path):
        self._completed(tmp_path)
        self._rewrite_cursor(
            tmp_path,
            lambda c: c["blocks"][1]["sample_sites"].update(beta=[NUM_CHAINS, 4, 5]),
        )
        with pytest.raises(CheckpointError, match="shapes differ"):
            _store(tmp_path).resume()

    def test_site_shape_disagreeing_with_the_stored_array_refuses(self, tmp_path):
        self._completed(tmp_path)
        # Consistent across blocks, so it survives resume and must be caught
        # when the arrays themselves are read.
        self._rewrite_cursor(
            tmp_path,
            lambda c: [
                b["sample_sites"].update(beta=[NUM_CHAINS, b["draws"], 4]) for b in c["blocks"]
            ],
        )
        store = _store(tmp_path)
        store.resume()
        with pytest.raises(CheckpointError, match="site 'beta' has shape"):
            store.load_blocks()

    def test_block_holding_unexpected_keys_refuses(self, tmp_path):
        self._completed(tmp_path)
        store = _store(tmp_path)
        store.resume()
        record = store._records[0]
        # Rewrite the block with an extra array and re-point the cursor hash at
        # it: the key set alone must still refuse.
        path = tmp_path / record.block_file
        with np.load(path, allow_pickle=True) as archive:
            payload = {name: archive[name] for name in archive.files}
        payload["s.rogue"] = np.zeros((NUM_CHAINS, record.draws))
        with atomic_write(path) as handle:
            np.savez(handle, allow_pickle=True, **payload)
        self._rewrite_cursor(
            tmp_path,
            lambda c: c["blocks"][0].update(block_sha256=ckpt._sha256_file(path)),
        )
        store = _store(tmp_path)
        store.resume()
        with pytest.raises(CheckpointError, match="holds keys"):
            store.load_blocks()

    def test_load_before_every_block_is_committed_refuses(self, tmp_path):
        store = _store(tmp_path)
        state = None
        samples, extras, state = _block_arrays(state, BLOCK_SIZES[0])
        store.append(0, samples, extras, state)
        with pytest.raises(CheckpointError, match="incomplete"):
            store.load_blocks()


class TestAppendGuards:
    def test_out_of_order_append_refuses(self, tmp_path):
        store = _store(tmp_path)
        samples, extras, state = _block_arrays(None, BLOCK_SIZES[0])
        with pytest.raises(CheckpointError, match="out of order"):
            store.append(1, samples, extras, state)

    def test_append_past_the_layout_refuses(self, tmp_path):
        store = _store(tmp_path, block_sizes=[BLOCK_SIZES[0]])
        samples, extras, state = _block_arrays(None, BLOCK_SIZES[0])
        store.append(0, samples, extras, state)
        samples, extras, state = _block_arrays(state, BLOCK_SIZES[0])
        with pytest.raises(CheckpointError, match="past the"):
            store.append(1, samples, extras, state)

    def test_wrong_draw_count_refuses(self, tmp_path):
        store = _store(tmp_path)
        samples, extras, state = _block_arrays(None, BLOCK_SIZES[0] + 1)
        with pytest.raises(CheckpointError, match="expected"):
            store.append(0, samples, extras, state)

    def test_wrong_chain_count_refuses(self, tmp_path):
        store = _store(tmp_path)
        samples, extras, state = _block_arrays(None, BLOCK_SIZES[0], num_chains=NUM_CHAINS + 1)
        with pytest.raises(CheckpointError, match="expected"):
            store.append(0, samples, extras, state)

    def test_empty_block_refuses(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(CheckpointError, match="no sample sites"):
            store.append(0, {}, {}, {"draws_done": 0})

    def test_site_set_changing_between_blocks_refuses(self, tmp_path):
        store = _store(tmp_path)
        samples, extras, state = _block_arrays(None, BLOCK_SIZES[0])
        store.append(0, samples, extras, state)
        samples, extras, state = _block_arrays(state, BLOCK_SIZES[1])
        samples.pop("beta")
        with pytest.raises(CheckpointError, match="sites"):
            store.append(1, samples, extras, state)

    def test_a_refused_append_leaves_the_cursor_untouched(self, tmp_path):
        store = _store(tmp_path)
        samples, extras, state = _block_arrays(None, BLOCK_SIZES[0])
        store.append(0, samples, extras, state)
        committed = _cursor(tmp_path)

        bad_samples, bad_extras, bad_state = _block_arrays(state, BLOCK_SIZES[1] + 1)
        with pytest.raises(CheckpointError):
            store.append(1, bad_samples, bad_extras, bad_state)
        assert _cursor(tmp_path) == committed
        assert not _tmp_files(tmp_path)
