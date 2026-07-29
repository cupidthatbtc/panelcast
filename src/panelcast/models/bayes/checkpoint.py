"""Transactional on-disk store for checkpointed MCMC sampling (#366).

The store exists so a killed fit resumes at the last block instead of the
start, which only helps if the resume is provably the same Markov chain. Three
rules make that true:

- **Immutable once committed.** Block ``i`` writes ``block_i.npz`` (its draws)
  and ``state_i.pkl`` (the sampler state it ended in). Nothing named by the
  cursor is rewritten, so a state can never move ahead of its commit record.
- **Atomic artifacts.** Every file is written to a temporary name, flushed,
  fsynced, then renamed over its final path. A reader sees the whole file or
  no file, never a truncated one.
- **Cursor last.** ``cursor.json`` is the commit record and is written only
  after every artifact it references is durable. It names each block and
  state by SHA-256, so a resume can prove it is loading the exact pair the
  committing process wrote.

A crash therefore leaves the previous commit or the new one, and load
validates format, identity, block count, draw counts, chain counts, site
names, shapes, and hashes before a single draw is trusted. Anything that does
not check out refuses loudly — silently mixing two fits' draws is the failure
this module exists to prevent. A checkpoint directory is single-writer; two
concurrent fits sharing it will fail hash/identity checks rather than coordinate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4

import numpy as np

__all__ = [
    "CHECKPOINT_FORMAT",
    "BlockRecord",
    "CheckpointError",
    "CheckpointStore",
    "atomic_write",
]

logger = logging.getLogger(__name__)

# Bump whenever the on-disk layout changes. Older layouts are refused, not
# guessed at: a mutable state.pkl (format 1) cannot be shown to match its
# cursor, so resuming one is exactly the corruption this module prevents.
CHECKPOINT_FORMAT = 2

CURSOR_NAME = "cursor.json"
_TMP_MARKER = ".tmp-"


class CheckpointError(ValueError):
    """The checkpoint cannot be proven to belong to this fit — refuse it.

    Subclasses ValueError so existing callers that catch the checkpoint
    refusal keep working.
    """


def _fsync_dir(directory: Path) -> None:
    """Persist a rename in the directory entry itself (POSIX only)."""
    if os.name == "nt":  # no directory handles to fsync on Windows
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/permission dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems reject it
        pass
    finally:
        os.close(fd)


@contextmanager
def atomic_write(path: Path) -> Iterator[BinaryIO]:
    """Yield a handle whose contents replace ``path`` atomically, or not at all.

    The temporary lives beside the target so the rename stays within one
    filesystem, and it is removed on any failure — an interrupted write leaves
    the previous file untouched and no debris behind.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}{_TMP_MARKER}{os.getpid()}-{uuid4().hex[:8]}")
    committed = False
    try:
        with tmp.open("wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        committed = True
    finally:
        if not committed:
            tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_map(arrays: dict[str, Any]) -> dict[str, list[int]]:
    return {name: [int(d) for d in np.asarray(value).shape] for name, value in arrays.items()}


@dataclass(frozen=True)
class BlockRecord:
    """What the cursor commits about one block: identity plus content hashes."""

    index: int
    draws: int
    num_chains: int
    sample_sites: dict[str, list[int]]
    extra_sites: dict[str, list[int]]
    block_file: str
    block_sha256: str
    state_file: str
    state_sha256: str

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "draws": self.draws,
            "num_chains": self.num_chains,
            "sample_sites": self.sample_sites,
            "extra_sites": self.extra_sites,
            "block_file": self.block_file,
            "block_sha256": self.block_sha256,
            "state_file": self.state_file,
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_json(cls, payload: Any, position: int) -> BlockRecord:
        if not isinstance(payload, dict):
            raise CheckpointError(f"block record {position} is not an object")
        try:
            record = cls(
                index=_as_int(payload["index"], f"block {position} index"),
                draws=_as_int(payload["draws"], f"block {position} draws"),
                num_chains=_as_int(payload["num_chains"], f"block {position} num_chains"),
                sample_sites=_as_shape_map(payload["sample_sites"], position, "sample_sites"),
                extra_sites=_as_shape_map(payload["extra_sites"], position, "extra_sites"),
                block_file=_as_name(payload["block_file"], f"block {position} block_file"),
                block_sha256=_as_hex(payload["block_sha256"], f"block {position} block_sha256"),
                state_file=_as_name(payload["state_file"], f"block {position} state_file"),
                state_sha256=_as_hex(payload["state_sha256"], f"block {position} state_sha256"),
            )
        except KeyError as exc:
            raise CheckpointError(f"block record {position} is missing {exc}") from exc
        if record.index != position:
            raise CheckpointError(
                f"block record at position {position} claims index {record.index}"
            )
        return record


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointError(f"{what} is not an integer: {value!r}")
    return value


def _as_hex(value: Any, what: str) -> str:
    is_digest = (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )
    if not is_digest:
        raise CheckpointError(f"{what} is not a sha256 digest: {value!r}")
    return value


def _as_name(value: Any, what: str) -> str:
    """A bare file name — never a path that could escape the checkpoint dir."""
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise CheckpointError(f"{what} is not a plain file name: {value!r}")
    return value


def _as_shape_map(value: Any, position: int, what: str) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        raise CheckpointError(f"block {position} {what} is not an object")
    out: dict[str, list[int]] = {}
    for name, shape in value.items():
        if not isinstance(name, str):
            raise CheckpointError(f"block {position} {what} has a non-string site name")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) for d in shape
        ):
            raise CheckpointError(f"block {position} {what}['{name}'] is not a shape")
        out[name] = list(shape)
    return out


class CheckpointStore:
    """Immutable, hash-verified block store with cursor-last commit semantics."""

    def __init__(
        self,
        directory: Path,
        *,
        identity: dict,
        block_sizes: list[int],
        num_chains: int,
    ) -> None:
        self.directory = Path(directory)
        self.identity = identity
        self.block_sizes = [int(size) for size in block_sizes]
        self.num_chains = int(num_chains)
        self._records: list[BlockRecord] = []

    # -- paths -------------------------------------------------------------

    @property
    def cursor_path(self) -> Path:
        return self.directory / CURSOR_NAME

    def block_path(self, index: int) -> Path:
        return self.directory / f"block_{index:04d}.npz"

    def state_path(self, index: int) -> Path:
        return self.directory / f"state_{index:04d}.pkl"

    @property
    def blocks_done(self) -> int:
        return len(self._records)

    # -- resume ------------------------------------------------------------

    def resume(self) -> tuple[int, Any]:
        """(committed block count, sampler state to continue from).

        Validates everything the cursor claims before returning: format,
        identity, block layout, chain count, per-block draw counts, and the
        SHA-256 of every artifact it references. A fresh directory resumes at
        block 0 with no state; anything unverifiable raises.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        self._remove_orphan_temps()
        if not self.cursor_path.exists():
            return 0, None

        payload = self._read_cursor()
        records = self._validate_cursor(payload)
        for record in records:
            block_path = self.directory / record.block_file
            self._verify_artifact(block_path, record.block_sha256, record)
            self._validate_block_headers(record)
            self._verify_artifact(self.directory / record.state_file, record.state_sha256, record)
        self._records = records
        if not records:
            return 0, None

        state_path = self.directory / records[-1].state_file
        try:
            with state_path.open("rb") as handle:
                state = pickle.load(handle)
        except Exception as exc:  # a hash-verified file that will not unpickle
            raise CheckpointError(
                f"checkpoint state {state_path} is unreadable ({exc}); delete the "
                "checkpoint directory to start over"
            ) from exc
        logger.info(
            "Resuming from checkpoint: %d/%d blocks done", len(records), len(self.block_sizes)
        )
        return len(records), state

    def _remove_orphan_temps(self) -> None:
        """Delete uncommitted temp files left by a killed writer."""
        for path in self.directory.glob(f"*{_TMP_MARKER}*"):
            try:
                path.unlink()
            except OSError as exc:
                raise CheckpointError(
                    f"checkpoint temp artifact {path} cannot be removed ({exc}); "
                    "delete the checkpoint directory to start over"
                ) from exc
            logger.info("Removed orphan checkpoint temp artifact %s", path)

    def _read_cursor(self) -> dict:
        try:
            payload = json.loads(self.cursor_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"checkpoint cursor {self.cursor_path} is unreadable ({exc}); delete the "
                "checkpoint directory to start over"
            ) from exc
        if not isinstance(payload, dict):
            raise CheckpointError(f"checkpoint cursor {self.cursor_path} is not an object")
        return payload

    def _validate_cursor(self, payload: dict) -> list[BlockRecord]:
        fmt = payload.get("format")
        if fmt != CHECKPOINT_FORMAT:
            raise CheckpointError(
                f"checkpoint at {self.directory} uses format {fmt!r}, this build writes "
                f"{CHECKPOINT_FORMAT}; delete it to start over"
            )
        if payload.get("identity") != self.identity:
            raise CheckpointError(
                f"checkpoint at {self.directory} belongs to a different fit "
                "(config, data, model, or numpyro/jax version changed); delete it to start over"
            )
        if payload.get("block_sizes") != self.block_sizes:
            raise CheckpointError(
                f"checkpoint at {self.directory} was written with block sizes "
                f"{payload.get('block_sizes')!r}, this fit needs {self.block_sizes!r}; "
                "delete it to start over"
            )
        if payload.get("num_chains") != self.num_chains:
            raise CheckpointError(
                f"checkpoint at {self.directory} holds {payload.get('num_chains')!r} chains, "
                f"this fit runs {self.num_chains}; delete it to start over"
            )

        blocks_done = _as_int(payload.get("blocks_done"), "blocks_done")
        if not 0 <= blocks_done <= len(self.block_sizes):
            raise CheckpointError(
                f"checkpoint cursor claims {blocks_done} of {len(self.block_sizes)} blocks done"
            )
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list) or len(raw_blocks) != blocks_done:
            raise CheckpointError(
                f"checkpoint cursor claims {blocks_done} blocks done but lists "
                f"{len(raw_blocks) if isinstance(raw_blocks, list) else 'no'} block records"
            )

        records = [BlockRecord.from_json(entry, i) for i, entry in enumerate(raw_blocks)]
        for record in records:
            if record.draws != self.block_sizes[record.index]:
                raise CheckpointError(
                    f"checkpoint block {record.index} holds {record.draws} draws, "
                    f"this fit needs {self.block_sizes[record.index]}"
                )
            if record.num_chains != self.num_chains:
                raise CheckpointError(
                    f"checkpoint block {record.index} holds {record.num_chains} chains, "
                    f"this fit runs {self.num_chains}"
                )
            if record.block_file != self.block_path(record.index).name:
                raise CheckpointError(
                    f"checkpoint block {record.index} names an unexpected draws file "
                    f"{record.block_file!r}"
                )
            if record.state_file != self.state_path(record.index).name:
                raise CheckpointError(
                    f"checkpoint block {record.index} names an unexpected state file "
                    f"{record.state_file!r}"
                )
        _require_consistent_sites(records)
        return records

    def _verify_artifact(self, path: Path, expected: str, record: BlockRecord) -> None:
        if not path.exists():
            raise CheckpointError(
                f"checkpoint block {record.index} references missing artifact {path}; "
                "delete the checkpoint directory to start over"
            )
        actual = _sha256_file(path)
        if actual != expected:
            raise CheckpointError(
                f"checkpoint artifact {path} does not match the hash the cursor committed "
                f"({actual[:12]} != {expected[:12]}); delete the checkpoint directory to start over"
            )

    def _validate_block_headers(self, record: BlockRecord) -> None:
        """Validate NPZ keys and shapes without materializing array payloads."""
        path = self.directory / record.block_file
        expected = {f"s.{name}": shape for name, shape in record.sample_sites.items()}
        expected.update({f"e.{name}": shape for name, shape in record.extra_sites.items()})
        try:
            with zipfile.ZipFile(path) as archive:
                found = {
                    name.removesuffix(".npy")
                    for name in archive.namelist()
                    if name.endswith(".npy")
                }
                if found != set(expected):
                    raise CheckpointError(
                        f"checkpoint block {record.index} holds keys {sorted(found)}, "
                        f"the cursor committed {sorted(expected)}"
                    )
                for name, shape in expected.items():
                    with archive.open(f"{name}.npy") as handle:
                        version = np.lib.format.read_magic(handle)
                        if version == (1, 0):
                            actual_shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                        elif version in ((2, 0), (3, 0)):
                            actual_shape, _, _ = np.lib.format.read_array_header_2_0(handle)
                        else:
                            raise ValueError(f"unsupported NPY version {version}")
                    if list(actual_shape) != list(shape):
                        raise CheckpointError(
                            f"checkpoint block {record.index} site '{name[2:]}' has shape "
                            f"{list(actual_shape)}, the cursor committed {shape}"
                        )
        except CheckpointError:
            raise
        except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
            raise CheckpointError(
                f"checkpoint block {record.index} archive {path} is unreadable ({exc}); "
                "delete the checkpoint directory to start over"
            ) from exc

    # -- append ------------------------------------------------------------

    def append(
        self,
        index: int,
        samples: dict[str, Any],
        extras: dict[str, Any],
        state: Any,
    ) -> None:
        """Commit one block: draws, then state, then the cursor that names both."""
        if index != len(self._records):
            raise CheckpointError(
                f"block {index} appended out of order; {len(self._records)} blocks are committed"
            )
        if index >= len(self.block_sizes):
            raise CheckpointError(
                f"block {index} is past the {len(self.block_sizes)}-block layout of this fit"
            )
        if not samples:
            raise CheckpointError(f"block {index} carries no sample sites")

        sample_shapes = _shape_map(samples)
        extra_shapes = _shape_map(extras)
        expected_draws = self.block_sizes[index]
        for name, shape in {**sample_shapes, **extra_shapes}.items():
            if len(shape) < 2 or shape[0] != self.num_chains or shape[1] != expected_draws:
                raise CheckpointError(
                    f"block {index} site '{name}' has shape {shape}, expected "
                    f"(chains={self.num_chains}, draws={expected_draws}, ...)"
                )

        payload = {f"s.{name}": np.asarray(value) for name, value in samples.items()}
        payload.update({f"e.{name}": np.asarray(value) for name, value in extras.items()})

        block_path = self.block_path(index)
        with atomic_write(block_path) as handle:
            # NumPy <2.2 treats `allow_pickle` as an archive member, so do not
            # pass it merely to satisfy newer stubs; the runtime default is True.
            cast(Any, np.savez)(handle, **payload)
        with np.load(block_path, allow_pickle=True) as written:
            # Parsing the archive directory proves the file landed complete.
            written.files
        block_sha = _sha256_file(block_path)

        state_path = self.state_path(index)
        with atomic_write(state_path) as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        state_sha = _sha256_file(state_path)

        record = BlockRecord(
            index=index,
            draws=expected_draws,
            num_chains=self.num_chains,
            sample_sites=sample_shapes,
            extra_sites=extra_shapes,
            block_file=block_path.name,
            block_sha256=block_sha,
            state_file=state_path.name,
            state_sha256=state_sha,
        )
        _require_consistent_sites([*self._records, record])
        self._commit(record)

    def _commit(self, record: BlockRecord) -> None:
        """Publish the block by writing the cursor — the only durable commit point."""
        records = [*self._records, record]
        payload = {
            "format": CHECKPOINT_FORMAT,
            "identity": self.identity,
            "num_chains": self.num_chains,
            "block_sizes": self.block_sizes,
            "blocks_done": len(records),
            "blocks": [item.to_json() for item in records],
        }
        with atomic_write(self.cursor_path) as handle:
            handle.write(json.dumps(payload, sort_keys=True).encode("utf-8"))
        self._records = records
        logger.info("Checkpoint block %d/%d committed", len(records), len(self.block_sizes))

    # -- load --------------------------------------------------------------

    def load_blocks(self) -> tuple[dict, dict]:
        """(grouped samples, chain-major-flat extra fields) across every block.

        Re-verifies each artifact's hash as it reads, so a block corrupted
        after its commit fails here instead of entering the posterior.
        """
        if len(self._records) != len(self.block_sizes):
            raise CheckpointError(
                f"checkpoint holds {len(self._records)} of {len(self.block_sizes)} blocks; "
                "the fit is incomplete"
            )
        samples: dict[str, list] = {}
        extra: dict[str, list] = {}
        for record in self._records:
            block_samples, block_extra = self._read_block(record)
            for name, value in block_samples.items():
                samples.setdefault(name, []).append(value)
            for name, value in block_extra.items():
                extra.setdefault(name, []).append(value)

        samples_grouped = {name: np.concatenate(v, axis=1) for name, v in samples.items()}
        extra_flat = {}
        for name, values in extra.items():
            grouped = np.concatenate(values, axis=1)
            extra_flat[name] = grouped.reshape(-1, *grouped.shape[2:])

        total_draws = sum(self.block_sizes)
        for name, value in samples_grouped.items():
            if value.shape[:2] != (self.num_chains, total_draws):
                raise CheckpointError(
                    f"reassembled site '{name}' has shape {value.shape}, expected "
                    f"(chains={self.num_chains}, draws={total_draws}, ...)"
                )
        return samples_grouped, extra_flat

    def _read_block(self, record: BlockRecord) -> tuple[dict, dict]:
        path = self.directory / record.block_file
        self._verify_artifact(path, record.block_sha256, record)

        expected_keys = {f"s.{name}" for name in record.sample_sites}
        expected_keys |= {f"e.{name}" for name in record.extra_sites}
        with np.load(path, allow_pickle=True) as archive:
            found = set(archive.files)
            if found != expected_keys:
                raise CheckpointError(
                    f"checkpoint block {record.index} holds keys {sorted(found)}, "
                    f"the cursor committed {sorted(expected_keys)}"
                )
            samples = {
                name: self._checked_array(archive[f"s.{name}"], name, shape, record)
                for name, shape in record.sample_sites.items()
            }
            extras = {
                name: self._checked_array(archive[f"e.{name}"], name, shape, record)
                for name, shape in record.extra_sites.items()
            }
        return samples, extras

    def _checked_array(
        self, value: np.ndarray, name: str, shape: list[int], record: BlockRecord
    ) -> np.ndarray:
        if list(value.shape) != list(shape):
            raise CheckpointError(
                f"checkpoint block {record.index} site '{name}' has shape {list(value.shape)}, "
                f"the cursor committed {shape}"
            )
        if len(shape) < 2 or shape[0] != record.num_chains or shape[1] != record.draws:
            raise CheckpointError(
                f"checkpoint block {record.index} site '{name}' has shape {shape}, expected "
                f"(chains={record.num_chains}, draws={record.draws}, ...)"
            )
        return value


def _require_consistent_sites(records: list[BlockRecord]) -> None:
    """Every block must carry the same sites with the same per-draw shapes.

    A site that appears, vanishes, or changes trailing shape between blocks
    means the blocks came from different models, and concatenating them would
    build a posterior that never existed.
    """
    if not records:
        return
    first = records[0]

    def trailing(shapes: dict[str, list[int]]) -> dict[str, list[int]]:
        return {name: shape[2:] for name, shape in shapes.items()}

    for record in records[1:]:
        for what, a, b in (
            ("sample", first.sample_sites, record.sample_sites),
            ("extra", first.extra_sites, record.extra_sites),
        ):
            if set(a) != set(b):
                raise CheckpointError(
                    f"checkpoint block {record.index} {what} sites {sorted(b)} differ from "
                    f"block {first.index}'s {sorted(a)}"
                )
            if trailing(a) != trailing(b):
                raise CheckpointError(
                    f"checkpoint block {record.index} {what} site shapes differ from "
                    f"block {first.index}'s"
                )
