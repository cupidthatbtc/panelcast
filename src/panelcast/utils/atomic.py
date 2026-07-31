"""Replace a file whole, or leave the previous one alone.

Every durable record panelcast keeps — the run manifest, data-root stamps, the
`latest` pointer, checkpoint blocks — is read back by a later process that has
no way to tell a truncated file from a short one. So they are all written the
same way: to a temporary beside the target, flushed and fsynced, then renamed
over it. A kill anywhere in that sequence leaves the old file intact and no
debris behind.

This lives here, rather than beside any one caller, because the alternative is
each caller re-deriving it and getting a different subset right (#424).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

__all__ = ["TMP_MARKER", "atomic_write", "atomic_write_text", "fsync_dir"]

# In the temporary's name so a crashed writer's leftovers are identifiable.
TMP_MARKER = ".tmp-"


def fsync_dir(directory: Path) -> None:
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
    tmp = path.with_name(f"{path.name}{TMP_MARKER}{os.getpid()}-{uuid4().hex[:8]}")
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
    fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace ``path`` with ``text``.

    Encoded before the temporary is opened, so a value that cannot be encoded
    fails without having created a file at all.
    """
    data = text.encode(encoding)
    with atomic_write(path) as handle:
        handle.write(data)
