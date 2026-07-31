"""Replace a file whole, or leave the previous one alone.

Every durable record panelcast keeps — the run manifest, data-root stamps, the
`latest` pointer, checkpoint blocks — is read back by a later process that has
no way to tell a truncated file from a short one. So they are all written the
same way: to a temporary beside the target, flushed and fsynced, then renamed
over it. A reader sees the previous file or the new one, never a partial one.

This lives here, rather than beside any one caller, because the alternative is
each caller re-deriving it and getting a different subset right (#424).

Three consequences of committing by rename, all deliberate:

- The target gets a new inode on every write. Anything holding the *file*
  rather than the path — a hard link, a file-granular bind mount, ``tail -f``
  — keeps seeing the old contents, and a symlink *at* the target path is
  replaced rather than written through, which is what ``write_text`` did.
  `latest.json` is the plausible victim; follow the directory, not the file.
- Bytes are written through a binary handle, so line endings are whatever the
  caller passed. `Path.write_text` translated ``\\n`` to ``\\r\\n`` on Windows,
  and the records that moved here (manifest, stamp, pointer) are LF on every
  platform now, matching the checkpoint artifacts that were always binary.
- A killed process leaves its temporary behind, where a fixed temp name would
  have been overwritten by the next attempt. Reclaiming those is #445.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

__all__ = ["TMP_MARKER", "atomic_write", "atomic_write_text", "fsync_dir"]

# In the temporary's name so a killed writer's leftovers are identifiable, and
# so two writers racing on one target cannot end up sharing a scratch file.
TMP_MARKER = ".tmp-"


def fsync_dir(directory: Path) -> None:
    """Persist a rename in the directory entry itself (POSIX only)."""
    if os.name == "nt":  # pragma: no cover - no directory handles to fsync on Windows
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


def _target_mode(path: Path) -> int | None:
    """Permission bits of the file being replaced, or None when there is none.

    ``write_text`` truncated in place and kept the mode; a rename brings the
    replacement's own. The manifest records the command line and flags
    verbatim, so an operator who tightened permissions on one must not have
    them widened again at the next stage boundary.

    Follows symlinks, so replacing a symlinked target inherits the mode of the
    file it pointed at — the one whose content the reader was getting.

    Permission bits only. A rename still drops POSIX ACLs, xattrs and SELinux
    labels that in-place truncation preserved; nothing here restores those.
    """
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return None


def _open_temp(tmp: Path, mode: int | None) -> BinaryIO:
    """Create ``tmp`` already at the mode it will be renamed with.

    Not created-then-narrowed: a file exists between those two calls, and
    permission is checked at ``open``, so a reader who gets a descriptor in
    that window keeps reading everything written afterwards. With no previous
    target there is nothing to inherit and the umask default stands, which is
    what ``write_text`` gave.

    ``O_EXCL`` makes the name collision the pid-and-uuid suffix guards against
    an error rather than a silent share.
    """
    if mode is None:
        return tmp.open("wb")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    return os.fdopen(fd, "wb")


@contextmanager
def atomic_write(path: Path) -> Iterator[BinaryIO]:
    """Yield a handle whose contents replace ``path`` atomically, or not at all.

    The temporary lives beside the target so the rename stays within one
    filesystem, and it is removed on any failure — an interrupted write leaves
    the previous file untouched. A process killed outright cannot run that
    cleanup, and what it leaves is #445.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}{TMP_MARKER}{os.getpid()}-{uuid4().hex[:8]}")
    mode = _target_mode(path)
    committed = False
    try:
        with _open_temp(tmp, mode) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        committed = True
    finally:
        if not committed:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                # Never let tidying up replace the failure that got us here —
                # that failure is the whole diagnostic.
                pass
    fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace ``path`` with ``text``.

    Encoded before the temporary is opened, so a value that cannot be encoded
    fails without having created a file at all. Written as bytes, so the text
    lands exactly as given — no platform newline translation.
    """
    data = text.encode(encoding)
    with atomic_write(path) as handle:
        handle.write(data)
