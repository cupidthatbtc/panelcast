"""Replace a file whole, or leave the previous one alone.

Every durable record panelcast keeps — the run manifest, data-root stamps, the
`latest` pointer, checkpoint blocks — is read back by a later process that has
no way to tell a truncated file from a short one. So they are all written the
same way: to a temporary beside the target, flushed and fsynced, then renamed
over it. A reader sees the previous file or the new one, never a partial one.

This lives here, rather than beside any one caller, because the alternative is
each caller re-deriving it and getting a different subset right (#424).

Two consequences of committing by rename, both deliberate:

- The target gets a new inode on every write. Anything holding the *file*
  rather than the path — a hard link, a file-granular bind mount, ``tail -f``
  — keeps seeing the old contents. `latest.json` is the one plausible victim;
  follow the directory, not the file.
- Bytes are written through a binary handle, so line endings are whatever the
  caller passed. `Path.write_text` translated ``\\n`` to ``\\r\\n`` on Windows,
  and the records that moved here (manifest, stamp, pointer) are LF on every
  platform now, matching the checkpoint artifacts that were always binary.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

__all__ = [
    "TMP_MARKER",
    "atomic_write",
    "atomic_write_text",
    "fsync_dir",
    "sweep_orphan_temps",
]

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


def sweep_orphan_temps(directory: Path) -> list[Path]:
    """Remove temporaries a killed writer left in ``directory``; return them.

    The cleanup inside :func:`atomic_write` is a ``finally``, which covers an
    exception and nothing else — ``SIGKILL`` and power loss leave the temporary
    where it fell. Since each one is uniquely named, they accumulate rather
    than being overwritten by the next attempt, so somebody has to reclaim
    them.

    Deliberately *not* called from ``atomic_write`` itself: a concurrent writer
    on the same target is holding a temporary that is alive, not orphaned, and
    nothing about the file distinguishes the two. Call this where sole
    ownership of the directory is already established — the orchestrator takes
    a quarantined run directory back for a resume, and sweeps it there.

    Best-effort throughout: ``glob`` swallows its own directory-listing errors
    and yields nothing, and a temporary that will not delete is skipped, since
    failing to tidy is never worth failing the run over.
    """
    removed: list[Path] = []
    for path in sorted(Path(directory).glob(f"*{TMP_MARKER}*")):
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(path)
    return removed


def _inherit_mode(target: Path, handle: BinaryIO) -> None:
    """Give the replacement file an existing target's permission bits.

    ``write_text`` truncated in place and kept the mode; a rename brings the
    temporary's own, created under the current umask. The manifest records the
    command line and flags verbatim, so an operator who tightened permissions
    on one must not have them widened again at the next stage boundary.

    Applied to the open handle *before* the caller writes anything, not to the
    finished temporary: a chmod at commit time would leave the payload sitting
    at the umask default for the whole write and fsync — the longest and most
    interruptible part — which is the exposure the tightened mode was for.

    Permission bits only. A rename still drops POSIX ACLs, xattrs and SELinux
    labels that in-place truncation preserved; nothing here restores those.
    """
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except OSError:
        return  # nothing there yet: the umask default is the right answer
    try:
        os.fchmod(handle.fileno(), mode)
    except (AttributeError, OSError):  # no fchmod on Windows; modes are advisory there
        pass


@contextmanager
def atomic_write(path: Path) -> Iterator[BinaryIO]:
    """Yield a handle whose contents replace ``path`` atomically, or not at all.

    The temporary lives beside the target so the rename stays within one
    filesystem, and it is removed on any failure — an interrupted write leaves
    the previous file untouched. A process killed outright cannot run that
    cleanup; :func:`sweep_orphan_temps` reclaims what it leaves.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}{TMP_MARKER}{os.getpid()}-{uuid4().hex[:8]}")
    committed = False
    try:
        with tmp.open("wb") as handle:
            _inherit_mode(path, handle)
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
