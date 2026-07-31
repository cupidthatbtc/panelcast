"""Replace a file whole, or leave the previous one alone.

Every durable record panelcast keeps — the run manifest, data-root stamps, the
`latest` pointer, checkpoint blocks — is read back by a later process that has
no way to tell a truncated file from a short one. So they are all written the
same way: to a temporary beside the target, flushed and fsynced, then renamed
over it. A reader sees the previous file or the new one, never a partial one.

This lives here, rather than beside any one caller, because the alternative is
each caller re-deriving it and getting a different subset right (#424).

Four consequences of committing by rename, all deliberate:

- The target gets a new inode on every write. Anything holding the *file*
  rather than the path — a hard link, a file-granular bind mount, ``tail -f``
  — keeps seeing the old contents, and a symlink *at* the target path is
  replaced rather than written through, which is what ``write_text`` did.
  `latest.json` is the plausible victim; follow the directory, not the file.
  The replacement is also owned by whoever wrote it, where truncating in place
  kept the original owner and group.
- Bytes are written through a binary handle, so line endings are whatever the
  caller passed. `Path.write_text` translated ``\\n`` to ``\\r\\n`` on Windows,
  and the records that moved here (manifest, stamp, pointer) are LF on every
  platform now, matching the checkpoint artifacts that were always binary.
- A killed process leaves its temporary behind, where a fixed temp name would
  have been overwritten by the next attempt. Reclaiming those is #445.
- On Windows the rename needs DELETE access on the destination, which another
  process holding it open denies, where truncating in place did not.
  ``_commit`` retries briefly rather than failing the write.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

__all__ = ["TMP_MARKER", "atomic_write", "atomic_write_text", "fsync_dir"]

# Stdlib logging, matching the checkpoint store rather than the pipeline: this
# module is reachable as a library, without the CLI having configured
# anything, and an unconfigured structlog prints unfiltered to stdout.
logger = logging.getLogger(__name__)

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

    Only a *missing* target answers None. Any other ``stat`` failure — a
    symlink loop, an unreadable path component — propagates rather than being
    read as "nothing there", which would rewrite an existing restrictive file
    at the umask default and silently widen it. Those failures would defeat
    the create a few lines later anyway; better to say so at the cause.

    Follows symlinks, so replacing a symlinked target inherits the mode of the
    file it pointed at — the one whose content the reader was getting.

    Permission bits only. A rename still drops owner, group, POSIX ACLs,
    xattrs and SELinux labels that in-place truncation preserved; nothing here
    restores those.
    """
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        return None


def _open_temp(tmp: Path, mode: int | None) -> BinaryIO:
    """Create ``tmp`` already at the mode it will be renamed with.

    Not created-then-narrowed: a file exists between those two calls, and
    permission is checked at ``open``, so a reader who gets a descriptor in
    that window keeps reading everything written afterwards.

    ``O_EXCL`` makes the name collision the pid-and-uuid suffix guards against
    an error rather than a silent share, and ``O_BINARY`` (nothing on POSIX)
    keeps the descriptor out of the Windows CRT's text mode, which would turn
    every ``\\n`` in a payload — including a checkpoint's — into ``\\r\\n``.

    With no previous target the create asks for ``0o666`` and the umask trims
    it to exactly what ``write_text`` would have produced; with one, the umask
    would *narrow* the bits being inherited, so ``fchmod`` restores them —
    after a create that was never wider than the target, so there is still no
    window. A filesystem that refuses ``chmod`` at all (CIFS without the unix
    extensions, exfat, some FUSE mounts) leaves the create's bits, which are
    never wider than the target's; refusing to write the manifest over that
    would be a worse answer than a file the umask narrowed.
    """
    fchmod = getattr(os, "fchmod", None)  # POSIX only; modes are advisory on Windows
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o666 if mode is None else mode)
    ours = True
    try:
        if mode is not None and fchmod is not None:
            try:
                fchmod(fd, mode)
            except OSError:
                pass
        # From here the descriptor belongs to ``fdopen``, which closes it on
        # its own error path. Closing it again could land on one the runtime
        # has since handed to something else.
        ours = False
        return os.fdopen(fd, "wb")
    except BaseException:
        if ours:
            with suppress(OSError):
                os.close(fd)
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        # Neither cleanup may stand in front of the failure that got us here.
        raise


_COMMIT_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def _commit(tmp: Path, path: Path) -> None:
    """Rename ``tmp`` over ``path``, retrying a Windows sharing violation.

    ``os.replace`` needs DELETE access on the destination, and CPython opens
    files without ``FILE_SHARE_DELETE``, so on Windows any other process
    holding the target open — a concurrent ``runs show``, a virus scanner
    touching a freshly written manifest — makes the rename raise
    ``PermissionError`` where the truncate-in-place it replaced would have
    succeeded. Every such holder is transient, so a bounded backoff turns a
    run killed at a stage boundary into a few milliseconds of delay.

    Not gated on the platform. A ``PermissionError`` that POSIX raises here is
    about the directory rather than a holder and will not clear, but it costs
    under a second to find that out, and one code path beats two.

    Logged, because a scanner touching every file it sees is a property of the
    machine rather than a one-off, and the checkpoint store commits once per
    sampling block: a fit that pays the backoff on every block is a fit that
    got mysteriously slower. Silently slow is worse to diagnose than loudly
    failed — and the outcome of the last attempt is logged, not just the fact
    that it was reached, so a write that recovered and one that did not do not
    read the same.
    """
    for attempt, delay in enumerate(_COMMIT_RETRY_DELAYS, start=1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            logger.debug("Rename of %s denied (attempt %d); retrying", path, attempt)
            time.sleep(delay)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Both causes, because the retry is deliberately not platform-gated:
        # naming only the Windows one sends a Linux operator hunting for a
        # process that does not exist.
        logger.warning(
            "Rename of %s still denied after %d retries: another process is holding it "
            "open, or this directory does not permit the rename",
            path,
            len(_COMMIT_RETRY_DELAYS),
        )
        raise
    logger.warning(
        "Rename of %s succeeded only on the final retry, after %.2fs of waiting",
        path,
        sum(_COMMIT_RETRY_DELAYS),
    )


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
    # Outside the try: a name collision means the file belongs to another
    # writer, and the cleanup below must never delete one we did not create.
    handle = _open_temp(tmp, mode)
    committed = False
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        _commit(tmp, path)
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
