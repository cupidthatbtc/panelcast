"""One output-integrity primitive behind `--skip-existing` and `runs verify` (#385).

Both callers re-hash the outputs a manifest recorded, and both have to answer
the same question: does what is on disk still prove the manifest's claim? They
differ only in what they do with the answer and in where the run directory
lives — `runs verify` may be looking at a run quarantined under
`outputs/failed/`, while the skip path only ever follows the active pointer.
Everything else — which keys are verifiable at all, containment, declared-path
binding, directory hashing — is one implementation here, so the two cannot
drift apart in what they accept as proof.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from panelcast.utils.hashing import sha256_path

# Where the orchestrator moves a failed run; `paths.py` reserves the name.
QUARANTINE_DIR = "failed"

# Verdict labels, doubling as the `runs verify` status column.
OK = "OK"
MISSING = "MISSING"
MODIFIED = "MODIFIED"
UNBOUND = "UNBOUND"
UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class OutputVerdict:
    """What one manifest output key proves, and what it does not.

    ``untrusted`` means disk actively contradicts the manifest; a not-ok
    verdict that is not untrusted is merely unproven — a legacy manifest, a
    half-recorded key, a dynamic output that is gone. A caller that consumes
    artifacts must refuse both; only the wording differs.
    """

    key: str
    label: str
    reason: str = ""
    path: Path | None = None
    untrusted: bool = False

    @property
    def ok(self) -> bool:
        return self.label == OK


def contained_path(path: Path, roots: Sequence[Path]) -> Path | None:
    """The *resolved* ``path``, but only if it lands inside one of ``roots``.

    Manifests are just files on disk; a tampered one must not be able to aim
    the integrity check at something outside the workspace it describes.
    Resolution is what makes a symlinked component that leaves the root fail
    here rather than at the eventual read — and the resolved form is what is
    returned, so the caller reads and hashes the path this proved rather than
    walking the same symlink a second time and possibly landing elsewhere.

    A root that will not resolve is skipped rather than fatal. The roots are a
    shared workspace an operator may have symlinked, and one bad entry among
    eight must not turn every output in every run into an apparent escape.
    """
    try:
        resolved = path.resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    for root in roots:
        try:
            resolved_root = root.resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        if resolved == resolved_root or resolved_root in resolved.parents:
            return resolved
    return None


def reroot_under(path: Path, run_dir: Path) -> Path:
    """``path`` re-rooted at ``run_dir`` when the run itself was moved.

    A quarantined run keeps the manifest it was written with, so its recorded
    paths still name ``outputs/<id>/...`` while the artifacts now live under
    ``outputs/failed/<id>/...``. Deliberately generic over what the path is —
    the same mapping applies to run-owned *inputs* (#420), which this module
    does not verify but must not stand in the way of.

    "Run-owned" means recorded as ``<output base>/<run id>/<rest>``: the run
    id alone is not enough, since a bare-name match would launder
    ``/somewhere/else/<id>/metrics.json`` into the run directory and let a
    manifest describing a different workspace verify clean.

    The base is matched by *name*, not by identity. A manifest records whatever
    ``--output-base`` was spelled as — usually the relative default ``outputs``
    — while the run directory now in hand may be absolute, relocated, or both,
    and comparing them as paths would resolve the recorded one against whatever
    directory the command happens to run in, so a relocated workspace would
    read as tampering. The name comes from the run's parent, or its grandparent
    when the run sits under the quarantine directory.

    Be precise about what that admits: it refuses a tree whose base is spelled
    differently, not every foreign one. Another checkout of the *same* project
    also spells its base ``outputs``, so a recorded path from it maps here —
    and with a relative recorded base there is nothing left to tell relocation
    and impersonation apart, since they are the same string. What stands behind
    it is that the mapping only ever aims *into* this run's directory: the
    result is judged by containment and then by the recorded hash, so a foreign
    manifest passes only where the run id, the run-relative path and the bytes
    all already agree.

    A run-owned path maps unconditionally — not only when the recorded location
    has gone missing. The run directory in hand is the authority for that run's
    own artifacts, so consulting the recorded path's existence first would let
    the working directory decide which copy gets verified: a stale relative
    copy under cwd would be checked, and then refused as outside the roots,
    while the real artifact under ``--output-base`` went unread. It also means
    a deleted artifact reports where it *should* be rather than turning a plain
    deletion into an apparent escape. For an active run the mapping is a no-op.

    Paths the run does not own — shared data roots, external files — are
    returned untouched, and containment judges them where they lie.

    Passed by callers that may be looking at a moved run: `runs verify`
    resolves active and quarantined runs alike and cannot know which it has
    until it looks. The skip path follows the active pointer and never passes
    it.
    """
    base_name = _output_base_name(run_dir)
    parts = path.parts
    if not base_name:
        # An output base of `.` contributes no name, so the recorded path is
        # already run-relative and the id must be its *first* component.
        # Scanning for it anywhere would be the bare-name match this refuses.
        if parts[:1] == (run_dir.name,):
            return run_dir.joinpath(*parts[1:])
        return path
    # `<base>/<id>` as a unit, matched right to left. Scanning for the run id
    # alone and then looking left would compare the wrong pair whenever the id
    # also appears inside the output base's own path, and taking the first
    # match would pick that earlier occurrence over the real one.
    marker = (base_name, run_dir.name)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index : index + 2] == marker:
            return run_dir.joinpath(*parts[index + 2 :])
    return path


def _output_base_name(run_dir: Path) -> str:
    """The name of the output base ``run_dir`` sits in, seeing past quarantine."""
    parent = run_dir.parent
    return parent.parent.name if parent.name == QUARANTINE_DIR else parent.name


def verify_output_records(
    outputs: Mapping[str, str] | None,
    output_hashes: Mapping[str, str] | None,
    *,
    roots: Sequence[Path],
    prefix: str = "",
    declared: Mapping[str, Path] | None = None,
    reroot: Path | None = None,
) -> Iterator[OutputVerdict]:
    """Verdicts for every recorded output key, in key order, lazily.

    Yielding lets a caller that only needs the first failure stop before
    hashing the rest, while `runs verify` drains it for a full report.

    The key set is the union of ``outputs`` and ``output_hashes`` (filtered by
    ``prefix``): a key present on only one side proves nothing, so it is
    unverifiable rather than silently skipped by whichever map the caller
    happened to iterate.

    ``declared`` binds a key to the path the stage says it writes, so a
    manifest cannot redirect a static output at another file that happens to
    hash correctly — including one inside the run's own directory, where
    containment has nothing to say. It also separates a *declared* output going
    missing (disk contradicts the manifest) from a dynamic one (nothing said
    this run still owned it). A caller without stage objects passes nothing and
    gets neither; the manifest does not record which keys were declared, so
    there is nothing to read it from, and inferring it from key shape would be
    a guess about every run_fn label ever written (#439). The binding is
    compared in *recorded* coordinates, before ``reroot`` is applied, so a
    caller passing both must give the pre-move spelling the manifest uses
    rather than the moved run's — otherwise every key reads as unbound.
    """
    recorded = {k: v for k, v in (outputs or {}).items() if k.startswith(prefix)}
    hashes = {k: v for k, v in (output_hashes or {}).items() if k.startswith(prefix)}
    declared = declared or {}

    for key in sorted(set(recorded) | set(hashes)):
        expected = hashes.get(key)
        path_str = recorded.get(key)
        if not expected:
            yield OutputVerdict(key, UNVERIFIABLE, "recorded output has no hash")
            continue
        if not path_str:
            yield OutputVerdict(key, UNVERIFIABLE, "hashed output has no recorded path")
            continue
        path = Path(path_str)
        if key in declared:
            try:
                if path.resolve() != declared[key].resolve():
                    yield OutputVerdict(
                        key,
                        UNBOUND,
                        "recorded output path disagrees with its manifest key",
                        untrusted=True,
                    )
                    continue
            except (OSError, ValueError, RuntimeError):
                yield OutputVerdict(
                    key, UNBOUND, "recorded output path is unreadable", untrusted=True
                )
                continue
        if reroot is not None:
            path = reroot_under(path, reroot)
        contained = contained_path(path, roots)
        if contained is None:
            yield OutputVerdict(
                key, UNBOUND, "recorded output path escapes the run roots", untrusted=True
            )
            continue
        path = contained
        if not path.exists():
            # Same reason either way; the difference is whether disk is
            # *contradicting* the manifest. Without a declared path behind the
            # key nothing says this run still owns it, so its absence is
            # unproven rather than corrupt.
            yield OutputVerdict(
                key, MISSING, "recorded output is missing", path, untrusted=key in declared
            )
            continue
        try:
            actual = sha256_path(path)
        except OSError as exc:
            yield OutputVerdict(
                key, MISSING, f"recorded output unreadable: {exc}", path, untrusted=True
            )
            continue
        if actual != expected:
            yield OutputVerdict(
                key, MODIFIED, "recorded output no longer matches its hash", path, untrusted=True
            )
            continue
        yield OutputVerdict(key, OK, path=path)


__all__ = [
    "MISSING",
    "MODIFIED",
    "OK",
    "UNBOUND",
    "UNVERIFIABLE",
    "OutputVerdict",
    "contained_path",
    "reroot_under",
    "verify_output_records",
]
