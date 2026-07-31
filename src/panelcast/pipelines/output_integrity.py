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

    A run-owned path maps whether or not the target survived the move: where a
    deleted artifact *should* be is the useful answer, and reporting the
    pre-move location instead turns a plain deletion into an apparent escape
    from the run root. Paths the run does not own — shared data roots, external
    files — are returned untouched.

    Passed by callers that may be looking at a moved run — `runs verify`
    resolves active and quarantined runs alike and cannot know which it has
    until it looks. An active run's paths exist, so this returns before mapping
    anything. The skip path follows the active pointer and never passes it.
    """
    if path.exists():
        return path
    parts = path.parts
    if run_dir.name in parts:
        return run_dir.joinpath(*parts[parts.index(run_dir.name) + 1 :])
    return path


def _key_encoded_path(key: str) -> Path | None:
    """The declared path a manifest key encodes, or None for a dynamic key.

    A stage records its declared outputs as ``<stage>:<path.as_posix()>``, so
    the key carries the path the stage said it would write and a recorded value
    that disagrees with its own key is a redirect. Reading it from the key
    means both callers can refuse that, not only the one holding stage objects.

    Dynamic keys are ``<stage>:<name>`` — a run_fn result label, not a path —
    and are left unbound, which is what makes containment the only thing behind
    them. The separator is the discriminator: a declared output always has one,
    since every artifact root is a directory.
    """
    _, _, suffix = key.partition(":")
    return Path(suffix) if "/" in suffix else None


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

    A recorded path is bound to the path its key encodes, so a manifest cannot
    redirect a static output at another file that happens to hash correctly —
    including one inside its own run directory, where containment has nothing
    to say. ``declared`` lets a caller holding stage objects state that binding
    directly, and is also what separates a *declared* output going missing
    (disk contradicts the manifest) from a dynamic one (nothing said this run
    still owned it). Both are compared in *recorded* coordinates, before
    ``reroot`` is applied, so a caller passing ``declared`` and ``reroot``
    together must give the pre-move spelling the manifest uses rather than the
    moved run's — otherwise every key reads as unbound.
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
        bound = declared.get(key) or _key_encoded_path(key)
        if bound is not None:
            try:
                if path.resolve() != bound.resolve():
                    yield OutputVerdict(
                        key,
                        UNBOUND,
                        "recorded output path disagrees with its manifest key",
                        untrusted=True,
                    )
                    continue
            except (OSError, ValueError):
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
