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

from panelcast.paths import QUARANTINE_DIR
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

    ``untrusted`` separates a verdict that contradicts the manifest from one
    that merely fails to confirm it — sorted by what is *known*, not by which
    way the check failed, so the physical failure alone does not decide it.
    Stated as a rule rather than a list of cases, since the cases keep growing:
    an ``UNBOUND`` verdict is always untrusted — the manifest named a path that
    cannot be tied to this run, which is a fact about its claim — while
    ``MISSING`` takes it from whether a declared path stands behind the key,
    and ``UNVERIFIABLE`` never does, because nothing was shown either way. A
    caller that consumes artifacts must refuse all of them; only the wording
    differs.
    """

    key: str
    label: str
    reason: str = ""
    path: Path | None = None
    untrusted: bool = False

    @property
    def ok(self) -> bool:
        return self.label == OK


def _resolved_roots(roots: Sequence[Path]) -> tuple[Path, ...]:
    """The roots, resolved once, with the ones that will not resolve dropped.

    Hoisted out of the per-key check because the roots do not change while a
    manifest is being walked: resolving them inside it made the number of
    ``realpath`` walks the product of roots and recorded outputs, on the path
    whose whole claim is that checking is cheaper than recomputing.

    A root that will not resolve is skipped rather than fatal. The roots are a
    shared workspace an operator may have symlinked, and one bad entry among
    eight must not turn every output in every run into an apparent escape.
    Dropping it here rather than at the comparison keeps that decision in one
    place; an empty result then means nothing is contained, which is the
    refusal the callers already treat as "not contained".
    """
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except (OSError, ValueError, RuntimeError):
            continue
    return tuple(resolved)


def _is_contained(resolved: Path, roots: Sequence[Path]) -> bool:
    """Whether ``resolved`` — an already-resolved path — lands inside a root.

    ``roots`` must already be resolved too; ``_resolved_roots`` does that once
    per manifest walk.

    Module-private, because the precondition is load-bearing and unenforceable:
    handed an *unresolved* path, this answers about the literal join, so
    ``<root>/link/secret`` with ``link`` pointing out of the root is contained
    and the caller goes on to read it. Resolution lives in the caller so that
    "will not resolve" and "resolves outside" stay two answers rather than one
    — and keeping the function private is what stops that trade becoming a
    safety property some future caller has to remember.

    Manifests are just files on disk; a tampered one must not be able to aim
    the integrity check at something outside the workspace it describes.
    Resolution is what makes a symlinked component that leaves the root fail
    before the read rather than at it, and the caller reads and hashes the
    resolved path this answered about rather than walking the same symlink a
    second time and possibly landing elsewhere.

    """
    return any(
        resolved == root or root in resolved.parents for root in roots
    )


def reroot_under(path: Path, run_dir: Path) -> Path:
    """``path`` re-rooted at ``run_dir`` when the run itself was moved.

    A quarantined run keeps the manifest it was written with, so its recorded
    paths still name ``outputs/<id>/...`` while the artifacts now live under
    ``outputs/failed/<id>/...``. Deliberately generic over what the path is: the
    same mapping applies to run-owned *inputs*, which this module does not
    verify but does locate for the callers that do, through ``run_owned_path``
    (#420).

    "Run-owned" means recorded as ``<output base>/<run id>/<rest>``: the run
    id alone is not enough, since a bare-name match would launder
    ``/somewhere/else/<id>/metrics.json`` into the run directory and let a
    manifest describing a different workspace verify clean. An output base with
    no name — ``.`` or ``/`` — has no half to pair with, so that case trades
    the pair for a position and requires the id to be the path's *first*
    component, which a foreign path cannot satisfy without being run-relative.

    The base is matched by *name*, not by identity. A manifest records whatever
    ``--output-base`` was spelled as — usually the relative default ``outputs``
    — while the run directory now in hand may be absolute, relocated, or both,
    and comparing them as paths would resolve the recorded one against whatever
    directory the command happens to run in, so a relocated workspace would
    read as tampering. The name comes from the run's parent, or its grandparent
    when the run sits under the quarantine directory.

    Matching by name also means matching case-sensitively, on every platform,
    while the containment check that judges the result compares ``Path``s —
    which are case-*insensitive* on Windows. So a differently-cased
    ``--output-base`` against a case-insensitive filesystem is the one input
    whose verdict is platform-dependent, and the split is not symmetric:
    nothing re-roots either way, and then Windows finds the unmapped path
    contained and verifies the right bytes, while macOS — ``PosixPath``
    comparison over a filesystem that does not care about case — finds it
    outside the roots and reports a healthy run ``UNBOUND``, at tampering
    severity. Neither admits a foreign path, but one of them is wrong about a
    run that is fine. Making them agree means deciding case-folding for the
    layout as a whole, which is #440 rather than this module's to settle.

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


def run_owned_path(path: Path, run_dir: Path) -> Path | None:
    """Where ``run_dir`` holds ``path``, or None if the run does not own it.

    ``reroot_under`` followed by the containment step, returning the ownership
    answer the mapping already had to compute. `runs verify`'s input pass
    re-hashes a run-owned input at the location this gives it; `runs
    reproduce`'s pre-flight gate uses the None to tell an *external* input —
    raw data whose drift would invalidate the comparison — from one of the
    run's own products, which a fresh reproduction regenerates (#420).

    Two things ``reroot_under`` alone cannot give a caller that only stats and
    hashes. It maps unconditionally once the pair matches, so a recorded tail
    that climbs back out — ``outputs/<id>/../../etc/shadow`` — would be aimed
    into this run's directory and read there; ``verify_output_records`` catches
    that with the containment step it runs afterwards, and these callers have
    no such step, so here the guard travels with the mapping and an escaping
    tail reads as unowned. Unowned is all it means: the caller then treats the
    path as external and checks it wherever the manifest recorded it, which is
    what it did before any of this existed. What is bounded is where the
    *mapping* may aim, not what the manifest may name. And for an
    *active* run the mapping moves nothing — though it still rewrites the
    spelling onto ``run_dir``, which is why a changed path cannot answer
    ownership either way. Containment can, and it is the same question.

    Ownership is decided on the *resolved* location, so it also declines an
    artifact reached through a symlink inside the run directory that leaves it
    — a run whose ``models/`` points at shared storage, say. That is the bound
    on purpose, not an oversight about the recorded tail: it is the same bound
    ``verify_output_records`` applies to a path in the same place.

    Be exact about what that costs, since unowned does not mean unread. An
    unowned path is checked where the manifest recorded it, so on an active run
    a symlinked-out product still verifies through the link — the same file the
    output side refuses, which is an asymmetry in the *consequence* even though
    ownership is symmetric. What is given up is the re-rooting: quarantine that
    run and the recorded location is gone, so the input reads ``MISSING`` and
    the reproduce gate treats the product as external. That is #420's own
    failure surviving for one layout, and what makes the trade worth taking is
    that such a run already fails `runs verify` on its *outputs* under the same
    directory, which are ``UNBOUND`` — so the price is paid on runs that were
    not going to verify anyway, and it buys the two callers never disagreeing
    about which paths a run owns.

    Not folded into ``reroot_under`` itself, because for an *output* an
    escaping tail is worth reporting rather than quietly declining: the
    manifest is claiming something about a path outside the run, and only a
    caller that produces verdicts can say so.
    """
    # A run dir that will not resolve is dropped here, leaving no roots, so
    # nothing is contained and the answer is None — the same answer an escape
    # gets, which each caller then reads its own way.
    roots = _resolved_roots((run_dir,))
    mapped = reroot_under(path, run_dir)
    try:
        resolved = mapped.resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    return mapped if _is_contained(resolved, roots) else None


def _output_base_name(run_dir: Path) -> str:
    """The name of the output base ``run_dir`` sits in, seeing past quarantine.

    A run *id* of ``failed`` is impossible — the layout reserves it — but the
    output *base* is the operator's to name and nothing reserves anything
    there, so ``--output-base ./failed`` makes this read the grandparent when
    the run was never quarantined.

    That *widens* the match rather than narrowing it: the marker becomes
    ``(<grandparent>, <run id>)``, so a recorded path from a foreign workspace
    whose base happens to be named like this run's grandparent now pairs and
    gets mapped in, where the true base name would not have matched it. It is
    not a new class of hole — it is the same laundering the name-matching
    paragraph above already admits for another checkout spelling its base the
    same way, reached by a different spelling — and it is still bounded the
    same way: the mapping only aims *into* this run's directory, so the result
    must clear containment and then the recorded hash. But the effect is to
    accept more, not to leave more unmapped.
    """
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
    there is nothing to read it from. A declared key already encodes its path —
    what is missing is a way to tell a *label* apart from one, and inferring
    that from key shape would be a standing claim about every run_fn label ever
    written (#439). The binding is
    compared in *recorded* coordinates, before ``reroot`` is applied, so a
    caller passing both must give the pre-move spelling the manifest uses
    rather than the moved run's — otherwise every key reads as unbound.
    """
    recorded = {k: v for k, v in (outputs or {}).items() if k.startswith(prefix)}
    hashes = {k: v for k, v in (output_hashes or {}).items() if k.startswith(prefix)}
    declared = declared or {}
    contain_roots = _resolved_roots(roots)

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
        resolved: Path | None = None
        if key in declared:
            # Resolved separately so the verdict names the side that failed.
            # The recorded path is the manifest's claim; the declared one is
            # the stage's own configuration, and a workspace problem there
            # says nothing about the manifest.
            try:
                resolved = path.resolve()
            except (OSError, ValueError, RuntimeError):
                yield OutputVerdict(
                    key, UNBOUND, "recorded output path is unreadable", untrusted=True
                )
                continue
            try:
                declared_at = declared[key].resolve()
            except (OSError, ValueError, RuntimeError) as exc:
                yield OutputVerdict(
                    key, UNVERIFIABLE, f"the stage's declared path is unreadable: {exc}"
                )
                continue
            if resolved != declared_at:
                yield OutputVerdict(
                    key,
                    UNBOUND,
                    "recorded output path disagrees with its manifest key",
                    untrusted=True,
                )
                continue
        if reroot is not None:
            rerooted = reroot_under(path, reroot)
            if rerooted != path:
                resolved = None  # a different path, so the walk above is stale
            path = rerooted
        # Resolved here, not inside `_is_contained`, so a path the tool could
        # not locate is never reported as one that escaped: it may be sitting
        # squarely inside the run root. Reused from the binding when that ran
        # on this same path, so a declared key walks its symlink chain once and
        # containment proves the target the read will use.
        if resolved is None:
            try:
                resolved = path.resolve()
            except (OSError, ValueError, RuntimeError):
                yield OutputVerdict(
                    key, UNBOUND, "recorded output path is unreadable", untrusted=True
                )
                continue
        if not _is_contained(resolved, contain_roots):
            yield OutputVerdict(
                key, UNBOUND, "recorded output path escapes the run roots", untrusted=True
            )
            continue
        path = resolved
        # Both ways of failing to read it are the same kind of fact, so they
        # take the severity from the same place: with a declared path behind
        # the key, disk is *contradicting* the manifest; without one, nothing
        # says this run still owns the file, so it is unproven rather than
        # corrupt — and that is as true of an unreadable file as a missing one.
        if not path.exists():
            yield OutputVerdict(
                key, MISSING, "recorded output is missing", path, untrusted=key in declared
            )
            continue
        try:
            actual = sha256_path(path)
        except (OSError, ValueError, RuntimeError) as exc:
            # The same triple the resolve sites catch: this is the call that
            # walks directory trees, so it is the most exposed of the three.
            yield OutputVerdict(
                key,
                MISSING,
                f"recorded output unreadable: {exc}",
                path,
                untrusted=key in declared,
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
    "reroot_under",
    "run_owned_path",
    "verify_output_records",
]
