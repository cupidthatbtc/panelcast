"""Artifact root directories for pipeline stages.

Data roots (processed/splits/features) are a deterministic cross-run cache
and stay flat in both layouts; mutable products (models/evaluation/
predictions/reports) can be scoped under a run directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True)
class ArtifactPaths:
    """Root directories every stage writer/consumer resolves paths against."""

    processed: Path
    splits: Path
    features: Path
    models: Path
    evaluation: Path
    predictions: Path
    reports: Path

    @classmethod
    def flat(cls) -> ArtifactPaths:
        """The legacy flat repository layout."""
        return cls(
            processed=Path("data/processed"),
            splits=Path("data/splits"),
            features=Path("data/features"),
            models=Path("models"),
            evaluation=Path("outputs/evaluation"),
            predictions=Path("outputs/predictions"),
            reports=Path("reports"),
        )

    @classmethod
    def for_run(cls, run_dir: Path) -> ArtifactPaths:
        """Run-scoped layout: mutable products live under ``run_dir``."""
        return cls(
            processed=Path("data/processed"),
            splits=Path("data/splits"),
            features=Path("data/features"),
            models=run_dir / "models",
            evaluation=run_dir / "evaluation",
            predictions=run_dir / "predictions",
            reports=run_dir / "reports",
        )

    @classmethod
    def from_ctx(cls, ctx: object) -> ArtifactPaths:
        """Paths carried by a stage context; flat layout when absent.

        The isinstance check keeps bare test contexts (SimpleNamespace,
        MagicMock) on the legacy flat layout.
        """
        paths = getattr(ctx, "paths", None)
        return paths if isinstance(paths, cls) else cls.flat()


class RunPathError(ValueError):
    """A run identifier is malformed or escapes its output root."""


# Reserved by the layout itself: `latest` is the pointer link, `failed` the
# quarantine root. Device names are rejected on every platform so a run id
# minted on Linux stays usable on Windows.
_RESERVED_RUN_IDS = frozenset({"latest", "failed"})
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _run_id_rejection(run_id: str) -> str | None:
    """Why ``run_id`` is not a bare, portable directory name (None if it is)."""
    if not isinstance(run_id, str):
        return "must be a string"
    if not run_id:
        return "must not be empty"
    if len(run_id) > 255:
        return "must be at most 255 characters"
    if "/" in run_id or "\\" in run_id:
        return "must not contain a path separator"
    if ":" in run_id:
        return "must not contain ':' (drive letter or NTFS data stream)"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in run_id):
        return "must not contain control characters"
    if run_id.startswith("."):  # covers "." and ".."
        return "must not start with '.'"
    # Windows silently strips these, so two distinct ids can name one directory.
    if run_id != run_id.strip() or run_id.endswith("."):
        return "must not start or end with whitespace, nor end with '.'"
    if run_id.lower() in _RESERVED_RUN_IDS:
        return "is reserved by the output layout"
    if run_id.split(".", 1)[0].lower() in _WINDOWS_DEVICE_NAMES:
        return "is a reserved Windows device name"
    if PurePosixPath(run_id).is_absolute() or PureWindowsPath(run_id).is_absolute():
        return "must not be an absolute path"
    return None


def validate_run_id(run_id: str, *, field: str = "run_id") -> str:
    """Return ``run_id`` unchanged if it is a bare directory name, else raise."""
    reason = _run_id_rejection(run_id)
    if reason is not None:
        raise RunPathError(
            f"Invalid {field}: {run_id!r} {reason}. Must be a bare directory name "
            "(no path separators, no traversal, not a reserved name)."
        )
    return run_id


def path_is_within(candidate: Path, root: Path) -> bool:
    """Whether ``candidate`` resolves to a strict descendant of ``root``.

    Both sides are resolved, so a symlinked component that leaves the root is
    caught even when the literal join looks contained.
    """
    try:
        return Path(root).resolve() in Path(candidate).resolve().parents
    except OSError:
        return False


def safe_run_dir(
    output_base: Path,
    run_id: str,
    *,
    subdir: str | None = None,
    field: str = "run_id",
) -> Path:
    """The one containment-enforcing resolver for run directories.

    Returns the *unresolved* join so callers keep their own spelling of
    ``output_base``; containment is proven against the resolved pair, which is
    what makes a symlinked run name pointing outside the root fail here rather
    than at the eventual read, move, or delete.
    """
    validate_run_id(run_id, field=field)
    root = Path(output_base)
    candidate = root / subdir / run_id if subdir else root / run_id
    if not path_is_within(candidate, root):
        raise RunPathError(
            f"Invalid {field}: {run_id!r} resolves outside the output root {root}."
        )
    return candidate


def _is_dry_run_dir(run_dir: Path) -> bool:
    """Whether a run dir's manifest marks it as a dry run (no artifacts)."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        return bool(manifest.get("flags", {}).get("dry_run"))
    except (OSError, ValueError, AttributeError):
        return False


def resolve_latest(output_base: Path = Path("outputs")) -> Path | None:
    """Locate the most recent successful run directory.

    Prefers the ``latest.json`` pointer the orchestrator writes on success;
    falls back to the ``latest`` link for outputs written by older checkouts.
    Pointers left behind by older checkouts that targeted dry runs are
    ignored (a dry-run dir holds only a manifest). A pointer or link aiming
    outside ``output_base`` is treated as unusable rather than followed.
    Returns None when nothing usable exists.
    """
    try:
        data = json.loads((output_base / "latest.json").read_text(encoding="utf-8"))
        run_dir = safe_run_dir(output_base, str(data["run_dir"]), field="latest.json run_dir")
        if run_dir.exists() and not _is_dry_run_dir(run_dir):
            return run_dir
    except (OSError, ValueError, KeyError, TypeError):
        pass
    link = output_base / "latest"
    try:
        if link.exists() and path_is_within(link, output_base) and not _is_dry_run_dir(link):
            return link
    except OSError:
        pass
    return None


def resolve_evaluation_dir(output_base: Path = Path("outputs")) -> Path:
    """Latest run's evaluation dir, or the legacy flat location."""
    latest = resolve_latest(output_base)
    return latest / "evaluation" if latest is not None else output_base / "evaluation"


def resolve_reports_dir(output_base: Path = Path("outputs")) -> Path:
    """Latest run's reports dir, or the legacy flat location."""
    latest = resolve_latest(output_base)
    return latest / "reports" if latest is not None else Path("reports")
