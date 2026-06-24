"""Canonical split-strategy identifiers and backward-compatible aliases.

panelcast evaluates on two leak-safe split strategies. Their identifiers are
role-based (``entity``), not domain-flavored: the flagship AOTY domain happens
to call entities "artists", but the pipeline is domain-portable, so the split
names must not be.

Older artifacts (manifests, split/feature directories, summary keys) were
written with the AOTY-flavored literals ``within_artist_temporal`` and
``artist_disjoint``. New runs write the canonical names; the resolvers here let
old artifacts still load via an alias map.

- :class:`SplitType` — the two canonical identifiers (a ``StrEnum``, so each
  member is usable directly as a string / path segment).
- :func:`resolve_split_type` — normalize any legacy or canonical literal to a
  :class:`SplitType`.
- :func:`split_dir_name` / :func:`legacy_split_name` — canonical and legacy
  directory segments for a split type.
- :func:`resolve_split_dir` — locate a split's directory on disk, preferring the
  canonical name but falling back to a legacy-named directory if only that
  exists (reading pre-rename artifacts).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

__all__ = [
    "SplitType",
    "LEGACY_SPLIT_ALIASES",
    "resolve_split_type",
    "split_dir_name",
    "legacy_split_name",
    "resolve_split_dir",
]


class SplitType(StrEnum):
    """Canonical, role-based split-strategy identifiers."""

    WITHIN_ENTITY_TEMPORAL = "within_entity_temporal"
    ENTITY_DISJOINT = "entity_disjoint"


# AOTY-flavored literals written by pre-rename runs -> canonical split type.
LEGACY_SPLIT_ALIASES: dict[str, SplitType] = {
    "within_artist_temporal": SplitType.WITHIN_ENTITY_TEMPORAL,
    "artist_disjoint": SplitType.ENTITY_DISJOINT,
}

# Canonical -> legacy directory segment (for on-disk fallback reads).
_CANONICAL_TO_LEGACY: dict[SplitType, str] = {
    canonical: legacy for legacy, canonical in LEGACY_SPLIT_ALIASES.items()
}


def resolve_split_type(name: str | SplitType) -> SplitType:
    """Normalize a split identifier (legacy or canonical) to a ``SplitType``.

    Raises:
        ValueError: if ``name`` is neither a canonical value nor a known legacy
            alias.
    """
    if isinstance(name, SplitType):
        return name
    if name in LEGACY_SPLIT_ALIASES:
        return LEGACY_SPLIT_ALIASES[name]
    return SplitType(name)


def split_dir_name(split_type: str | SplitType) -> str:
    """Return the canonical directory segment for a split type."""
    return str(resolve_split_type(split_type).value)


def legacy_split_name(split_type: str | SplitType) -> str | None:
    """Return the legacy directory segment for a split type, if one exists."""
    return _CANONICAL_TO_LEGACY.get(resolve_split_type(split_type))


def resolve_split_dir(base: Path | str, split_type: str | SplitType) -> Path:
    """Locate a split's directory under ``base``.

    Prefers the canonical directory name. If the canonical directory does not
    exist but a legacy-named directory does, returns the legacy path so
    pre-rename artifacts still load. When neither exists, returns the canonical
    path (the natural target for a fresh write).
    """
    base = Path(base)
    canonical = base / split_dir_name(split_type)
    if canonical.exists():
        return canonical
    legacy = legacy_split_name(split_type)
    if legacy is not None:
        legacy_path = base / legacy
        if legacy_path.exists():
            return legacy_path
    return canonical
