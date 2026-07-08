"""Split manifest schema and I/O for reproducibility."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from panelcast.data.split_types import SplitType, resolve_split_type


class SourceDataset(TypedDict):
    """The source-dataset provenance block of a split manifest."""

    path: str
    sha256: str
    row_count: int
    unique_artists: int


class SplitParameters(TypedDict, total=False):
    """Split parameters. The keys present depend on the split type.

    within-entity temporal: ``test_albums`` / ``val_albums`` / ``min_train_albums``
    / ``origin_offset``.
    entity-disjoint: ``test_size`` / ``val_size`` / ``random_state``.
    """

    test_albums: int
    val_albums: int
    min_train_albums: int
    origin_offset: int
    test_size: float
    val_size: float
    random_state: int


@dataclass
class SplitAssignment:
    """Per-row split assignment with reasoning."""

    original_row_id: int
    split: str  # "train", "validation", or "test"
    reason: str  # e.g., "last_album_for_artist", "artist_in_test_group"


@dataclass
class SplitStats:
    """Statistics for a single split."""

    row_count: int
    unique_artists: int
    sha256: str


@dataclass
class SplitManifest:
    """
    Complete manifest for a split operation.

    Records all metadata needed to reproduce and audit the split.
    """

    version: str
    created_at: str
    split_type: str  # SplitType value; legacy literals are resolved on load
    parameters: SplitParameters
    source_dataset: SourceDataset
    splits: dict[str, SplitStats]  # train, validation, test stats
    assignments: list[SplitAssignment] = field(default_factory=list)
    content_hash: str = ""  # Combined hash of all splits

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Uses dataclasses.asdict() which recursively converts nested
        dataclasses (SplitStats, SplitAssignment) to dicts automatically.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitManifest":
        """Create from dictionary.

        ``split_type`` is normalized through the legacy alias map, so a manifest
        written with the old AOTY-flavored literals (``within_artist_temporal``
        / ``artist_disjoint``) loads with the canonical role-based value.
        """
        d = dict(d)
        if "split_type" in d:
            d["split_type"] = str(resolve_split_type(d["split_type"]).value)
        d["splits"] = {k: SplitStats(**v) for k, v in d["splits"].items()}
        d["assignments"] = [SplitAssignment(**a) for a in d.get("assignments", [])]
        return cls(**d)


def generate_manifest_filename(version: str, content_hash: str) -> str:
    """
    Generate manifest filename with version, timestamp, and hash.

    Format: split_{version}_{timestamp}_{hash_prefix}.json
    Example: split_v1_20260118_143052_abc123de.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_prefix = content_hash[:8]
    return f"split_{version}_{timestamp}_{hash_prefix}.json"


def save_manifest(manifest: SplitManifest, output_dir: Path) -> Path:
    """
    Save manifest to JSON file.

    Args:
        manifest: SplitManifest to save
        output_dir: Directory to save manifest in

    Returns:
        Path to saved manifest file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = manifest.to_dict()

    # Keep an immutable, content-addressed artifact for audit history.
    versioned_filename = generate_manifest_filename(manifest.version, manifest.content_hash)
    versioned_path = output_dir / versioned_filename
    with open(versioned_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # Also publish a stable path for downstream tooling and docs.
    canonical_path = output_dir / "manifest.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return canonical_path


def load_manifest(filepath: Path) -> SplitManifest:
    """
    Load manifest from JSON file.

    Args:
        filepath: Path to manifest JSON file

    Returns:
        SplitManifest object
    """
    with open(filepath, encoding="utf-8") as f:
        d = json.load(f)
    return SplitManifest.from_dict(d)


def create_split_assignments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str | SplitType,
    entity_col: str = "Artist",
) -> list[SplitAssignment]:
    """
    Create per-row split assignments with reasoning.

    Args:
        train_df, val_df, test_df: Split DataFrames
        split_type: SplitType (or a legacy literal, resolved via alias)
        entity_col: Column name for the entity

    Returns:
        List of SplitAssignment objects
    """

    # Build from plain arrays instead of iterrows (which materializes a
    # Series per row and dominated runtime on full-dataset splits).
    def _bulk(df: pd.DataFrame, split: str, reason_prefix: str | None, reason: str | None):
        row_ids = df["original_row_id"].astype(int).tolist()
        if reason_prefix is not None:
            entities = df[entity_col].astype(str).str[:50].tolist()
            return [
                SplitAssignment(
                    original_row_id=rid,
                    split=split,
                    reason=f"{reason_prefix}{entity}",
                )
                for rid, entity in zip(row_ids, entities, strict=True)
            ]
        return [
            SplitAssignment(original_row_id=rid, split=split, reason=reason or "")
            for rid in row_ids
        ]

    assignments: list[SplitAssignment] = []
    if resolve_split_type(split_type) is SplitType.WITHIN_ENTITY_TEMPORAL:
        # For temporal splits, reason includes event position
        assignments.extend(_bulk(test_df, "test", "last_album_for_", None))
        assignments.extend(_bulk(val_df, "validation", "second_last_album_for_", None))
        assignments.extend(_bulk(train_df, "train", "earlier_album_for_", None))
    else:  # entity_disjoint
        assignments.extend(_bulk(test_df, "test", None, "artist_in_test_group"))
        assignments.extend(_bulk(val_df, "validation", None, "artist_in_validation_group"))
        assignments.extend(_bulk(train_df, "train", None, "artist_in_train_group"))

    return assignments
