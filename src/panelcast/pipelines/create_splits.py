"""End-to-end pipeline for creating train/validation/test splits."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from panelcast.data.manifests import (
    SplitManifest,
    SplitStats,
    create_split_assignments,
    save_manifest,
)
from panelcast.data.split import (
    assert_no_artist_overlap,
    entity_disjoint_split,
    validate_temporal_split,
    within_entity_temporal_split,
)
from panelcast.data.split_types import SplitType, split_dir_name
from panelcast.utils.hashing import hash_dataframe
from panelcast.utils.logging import setup_pipeline_logging


@dataclass
class SplitConfig:
    """Configuration for split pipeline."""

    min_ratings: int = 10
    output_dir: Path = Path("data/splits")
    version: str = "v1"
    random_state: int = 42

    # Within-artist temporal parameters
    test_albums: int = 1
    val_albums: int = 0
    min_train_albums: int = 1
    # Rolling-origin backtest offset (0 = the standard split)
    origin_offset: int = 0

    # Artist-disjoint parameters
    disjoint_test_size: float = 0.15
    disjoint_val_size: float = 0.15

    # Dataset identity columns (from the descriptor; defaults = AOTY)
    entity_col: str = "Artist"
    date_col: str = "Release_Date_Parsed"
    event_col: str = "Album"

    # Computed source path based on min_ratings (set in __post_init__)
    source_path: Path | None = None

    def __post_init__(self):
        """Compute source_path from min_ratings if not provided."""
        if self.source_path is None:
            self.source_path = Path(
                f"data/processed/user_score_minratings_{self.min_ratings}.parquet"
            )


@dataclass
class SplitResult:
    """Result of split pipeline execution."""

    source_path: Path
    temporal_manifest_path: Path
    disjoint_manifest_path: Path
    temporal_splits: dict[str, Path]
    disjoint_splits: dict[str, Path]
    summary: dict[str, Any]


def save_split_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet with snappy compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def _hash_split(df: pd.DataFrame) -> str:
    """Hash a split as persisted: parquet is written with index=False, so the
    filtered non-contiguous index must not enter the digest or reloaded
    parquets could never match the manifest."""
    return hash_dataframe(df.reset_index(drop=True))


def create_splits(config: SplitConfig | None = None) -> SplitResult:
    """
    Create train/validation/test splits from cleaned dataset.

    Produces:
    - Within-artist temporal splits (primary evaluation)
    - Artist-disjoint splits (cold-start robustness)
    - Full manifests with per-row assignment reasoning
    - Pipeline summary

    Args:
        config: SplitConfig with paths and parameters

    Returns:
        SplitResult with paths and summary
    """
    if config is None:
        config = SplitConfig()

    log = structlog.get_logger()
    log.info(
        "split_pipeline_start",
        source=str(config.source_path),
        min_ratings=config.min_ratings,
    )

    # Load source dataset
    source_df = pd.read_parquet(config.source_path)
    source_hash = hash_dataframe(source_df)
    log.info(
        "source_loaded",
        rows=len(source_df),
        artists=source_df[config.entity_col].nunique(),
        hash=source_hash[:16],
    )

    # Duplicate (entity, event) keys can straddle the temporal boundary: one
    # copy of the latest event lands in test while its twin stays in train,
    # and no split validator can see it. Real AOTY data contains legitimate
    # title collisions (re-recordings, same-titled albums), so warn loudly
    # and record the count instead of dropping rows.
    n_duplicate_key_rows = 0
    if config.event_col in source_df.columns:
        dup_mask = source_df.duplicated(subset=[config.entity_col, config.event_col], keep=False)
        n_duplicate_key_rows = int(dup_mask.sum())
    if n_duplicate_key_rows:
        examples = (
            source_df.loc[dup_mask, [config.entity_col, config.event_col]]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .values.tolist()
        )
        log.warning(
            "duplicate_entity_event_keys",
            n_rows=n_duplicate_key_rows,
            examples=examples,
            risk="duplicate keys may straddle the temporal split boundary (leakage)",
            action="rows kept; verify duplicates are distinct events, not repeats",
        )

    results: dict[str, dict] = {"temporal": {}, "disjoint": {}}

    # ===== WITHIN-ENTITY TEMPORAL SPLIT =====
    temporal_dir = config.output_dir / split_dir_name(SplitType.WITHIN_ENTITY_TEMPORAL)
    temporal_dir.mkdir(parents=True, exist_ok=True)

    train_t, val_t, test_t = within_entity_temporal_split(
        source_df,
        entity_col=config.entity_col,
        date_col=config.date_col,
        test_albums=config.test_albums,
        val_albums=config.val_albums,
        min_train_albums=config.min_train_albums,
        event_col=config.event_col,
        origin_offset=config.origin_offset,
    )

    # Validate temporal ordering
    validate_temporal_split(
        train_t, val_t, test_t, entity_col=config.entity_col, date_col=config.date_col
    )
    log.info("temporal_split_validated")

    # Save parquet files
    temporal_paths = {
        "train": temporal_dir / "train.parquet",
        "validation": temporal_dir / "validation.parquet",
        "test": temporal_dir / "test.parquet",
    }
    save_split_parquet(train_t, temporal_paths["train"])
    save_split_parquet(val_t, temporal_paths["validation"])
    save_split_parquet(test_t, temporal_paths["test"])

    # Compute hashes
    train_t_hash = _hash_split(train_t)
    val_t_hash = _hash_split(val_t)
    test_t_hash = _hash_split(test_t)

    # Combined content hash
    combined_t = hashlib.sha256((train_t_hash + val_t_hash + test_t_hash).encode()).hexdigest()

    # Create manifest
    temporal_manifest = SplitManifest(
        version=config.version,
        created_at=datetime.now(UTC).isoformat(),
        split_type=str(SplitType.WITHIN_ENTITY_TEMPORAL.value),
        parameters={
            "test_albums": config.test_albums,
            "val_albums": config.val_albums,
            "min_train_albums": config.min_train_albums,
            "origin_offset": config.origin_offset,
        },
        source_dataset={
            "path": str(config.source_path),
            "sha256": source_hash,
            "row_count": len(source_df),
            "unique_artists": source_df[config.entity_col].nunique(),
        },
        splits={
            "train": SplitStats(
                row_count=len(train_t),
                unique_artists=train_t[config.entity_col].nunique(),
                sha256=train_t_hash,
            ),
            "validation": SplitStats(
                row_count=len(val_t),
                unique_artists=val_t[config.entity_col].nunique(),
                sha256=val_t_hash,
            ),
            "test": SplitStats(
                row_count=len(test_t),
                unique_artists=test_t[config.entity_col].nunique(),
                sha256=test_t_hash,
            ),
        },
        assignments=create_split_assignments(
            train_t, val_t, test_t, SplitType.WITHIN_ENTITY_TEMPORAL, entity_col=config.entity_col
        ),
        content_hash=combined_t,
    )
    temporal_manifest_path = save_manifest(temporal_manifest, temporal_dir)

    log.info(
        "temporal_split_complete",
        train=len(train_t),
        val=len(val_t),
        test=len(test_t),
        artists_included=train_t[config.entity_col].nunique(),
        manifest=str(temporal_manifest_path.name),
    )

    results["temporal"] = {
        "train": len(train_t),
        "validation": len(val_t),
        "test": len(test_t),
        "artists": train_t[config.entity_col].nunique(),
    }

    # ===== ENTITY-DISJOINT SPLIT =====
    disjoint_dir = config.output_dir / split_dir_name(SplitType.ENTITY_DISJOINT)
    disjoint_dir.mkdir(parents=True, exist_ok=True)

    train_d, val_d, test_d = entity_disjoint_split(
        source_df,
        entity_col=config.entity_col,
        test_size=config.disjoint_test_size,
        val_size=config.disjoint_val_size,
        random_state=config.random_state,
    )

    # Validate no overlap
    assert_no_artist_overlap(train_d, val_d, test_d, entity_col=config.entity_col)
    log.info("disjoint_split_validated", overlap="none")

    # Save parquet files
    disjoint_paths = {
        "train": disjoint_dir / "train.parquet",
        "validation": disjoint_dir / "validation.parquet",
        "test": disjoint_dir / "test.parquet",
    }
    save_split_parquet(train_d, disjoint_paths["train"])
    save_split_parquet(val_d, disjoint_paths["validation"])
    save_split_parquet(test_d, disjoint_paths["test"])

    # Compute hashes
    train_d_hash = _hash_split(train_d)
    val_d_hash = _hash_split(val_d)
    test_d_hash = _hash_split(test_d)
    combined_d = hashlib.sha256((train_d_hash + val_d_hash + test_d_hash).encode()).hexdigest()

    # Create manifest
    disjoint_manifest = SplitManifest(
        version=config.version,
        created_at=datetime.now(UTC).isoformat(),
        split_type=str(SplitType.ENTITY_DISJOINT.value),
        parameters={
            "test_size": config.disjoint_test_size,
            "val_size": config.disjoint_val_size,
            "random_state": config.random_state,
        },
        source_dataset={
            "path": str(config.source_path),
            "sha256": source_hash,
            "row_count": len(source_df),
            "unique_artists": source_df[config.entity_col].nunique(),
        },
        splits={
            "train": SplitStats(
                row_count=len(train_d),
                unique_artists=train_d[config.entity_col].nunique(),
                sha256=train_d_hash,
            ),
            "validation": SplitStats(
                row_count=len(val_d),
                unique_artists=val_d[config.entity_col].nunique(),
                sha256=val_d_hash,
            ),
            "test": SplitStats(
                row_count=len(test_d),
                unique_artists=test_d[config.entity_col].nunique(),
                sha256=test_d_hash,
            ),
        },
        assignments=create_split_assignments(
            train_d, val_d, test_d, SplitType.ENTITY_DISJOINT, entity_col=config.entity_col
        ),
        content_hash=combined_d,
    )
    disjoint_manifest_path = save_manifest(disjoint_manifest, disjoint_dir)

    log.info(
        "disjoint_split_complete",
        train=len(train_d),
        val=len(val_d),
        test=len(test_d),
        train_artists=train_d[config.entity_col].nunique(),
        val_artists=val_d[config.entity_col].nunique(),
        test_artists=test_d[config.entity_col].nunique(),
        manifest=str(disjoint_manifest_path.name),
    )

    results["disjoint"] = {
        "train": len(train_d),
        "validation": len(val_d),
        "test": len(test_d),
        "train_artists": train_d[config.entity_col].nunique(),
        "val_artists": val_d[config.entity_col].nunique(),
        "test_artists": test_d[config.entity_col].nunique(),
    }

    # ===== PIPELINE SUMMARY =====
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "source": {
            "path": str(config.source_path),
            "rows": len(source_df),
            "artists": source_df[config.entity_col].nunique(),
            "sha256": source_hash,
            "duplicate_key_rows": n_duplicate_key_rows,
        },
        "within_entity_temporal": {
            "train_rows": results["temporal"]["train"],
            "val_rows": results["temporal"]["validation"],
            "test_rows": results["temporal"]["test"],
            "artists_included": results["temporal"]["artists"],
            "artists_excluded": source_df[config.entity_col].nunique()
            - results["temporal"]["artists"],
            "manifest": str(temporal_manifest_path),
        },
        "entity_disjoint": {
            "train_rows": results["disjoint"]["train"],
            "val_rows": results["disjoint"]["validation"],
            "test_rows": results["disjoint"]["test"],
            "train_artists": results["disjoint"]["train_artists"],
            "val_artists": results["disjoint"]["val_artists"],
            "test_artists": results["disjoint"]["test_artists"],
            "manifest": str(disjoint_manifest_path),
        },
    }

    # Save summary
    summary_path = config.output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("pipeline_complete", summary_path=str(summary_path))

    return SplitResult(
        source_path=config.source_path,
        temporal_manifest_path=temporal_manifest_path,
        disjoint_manifest_path=disjoint_manifest_path,
        temporal_splits=temporal_paths,
        disjoint_splits=disjoint_paths,
        summary=summary,
    )


def main() -> None:
    """CLI entry point for split pipeline."""
    # Logging configuration is owned by utils.logging.setup_pipeline_logging.
    setup_pipeline_logging()

    result = create_splits()

    # Print formatted summary
    print("\n" + "=" * 60)
    print("SPLIT PIPELINE COMPLETE")
    print("=" * 60)

    s = result.summary
    print(f"\nSource: {s['source']['path']}")
    print(f"  Rows: {s['source']['rows']:,}")
    print(f"  Artists: {s['source']['artists']:,}")

    print("\nWithin-Entity Temporal Split:")
    print(f"  Train:      {s['within_entity_temporal']['train_rows']:,} rows")
    print(f"  Validation: {s['within_entity_temporal']['val_rows']:,} rows")
    print(f"  Test:       {s['within_entity_temporal']['test_rows']:,} rows")
    print(f"  Entities included: {s['within_entity_temporal']['artists_included']:,}")
    excl = s["within_entity_temporal"]["artists_excluded"]
    print(f"  Entities excluded: {excl:,} (insufficient events)")

    print("\nEntity-Disjoint Split:")
    ad = s["entity_disjoint"]
    print(f"  Train:      {ad['train_rows']:,} rows ({ad['train_artists']:,} artists)")
    print(f"  Validation: {ad['val_rows']:,} rows ({ad['val_artists']:,} artists)")
    print(f"  Test:       {ad['test_rows']:,} rows ({ad['test_artists']:,} artists)")

    print(f"\nOutput directory: {result.temporal_splits['train'].parent.parent}")
    print("=" * 60)


if __name__ == "__main__":
    main()
