"""End-to-end dataset preparation pipeline."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import structlog

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.data.cleaning import (
    CleaningConfig,
    clean_albums,
    filter_for_target_model,
)
from panelcast.data.ingest import LoadMetadata, load_raw_albums
from panelcast.data.lineage import AuditLogger

log = structlog.get_logger()


def _default_raw_dataset_path() -> str:
    """Resolve raw dataset path at runtime from environment when available."""
    return os.environ.get("AOTY_DATASET_PATH", "data/raw/all_albums_full.csv")


@dataclass
class PrepareConfig:
    """Configuration for dataset preparation.

    Fields left as ``None`` resolve from ``descriptor`` in ``__post_init__``,
    so the no-argument construction reproduces AOTY behavior exactly while a
    descriptor-only construction retargets every name and threshold.
    """

    raw_path: str | None = None
    output_dir: str = "data/processed"
    audit_dir: str = "data/audit"
    min_ratings_thresholds: list[int] | None = None
    # Threshold of the primary modeling dataset (must be one of the thresholds
    # above). Replaces the old positional thresholds[1] pick, which silently
    # changed meaning when the list was reordered or shortened.
    primary_min_ratings: int | None = None
    min_critic_reviews: int = 1
    cleaning: CleaningConfig | None = None
    dataset_hash_output: str | None = None
    validate_raw_schema: bool = False
    descriptor: DatasetDescriptor = field(default_factory=DatasetDescriptor)

    def __post_init__(self) -> None:
        if self.raw_path is None:
            self.raw_path = os.environ.get(
                self.descriptor.raw_path_env, self.descriptor.raw_path_default
            )
        if self.min_ratings_thresholds is None:
            self.min_ratings_thresholds = list(self.descriptor.min_obs_thresholds)
        if self.primary_min_ratings is None:
            self.primary_min_ratings = self.descriptor.primary_min_obs
        if self.cleaning is None:
            self.cleaning = CleaningConfig(
                min_year=self.descriptor.min_year,
                descriptor=self.descriptor,
            )
        if self.primary_min_ratings not in self.min_ratings_thresholds:
            raise ValueError(
                f"primary_min_ratings={self.primary_min_ratings} is not one of "
                f"min_ratings_thresholds={self.min_ratings_thresholds}."
            )


@dataclass
class PrepareResult:
    """Result of dataset preparation."""

    load_metadata: LoadMetadata
    datasets_created: dict[str, Path]
    audit_paths: dict[str, Path]
    summary: dict


def save_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> dict[str, Path]:
    """Save dataset in both Parquet and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"

    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    df.to_csv(csv_path, index=False)

    log.info(
        "dataset_saved",
        name=name,
        rows=len(df),
        parquet=str(parquet_path),
        csv=str(csv_path),
    )

    return {"parquet": parquet_path, "csv": csv_path}


def prepare_datasets(config: PrepareConfig | None = None) -> PrepareResult:
    """
    Prepare all cleaned datasets from raw CSV.

    Creates:
    - Primary-target datasets at each observation threshold
    - Secondary-target dataset (when the descriptor defines one)
    - Audit logs with all exclusions

    Args:
        config: Pipeline configuration

    Returns:
        PrepareResult with paths to all created files
    """
    config = config or PrepareConfig()
    descriptor = config.descriptor
    # Resolved in __post_init__; narrow the Optionals for type checkers.
    assert config.raw_path is not None
    assert config.min_ratings_thresholds is not None
    assert config.primary_min_ratings is not None
    output_dir = Path(config.output_dir)
    audit_dir = Path(config.audit_dir)

    # Initialize audit logger
    logger = AuditLogger(output_dir=audit_dir)

    log.info(
        "pipeline_start",
        raw_path=config.raw_path,
        validate_raw_schema=config.validate_raw_schema,
    )

    # Step 1: Load raw data
    raw_df, load_meta = load_raw_albums(
        config.raw_path,
        validate=config.validate_raw_schema,
        descriptor=descriptor,
    )
    log.info(
        "raw_loaded",
        rows=load_meta.row_count,
        hash=load_meta.file_hash[:16] + "...",
    )

    # Step 2: Apply cleaning transformations
    cleaned_df = clean_albums(raw_df, config=config.cleaning, logger=logger)
    log.info("cleaning_complete", rows=len(cleaned_df))

    # Step 3: Generate primary-target datasets at multiple thresholds
    datasets_created: dict[str, Path] = {}
    dataset_rows: dict[str, int] = {}
    primary_df: pd.DataFrame | None = None

    for threshold in config.min_ratings_thresholds:
        # Create a fresh logger section for this threshold
        user_df = filter_for_target_model(
            cleaned_df.copy(),
            descriptor,
            threshold,
            logger=logger,
        )

        # Save dataset
        name = descriptor.processed_name(threshold)
        paths = save_dataset(user_df, output_dir, name)
        datasets_created[name] = paths["parquet"]
        dataset_rows[name] = len(user_df)
        if threshold == config.primary_min_ratings:
            primary_df = user_df

        log.info(
            "user_dataset_created",
            threshold=threshold,
            rows=len(user_df),
            unique_artists=user_df[descriptor.entity_col].nunique(),
        )

    # Step 4: Generate secondary-target dataset (AOTY: critic score)
    if descriptor.secondary_target_col is not None:
        critic_df = filter_for_target_model(
            cleaned_df.copy(),
            descriptor,
            config.min_critic_reviews,
            target="secondary",
            logger=logger,
        )

        name = f"{descriptor.secondary_prefix}_score"
        paths = save_dataset(critic_df, output_dir, name)
        datasets_created[name] = paths["parquet"]
        dataset_rows[name] = len(critic_df)

        log.info(
            "critic_dataset_created",
            rows=len(critic_df),
            unique_artists=critic_df[descriptor.entity_col].nunique(),
        )

    # Step 5: Save full cleaned dataset (before score filtering)
    name = "cleaned_all"
    paths = save_dataset(cleaned_df, output_dir, name)
    datasets_created[name] = paths["parquet"]
    dataset_rows[name] = len(cleaned_df)

    # Step 5b: Save dataset-level statistics for downstream stages.
    # The global_mean_score is computed from the primary dataset BEFORE
    # train/test splitting, so it is stable across splits and avoids
    # data-dependency in debut prev_score.
    primary_key = descriptor.processed_name(config.primary_min_ratings)
    if primary_key in datasets_created and primary_df is not None:
        dataset_stats = {
            "global_mean_score": float(primary_df[descriptor.target_col].mean()),
            "global_std_score": float(primary_df[descriptor.target_col].std()),
            "n_albums": len(primary_df),
            "n_artists": int(primary_df[descriptor.entity_col].nunique()),
            "source_dataset": primary_key,
        }
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2)
        log.info(
            "dataset_stats_saved",
            path=str(stats_path),
            global_mean=dataset_stats["global_mean_score"],
        )

    # Step 6: Save audit logs
    audit_paths = logger.save()

    # Build summary
    summary = {
        "raw_rows": load_meta.row_count,
        "raw_hash": load_meta.file_hash,
        "cleaned_rows": len(cleaned_df),
        "datasets": {
            # Row counts come from the in-memory frames; re-reading every
            # just-written parquet doubled the I/O of the data stage.
            name: {"path": str(path), "rows": dataset_rows[name]}
            for name, path in datasets_created.items()
        },
        "exclusions": logger.get_summary(),
    }

    # Persist a dataset hash artifact for run-level provenance when requested.
    if config.dataset_hash_output is not None:
        hash_path = Path(config.dataset_hash_output)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(f"{load_meta.file_hash}\n", encoding="utf-8")
        log.info("dataset_hash_written", path=str(hash_path))

    log.info("pipeline_complete", datasets=len(datasets_created))

    return PrepareResult(
        load_metadata=load_meta,
        datasets_created=datasets_created,
        audit_paths=audit_paths,
        summary=summary,
    )


def main() -> None:
    """CLI entry point for dataset preparation."""
    result = prepare_datasets()

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nRaw data: {result.load_metadata.row_count:,} rows")
    print(f"File hash: {result.load_metadata.file_hash[:32]}...")
    print("\nDatasets created:")
    for name, path in result.datasets_created.items():
        rows = result.summary["datasets"][name]["rows"]
        print(f"  - {name}: {rows:,} rows")
    print(f"\nAudit log: {result.audit_paths.get('summary')}")
    print(f"Total exclusions: {result.summary['exclusions']['total_exclusions']:,}")
    print("\nExclusions by reason:")
    for reason, count in list(result.summary["exclusions"]["exclusions_by_reason"].items())[:10]:
        print(f"  - {reason}: {count:,}")


if __name__ == "__main__":
    main()
