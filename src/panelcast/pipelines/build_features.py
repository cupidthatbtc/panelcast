"""Feature matrix building pipeline.

Builds combined feature matrices from configured feature blocks for all splits
(train, validation, test) and saves them for reuse in model training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.data.alignment import ROW_ID_COL
from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.features.base import FeatureContext
from panelcast.features.pipeline import FeaturePipeline
from panelcast.features.registry import FeatureSpec, build_default_registry
from panelcast.paths import ArtifactPaths

if TYPE_CHECKING:
    from panelcast.pipelines.stages import StageContext

log = structlog.get_logger()


def _safe_split_stats(features: pd.DataFrame, path: Path) -> dict:
    """Build feature stats dict, handling empty DataFrames (e.g. no val split)."""
    has_data = len(features) > 0
    has_reviews = has_data and "n_reviews" in features.columns
    return {
        "path": str(path),
        "rows": int(features.shape[0]),
        "cols": int(features.shape[1]) if has_data else 0,
        "n_reviews_min": int(features["n_reviews"].min()) if has_reviews else 0,
        "n_reviews_max": int(features["n_reviews"].max()) if has_reviews else 0,
        "n_reviews_median": int(features["n_reviews"].median()) if has_reviews else 0,
    }


def _partition_schema(features: pd.DataFrame) -> dict:
    """Ordered column name/dtype schema with a stable content hash (#295)."""
    columns = [{"name": str(col), "dtype": str(dtype)} for col, dtype in features.dtypes.items()]
    payload = json.dumps(columns, sort_keys=True)
    return {
        "columns": columns,
        "hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _schema_mismatch(reference: dict, other: dict) -> str | None:
    """Human-readable difference between two partition schemas, or None."""
    ref_cols = [c["name"] for c in reference["columns"]]
    oth_cols = [c["name"] for c in other["columns"]]
    missing = [c for c in ref_cols if c not in oth_cols]
    extra = [c for c in oth_cols if c not in ref_cols]
    if missing or extra:
        return f"missing columns {missing}, unexpected columns {extra}"
    if ref_cols != oth_cols:
        return f"columns reordered ({oth_cols} vs {ref_cols})"
    drifted = [
        f"{r['name']} ({o['dtype']} vs {r['dtype']})"
        for r, o in zip(reference["columns"], other["columns"])
        if r["dtype"] != o["dtype"]
    ]
    if drifted:
        return "dtype drift: " + ", ".join(drifted)
    return None


def _basis_curve_specs(descriptor: DatasetDescriptor) -> dict[str, dict[str, Any]]:
    return {
        name: curve.model_dump(mode="json")
        for name, curve in descriptor.basis_curves.items()
    }


def get_feature_blocks(
    enable_genre: bool = True,
    enable_artist: bool = True,
    enable_temporal: bool = True,
    descriptor: DatasetDescriptor | None = None,
    gbm_offset: bool = False,
) -> list:
    """Get feature blocks for a descriptor, filtered by ablation flags.

    The block list and parameters come from ``descriptor.feature_blocks``
    (instantiated through the descriptor-aware registry); the boolean flags
    disable the block groups named in ``descriptor.ablation_groups``.

    Args:
        enable_genre: Keep the descriptor's "genre" ablation group if True.
        enable_artist: Keep the descriptor's "artist" ablation group if True.
        enable_temporal: Keep the descriptor's "temporal" ablation group if True.
        descriptor: Dataset descriptor (None = AOTY defaults).
        gbm_offset: Append the stacked-GBM offset block over the enabled
            blocks' outputs (#86). Default off: block absent, X unchanged.

    Returns:
        List of enabled feature blocks in dependency order.
    """
    descriptor = descriptor or DatasetDescriptor()
    registry = build_default_registry(descriptor)

    disabled: set[str] = set()
    for group, enabled in (
        ("genre", enable_genre),
        ("artist", enable_artist),
        ("temporal", enable_temporal),
    ):
        if not enabled:
            disabled.update(descriptor.ablation_groups.get(group, []))

    specs = [
        FeatureSpec(name=spec.name, params=dict(spec.params))
        for spec in descriptor.feature_blocks
        if spec.name not in disabled
    ]
    if descriptor.basis_curves:
        specs.append(
            FeatureSpec(
                name="basis",
                params={"curves": _basis_curve_specs(descriptor)},
            )
        )
    blocks = registry.build_all(specs)
    if gbm_offset:
        from panelcast.features.gbm_offset import GbmOffsetBlock

        blocks.append(
            GbmOffsetBlock(
                list(blocks),
                target_col=descriptor.target_col,
                entity_col=descriptor.entity_col,
                date_col=descriptor.parsed_date_col,
                event_col=descriptor.event_col,
            )
        )
    return blocks


def get_default_feature_blocks() -> list:
    """Get the default feature blocks for the pipeline.

    Legacy function for backward compatibility. Prefer get_feature_blocks()
    with explicit flags for new code.

    Returns:
        List of all feature blocks in dependency order.
    """
    return get_feature_blocks(
        enable_genre=True,
        enable_artist=True,
        enable_temporal=True,
    )


def _assign_n_reviews(
    features_df: pd.DataFrame,
    n_reviews: pd.Series,
    name: str,
) -> pd.DataFrame:
    """Assign n_reviews to features DataFrame with alignment validation."""
    aligned = n_reviews.reindex(features_df.index)
    null_count = aligned.isna().sum()
    if null_count > 0:
        missing_indices = features_df.index[aligned.isna()].tolist()[:5]
        raise ValueError(
            f"{name}_n_reviews has {null_count} null values after reindexing to "
            f"{name}_features index. First missing indices: {missing_indices}. "
            f"Ensure {name}_n_reviews index matches {name}_features index."
        )
    out = features_df.copy()
    out["n_reviews"] = aligned
    return out


def _attach_row_ids(
    features_df: pd.DataFrame,
    source_df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """Carry the stable row-identity key from the split into the feature matrix.

    Downstream stages join splits with features on ``original_row_id`` instead
    of relying on positional index equality, which cannot detect reordering.
    """
    if ROW_ID_COL not in source_df.columns:
        log.warning(
            "row_id_missing_from_split",
            split=name,
            reason=f"'{ROW_ID_COL}' not in split columns; keyed join unavailable",
        )
        return features_df
    aligned = source_df[ROW_ID_COL].reindex(features_df.index)
    if aligned.isna().any():
        raise ValueError(
            f"{name}: '{ROW_ID_COL}' has null values after aligning split rows "
            "to the feature matrix index. Split and feature rows do not match."
        )
    out = features_df.copy()
    out[ROW_ID_COL] = aligned.astype(np.int64)
    return out


def _transform_with_train_history(
    pipeline: FeaturePipeline,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_ctx: FeatureContext,
    mask_target_score_cols: tuple[str, ...] = ("User_Score", "Critic_Score"),
) -> pd.DataFrame:
    """Transform target split while preserving train-history semantics.

    Target labels are masked before concatenation so history-based blocks
    (e.g., ArtistHistoryBlock) cannot read held-out score labels from the
    target split. This prevents within-split label leakage.
    """
    train_work = train_df.copy()
    target_work = target_df.copy()

    # Prevent feature leakage from held-out score labels.
    for col in mask_target_score_cols:
        if col in target_work.columns:
            target_work[col] = np.nan

    # Renumber to a combined unique RangeIndex so target rows can be recovered
    # unambiguously even when train/target indices overlap or contain duplicates.
    train_work.index = pd.RangeIndex(0, len(train_work))
    target_index = pd.RangeIndex(len(train_work), len(train_work) + len(target_work))
    target_work.index = target_index

    combined = pd.concat([train_work, target_work], axis=0)
    combined_features = pipeline.transform(combined, feature_ctx).data
    if len(combined_features) != len(combined):
        raise ValueError(
            "Feature pipeline changed the row count during transform "
            f"({len(combined)} in, {len(combined_features)} out). Target rows "
            "can no longer be recovered reliably."
        )
    target_features = combined_features.loc[target_index].copy()
    target_features.index = target_df.index
    return target_features


def build_features(ctx: StageContext) -> dict:
    """Build feature matrices for all splits.

    Fits feature pipeline on training data only, then transforms all splits
    to prevent data leakage. Respects feature ablation flags from CLI.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with paths to created feature matrices and metadata.
    """
    descriptor = getattr(ctx, "descriptor", None) or DatasetDescriptor()
    # Mask all target labels in held-out splits before history transforms.
    mask_score_cols = tuple(
        col for col in (descriptor.target_col, descriptor.secondary_target_col) if col is not None
    )

    gbm_offset = bool(getattr(ctx, "gbm_offset", False))

    log.info(
        "feature_pipeline_start",
        seed=ctx.seed,
        enable_genre=ctx.enable_genre,
        enable_artist=ctx.enable_artist,
        enable_temporal=ctx.enable_temporal,
        gbm_offset=gbm_offset,
    )

    # Roots come from ctx.paths but are rebuilt through the module-local Path
    # so test patches keep applying.
    paths = ArtifactPaths.from_ctx(ctx)
    splits_root = Path(paths.splits)
    features_dir = Path(paths.features)
    features_dir.mkdir(parents=True, exist_ok=True)
    split_names = [
        str(SplitType.WITHIN_ENTITY_TEMPORAL.value),
        str(SplitType.ENTITY_DISJOINT.value),
    ]

    # Create feature context
    feature_ctx = FeatureContext(
        config={},  # Using default configs
        random_state=ctx.seed,
    )

    split_manifests: dict[str, dict] = {}
    basis_states: dict[str, dict] = {}
    feature_schemas: dict[str, dict[str, dict]] = {}

    for split_name in split_names:
        # Read from the canonical directory, falling back to a legacy-named
        # directory if only a pre-rename one exists on disk.
        split_dir = resolve_split_dir(splits_root, split_name)
        feature_split_dir = features_dir / split_name
        feature_split_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_parquet(split_dir / "train.parquet")
        val_df = pd.read_parquet(split_dir / "validation.parquet")
        test_df = pd.read_parquet(split_dir / "test.parquet")

        log.info(
            "splits_loaded",
            split=split_name,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
        )

        train_n_reviews = train_df[descriptor.n_obs_col].rename("n_reviews")
        val_n_reviews = val_df[descriptor.n_obs_col].rename("n_reviews")
        test_n_reviews = test_df[descriptor.n_obs_col].rename("n_reviews")

        blocks = get_feature_blocks(
            enable_genre=ctx.enable_genre,
            enable_artist=ctx.enable_artist,
            enable_temporal=ctx.enable_temporal,
            descriptor=descriptor,
            gbm_offset=gbm_offset,
        )
        pipeline = FeaturePipeline(blocks)

        log.info(
            "fitting_features",
            split=split_name,
            blocks=[b.name for b in blocks],
            ablated={
                "genre": not ctx.enable_genre,
                "artist": not ctx.enable_artist,
                "temporal": not ctx.enable_temporal,
            },
        )
        pipeline.fit(train_df, feature_ctx)

        train_output = pipeline.transform(train_df, feature_ctx)
        train_features = train_output.data
        for block_metadata in train_output.metadata["blocks"]:
            if block_metadata.get("name") == "basis":
                basis_states[split_name] = block_metadata["curves"]
        val_features = _transform_with_train_history(
            pipeline, train_df, val_df, feature_ctx, mask_target_score_cols=mask_score_cols
        )
        test_features = _transform_with_train_history(
            pipeline, train_df, test_df, feature_ctx, mask_target_score_cols=mask_score_cols
        )

        train_features = _assign_n_reviews(train_features, train_n_reviews, f"{split_name}_train")
        val_features = _assign_n_reviews(val_features, val_n_reviews, f"{split_name}_val")
        test_features = _assign_n_reviews(test_features, test_n_reviews, f"{split_name}_test")

        train_features = _attach_row_ids(train_features, train_df, f"{split_name}_train")
        val_features = _attach_row_ids(val_features, val_df, f"{split_name}_val")
        test_features = _attach_row_ids(test_features, test_df, f"{split_name}_test")

        # Record and enforce per-partition schemas before anything is written
        # (#295): a train/validation/test drift must fail feature building, not
        # surface as a shape error mid-fit.
        schemas = {
            "train": _partition_schema(train_features),
            "validation": _partition_schema(val_features),
            "test": _partition_schema(test_features),
        }
        for partition in ("validation", "test"):
            mismatch = _schema_mismatch(schemas["train"], schemas[partition])
            if mismatch:
                raise ValueError(
                    f"Feature schema drift in split '{split_name}': {partition} "
                    f"disagrees with train — {mismatch}."
                )
        feature_schemas[split_name] = schemas

        train_path = feature_split_dir / "train_features.parquet"
        val_path = feature_split_dir / "validation_features.parquet"
        test_path = feature_split_dir / "test_features.parquet"

        train_features.to_parquet(train_path, index=True)
        val_features.to_parquet(val_path, index=True)
        test_features.to_parquet(test_path, index=True)

        # Backward compatibility: root feature paths follow primary split.
        if split_name == SplitType.WITHIN_ENTITY_TEMPORAL:
            train_features.to_parquet(features_dir / "train_features.parquet", index=True)
            val_features.to_parquet(features_dir / "validation_features.parquet", index=True)
            test_features.to_parquet(features_dir / "test_features.parquet", index=True)

        split_manifests[split_name] = {
            "train": _safe_split_stats(train_features, train_path),
            "validation": _safe_split_stats(val_features, val_path),
            "test": _safe_split_stats(test_features, test_path),
        }
        gbm_block = next((block for block in blocks if block.name == "gbm_offset"), None)
        if gbm_block is not None:
            split_manifests[split_name]["gbm_oof_folds"] = gbm_block.fold_manifest
            split_manifests[split_name]["gbm_deployment_refit"] = {
                "protocol": "all_training_rows_admissible_before_held_out_prediction",
                "n_rows": len(train_df),
            }

    # Save manifest
    block_names = [
        b.name
        for b in get_feature_blocks(
            enable_genre=ctx.enable_genre,
            enable_artist=ctx.enable_artist,
            enable_temporal=ctx.enable_temporal,
            descriptor=descriptor,
            gbm_offset=gbm_offset,
        )
    ]
    primary_split = str(SplitType.WITHIN_ENTITY_TEMPORAL.value)
    train_schema_hashes = {name: feature_schemas[name]["train"]["hash"] for name in split_names}
    schemas_identical = len(set(train_schema_hashes.values())) == 1
    schema_identity: dict[str, Any] = {
        "identical_across_splits": schemas_identical,
        "train_schema_hashes": train_schema_hashes,
        "canonical_split": primary_split,
    }
    if not schemas_identical:
        schema_identity["reason"] = (
            "feature vocabularies are fit on each split's own training rows, so "
            "splits with different training data can derive different columns; "
            "feature_schemas records each split's actual schema"
        )
    # Legacy compatibility projection: the primary split's train schema minus
    # the row-identity join key. Kept because external tooling reads it; the
    # per-split truth lives in feature_schemas.
    feature_names = [
        c["name"]
        for c in feature_schemas[primary_split]["train"]["columns"]
        if c["name"] != ROW_ID_COL
    ]

    manifest = {
        "seed": ctx.seed,
        "blocks": block_names,
        "feature_ablation": {
            "enable_genre": ctx.enable_genre,
            "enable_artist": ctx.enable_artist,
            "enable_temporal": ctx.enable_temporal,
        },
        "gbm_offset": gbm_offset,
        "feature_names": feature_names,
        "feature_schemas": feature_schemas,
        "feature_schema_identity": schema_identity,
        "n_reviews_included": True,
        "target_label_leakage_prevention": {
            "masked_score_columns": list(mask_score_cols),
            "applies_to_splits": ["validation", "test"],
        },
        "split_features": split_manifests,
        "legacy_primary_split": str(SplitType.WITHIN_ENTITY_TEMPORAL.value),
    }

    if descriptor.basis_curves:
        manifest["basis_curves"] = {
            "specs": _basis_curve_specs(descriptor),
            "fitted_by_split": basis_states,
        }

    manifest_path = features_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("feature_pipeline_complete", manifest_path=str(manifest_path))

    return manifest
