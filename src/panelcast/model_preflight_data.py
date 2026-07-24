"""Assemble the exact fit inputs `panelcast preflight` audits.

Kept apart from :mod:`panelcast.model_preflight` so the pure check functions
import without pandas / pyarrow / jax and stay trivially unit-testable. This
module reuses the run's data-loading and prior-resolution path so the audit
sees precisely the X / artist_idx / y and PriorConfig the fit would.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PreflightInputs:
    X: np.ndarray
    artist_idx: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    group_idx_by_artist: np.ndarray | None
    priors: object
    likelihood_family: str
    n_obs_is_aggregation_count: bool
    target_bounds: tuple[float, float]


def _resolve_config(dataset: str | None, config_files: list[str] | None):
    from panelcast.pipelines.orchestrator import PipelineConfig

    config_kwargs: dict = {"dataset": dataset}
    if config_files:
        from panelcast.config.loader import load_yaml_config
        from panelcast.config.pipeline_yaml import apply_yaml_overrides

        yaml_data = load_yaml_config(list(config_files))
        # An explicit --dataset must win over a config's `dataset:` key, exactly
        # as it does for `panelcast run`; a config may still set it when --dataset
        # was omitted.
        explicit = {"dataset"} if dataset is not None else set()
        config_kwargs = apply_yaml_overrides(config_kwargs, yaml_data, explicit)
    config = PipelineConfig(**config_kwargs)
    # Mirror the orchestrator's descriptor-owned model-fact resolution (#268)
    # so the audit sees the same resolved family/transform the run would.
    from panelcast.config.descriptor import load_descriptor

    descriptor = load_descriptor(config.dataset)
    if config.likelihood_family is None:
        config.likelihood_family = descriptor.likelihood_family or "studentt"
    if config.target_transform is None:
        config.target_transform = descriptor.target_transform or "offset_logit"
    if config.max_albums is None:
        config.max_albums = descriptor.max_events or 50
    config._validate()
    return config


def assemble_preflight_inputs(
    dataset: str | None = None,
    config_files: list[str] | None = None,
) -> PreflightInputs:
    """Resolve config + priors and load the prepared training arrays.

    Reads the same features/splits the training stage does; raises if they are
    missing (features must be built first) so the CLI can report it cleanly.
    """
    import pyarrow.parquet as pq

    from panelcast.config.descriptor import load_descriptor
    from panelcast.data.split_types import SplitType, resolve_split_dir
    from panelcast.pipelines.train_bayes import (
        build_training_priors,
        load_training_data,
        resolve_entity_group_pooling,
    )

    config = _resolve_config(dataset, config_files)
    descriptor = load_descriptor(config.dataset)

    features_path = Path("data/features/train_features.parquet")
    splits_path = resolve_split_dir(Path("data/splits"), SplitType.WITHIN_ENTITY_TEMPORAL) / (
        "train.parquet"
    )
    if not features_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            "preflight needs prepared data. Missing "
            f"{features_path if not features_path.exists() else splits_path}. "
            "Run the data/splits/features stages first."
        )

    train_columns = set(pq.read_schema(splits_path).names) | set(
        pq.read_schema(features_path).names
    )
    entity_group_pooling = resolve_entity_group_pooling(
        getattr(config, "entity_group_pooling", None), descriptor, train_columns
    )

    model_args, feature_cols, _train_df, _imputation = load_training_data(
        features_path=features_path,
        splits_path=splits_path,
        min_albums_filter=config.min_albums_filter,
        descriptor=descriptor,
        debut_prev_score_source=config.debut_prev_score_source,
        target_transform=config.target_transform,
        logit_offset=config.logit_offset,
        ar_center=config.ar_center,
        entity_group_pooling=entity_group_pooling,
        impute_missing=bool(getattr(config, "impute_missing", False)),
    )

    priors = build_training_priors(
        config,
        target_transform=config.target_transform,
        logit_offset=config.logit_offset,
        ar_center=config.ar_center,
        entity_group_pooling=entity_group_pooling,
        effective_ceiling=model_args.get("effective_ceiling"),
        ar_center_value=model_args["ar_center_value"],
        target_bounds=tuple(descriptor.target_bounds),
    )

    return PreflightInputs(
        X=np.asarray(model_args["X"], dtype=float),
        artist_idx=np.asarray(model_args["artist_idx"]),
        y=np.asarray(model_args["y"], dtype=float),
        feature_names=list(feature_cols),
        group_idx_by_artist=(
            np.asarray(model_args["group_idx_by_artist"])
            if "group_idx_by_artist" in model_args
            else None
        ),
        priors=priors,
        likelihood_family=config.likelihood_family,
        n_obs_is_aggregation_count=descriptor.n_obs_is_aggregation_count,
        target_bounds=tuple(descriptor.target_bounds),
    )
