"""YAML -> PipelineConfig mapping layer.

``panelcast run --config file.yaml`` loads one or more YAML files
(deep-merged in order by :func:`panelcast.config.loader.load_yaml_config`)
whose top-level keys mirror :class:`PipelineConfig` field names — see
``configs/publication.yaml`` for the canonical example.

Precedence: built-in defaults < YAML files (later files win) < options given
explicitly on the command line. Keys with no mapping produce one structured
warning and are otherwise ignored.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

VALID_CHAIN_METHODS = ("sequential", "vectorized", "parallel", "auto")


def _passthrough(value: Any) -> Any:
    return value


def _as_stage_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"stages must be a list of stage names or a comma string, got {value!r}.")


def _as_calibration_levels(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"calibration_intervals must be a list of levels, got {value!r}.")
    # Same normalization the CLI applies: unique, sorted.
    return tuple(sorted({float(v) for v in value}))


def _as_chain_method(value: Any) -> str:
    normalized = str(value).lower()
    if normalized not in VALID_CHAIN_METHODS:
        raise ValueError(
            f"Invalid chain_method '{value}'. Must be one of: {', '.join(VALID_CHAIN_METHODS)}"
        )
    return normalized


@dataclass(frozen=True)
class YamlKeySpec:
    """One supported YAML key.

    Attributes:
        config_field: PipelineConfig field the value lands on.
        cli_param: Name of the ``run()`` parameter that guards CLI precedence
            (None for fields with no CLI flag — YAML always applies).
        transform: Normalization applied to the YAML value.
    """

    config_field: str
    cli_param: str | None = None
    transform: Callable[[Any], Any] = field(default=_passthrough)


def _spec(config_field: str, cli_param: str | None = None, transform=_passthrough) -> YamlKeySpec:
    return YamlKeySpec(config_field=config_field, cli_param=cli_param, transform=transform)


# Explicit mapping table: YAML key -> PipelineConfig field. YAML keys equal
# field names except where noted (the CLI param differs for a few flags).
PIPELINE_YAML_MAPPING: dict[str, YamlKeySpec] = {
    # Run control
    "seed": _spec("seed", "seed"),
    "skip_existing": _spec("skip_existing", "skip_existing"),
    "stages": _spec("stages", "stages", _as_stage_list),
    "dry_run": _spec("dry_run", "dry_run"),
    "strict": _spec("strict", "strict"),
    "enforce_lockfile": _spec("enforce_lockfile", "allow_unlocked_env"),
    "verbose": _spec("verbose", "verbose"),
    "progress_bar": _spec("progress_bar", "no_progress"),
    "max_albums": _spec("max_albums", "max_albums"),
    # MCMC configuration
    "num_chains": _spec("num_chains", "num_chains"),
    "num_samples": _spec("num_samples", "num_samples"),
    "num_warmup": _spec("num_warmup", "num_warmup"),
    "target_accept": _spec("target_accept", "target_accept"),
    "max_tree_depth": _spec("max_tree_depth", "max_tree_depth"),
    "chain_method": _spec("chain_method", "chain_method", _as_chain_method),
    "checkpoint_every": _spec("checkpoint_every_draws", "checkpoint_every"),
    # Warmup-transfer seams (no CLI flags; the select runner writes them per arm).
    "warmup_export_path": _spec("warmup_export_path"),
    "warmup_import_path": _spec("warmup_import_path"),
    # Caller-supplied run-dir name (#167; the select runner writes it per arm).
    "run_id": _spec("run_id"),
    # Convergence thresholds
    "rhat_threshold": _spec("rhat_threshold", "rhat_threshold"),
    "ess_threshold": _spec("ess_threshold", "ess_threshold"),
    "allow_divergences": _spec("allow_divergences", "allow_divergences"),
    # Data filtering
    "min_ratings": _spec("min_ratings", "min_ratings"),
    "min_albums_filter": _spec("min_albums_filter", "min_albums"),
    # Feature ablation
    "enable_genre": _spec("enable_genre", "enable_genre"),
    "enable_artist": _spec("enable_artist", "enable_artist"),
    "enable_temporal": _spec("enable_temporal", "enable_temporal"),
    # Heteroscedastic noise
    "n_exponent": _spec("n_exponent", "n_exponent"),
    "learn_n_exponent": _spec("learn_n_exponent", "learn_n_exponent"),
    "n_exponent_alpha": _spec("n_exponent_alpha", "n_exponent_alpha"),
    "n_exponent_beta": _spec("n_exponent_beta", "n_exponent_beta"),
    "n_exponent_prior": _spec("n_exponent_prior", "n_exponent_prior"),
    # Likelihood / model gates
    "likelihood_df": _spec("likelihood_df", "likelihood_df"),
    "likelihood_family": _spec("likelihood_family", "likelihood_family"),
    "discretize_observation": _spec("discretize_observation", "discretize_observation"),
    "debut_prev_score_source": _spec("debut_prev_score_source", "debut_prev_score_source"),
    "target_transform": _spec("target_transform", "target_transform"),
    "logit_offset": _spec("logit_offset", None),
    "ar_center": _spec("ar_center", "ar_center"),
    "latent_process": _spec("latent_process", "latent_process"),
    # sigma_obs prior family + entity-level overdispersion gate (no CLI flags;
    # configured per-domain via run_config.yaml, like logit_offset).
    "sigma_obs_prior_type": _spec("sigma_obs_prior_type", None),
    # sigma_artist prior family, artist-effect parameterization, and NUTS init
    # strategy (no CLI flags; configured per-domain via run_config.yaml).
    "sigma_artist_prior_type": _spec("sigma_artist_prior_type", None),
    "artist_effect_param": _spec("artist_effect_param", None),
    "init_strategy": _spec("init_strategy", None),
    # Covariate-block prior gate (#155; no CLI flag).
    "beta_prior_type": _spec("beta_prior_type", None),
    "hs_global_scale": _spec("hs_global_scale", None),
    "heteroscedastic_entity_obs": _spec("heteroscedastic_entity_obs", None),
    "tau_entity_scale": _spec("tau_entity_scale", None),
    # model-v2 gates (no CLI flags; configured per-domain via run_config.yaml).
    "errors_in_variables": _spec("errors_in_variables", None),
    "propagate_rw_horizon": _spec("propagate_rw_horizon", None),
    # Genre/group pooling gate (#41; no CLI flag).
    "entity_group_pooling": _spec("entity_group_pooling", None),
    # Stacked-GBM offset feature block (#86; no CLI flag).
    "gbm_offset": _spec("gbm_offset", None),
    # Train-median imputation gate (#158; no CLI flag).
    "impute_missing": _spec("impute_missing", None),
    "exclude_rw_raw_from_collection": _spec(
        "exclude_rw_raw_from_collection", "exclude_rw_raw_from_collection"
    ),
    # Split configuration
    "val_albums": _spec("val_albums", "val_albums"),
    "origin_offset": _spec("origin_offset", "origin_offset"),
    # Conformal wrapper gate (#156; no CLI flag).
    "conformal_calibration": _spec("conformal_calibration", None),
    # Multi-step rollout evaluation depth (#157; no CLI flag).
    "eval_horizon": _spec("eval_horizon", None),
    "min_train_albums": _spec("min_train_albums", "min_train_albums"),
    # Evaluation configuration
    "calibration_intervals": _spec(
        "calibration_intervals", "calibration_intervals", _as_calibration_levels
    ),
    "coverage_tolerance": _spec("coverage_tolerance", "coverage_tolerance"),
    "prediction_interval": _spec("prediction_interval", "prediction_interval"),
    "evaluate_secondary_split": _spec("evaluate_secondary_split", "secondary_split"),
    # Prediction batching (no CLI flags)
    "predictive_batch_size": _spec("predictive_batch_size", None),
    "predict_artist_batch_size": _spec("predict_artist_batch_size", None),
    # Dataset descriptor reference
    "dataset": _spec("dataset", "dataset"),
}


def apply_yaml_overrides(
    kwargs: dict[str, Any],
    yaml_data: dict[str, Any],
    explicit_cli_params: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Overlay YAML config values onto PipelineConfig kwargs.

    Args:
        kwargs: PipelineConfig keyword arguments built from CLI values.
        yaml_data: Deep-merged YAML mapping (from ``load_yaml_config``).
        explicit_cli_params: ``run()`` parameter names the user set explicitly
            on the command line; those keep their CLI values.

    Returns:
        New kwargs dict with YAML overrides applied. Unmapped YAML keys are
        ignored with a single structured warning.

    Raises:
        ValueError: If a mapped value fails its normalization.
    """
    out = dict(kwargs)
    unmapped: list[str] = []
    for key, value in yaml_data.items():
        spec = PIPELINE_YAML_MAPPING.get(key)
        if spec is None:
            unmapped.append(key)
            continue
        if spec.cli_param is not None and spec.cli_param in explicit_cli_params:
            continue  # explicit CLI option wins over YAML
        out[spec.config_field] = spec.transform(value)
    if unmapped:
        structlog.get_logger().warning(
            "yaml_config_keys_ignored",
            keys=sorted(unmapped),
            hint=(
                "These top-level keys have no PipelineConfig mapping. Dataset/"
                "domain settings belong in a dataset descriptor "
                "(configs/datasets/*.yaml via --dataset), not the pipeline config."
            ),
        )
    return out


def dump_resolved_config(config: Any) -> str:
    """The fully-resolved PipelineConfig as YAML that round-trips losslessly.

    Emits every mapped YAML key from the post-layering config object (preset +
    --config overlays + CLI wins + descriptor-resolved values collapsed), so a
    run can be re-executed from its run directory alone. Round-trip identity
    (config -> yaml -> config) is test-pinned, which permanently prevents new
    config knobs from escaping provenance.
    """
    import yaml  # type: ignore[import-untyped]

    payload: dict[str, Any] = {}
    for yaml_key, spec in PIPELINE_YAML_MAPPING.items():
        value = getattr(config, spec.config_field, None)
        if value is None:
            # None is "unset" for every mapped knob (stages=all, dataset=AOTY,
            # checkpointing off); omitting round-trips to the same default and
            # keeps transforms that reject None out of the load path.
            continue
        if isinstance(value, tuple):
            value = list(value)
        payload[yaml_key] = value
    return yaml.safe_dump(payload, sort_keys=True)


def load_resolved_config(path: Any) -> dict[str, Any]:
    """PipelineConfig kwargs from a resolved_config.yaml (inverse of dump)."""
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    yaml_data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return apply_yaml_overrides({}, yaml_data)
