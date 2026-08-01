"""Assemble the :class:`StageContext` every stage runs against.

One place where a config knob becomes the value a stage sees, so adding a gate
means touching the dataclass and this mapping rather than hunting the lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from panelcast.paths import ArtifactPaths
from panelcast.pipelines.manifest import RunManifest
from panelcast.pipelines.pipeline_config import PipelineConfig
from panelcast.pipelines.stages import StageContext


def build_stage_context(
    config: PipelineConfig,
    descriptor: Any,
    *,
    run_dir: Path,
    paths: ArtifactPaths,
    manifest: RunManifest | None,
    max_events: int,
    min_ratings: int,
    likelihood_family: str,
    target_transform: str,
) -> StageContext:
    """Create StageContext for stage execution.

    Returns:
        StageContext with current configuration.
    """
    return StageContext(
        run_dir=run_dir,
        paths=paths,
        seed=config.seed,
        strict=config.strict,
        verbose=config.verbose,
        progress_bar=config.progress_bar,
        manifest=manifest,
        max_events=max_events,
        # MCMC configuration
        num_chains=config.num_chains,
        num_samples=config.num_samples,
        num_warmup=config.num_warmup,
        target_accept=config.target_accept,
        max_tree_depth=config.max_tree_depth,
        init_strategy=config.init_strategy,
        chain_method=config.chain_method,
        checkpoint_every_draws=config.checkpoint_every_draws,
        caged_chain_retries=config.caged_chain_retries,
        caged_chain_tree_depth_fraction=config.caged_chain_tree_depth_fraction,
        caged_chain_boundary_sigma=config.caged_chain_boundary_sigma,
        caged_chain_consensus_ratio=config.caged_chain_consensus_ratio,
        # Convergence thresholds
        rhat_threshold=config.rhat_threshold,
        ess_threshold=config.ess_threshold,
        allow_divergences=config.allow_divergences,
        # Data filtering
        min_ratings=min_ratings,
        min_events_filter=config.min_events_filter,
        # Feature flags
        enable_genre=config.enable_genre,
        enable_artist=config.enable_artist,
        enable_temporal=config.enable_temporal,
        # Heteroscedastic noise configuration
        n_exponent=config.n_exponent,
        learn_n_exponent=config.learn_n_exponent,
        n_exponent_alpha=config.n_exponent_alpha,
        n_exponent_beta=config.n_exponent_beta,
        n_exponent_prior=config.n_exponent_prior,
        likelihood_df=config.likelihood_df,
        likelihood_family=likelihood_family,
        auto_priors=bool(config.auto_priors),
        discretize_observation=config.discretize_observation,
        debut_prev_score_source=config.debut_prev_score_source,
        target_transform=target_transform,
        logit_offset=config.logit_offset,
        ar_center=config.ar_center,
        latent_process=config.latent_process,
        sigma_obs_prior_type=config.sigma_obs_prior_type,
        sigma_artist_prior_type=config.sigma_artist_prior_type,
        artist_effect_param=config.artist_effect_param,
        sigma_rw_lognormal_loc=config.sigma_rw_lognormal_loc,
        sigma_rw_lognormal_sigma=config.sigma_rw_lognormal_sigma,
        sigma_artist_lognormal_loc=config.sigma_artist_lognormal_loc,
        sigma_artist_lognormal_sigma=config.sigma_artist_lognormal_sigma,
        rho_loc=config.rho_loc,
        rho_scale=config.rho_scale,
        beta_prior_type=config.beta_prior_type,
        hs_global_scale=config.hs_global_scale,
        heteroscedastic_entity_obs=config.heteroscedastic_entity_obs,
        tau_entity_scale=config.tau_entity_scale,
        errors_in_variables=config.errors_in_variables,
        propagate_rw_horizon=config.propagate_rw_horizon,
        entity_group_pooling=config.entity_group_pooling,
        group_variance=config.group_variance,
        tau_group_sigma_scale=config.tau_group_sigma_scale,
        entity_effect_prior_type=config.entity_effect_prior_type,
        entity_skew_alpha_scale=config.entity_skew_alpha_scale,
        rw_innovation_type=config.rw_innovation_type,
        rw_skew_alpha_scale=config.rw_skew_alpha_scale,
        censor_at_bounds=config.censor_at_bounds,
        period_effects=config.period_effects,
        period_constraint=config.period_constraint,
        sigma_period_scale=config.sigma_period_scale,
        impute_missing=config.impute_missing,
        gbm_offset=config.gbm_offset,
        exclude_rw_raw_from_collection=config.exclude_rw_raw_from_collection,
        warmup_export_path=config.warmup_export_path,
        warmup_import_path=config.warmup_import_path,
        val_events=config.val_events,
        test_events=config.test_events,
        origin_offset=config.origin_offset,
        conformal_calibration=config.conformal_calibration,
        eval_horizon=config.eval_horizon,
        min_train_events=config.min_train_events,
        calibration_intervals=config.calibration_intervals,
        coverage_tolerance=config.coverage_tolerance,
        prediction_interval=config.prediction_interval,
        evaluate_secondary_split=config.evaluate_secondary_split,
        predictive_batch_size=config.predictive_batch_size,
        predict_entity_batch_size=config.predict_entity_batch_size,
        descriptor=descriptor,
    )
