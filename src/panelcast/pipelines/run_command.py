"""Reconstruct the ``panelcast run`` command a config represents.

The string is manifest provenance: only knobs that differ from their effective
default appear, where "effective" means the descriptor-resolved value for the
model facts a dataset owns (#268).
"""

from __future__ import annotations

from typing import Any

from panelcast.pipelines.pipeline_config import PipelineConfig, _get_default_config


def build_command_string(  # noqa: C901  # tracked complexity debt
    config: PipelineConfig, descriptor: Any
) -> str:
    """Build command string representation for manifest."""
    parts = ["panelcast run"]
    defaults = _get_default_config()
    # Descriptor-owned model facts (#268): the effective default is what
    # the descriptor resolves to, so a domain's default run needs no flag.
    eff_family = descriptor.likelihood_family or "studentt"
    eff_transform = descriptor.target_transform or "offset_logit"
    eff_max_events = descriptor.max_events or 50

    if config.seed != defaults.seed:
        parts.append(f"--seed {config.seed}")
    if config.skip_existing:
        parts.append("--skip-existing")
    if config.stages:
        parts.append(f"--stages {','.join(config.stages)}")
    if config.dry_run:
        parts.append("--dry-run")
    if config.strict:
        parts.append("--strict")
    if not config.enforce_lockfile:
        parts.append("--allow-unlocked-env")
    if config.verbose:
        parts.append("--verbose")
    if config.progress_bar is False:
        parts.append("--no-progress")
    if config.max_events != eff_max_events:
        parts.append(f"--max-events {config.max_events}")
    # MCMC config
    if config.num_chains != defaults.num_chains:
        parts.append(f"--num-chains {config.num_chains}")
    if config.num_samples != defaults.num_samples:
        parts.append(f"--num-samples {config.num_samples}")
    if config.num_warmup != defaults.num_warmup:
        parts.append(f"--num-warmup {config.num_warmup}")
    if config.target_accept != defaults.target_accept:
        parts.append(f"--target-accept {config.target_accept}")
    if config.max_tree_depth != defaults.max_tree_depth:
        parts.append(f"--max-tree-depth {config.max_tree_depth}")
    if config.init_strategy != defaults.init_strategy:
        parts.append(f"--init-strategy {config.init_strategy}")
    if config.chain_method != defaults.chain_method:
        parts.append(f"--chain-method {config.chain_method}")
    if config.checkpoint_every_draws is not None:
        parts.append(f"--checkpoint-every {config.checkpoint_every_draws}")
    if config.caged_chain_retries:
        parts.append(f"--caged-chain-retries {config.caged_chain_retries}")
    if config.caged_chain_tree_depth_fraction != defaults.caged_chain_tree_depth_fraction:
        parts.append(
            f"--caged-chain-tree-depth-fraction {config.caged_chain_tree_depth_fraction}"
        )
    if config.caged_chain_boundary_sigma != defaults.caged_chain_boundary_sigma:
        parts.append(f"--caged-chain-boundary-sigma {config.caged_chain_boundary_sigma}")
    if config.caged_chain_consensus_ratio != defaults.caged_chain_consensus_ratio:
        parts.append(f"--caged-chain-consensus-ratio {config.caged_chain_consensus_ratio}")
    # Convergence thresholds
    if config.rhat_threshold != defaults.rhat_threshold:
        parts.append(f"--rhat-threshold {config.rhat_threshold}")
    if config.ess_threshold != defaults.ess_threshold:
        parts.append(f"--ess-threshold {config.ess_threshold}")
    if config.allow_divergences:
        parts.append("--allow-divergences")
    # Data filtering. Record --min-ratings only when it differs from the
    # descriptor default it would otherwise resolve to (config.min_ratings
    # is already resolved to an int by __init__).
    if config.min_ratings != descriptor.primary_min_obs:
        parts.append(f"--min-ratings {config.min_ratings}")
    if config.min_events_filter != defaults.min_events_filter:
        parts.append(f"--min-events {config.min_events_filter}")
    # Feature flags
    if not config.enable_genre:
        parts.append("--no-genre")
    if not config.enable_artist:
        parts.append("--no-artist")
    if not config.enable_temporal:
        parts.append("--no-temporal")
    # Heteroscedastic noise (only if non-default and not learning)
    if config.n_exponent != defaults.n_exponent and not config.learn_n_exponent:
        parts.append(f"--n-exponent {config.n_exponent}")
    if config.learn_n_exponent:
        parts.append("--learn-n-exponent")
        if config.n_exponent_prior != defaults.n_exponent_prior:
            parts.append(f"--n-exponent-prior {config.n_exponent_prior}")
        # Only emit beta prior params when using beta prior
        if config.n_exponent_prior == "beta":
            if config.n_exponent_alpha != defaults.n_exponent_alpha:
                parts.append(f"--n-exponent-alpha {config.n_exponent_alpha}")
            if config.n_exponent_beta != defaults.n_exponent_beta:
                parts.append(f"--n-exponent-beta {config.n_exponent_beta}")
    if config.likelihood_df != defaults.likelihood_df:
        parts.append(f"--likelihood-df {config.likelihood_df}")
    if config.likelihood_family != eff_family:
        parts.append(f"--likelihood-family {config.likelihood_family}")
    if config.discretize_observation != defaults.discretize_observation:
        parts.append("--discretize-observation")
    # Model gates. The YAML-only knobs (logit_offset through the period
    # gates) have no CLI flags — they are recorded flag-style for
    # provenance and reproduced via run_config.yaml.
    if config.target_transform != eff_transform:
        parts.append(f"--target-transform {config.target_transform}")
    if config.logit_offset != defaults.logit_offset:
        parts.append(f"--logit-offset {config.logit_offset}")
    if config.ar_center != defaults.ar_center:
        parts.append(f"--ar-center {config.ar_center}")
    if config.latent_process != defaults.latent_process:
        parts.append(f"--latent-process {config.latent_process}")
    if config.debut_prev_score_source != defaults.debut_prev_score_source:
        parts.append(f"--debut-prev-score-source {config.debut_prev_score_source}")
    if config.sigma_obs_prior_type != defaults.sigma_obs_prior_type:
        parts.append(f"--sigma-obs-prior-type {config.sigma_obs_prior_type}")
    if config.sigma_artist_prior_type != defaults.sigma_artist_prior_type:
        parts.append(f"--sigma-artist-prior-type {config.sigma_artist_prior_type}")
    if config.artist_effect_param != defaults.artist_effect_param:
        parts.append(f"--artist-effect-param {config.artist_effect_param}")
    if config.sigma_rw_lognormal_loc != defaults.sigma_rw_lognormal_loc:
        parts.append(f"--sigma-rw-lognormal-loc {config.sigma_rw_lognormal_loc}")
    if config.sigma_rw_lognormal_sigma != defaults.sigma_rw_lognormal_sigma:
        parts.append(f"--sigma-rw-lognormal-sigma {config.sigma_rw_lognormal_sigma}")
    if config.sigma_artist_lognormal_loc != defaults.sigma_artist_lognormal_loc:
        parts.append(f"--sigma-artist-lognormal-loc {config.sigma_artist_lognormal_loc}")
    if config.sigma_artist_lognormal_sigma != defaults.sigma_artist_lognormal_sigma:
        parts.append(
            f"--sigma-artist-lognormal-sigma {config.sigma_artist_lognormal_sigma}"
        )
    if config.rho_loc != defaults.rho_loc:
        parts.append(f"--rho-loc {config.rho_loc}")
    if config.rho_scale != defaults.rho_scale:
        parts.append(f"--rho-scale {config.rho_scale}")
    if config.beta_prior_type != defaults.beta_prior_type:
        parts.append(f"--beta-prior-type {config.beta_prior_type}")
    if config.hs_global_scale != defaults.hs_global_scale:
        parts.append(f"--hs-global-scale {config.hs_global_scale}")
    if config.heteroscedastic_entity_obs != defaults.heteroscedastic_entity_obs:
        parts.append(
            "--heteroscedastic-entity-obs"
            if config.heteroscedastic_entity_obs
            else "--no-heteroscedastic-entity-obs"
        )
    if config.tau_entity_scale != defaults.tau_entity_scale:
        parts.append(f"--tau-entity-scale {config.tau_entity_scale}")
    if config.errors_in_variables:
        parts.append("--errors-in-variables")
    if config.propagate_rw_horizon:
        parts.append("--propagate-rw-horizon")
    if config.entity_group_pooling is not None:
        parts.append(
            "--entity-group-pooling"
            if config.entity_group_pooling
            else "--no-entity-group-pooling"
        )
    if config.group_variance != defaults.group_variance:
        parts.append(f"--group-variance {config.group_variance}")
    if config.entity_effect_prior_type != defaults.entity_effect_prior_type:
        parts.append(
            f"--entity-effect-prior-type {config.entity_effect_prior_type}"
        )
    if config.rw_innovation_type != defaults.rw_innovation_type:
        parts.append(f"--rw-innovation-type {config.rw_innovation_type}")
    if config.censor_at_bounds:
        parts.append("--censor-at-bounds")
    if config.period_effects:
        parts.append("--period-effects")
        if config.period_constraint != defaults.period_constraint:
            parts.append(f"--period-constraint {config.period_constraint}")
    if config.impute_missing:
        parts.append("--impute-missing")
    if config.gbm_offset != defaults.gbm_offset:
        parts.append("--gbm-offset" if config.gbm_offset else "--no-gbm-offset")
    if config.val_events != defaults.val_events:
        parts.append(f"--val-events {config.val_events}")
    if config.origin_offset != defaults.origin_offset:
        parts.append(f"--origin-offset {config.origin_offset}")
    if config.calibration_intervals != defaults.calibration_intervals:
        interval_str = ",".join(f"{p:.4g}" for p in config.calibration_intervals)
        parts.append(f"--calibration-intervals {interval_str}")
    if config.coverage_tolerance != defaults.coverage_tolerance:
        parts.append(f"--coverage-tolerance {config.coverage_tolerance}")
    if config.prediction_interval != defaults.prediction_interval:
        parts.append(f"--prediction-interval {config.prediction_interval}")
    if not config.evaluate_secondary_split:
        parts.append("--no-secondary-split")
    if config.dataset is not None:
        parts.append(f"--dataset {config.dataset}")
    if config.tag is not None:
        parts.append(f"--tag {config.tag}")

    return " ".join(parts)
