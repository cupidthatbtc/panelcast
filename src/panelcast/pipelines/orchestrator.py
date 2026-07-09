"""Pipeline orchestrator for end-to-end execution with progress tracking.

This module provides the PipelineOrchestrator class that executes pipeline
stages in dependency order, with features for:
- Progress display using Rich
- Hash-based skip logic for incremental runs
- Environment verification via pixi.lock
- Error handling with fail-fast semantics
- Manifest tracking for reproducibility
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import structlog
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from panelcast import __version__ as panelcast_version
from panelcast.config.descriptor import load_descriptor, resolve_descriptor_path
from panelcast.config.gates import (
    ArCenter,
    BetaPriorType,
    ChainMethod,
    DebutPrevScoreSource,
    LatentProcess,
    LikelihoodFamily,
    NExponentPrior,
    SigmaObsPriorType,
    TargetTransform,
)
from panelcast.paths import ArtifactPaths, resolve_latest
from panelcast.pipelines.errors import (
    ConvergenceError,
    EnvironmentError,
    PipelineError,
    StageSkipped,
)
from panelcast.pipelines.manifest import (
    GitStateModel,
    RunManifest,
    capture_environment,
    flag_differences,
    generate_run_id,
    load_run_manifest,
    save_run_manifest,
)
from panelcast.pipelines.stages import PipelineStage, StageContext, get_execution_order
from panelcast.pipelines.stamps import (
    CONSUMER_STAGES,
    DATA_STAGE_ROOTS,
    read_stamp,
    verify_stamps,
    write_stamp,
)
from panelcast.utils.environment import ensure_environment_locked, verify_environment
from panelcast.utils.git_state import capture_git_state
from panelcast.utils.hashing import sha256_path
from panelcast.utils.logging import is_interactive, setup_pipeline_logging
from panelcast.utils.random import set_seeds

log = structlog.get_logger()

# Module-level reference for default config values (used to detect non-default flags)
_DEFAULT_CONFIG: PipelineConfig | None = None


def _get_default_config() -> PipelineConfig:
    """Get a cached default PipelineConfig instance for comparison."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = PipelineConfig()
    return _DEFAULT_CONFIG


def _reset_default_config() -> None:
    """Reset cached default config (for testing only)."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        seed: Random seed for reproducibility (default 42).
        skip_existing: If True, skip stages with unchanged inputs (default False).
        stages: List of stage names to run, or None for all stages.
        dry_run: If True, log what would run without executing (default False).
        strict: If True, fail on convergence warnings (default False).
        enforce_lockfile: If True, fail if pixi.lock missing (default True).
        verbose: If True, enable DEBUG logging (default False).
        resume: Run ID to resume, or None for fresh run.
        max_albums: Maximum albums per artist for model training (default 50).
        num_chains: Number of parallel MCMC chains (default 4).
        num_samples: Post-warmup samples per chain (default 1000).
        num_warmup: Warmup iterations per chain (default 1000).
        target_accept: Target acceptance probability (default 0.90).
        max_tree_depth: Maximum tree depth for NUTS (default 10).
        rhat_threshold: Maximum acceptable R-hat (default 1.01).
        ess_threshold: Minimum ESS per chain (default 400).
        allow_divergences: If True, don't fail on divergences (default False).
        min_ratings: Minimum primary observations per event, or None to resolve
            from the descriptor's ``primary_min_obs`` at run time (default None).
        min_albums_filter: Minimum albums per artist for dynamic effects (default 2).
        enable_genre: If False, disable genre features (default True).
        enable_artist: If False, disable artist features (default True).
        enable_temporal: If False, disable temporal features (default True).
        n_exponent: Scaling exponent for review count noise adjustment (default 0.0).
        learn_n_exponent: If True, learn exponent from data using prior (default False).
        n_exponent_alpha: Beta prior alpha parameter for learned exponent (default 2.0).
        n_exponent_beta: Beta prior beta parameter for learned exponent (default 4.0).
        n_exponent_prior: Prior for learned exponent: 'logit-normal' or 'beta'.

    Example:
        >>> config = PipelineConfig(seed=42, dry_run=True)
        >>> config.stages is None  # Run all stages
        True
    """

    seed: int = 42
    skip_existing: bool = False
    stages: list[str] | None = None
    dry_run: bool = False
    strict: bool = False
    enforce_lockfile: bool = True
    verbose: bool = False
    # MCMC progress bars: None = auto (stderr TTY only), False = --no-progress.
    # Execution mechanics only — never affects outputs, skip detection, or resume.
    progress_bar: bool | None = None
    resume: str | None = None
    # Free-form run label recorded in the manifest (surfaced by `runs history`).
    # Provenance only — never affects outputs or skip detection.
    tag: str | None = None
    max_albums: int = 50
    # MCMC configuration
    num_chains: int = 4
    num_samples: int = 1000
    num_warmup: int = 1000
    target_accept: float = 0.90
    max_tree_depth: int = 10
    chain_method: ChainMethod = "sequential"
    checkpoint_every_draws: int | None = None
    # Warmup-transfer seams (YAML-only; the select runner writes them per arm).
    warmup_export_path: str | None = None
    warmup_import_path: str | None = None
    # Convergence thresholds
    rhat_threshold: float = 1.01
    ess_threshold: int = 400
    allow_divergences: bool = False
    # Data filtering. min_ratings=None defers to the descriptor's primary_min_obs
    # (resolved in the orchestrator), so a retargeted domain needs no
    # --min-ratings on the command line. An explicit value (CLI/YAML) wins.
    min_ratings: int | None = None
    min_albums_filter: int = 2
    # Feature flags
    enable_genre: bool = True
    enable_artist: bool = True
    enable_temporal: bool = True
    # Heteroscedastic noise configuration
    n_exponent: float = 0.0
    learn_n_exponent: bool = False
    n_exponent_alpha: float = 2.0
    n_exponent_beta: float = 4.0
    n_exponent_prior: NExponentPrior = "logit-normal"
    # Likelihood configuration
    likelihood_df: float = 4.0
    # Likelihood family gate: "studentt" (legacy) | "normal" | "skew_studentt" /
    # "skew_normal" (sinh-arcsinh skew) | "split_normal" (two-piece) | "beta"
    # (bounded mean-precision Beta on [low, high]).
    likelihood_family: LikelihoodFamily = "studentt"
    # Discretization gate: interval-censor the observation to integers (default
    # off => continuous likelihood). Location-scale families only; not for beta.
    discretize_observation: bool = False
    # Debut prev_score fill source: "train_mean" | "dataset_stats" (legacy)
    debut_prev_score_source: DebutPrevScoreSource = "train_mean"
    # Target transform gate: "offset_logit" (default since 0.5.0 — promoted on
    # the corrected #63 ledger, +22 held-out elpd) | "identity" (former default)
    target_transform: TargetTransform = "offset_logit"
    logit_offset: float = 0.5
    # AR(1) centering gate: "global" | "none" (legacy) | "artist_running"
    ar_center: ArCenter = "global"
    # Latent artist-effect process gate: "rw" (legacy) | "ar1" (experimental)
    latent_process: LatentProcess = "rw"
    # sigma_obs prior family gate: "halfnormal" (legacy default) | "lognormal"
    # (removes the zero-boundary pile-up behind the econ variance-collapse).
    sigma_obs_prior_type: SigmaObsPriorType = "halfnormal"
    # Covariate-block prior gate (#155): "normal" (legacy default, bit-identical
    # RNG path) | "horseshoe" (regularized horseshoe; global-local shrinkage
    # against the #76 coefficient dilution). No CLI flag; via run_config.yaml.
    beta_prior_type: BetaPriorType = "normal"
    # Horseshoe global scale (tau_0), the sparsity knob a bake-off sweeps.
    # Read only when beta_prior_type="horseshoe".
    hs_global_scale: float = 0.1
    # Entity-level observation overdispersion gate: False (legacy default,
    # bit-identical RNG path) | True (per-entity multiplicative noise inflation
    # widening intervals for noisy series). tau_entity_scale sets the prior
    # HalfNormal scale on the entity-noise dispersion.
    heteroscedastic_entity_obs: bool = False
    tau_entity_scale: float = 0.25
    # Errors-in-variables gate (model-v2): de-noise the AR(1) lagged regressor
    # with a measurement-error latent so rho de-attenuates. Default off => legacy
    # bit-identical path. No CLI flag; configured via run_config.yaml.
    errors_in_variables: bool = False
    # Long-horizon random-walk variance gate (model-v2): at prediction time drop
    # the album_seq clamp at max_seq_train so deep-extrapolation intervals widen.
    # Default off => legacy clamp. No CLI flag.
    propagate_rw_horizon: bool = False
    # Genre/group pooling level between the global mean and the entity effects
    # (#41): each entity's init-effect location shifts by a learned zero-sum
    # group offset. None = auto (default since 0.6.0, promoted on the #85
    # screening + publication confirmation): on where the domain supports it —
    # the descriptor names an entity_group_col and the training split has that
    # column. Explicit True/False always wins (True hard-fails on unsupported
    # domains). No CLI flag.
    entity_group_pooling: bool | None = None
    # Stacked-GBM offset feature block (#86): a gradient-boosted prediction of
    # the target from the other blocks' outputs enters X as one more covariate
    # (out-of-fold for train rows). Default on since 0.6.0 (promoted on the
    # #86 screening + publication confirmation: +224 paired held-out ELPD and
    # better point accuracy at nominal coverage); works for every domain since
    # it needs only the descriptor target and row ids. No CLI flag.
    gbm_offset: bool = True
    # Opt-in in-sampler exclusion of the rw_raw tensor: never store its draws
    # on device during sampling (~96% peak-GPU cut at production settings;
    # posterior parity for all other sites guarded by tests).
    exclude_rw_raw_from_collection: bool = False
    # Split configuration. min_train_albums matches the documented `run` CLI
    # default (2) so `stage splits` / `demo` build the same split population.
    val_albums: int = 0
    min_train_albums: int = 2
    # Rolling-origin backtest offset (0 = the standard split)
    origin_offset: int = 0
    # Conformal calibration wrapper on the predictive (#156; needs val_albums >= 1)
    conformal_calibration: bool = False
    # Evaluation configuration
    calibration_intervals: tuple[float, ...] = (0.80, 0.95)
    coverage_tolerance: float = 0.03
    prediction_interval: float = 0.95
    evaluate_secondary_split: bool = True
    # Prediction batching (memory/speed trade-off, not statistically relevant)
    predictive_batch_size: int = 500
    predict_artist_batch_size: int = 50
    # Dataset descriptor reference (bare name or YAML path; None = AOTY defaults)
    dataset: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Called by __post_init__ and can be called after setattr modifications
        (e.g., after restoring config from manifest).
        """
        valid_priors = ("logit-normal", "beta")
        if self.n_exponent_prior not in valid_priors:
            raise ValueError(
                f"Invalid n_exponent_prior: '{self.n_exponent_prior}'. "
                f"Must be one of {valid_priors}."
            )
        if not 5 <= self.max_tree_depth <= 15:
            raise ValueError(
                f"Invalid max_tree_depth: {self.max_tree_depth}. Must be between 5 and 15."
            )
        if len(self.calibration_intervals) == 0:
            raise ValueError("calibration_intervals must contain at least one probability level.")
        for prob in self.calibration_intervals:
            if not 0.0 < prob < 1.0:
                raise ValueError(f"Invalid calibration interval {prob}. Must be in (0, 1).")
        if self.target_transform not in ("identity", "offset_logit"):
            raise ValueError(
                f"Invalid target_transform: '{self.target_transform}'. "
                "Must be 'identity' or 'offset_logit'."
            )
        self._validate_likelihood()
        if self.debut_prev_score_source not in ("train_mean", "dataset_stats"):
            raise ValueError(
                f"Invalid debut_prev_score_source: '{self.debut_prev_score_source}'. "
                "Must be 'train_mean' or 'dataset_stats'."
            )
        if self.ar_center not in ("global", "none", "artist_running"):
            raise ValueError(
                f"Invalid ar_center: '{self.ar_center}'. "
                "Must be 'global', 'none', or 'artist_running'."
            )
        if self.latent_process not in ("rw", "ar1"):
            raise ValueError(
                f"Invalid latent_process: '{self.latent_process}'. Must be 'rw' or 'ar1'."
            )
        if self.sigma_obs_prior_type not in ("halfnormal", "lognormal"):
            raise ValueError(
                f"Invalid sigma_obs_prior_type: '{self.sigma_obs_prior_type}'. "
                "Must be 'halfnormal' or 'lognormal'."
            )
        if self.beta_prior_type not in ("normal", "horseshoe"):
            raise ValueError(
                f"Invalid beta_prior_type: '{self.beta_prior_type}'. "
                "Must be 'normal' or 'horseshoe'."
            )
        if self.hs_global_scale <= 0.0:
            raise ValueError(f"Invalid hs_global_scale: {self.hs_global_scale}. Must be > 0.")
        if self.tau_entity_scale <= 0.0:
            raise ValueError(f"Invalid tau_entity_scale: {self.tau_entity_scale}. Must be > 0.")
        if self.coverage_tolerance < 0.0:
            raise ValueError("coverage_tolerance must be >= 0.")
        if not 0.0 < self.prediction_interval < 1.0:
            raise ValueError("prediction_interval must be in (0, 1).")
        if self.num_chains < 1:
            raise ValueError("num_chains must be >= 1.")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if self.checkpoint_every_draws is not None and self.checkpoint_every_draws < 1:
            raise ValueError("checkpoint_every_draws must be >= 1 when set.")
        if self.ess_threshold < 1:
            raise ValueError("ess_threshold must be >= 1.")
        if self.strict and self.num_chains < 2:
            raise ValueError(
                "strict mode requires num_chains >= 2 for R-hat diagnostics. "
                "Increase --num-chains or disable --strict."
            )
        if self.strict and self.num_samples < self.ess_threshold:
            raise ValueError(
                "strict mode requires num_samples >= ess_threshold per chain for ESS checks. "
                f"Got num_samples={self.num_samples}, ess_threshold={self.ess_threshold}."
            )

    def _validate_likelihood(self) -> None:
        """Validate the likelihood family and its structural constraints."""
        from panelcast.models.bayes.likelihoods import REGISTRY

        valid_families = tuple(REGISTRY)
        if self.likelihood_family not in valid_families:
            raise ValueError(
                f"Invalid likelihood_family: '{self.likelihood_family}'. "
                f"Must be one of: {', '.join(valid_families)}."
            )
        spec = REGISTRY[self.likelihood_family]
        if self.discretize_observation and not spec.supports_discretization:
            supported = [f for f, s in REGISTRY.items() if s.supports_discretization]
            raise ValueError(
                f"discretize_observation=True is not supported by likelihood_family "
                f"'{self.likelihood_family}'. Supported: {', '.join(supported)}."
            )
        if self.discretize_observation and self.target_transform != "identity":
            raise ValueError(
                "discretize_observation=True requires target_transform='identity': "
                "discretization interval-censors integers on the raw score scale, "
                f"but target_transform='{self.target_transform}' moves y off that scale."
            )
        if spec.requires_identity_transform and self.target_transform != "identity":
            raise ValueError(
                f"likelihood_family='{self.likelihood_family}' requires "
                f"target_transform='identity' (got '{self.target_transform}'): "
                "the bounded likelihood assumes mu is on the score scale."
            )
        if not spec.uses_sigma:
            inert = [
                knob
                for knob, enabled in (
                    ("learn_n_exponent", self.learn_n_exponent),
                    ("heteroscedastic_entity_obs", self.heteroscedastic_entity_obs),
                    ("n_exponent", self.n_exponent != 0.0),
                )
                if enabled
            ]
            if inert:
                raise ValueError(
                    f"{', '.join(inert)} cannot be used with likelihood_family="
                    f"'{self.likelihood_family}': the family draws its own precision "
                    "and ignores sigma, so these options would be silently inert."
                )


class PipelineOrchestrator:
    """Orchestrates pipeline execution with progress tracking and error handling.

    The orchestrator manages the full pipeline lifecycle:
    1. Verify environment (pixi.lock check)
    2. Create run directory and manifest
    3. Execute stages in dependency order
    4. Track progress with Rich display
    5. Handle errors with fail-fast semantics
    6. Create outputs/latest symlink on success

    Attributes:
        config: Pipeline configuration options.
        output_base: Base directory for output runs (default "outputs").
        run_dir: Path to current run directory (set during run).
        manifest: Current run manifest (set during run).

    Example:
        >>> config = PipelineConfig(seed=42, dry_run=True)
        >>> orchestrator = PipelineOrchestrator(config)
        >>> exit_code = orchestrator.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_base: Path | str = Path("outputs"),
    ) -> None:
        """Initialize orchestrator with configuration.

        Args:
            config: Pipeline configuration.
            output_base: Base directory for outputs (default "outputs").
        """
        self.config = config
        self.output_base = Path(output_base)
        self.run_dir: Path | None = None
        self.manifest: RunManifest | None = None
        self._start_time: float = 0.0
        self._resolved_paths: ArtifactPaths | None = None
        # Resolve the dataset descriptor once; every stage reads it from the
        # StageContext rather than re-deriving domain names from literals.
        self.descriptor = load_descriptor(config.dataset)
        self.descriptor_path = resolve_descriptor_path(config.dataset)
        # beta_binomial models the target as the mean of n aggregated ratings, so
        # it only makes sense when n_obs_col is a true count of independent raters.
        if (
            config.likelihood_family == "beta_binomial"
            and not self.descriptor.n_obs_is_aggregation_count
        ):
            raise ValueError(
                "likelihood_family='beta_binomial' models the target as the mean of "
                f"n={self.descriptor.n_obs_col} aggregated ratings, but descriptor "
                f"'{self.descriptor.name}' sets n_obs_is_aggregation_count=false "
                f"({self.descriptor.n_obs_col} is not a count of independent raters). "
                "Use an aggregation-count domain or a different likelihood_family."
            )
        # Resolve the observation threshold: an explicit CLI/YAML value wins;
        # otherwise fall back to the descriptor's primary_min_obs so retargeted
        # domains don't need --min-ratings on the command line. The data stage
        # only materializes parquets at the descriptor's thresholds, so any
        # other value would die hours later at the splits stage.
        if config.min_ratings is None:
            config.min_ratings = self.descriptor.primary_min_obs
        elif config.min_ratings not in self.descriptor.min_obs_thresholds:
            raise ValueError(
                f"min_ratings={config.min_ratings} has no materialized dataset for "
                f"descriptor '{self.descriptor.name}': the data stage only writes "
                f"thresholds {sorted(self.descriptor.min_obs_thresholds)}. Pick one "
                "of those, or add the value to the descriptor's min_obs_thresholds."
            )

    def run(self) -> int:
        """Execute the pipeline and return exit code.

        Runs all configured stages in dependency order with progress tracking.
        Creates run manifest, handles errors, and maintains output structure.

        Returns:
            Exit code: 0 on success, error's exit_code on failure.

        Raises:
            EnvironmentError: If strict=True and pixi.lock missing.
        """
        self._start_time = time()

        # 1. Verify environment
        try:
            self._verify_environment()
        except EnvironmentError as e:
            log.error("environment_verification_failed", error=str(e))
            return e.exit_code

        # 2. Set up run directory and manifest
        self._setup_run()

        # 3. Set up logging
        log_file = self.run_dir / "pipeline.log.json" if self.run_dir else None
        setup_pipeline_logging(verbose=self.config.verbose, log_file=log_file)

        # 4. Set random seeds
        set_seeds(self.config.seed)

        # Check for config conflicts
        if self.config.learn_n_exponent and self.config.n_exponent != 0.0:
            log.warning(
                "config_conflict",
                message="Both --n-exponent and --learn-n-exponent set; using learned mode",
            )
            # Clear the fixed exponent to prevent manifest recording stale value
            self.config.n_exponent = 0.0

        log.info(
            "pipeline_started",
            run_id=self.manifest.run_id if self.manifest else "unknown",
            seed=self.config.seed,
            dry_run=self.config.dry_run,
            stages=self.config.stages,
            n_exponent=self.config.n_exponent,
            learn_n_exponent=self.config.learn_n_exponent,
        )

        # 5. Get execution order (pass min_ratings for correct input_paths)
        try:
            stages = get_execution_order(
                self.config.stages,
                min_ratings=self.config.min_ratings,
                descriptor=self.descriptor,
                descriptor_path=self.descriptor_path,
                paths=self._artifact_paths(),
            )
        except KeyError as e:
            log.error("invalid_stage", error=str(e))
            return 1
        except PipelineError as e:
            # Consumer-only invocations fail here when no prior run supplies
            # their inputs; route through the normal failure path.
            self._handle_failure(e, e.stage)
            return e.exit_code

        if not stages:
            log.warning("no_stages_to_execute")
            self._finalize_success()
            return 0

        # 6. Execute stages
        try:
            self._execute_stages(stages)
            self._finalize_success()
            return 0
        except PipelineError as e:
            self._handle_failure(e, e.stage)
            return e.exit_code
        except Exception as e:
            self._handle_failure(e, "unknown")
            return 1

    def _verify_environment(self) -> None:
        """Verify environment is locked for reproducibility.

        Raises EnvironmentError if pixi.lock is not found when
        config.enforce_lockfile=True.
        """
        log.debug("verifying_environment", enforce_lockfile=self.config.enforce_lockfile)

        try:
            ensure_environment_locked(strict=self.config.enforce_lockfile)
        except Exception as e:
            # Re-raise as our EnvironmentError for consistent exit code
            raise EnvironmentError(str(e)) from e

        # Log environment status
        status = verify_environment()
        if status.is_reproducible:
            log.info(
                "environment_verified",
                pixi_lock_hash=status.pixi_lock_hash[:12] if status.pixi_lock_hash else None,
            )
        else:
            log.warning("environment_not_locked", warnings=status.warnings)

    def _setup_run(self) -> None:
        """Create run directory and initialize manifest."""
        # Handle resume vs fresh run
        if self.config.resume:
            self._setup_resume()
            return

        # Generate new run ID and create its directory EXCLUSIVELY: a second
        # run minting the same id must retry with a fresh one instead of
        # silently sharing (and, on failure, rmtree'ing) this run's dir.
        run_id = ""
        for _ in range(10):
            run_id = generate_run_id()
            self.run_dir = self.output_base / run_id
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise PipelineError(
                f"Could not create a unique run directory under {self.output_base} "
                f"after 10 attempts (last id: {run_id}).",
                stage="setup",
            )

        # Capture git state and environment
        git_state = capture_git_state()
        environment = capture_environment()

        # Build command string for manifest
        command = self._build_command_string()

        # Create manifest
        self.manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            version=panelcast_version,
            tag=self.config.tag,
            command=command,
            flags={
                "seed": self.config.seed,
                "skip_existing": self.config.skip_existing,
                "stages": self.config.stages,
                "dry_run": self.config.dry_run,
                "strict": self.config.strict,
                "enforce_lockfile": self.config.enforce_lockfile,
                "verbose": self.config.verbose,
                "progress_bar": self.config.progress_bar,
                "resume": self.config.resume,
                "max_albums": self.config.max_albums,
                # MCMC config
                "num_chains": self.config.num_chains,
                "num_samples": self.config.num_samples,
                "num_warmup": self.config.num_warmup,
                "target_accept": self.config.target_accept,
                "max_tree_depth": self.config.max_tree_depth,
                "chain_method": self.config.chain_method,
                # Convergence thresholds
                "rhat_threshold": self.config.rhat_threshold,
                "ess_threshold": self.config.ess_threshold,
                "allow_divergences": self.config.allow_divergences,
                # Data filtering
                "min_ratings": self.config.min_ratings,
                "min_albums_filter": self.config.min_albums_filter,
                # Feature flags
                "enable_genre": self.config.enable_genre,
                "enable_artist": self.config.enable_artist,
                "enable_temporal": self.config.enable_temporal,
                # Heteroscedastic noise
                "n_exponent": self.config.n_exponent,
                "learn_n_exponent": self.config.learn_n_exponent,
                "n_exponent_alpha": self.config.n_exponent_alpha,
                "n_exponent_beta": self.config.n_exponent_beta,
                "n_exponent_prior": self.config.n_exponent_prior,
                "likelihood_df": self.config.likelihood_df,
                "likelihood_family": self.config.likelihood_family,
                "discretize_observation": self.config.discretize_observation,
                "debut_prev_score_source": self.config.debut_prev_score_source,
                "target_transform": self.config.target_transform,
                "logit_offset": self.config.logit_offset,
                "ar_center": self.config.ar_center,
                "latent_process": self.config.latent_process,
                "sigma_obs_prior_type": self.config.sigma_obs_prior_type,
                "beta_prior_type": self.config.beta_prior_type,
                "hs_global_scale": self.config.hs_global_scale,
                "heteroscedastic_entity_obs": self.config.heteroscedastic_entity_obs,
                "tau_entity_scale": self.config.tau_entity_scale,
                "errors_in_variables": self.config.errors_in_variables,
                "propagate_rw_horizon": self.config.propagate_rw_horizon,
                "entity_group_pooling": self.config.entity_group_pooling,
                "gbm_offset": self.config.gbm_offset,
                "exclude_rw_raw_from_collection": self.config.exclude_rw_raw_from_collection,
                "val_albums": self.config.val_albums,
                "origin_offset": self.config.origin_offset,
                "conformal_calibration": self.config.conformal_calibration,
                "min_train_albums": self.config.min_train_albums,
                # Evaluation
                "calibration_intervals": list(self.config.calibration_intervals),
                "coverage_tolerance": self.config.coverage_tolerance,
                "prediction_interval": self.config.prediction_interval,
                "evaluate_secondary_split": self.config.evaluate_secondary_split,
                # Dataset descriptor provenance
                "dataset": self.config.dataset,
                "dataset_descriptor_hash": self.descriptor.descriptor_hash(),
            },
            seed=self.config.seed,
            git=GitStateModel.from_git_state(git_state),
            environment=environment,
            input_hashes={},
            stage_hashes={},
            stages_completed=[],
            stages_skipped=[],
            outputs={},
            success=False,
            error=None,
            duration_seconds=0.0,
        )

        # Save initial manifest
        save_run_manifest(self.manifest, self.run_dir)

        # The post-layering truth (preset + YAML overlays + CLI wins +
        # descriptor-resolved values) — the manifest command string cannot
        # express YAML-only gates, so this is what `runs reproduce` re-executes.
        from panelcast.config.pipeline_yaml import dump_resolved_config

        (self.run_dir / "resolved_config.yaml").write_text(
            dump_resolved_config(self.config), encoding="utf-8"
        )

    # MCMC config keys that should be restored from manifest on resume
    RESUME_CONFIG_KEYS = (
        # The RNG seed governs the whole MCMC draw; a resume that reverts to the
        # CLI default silently re-fits with a different posterior than the run it
        # claims to continue (and the manifest still records the original seed).
        "seed",
        "target_accept",
        "max_tree_depth",
        "chain_method",
        "num_chains",
        "num_samples",
        "num_warmup",
        # Resume must reuse the checkpoint identity, which hashes the MCMC config.
        "checkpoint_every_draws",
        "n_exponent",
        "learn_n_exponent",
        "n_exponent_prior",
        "n_exponent_alpha",
        "n_exponent_beta",
        "likelihood_df",
        "likelihood_family",
        "discretize_observation",
        "debut_prev_score_source",
        "target_transform",
        "logit_offset",
        "ar_center",
        "latent_process",
        "sigma_obs_prior_type",
        "beta_prior_type",
        "hs_global_scale",
        "heteroscedastic_entity_obs",
        "tau_entity_scale",
        "errors_in_variables",
        "propagate_rw_horizon",
        "entity_group_pooling",
        "gbm_offset",
        "exclude_rw_raw_from_collection",
        "max_albums",
        "min_ratings",
        "min_albums_filter",
        "min_train_albums",
        # Split geometry: a resume that reverts these regenerates different
        # splits than the run it claims to continue.
        "val_albums",
        "origin_offset",
        "calibration_intervals",
        "coverage_tolerance",
        "prediction_interval",
        "evaluate_secondary_split",
        "conformal_calibration",
        "enforce_lockfile",
        "dataset",
    )
    # Flags that should not invalidate input-hash skip detection.
    # These only affect execution mechanics, not stage outputs.
    SKIP_FLAG_IGNORE = frozenset(
        {"skip_existing", "dry_run", "verbose", "resume", "progress_bar"}
    )

    def _skip_flag_differences(self, previous_manifest: RunManifest) -> list[str]:
        """Return output-affecting flag keys that changed since previous run."""
        if self.manifest is None:
            return []
        return [
            key
            for key, _, _ in flag_differences(
                self.manifest.flags,
                previous_manifest.flags,
                _get_default_config(),
                ignore=self.SKIP_FLAG_IGNORE,
            )
        ]

    def _setup_resume(self) -> None:
        """Set up for resuming a previous run."""
        resume_id = self.config.resume

        # Try to find the run directory
        run_dir = self.output_base / resume_id
        failed_dir = self.output_base / "failed" / resume_id

        if run_dir.exists():
            self.run_dir = run_dir
        elif failed_dir.exists():
            # Move back from failed for retry
            self.run_dir = run_dir
            shutil.move(str(failed_dir), str(run_dir))
        else:
            raise PipelineError(
                f"Cannot find run to resume: {resume_id}",
                stage="setup",
            )

        # Load existing manifest
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            raise PipelineError(
                f"No manifest.json in run directory: {resume_id}",
                stage="setup",
            )

        self.manifest = load_run_manifest(manifest_path)

        # Restore MCMC config from manifest to prevent config drift
        self._restore_config_from_manifest()

        log.info(
            "resuming_run",
            run_id=resume_id,
            completed_stages=self.manifest.stages_completed,
        )

    def _restore_config_from_manifest(self) -> None:
        """Restore MCMC config values from manifest flags.

        For each key in RESUME_CONFIG_KEYS:
        - If present in manifest flags, restore it to self.config
        - If missing, emit a warning about potential config drift
        """
        if self.manifest is None:
            return

        for key in self.RESUME_CONFIG_KEYS:
            if key in self.manifest.flags:
                manifest_value = self.manifest.flags[key]
                setattr(self.config, key, manifest_value)
                log.debug("resume_config_restored", key=key, value=manifest_value)
            else:
                current_default = getattr(self.config, key)
                log.warning(
                    "resume_config_missing",
                    key=key,
                    current_default=current_default,
                    message=(
                        f"manifest missing '{key}', using current default {current_default} "
                        "- verify this matches original run"
                    ),
                )

        # Re-resolve the descriptor for the restored dataset reference and
        # guard against descriptor drift: resuming a run whose descriptor
        # YAML has changed since the original run would silently mix domains.
        self.descriptor = load_descriptor(self.config.dataset)
        self.descriptor_path = resolve_descriptor_path(self.config.dataset)
        # __init__ resolved min_ratings against the pre-resume (CLI/default)
        # descriptor; the manifest restore above re-pointed the dataset, so
        # re-derive the threshold from the restored descriptor when it wasn't
        # pinned in the manifest. Without this a resumed cross-domain run keeps
        # the wrong threshold and reads the wrong processed parquet.
        if self.config.min_ratings is None:
            self.config.min_ratings = self.descriptor.primary_min_obs
        recorded_hash = self.manifest.flags.get("dataset_descriptor_hash")
        if recorded_hash is None:
            log.warning(
                "resume_descriptor_hash_missing",
                message=(
                    "manifest predates descriptor tracking; assuming AOTY "
                    "defaults match the original run"
                ),
            )
        elif recorded_hash != self.descriptor.descriptor_hash():
            raise PipelineError(
                "Dataset descriptor changed since the original run "
                f"(recorded hash {recorded_hash[:12]}…, current "
                f"{self.descriptor.descriptor_hash()[:12]}…). Resuming would mix "
                "artifacts from different dataset definitions. Start a fresh "
                "run instead.",
                stage="setup",
            )

        # Re-validate after restoration (catches corrupted/invalid manifest values)
        self.config._validate()

    def _build_command_string(self) -> str:  # noqa: C901  # tracked complexity debt
        """Build command string representation for manifest."""
        parts = ["panelcast run"]
        defaults = _get_default_config()

        if self.config.seed != defaults.seed:
            parts.append(f"--seed {self.config.seed}")
        if self.config.skip_existing:
            parts.append("--skip-existing")
        if self.config.stages:
            parts.append(f"--stages {','.join(self.config.stages)}")
        if self.config.dry_run:
            parts.append("--dry-run")
        if self.config.strict:
            parts.append("--strict")
        if not self.config.enforce_lockfile:
            parts.append("--allow-unlocked-env")
        if self.config.verbose:
            parts.append("--verbose")
        if self.config.progress_bar is False:
            parts.append("--no-progress")
        if self.config.max_albums != defaults.max_albums:
            parts.append(f"--max-albums {self.config.max_albums}")
        # MCMC config
        if self.config.num_chains != defaults.num_chains:
            parts.append(f"--num-chains {self.config.num_chains}")
        if self.config.num_samples != defaults.num_samples:
            parts.append(f"--num-samples {self.config.num_samples}")
        if self.config.num_warmup != defaults.num_warmup:
            parts.append(f"--num-warmup {self.config.num_warmup}")
        if self.config.target_accept != defaults.target_accept:
            parts.append(f"--target-accept {self.config.target_accept}")
        if self.config.max_tree_depth != defaults.max_tree_depth:
            parts.append(f"--max-tree-depth {self.config.max_tree_depth}")
        if self.config.chain_method != defaults.chain_method:
            parts.append(f"--chain-method {self.config.chain_method}")
        if self.config.checkpoint_every_draws is not None:
            parts.append(f"--checkpoint-every {self.config.checkpoint_every_draws}")
        # Convergence thresholds
        if self.config.rhat_threshold != defaults.rhat_threshold:
            parts.append(f"--rhat-threshold {self.config.rhat_threshold}")
        if self.config.ess_threshold != defaults.ess_threshold:
            parts.append(f"--ess-threshold {self.config.ess_threshold}")
        if self.config.allow_divergences:
            parts.append("--allow-divergences")
        # Data filtering. Record --min-ratings only when it differs from the
        # descriptor default it would otherwise resolve to (config.min_ratings
        # is already resolved to an int by __init__).
        if self.config.min_ratings != self.descriptor.primary_min_obs:
            parts.append(f"--min-ratings {self.config.min_ratings}")
        if self.config.min_albums_filter != defaults.min_albums_filter:
            parts.append(f"--min-albums {self.config.min_albums_filter}")
        # Feature flags
        if not self.config.enable_genre:
            parts.append("--no-genre")
        if not self.config.enable_artist:
            parts.append("--no-artist")
        if not self.config.enable_temporal:
            parts.append("--no-temporal")
        # Heteroscedastic noise (only if non-default and not learning)
        if self.config.n_exponent != defaults.n_exponent and not self.config.learn_n_exponent:
            parts.append(f"--n-exponent {self.config.n_exponent}")
        if self.config.learn_n_exponent:
            parts.append("--learn-n-exponent")
            if self.config.n_exponent_prior != defaults.n_exponent_prior:
                parts.append(f"--n-exponent-prior {self.config.n_exponent_prior}")
            # Only emit beta prior params when using beta prior
            if self.config.n_exponent_prior == "beta":
                if self.config.n_exponent_alpha != defaults.n_exponent_alpha:
                    parts.append(f"--n-exponent-alpha {self.config.n_exponent_alpha}")
                if self.config.n_exponent_beta != defaults.n_exponent_beta:
                    parts.append(f"--n-exponent-beta {self.config.n_exponent_beta}")
        if self.config.likelihood_df != defaults.likelihood_df:
            parts.append(f"--likelihood-df {self.config.likelihood_df}")
        if self.config.likelihood_family != defaults.likelihood_family:
            parts.append(f"--likelihood-family {self.config.likelihood_family}")
        if self.config.discretize_observation != defaults.discretize_observation:
            parts.append("--discretize-observation")
        # Model gates. The YAML-only knobs (logit_offset through
        # entity_group_pooling) have no CLI flags — they are recorded
        # flag-style for provenance and reproduced via run_config.yaml.
        if self.config.target_transform != defaults.target_transform:
            parts.append(f"--target-transform {self.config.target_transform}")
        if self.config.logit_offset != defaults.logit_offset:
            parts.append(f"--logit-offset {self.config.logit_offset}")
        if self.config.ar_center != defaults.ar_center:
            parts.append(f"--ar-center {self.config.ar_center}")
        if self.config.latent_process != defaults.latent_process:
            parts.append(f"--latent-process {self.config.latent_process}")
        if self.config.debut_prev_score_source != defaults.debut_prev_score_source:
            parts.append(f"--debut-prev-score-source {self.config.debut_prev_score_source}")
        if self.config.sigma_obs_prior_type != defaults.sigma_obs_prior_type:
            parts.append(f"--sigma-obs-prior-type {self.config.sigma_obs_prior_type}")
        if self.config.beta_prior_type != defaults.beta_prior_type:
            parts.append(f"--beta-prior-type {self.config.beta_prior_type}")
        if self.config.hs_global_scale != defaults.hs_global_scale:
            parts.append(f"--hs-global-scale {self.config.hs_global_scale}")
        if self.config.heteroscedastic_entity_obs:
            parts.append("--heteroscedastic-entity-obs")
        if self.config.tau_entity_scale != defaults.tau_entity_scale:
            parts.append(f"--tau-entity-scale {self.config.tau_entity_scale}")
        if self.config.errors_in_variables:
            parts.append("--errors-in-variables")
        if self.config.propagate_rw_horizon:
            parts.append("--propagate-rw-horizon")
        if self.config.entity_group_pooling is not None:
            parts.append(
                "--entity-group-pooling"
                if self.config.entity_group_pooling
                else "--no-entity-group-pooling"
            )
        if self.config.gbm_offset != defaults.gbm_offset:
            parts.append("--gbm-offset" if self.config.gbm_offset else "--no-gbm-offset")
        if self.config.val_albums != defaults.val_albums:
            parts.append(f"--val-albums {self.config.val_albums}")
        if self.config.origin_offset != defaults.origin_offset:
            parts.append(f"--origin-offset {self.config.origin_offset}")
        if self.config.calibration_intervals != defaults.calibration_intervals:
            interval_str = ",".join(f"{p:.4g}" for p in self.config.calibration_intervals)
            parts.append(f"--calibration-intervals {interval_str}")
        if self.config.coverage_tolerance != defaults.coverage_tolerance:
            parts.append(f"--coverage-tolerance {self.config.coverage_tolerance}")
        if self.config.prediction_interval != defaults.prediction_interval:
            parts.append(f"--prediction-interval {self.config.prediction_interval}")
        if not self.config.evaluate_secondary_split:
            parts.append("--no-secondary-split")
        if self.config.dataset is not None:
            parts.append(f"--dataset {self.config.dataset}")
        if self.config.tag is not None:
            parts.append(f"--tag {self.config.tag}")

        return " ".join(parts)

    def _execute_stages(self, stages: list[PipelineStage]) -> None:
        """Execute stages with progress display.

        Args:
            stages: List of stages in execution order.
        """
        # Load previous manifest for skip detection
        previous_manifest: RunManifest | None = None
        if self.config.skip_existing and self.run_dir:
            previous_run = resolve_latest(self.output_base)
            if previous_run is not None:
                try:
                    prev_manifest_path = previous_run / "manifest.json"
                    if prev_manifest_path.exists():
                        previous_manifest = load_run_manifest(prev_manifest_path)
                except OSError as e:
                    log.debug("could_not_load_previous_manifest", error=str(e), exc_info=True)
                except Exception as e:
                    log.debug("could_not_load_previous_manifest", error=str(e), exc_info=True)

        # Defensive: a latest pointer written by an older checkout may still
        # target a dry run, whose recorded hashes cover nothing on disk.
        if previous_manifest is not None and previous_manifest.flags.get("dry_run"):
            log.debug("skip_existing_ignores_dry_run_manifest", run_id=previous_manifest.run_id)
            previous_manifest = None

        # Progress weighting reads durations before the flag-change reset below:
        # a config change invalidates skip detection, not how long stages take.
        previous_durations: dict[str, float] = (
            dict(previous_manifest.stage_durations or {}) if previous_manifest else {}
        )

        if previous_manifest is not None:
            changed_flags = self._skip_flag_differences(previous_manifest)
            if changed_flags:
                log.info(
                    "skip_existing_disabled_due_flag_change",
                    previous_run_id=previous_manifest.run_id,
                    n_changed=len(changed_flags),
                    changed_flags=changed_flags[:10],
                )
                previous_manifest = None

        # Set up progress display; stages advance by predicted duration, not by
        # count, so a 6-hour train doesn't move the bar like a 4-second stage.
        weights = self._stage_weights(stages, previous_durations)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not is_interactive(),
        ) as progress:
            task_id = progress.add_task("Pipeline", total=sum(weights.values()))

            for stage in stages:
                progress.update(task_id, description=f"[cyan]{stage.name}")

                # Check if this stage was already completed (for resume)
                if self.manifest and stage.name in self.manifest.stages_completed:
                    log.info(
                        "stage_already_completed",
                        stage=stage.name,
                    )
                    progress.advance(task_id, weights[stage.name])
                    continue

                # Check if stage should be skipped
                if self.config.skip_existing and not self.config.dry_run:
                    if stage.should_skip(previous_manifest, force=False):
                        log.info(
                            "stage_skipped",
                            stage=stage.name,
                            reason="inputs unchanged",
                        )
                        if self.manifest:
                            self.manifest.stages_skipped.append(stage.name)
                            save_run_manifest(self.manifest, self.run_dir)
                        progress.advance(task_id, weights[stage.name])
                        continue

                # Execute stage
                self._execute_stage(stage)
                progress.advance(task_id, weights[stage.name])

    _FALLBACK_STAGE_SECONDS = 30.0

    def _stage_weights(
        self, stages: list[PipelineStage], previous_durations: dict[str, float]
    ) -> dict[str, float]:
        """Predicted seconds per stage for progress weighting (presentation only).

        Train comes from the runtime predictor when prepared features exist
        (config-aware: it prices the sampler settings this run will use), other
        stages from the previous manifest's durations. With no history at all,
        degrade to equal weights — a bar with made-up proportions is worse than
        a stage-counted one.
        """
        train_predicted = self._predicted_train_seconds() if "train" in (
            s.name for s in stages
        ) else None
        if not previous_durations and train_predicted is None:
            return {s.name: 1.0 for s in stages}
        weights: dict[str, float] = {}
        for s in stages:
            w: float | None = previous_durations.get(s.name)
            if s.name == "train" and train_predicted is not None:
                w = train_predicted
            weights[s.name] = float(w) if w and w > 0 else self._FALLBACK_STAGE_SECONDS
        return weights

    def _predicted_train_seconds(self) -> float | None:
        """Runtime-predictor train weight; None without prepared features to size from."""
        features = Path("data/features/train_features.parquet")
        if not features.exists():
            return None
        try:
            import pandas as pd

            from panelcast.gpu_memory.runtime_predictor import predict_fit_seconds

            n_obs = int(len(pd.read_parquet(features, columns=[])))
            return predict_fit_seconds(
                self.config.num_chains,
                self.config.num_samples,
                self.config.num_warmup,
                n_obs,
                transform=self.config.target_transform,
            ).seconds
        except Exception:
            return None

    # Run-scoped product roots: the stages that write each root and the stages
    # that read it. A root stays in the current run dir when one of its writers
    # is part of this invocation; otherwise consumers read it from the most
    # recent successful run that produced it.
    PRODUCT_WRITERS: dict[str, tuple[str, ...]] = {
        "models": ("train",),
        "evaluation": ("evaluate",),
        "predictions": ("predict",),
        "reports": ("report", "sensitivity"),
    }
    PRODUCT_READERS: dict[str, tuple[str, ...]] = {
        "models": ("evaluate", "predict", "sensitivity"),
        "evaluation": ("report",),
        "predictions": ("report",),
    }

    def _artifact_paths(self) -> ArtifactPaths:
        """Artifact roots for this invocation; flat layout before a run dir exists."""
        if self.run_dir is None:
            return ArtifactPaths.flat()
        if self._resolved_paths is None:
            self._resolved_paths = self._resolve_artifact_paths()
        return self._resolved_paths

    def _resolve_artifact_paths(self) -> ArtifactPaths:
        """Run-scoped roots, with read roots redirected for consumer-only runs.

        A ``--stages`` selection that excludes a product's writer would
        otherwise look for that product in the just-created (empty) run dir.
        Each such root that a selected stage reads resolves to the most recent
        successful run that produced it; a producer present in the stage list
        wins over latest-run resolution, so ``--stages evaluate,report`` reads
        evaluate's fresh output. Writes always target the current run dir.
        """
        current = ArtifactPaths.for_run(self.run_dir)
        if self.config.stages is None:
            return current  # full run: every product is produced here
        selected = set(self.config.stages)
        overrides: dict[str, Path] = {}
        for product, writers in self.PRODUCT_WRITERS.items():
            if selected.intersection(writers):
                continue
            readers = selected.intersection(self.PRODUCT_READERS.get(product, ()))
            if not readers:
                continue
            source = self._find_run_with_product(product, writers)
            if source is None:
                if self.config.dry_run:
                    # A dry run only previews the plan; a missing source is
                    # worth a warning, not a failure.
                    log.warning(
                        "artifact_root_unresolved",
                        product=product,
                        readers=sorted(readers),
                    )
                    continue
                raise PipelineError(
                    f"Stage(s) {sorted(readers)} read '{product}' artifacts, but this "
                    f"invocation does not run {list(writers)} and no previous "
                    f"successful run under {self.output_base} contains '{product}'. "
                    f"Run `panelcast stage {writers[0]}` (or a full `panelcast run`) "
                    "first.",
                    stage="setup",
                )
            overrides[product] = source / product
            log.info(
                "artifact_root_from_previous_run",
                product=product,
                source_run=source.name,
                readers=sorted(readers),
            )
        return dataclass_replace(current, **overrides) if overrides else current

    def _find_run_with_product(self, product: str, writers: tuple[str, ...]) -> Path | None:
        """Most recent successful non-dry run whose dir contains ``product``."""
        try:
            candidates = sorted(
                (p for p in self.output_base.iterdir() if p.is_dir()), reverse=True
            )
        except OSError:
            return None
        for run_dir in candidates:
            if run_dir == self.run_dir or run_dir.name in ("latest", "failed"):
                continue
            try:
                manifest = load_run_manifest(run_dir / "manifest.json")
            except Exception:
                continue
            if not manifest.success or manifest.flags.get("dry_run"):
                continue
            if not any(w in manifest.stages_completed for w in writers):
                continue
            if (run_dir / product).is_dir():
                return run_dir
        return None

    def _create_stage_context(self) -> StageContext:
        """Create StageContext for stage execution.

        Returns:
            StageContext with current configuration.
        """
        return StageContext(
            run_dir=self.run_dir or Path("outputs"),
            paths=self._artifact_paths(),
            seed=self.config.seed,
            strict=self.config.strict,
            verbose=self.config.verbose,
            progress_bar=self.config.progress_bar,
            manifest=self.manifest,
            max_albums=self.config.max_albums,
            # MCMC configuration
            num_chains=self.config.num_chains,
            num_samples=self.config.num_samples,
            num_warmup=self.config.num_warmup,
            target_accept=self.config.target_accept,
            max_tree_depth=self.config.max_tree_depth,
            chain_method=self.config.chain_method,
            # Convergence thresholds
            rhat_threshold=self.config.rhat_threshold,
            ess_threshold=self.config.ess_threshold,
            allow_divergences=self.config.allow_divergences,
            # Data filtering
            min_ratings=self.config.min_ratings,
            min_albums_filter=self.config.min_albums_filter,
            # Feature flags
            enable_genre=self.config.enable_genre,
            enable_artist=self.config.enable_artist,
            enable_temporal=self.config.enable_temporal,
            # Heteroscedastic noise configuration
            n_exponent=self.config.n_exponent,
            learn_n_exponent=self.config.learn_n_exponent,
            n_exponent_alpha=self.config.n_exponent_alpha,
            n_exponent_beta=self.config.n_exponent_beta,
            n_exponent_prior=self.config.n_exponent_prior,
            likelihood_df=self.config.likelihood_df,
            likelihood_family=self.config.likelihood_family,
            discretize_observation=self.config.discretize_observation,
            debut_prev_score_source=self.config.debut_prev_score_source,
            target_transform=self.config.target_transform,
            logit_offset=self.config.logit_offset,
            ar_center=self.config.ar_center,
            latent_process=self.config.latent_process,
            sigma_obs_prior_type=self.config.sigma_obs_prior_type,
            beta_prior_type=self.config.beta_prior_type,
            hs_global_scale=self.config.hs_global_scale,
            heteroscedastic_entity_obs=self.config.heteroscedastic_entity_obs,
            tau_entity_scale=self.config.tau_entity_scale,
            errors_in_variables=self.config.errors_in_variables,
            propagate_rw_horizon=self.config.propagate_rw_horizon,
            entity_group_pooling=self.config.entity_group_pooling,
            gbm_offset=self.config.gbm_offset,
            exclude_rw_raw_from_collection=self.config.exclude_rw_raw_from_collection,
            warmup_export_path=self.config.warmup_export_path,
            warmup_import_path=self.config.warmup_import_path,
            val_albums=self.config.val_albums,
            origin_offset=self.config.origin_offset,
            conformal_calibration=self.config.conformal_calibration,
            min_train_albums=self.config.min_train_albums,
            calibration_intervals=self.config.calibration_intervals,
            coverage_tolerance=self.config.coverage_tolerance,
            prediction_interval=self.config.prediction_interval,
            evaluate_secondary_split=self.config.evaluate_secondary_split,
            predictive_batch_size=self.config.predictive_batch_size,
            predict_artist_batch_size=self.config.predict_artist_batch_size,
            descriptor=self.descriptor,
        )

    def _observe_data_stamps(self) -> None:
        """Record on-disk data-root stamps this run hasn't produced or seen yet.

        Covers consumer-only runs (``--stages train,evaluate``) and skipped
        data stages: the first consumer pins the world it starts from, so a
        later consumer in the same run detects a foreign regeneration.
        """
        if self.manifest is None:
            return
        for stage_name, root in DATA_STAGE_ROOTS.items():
            if stage_name in self.manifest.data_stamps:
                continue
            current = read_stamp(root)
            if current is not None:
                self.manifest.data_stamps[stage_name] = current

    def _capture_stage_input_hashes(self, stage: PipelineStage) -> dict[str, str]:
        """Capture per-path input hashes for manifest provenance."""
        hashes: dict[str, str] = {}
        for path in stage.input_paths:
            if not path.exists():
                continue
            try:
                hashes[str(path)] = sha256_path(path)
            except Exception as e:
                log.warning(
                    "input_hash_failed",
                    stage=stage.name,
                    path=str(path),
                    error=str(e),
                )
        return hashes

    def _record_stage_outputs(
        self,
        stage: PipelineStage,
        run_result: Any | None,
    ) -> None:
        """Record stage outputs (and their content hashes) in the manifest."""
        if self.manifest is None:
            return

        recorded: dict[str, str] = {}
        # Static stage output declarations
        for output_path in stage.output_paths:
            if output_path.exists():
                recorded[f"{stage.name}:{output_path.as_posix()}"] = str(output_path)

        # Dynamic run_fn result paths
        if isinstance(run_result, dict):
            for key, value in run_result.items():
                if isinstance(value, (str, Path)):
                    candidate = Path(value)
                    if candidate.exists():
                        recorded[f"{stage.name}:{key}"] = str(candidate)

        self.manifest.outputs.update(recorded)
        hash_started = time()
        for manifest_key, path_str in recorded.items():
            try:
                self.manifest.output_hashes[manifest_key] = sha256_path(path_str)
            except OSError as e:
                log.debug("output_hash_failed", key=manifest_key, error=str(e))
        if recorded:
            log.debug(
                "outputs_hashed",
                stage=stage.name,
                n=len(recorded),
                seconds=round(time() - hash_started, 3),
            )

    def _execute_stage(self, stage: PipelineStage) -> None:
        """Execute a single pipeline stage.

        Args:
            stage: Stage to execute.
        """
        log.info("stage_started", stage=stage.name, description=stage.description)

        if self.manifest:
            self.manifest.input_hashes.update(self._capture_stage_input_hashes(stage))

        if self.config.dry_run:
            # Record nothing beyond the plan: completed stages, stage hashes,
            # or outputs from a run that executed nothing would poison
            # --skip-existing and latest-run resolution with stale state.
            log.info("stage_dry_run", stage=stage.name, would_run=stage.description)
            return

        if stage.name in CONSUMER_STAGES and self.manifest is not None:
            self._observe_data_stamps()
            verify_stamps(self.manifest.data_stamps, stage.name)

        # Create stage context
        ctx = self._create_stage_context()

        stage_started = time()

        # Execute the stage's run function
        if stage.run_fn is None:
            log.warning(
                "stage_no_run_fn",
                stage=stage.name,
                message="Stage has no run function defined",
            )
            run_result = None
        else:
            try:
                run_result = stage.run_fn(ctx)
            except StageSkipped as e:
                log.info("stage_skipped", stage=stage.name, reason=e.message)
                if self.manifest:
                    self.manifest.stages_skipped.append(stage.name)
                    save_run_manifest(self.manifest, self.run_dir)
                return
            except ConvergenceError as e:
                # Handle convergence errors: fail in strict mode, warn otherwise
                if self.config.strict:
                    raise
                log.warning(
                    "convergence_warning",
                    stage=stage.name,
                    error=str(e),
                    message="Continuing despite convergence issues (strict=False)",
                )
                # The fit raised before binding run_result; leave it unset so the
                # manifest update below treats the stage like a no-result run
                # instead of raising UnboundLocalError (defeating the "continue").
                run_result = None
            except PipelineError:
                raise
            except Exception as e:
                # Wrap unexpected errors
                raise PipelineError(str(e), stage=stage.name) from e

        # Update manifest
        if self.manifest:
            self.manifest.stages_completed.append(stage.name)
            self.manifest.stage_hashes[stage.name] = stage.compute_input_hash()
            self.manifest.stage_durations[stage.name] = round(time() - stage_started, 3)
            if isinstance(run_result, dict) and isinstance(
                run_result.get("resource_usage"), dict
            ):
                self.manifest.resources[stage.name] = run_result["resource_usage"]
            if stage.name in DATA_STAGE_ROOTS:
                self.manifest.data_stamps[stage.name] = write_stamp(
                    DATA_STAGE_ROOTS[stage.name],
                    stage.name,
                    self.manifest.stage_hashes[stage.name],
                    self.manifest.run_id,
                )
            self._record_stage_outputs(stage, run_result=run_result)
            save_run_manifest(self.manifest, self.run_dir)

        log.info(
            "stage_completed",
            stage=stage.name,
            duration_seconds=round(time() - stage_started, 3),
        )

    def _handle_failure(self, error: Exception, stage: str) -> None:
        """Handle pipeline failure with cleanup.

        Args:
            error: The exception that caused failure.
            stage: Name of the stage that failed.
        """
        log.error(
            "pipeline_failed",
            stage=stage,
            error=str(error),
            exc_info=True,
        )

        # Update manifest
        if self.manifest:
            self.manifest.success = False
            self.manifest.error = str(error)
            self.manifest.duration_seconds = time() - self._start_time
            if self.run_dir:
                save_run_manifest(self.manifest, self.run_dir)

        # failure.json is written BEFORE the move so it survives it.
        self._write_failure_payload(error, stage)

        # Close logging handlers before moving directory (Windows file lock issue)
        self._close_log_handlers()

        # Move to failed directory
        final_path = self.run_dir
        if self.run_dir and self.run_dir.exists():
            failed_dir = self.output_base / "failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / self.run_dir.name

            # Remove existing failed dir if present
            if failed_path.exists():
                shutil.rmtree(failed_path)

            try:
                shutil.move(str(self.run_dir), str(failed_path))
                final_path = failed_path
                log.info("run_moved_to_failed", path=str(failed_path))
            except PermissionError as e:
                # On Windows, file locks can persist; log but don't fail
                log.warning(
                    "failed_to_move_to_failed",
                    error=str(e),
                    run_dir=str(self.run_dir),
                )

        self._print_failure_epilogue(error, stage, final_path)

    def _write_failure_payload(self, error: Exception, stage: str) -> None:
        """Structured forensics for `runs why`; must never raise."""
        if self.run_dir is None or not self.run_dir.exists():
            return
        import traceback

        from panelcast.pipelines.errors import failure_hint
        from panelcast.utils.logging import recent_events

        try:
            payload = {
                "run_id": self.manifest.run_id if self.manifest else None,
                "stage": stage,
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback_tail": traceback.format_exception(error)[-8:],
                "stages_completed": (
                    list(self.manifest.stages_completed) if self.manifest else []
                ),
                "hint": failure_hint(error),
                "resume_command": f"panelcast run --resume {self.run_dir.name}",
                "recent_events": recent_events(),
            }
            (self.run_dir / "failure.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:  # forensics must never mask the real failure
            log.debug("failure_payload_write_failed", error=str(e))

    def _print_failure_epilogue(self, error: Exception, stage: str, final_path) -> None:
        """The 10-second answer to 'what happened and what do I type next'."""
        from rich.console import Console

        from panelcast.pipelines.errors import failure_hint

        console = Console(stderr=True)
        run_name = final_path.name if final_path is not None else "unknown"
        console.print(f"\n[red bold]{stage} failed:[/] {type(error).__name__}: {error}")
        if final_path is not None:
            console.print(f"run moved to: {final_path}")
        console.print(f"resume with:  panelcast run --resume {run_name}")
        hint = failure_hint(error)
        if hint:
            console.print(f"hint:         {hint}")
        console.print(f"details:      panelcast runs why {run_name}")

    def _close_log_handlers(self) -> None:
        """Close file handlers to release locks (needed for Windows).

        On Windows, file handlers keep files locked which prevents moving
        directories containing log files. This closes all handlers on the
        root logger to release those locks.
        """
        root_logger = logging.getLogger()
        handlers_to_remove = []

        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                handlers_to_remove.append(handler)

        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)

    def _finalize_success(self) -> None:
        """Finalize successful run with manifest update and latest pointer."""
        # Update manifest
        if self.manifest:
            self.manifest.success = True
            self.manifest.duration_seconds = time() - self._start_time
            if self.run_dir:
                save_run_manifest(self.manifest, self.run_dir)

        # latest.json is the authoritative pointer; the link is opportunistic
        # convenience (symlink/junction creation can fail on Windows/NTFS).
        # Dry runs never take the pointer: their run dir holds no artifacts,
        # so latest-run consumers (diagnose, compare, dashboards) would
        # resolve to an empty run.
        if self.run_dir and not self.config.dry_run:
            self._write_latest_pointer()
            self._create_latest_link()

        log.info(
            "pipeline_completed",
            run_id=self.manifest.run_id if self.manifest else "unknown",
            duration=f"{self.manifest.duration_seconds:.2f}s" if self.manifest else "unknown",
            stages_completed=len(self.manifest.stages_completed) if self.manifest else 0,
            stages_skipped=len(self.manifest.stages_skipped) if self.manifest else 0,
        )

    def _write_latest_pointer(self) -> None:
        """Atomically write outputs/latest.json pointing at the current run."""
        if not self.run_dir:
            return
        try:
            run_dir_rel = self.run_dir.relative_to(self.output_base)
        except ValueError:
            run_dir_rel = Path(self.run_dir.name)
        payload = {
            "run_id": self.manifest.run_id if self.manifest else self.run_dir.name,
            "run_dir": run_dir_rel.as_posix(),
        }
        pointer = self.output_base / "latest.json"
        tmp = self.output_base / "latest.json.tmp"
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, pointer)
        except OSError as e:
            log.warning("failed_to_write_latest_pointer", error=str(e))

    def _create_latest_link(self) -> None:
        """Create outputs/latest symlink/junction to current run."""
        if not self.run_dir:
            return

        latest_link = self.output_base / "latest"

        # Remove existing link/junction
        # Wrap exists()/is_symlink() in try/except for WSL where broken
        # symlinks on NTFS can raise OSError (WinError 1920).
        try:
            should_remove = latest_link.exists() or latest_link.is_symlink()
        except OSError:
            should_remove = True
        if should_remove:
            try:
                if sys.platform == "win32" and latest_link.is_dir():
                    # On Windows, junctions appear as directories
                    os.rmdir(latest_link)
                else:
                    latest_link.unlink()
            except Exception as e:
                log.warning("failed_to_remove_latest_link", error=str(e))
                return

        # Create new link
        try:
            if sys.platform == "win32":
                # Try symlink first (requires Developer Mode or admin)
                try:
                    os.symlink(self.run_dir, latest_link, target_is_directory=True)
                    log.debug("created_symlink", target=str(self.run_dir))
                except OSError:
                    # Fall back to directory junction (no special permissions)
                    # Validate paths don't contain shell metacharacters
                    link_str = str(latest_link)
                    target_str = str(self.run_dir)
                    if any(c in link_str + target_str for c in "&|;<>`$^%\r\n"):
                        log.warning("unsafe_path_characters", link=link_str, target=target_str)
                        return
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", link_str, target_str],
                        capture_output=True,
                        check=True,
                    )
                    log.debug("created_junction", target=target_str)
            else:
                os.symlink(self.run_dir, latest_link, target_is_directory=True)
                log.debug("created_symlink", target=str(self.run_dir))
        except Exception as e:
            log.warning("failed_to_create_latest_link", error=str(e))


def run_pipeline(config: PipelineConfig, output_base: Path | str = Path("outputs")) -> int:
    """Convenience function to run pipeline with given configuration.

    Creates an orchestrator and runs the pipeline, returning the exit code.

    Args:
        config: Pipeline configuration.
        output_base: Base directory for outputs (default "outputs").

    Returns:
        Exit code: 0 on success, error's exit_code on failure.

    Example:
        >>> exit_code = run_pipeline(PipelineConfig(seed=42))
    """
    orchestrator = PipelineOrchestrator(config, output_base=output_base)
    return orchestrator.run()
