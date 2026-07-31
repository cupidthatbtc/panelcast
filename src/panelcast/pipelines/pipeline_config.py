"""Pipeline configuration: the resolved experiment a run executes.

Holds :class:`PipelineConfig` and its validation, the descriptor-owned
model-fact resolution (:func:`resolve_model_facts`), and the field sets the
orchestrator uses for resume restoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

from panelcast.config.gates import (
    ArCenter,
    ArtistEffectParam,
    BetaPriorType,
    ChainMethod,
    DebutPrevScoreSource,
    InitStrategy,
    LatentProcess,
    NExponentPrior,
    SigmaObsPriorType,
)
from panelcast.paths import validate_run_id

# One-directional by design: training_summary owns the recorded-artifact
# contract and must never import config, so the write and read paths can share
# the offset domain without a cycle.
from panelcast.pipelines.training_summary import (
    DEFAULT_LOGIT_OFFSET,
    coerce_logit_offset,
    coerce_target_transform,
)

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


def _field_defaults() -> dict[str, Any]:
    """Shipped defaults, read off the fields rather than an instance.

    `__post_init__` needs these, and `_get_default_config()` would re-enter it.
    """
    return {f.name: f.default for f in dataclass_fields(PipelineConfig)}


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
        max_events: Maximum events per entity for model training (default 50).
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
        min_events_filter: Minimum events per entity for dynamic effects (default 2).
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
    # Caller-supplied run directory name (#167): lets the select runner name
    # each arm's run up front instead of racing the mutable `latest` pointer.
    # None (default) keeps the generated timestamp ids. No CLI flag.
    run_id: str | None = None
    # None resolves to the descriptor's max_events, else 50 (#268).
    max_events: int | None = None
    # MCMC configuration
    num_chains: int = 4
    num_samples: int = 1000
    num_warmup: int = 1000
    target_accept: float = 0.90
    max_tree_depth: int = 10
    # NUTS initialization strategy: "uniform" (legacy default) | "median" |
    # "feasible". No CLI flag; via run_config.yaml. External domains whose chains
    # trap in a degenerate init corner (e.g. the baseball beta_binomial
    # replication) switch to "median"/"feasible" here.
    init_strategy: InitStrategy = "uniform"
    chain_method: ChainMethod = "sequential"
    checkpoint_every_draws: int | None = None
    caged_chain_retries: int = 0
    caged_chain_tree_depth_fraction: float = 0.95
    caged_chain_boundary_sigma: float = 0.005
    caged_chain_consensus_ratio: float = 5.0
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
    min_events_filter: int = 2
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
    # Builtins are the LikelihoodFamily Literal; entry-point plugin families
    # (#172) are also legal, so the boundary widens to str and the runtime
    # registry check in _validate_likelihood is the contract.
    # None resolves to the descriptor's likelihood_family, else "studentt".
    likelihood_family: str | None = None
    # Discretization gate: interval-censor the observation to integers (default
    # off => continuous likelihood). Location-scale families only; not for beta.
    discretize_observation: bool = False
    # Debut prev_score fill source: "train_mean" | "dataset_stats" (legacy)
    debut_prev_score_source: DebutPrevScoreSource = "train_mean"
    # Target transform gate: "offset_logit" (default since 0.5.0 — promoted on
    # the corrected #63 ledger, +22 held-out elpd) | "identity" (former default)
    # None resolves to the descriptor's target_transform, else "offset_logit"
    # (the default since 0.5.0 — promoted on the corrected #63 ledger).
    target_transform: str | None = None
    logit_offset: float = DEFAULT_LOGIT_OFFSET
    # AR(1) centering gate: "global" | "none" (legacy) | "artist_running"
    ar_center: ArCenter = "global"
    # Latent artist-effect process gate: "rw" (legacy) | "ar1" (experimental)
    latent_process: LatentProcess = "rw"
    # sigma_obs prior family gate: "halfnormal" (legacy default) | "lognormal"
    # (removes the zero-boundary pile-up behind the econ variance-collapse).
    sigma_obs_prior_type: SigmaObsPriorType = "halfnormal"
    # sigma_artist prior family gate: "halfnormal" (legacy default) | "lognormal"
    # (removes the zero-boundary pile-up; mirrors sigma_rw/sigma_obs). No CLI flag.
    sigma_artist_prior_type: SigmaObsPriorType = "halfnormal"
    # Artist-effect parameterization: "noncentered" (legacy default) | "zerosum"
    # (ZeroSumNormal deviations around mu_artist — removes the mu_artist<->effects
    # location ridge that throttles sigma_artist ESS). No CLI flag.
    artist_effect_param: ArtistEffectParam = "noncentered"
    # LogNormal(loc, sigma) parameters for the sigma_rw / sigma_artist priors
    # (used only when the respective *_prior_type is "lognormal"). The default
    # locations are sized for the AOTY score scale; external domains on other
    # scales (e.g. the baseball beta_binomial replication, sigma ~1e-2/1e-3)
    # right-size them here to avoid prior-likelihood conflict. No CLI flag.
    sigma_rw_lognormal_loc: float = -2.8
    sigma_rw_lognormal_sigma: float = 0.6
    sigma_artist_lognormal_loc: float = -0.9
    sigma_artist_lognormal_sigma: float = 0.6
    # Normal(loc, scale) parameters for the AR(1) coefficient prior. The default
    # centers rho at zero with moderate spread; external domains where the AR
    # term competes with the artist effects as an alternative persistence channel
    # (e.g. the baseball replication's previous-season average) set rho_scale
    # small to pin rho near zero and disable the channel. No CLI flag.
    rho_loc: float = 0.0
    rho_scale: float = 0.3
    # Covariate-block prior gate (#155): "normal" (legacy default, bit-identical
    # RNG path) | "horseshoe" (regularized horseshoe; global-local shrinkage
    # against the #76 coefficient dilution). No CLI flag; via run_config.yaml.
    beta_prior_type: BetaPriorType = "normal"
    # Horseshoe global scale (tau_0), the sparsity knob a bake-off sweeps.
    # Read only when beta_prior_type="horseshoe".
    hs_global_scale: float = 0.1
    # Entity-level observation overdispersion gate: True (AOTY default since
    # 0.13.0, #238 — per-entity multiplicative noise inflation, held-out ELPD
    # +29.8+/-7.0) | False (legacy bit-identical RNG path, pinned by IMDb/econ).
    # tau_entity_scale sets the prior HalfNormal scale on the entity dispersion.
    heteroscedastic_entity_obs: bool = True
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
    # Per-group entity-effect variances (#271): each entity group gets its own
    # sigma_artist via log-scale partial pooling around the shared draw.
    # "shared" (default) is the legacy bit-identical path. Requires
    # entity_group_pooling (checked against the resolved gate at train time).
    # No CLI flag.
    group_variance: str = "shared"
    tau_group_sigma_scale: float = 0.3
    # Latent-population shape seam (#232): "normal" (legacy bit-identical) |
    # "skew_normal" (learned-alpha skew-normal population on the initial
    # entity effects — the skewness-pin candidate). Requires
    # artist_effect_param="noncentered". No CLI flag.
    entity_effect_prior_type: str = "normal"
    entity_skew_alpha_scale: float = 2.0
    # Innovation-shape seam (#233): "normal" (legacy bit-identical) |
    # "skew_normal" (learned-alpha skew-normal random-walk innovations — the
    # skewness-via-dynamics candidate; doubles the dominant latent tensor).
    # No CLI flag.
    rw_innovation_type: str = "normal"
    rw_skew_alpha_scale: float = 2.0
    # Boundary censoring (#234): bound observations contribute CDF mass, the
    # max-pin candidate. Mutually exclusive with discretize_observation.
    # No CLI flag.
    censor_at_bounds: bool = False
    # Period (calendar-time) effects gate (#269): a constrained additive
    # offset per period_col value. The declared constraint (zero_sum, or
    # pin_first / pin_last) is what identifies the block against entity
    # intercepts + cohorts + age-like covariates (the APC rank deficiency).
    # Default off => legacy bit-identical path. Requires the descriptor to
    # name a period_col. No CLI flag.
    period_effects: bool = False
    period_constraint: str = "zero_sum"
    sigma_period_scale: float = 0.5
    # Missing-covariate treatment gate (#158): train-median imputation plus
    # <col>__missing indicator columns instead of the legacy fillna(0).
    # Feature-affecting; default off => byte-identical outputs. No CLI flag.
    impute_missing: bool = False
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
    # Split configuration. min_train_events matches the documented `run` CLI
    # default (2) so `stage splits` / `demo` build the same split population.
    val_events: int = 0
    min_train_events: int = 2
    # Rolling-origin backtest offset (0 = the standard split)
    origin_offset: int = 0
    # Conformal calibration wrapper on the predictive (#156; needs val_events >= 1)
    conformal_calibration: bool = False
    # Multi-step ancestral rollout depth for evaluation (#157). 0 = off (the
    # default; byte-identical). H > 0 scores h=1..H forecasts into the
    # separate horizon_rollout.json artifact. No CLI flag.
    eval_horizon: int = 0
    # Evaluation configuration
    calibration_intervals: tuple[float, ...] = (0.80, 0.95)
    coverage_tolerance: float = 0.03
    prediction_interval: float = 0.95
    evaluate_secondary_split: bool = True
    # Prediction batching (memory/speed trade-off, not statistically relevant)
    predictive_batch_size: int = 500
    predict_entity_batch_size: int = 50
    # priors: auto (#267): derive sigma lognormal locs from train-data moments
    # at fit time. None resolves to the descriptor's auto_priors, else False.
    auto_priors: bool | None = None
    # Dataset descriptor reference (bare name or YAML path; None = AOTY defaults)
    dataset: str | None = None
    # YAML keys ignored under --allow-unknown-config-keys (#297). Provenance
    # only: preserved in the run manifest, never applied, never dumped into
    # resolved_config.yaml.
    unknown_config_keys: dict[str, Any] = field(default_factory=dict)

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
        self._validate_run_id()
        if self.max_events is not None and self.max_events < 1:
            raise ValueError(f"Invalid max_events: {self.max_events}. Must be >= 1.")
        if not 5 <= self.max_tree_depth <= 15:
            raise ValueError(
                f"Invalid max_tree_depth: {self.max_tree_depth}. Must be between 5 and 15."
            )
        if len(self.calibration_intervals) == 0:
            raise ValueError("calibration_intervals must contain at least one probability level.")
        for prob in self.calibration_intervals:
            if not 0.0 < prob < 1.0:
                raise ValueError(f"Invalid calibration interval {prob}. Must be in (0, 1).")
        if self.target_transform is not None:
            self.target_transform = coerce_target_transform(
                self.target_transform, context="the run config"
            )
        self._validate_logit_offset()
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
        if self.sigma_artist_prior_type not in ("halfnormal", "lognormal"):
            raise ValueError(
                f"Invalid sigma_artist_prior_type: '{self.sigma_artist_prior_type}'. "
                "Must be 'halfnormal' or 'lognormal'."
            )
        if self.artist_effect_param not in ("noncentered", "zerosum"):
            raise ValueError(
                f"Invalid artist_effect_param: '{self.artist_effect_param}'. "
                "Must be 'noncentered' or 'zerosum'."
            )
        if self.init_strategy not in ("uniform", "median", "feasible"):
            raise ValueError(
                f"Invalid init_strategy: '{self.init_strategy}'. "
                "Must be 'uniform', 'median', or 'feasible'."
            )
        if self.beta_prior_type not in ("normal", "horseshoe"):
            raise ValueError(
                f"Invalid beta_prior_type: '{self.beta_prior_type}'. "
                "Must be 'normal' or 'horseshoe'."
            )
        self._validate_structural_gates()
        for scale_field in (
            "sigma_rw_lognormal_sigma",
            "sigma_artist_lognormal_sigma",
            "rho_scale",
            "hs_global_scale",
            "tau_entity_scale",
            "sigma_period_scale",
            "tau_group_sigma_scale",
            "entity_skew_alpha_scale",
            "rw_skew_alpha_scale",
        ):
            value = getattr(self, scale_field)
            if value <= 0.0:
                raise ValueError(f"Invalid {scale_field}: {value}. Must be > 0.")
        self._validate_auto_priors()
        if self.coverage_tolerance < 0.0:
            raise ValueError("coverage_tolerance must be >= 0.")
        if not 0.0 < self.prediction_interval < 1.0:
            raise ValueError("prediction_interval must be in (0, 1).")
        if self.eval_horizon < 0:
            raise ValueError(f"eval_horizon must be >= 0, got {self.eval_horizon}.")
        self._validate_sampling()

    def _validate_sampling(self) -> None:
        """Validate sampler settings and strict-mode requirements."""
        if self.num_chains < 1:
            raise ValueError("num_chains must be >= 1.")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if self.num_warmup < 0:
            raise ValueError("num_warmup must be >= 0.")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must be in (0, 1).")
        if self.checkpoint_every_draws is not None and self.checkpoint_every_draws < 1:
            raise ValueError("checkpoint_every_draws must be >= 1 when set.")
        if type(self.caged_chain_retries) is not int:
            raise ValueError("caged_chain_retries must be an integer.")
        if not 0 <= self.caged_chain_retries <= 10:
            raise ValueError("caged_chain_retries must be between 0 and 10.")
        for name, value in (
            ("caged_chain_tree_depth_fraction", self.caged_chain_tree_depth_fraction),
            ("caged_chain_boundary_sigma", self.caged_chain_boundary_sigma),
            ("caged_chain_consensus_ratio", self.caged_chain_consensus_ratio),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite number.")
            try:
                finite = math.isfinite(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a finite number.") from exc
            if not finite:
                raise ValueError(f"{name} must be a finite number.")
        if not 0.0 < self.caged_chain_tree_depth_fraction <= 1.0:
            raise ValueError("caged_chain_tree_depth_fraction must be in (0, 1].")
        if self.caged_chain_boundary_sigma <= 0.0:
            raise ValueError("caged_chain_boundary_sigma must be > 0.")
        if self.caged_chain_consensus_ratio <= 1.0:
            raise ValueError("caged_chain_consensus_ratio must be > 1.")
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

    def _validate_run_id(self) -> None:
        """Caller-supplied run and resume ids must be bare names (#167, #365).

        Resume is validated here too, so a YAML or direct-API id is rejected
        before anything on disk is looked up, moved, or deleted.
        """
        for name in ("run_id", "resume"):
            value = getattr(self, name)
            if value is not None:
                validate_run_id(value, field=name)

    def _validate_structural_gates(self) -> None:
        """Enum and coherence checks for the #269/#271 structural gates."""
        if self.period_constraint not in ("zero_sum", "pin_first", "pin_last"):
            raise ValueError(
                f"Invalid period_constraint: '{self.period_constraint}'. "
                "Must be 'zero_sum', 'pin_first', or 'pin_last'."
            )
        if self.group_variance not in ("shared", "per_group"):
            raise ValueError(
                f"Invalid group_variance: '{self.group_variance}'. "
                "Must be 'shared' or 'per_group'."
            )
        if self.group_variance == "per_group" and self.entity_group_pooling is False:
            raise ValueError(
                "group_variance='per_group' requires entity_group_pooling: "
                "per-group sigmas are indexed by the entity-group mapping."
            )
        if self.entity_effect_prior_type not in ("normal", "skew_normal"):
            raise ValueError(
                f"Invalid entity_effect_prior_type: '{self.entity_effect_prior_type}'. "
                "Must be 'normal' or 'skew_normal'."
            )
        if (
            self.entity_effect_prior_type == "skew_normal"
            and self.artist_effect_param != "noncentered"
        ):
            raise ValueError(
                "entity_effect_prior_type='skew_normal' requires "
                "artist_effect_param='noncentered'."
            )
        if self.rw_innovation_type not in ("normal", "skew_normal"):
            raise ValueError(
                f"Invalid rw_innovation_type: '{self.rw_innovation_type}'. "
                "Must be 'normal' or 'skew_normal'."
            )
        if self.censor_at_bounds and self.discretize_observation:
            raise ValueError(
                "censor_at_bounds and discretize_observation are mutually "
                "exclusive: censoring puts mass at the exact bounds while "
                "discretization interval-censors every integer."
            )

    def _validate_auto_priors(self) -> None:
        """auto_priors derives the sigma locs; explicit values conflict (#267)."""
        if not self.auto_priors:
            return
        defaults = _field_defaults()
        explicit = [
            name
            for name in (
                "sigma_rw_lognormal_loc",
                "sigma_rw_lognormal_sigma",
                "sigma_artist_lognormal_loc",
                "sigma_artist_lognormal_sigma",
            )
            if getattr(self, name) != defaults[name]
        ]
        if explicit:
            raise ValueError(
                f"auto_priors=True derives {', '.join(explicit)} from the "
                "training data; remove the explicit value(s) or turn auto off."
            )

    def _validate_logit_offset(self) -> None:
        """Zero is a supported offset (the plain logit) and is propagated as
        recorded; a negative or non-finite one puts the offset-logit argument
        outside (0, 1) and yields NaN log-likelihoods instead of an error.

        Normalizes rather than only checking: the coerced float is written
        back, so a YAML string never reaches the summary as a string that the
        read-side resolver would then reject after training has already run.
        ``None`` is the unset sentinel on both ends -- ``logit_offset: null``
        in a config resolves to the default here exactly as a recorded null
        does in the resolver. The field stays annotated ``float`` because that
        is what every reader gets: YAML reaches the dataclass through an
        untyped mapping, and this is where that boundary is normalized."""
        if self.logit_offset is None:
            self.logit_offset = DEFAULT_LOGIT_OFFSET
            return
        self.logit_offset = coerce_logit_offset(self.logit_offset, context="the run config")

    def _validate_likelihood(self) -> None:
        """Validate the likelihood family and its structural constraints."""
        from panelcast.models.bayes.likelihoods import all_likelihoods, find_likelihood

        # Family-independent coupling first: it must hold even while the
        # family is an unresolved sentinel (#268).
        if (
            self.discretize_observation
            and self.target_transform is not None
            and self.target_transform != "identity"
        ):
            raise ValueError(
                "discretize_observation=True requires target_transform='identity': "
                "discretization interval-censors integers on the raw score scale, "
                f"but target_transform='{self.target_transform}' moves y off that scale."
            )
        if self.likelihood_family is None:
            # Unresolved sentinel: the orchestrator resolves it from the
            # descriptor and re-validates (#268).
            return
        spec = find_likelihood(self.likelihood_family)
        if spec is None:
            raise ValueError(
                f"Invalid likelihood_family: '{self.likelihood_family}'. "
                f"Must be one of: {', '.join(sorted(all_likelihoods()))}."
            )
        if self.discretize_observation and not spec.supports_discretization:
            supported = [f for f, s in all_likelihoods().items() if s.supports_discretization]
            raise ValueError(
                f"discretize_observation=True is not supported by likelihood_family "
                f"'{self.likelihood_family}'. Supported: {', '.join(supported)}."
            )
        if (
            spec.requires_identity_transform
            and self.target_transform is not None
            and self.target_transform != "identity"
        ):
            raise ValueError(
                f"likelihood_family='{self.likelihood_family}' requires "
                f"target_transform='identity' (got '{self.target_transform}'): "
                "the bounded likelihood assumes mu is on the score scale."
            )
        if spec.samples_bare_phi and self.latent_process == "ar1":
            raise ValueError(
                f"likelihood_family='{self.likelihood_family}' cannot be combined "
                "with latent_process='ar1': both sample a 'phi' site and NUTS "
                "requires unique site names. Use latent_process='rw', or a "
                "different likelihood_family."
            )
        if not spec.uses_sigma:
            # Fire only on knobs moved OFF their shipped default: the point is to
            # catch a request the family would silently ignore, and inheriting a
            # default is not a request. Comparing to the default rather than to
            # truthiness keeps that true if one of these is ever promoted on —
            # the model already no-ops them here (model.py, family_uses_sigma).
            defaults = _field_defaults()
            inert = [
                name
                for name, value in (
                    ("learn_n_exponent", self.learn_n_exponent),
                    ("heteroscedastic_entity_obs", self.heteroscedastic_entity_obs),
                    ("n_exponent", self.n_exponent),
                )
                if value != defaults[name]
            ]
            if inert:
                raise ValueError(
                    f"{', '.join(inert)} cannot be used with likelihood_family="
                    f"'{self.likelihood_family}': the family draws its own precision "
                    "and ignores sigma, so these options would be silently inert."
                )


def resolve_model_facts(config: PipelineConfig, descriptor) -> None:
    """Resolve descriptor-owned model facts onto a config in place (#268).

    These are properties of the data, so an explicit CLI/YAML value wins, the
    descriptor is next, and the historical pipeline defaults are last.
    Re-validates afterwards so family/transform coupling rules see resolved
    values, then enforces descriptor coherence for beta_binomial. Idempotent:
    a config that already ran through resolution passes through unchanged.
    """
    if config.likelihood_family is None:
        config.likelihood_family = (
            descriptor.likelihood_family
            if descriptor.likelihood_family is not None
            else "studentt"
        )
    if config.target_transform is None:
        config.target_transform = (
            descriptor.target_transform
            if descriptor.target_transform is not None
            else "offset_logit"
        )
    if config.max_events is None:
        config.max_events = descriptor.max_events if descriptor.max_events is not None else 50
    if config.auto_priors is None:
        config.auto_priors = (
            descriptor.auto_priors if descriptor.auto_priors is not None else False
        )
    config._validate()
    # beta_binomial models the target as the mean of n aggregated ratings, so
    # it only makes sense when n_obs_col is a true count of independent raters.
    if config.likelihood_family == "beta_binomial" and not descriptor.n_obs_is_aggregation_count:
        raise ValueError(
            "likelihood_family='beta_binomial' models the target as the mean of "
            f"n={descriptor.n_obs_col} aggregated ratings, but descriptor "
            f"'{descriptor.name}' sets n_obs_is_aggregation_count=false "
            f"({descriptor.n_obs_col} is not a count of independent raters). "
            "Use an aggregation-count domain or a different likelihood_family."
        )
    if config.period_effects and descriptor.period_col is None:
        raise ValueError(
            "period_effects=true requires the dataset descriptor to declare "
            f"period_col (descriptor '{descriptor.name}' does not) — the block "
            "needs to know which column holds calendar time."
        )


# Execution mechanics and per-invocation provenance excluded from resume
# restore (#296); everything else on PipelineConfig is restored. strict stays a
# per-invocation gate: it aborts on warnings but never changes outputs, and a
# user resuming with --strict means it.
_RESUME_EXCLUDED_KEYS = frozenset(
    {"resume", "skip_existing", "dry_run", "verbose", "progress_bar",
     "tag", "run_id", "unknown_config_keys", "strict"}
)
_RESUME_CONFIG_KEYS = tuple(
    f.name for f in dataclass_fields(PipelineConfig) if f.name not in _RESUME_EXCLUDED_KEYS
)
