"""Pipeline stage definitions with hash-based skip detection.

This module defines the computational graph of pipeline stages with their
dependencies, input/output paths, and hash-based skip logic for incremental runs.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.paths import ArtifactPaths
from panelcast.utils.hashing import sha256_path

if TYPE_CHECKING:
    from panelcast.pipelines.manifest import RunManifest


@dataclass
class StageContext:
    """Context passed to stage run functions.

    Provides access to run configuration and shared state for stage execution.

    Attributes:
        run_dir: Directory for this pipeline run (outputs/{timestamp}/).
        seed: Random seed for reproducibility.
        strict: If True, fail on convergence warnings.
        verbose: If True, enable verbose logging.
        manifest: Current run manifest for tracking progress.
        max_albums: Maximum albums per artist for model training.
        num_chains: Number of parallel MCMC chains.
        num_samples: Post-warmup samples per chain.
        num_warmup: Warmup iterations per chain.
        target_accept: Target acceptance probability (default 0.90).
        max_tree_depth: Maximum tree depth for NUTS (default 10).
        chain_method: MCMC chain parallelization method ('sequential', 'vectorized', 'parallel').
        rhat_threshold: Maximum acceptable R-hat.
        ess_threshold: Minimum ESS per chain.
        allow_divergences: If True, don't fail on divergences.
        min_ratings: Minimum user ratings per album.
        min_albums_filter: Minimum albums per artist for dynamic effects.
        enable_genre: If False, disable genre features.
        enable_artist: If False, disable artist features.
        enable_temporal: If False, disable temporal features.
        n_exponent: Scaling exponent for review count noise adjustment.
        learn_n_exponent: If True, learn exponent from data using prior.
        n_exponent_alpha: Beta prior alpha parameter for learned exponent.
        n_exponent_beta: Beta prior beta parameter for learned exponent.
        n_exponent_prior: Prior type for learned exponent: 'logit-normal' or 'beta'.
        calibration_intervals: Credible interval levels for calibration checks.
        coverage_tolerance: Allowed absolute calibration error tolerance.
        prediction_interval: Interval level used for saved prediction bands.
        evaluate_secondary_split: Whether to evaluate artist-disjoint split.

    Example:
        >>> ctx = StageContext(
        ...     run_dir=Path("outputs/2026-01-19_143052"),
        ...     seed=42,
        ...     strict=False,
        ...     verbose=True,
        ...     manifest=manifest,
        ...     max_albums=50,
        ... )
    """

    run_dir: Path
    seed: int
    strict: bool
    verbose: bool
    manifest: RunManifest
    # MCMC progress bars: None = auto (stderr TTY only), False = disabled.
    progress_bar: bool | None = None
    max_albums: int = 50
    # MCMC configuration
    num_chains: int = 4
    num_samples: int = 1000
    num_warmup: int = 1000
    target_accept: float = 0.90
    max_tree_depth: int = 10
    # NUTS init strategy: "uniform" (legacy) | "median" | "feasible"
    init_strategy: str = "uniform"
    chain_method: str = "sequential"
    # Blocked-sampling checkpoint cadence; None = single-shot. train_bayes reads
    # this via getattr, so a dropped field silently disables checkpointing.
    checkpoint_every_draws: int | None = None
    # Convergence thresholds
    rhat_threshold: float = 1.01
    ess_threshold: int = 400
    allow_divergences: bool = False
    # Data filtering
    min_ratings: int = 10
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
    n_exponent_prior: str = "logit-normal"
    # Likelihood configuration
    likelihood_df: float = 4.0
    # Likelihood family: "studentt" (legacy) | "normal" | "skew_studentt" |
    # "skew_normal" | "split_normal" | "beta"
    likelihood_family: str = "studentt"
    # Interval-censor the observation to integers (default off => continuous).
    discretize_observation: bool = False
    # Debut prev_score fill source: "train_mean" | "dataset_stats" (legacy)
    debut_prev_score_source: str = "train_mean"
    # Target transform gate: "identity" (legacy) | "offset_logit"
    target_transform: str = "identity"
    logit_offset: float = 0.5
    # AR(1) centering gate: "global" | "none" (legacy) | "artist_running"
    ar_center: str = "global"
    # Latent artist-effect process gate: "rw" (legacy) | "ar1" (experimental)
    latent_process: str = "rw"
    # sigma_obs prior family gate: "halfnormal" (legacy) | "lognormal"
    sigma_obs_prior_type: str = "halfnormal"
    # sigma_artist prior family gate: "halfnormal" (legacy) | "lognormal"
    sigma_artist_prior_type: str = "halfnormal"
    # Artist-effect parameterization: "noncentered" (legacy) | "zerosum"
    artist_effect_param: str = "noncentered"
    # LogNormal(loc, sigma) params for the sigma_rw / sigma_artist priors, sized
    # for the AOTY scale; external domains right-size them per scale.
    sigma_rw_lognormal_loc: float = -2.8
    sigma_rw_lognormal_sigma: float = 0.6
    sigma_artist_lognormal_loc: float = -0.9
    sigma_artist_lognormal_sigma: float = 0.6
    # Normal(loc, scale) params for the AR(1) coefficient prior; domains set
    # rho_scale small to pin rho near zero and disable the AR channel.
    rho_loc: float = 0.0
    rho_scale: float = 0.3
    # Covariate-block prior gate (#155): "normal" (legacy) | "horseshoe"
    beta_prior_type: str = "normal"
    hs_global_scale: float = 0.1
    # Entity-level observation overdispersion gate (AOTY default since 0.13.0,
    # #238; False reproduces the legacy bit-identical path)
    heteroscedastic_entity_obs: bool = True
    tau_entity_scale: float = 0.25
    # Errors-in-variables on the AR(1) regressor (model-v2; default off)
    errors_in_variables: bool = False
    # Propagate the random walk past the training horizon at predict time (model-v2)
    propagate_rw_horizon: bool = False
    # Genre/group pooling level between the global mean and entity effects
    # (#41). None = auto: on where the descriptor names an entity_group_col
    # present in the training split (the orchestrator always passes the
    # configured PipelineConfig value; train resolves it).
    entity_group_pooling: bool | None = None
    # Missing-covariate treatment gate (#158): median + indicators vs fillna(0)
    impute_missing: bool = False
    # Stacked-GBM offset feature block (#86). Stays False here so bare test
    # contexts keep the legacy roster; the orchestrator always passes the
    # config value (default on since 0.6.0), mirroring the 0.5.0
    # target_transform flip that left this context at "identity".
    gbm_offset: bool = False
    # Opt-in in-sampler exclusion of the rw_raw tensor (peak-GPU cut)
    exclude_rw_raw_from_collection: bool = False
    # Warm-start adaptation transfer (select --warmup-transfer): the reference
    # arm exports, later signature-matched arms import.
    warmup_export_path: str | None = None
    warmup_import_path: str | None = None
    # Split configuration
    val_albums: int = 0
    origin_offset: int = 0
    conformal_calibration: bool = False
    # Multi-step ancestral rollout depth for evaluation (#157); 0 = off
    eval_horizon: int = 0
    min_train_albums: int = 1
    # Evaluation configuration
    calibration_intervals: tuple[float, ...] = (0.80, 0.95)
    coverage_tolerance: float = 0.03
    prediction_interval: float = 0.95
    evaluate_secondary_split: bool = True
    # Prediction batching (memory/speed trade-off, not statistically relevant)
    predictive_batch_size: int = 500
    predict_artist_batch_size: int = 50
    # Dataset descriptor (default reproduces AOTY behavior exactly)
    descriptor: DatasetDescriptor = field(default_factory=DatasetDescriptor)
    # Artifact roots (flat default reproduces the legacy layout exactly)
    paths: ArtifactPaths = field(default_factory=ArtifactPaths.flat)


@dataclass
class PipelineStage:
    """A pipeline stage with input tracking for incremental runs.

    Each stage defines its inputs, outputs, and dependencies on other stages.
    The compute_input_hash method enables skip detection by comparing current
    input hashes against previously recorded values.

    Attributes:
        name: Unique identifier for the stage (e.g., "data", "splits", "train")
        description: Human-readable description of what the stage does
        run_fn: Function to execute the stage, or None for placeholder stages
        input_paths: List of file paths this stage reads from
        output_paths: List of file paths this stage creates
        depends_on: List of stage names that must run before this stage
        input_values: Semantic values included in incremental-run hashes

    Example:
        >>> stage = PipelineStage(
        ...     name="data",
        ...     description="Prepare and clean raw data",
        ...     run_fn=None,
        ...     input_paths=[Path("data/raw/albums.csv")],
        ...     output_paths=[Path("data/processed/cleaned.parquet")],
        ...     depends_on=[],
        ... )
        >>> stage.compute_input_hash()  # Returns hash of input files
    """

    name: str
    description: str
    run_fn: Callable[..., None] | None
    input_paths: list[Path] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    input_values: list[str] = field(default_factory=list)

    def compute_input_hash(self) -> str:
        """Compute combined hash of all input files.

        Hashes all existing input files and combines them into a single
        hash for comparison during skip detection.

        Returns:
            Combined SHA256 hash of all input files, or empty string if
            no input files exist.

        Example:
            >>> stage.compute_input_hash()
            'abc123def456...'
        """
        # Hash ordered (path, hash) pairs: combining bare sorted hashes would
        # decouple each hash from the file it belongs to, so two inputs
        # swapping contents would go undetected.
        pairs: list[str] = []

        for path in sorted(self.input_paths):
            if path.exists():
                pairs.append(f"{path.as_posix()}:{sha256_path(path)}")
        pairs.extend(f"value:{value}" for value in sorted(self.input_values))

        if not pairs:
            return ""

        combined = hashlib.sha256("\n".join(pairs).encode()).hexdigest()
        return combined

    def should_skip(
        self,
        manifest: RunManifest | None,
        force: bool = False,
    ) -> bool:
        """Check if stage can be skipped (outputs exist, inputs unchanged).

        A stage can be skipped only if:
        1. force is False
        2. A previous manifest exists
        3. The stage was run in that manifest
        4. The current input hash matches the recorded hash
        5. All output files exist

        Args:
            manifest: Previous run manifest to compare against, or None.
            force: If True, never skip (always return False).

        Returns:
            True if stage can be safely skipped, False otherwise.

        Example:
            >>> if stage.should_skip(previous_manifest):
            ...     print(f"Skipping {stage.name} (inputs unchanged)")
        """
        if force:
            return False

        if manifest is None:
            return False

        # Check if this stage was run before
        prev_hash = manifest.stage_hashes.get(self.name)
        if prev_hash is None:
            return False

        # Check if inputs have changed
        current_hash = self.compute_input_hash()
        if current_hash != prev_hash:
            return False

        # Check all outputs exist
        if not all(p.exists() for p in self.output_paths):
            return False

        return True


def _topological_sort(
    stages: list[PipelineStage],
    stage_names: set[str] | None = None,
) -> list[PipelineStage]:
    """Sort stages by dependencies using Kahn's algorithm.

    Args:
        stages: List of stages to sort.
        stage_names: Optional set of stage names to include. If None, include all.

    Returns:
        Stages in dependency order (dependencies first).

    Raises:
        ValueError: If there is a cycle in dependencies or missing dependency.
    """
    # Build adjacency list and in-degree count
    name_to_stage = {s.name: s for s in stages}

    # Filter to requested stages if specified
    if stage_names is not None:
        stages = [s for s in stages if s.name in stage_names]

    # Validate all dependencies exist
    all_stage_names = set(name_to_stage.keys())
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in all_stage_names:
                raise ValueError(f"Stage '{stage.name}' depends on unknown stage '{dep}'")

    # Build in-degree map: count dependencies within our stage set
    stage_name_set = {s.name for s in stages}
    in_degree: dict[str, int] = {s.name: 0 for s in stages}
    for stage in stages:
        for dep in stage.depends_on:
            if dep in stage_name_set:
                in_degree[stage.name] += 1

    # Kahn's algorithm
    queue = deque([s for s in stages if in_degree[s.name] == 0])
    result: list[PipelineStage] = []

    while queue:
        current = queue.popleft()
        result.append(current)

        # Decrease in-degree for stages that depend on current
        for stage in stages:
            if current.name in stage.depends_on:
                in_degree[stage.name] -= 1
                if in_degree[stage.name] == 0:
                    queue.append(stage)

    if len(result) != len(stages):
        # Cycle detected
        remaining = [s.name for s in stages if s not in result]
        raise ValueError(f"Circular dependency detected among stages: {remaining}")

    return result


# ============================================================================
# Stage Factory Functions
# ============================================================================


def _run_data_stage(ctx: StageContext):
    """Run data preparation stage."""
    from panelcast.data.cleaning import CleaningConfig
    from panelcast.pipelines.prepare_dataset import PrepareConfig, prepare_datasets

    descriptor = ctx.descriptor
    result = prepare_datasets(
        PrepareConfig(
            raw_path=str(_resolve_raw_dataset_path(descriptor)),
            output_dir=ctx.paths.processed.as_posix(),
            min_ratings_thresholds=list(descriptor.min_obs_thresholds),
            primary_min_ratings=descriptor.primary_min_obs,
            dataset_hash_output=str(ctx.run_dir / "dataset_hash.txt"),
            # Publication-mode runs should fail fast on schema drift.
            # Non-strict runs keep permissive behavior for exploratory work.
            validate_raw_schema=ctx.strict,
            cleaning=CleaningConfig(
                min_year=descriptor.min_year,
                strict_validation=ctx.strict,
                descriptor=descriptor,
            ),
            descriptor=descriptor,
        )
    )
    outputs = {"dataset_hash": str(ctx.run_dir / "dataset_hash.txt")}
    cleaned = result.datasets_created.get("cleaned_all")
    if cleaned is not None:
        outputs["cleaned_dataset"] = str(cleaned)
    return outputs


def _run_splits_stage(ctx: StageContext):
    """Run splits creation stage."""
    from panelcast.pipelines.create_splits import SplitConfig, create_splits

    # Pass seed and min_ratings from context for reproducibility
    config = SplitConfig(
        random_state=ctx.seed,
        min_ratings=ctx.min_ratings,
        val_albums=getattr(ctx, "val_albums", 0),
        origin_offset=getattr(ctx, "origin_offset", 0),
        min_train_albums=getattr(ctx, "min_train_albums", 1),
        entity_col=ctx.descriptor.entity_col,
        date_col=ctx.descriptor.parsed_date_col,
        event_col=ctx.descriptor.event_col,
        output_dir=ctx.paths.splits,
        source_path=ctx.paths.processed
        / f"{ctx.descriptor.processed_name(ctx.min_ratings)}.parquet",
    )
    return create_splits(config)


def _run_features_stage(ctx: StageContext):
    """Run feature building stage."""
    from panelcast.pipelines.build_features import build_features

    return build_features(ctx)


def _run_train_stage(ctx: StageContext):
    """Run model training stage."""
    from panelcast.pipelines.train_bayes import train_models

    return train_models(ctx)


def _run_evaluate_stage(ctx: StageContext):
    """Run model evaluation stage."""
    from panelcast.pipelines.evaluate import evaluate_models

    return evaluate_models(ctx)


def _run_predict_stage(ctx: StageContext):
    """Run next-event prediction stage."""
    from panelcast.pipelines.predict_next import predict_next_events

    return predict_next_events(ctx)


def _run_report_stage(ctx: StageContext):
    """Run publication artifact generation stage."""
    from panelcast.pipelines.publication import generate_publication_artifacts

    return generate_publication_artifacts(ctx)


def _run_sensitivity_stage(ctx: StageContext):
    """Run the opt-in sensitivity analysis stage."""
    from panelcast.pipelines.sensitivity import run_sensitivity_suite

    return run_sensitivity_suite(ctx)


def _resolve_raw_dataset_path(descriptor: DatasetDescriptor | None = None) -> Path:
    """Resolve raw dataset path from environment with descriptor default."""
    descriptor = descriptor or DatasetDescriptor()
    return descriptor.resolve_raw_path()


def make_stage_data(
    descriptor: DatasetDescriptor | None = None,
) -> PipelineStage:
    """Create data preparation stage.

    Args:
        descriptor: Dataset descriptor (None = AOTY defaults).
    """
    descriptor = descriptor or DatasetDescriptor()
    output_paths = [Path("data/processed/cleaned_all.parquet")]
    output_paths.extend(
        Path("data/processed") / f"{descriptor.processed_name(t)}.parquet"
        for t in descriptor.min_obs_thresholds
    )
    if descriptor.secondary_target_col is not None:
        secondary_name = f"{descriptor.secondary_prefix}_score.parquet"
        output_paths.append(Path("data/processed") / secondary_name)
    return PipelineStage(
        name="data",
        description="Prepare and clean raw album data",
        run_fn=_run_data_stage,
        input_paths=[_resolve_raw_dataset_path(descriptor)],
        output_paths=output_paths,
        depends_on=[],
        input_values=[descriptor.descriptor_hash()],
    )


def make_stage_splits(
    min_ratings: int = 10,
    descriptor: DatasetDescriptor | None = None,
) -> PipelineStage:
    """Create splits stage.

    Args:
        min_ratings: Minimum user ratings per album. Determines input file path.
        descriptor: Dataset descriptor (None = AOTY defaults).
    """
    descriptor = descriptor or DatasetDescriptor()
    return PipelineStage(
        name="splits",
        description="Create train/validation/test splits",
        run_fn=_run_splits_stage,
        input_paths=[Path("data/processed") / f"{descriptor.processed_name(min_ratings)}.parquet"],
        output_paths=[
            Path("data/splits/within_entity_temporal/train.parquet"),
            Path("data/splits/within_entity_temporal/validation.parquet"),
            Path("data/splits/within_entity_temporal/test.parquet"),
            Path("data/splits/within_entity_temporal/manifest.json"),
            Path("data/splits/entity_disjoint/train.parquet"),
            Path("data/splits/entity_disjoint/validation.parquet"),
            Path("data/splits/entity_disjoint/test.parquet"),
            Path("data/splits/entity_disjoint/manifest.json"),
            Path("data/splits/pipeline_summary.json"),
        ],
        depends_on=["data"],
    )


def make_stage_features() -> PipelineStage:
    """Create feature building stage."""
    return PipelineStage(
        name="features",
        description="Build feature matrices from split data",
        run_fn=_run_features_stage,
        input_paths=[
            Path("data/splits/within_entity_temporal/train.parquet"),
            Path("data/splits/within_entity_temporal/validation.parquet"),
            Path("data/splits/within_entity_temporal/test.parquet"),
            Path("data/splits/entity_disjoint/train.parquet"),
            Path("data/splits/entity_disjoint/validation.parquet"),
            Path("data/splits/entity_disjoint/test.parquet"),
        ],
        output_paths=[
            Path("data/features/within_entity_temporal/train_features.parquet"),
            Path("data/features/within_entity_temporal/validation_features.parquet"),
            Path("data/features/within_entity_temporal/test_features.parquet"),
            Path("data/features/entity_disjoint/train_features.parquet"),
            Path("data/features/entity_disjoint/validation_features.parquet"),
            Path("data/features/entity_disjoint/test_features.parquet"),
            Path("data/features/train_features.parquet"),
            Path("data/features/validation_features.parquet"),
            Path("data/features/test_features.parquet"),
        ],
        depends_on=["splits"],
    )


def make_stage_train(paths: ArtifactPaths | None = None) -> PipelineStage:
    """Create model training stage.

    Args:
        paths: Artifact roots for the mutable products (None = legacy flat
            layout). Skip detection and output recording resolve against
            these, so a run-scoped orchestrator must pass its own.
    """
    paths = paths or ArtifactPaths.flat()
    return PipelineStage(
        name="train",
        description="Fit Bayesian models on training data",
        run_fn=_run_train_stage,
        input_paths=[
            Path("data/features/train_features.parquet"),
            Path("data/features/validation_features.parquet"),
        ],
        output_paths=[
            paths.models / "manifest.json",
            paths.models / "training_summary.json",
        ],
        depends_on=["features"],
    )


def make_stage_evaluate(paths: ArtifactPaths | None = None) -> PipelineStage:
    """Create model evaluation stage."""
    paths = paths or ArtifactPaths.flat()
    return PipelineStage(
        name="evaluate",
        description="Run model evaluation and diagnostics",
        run_fn=_run_evaluate_stage,
        input_paths=[
            paths.models / "manifest.json",
            paths.models / "training_summary.json",
            Path("data/features/within_entity_temporal/test_features.parquet"),
            Path("data/features/entity_disjoint/test_features.parquet"),
        ],
        output_paths=[
            paths.evaluation / "metrics.json",
            paths.evaluation / "diagnostics.json",
            paths.evaluation / "within_entity_temporal/predictions.json",
            paths.evaluation / "entity_disjoint/predictions.json",
        ],
        depends_on=["train"],
    )


def make_stage_predict(paths: ArtifactPaths | None = None) -> PipelineStage:
    """Create next-event prediction stage."""
    paths = paths or ArtifactPaths.flat()
    return PipelineStage(
        name="predict",
        description="Generate next-event predictions for known and new entities",
        run_fn=_run_predict_stage,
        input_paths=[
            paths.models / "manifest.json",
            paths.models / "training_summary.json",
            Path("data/splits/within_entity_temporal/train.parquet"),
            Path("data/features/train_features.parquet"),
        ],
        output_paths=[
            paths.predictions / "next_event_known_entities.csv",
            paths.predictions / "next_event_new_entity.csv",
            paths.predictions / "prediction_summary.json",
        ],
        depends_on=["evaluate"],
    )


def make_stage_report(paths: ArtifactPaths | None = None) -> PipelineStage:
    """Create publication artifacts stage."""
    paths = paths or ArtifactPaths.flat()
    return PipelineStage(
        name="report",
        description="Generate publication artifacts (figures, tables)",
        run_fn=_run_report_stage,
        input_paths=[
            paths.evaluation / "metrics.json",
            paths.evaluation / "diagnostics.json",
            paths.predictions / "prediction_summary.json",
        ],
        output_paths=[
            paths.reports / "artifact_status.json",
            paths.reports / "tables/coefficients.csv",
            paths.reports / "tables/diagnostics.csv",
            paths.reports / "tables/metrics_summary.csv",
            paths.reports / "figures/trace_plot.pdf",
            paths.reports / "figures/posterior_plot.pdf",
            paths.reports / "MODEL_CARD.md",
        ],
        depends_on=["predict"],
    )


def make_stage_sensitivity(paths: ArtifactPaths | None = None) -> PipelineStage:
    """Create the opt-in sensitivity analysis stage.

    Not part of the default stage list: it refits the model several times
    (prior variants, feature ablations) and is only run when named
    explicitly (``--stages sensitivity`` or ``panelcast stage
    sensitivity``) after an evaluate run has produced its artifacts.
    """
    paths = paths or ArtifactPaths.flat()
    return PipelineStage(
        name="sensitivity",
        description="Sensitivity analyses (priors, ablations, split seed)",
        run_fn=_run_sensitivity_stage,
        input_paths=[
            paths.models / "training_summary.json",
            Path("data/features/train_features.parquet"),
            Path("data/splits/within_entity_temporal/train.parquet"),
        ],
        output_paths=[
            paths.reports / "sensitivity/sensitivity_results.json",
        ],
        depends_on=["evaluate"],
    )


def build_pipeline_stages(
    min_ratings: int = 10,
    descriptor: DatasetDescriptor | None = None,
    descriptor_path: Path | None = None,
    paths: ArtifactPaths | None = None,
) -> list[PipelineStage]:
    """Build pipeline stages list with runtime configuration.

    Args:
        min_ratings: Minimum user ratings per album. Passed to make_stage_splits()
            to ensure input_paths point to the correct parquet file.
        descriptor: Dataset descriptor (None = AOTY defaults).
        descriptor_path: Retained for caller compatibility; the semantic
            descriptor hash controls stage invalidation.
        paths: Artifact roots for the mutable-product stages (None = legacy
            flat layout). Data stages always declare flat paths.

    Returns:
        List of PipelineStage objects configured for the given dataset.
        Opt-in stages (sensitivity) are NOT included; see
        build_optional_stages().
    """
    return [
        make_stage_data(descriptor=descriptor),
        make_stage_splits(min_ratings=min_ratings, descriptor=descriptor),
        make_stage_features(),
        make_stage_train(paths=paths),
        make_stage_evaluate(paths=paths),
        make_stage_predict(paths=paths),
        make_stage_report(paths=paths),
    ]


def build_optional_stages(paths: ArtifactPaths | None = None) -> list[PipelineStage]:
    """Opt-in stages: available by name, never part of a default run."""
    return [make_stage_sensitivity(paths=paths)]


def get_execution_order(
    stages: list[str] | None = None,
    min_ratings: int = 10,
    descriptor: DatasetDescriptor | None = None,
    descriptor_path: Path | None = None,
    paths: ArtifactPaths | None = None,
) -> list[PipelineStage]:
    """Get stages in dependency-respecting execution order.

    Args:
        stages: List of stage names to include, or None for all stages.
            If provided, stages are returned in topological order respecting
            dependencies between the specified stages.
        min_ratings: Minimum user ratings per album. Determines which parquet
            file the splits stage uses as input.
        paths: Artifact roots the mutable-product stages declare their
            input/output paths against (None = legacy flat layout).

    Returns:
        List of PipelineStage objects in execution order.

    Raises:
        KeyError: If an unknown stage name is provided.
        ValueError: If there is a circular dependency.

    Example:
        >>> order = get_execution_order()
        >>> [s.name for s in order]
        ['data', 'splits', 'features', 'train', 'evaluate', 'report']

        >>> order = get_execution_order(["features", "splits"], min_ratings=30)
        >>> [s.name for s in order]
        ['splits', 'features']
    """
    # Build stages with runtime min_ratings to ensure correct input_paths
    pipeline_stages = build_pipeline_stages(
        min_ratings=min_ratings,
        descriptor=descriptor,
        descriptor_path=descriptor_path,
        paths=paths,
    )

    if stages is None:
        # Default run: opt-in stages are deliberately excluded.
        return _topological_sort(pipeline_stages)

    # Named selection may include opt-in stages.
    all_stages = pipeline_stages + build_optional_stages(paths=paths)

    # Validate stage names
    valid_names = {s.name for s in all_stages}
    for name in stages:
        if name not in valid_names:
            raise KeyError(f"Unknown stage: '{name}'. Valid stages: {sorted(valid_names)}")

    # Filter and sort
    stage_set = set(stages)
    return _topological_sort(all_stages, stage_set)


def get_stage(name: str, min_ratings: int = 10) -> PipelineStage:
    """Look up a stage by name.

    Args:
        name: Stage name to look up.
        min_ratings: Minimum user ratings per album. Determines which parquet
            file the splits stage uses as input.

    Returns:
        PipelineStage with the given name.

    Raises:
        KeyError: If no stage with that name exists.

    Example:
        >>> stage = get_stage("train")
        >>> stage.description
        'Fit Bayesian models on training data'
    """
    all_stages = build_pipeline_stages(min_ratings=min_ratings) + build_optional_stages()
    for stage in all_stages:
        if stage.name == name:
            return stage

    valid_names = sorted(s.name for s in all_stages)
    raise KeyError(f"Unknown stage: '{name}'. Valid stages: {valid_names}")
