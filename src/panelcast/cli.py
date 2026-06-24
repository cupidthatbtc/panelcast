"""Command-line interface for the panelcast prediction pipeline.

This module provides CLI entry points for running the full pipeline or
individual stages. The primary entry point is `panelcast run`, which
executes all stages in dependency order with progress tracking.

Usage:
    panelcast run --seed 42
    panelcast run --dry-run --verbose
    panelcast stage data --verbose
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# Quick preflight uses fixed estimates rather than loading actual data (~1s vs ~30-60s).
# These are conservative defaults for the memory estimation formula.
# For accurate checking with real data dimensions, use --preflight-full.
QUICK_PREFLIGHT_OBSERVATIONS = 1000  # Conservative observation count
QUICK_PREFLIGHT_FEATURES = 20  # Typical feature count from feature builder
QUICK_PREFLIGHT_ARTISTS = 100  # Moderate artist count

app = typer.Typer(
    add_completion=False,
    help="panelcast - hierarchical Bayesian prediction for bounded scores over entity histories.",
    invoke_without_command=True,
)

# Stage subcommand group
stage_app = typer.Typer(help="Run individual pipeline stages")
app.add_typer(stage_app, name="stage")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
    ),
    setup_guide: bool = typer.Option(
        False,
        "--setup-guide",
        help="Show path to step-by-step setup guide and exit.",
    ),
) -> None:
    """panelcast - hierarchical Bayesian prediction, reproducible ML workflow."""
    if version:
        typer.echo(f"panelcast version {__version__}")
        raise typer.Exit()
    if setup_guide:
        typer.echo("Step-by-step setup guide: docs/GETTING_STARTED.md")
        typer.echo("")
        typer.echo("Covers: prerequisites, installation, data setup, GPU config,")
        typer.echo("verification, running the pipeline, and troubleshooting.")
        typer.echo("")
        typer.echo(
            "View online: https://github.com/cupidthatbtc/panelcast/blob/main/docs/GETTING_STARTED.md"
        )
        raise typer.Exit()
    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("run")
def run(
    ctx: typer.Context,
    config_files: Optional[list[str]] = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "YAML config file(s) with PipelineConfig keys (e.g. "
            "configs/publication.yaml). Repeatable; later files override "
            "earlier ones. Explicit CLI options always win over YAML."
        ),
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help=(
            "Named config preset {quick,dev,diagnostic,publication} — sugar for "
            "--config configs/<preset>.yaml, layered first so any --config files "
            "and explicit CLI options still win."
        ),
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip stages with unchanged inputs",
    ),
    stages: Optional[str] = typer.Option(
        None,
        "--stages",
        "-s",
        help="Comma-separated list of stages to run (e.g., 'data,splits,train')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show execution plan without running stages",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on convergence warnings and calibration tolerance failures",
    ),
    allow_unlocked_env: bool = typer.Option(
        False,
        "--allow-unlocked-env",
        help="Allow runs without pixi.lock (not publication-safe)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging",
    ),
    preflight: bool = typer.Option(
        False,
        "--preflight",
        help="Quick memory check (~1s) with fixed estimates; use --preflight-full for accuracy",
    ),
    preflight_only: bool = typer.Option(
        False,
        "--preflight-only",
        help="Run memory check and exit (0=pass, 1=fail, 2=warning/cannot-check)",
    ),
    force_run: bool = typer.Option(
        False,
        "--force-run",
        help="Override preflight failure and continue anyway (use with --preflight)",
    ),
    preflight_full: bool = typer.Option(
        False,
        "--preflight-full",
        help="Run calibration mini-MCMC and extrapolate to target sample count",
    ),
    recalibrate: bool = typer.Option(
        False,
        "--recalibrate",
        help="Force fresh calibration even if cached calibration exists",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume failed run by run-id (e.g., '2026-01-19_143052')",
    ),
    max_albums: Annotated[
        int,
        typer.Option(
            min=1,
            help="Max albums per artist for training. Beyond this use same artist effect.",
        ),
    ] = 50,
    # MCMC Configuration
    num_chains: Annotated[
        int,
        typer.Option(
            min=1,
            help="Number of parallel MCMC chains (default 4)",
        ),
    ] = 4,
    num_samples: Annotated[
        int,
        typer.Option(
            min=100,
            help="Post-warmup samples per chain (default 1000)",
        ),
    ] = 1000,
    num_warmup: Annotated[
        int,
        typer.Option(
            min=50,
            help="Warmup iterations per chain (default 1000)",
        ),
    ] = 1000,
    target_accept: Annotated[
        float,
        typer.Option(
            min=0.5,
            max=0.999,
            help="Target acceptance probability (default 0.90)",
        ),
    ] = 0.90,
    max_tree_depth: Annotated[
        int,
        typer.Option(
            min=5,
            max=15,
            help="Maximum tree depth for NUTS (default 10)",
        ),
    ] = 10,
    chain_method: Annotated[
        str,
        typer.Option(
            "--chain-method",
            help="Chain method: 'sequential', 'vectorized', or 'parallel' (multi-GPU)",
        ),
    ] = "sequential",
    # Convergence Thresholds
    rhat_threshold: Annotated[
        float,
        typer.Option(
            min=1.0,
            max=1.1,
            help="Maximum acceptable R-hat (default 1.01)",
        ),
    ] = 1.01,
    ess_threshold: Annotated[
        int,
        typer.Option(
            min=100,
            help="Minimum ESS per chain (default 400)",
        ),
    ] = 400,
    allow_divergences: bool = typer.Option(
        False,
        "--allow-divergences",
        help="Don't fail on divergences (for exploratory runs)",
    ),
    # Data Filtering
    min_ratings: Annotated[
        int,
        typer.Option(
            min=1,
            help="Minimum user ratings per album (default 10)",
        ),
    ] = 10,
    min_albums: Annotated[
        int,
        typer.Option(
            min=1,
            help="Minimum albums per artist for dynamic effects (default 2)",
        ),
    ] = 2,
    # Feature Ablation flags
    enable_genre: Annotated[
        bool,
        typer.Option(
            " /--no-genre",
            help="Disable genre features",
        ),
    ] = True,
    enable_artist: Annotated[
        bool,
        typer.Option(
            " /--no-artist",
            help="Disable artist reputation features",
        ),
    ] = True,
    enable_temporal: Annotated[
        bool,
        typer.Option(
            " /--no-temporal",
            help="Disable temporal features",
        ),
    ] = True,
    # Heteroscedastic noise configuration
    n_exponent: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Noise exponent (0.0=homoscedastic, 0.5=sqrt scaling)",
        ),
    ] = 0.0,
    learn_n_exponent: bool = typer.Option(
        False,
        "--learn-n-exponent",
        help="Learn exponent from data (ignores --n-exponent if set)",
    ),
    n_exponent_alpha: Annotated[
        float,
        typer.Option(
            min=0.01,
            help="Beta prior alpha parameter for learned exponent (advanced, default 2.0)",
        ),
    ] = 2.0,
    n_exponent_beta: Annotated[
        float,
        typer.Option(
            min=0.01,
            help="Beta prior beta parameter for learned exponent (advanced, default 4.0)",
        ),
    ] = 4.0,
    n_exponent_prior: str = typer.Option(
        "logit-normal",
        "--n-exponent-prior",
        help="Prior for learned n_exponent: 'logit-normal' (default) or 'beta' (legacy)",
    ),
    likelihood_df: Annotated[
        float,
        typer.Option(
            min=1.0,
            help=(
                "Degrees of freedom for Student-t likelihood. "
                "Lower values give heavier tails (default 4.0). "
                "Set >= 100 for Normal likelihood."
            ),
        ),
    ] = 4.0,
    likelihood_family: str = typer.Option(
        "studentt",
        "--likelihood-family",
        help=(
            "Observation likelihood: 'studentt' (default) or 'normal' (symmetric); "
            "the skew candidates 'skew_studentt' / 'skew_normal' (sinh-arcsinh) and "
            "'split_normal' (two-piece); or 'beta' (bounded mean-precision Beta)."
        ),
    ),
    discretize_observation: bool = typer.Option(
        False,
        "--discretize-observation",
        help=(
            "Interval-censor the observation to integers: integer k contributes "
            "log(F(k+0.5)-F(k-0.5)) and replicated draws are rounded (honest PPC "
            "for integer-valued scores). Location-scale families only "
            "(studentt, normal, skew_normal, split_normal); rejected for beta."
        ),
    ),
    val_albums: Annotated[
        int,
        typer.Option(
            min=0,
            help=(
                "Number of albums per artist to hold out for validation "
                "(default 0 = no validation split)."
            ),
        ),
    ] = 0,
    min_train_albums: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "Minimum training albums per artist. Artists with fewer than "
                "(min_train_albums + val_albums + 1) total albums are excluded. "
                "Higher values = fewer artists but more history per artist (default 2)."
            ),
        ),
    ] = 2,
    calibration_intervals: str = typer.Option(
        "0.80,0.95",
        "--calibration-intervals",
        help="Comma-separated interval levels for calibration checks (e.g. 0.80,0.95)",
    ),
    coverage_tolerance: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Allowed absolute coverage error for calibration checks (default 0.03)",
        ),
    ] = 0.03,
    prediction_interval: Annotated[
        float,
        typer.Option(
            min=0.01,
            max=0.99,
            help="Interval level used for saved prediction bands (default 0.95)",
        ),
    ] = 0.95,
    secondary_split: bool = typer.Option(
        True,
        "--secondary-split/--no-secondary-split",
        help="Enable artist-disjoint secondary evaluation split",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help=(
            "Dataset descriptor: bare name (resolves to configs/datasets/{name}.yaml) "
            "or YAML path. Omit for built-in AOTY defaults."
        ),
    ),
    debut_prev_score_source: str = typer.Option(
        "train_mean",
        "--debut-prev-score-source",
        help=(
            "Debut prev_score fill: 'train_mean' (train-split only, default) or "
            "'dataset_stats' (legacy pre-split mean; mild test leakage)."
        ),
    ),
    target_transform: str = typer.Option(
        "identity",
        "--target-transform",
        help=(
            "Score-scale transform: 'identity' (legacy soft-clip) or "
            "'offset_logit' (model runs on the Smithson-Verkuilen logit scale; "
            "tested but held — did not mix; see LIKELIHOOD_CANDIDATES.md)."
        ),
    ),
    ar_center: str = typer.Option(
        "global",
        "--ar-center",
        help=(
            "AR(1) centering: 'global' (default; subtract the train-mean "
            "prev_score so debut AR terms are zero and rho decouples from "
            "the artist-effect level), 'none' (legacy uncentered), or "
            "'artist_running' (per-artist running mean; sensitivity only)."
        ),
    ),
    latent_process: str = typer.Option(
        "rw",
        "--latent-process",
        help=(
            "Latent artist-effect process: 'rw' (random walk, default) or "
            "'ar1' (stationary deviations with persistence phi; nests rw "
            "at phi=1). Experimental — adopt only if LOO clearly wins."
        ),
    ),
    exclude_rw_raw_from_collection: bool = typer.Option(
        False,
        "--exclude-rw-raw-from-collection",
        help=(
            "Never store rw_raw draws on device during sampling (~96% peak-GPU "
            "cut at production settings). Posterior for all other sites is "
            "unchanged; required for the 4-chain publication run on 24 GB GPUs."
        ),
    ),
) -> None:
    """Execute full pipeline from raw data to publication artifacts.

    Runs all pipeline stages in dependency order: data -> splits -> features ->
    train -> evaluate -> report. Creates a timestamped output directory with
    manifest for reproducibility.

    Examples:
        # Default run
        panelcast run

        # High-accuracy run with more chains and samples
        panelcast run --num-chains 8 --num-samples 2000 --target-accept 0.95

        # Fast exploratory run
        panelcast run --num-chains 1 --num-samples 500 --num-warmup 500

        # Feature ablation
        panelcast run --no-genre --no-temporal

        # Relaxed convergence for testing
        panelcast run --rhat-threshold 1.05 --allow-divergences

        # Resume a failed run
        panelcast run --resume 2026-01-19_143052

        # Check memory before running
        panelcast run --preflight

        # Check memory only (CI/scripting)
        panelcast run --preflight-only

        # Force run despite preflight failure
        panelcast run --preflight --force-run
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    # Parse stages from comma-separated string
    stage_list: list[str] | None = None
    if stages:
        stage_list = [s.strip() for s in stages.split(",") if s.strip()]

    # Validate chain_method (case-insensitive)
    valid_chain_methods = ("sequential", "vectorized", "parallel")
    chain_method_normalized = chain_method.lower()
    if chain_method_normalized not in valid_chain_methods:
        typer.echo(
            f"Error: Invalid --chain-method '{chain_method}'. "
            f"Must be one of: {', '.join(valid_chain_methods)}"
        )
        raise typer.Exit(code=1)

    # Validate n_exponent_prior
    valid_priors = ("logit-normal", "beta")
    if n_exponent_prior not in valid_priors:
        typer.echo(
            f"Error: Invalid --n-exponent-prior '{n_exponent_prior}'. "
            f"Must be one of: {', '.join(valid_priors)}"
        )
        raise typer.Exit(code=1)

    # Validate likelihood_family (single source of truth: the registry)
    from panelcast.models.bayes.likelihoods import REGISTRY

    valid_families = tuple(REGISTRY)
    if likelihood_family not in valid_families:
        typer.echo(
            f"Error: Invalid --likelihood-family '{likelihood_family}'. "
            f"Must be one of: {', '.join(valid_families)}"
        )
        raise typer.Exit(code=1)
    chain_method = chain_method_normalized  # Use normalized value downstream

    # Parse calibration interval levels
    try:
        calibration_levels = tuple(
            sorted({float(x.strip()) for x in calibration_intervals.split(",") if x.strip()})
        )
    except ValueError as e:
        typer.echo(f"Error: Invalid --calibration-intervals: {e}")
        raise typer.Exit(code=1) from e
    if not calibration_levels:
        typer.echo("Error: --calibration-intervals must contain at least one value.")
        raise typer.Exit(code=1)
    if any(level <= 0.0 or level >= 1.0 for level in calibration_levels):
        typer.echo("Error: --calibration-intervals values must be in (0, 1).")
        raise typer.Exit(code=1)

    # Full preflight mode (--preflight-full) takes precedence over quick preflight
    if preflight_full:
        from pathlib import Path

        import numpy as np
        from rich.console import Console

        from panelcast.pipelines.train_bayes import load_training_data
        from panelcast.preflight import (
            PreflightStatus,
            render_extrapolation_result,
            run_extrapolated_preflight_check,
        )
        from panelcast.preflight.full_check import _derive_dimensions_from_model_args

        console = Console()

        # Check if required data exists
        from panelcast.data.split_types import SplitType, resolve_split_dir

        features_path = Path("data/features/train_features.parquet")
        splits_path = resolve_split_dir(Path("data/splits"), SplitType.WITHIN_ENTITY_TEMPORAL) / (
            "train.parquet"
        )

        if not features_path.exists() or not splits_path.exists():
            console.print(
                "[bold red]Error:[/bold red] --preflight-full requires processed data.\n"
                "Missing files:\n"
                f"  - {features_path}: {'exists' if features_path.exists() else 'MISSING'}\n"
                f"  - {splits_path}: {'exists' if splits_path.exists() else 'MISSING'}\n\n"
                "Run data stages first, or use [bold]--preflight[/bold] for quick estimation."
            )
            raise typer.Exit(2)  # CANNOT_CHECK exit code

        from panelcast.config.descriptor import load_descriptor

        preflight_descriptor = load_descriptor(dataset)

        # Load data and build model_args using shared function
        model_args, _, _ = load_training_data(
            features_path=features_path,
            splits_path=splits_path,
            min_albums_filter=min_albums,
            descriptor=preflight_descriptor,
        )

        # Remove artist_album_counts (not needed for preflight)
        model_args.pop("artist_album_counts", None)

        # Apply max_albums cap to model_args["album_seq"]
        album_seq = model_args["album_seq"]
        model_args["album_seq"] = np.clip(album_seq, 1, max_albums)
        model_args["max_seq"] = max_albums

        # Add heteroscedastic params (use CLI-parsed values)
        model_args["n_exponent"] = n_exponent
        model_args["learn_n_exponent"] = learn_n_exponent
        model_args["likelihood_df"] = likelihood_df

        # Use shared dimension derivation for consistent validation
        n_observations, n_artists_dim, n_features, _ = _derive_dimensions_from_model_args(
            model_args
        )

        # Target is POST-WARMUP samples per chain: warmup draws are never
        # stored (measured: identical peaks at warmup 50 vs 250), and the
        # calibration runs at the production chain count so multi-chain
        # accumulation is measured rather than modeled.
        target_samples = num_samples

        # Show progress indicator
        progress_msg = (
            "[bold blue]Running calibration (10+50 samples)...[/bold blue]"
            if not recalibrate
            else "[bold blue]Running fresh calibration...[/bold blue]"
        )
        # Structural gates for the calibration cache key: a calibration must
        # never serve projections for a structurally different model.
        model_signature = {
            "descriptor_hash": preflight_descriptor.descriptor_hash(),
            "latent_process": latent_process,
            "target_transform": target_transform,
            "n_exponent": n_exponent,
            "learn_n_exponent": learn_n_exponent,
            "likelihood_df": likelihood_df,
            "likelihood_family": likelihood_family,
            "discretize_observation": discretize_observation,
            "exclude_rw_raw_from_collection": exclude_rw_raw_from_collection,
        }
        # Mirror the production fit's memory gate in the calibration runs:
        # with the rw_raw exclusion on, the dominant memory term disappears
        # and the projection must reflect that.
        preflight_exclude_collection = (
            (f"{preflight_descriptor.model_prefix}_rw_raw",)
            if exclude_rw_raw_from_collection
            else ()
        )

        with console.status(progress_msg):
            full_result = run_extrapolated_preflight_check(
                model_args=model_args,
                target_samples=target_samples,
                n_observations=n_observations,
                n_artists=n_artists_dim,
                n_features=n_features,
                max_seq=max_albums,
                headroom_target=0.20,
                # Sequential chains run one after another in the calibration
                # mini-runs; scale the per-run timeout with the chain count.
                timeout_seconds=120 * max(1, num_chains),
                recalibrate=recalibrate,
                model_signature=model_signature,
                exclude_collection=preflight_exclude_collection,
                num_chains=num_chains,
            )

        render_extrapolation_result(full_result, verbose=verbose)

        if preflight_only:
            raise typer.Exit(full_result.exit_code)

        if full_result.status == PreflightStatus.FAIL:
            if force_run:
                typer.echo("Warning: Continuing despite preflight failure (--force-run)")
            else:
                typer.echo("Use --force-run to override preflight failure")
                raise typer.Exit(full_result.exit_code)

    # Run preflight check if requested (quick estimation mode)
    # Note: Preflight runs BEFORE building PipelineConfig to fail fast
    # Skip quick preflight when full preflight was already run (--preflight-full takes precedence)
    if (preflight or preflight_only) and not preflight_full:
        import os

        from panelcast.config.descriptor import load_descriptor
        from panelcast.data.ingest import extract_data_dimensions
        from panelcast.preflight import (
            PreflightStatus,
            render_preflight_result,
            run_preflight_check,
        )

        # Resolve the dataset descriptor and its raw CSV path so the quick
        # preflight reads the configured domain's data and columns, not just
        # AOTY's. dataset=None -> AOTY defaults.
        quick_descriptor = load_descriptor(dataset)
        quick_csv_path = os.environ.get(
            quick_descriptor.raw_path_env, quick_descriptor.raw_path_default
        )

        # Extract actual data dimensions (graceful fallback to defaults)
        dimensions = extract_data_dimensions(
            csv_path=quick_csv_path,
            min_ratings=min_ratings,
            descriptor=quick_descriptor,
        )

        result = run_preflight_check(
            n_observations=dimensions.n_observations,
            n_features=QUICK_PREFLIGHT_FEATURES,  # Features built from columns, not counted
            n_artists=dimensions.n_artists,
            max_seq=max_albums,
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
            exclude_rw_raw_from_collection=exclude_rw_raw_from_collection,
        )

        render_preflight_result(result, verbose=verbose, dimensions=dimensions)

        if preflight_only:
            raise typer.Exit(code=result.exit_code)

        if result.status == PreflightStatus.FAIL:
            if force_run:
                typer.echo("Warning: Continuing despite preflight failure (--force-run)")
            else:
                typer.echo("Aborting. Use --force-run to override.")
                raise typer.Exit(code=result.exit_code)

    config_kwargs: dict = dict(
        seed=seed,
        skip_existing=skip_existing,
        stages=stage_list,
        dry_run=dry_run,
        strict=strict,
        enforce_lockfile=not allow_unlocked_env,
        verbose=verbose,
        resume=resume,
        max_albums=max_albums,
        # MCMC config
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        chain_method=chain_method,
        # Convergence thresholds
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        allow_divergences=allow_divergences,
        # Data filtering
        min_ratings=min_ratings,
        min_albums_filter=min_albums,
        # Feature flags
        enable_genre=enable_genre,
        enable_artist=enable_artist,
        enable_temporal=enable_temporal,
        # Heteroscedastic noise
        n_exponent=n_exponent,
        learn_n_exponent=learn_n_exponent,
        n_exponent_alpha=n_exponent_alpha,
        n_exponent_beta=n_exponent_beta,
        n_exponent_prior=n_exponent_prior,
        likelihood_df=likelihood_df,
        likelihood_family=likelihood_family,
        discretize_observation=discretize_observation,
        val_albums=val_albums,
        min_train_albums=min_train_albums,
        calibration_intervals=calibration_levels,
        coverage_tolerance=coverage_tolerance,
        prediction_interval=prediction_interval,
        evaluate_secondary_split=secondary_split,
        dataset=dataset,
        debut_prev_score_source=debut_prev_score_source,
        target_transform=target_transform,
        ar_center=ar_center,
        latent_process=latent_process,
        exclude_rw_raw_from_collection=exclude_rw_raw_from_collection,
    )

    # Resolve --preset to a config file layered FIRST, so explicit --config
    # files and CLI options still win (later layers override earlier ones).
    effective_config_files: list[str] = []
    if preset is not None:
        from pathlib import Path

        valid_presets = ("quick", "dev", "diagnostic", "publication")
        if preset not in valid_presets:
            typer.echo(
                f"Error: unknown --preset '{preset}'. "
                f"Choose one of: {', '.join(valid_presets)}"
            )
            raise typer.Exit(code=1)
        preset_path = Path("configs") / f"{preset}.yaml"
        if not preset_path.exists():
            # Fall back to the repo-bundled configs (relative to this file) so
            # --preset works when running from a separate domain directory
            # (docs/PORTING.md), not only from the repo root.
            bundled = Path(__file__).resolve().parents[2] / "configs" / f"{preset}.yaml"
            if bundled.exists():
                preset_path = bundled
            else:
                typer.echo(f"Error: preset config not found at {preset_path}.")
                raise typer.Exit(code=1)
        effective_config_files.append(str(preset_path))
    if config_files:
        effective_config_files.extend(config_files)

    try:
        if effective_config_files:
            from panelcast.config.loader import load_yaml_config
            from panelcast.config.pipeline_yaml import apply_yaml_overrides

            yaml_data = load_yaml_config(effective_config_files)
            # Params set explicitly on the command line win over YAML. click's
            # parameter-source API lives in internals that have drifted across
            # click/typer versions, so compare the source by its enum member
            # name (stable) rather than importing or identity-checking the enum,
            # and degrade gracefully if the API is unavailable.
            explicit_cli_params: set[str] = set()
            get_source = getattr(ctx, "get_parameter_source", None)
            if get_source is not None:
                for name in ctx.params:
                    source = get_source(name)
                    if source is not None and getattr(source, "name", "") == "COMMANDLINE":
                        explicit_cli_params.add(name)
            config_kwargs = apply_yaml_overrides(config_kwargs, yaml_data, explicit_cli_params)

        config = PipelineConfig(**config_kwargs)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


# Individual stage commands
@stage_app.command("data")
def stage_data(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run data preparation stage only.

    Loads raw album data, applies cleaning transformations, and creates
    processed datasets at multiple rating thresholds.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["data"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("splits")
def stage_splits(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run split creation stage only.

    Creates train/validation/test splits using within-artist temporal
    and artist-disjoint strategies.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["splits"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("features")
def stage_features(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run feature building stage only.

    Builds feature matrices from split data using configured feature blocks.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["features"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("train")
def stage_train(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on convergence warnings",
    ),
    rhat_threshold: Annotated[
        float,
        typer.Option(
            min=1.0,
            max=1.1,
            help="Maximum acceptable R-hat (default 1.01)",
        ),
    ] = 1.01,
    ess_threshold: Annotated[
        int,
        typer.Option(
            min=100,
            help="Minimum ESS per chain (default 400)",
        ),
    ] = 400,
    allow_divergences: bool = typer.Option(
        False,
        "--allow-divergences",
        help="Don't fail on divergences (for exploratory runs)",
    ),
) -> None:
    """Run model training stage only.

    Fits Bayesian models on training data using NumPyro MCMC.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["train"],
        strict=strict,
        verbose=verbose,
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        allow_divergences=allow_divergences,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("evaluate")
def stage_evaluate(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run evaluation stage only.

    Computes model diagnostics, calibration metrics, and LOO-CV.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["evaluate"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("predict")
def stage_predict(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run next-album prediction stage only.

    Generates predictions for known and new artists under multiple scenarios.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["predict"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("report")
def stage_report(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail if publication-readiness checks are not satisfied",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run report generation stage only.

    Generates publication artifacts: figures, tables, and model cards.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["report"],
        strict=strict,
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("sensitivity")
def stage_sensitivity(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    num_chains: Annotated[
        int,
        typer.Option(min=1, help="MCMC chains per sensitivity refit (default 2)"),
    ] = 2,
    num_samples: Annotated[
        int,
        typer.Option(min=100, help="Post-warmup samples per refit (default 500)"),
    ] = 500,
    num_warmup: Annotated[
        int,
        typer.Option(min=50, help="Warmup iterations per refit (default 500)"),
    ] = 500,
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
    ),
    exclude_rw_raw_from_collection: bool = typer.Option(
        False,
        "--exclude-rw-raw-from-collection",
        help="Apply the in-sampler memory gate to the sensitivity refits.",
    ),
) -> None:
    """Run the opt-in sensitivity analysis stage.

    Refits the model under prior variants and feature ablations (reusing the
    default-prior fit as the ablation baseline) and scores cold-start
    coverage across artist-disjoint split seeds. Requires train/evaluate
    artifacts; expensive (several MCMC refits at the configured settings).
    """
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["sensitivity"],
        verbose=verbose,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        dataset=dataset,
        exclude_rw_raw_from_collection=exclude_rw_raw_from_collection,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


# Visualization commands
@app.command("export-figures")
def export_figures(
    output_dir: str = typer.Option(
        "reports/interactive", "--output", "-o", help="Output directory"
    ),
    formats: str = typer.Option(
        "svg,png", "--formats", "-f", help="Comma-separated formats (svg,png,pdf)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Figure width in pixels"),
    height: int = typer.Option(600, "--height", help="Figure height in pixels"),
    scale: float = typer.Option(
        2.0, "--scale", "-s", help="Scale factor for raster output (2.0 = ~300dpi)"
    ),
    run_dir: Optional[str] = typer.Option(
        None, "--run", "-r", help="Path to pipeline run directory"
    ),
) -> None:
    """Export all visualization figures to static formats.

    Generates publication-quality SVG and PNG files from the
    interactive dashboard figures.

    Examples:
        panelcast export-figures
        panelcast export-figures --output figs/ --formats svg,png,pdf
        panelcast export-figures --width 1200 --height 800 --scale 3.0
    """
    from pathlib import Path

    import plotly.graph_objects as go

    from panelcast.visualization.charts import (
        create_forest_plot,
        create_predictions_plot,
        create_reliability_plot,
    )
    from panelcast.visualization.dashboard import load_dashboard_data
    from panelcast.visualization.export import ensure_kaleido_chrome, export_all_figures

    # Parse formats
    format_list = tuple(f.strip() for f in formats.split(",") if f.strip())

    # Ensure Kaleido Chrome is available for raster formats
    if any(fmt in ("png", "jpeg", "webp") for fmt in format_list):
        if not ensure_kaleido_chrome():
            typer.echo("Warning: Kaleido Chrome not available, PNG export may fail", err=True)

    # Load data
    run_path = Path(run_dir) if run_dir else None
    data = load_dashboard_data(run_path)

    # Create figures as go.Figure objects (not HTML strings)
    figures: dict[str, go.Figure] = {}

    if data.predictions is not None:
        pred = data.predictions
        required = ["y_true", "y_pred_mean", "y_pred_lower", "y_pred_upper"]
        if all(k in pred for k in required):
            figures["predictions"] = create_predictions_plot(
                pred["y_true"],
                pred["y_pred_mean"],
                pred["y_pred_lower"],
                pred["y_pred_upper"],
            )

    if data.coefficients is not None:
        figures["coefficients"] = create_forest_plot(data.coefficients)

    if data.reliability is not None:
        rel = data.reliability
        required = ["predicted_probs", "observed_freq", "counts"]
        if all(k in rel for k in required):
            figures["reliability"] = create_reliability_plot(
                rel["predicted_probs"],
                rel["observed_freq"],
                rel["counts"],
            )

    # Add trace/posterior plots if idata available
    if data.idata is not None:
        try:
            from panelcast.visualization.charts import create_trace_plot

            posterior = data.idata.posterior
            if hasattr(posterior, "data_vars"):
                var_names = list(posterior.data_vars)
                if var_names:
                    var_name = var_names[0]
                    samples = posterior[var_name].values
                    # Handle multi-dimensional samples
                    if samples.ndim > 2:
                        samples = samples.reshape(samples.shape[0], -1)[:, 0:100]
                    elif samples.ndim == 1:
                        samples = samples.reshape(1, -1)
                    figures["trace"] = create_trace_plot(samples, var_name)
        except Exception as e:  # Broad catch intentional: idata format varies widely
            logger.debug("trace_plot_skipped", reason="unexpected_idata_format", error=str(e))

    if not figures:
        typer.echo("No data available for export. Run pipeline first.", err=True)
        raise typer.Exit(code=1)

    # Export
    output_path = Path(output_dir)
    results = export_all_figures(
        output_dir=output_path,
        figures=figures,
        formats=format_list,
        width=width,
        height=height,
        scale=scale,
    )

    typer.echo(f"Exported {len(results)} figures to {output_path}")
    for name, paths in results.items():
        typer.echo(f"  {name}: {', '.join(p.name for p in paths)}")


@app.command("demo")
def demo(
    descriptor_path: str = typer.Option(
        "examples/aerospace/descriptor.yaml",
        "--descriptor",
        help="Descriptor YAML for the demo dataset.",
    ),
    num_chains: int = typer.Option(1, "--num-chains", min=1, help="MCMC chains (default 1)."),
    num_samples: int = typer.Option(
        300, "--num-samples", min=50, help="Post-warmup samples per chain (default 300)."
    ),
    num_warmup: int = typer.Option(
        300, "--num-warmup", min=50, help="Warmup iterations per chain (default 300)."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Run the whole pipeline end-to-end on the bundled aerospace example.

    A tiny, self-contained demonstration: airframes flying scored test flights,
    selected entirely by a one-file descriptor with zero source changes. Runs
    data → splits → features → train → evaluate → predict → report at small
    scale and finishes with a generated model card under reports/.

    Examples:
        panelcast demo
        panelcast demo --num-chains 2 --num-samples 500
    """
    from pathlib import Path

    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    if not Path(descriptor_path).exists():
        typer.echo(
            f"Error: demo descriptor not found at {descriptor_path}.\n"
            "Regenerate the example with: python scripts/generate_aero_example.py"
        )
        raise typer.Exit(code=1)

    typer.echo(f"Running the panelcast demo on {descriptor_path} (tiny scale)...\n")

    # Tiny, tolerant settings: this is a smoke demonstration, not a publication
    # run, so convergence gates are relaxed and divergences are allowed.
    config = PipelineConfig(
        seed=seed,
        dataset=descriptor_path,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        min_ratings=5,
        max_albums=10,
        min_albums_filter=2,
        rhat_threshold=1.1,
        ess_threshold=100,
        allow_divergences=True,
        strict=False,
        verbose=verbose,
        enforce_lockfile=False,
    )
    exit_code = run_pipeline(config)

    if exit_code == 0:
        typer.echo("\nDemo complete. Generated artifacts:")
        for artifact in (
            "reports/MODEL_CARD.md",
            "reports/tables/metrics_summary.csv",
            "outputs/evaluation/metrics.json",
        ):
            marker = "✓" if Path(artifact).exists() else "·"
            typer.echo(f"  {marker} {artifact}")
    raise typer.Exit(code=exit_code)


@app.command("compare")
def compare(
    baselines: bool = typer.Option(
        False,
        "--baselines",
        help="Fit the baseline predictors and emit the benchmark comparison table.",
    ),
    dataset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
    ),
    output_dir: str = typer.Option(
        "reports/baselines", "--output", "-o", help="Directory for the comparison artifacts."
    ),
    num_samples: int = typer.Option(
        1000, "--num-samples", min=2, help="Predictive samples per baseline for interval scoring."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for predictive sampling."),
    include_bayes: bool = typer.Option(
        True,
        "--bayes/--no-bayes",
        help="Append the current Bayesian model's metrics from outputs/evaluation/metrics.json.",
    ),
) -> None:
    """Benchmark simple baselines against the model on the existing splits.

    Fits global-mean, entity-mean, last-score (persistence), ridge, and gradient
    boosting baselines on the within-entity-temporal and entity-disjoint splits,
    scores them through the same metrics/calibration/CRPS toolkit as the model,
    and writes a populated comparison table (CSV + Markdown + JSON).

    Requires the splits and features stages to have run (run them first with
    `panelcast run --stages splits,features`).

    Examples:
        panelcast compare --baselines
        panelcast compare --baselines --dataset aero --output reports/aero_baselines
    """
    from pathlib import Path

    if not baselines:
        typer.echo("Nothing to do. Pass --baselines to run the baseline benchmark.")
        raise typer.Exit(code=0)

    from panelcast.pipelines.compare_baselines import run_baseline_comparison

    try:
        result = run_baseline_comparison(
            dataset=dataset,
            n_samples=num_samples,
            seed=seed,
            output_dir=Path(output_dir),
            include_bayes=include_bayes,
        )
    except FileNotFoundError as e:
        typer.echo(
            "Error: split/feature artifacts not found. Run "
            "`panelcast run --stages splits,features` first.\n"
            f"  ({e})"
        )
        raise typer.Exit(code=1) from e

    typer.echo(result.table.to_string(index=False))
    typer.echo("")
    for path in result.artifacts:
        typer.echo(f"  wrote {path}")


@app.command("diagnose")
def diagnose(
    eval_dir: str = typer.Option(
        "outputs/evaluation",
        "--eval-dir",
        help="Directory holding diagnostics.json / metrics.json from an evaluate run.",
    ),
    output_dir: str = typer.Option(
        "reports/diagnostics", "--output", "-o", help="Directory for the diagnostics report."
    ),
) -> None:
    """Summarize convergence + PPC over an existing evaluation run.

    Re-presents the two things the review flagged — the convergence gate and the
    posterior-predictive-check p-values — from artifacts the evaluate stage
    already wrote. PPC statistics pinned near 0/1 are flagged as the signature of
    likelihood misspecification. No model refit.

    Examples:
        panelcast diagnose
        panelcast diagnose --eval-dir outputs/2026-06-23_192630/evaluation
    """
    from pathlib import Path

    from panelcast.pipelines.diagnose import run_diagnose

    try:
        report = run_diagnose(eval_dir=Path(eval_dir), output_dir=Path(output_dir))
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e

    typer.echo(f"Verdict: {report.verdict}\n")
    c = report.convergence
    if c:
        rhat = c.get("rhat_max")
        typer.echo("Convergence:")
        typer.echo(f"  status:      {'PASS' if c.get('passed') else 'FAIL'}")
        typer.echo(f"  R-hat (max): {rhat if rhat is not None else 'n/a (single chain)'}")
        ess_min = c.get("ess_bulk_min", "?")
        ess_thr = c.get("ess_threshold", "?")
        typer.echo(f"  ESS bulk:    {ess_min} (>= {ess_thr})")
        typer.echo(f"  divergences: {c.get('divergences', '?')}")
    if report.ppc:
        typer.echo("\nPPC (statistic: p-value [flag]):")
        for row in report.ppc:
            typer.echo(f"  {row['statistic']:<10} {row['p_value']:.3f}  [{row['flag']}]")
    typer.echo("")
    for path in report.artifacts:
        typer.echo(f"  wrote {path}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
