"""The ``panelcast run`` command and its config-layering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from panelcast.cli import app
from panelcast.cli.main import QUICK_PREFLIGHT_FEATURES

if TYPE_CHECKING:
    from panelcast.pipelines.orchestrator import PipelineConfig


def _resolve_effective_config_files(
    preset: str | None,
    config_files: list[str] | None,
) -> list[str]:
    """Resolve ``--preset`` (layered first) + ``--config`` into an ordered list.

    The preset is layered before any ``--config`` files so explicit ``--config``
    files and CLI options still win (later layers override earlier ones). Exits
    with a clear message on an unknown or missing preset.
    """
    from pathlib import Path

    effective: list[str] = []
    if preset is not None:
        valid_presets = ("quick", "dev", "diagnostic", "publication")
        if preset not in valid_presets:
            typer.echo(
                f"Error: unknown --preset '{preset}'. Choose one of: {', '.join(valid_presets)}"
            )
            raise typer.Exit(code=1)
        preset_path = Path("configs") / f"{preset}.yaml"
        if not preset_path.exists():
            # Fall back to the repo-bundled configs (relative to this file) so
            # --preset works when running from a separate domain directory
            # (docs/PORTING.md), not only from the repo root. This file lives at
            # src/panelcast/cli/run.py, so the repo root is parents[3].
            bundled = Path(__file__).resolve().parents[3] / "configs" / f"{preset}.yaml"
            if bundled.exists():
                preset_path = bundled
            else:  # pragma: no cover - every validated preset ships a bundled config
                typer.echo(f"Error: preset config not found at {preset_path}.")
                raise typer.Exit(code=1)
        effective.append(str(preset_path))
    if config_files:
        effective.extend(config_files)
    return effective


def _apply_config_layers(
    ctx: typer.Context,
    config_kwargs: dict,
    effective_config_files: list[str],
) -> dict:
    """Overlay preset/config YAML onto PipelineConfig kwargs (explicit CLI wins)."""
    if not effective_config_files:
        return config_kwargs

    from panelcast.config.loader import load_yaml_config
    from panelcast.config.pipeline_yaml import apply_yaml_overrides

    yaml_data = load_yaml_config(effective_config_files)
    # Params set explicitly on the command line win over YAML. click's
    # parameter-source API lives in internals that have drifted across
    # click/typer versions, so compare the source by its enum member name
    # (stable) rather than importing or identity-checking the enum, and degrade
    # gracefully if the API is unavailable.
    explicit_cli_params: set[str] = set()
    get_source = getattr(ctx, "get_parameter_source", None)
    if get_source is not None:
        for name in ctx.params:
            source = get_source(name)
            if source is not None and getattr(source, "name", "") == "COMMANDLINE":
                explicit_cli_params.add(name)
    return apply_yaml_overrides(config_kwargs, yaml_data, explicit_cli_params)


def _build_stage_config(
    ctx: typer.Context,
    stage_name: str,
    *,
    seed: int,
    verbose: bool,
    dataset: str | None,
    config_files: list[str] | None,
    preset: str | None,
    **extra: object,
) -> PipelineConfig:
    """Build a single-stage ``PipelineConfig`` with ``run``'s option-wiring.

    Shares the ``--preset`` / ``--config`` / ``--dataset`` resolution used by
    ``run`` so per-stage commands honor the same layering. The requested stage
    is forced after the YAML overlay so a config's ``stages`` key cannot
    redirect an explicit ``panelcast stage <name>`` invocation.
    """
    from panelcast.pipelines.orchestrator import PipelineConfig

    config_kwargs: dict = dict(
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        **extra,
    )
    effective_config_files = _resolve_effective_config_files(preset, config_files)
    try:
        config_kwargs = _apply_config_layers(ctx, config_kwargs, effective_config_files)
        config_kwargs["stages"] = [stage_name]
        return PipelineConfig(**config_kwargs)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _validate_run_options(
    chain_method: str,
    n_exponent_prior: str,
    likelihood_family: str,
    calibration_intervals: str,
) -> tuple[str, tuple[float, ...]]:
    """Validate the free-form run options, returning the normalized chain method
    and parsed calibration levels. Raises typer.Exit(1) with a message on any
    invalid value (Typer can't express these constraints declaratively)."""
    valid_chain_methods = ("sequential", "vectorized", "parallel")
    chain_method_normalized = chain_method.lower()
    if chain_method_normalized not in valid_chain_methods:
        typer.echo(
            f"Error: Invalid --chain-method '{chain_method}'. "
            f"Must be one of: {', '.join(valid_chain_methods)}"
        )
        raise typer.Exit(code=1)

    valid_priors = ("logit-normal", "beta")
    if n_exponent_prior not in valid_priors:
        typer.echo(
            f"Error: Invalid --n-exponent-prior '{n_exponent_prior}'. "
            f"Must be one of: {', '.join(valid_priors)}"
        )
        raise typer.Exit(code=1)

    # Single source of truth: the likelihood registry.
    from panelcast.models.bayes.likelihoods import REGISTRY

    valid_families = tuple(REGISTRY)
    if likelihood_family not in valid_families:
        typer.echo(
            f"Error: Invalid --likelihood-family '{likelihood_family}'. "
            f"Must be one of: {', '.join(valid_families)}"
        )
        raise typer.Exit(code=1)

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

    return chain_method_normalized, calibration_levels


def _run_full_preflight(
    config, *, recalibrate: bool, preflight_only: bool, force_run: bool
) -> None:
    """Run the calibrated full preflight (--preflight-full). Raises typer.Exit
    on preflight-only or on a failure not overridden by --force-run."""
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

    preflight_descriptor = load_descriptor(config.dataset)

    # Load data and build model_args using shared function
    model_args, _, _ = load_training_data(
        features_path=features_path,
        splits_path=splits_path,
        min_albums_filter=config.min_albums_filter,
        descriptor=preflight_descriptor,
    )

    # Remove artist_album_counts (not needed for preflight)
    model_args.pop("artist_album_counts", None)

    # Apply max_albums cap to model_args["album_seq"]
    album_seq = model_args["album_seq"]
    model_args["album_seq"] = np.clip(album_seq, 1, config.max_albums)
    model_args["max_seq"] = config.max_albums

    # Add heteroscedastic params (use effective-config values)
    model_args["n_exponent"] = config.n_exponent
    model_args["learn_n_exponent"] = config.learn_n_exponent
    model_args["likelihood_df"] = config.likelihood_df

    # Use shared dimension derivation for consistent validation
    n_observations, n_artists_dim, n_features, _ = _derive_dimensions_from_model_args(
        model_args
    )

    # Target is POST-WARMUP samples per chain: warmup draws are never
    # stored (measured: identical peaks at warmup 50 vs 250), and the
    # calibration runs at the production chain count so multi-chain
    # accumulation is measured rather than modeled.
    target_samples = config.num_samples

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
        "latent_process": config.latent_process,
        "target_transform": config.target_transform,
        "n_exponent": config.n_exponent,
        "learn_n_exponent": config.learn_n_exponent,
        "likelihood_df": config.likelihood_df,
        "likelihood_family": config.likelihood_family,
        "discretize_observation": config.discretize_observation,
        "exclude_rw_raw_from_collection": config.exclude_rw_raw_from_collection,
    }
    # Mirror the production fit's memory gate in the calibration runs:
    # with the rw_raw exclusion on, the dominant memory term disappears
    # and the projection must reflect that.
    preflight_exclude_collection = (
        (f"{preflight_descriptor.model_prefix}_rw_raw",)
        if config.exclude_rw_raw_from_collection
        else ()
    )

    with console.status(progress_msg):
        full_result = run_extrapolated_preflight_check(
            model_args=model_args,
            target_samples=target_samples,
            n_observations=n_observations,
            n_artists=n_artists_dim,
            n_features=n_features,
            max_seq=config.max_albums,
            headroom_target=0.20,
            # Sequential chains run one after another in the calibration
            # mini-runs; scale the per-run timeout with the chain count.
            timeout_seconds=120 * max(1, config.num_chains),
            recalibrate=recalibrate,
            model_signature=model_signature,
            exclude_collection=preflight_exclude_collection,
            num_chains=config.num_chains,
        )

    render_extrapolation_result(full_result, verbose=config.verbose)

    if preflight_only:
        raise typer.Exit(full_result.exit_code)

    if full_result.status == PreflightStatus.FAIL:
        if force_run:
            typer.echo("Warning: Continuing despite preflight failure (--force-run)")
        else:
            typer.echo("Use --force-run to override preflight failure")
            raise typer.Exit(full_result.exit_code)


def _run_quick_preflight(config, *, preflight_only: bool, force_run: bool) -> None:
    """Run the quick estimation preflight (--preflight). Raises typer.Exit on
    preflight-only or on a failure not overridden by --force-run."""
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
    # AOTY's. config.dataset=None -> AOTY defaults.
    quick_descriptor = load_descriptor(config.dataset)
    quick_csv_path = os.environ.get(
        quick_descriptor.raw_path_env, quick_descriptor.raw_path_default
    )
    # Mirror the orchestrator's resolution so the preflight reads the same
    # threshold a default run would use.
    effective_min_ratings = (
        config.min_ratings
        if config.min_ratings is not None
        else quick_descriptor.primary_min_obs
    )

    # Extract actual data dimensions (graceful fallback to defaults)
    dimensions = extract_data_dimensions(
        csv_path=quick_csv_path,
        min_ratings=effective_min_ratings,
        descriptor=quick_descriptor,
    )

    result = run_preflight_check(
        n_observations=dimensions.n_observations,
        n_features=QUICK_PREFLIGHT_FEATURES,  # Features built from columns, not counted
        n_artists=dimensions.n_artists,
        max_seq=config.max_albums,
        num_chains=config.num_chains,
        num_samples=config.num_samples,
        num_warmup=config.num_warmup,
        exclude_rw_raw_from_collection=config.exclude_rw_raw_from_collection,
    )

    render_preflight_result(result, verbose=config.verbose, dimensions=dimensions)

    if preflight_only:
        raise typer.Exit(code=result.exit_code)

    if result.status == PreflightStatus.FAIL:
        if force_run:
            typer.echo("Warning: Continuing despite preflight failure (--force-run)")
        else:
            typer.echo("Aborting. Use --force-run to override.")
            raise typer.Exit(code=result.exit_code)


@app.command("run")
def run(
    ctx: typer.Context,
    config_files: list[str] | None = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "YAML config file(s) with PipelineConfig keys (e.g. "
            "configs/publication.yaml). Repeatable; later files override "
            "earlier ones. Explicit CLI options always win over YAML."
        ),
    ),
    preset: str | None = typer.Option(
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
    stages: str | None = typer.Option(
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
    resume: str | None = typer.Option(
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
        int | None,
        typer.Option(
            min=1,
            help=(
                "Minimum primary observations per event. Omit to use the "
                "dataset descriptor's primary_min_obs (10 for the AOTY default)."
            ),
        ),
    ] = None,
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
            "'split_normal' (two-piece); 'beta' (bounded mean-precision Beta); or "
            "'beta_binomial' (target as the mean of n aggregated ratings)."
        ),
    ),
    discretize_observation: bool = typer.Option(
        False,
        "--discretize-observation",
        help=(
            "Make the observation integer-aware via dequantization: condition on "
            "y + u, u ~ Uniform(-0.5, 0.5) a single fixed jitter, and round "
            "replicated draws (honest PPC for integer-valued scores). "
            "Location-scale families only (studentt, normal, skew_normal, "
            "split_normal); rejected for beta."
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
    dataset: str | None = typer.Option(
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

    chain_method, calibration_levels = _validate_run_options(
        chain_method, n_exponent_prior, likelihood_family, calibration_intervals
    )

    # Build the effective config (CLI + --preset/--config overlay) BEFORE the
    # preflight branches so --preflight[-full] check the merged dataset, sizes,
    # and model gates rather than raw CLI defaults. PipelineConfig only
    # validates here; the heavy run still happens after preflight.
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

    # Resolve --preset (layered FIRST) + --config, then overlay them so explicit
    # --config files and CLI options still win (later layers override earlier).
    effective_config_files = _resolve_effective_config_files(preset, config_files)
    try:
        config_kwargs = _apply_config_layers(ctx, config_kwargs, effective_config_files)
        config = PipelineConfig(**config_kwargs)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    # Full preflight mode (--preflight-full) takes precedence over quick preflight
    if preflight_full:
        _run_full_preflight(
            config, recalibrate=recalibrate, preflight_only=preflight_only, force_run=force_run
        )

    # Run preflight check if requested (quick estimation mode)
    # Skip quick preflight when full preflight was already run (--preflight-full takes precedence)
    if (preflight or preflight_only) and not preflight_full:
        _run_quick_preflight(config, preflight_only=preflight_only, force_run=force_run)

    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)
