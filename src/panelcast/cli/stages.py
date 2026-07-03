"""Per-stage ``panelcast stage <name>`` subcommands."""

from __future__ import annotations

from typing import Annotated

import typer

from panelcast.cli import stage_app
from panelcast.cli.run import _build_stage_config

# Shared option declarations for the per-stage commands. Each stage accepts the
# same --dataset / --config / --preset wiring as `run` so a single stage can be
# driven against a domain descriptor or a config preset (issue 2d).
_STAGE_DATASET_OPTION = typer.Option(
    None,
    "--dataset",
    help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
)
_STAGE_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="YAML config file(s) with PipelineConfig keys. Repeatable; later files win.",
)
_STAGE_PRESET_OPTION = typer.Option(
    None,
    "--preset",
    help="Named config preset {quick,dev,diagnostic,publication}, layered first.",
)
_STAGE_NO_PROGRESS_OPTION = typer.Option(
    False,
    "--no-progress",
    help=(
        "Disable MCMC progress bars. Without this flag they are shown "
        "only when stderr is a TTY (piped/redirected logs stay readable)."
    ),
)
_STAGE_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Show the execution plan without running the stage.",
)


# Individual stage commands
@stage_app.command("data")
def stage_data(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run data preparation stage only.

    Loads raw event data, applies cleaning transformations, and creates
    processed datasets at the descriptor's observation thresholds.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "data",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("splits")
def stage_splits(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run split creation stage only.

    Creates train/validation/test splits using within-entity temporal
    and entity-disjoint strategies.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "splits",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("features")
def stage_features(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run feature building stage only.

    Builds feature matrices from split data using configured feature blocks.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "features",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("train")
def stage_train(
    ctx: typer.Context,
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
    no_progress: bool = _STAGE_NO_PROGRESS_OPTION,
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run model training stage only.

    Fits Bayesian models on training data using NumPyro MCMC.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "train",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        strict=strict,
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        allow_divergences=allow_divergences,
        progress_bar=False if no_progress else None,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("evaluate")
def stage_evaluate(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run evaluation stage only.

    Computes model diagnostics, calibration metrics, and LOO-CV.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "evaluate",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("predict")
def stage_predict(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run next-event prediction stage only.

    Generates predictions for known and new entities under multiple scenarios.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "predict",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("report")
def stage_report(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail if publication-readiness checks are not satisfied",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    dataset: str | None = _STAGE_DATASET_OPTION,
    config_files: list[str] | None = _STAGE_CONFIG_OPTION,
    preset: str | None = _STAGE_PRESET_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
) -> None:
    """Run report generation stage only.

    Generates publication artifacts: figures, tables, and model cards.
    """
    from panelcast.pipelines.orchestrator import run_pipeline

    config = _build_stage_config(
        ctx,
        "report",
        seed=seed,
        verbose=verbose,
        dataset=dataset,
        config_files=config_files,
        preset=preset,
        strict=strict,
        dry_run=dry_run,
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
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
    ),
    exclude_rw_raw_from_collection: bool = typer.Option(
        False,
        "--exclude-rw-raw-from-collection",
        help="Apply the in-sampler memory gate to the sensitivity refits.",
    ),
    no_progress: bool = _STAGE_NO_PROGRESS_OPTION,
    dry_run: bool = _STAGE_DRY_RUN_OPTION,
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
        progress_bar=False if no_progress else None,
        dry_run=dry_run,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


# Visualization commands
