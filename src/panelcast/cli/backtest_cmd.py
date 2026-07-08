"""The `panelcast backtest` command (#179).

Rolling-origin backtest: runs the full leakage-safe stage chain once per
origin and reports every headline metric as mean ± SE across origins. A
killed backtest resumes at the next unfinished origin via the JSON ledger.
"""

from __future__ import annotations

from pathlib import Path

import typer

from panelcast.cli import app


@app.command("backtest")
def backtest(
    origins: int = typer.Option(
        3,
        "--origins",
        min=1,
        help="Number of rolling origins K: origin k holds out each entity's (last-k)-th event.",
    ),
    backtest_id: str = typer.Option(
        "default",
        "--backtest-id",
        help="Ledger/report directory name under outputs/backtest/ (reuse to resume).",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
    ),
    num_chains: int | None = typer.Option(
        None, "--num-chains", min=1, help="Chains per origin fit (default: pipeline default)."
    ),
    num_samples: int | None = typer.Option(
        None, "--num-samples", min=1, help="Draws per origin fit (default: pipeline default)."
    ),
    num_warmup: int | None = typer.Option(
        None, "--num-warmup", min=1, help="Warmup per origin fit (default: pipeline default)."
    ),
    origin_timeout: float | None = typer.Option(
        None,
        "--origin-timeout",
        min=1.0,
        help="Per-origin wall-clock timeout in seconds (default: none).",
    ),
    output_root: str = typer.Option(
        "outputs/backtest", "--output-root", help="Root directory for backtest ledgers/reports."
    ),
) -> None:
    """Run (or resume) a rolling-origin backtest and print the aggregate table.

    Each origin regenerates splits/features with fresh stamps, so the leakage
    controls hold unchanged; every origin's split content hash is recorded in
    the ledger. Deeper origins shrink the eligible entity set — the aggregate
    table reports n_test and n_entities per origin so cross-origin variation
    is framed honestly.

    Examples:
        panelcast backtest --origins 3
        panelcast backtest --origins 5 --num-chains 2 --num-samples 500
        panelcast backtest --backtest-id nightly  # rerun to resume
    """
    from panelcast.pipelines.backtest import BacktestConfig, run_backtest

    cfg = BacktestConfig(
        origins=origins,
        backtest_id=backtest_id,
        output_root=Path(output_root),
        dataset=dataset,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        origin_timeout_seconds=origin_timeout,
    )
    aggregate = run_backtest(cfg)

    typer.echo(
        f"\nBacktest '{backtest_id}': {aggregate['n_origins_completed']}"
        f"/{aggregate['n_origins_requested']} origins completed."
    )
    for name, block in aggregate["metrics"].items():
        if block is None:
            continue
        se = f" ± {block['se']:.4f}" if block["se"] is not None else ""
        typer.echo(f"  {name}: {block['mean']:.4f}{se}  [{block['min']:.4f}, {block['max']:.4f}]")
    typer.echo(f"\n  wrote {cfg.backtest_dir / 'backtest_metrics.json'}")
    typer.echo(f"  wrote {cfg.backtest_dir / 'backtest_report.md'}")
    if aggregate["n_origins_completed"] < aggregate["n_origins_requested"]:
        typer.echo("  incomplete origins remain — rerun the same command to resume.")
        raise typer.Exit(code=1)
