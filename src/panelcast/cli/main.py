"""Top-level ``panelcast`` callback and process entry point."""

from __future__ import annotations

import typer

from panelcast.cli import app

__version__ = "0.2.1"

# Quick preflight uses fixed estimates rather than loading actual data (~1s vs ~30-60s).
# These are conservative defaults for the memory estimation formula.
# For accurate checking with real data dimensions, use --preflight-full.
QUICK_PREFLIGHT_OBSERVATIONS = 1000  # Conservative observation count
QUICK_PREFLIGHT_FEATURES = 20  # Typical feature count from feature builder
QUICK_PREFLIGHT_ARTISTS = 100  # Moderate artist count


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


def main() -> None:
    """Entry point for CLI."""
    # Re-read ``app`` from the package at call time so tests can monkeypatch
    # ``panelcast.cli.app`` (the module-level import above binds at import time).
    from panelcast.cli import app

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
