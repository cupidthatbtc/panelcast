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

import typer

app = typer.Typer(
    add_completion=False,
    help="panelcast - hierarchical Bayesian prediction for bounded scores over entity histories.",
    invoke_without_command=True,
)

# Stage subcommand group
stage_app = typer.Typer(help="Run individual pipeline stages")
app.add_typer(stage_app, name="stage")

# Import the command submodules for their decorator side effects: importing each
# one runs the @app.command / @stage_app.command decorators that register the
# subcommands onto the shared app / stage_app created above. Order matches the
# original definition order so ``--help`` lists the commands unchanged.
# isort: off
from panelcast.cli import main as _main  # noqa: E402
from panelcast.cli import run as _run  # noqa: E402, F401
from panelcast.cli import stages as _stages  # noqa: E402, F401
from panelcast.cli import commands as _commands  # noqa: E402, F401
# isort: on

__version__ = _main.__version__
main = _main.main

__all__ = ["__version__", "app", "main", "stage_app"]
