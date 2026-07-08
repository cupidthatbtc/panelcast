"""The `panelcast stack` command (#154).

Predictive stacking over a completed sweep's arm ledger: fits stacking and
pseudo-BMA+ weights from the persisted per-point elpd snapshots, scores the
arm mixture against the champion and reference on each split with a
predictive snapshot, and writes the report next to the ledger. The headline
is only ever the split the weights were NOT fit on.
"""

from __future__ import annotations

from pathlib import Path

import typer

from panelcast.cli import app


@app.command("stack")
def stack(
    sweep_dir: str = typer.Argument(
        ..., help="Sweep directory (outputs/select/<sweep-id>) containing ledger.json."
    ),
    baselines: str | None = typer.Option(
        None,
        "--baselines",
        help="baseline_comparison.json rows for a baseline-floor section in the report.",
    ),
    seed: int = typer.Option(
        0, "--seed", help="Bayesian-bootstrap seed for the pseudo-BMA+ weights."
    ),
    out_dir: str | None = typer.Option(
        None, "--out-dir", help="Report destination (defaults to the sweep directory)."
    ),
) -> None:
    """Stack a sweep's arms into a weighted mixture and score it honestly.

    Examples:
        panelcast stack outputs/select/sweep
        panelcast stack outputs/select/sweep --baselines outputs/.../baseline_comparison.json
    """
    from panelcast.select.stacking import run_stack

    sweep_path = Path(sweep_dir)
    if not (sweep_path / "ledger.json").exists():
        raise typer.BadParameter(f"no ledger.json in {sweep_dir}", param_hint="SWEEP_DIR")
    baselines_path = Path(baselines) if baselines else None
    if baselines_path is not None and not baselines_path.exists():
        raise typer.BadParameter(f"file not found: {baselines}", param_hint="--baselines")
    try:
        result = run_stack(
            sweep_path,
            baselines_path=baselines_path,
            seed=seed,
            out_dir=Path(out_dir) if out_dir else None,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"Stacked {result['n_arms_stacked']} arms ({result['n_excluded']} excluded).")
    typer.echo(f"Report: {result['report_md']}")
    typer.echo(result["verdict"])
