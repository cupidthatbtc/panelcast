"""The `panelcast replicate` command (#272).

Grades a domain's claims.yaml against a fitted run's posterior and prints
the verdict table the replication READMEs assemble by hand. Two modes:

- ``--models <dir>`` grades an existing fit's artifacts directly.
- ``--dataset <yaml>`` first runs the full leakage-safe chain (data through
  train, preflight and convergence gates included), then grades the fresh
  fit.

Exit code: 0 = every claim met its target grade; 1 = divergences only
(findings, not errors); 2 = a claim failed every rung or the run itself
failed.
"""

from __future__ import annotations

from pathlib import Path

import typer

from panelcast.cli import app


@app.command("replicate")
def replicate(
    claims: Path = typer.Option(
        ...,
        "--claims",
        exists=True,
        dir_okay=False,
        help="claims.yaml declaring the paper's quantitative claims.",
    ),
    models: Path | None = typer.Option(
        None,
        "--models",
        exists=True,
        file_okay=False,
        help="Models directory of an existing fit (training_summary.json + .nc).",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor: run the full chain first, then grade the fresh fit.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--json",
        dir_okay=False,
        help="Also write the verdicts as JSON.",
    ),
) -> None:
    """Evaluate a domain's replication claims against a fitted posterior."""
    import dataclasses
    import json

    from rich.console import Console
    from rich.table import Table

    from panelcast.replicate import evaluate_claims, exit_code_for, load_claims
    from panelcast.replicate.extractors import load_bundle

    console = Console()
    if (models is None) == (dataset is None):
        console.print("[bold red]Error:[/bold red] pass exactly one of --models or --dataset.")
        raise typer.Exit(2)

    claims_file = load_claims(claims)

    if dataset is not None:
        models = _run_chain_for(dataset, console)

    assert models is not None
    verdicts = evaluate_claims(load_bundle(models), claims_file)

    table = Table(title=f"Replication verdicts — {claims}")
    for column in ("claim", "quantity", "expected", "observed", "grade", "verdict"):
        table.add_column(column)
    styles = {"PASS": "green", "DIVERGENCE": "yellow", "FAIL": "red"}
    for v in verdicts:
        table.add_row(
            v.name,
            v.quantity,
            v.expected,
            v.observed,
            f"{v.achieved} (target {v.target})",
            f"[{styles[v.verdict]}]{v.verdict}[/{styles[v.verdict]}]",
        )
        if v.detail:
            table.add_row("", f"[dim]{v.detail}[/dim]", "", "", "", "")
    console.print(table)

    if output_json is not None:
        payload = [dataclasses.asdict(v) for v in verdicts]
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"verdicts written to {output_json}")

    raise typer.Exit(exit_code_for(verdicts))


def _run_chain_for(dataset: str, console) -> Path:
    """Run data->train for the dataset and return the fresh models directory."""
    from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

    config = PipelineConfig(
        dataset=dataset,
        stages=["data", "splits", "features", "train"],
    )
    orchestrator = PipelineOrchestrator(config)
    exit_code = orchestrator.run()
    if exit_code != 0:
        console.print("[bold red]Error:[/bold red] the pipeline run failed; no fit to grade.")
        raise typer.Exit(2)
    run_dir = orchestrator.run_dir
    if run_dir is None:
        raise typer.Exit(2)
    models_dir = Path(run_dir) / "models"
    if not models_dir.exists():
        # Never fall back to a repo-level models/ here: grading a stale fit
        # while claiming it is fresh would be silently wrong.
        console.print(
            f"[bold red]Error:[/bold red] the run produced no models at {models_dir}."
        )
        raise typer.Exit(2)
    console.print(f"grading the fresh fit at {models_dir}")
    return models_dir
