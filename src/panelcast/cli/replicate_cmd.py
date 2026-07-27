"""The `panelcast replicate` command (#272, #276).

Grades a domain's claims against a fitted run's posterior and prints the
verdict table the replication READMEs assemble by hand. Modes:

- ``panelcast replicate <pack-dir>`` — run a domain pack end-to-end: build
  the panel if needed (gated on the manifest's expected_panel), run the
  leakage-safe chain, grade the pack's claims, write results to notes/.
- ``panelcast replicate --all <collection-dir>`` — run every immediate,
  non-template subfolder containing a pack.yaml and print a combined scoreboard.
- ``--models <dir> --claims <yaml>`` — grade an existing fit directly.
- ``--dataset <yaml> --claims <yaml>`` — run the chain, then grade.

Exit code: 0 = every claim met its target grade; 1 = divergences only
(findings, not errors); 2 = a claim failed every rung or a run failed.

`panelcast pack new <name>` scaffolds a skeleton pack.
"""

from __future__ import annotations

from pathlib import Path

import typer

from panelcast.cli import app

pack_app = typer.Typer(help="Domain-pack utilities (#276).")
app.add_typer(pack_app, name="pack")


@app.command("replicate")
def replicate(
    pack_dir: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        help="A domain pack directory (contains pack.yaml).",
    ),
    claims: Path | None = typer.Option(
        None,
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
    all_dir: Path | None = typer.Option(
        None,
        "--all",
        exists=True,
        file_okay=False,
        help="Collection mode: run every immediate subfolder holding a pack.yaml.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--json",
        dir_okay=False,
        help="Also write the verdicts as JSON.",
    ),
) -> None:
    """Evaluate replication claims against a fitted posterior."""
    from rich.console import Console

    console = Console()
    modes = [pack_dir is not None, all_dir is not None, models is not None, dataset is not None]
    if sum(modes) != 1:
        console.print(
            "[bold red]Error:[/bold red] pass exactly one of PACK_DIR, --all, "
            "--models, or --dataset."
        )
        raise typer.Exit(2)

    if claims is not None and (pack_dir is not None or all_dir is not None):
        console.print(
            "[bold red]Error:[/bold red] packs declare their own claims in "
            "pack.yaml — --claims only combines with --models/--dataset."
        )
        raise typer.Exit(2)

    if all_dir is not None:
        if output_json is not None:
            console.print(
                "[bold red]Error:[/bold red] --json is per-run; the collection "
                "scoreboard has no single verdict list. Each pack writes its own "
                "notes/replicate_verdicts.json."
            )
            raise typer.Exit(2)
        raise typer.Exit(_run_collection(all_dir, console))

    if pack_dir is not None:
        verdicts = _run_pack(pack_dir, console)
        _print_verdicts(verdicts, str(pack_dir), console)
        _maybe_write_json(verdicts, output_json, console)
        from panelcast.replicate import exit_code_for

        raise typer.Exit(exit_code_for(verdicts))

    if claims is None:
        console.print(
            "[bold red]Error:[/bold red] --claims is required with --models/--dataset."
        )
        raise typer.Exit(2)

    from panelcast.replicate import evaluate_claims, exit_code_for, load_claims
    from panelcast.replicate.extractors import load_bundle

    claims_file = load_claims(claims)
    if dataset is not None:
        models = _run_chain_for(dataset, console)
    assert models is not None
    verdicts = evaluate_claims(load_bundle(models), claims_file)
    _print_verdicts(verdicts, str(claims), console)
    _maybe_write_json(verdicts, output_json, console)
    raise typer.Exit(exit_code_for(verdicts))


@pack_app.command("new")
def pack_new(
    name: str = typer.Argument(..., help="Pack folder name (kebab/snake case)."),
    parent: Path = typer.Option(
        Path("."), "--parent", exists=True, file_okay=False, help="Where to create the pack."
    ),
) -> None:
    """Scaffold a skeleton domain pack."""
    from rich.console import Console

    from panelcast.replicate.pack import scaffold_pack

    console = Console()
    try:
        created = scaffold_pack(name, parent)
    except (ValueError, FileExistsError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(2) from exc
    console.print(
        f"created {created} — fill pack.yaml, descriptor.yaml, and build.py, "
        f"then run: panelcast replicate {created}"
    )


def _run_pack(pack_dir: Path, console) -> list:
    """Build (if needed), run the chain with the pack's overrides, grade."""
    import dataclasses
    import json

    from panelcast.replicate import evaluate_claims, load_claims
    from panelcast.replicate.extractors import load_bundle
    from panelcast.replicate.pack import ensure_panel, load_pack

    manifest, resolved = load_pack(pack_dir)
    console.print(f"[bold]pack {manifest.name}[/bold] — {manifest.paper.citation}")
    ensure_panel(manifest, resolved)
    models_dir = _run_chain_for(
        str(resolved / manifest.descriptor),
        console,
        fit_config=(resolved / manifest.fit) if manifest.fit else None,
        overrides=manifest.run,
    )
    if manifest.claims is None:
        console.print("pack declares no claims.yaml — chain ran, nothing to grade.")
        return []
    verdicts = evaluate_claims(
        load_bundle(models_dir), load_claims(resolved / manifest.claims)
    )
    notes_dir = resolved / "notes"
    notes_dir.mkdir(exist_ok=True)
    results_path = notes_dir / "replicate_verdicts.json"
    results_path.write_text(
        json.dumps([dataclasses.asdict(v) for v in verdicts], indent=2), encoding="utf-8"
    )
    console.print(f"verdicts written to {results_path}")
    return verdicts


def _run_collection(collection_dir: Path, console) -> int:
    """Run every pack under the collection and print one scoreboard."""
    from rich.table import Table

    from panelcast.replicate import exit_code_for

    pack_dirs = sorted(
        child
        for child in collection_dir.iterdir()
        if not child.name.startswith("_") and (child / "pack.yaml").exists()
    )
    if not pack_dirs:
        console.print(f"[bold red]Error:[/bold red] no pack.yaml under {collection_dir}/*.")
        return 2

    scoreboard = Table(title=f"Replication scoreboard — {collection_dir}")
    for column in ("pack", "claims", "pass", "divergence", "fail", "exit"):
        scoreboard.add_column(column)
    worst = 0
    for pack_dir in pack_dirs:
        try:
            verdicts = _run_pack(pack_dir, console)
        except Exception as exc:  # noqa: BLE001 — one pack must not sink the rest
            code = getattr(exc, "exit_code", 2) or 2
            # Surface what actually happened: a crash must be
            # distinguishable from a legitimate fit failure.
            console.print(f"[red]{pack_dir.name}: {type(exc).__name__}: {exc}[/red]")
            scoreboard.add_row(pack_dir.name, "—", "—", "—", "—", f"[red]{code}[/red]")
            worst = max(worst, 2)
            continue
        counts = {
            kind: sum(1 for v in verdicts if v.verdict == kind)
            for kind in ("PASS", "DIVERGENCE", "FAIL")
        }
        code = exit_code_for(verdicts)
        worst = max(worst, code)
        scoreboard.add_row(
            pack_dir.name,
            str(len(verdicts)),
            str(counts["PASS"]),
            str(counts["DIVERGENCE"]),
            str(counts["FAIL"]),
            str(code),
        )
    console.print(scoreboard)
    return worst


def _print_verdicts(verdicts: list, title: str, console) -> None:
    from rich.table import Table

    table = Table(title=f"Replication verdicts — {title}")
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


def _maybe_write_json(verdicts: list, output_json: Path | None, console) -> None:
    import dataclasses
    import json

    if output_json is None:
        return
    payload = [dataclasses.asdict(v) for v in verdicts]
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"verdicts written to {output_json}")


def _run_chain_for(
    dataset: str,
    console,
    fit_config: Path | None = None,
    overrides: dict | None = None,
) -> Path:
    """Run data->train for the dataset and return the fresh models directory."""
    from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

    config_kwargs: dict = {
        "dataset": dataset,
        "stages": ["data", "splits", "features", "train"],
    }
    if fit_config is not None:
        from panelcast.config.loader import load_yaml_config
        from panelcast.config.pipeline_yaml import apply_yaml_overrides

        yaml_data = load_yaml_config([str(fit_config)])
        # The pack's dataset/stages are authoritative over its fit.yaml.
        config_kwargs = apply_yaml_overrides(
            config_kwargs, yaml_data, {"dataset", "stages"}
        )
    if overrides:
        config_kwargs.update(overrides)
    config = PipelineConfig(**config_kwargs)
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
