"""The `panelcast preflight` command: pre-fit prior/data statistical checks.

Distinct from `panelcast run --preflight` (GPU-memory estimation). This one
audits the statistics of the fit — prior/data scale and covariate collinearity
given entity intercepts — reading the prepared splits + feature matrices.
"""

from __future__ import annotations

import typer

from panelcast.cli import app


@app.command("preflight")
def preflight(
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help=(
            "Dataset descriptor: bare name (configs/datasets/{name}.yaml) or YAML "
            "path. Omit for built-in AOTY defaults."
        ),
    ),
    config_files: list[str] | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Fit-config YAML(s), same as `panelcast run`, so priors match the fit.",
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit nonzero if any check FAILs (default: warn-only, exit 0)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output for CI."),
) -> None:
    """Pre-fit statistical sanity check on the prepared data (after features).

    Two checks, warn-only by default:
      A. resolved sigma_rw / sigma_artist prior medians vs the data moments they
         govern, on the model-training scale;
      B. condition number of the within-entity demeaned, standardized covariate
         matrix (plus cohort dummies when group pooling is active) — catches the
         age-period-cohort rank deficiency per-entity intercepts hide.

    Never touches the GPU or MCMC. Run it before the first fit on a new domain.
    """
    from panelcast.model_preflight import run_model_preflight

    if as_json:
        import contextlib
        import sys

        with contextlib.redirect_stdout(sys.stderr):
            results = run_model_preflight(dataset, config_files)
    else:
        results = run_model_preflight(dataset, config_files)

    if as_json:
        import json

        typer.echo(
            json.dumps(
                [
                    {
                        "name": r.name,
                        "status": r.status,
                        "detail": r.detail,
                        "suggestion": r.suggestion,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        for r in results:
            typer.echo(f"{r.status:<5} {r.name:<18} {r.detail}")
            if r.suggestion and r.status != "PASS":
                indented = "\n".join(f"        {line}" for line in r.suggestion.splitlines())
                typer.echo(indented)
        n_fail = sum(1 for r in results if r.status == "FAIL")
        n_warn = sum(1 for r in results if r.status == "WARN")
        typer.echo(
            f"\n{'FAIL' if n_fail else 'OK'}: "
            f"{len(results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail"
        )

    if strict and any(r.status == "FAIL" for r in results):
        raise typer.Exit(code=1)
