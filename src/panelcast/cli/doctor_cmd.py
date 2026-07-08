"""The `panelcast doctor` command (#162)."""

from __future__ import annotations

import typer

from panelcast.cli import app


@app.command("doctor")
def doctor(
    dataset: str | None = typer.Option(
        None, "--dataset", help="Dataset descriptor to check (bare name or YAML path)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output for CI."),
) -> None:
    """Read-only environment and reproducibility preflight; exit 1 on any FAIL.

    One screen: lockfile, package versions + fingerprint, accelerator, compile
    cache, git state, dataset resolution, data-root stamps, calibration-store
    status, and free disk — each with a fix hint on failure.
    """
    from panelcast.doctor import run_doctor

    if as_json:
        import contextlib
        import sys

        # Checks may emit structlog lines to stdout (e.g. the compile-cache
        # probe); JSON mode must keep stdout parseable for piping.
        with contextlib.redirect_stdout(sys.stderr):
            results = run_doctor(dataset)
    else:
        results = run_doctor(dataset)

    if as_json:
        import json

        typer.echo(
            json.dumps(
                [
                    {"name": r.name, "status": r.status, "detail": r.detail, "hint": r.hint}
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        for r in results:
            typer.echo(f"{r.status:<5} {r.name:<14} {r.detail}")
            if r.hint and r.status != "PASS":
                typer.echo(f"      {'':<14} -> {r.hint}")
        n_fail = sum(1 for r in results if r.status == "FAIL")
        n_warn = sum(1 for r in results if r.status == "WARN")
        typer.echo(
            f"\n{'FAIL' if n_fail else 'OK'}: "
            f"{len(results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail"
        )

    if any(r.status == "FAIL" for r in results):
        raise typer.Exit(code=1)
