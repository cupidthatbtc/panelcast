"""Provenance commands on the `panelcast runs` group (#169).

`runs verify` re-hashes a run's entire provenance chain — recorded outputs,
raw inputs, the shared data-root stamps, and the lockfile — turning the
manifest from a description into a checkable integrity contract.
"""

from __future__ import annotations

from pathlib import Path

import typer

from panelcast.cli import runs_app


def resolve_run_dir(run_id: str, output_base: Path = Path("outputs")) -> Path:
    """Resolve a run id against outputs/, outputs/failed/, or 'latest'."""
    if run_id == "latest":
        from panelcast.paths import resolve_latest

        run_dir = resolve_latest(output_base)
        if run_dir is None:
            raise typer.BadParameter("no latest run recorded", param_hint="RUN_ID")
        return Path(run_dir)
    for candidate in (output_base / run_id, output_base / "failed" / run_id):
        if (candidate / "manifest.json").exists():
            return candidate
    raise typer.BadParameter(
        f"no run '{run_id}' under {output_base}/ or {output_base}/failed/",
        param_hint="RUN_ID",
    )


def _resolve_recorded(path_str: str, run_dir: Path) -> Path:
    """Recorded path, re-rooted at the run dir when the run was moved (failed/)."""
    path = Path(path_str)
    if path.exists():
        return path
    parts = path.parts
    if run_dir.name in parts:
        rerooted = run_dir.joinpath(*parts[parts.index(run_dir.name) + 1 :])
        if rerooted.exists():
            return rerooted
    return path


def _verify_outputs(manifest, run_dir: Path, problems: list[str]) -> None:
    from panelcast.utils.hashing import sha256_path

    if not manifest.output_hashes:
        typer.echo("outputs: no hashes recorded for this run (pre-0.9.0 manifest)")
        return
    for key, recorded in sorted(manifest.output_hashes.items()):
        path_str = manifest.outputs.get(key)
        path = _resolve_recorded(path_str, run_dir) if path_str else None
        if path is None or not path.exists():
            typer.echo(f"MISSING  {key}")
            problems.append(key)
            continue
        try:
            current = sha256_path(path)
        except OSError as exc:
            typer.echo(f"MISSING  {key} ({exc})")
            problems.append(key)
            continue
        if current != recorded:
            typer.echo(f"MODIFIED {key}")
            problems.append(key)
        else:
            typer.echo(f"OK       {key}")


def _verify_inputs(manifest, problems: list[str]) -> None:
    from panelcast.utils.hashing import sha256_path

    for path_str, recorded in sorted(manifest.input_hashes.items()):
        path = Path(path_str)
        if not path.exists():
            typer.echo(f"MISSING  input {path_str}")
            problems.append(path_str)
            continue
        if sha256_path(path) != recorded:
            typer.echo(f"MODIFIED input {path_str} (raw data changed since this run)")
            problems.append(path_str)
        else:
            typer.echo(f"OK       input {path_str}")


def _verify_stamps(manifest, problems: list[str]) -> None:
    from panelcast.pipelines.errors import StaleArtifactError
    from panelcast.pipelines.stamps import verify_stamps

    try:
        verify_stamps(manifest.data_stamps or {}, consumer="runs verify")
    except StaleArtifactError as exc:
        typer.echo(f"STALE    data roots: {exc}")
        problems.append("data_stamps")
    else:
        if manifest.data_stamps:
            typer.echo("OK       data-root stamps")


def _verify_lockfile(manifest, problems: list[str]) -> None:
    from panelcast.utils.environment import verify_environment

    recorded = manifest.environment.pixi_lock_hash
    if recorded is None:
        return
    current = verify_environment().pixi_lock_hash
    if current != recorded:
        typer.echo(f"DRIFTED  pixi.lock (was {recorded[:12]}, now {(current or 'absent')[:12]})")
        problems.append("pixi_lock")
    else:
        typer.echo("OK       pixi.lock")


@runs_app.command("verify")
def runs_verify(
    run_id: str = typer.Argument("latest", help="Run id under outputs/ (or 'latest')."),
    output_base: Path = typer.Option(
        Path("outputs"), "--output-base", help="Run directory root."
    ),
) -> None:
    """Re-hash a run's recorded artifacts and provenance chain; exit 1 on any mismatch.

    Checks, in order: every recorded output hash (OK / MODIFIED / MISSING),
    every recorded raw-input hash, the shared data-root stamps, and the
    pixi.lock hash. Stamps protect the shared data roots *during* a run; this
    protects the entire run directory *after* it, indefinitely.
    """
    from panelcast.pipelines.manifest import load_run_manifest

    run_dir = resolve_run_dir(run_id, output_base)
    manifest = load_run_manifest(run_dir / "manifest.json")
    typer.echo(f"verifying {run_dir}")

    problems: list[str] = []
    _verify_outputs(manifest, run_dir, problems)
    _verify_inputs(manifest, problems)
    _verify_stamps(manifest, problems)
    _verify_lockfile(manifest, problems)

    if problems:
        typer.echo(f"\nFAILED: {len(problems)} mismatch(es)")
        raise typer.Exit(code=1)
    typer.echo("\nPASS: run directory matches its manifest")
