"""Provenance commands on the `panelcast runs` group (#169, #160).

`runs verify` re-hashes a run's entire provenance chain — recorded outputs,
raw inputs, the shared data-root stamps, and the lockfile — turning the
manifest from a description into a checkable integrity contract. `runs show`
renders one run's full provenance; `runs diff` compares two runs with
defaults-aware flag semantics, generic metric deltas, and run-fact deltas.
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


def _load_metrics(run_dir: Path) -> dict:
    import json

    try:
        return json.loads((run_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _numeric_leaves(payload, prefix: str = "") -> dict[str, float]:
    """Flatten nested dicts to dotted numeric leaves; lists and bools skipped.

    Generic on purpose: new metrics appear in `runs diff` without code changes.
    """
    out: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.update(_numeric_leaves(value, f"{prefix}{key}."))
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        out[prefix[:-1]] = float(payload)
    return out


@runs_app.command("show")
def runs_show(
    run_id: str = typer.Argument("latest", help="Run id under outputs/ (or 'latest')."),
    output_base: Path = typer.Option(
        Path("outputs"), "--output-base", help="Run directory root."
    ),
) -> None:
    """Render one run's full provenance: command, config, git, environment, metrics."""
    from panelcast.pipelines.manifest import load_run_manifest

    run_dir = resolve_run_dir(run_id, output_base)
    m = load_run_manifest(run_dir / "manifest.json")

    typer.echo(f"run       {m.run_id}  ({'ok' if m.success else 'incomplete/failed'})")
    typer.echo(f"created   {m.created_at}   duration {m.duration_seconds:.0f}s")
    if m.version or m.tag:
        typer.echo(f"version   {m.version or '?'}   tag {m.tag or '-'}")
    typer.echo(f"command   {m.command}")
    typer.echo(f"seed      {m.seed}")
    dataset = m.flags.get("dataset") or "aoty"
    descriptor_hash = m.flags.get("dataset_descriptor_hash")
    descriptor_note = f"   descriptor {descriptor_hash}" if descriptor_hash else ""
    typer.echo(f"dataset   {dataset}{descriptor_note}")
    typer.echo(
        f"git       {m.git.commit[:7]} on {m.git.branch}"
        + ("  DIRTY" if m.git.dirty else "")
        + (f"  ({m.git.untracked_count} untracked)" if m.git.untracked_count else "")
    )
    env = m.environment
    typer.echo(
        f"env       python {env.python_version}, jax {env.jax_version}"
        + (f"/{env.jaxlib_version}" if env.jaxlib_version else "")
        + f", numpyro {env.numpyro_version or '?'}"
    )
    if env.accelerator:
        typer.echo(f"device    {env.accelerator}: {env.device_kind or '?'}")
    if env.fingerprint:
        typer.echo(f"exactness fingerprint {env.fingerprint}")
    if env.pixi_lock_hash:
        typer.echo(f"lockfile  {env.pixi_lock_hash[:12]}")

    durations = m.stage_durations or {}
    completed = ", ".join(
        f"{s} ({durations[s]:.0f}s)" if s in durations else s for s in m.stages_completed
    )
    typer.echo(f"stages    {completed or '-'}")
    if m.stages_skipped:
        typer.echo(f"skipped   {', '.join(m.stages_skipped)}")
    typer.echo(f"outputs   {len(m.outputs)} recorded, {len(m.output_hashes)} hashed")
    if m.error:
        typer.echo(f"error     {m.error}")

    metrics = _load_metrics(run_dir)
    if metrics:
        from panelcast.cli.commands import _history_metrics

        h = _history_metrics(metrics)
        cov = ", ".join(
            f"{level}: {value:.3f}"
            for level, value in sorted((h.get("coverage") or {}).items())
            if value is not None
        )
        typer.echo(
            "metrics   "
            + "  ".join(
                part
                for part in (
                    f"mae {h['mae']:.3f}" if h.get("mae") is not None else None,
                    f"rmse {h['rmse']:.3f}" if h.get("rmse") is not None else None,
                    f"r2 {h['r2']:.3f}" if h.get("r2") is not None else None,
                    f"crps {h['crps']:.3f}" if h.get("crps") is not None else None,
                    (
                        f"elpd/obs {h['elpd_per_obs']:.4f}"
                        if h.get("elpd_per_obs") is not None
                        else None
                    ),
                )
                if part
            )
        )
        if cov:
            typer.echo(f"coverage  {cov}")


@runs_app.command("diff")
def runs_diff(
    run_a: str = typer.Argument(..., help="Left run id (or 'latest')."),
    run_b: str = typer.Argument(..., help="Right run id (or 'latest')."),
    output_base: Path = typer.Option(
        Path("outputs"), "--output-base", help="Run directory root."
    ),
) -> None:
    """Compare two runs: defaults-aware config delta, metric deltas, run facts.

    Emphasizes when the runs differ in dataset descriptor hash or git commit —
    such a diff is not a like-for-like comparison.
    """
    from panelcast.pipelines.manifest import flag_differences, load_run_manifest
    from panelcast.pipelines.orchestrator import PipelineConfig

    dir_a = resolve_run_dir(run_a, output_base)
    dir_b = resolve_run_dir(run_b, output_base)
    a = load_run_manifest(dir_a / "manifest.json")
    b = load_run_manifest(dir_b / "manifest.json")
    typer.echo(f"A: {a.run_id}\nB: {b.run_id}")

    # Not like-for-like guards first: these invalidate any metric comparison.
    if a.flags.get("dataset_descriptor_hash") != b.flags.get("dataset_descriptor_hash"):
        typer.echo("\nWARNING: dataset descriptor hashes differ — not a like-for-like comparison")
    if a.git.commit != b.git.commit:
        typer.echo(f"\ngit: {a.git.commit[:7]} vs {b.git.commit[:7]}")

    changed = flag_differences(a.flags, b.flags, PipelineConfig())
    typer.echo("\nconfig delta (output-affecting, defaults-aware):")
    if not changed:
        typer.echo("  none")
    for key, va, vb in changed:
        typer.echo(f"  {key}: {va} -> {vb}")

    leaves_a = _numeric_leaves(_load_metrics(dir_a))
    leaves_b = _numeric_leaves(_load_metrics(dir_b))
    keys = sorted(set(leaves_a) | set(leaves_b))
    typer.echo("\nmetric deltas (B - A):")
    if not keys:
        typer.echo("  no metrics on either side")
    for key in keys:
        va, vb = leaves_a.get(key), leaves_b.get(key)
        if va is None:
            typer.echo(f"  {key}: missing on A (B: {vb:.4g})")
        elif vb is None:
            typer.echo(f"  {key}: missing on B (A: {va:.4g})")
        elif va != vb:
            typer.echo(f"  {key}: {va:.4g} -> {vb:.4g}  ({vb - va:+.4g})")

    env_a, env_b = a.environment, b.environment
    typer.echo("\nrun facts:")
    facts = [
        ("seed", a.seed, b.seed),
        ("fingerprint", env_a.fingerprint, env_b.fingerprint),
        ("jax", env_a.jax_version, env_b.jax_version),
        ("numpyro", env_a.numpyro_version, env_b.numpyro_version),
        ("accelerator", env_a.accelerator, env_b.accelerator),
        ("pixi.lock", (env_a.pixi_lock_hash or "")[:12], (env_b.pixi_lock_hash or "")[:12]),
        ("version", a.version, b.version),
    ]
    same = [name for name, va, vb in facts if va == vb]
    for name, va, vb in facts:
        if va != vb:
            typer.echo(f"  {name}: {va} vs {vb}")
    if same:
        typer.echo(f"  identical: {', '.join(same)}")


def _reproduce_config(run_dir: Path, manifest) -> tuple["object", str]:
    """(PipelineConfig, provenance tier) for re-executing a recorded run."""
    from panelcast.config.pipeline_yaml import load_resolved_config
    from panelcast.pipelines.orchestrator import PipelineConfig

    resolved = run_dir / "resolved_config.yaml"
    if resolved.exists():
        return PipelineConfig(**load_resolved_config(resolved)), "resolved_config.yaml"
    # Pre-0.9.0 fallback: rebuild from the manifest's flags dict. Anything the
    # flags never recorded stays at the current default — weaker provenance.
    config = PipelineConfig()
    for key, value in (manifest.flags or {}).items():
        if key in ("resume", "dataset_descriptor_hash") or not hasattr(config, key):
            continue
        if isinstance(getattr(config, key), tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)
    return config, "manifest flags (pre-0.9.0 run; YAML-only gates may be lost)"


@runs_app.command("reproduce")
def runs_reproduce(
    run_id: str = typer.Argument(..., help="Run id under outputs/ (or 'latest')."),
    output_base: Path = typer.Option(
        Path("outputs"), "--output-base", help="Run directory root for old and new runs."
    ),
) -> None:
    """Re-execute a recorded run from its run directory alone, then compare.

    Guards before any compute: the dataset descriptor must hash-match the
    recorded one, and recorded raw inputs must be unchanged on disk. The
    environment fingerprint frames the expectation up front — bit-exact within
    a matching fingerprint, statistical otherwise — and the post-run
    comparison follows suit (exact output hashes vs headline-metric deltas).
    """
    from panelcast.config.descriptor import load_descriptor
    from panelcast.pipelines.manifest import capture_environment, load_run_manifest
    from panelcast.utils.hashing import sha256_path

    run_dir = resolve_run_dir(run_id, output_base)
    manifest = load_run_manifest(run_dir / "manifest.json")
    config, tier = _reproduce_config(run_dir, manifest)
    typer.echo(f"reproducing {manifest.run_id} from {tier}")

    recorded_hash = manifest.flags.get("dataset_descriptor_hash")
    if recorded_hash:
        current_hash = load_descriptor(config.dataset).descriptor_hash()
        if current_hash != recorded_hash:
            typer.echo(
                f"ABORT: dataset descriptor changed (recorded {recorded_hash}, "
                f"current {current_hash}); reproduction would not test the same model"
            )
            raise typer.Exit(code=1)

    for path_str, recorded in sorted((manifest.input_hashes or {}).items()):
        path = Path(path_str)
        if not path.exists():
            typer.echo(f"ABORT: recorded input missing: {path_str}")
            raise typer.Exit(code=1)
        if sha256_path(path) != recorded:
            typer.echo(f"ABORT: raw input changed since the run: {path_str}")
            raise typer.Exit(code=1)

    old_fingerprint = manifest.environment.fingerprint
    new_fingerprint = capture_environment().fingerprint
    bit_exact = bool(old_fingerprint) and old_fingerprint == new_fingerprint
    if bit_exact:
        typer.echo(f"fingerprint match ({new_fingerprint}) — expecting bit-exact outputs")
    else:
        typer.echo(
            f"fingerprint mismatch ({old_fingerprint or 'unrecorded'} vs {new_fingerprint}) "
            "— expecting statistical reproduction only"
        )

    # A reproduction is a full fresh execution: never resume, never skip.
    config.resume = None
    config.skip_existing = False
    from panelcast.pipelines.orchestrator import run_pipeline

    exit_code = run_pipeline(config, output_base=output_base)
    if exit_code != 0:
        typer.echo(f"reproduction run failed (exit {exit_code})")
        raise typer.Exit(code=exit_code)

    _compare_reproduction(manifest, output_base, bit_exact)


def _compare_reproduction(old_manifest, output_base: Path, bit_exact: bool) -> None:
    from panelcast.paths import resolve_latest
    from panelcast.pipelines.manifest import load_run_manifest

    new_dir = resolve_latest(output_base)
    if new_dir is None or not (Path(new_dir) / "manifest.json").exists():
        typer.echo("no new run manifest resolved; skipping comparison")
        return
    new_dir = Path(new_dir)
    new_manifest = load_run_manifest(new_dir / "manifest.json")
    typer.echo(f"new run: {new_manifest.run_id}")

    if bit_exact and old_manifest.output_hashes:
        old_keys = {k.split(":", 1)[1]: v for k, v in old_manifest.output_hashes.items()}
        new_keys = {k.split(":", 1)[1]: v for k, v in new_manifest.output_hashes.items()}
        shared = sorted(set(old_keys) & set(new_keys))
        exact = [k for k in shared if old_keys[k] == new_keys[k]]
        typer.echo(f"exact-match: {len(exact)}/{len(shared)} shared artifacts bit-identical")
        for key in shared:
            if old_keys[key] != new_keys[key]:
                typer.echo(f"  differs: {key}")
        return

    old_leaves = _numeric_leaves(_load_metrics_from_manifest_dir(old_manifest, output_base))
    new_leaves = _numeric_leaves(_load_metrics(new_dir))
    keys = sorted(set(old_leaves) & set(new_leaves))
    if not keys:
        typer.echo("no shared metrics to compare")
        return
    typer.echo("headline metric deltas (new - old):")
    for key in keys:
        delta = new_leaves[key] - old_leaves[key]
        if delta:
            typer.echo(f"  {key}: {old_leaves[key]:.4g} -> {new_leaves[key]:.4g} ({delta:+.4g})")


def _load_metrics_from_manifest_dir(manifest, output_base: Path) -> dict:
    for root in (output_base, output_base / "failed"):
        candidate = root / manifest.run_id
        if (candidate / "manifest.json").exists():
            return _load_metrics(candidate)
    return {}
