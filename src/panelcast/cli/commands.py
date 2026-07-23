"""Auxiliary ``panelcast`` commands: export-figures, demo, compare, diagnose."""

from __future__ import annotations

import logging

import typer

from panelcast.cli import app, runs_app

logger = logging.getLogger(__name__)


def _artifact_markers(encoding: str | None) -> tuple[str, str]:
    """Found/missing artifact markers, ASCII when the console can't encode them
    (Windows cp1252 stdout crashed on the check mark after a successful demo)."""
    try:
        "✓".encode(encoding or "utf-8")
        return "✓", "·"
    except (UnicodeEncodeError, LookupError):
        return "OK", "--"


def _coefficient_columns(coefficients) -> tuple[str, str, str, str] | None:
    """Resolve (estimate, lower, upper, label) columns of a coefficient table.

    Handles both arviz-summary columns (mean / hdi_3% / hdi_97% / param) and the
    report's human table (Estimate / CI Lower / CI Upper, param in the unnamed
    first column). Returns None when the input isn't a column-bearing table or the
    estimate/interval columns can't be found.
    """
    columns = getattr(coefficients, "columns", None)
    if columns is None:
        return None

    def _norm(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("%", "")

    by_norm = {_norm(c): c for c in columns}

    def pick(candidates: list[str]) -> str | None:
        for cand in candidates:
            if _norm(cand) in by_norm:
                return by_norm[_norm(cand)]
        return None

    estimate = pick(["mean", "estimate", "coef"])
    lower = pick(["hdi_3%", "hdi_2.5%", "ci_lower", "lower"])
    upper = pick(["hdi_97%", "hdi_97.5%", "ci_upper", "upper"])
    if not (estimate and lower and upper):
        return None
    label = pick(["param", "parameter", "name", "index"]) or columns[0]
    return estimate, lower, upper, label


@app.command("export-figures")
def export_figures(
    output_dir: str = typer.Option(
        "reports/interactive", "--output", "-o", help="Output directory"
    ),
    formats: str = typer.Option(
        "svg,png", "--formats", "-f", help="Comma-separated formats (svg,png,pdf)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Figure width in pixels"),
    height: int = typer.Option(600, "--height", help="Figure height in pixels"),
    scale: float = typer.Option(
        2.0, "--scale", "-s", help="Scale factor for raster output (2.0 = ~300dpi)"
    ),
    run_dir: str | None = typer.Option(None, "--run", "-r", help="Path to pipeline run directory"),
) -> None:
    """Export all visualization figures to static formats.

    Generates publication-quality SVG and PNG files from the
    interactive dashboard figures.

    Examples:
        panelcast export-figures
        panelcast export-figures --output figs/ --formats svg,png,pdf
        panelcast export-figures --width 1200 --height 800 --scale 3.0
    """
    from pathlib import Path

    import plotly.graph_objects as go

    from panelcast.visualization.charts import (
        create_forest_plot,
        create_predictions_plot,
        create_reliability_plot,
    )
    from panelcast.visualization.dashboard import _ci_label, load_dashboard_data
    from panelcast.visualization.export import ensure_kaleido_chrome, export_all_figures

    # Parse formats
    format_list = tuple(f.strip() for f in formats.split(",") if f.strip())

    # Ensure Kaleido Chrome is available for raster formats
    if any(fmt in ("png", "jpeg", "webp") for fmt in format_list):
        if not ensure_kaleido_chrome():
            typer.echo("Warning: Kaleido Chrome not available, PNG export may fail", err=True)

    # Load data
    run_path = Path(run_dir) if run_dir else None
    data = load_dashboard_data(run_path)

    # Create figures as go.Figure objects (not HTML strings)
    figures: dict[str, go.Figure] = {}

    if data.predictions is not None:
        pred = data.predictions
        required = ["y_true", "y_pred_mean", "y_pred_lower", "y_pred_upper"]
        if all(k in pred for k in required):
            figures["predictions"] = create_predictions_plot(
                pred["y_true"],
                pred["y_pred_mean"],
                pred["y_pred_lower"],
                pred["y_pred_upper"],
                ci_label=_ci_label(pred.get("interval_level")),
            )

    if data.coefficients is not None:
        cols = _coefficient_columns(data.coefficients)
        if cols is not None:
            estimate_col, lower_col, upper_col, label_col = cols
            figures["coefficients"] = create_forest_plot(
                data.coefficients,
                estimate_col=estimate_col,
                lower_col=lower_col,
                upper_col=upper_col,
                label_col=label_col,
            )
        else:
            typer.echo("Skipping coefficients figure: no estimate/HDI columns in fallback CSV.")

    if data.reliability is not None:
        rel = data.reliability
        required = ["predicted_probs", "observed_freq", "counts"]
        if all(k in rel for k in required):
            figures["reliability"] = create_reliability_plot(
                rel["predicted_probs"],
                rel["observed_freq"],
                rel["counts"],
            )

    # Add trace/posterior plots if idata available
    if data.idata is not None:
        try:
            from panelcast.visualization.charts import create_trace_plot

            posterior = data.idata.posterior
            if hasattr(posterior, "data_vars"):
                var_names = list(posterior.data_vars)
                if var_names:
                    var_name = var_names[0]
                    samples = posterior[var_name].values
                    # Multi-dimensional parameters: trace the first element
                    # only, keeping (chain, draw) intact.
                    if samples.ndim > 2:
                        samples = samples.reshape(samples.shape[0], samples.shape[1], -1)[:, :, 0]
                        var_name = f"{var_name}[0]"
                    elif samples.ndim == 1:
                        samples = samples.reshape(1, -1)
                    figures["trace"] = create_trace_plot(samples, var_name)
        except Exception as e:  # Broad catch intentional: idata format varies widely
            # Stdlib logger here, not structlog — kwargs would TypeError.
            logger.debug("trace_plot_skipped: unexpected idata format (%s)", e)

    if not figures:
        typer.echo("No data available for export. Run pipeline first.", err=True)
        raise typer.Exit(code=1)

    # Export
    output_path = Path(output_dir)
    results = export_all_figures(
        output_dir=output_path,
        figures=figures,
        formats=format_list,
        width=width,
        height=height,
        scale=scale,
    )

    typer.echo(f"Exported {len(results)} figures to {output_path}")
    for name, paths in results.items():
        typer.echo(f"  {name}: {', '.join(p.name for p in paths)}")


@app.command("demo")
def demo(
    descriptor_path: str = typer.Option(
        "aero",
        "--descriptor",
        "--dataset",
        help="Descriptor YAML for the demo dataset (--dataset is an alias, as in run/stage).",
    ),
    num_chains: int = typer.Option(1, "--num-chains", min=1, help="MCMC chains (default 1)."),
    num_samples: int = typer.Option(
        300, "--num-samples", min=50, help="Post-warmup samples per chain (default 300)."
    ),
    num_warmup: int = typer.Option(
        300, "--num-warmup", min=50, help="Warmup iterations per chain (default 300)."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    """Run the whole pipeline end-to-end on the bundled aerospace example.

    A tiny, self-contained demonstration: airframes flying scored test flights,
    selected entirely by a one-file descriptor with zero source changes. Runs
    data → splits → features → train → evaluate → predict → report at small
    scale and finishes with a generated model card under reports/.

    Examples:
        panelcast demo
        panelcast demo --num-chains 2 --num-samples 500
    """
    from pathlib import Path

    from panelcast.config.descriptor import resolve_demo_descriptor_path
    from panelcast.pipelines.orchestrator import PipelineConfig, run_pipeline

    resolved_descriptor = resolve_demo_descriptor_path(descriptor_path)
    if not resolved_descriptor.exists():
        typer.echo(f"Error: demo descriptor not found at {descriptor_path}.")
        raise typer.Exit(code=1)
    descriptor_path = str(resolved_descriptor)

    typer.echo(f"Running the panelcast demo on {descriptor_path} (tiny scale)...\n")

    # Tiny, tolerant settings: this is a smoke demonstration, not a publication
    # run, so convergence gates are relaxed and divergences are allowed.
    config = PipelineConfig(
        seed=seed,
        dataset=descriptor_path,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        # min_ratings unset: resolves to the aerospace descriptor's
        # primary_min_obs (5) in the orchestrator.
        max_albums=10,
        min_albums_filter=2,
        rhat_threshold=1.1,
        ess_threshold=100,
        allow_divergences=True,
        strict=False,
        verbose=verbose,
        enforce_lockfile=False,
    )
    exit_code = run_pipeline(config)

    if exit_code == 0:
        from panelcast.paths import resolve_latest

        run_dir = resolve_latest()
        if run_dir is not None:
            artifacts = (
                run_dir / "reports" / "MODEL_CARD.md",
                run_dir / "reports" / "tables" / "metrics_summary.csv",
                run_dir / "evaluation" / "metrics.json",
            )
        else:
            artifacts = (
                Path("reports/MODEL_CARD.md"),
                Path("reports/tables/metrics_summary.csv"),
                Path("outputs/evaluation/metrics.json"),
            )
        import sys

        typer.echo("\nDemo complete. Generated artifacts:")
        found_marker, missing_marker = _artifact_markers(
            getattr(sys.stdout, "encoding", None)
        )
        for artifact in artifacts:
            marker = found_marker if artifact.exists() else missing_marker
            typer.echo(f"  {marker} {artifact}")
    raise typer.Exit(code=exit_code)


@app.command("compare")
def compare(
    baselines: bool = typer.Option(
        False,
        "--baselines",
        help="Fit the baseline predictors and emit the benchmark comparison table.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Directory for the comparison artifacts. Default: the latest run's "
            "reports/baselines dir (flat reports/baselines when no run exists)."
        ),
    ),
    num_samples: int = typer.Option(
        1000, "--num-samples", min=2, help="Predictive samples per baseline for interval scoring."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for predictive sampling."),
    include_bayes: bool = typer.Option(
        True,
        "--bayes/--no-bayes",
        help="Append the current Bayesian model's metrics from the --metrics file.",
    ),
    metrics: str | None = typer.Option(
        None,
        "--metrics",
        help=(
            "Evaluation metrics.json supplying the Bayesian model's row (with "
            "--bayes). Default: the latest run's evaluation metrics."
        ),
    ),
) -> None:
    """Benchmark simple baselines against the model on the existing splits.

    Fits global-mean, entity-mean, last-score (persistence), ridge, and gradient
    boosting baselines on the within-entity-temporal and entity-disjoint splits,
    scores them through the same metrics/calibration/CRPS toolkit as the model,
    and writes a populated comparison table (CSV + Markdown + JSON).

    Requires the splits and features stages to have run (run them first with
    `panelcast run --stages splits,features`).

    Examples:
        panelcast compare --baselines
        panelcast compare --baselines --dataset aero --output reports/aero_baselines
    """
    from pathlib import Path

    if not baselines:
        typer.echo("Nothing to do. Pass --baselines to run the baseline benchmark.")
        raise typer.Exit(code=0)

    from panelcast.pipelines.compare_baselines import run_baseline_comparison

    try:
        result = run_baseline_comparison(
            dataset=dataset,
            n_samples=num_samples,
            seed=seed,
            output_dir=Path(output_dir) if output_dir is not None else None,
            include_bayes=include_bayes,
            metrics_path=Path(metrics) if metrics is not None else None,
        )
    except FileNotFoundError as e:
        typer.echo(
            "Error: split/feature artifacts not found. Run "
            "`panelcast run --stages splits,features` first.\n"
            f"  ({e})"
        )
        raise typer.Exit(code=1) from e

    typer.echo(result.table.to_string(index=False))
    typer.echo("")
    for path in result.artifacts:
        typer.echo(f"  wrote {path}")


@app.command("diagnose")
def diagnose(
    eval_dir: str | None = typer.Option(
        None,
        "--eval-dir",
        help=(
            "Directory holding diagnostics.json / metrics.json from an evaluate "
            "run (default: the latest run's evaluation dir)."
        ),
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Directory for the diagnostics report "
            "(default: reports/diagnostics; run-scoped for --errors)."
        ),
    ),
    errors: bool = typer.Option(
        False,
        "--errors",
        help=(
            "Also decompose per-row errors from the identified predictions "
            "artifact: entity/group/review-count rollups + worst-25 table."
        ),
    ),
) -> None:
    """Summarize convergence + PPC over an existing evaluation run.

    Re-presents the two things the review flagged — the convergence gate and the
    posterior-predictive-check p-values — from artifacts the evaluate stage
    already wrote. PPC statistics pinned near 0/1 are flagged as the signature of
    likelihood misspecification. No model refit.

    Examples:
        panelcast diagnose
        panelcast diagnose --eval-dir outputs/2026-06-23_192630/evaluation
        panelcast diagnose --errors
    """
    from pathlib import Path

    from panelcast.pipelines.diagnose import run_diagnose, run_error_decomposition

    try:
        report = run_diagnose(
            eval_dir=Path(eval_dir) if eval_dir else None,
            output_dir=Path(output_dir) if output_dir else Path("reports/diagnostics"),
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e

    error_artifacts: list[Path] = []
    if errors:
        try:
            error_artifacts = run_error_decomposition(
                eval_dir=Path(eval_dir) if eval_dir else None,
                output_dir=Path(output_dir) if output_dir else None,
            )
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error decomposition unavailable: {e}")
            raise typer.Exit(code=1) from e

    typer.echo(f"Verdict: {report.verdict}\n")
    c = report.convergence
    if c:
        rhat = c.get("rhat_max")
        typer.echo("Convergence:")
        typer.echo(f"  status:      {'PASS' if c.get('passed') else 'FAIL'}")
        typer.echo(f"  R-hat (max): {rhat if rhat is not None else 'n/a (single chain)'}")
        ess_min = c.get("ess_bulk_min", "?")
        ess_thr = c.get("ess_threshold", "?")
        typer.echo(f"  ESS bulk:    {ess_min} (>= {ess_thr})")
        typer.echo(f"  divergences: {c.get('divergences', '?')}")
    if report.ppc:
        typer.echo("\nPPC (statistic: p-value [flag]):")
        for row in report.ppc:
            typer.echo(f"  {row['statistic']:<10} {row['p_value']:.3f}  [{row['flag']}]")
    typer.echo("")
    for path in report.artifacts:
        typer.echo(f"  wrote {path}")
    for path in error_artifacts:
        typer.echo(f"  wrote {path}")


@app.command("report")
def report(
    run: str | None = typer.Option(
        None,
        "--run",
        help="Run directory to report on (default: the latest run).",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML path (default: <run>/reports/index.html).",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help=(
            "Interactive Plotly figures (self-contained, ~3-4 MB) vs embedded "
            "PNGs from reports/figures (smaller, quick shares)."
        ),
    ),
) -> None:
    """One self-contained HTML dashboard for a completed run.

    Composes the run's manifest, metrics, diagnostics, readiness verdicts,
    figures, and coefficient table into a single portable page that renders
    offline. Read-only over existing artifacts, so it works on any past run.

    Examples:
        panelcast report
        panelcast report --run outputs/2026-07-08_120000_000000_abcd
        panelcast report --no-interactive -o run_summary.html
    """
    from pathlib import Path

    from panelcast.reporting.html_report import write_run_report

    try:
        path = write_run_report(
            run_dir=Path(run) if run else None,
            output_path=Path(output) if output else None,
            interactive=interactive,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e
    typer.echo(f"  wrote {path}")


@runs_app.command("list")
def runs_list(
    output_dir: str = typer.Option(
        "outputs", "--output-dir", help="Directory holding the pipeline run directories."
    ),
) -> None:
    """List pipeline runs with their manifest summary.

    Shows each run directory (any directory with a manifest.json, skipping
    latest/failed) with its creation time, success status, and completed-stage
    count, and marks the run the latest symlink points at.

    Examples:
        panelcast runs list
        panelcast runs list --output-dir other_outputs
    """
    import json
    from pathlib import Path

    base = Path(output_dir)
    if not base.is_dir():
        typer.echo(f"No runs found: {base} does not exist.")
        raise typer.Exit(code=0)

    # Resolve latest via the authoritative latest.json pointer (falling back to
    # the opportunistic `latest` link, which resolve_latest handles internally).
    from panelcast.paths import resolve_latest

    latest_target: str | None = None
    try:
        latest_run = resolve_latest(base)
        if latest_run is not None:
            resolved = latest_run.resolve()
            if resolved.name not in ("latest", "failed"):
                latest_target = resolved.name
    except OSError:
        latest_target = None

    try:
        run_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        run_dirs = []

    rows: list[tuple[str, ...]] = []  # 11 columns, all preformatted strings
    for run_dir in run_dirs:
        if run_dir.name in ("latest", "failed"):
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if manifest is None:
            created_at, status, n_stages = "?", "corrupt", "?"
            seed = git = dataset = "?"
            mae = r2 = cov = "-"
        else:
            created_at = str(manifest.get("created_at") or "?")
            # Failed runs are moved to outputs/failed/, so a success=False
            # manifest still here is in-progress (or died before the move).
            status = "ok" if manifest.get("success") else "incomplete"
            n_stages = str(len(manifest.get("stages_completed") or []))
            seed = str(manifest.get("seed", "?"))
            g = manifest.get("git") or {}
            git = f"{str(g.get('commit', ''))[:7]}{'*' if g.get('dirty') else ''}" or "?"
            dataset = str((manifest.get("flags") or {}).get("dataset") or "aoty")
            mae = r2 = cov = "-"
            try:
                payload = json.loads(
                    (run_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8")
                )
                m = _history_metrics(payload)
                mae = f"{m['mae']:.2f}" if m["mae"] is not None else "-"
                r2 = f"{m['r2']:.3f}" if m["r2"] is not None else "-"
                cov95 = (m["coverage"] or {}).get("0.95")
                cov = f"{cov95:.2f}" if cov95 is not None else "-"
            except (OSError, ValueError):
                pass
        marker = "*" if run_dir.name == latest_target else " "
        rows.append(
            (marker, run_dir.name, created_at, status, n_stages, seed, dataset, git, mae, r2, cov)
        )

    if not rows:
        typer.echo(f"No runs found under {base}.")
        raise typer.Exit(code=0)

    typer.echo(
        f"  {'run_id':<24} {'created_at':<28} {'status':<10} {'stages':<6} "
        f"{'seed':<6} {'dataset':<10} {'git':<9} {'mae':<7} {'r2':<7} cov95"
    )
    for marker, run_id, created_at, status, n_stages, seed, dataset, git, mae, r2, cov in rows:
        typer.echo(
            f"{marker} {run_id:<24} {created_at:<28} {status:<10} {n_stages:<6} "
            f"{seed:<6} {dataset:<10} {git:<9} {mae:<7} {r2:<7} {cov}"
        )
    if latest_target is not None:
        typer.echo(f"\n* = {base / 'latest'} -> {latest_target}")


# Same default the evaluate stage uses (PipelineConfig.coverage_tolerance);
# each run's recorded calibration.coverage_tolerance wins when present.
_COVERAGE_TOLERANCE_DEFAULT = 0.03
# Relative regression vs the epoch best before mae / elpd_per_obs is flagged
# (raw worse-than-best would flag every non-best run over MCMC noise).
_DRIFT_REL_TOL = 0.02


def _history_metrics(payload: dict) -> dict:
    """Headline metrics from an evaluation metrics.json, null-tolerant.

    Reads the top-level (primary split) fields, which exist in both the
    current schema and legacy payloads.
    """
    point = payload.get("point_metrics") or {}
    cal = payload.get("calibration") or {}
    coverages = {
        level: (entry or {}).get("empirical")
        for level, entry in (cal.get("coverages") or {}).items()
    }
    info = payload.get("info_criteria")
    heldout = info.get("heldout_elpd") if isinstance(info, dict) else None
    return {
        "mae": point.get("mae"),
        "rmse": point.get("rmse"),
        "r2": point.get("r2"),
        "crps": (payload.get("crps") or {}).get("mean_crps"),
        "coverage": coverages,
        "wis": cal.get("wis"),
        "elpd_per_obs": (heldout or {}).get("elpd_per_obs"),
        "coverage_tolerance": cal.get("coverage_tolerance"),
    }


def _flag_history_drift(rows: list[dict]) -> None:
    """Mark within-epoch drift on each row (rows share one feature stamp).

    The reference is the epoch's best-MAE run. Coverage drifts when any
    interval level moves more than the coverage tolerance the evaluate stage
    recorded; mae / elpd_per_obs when worse than the epoch best by more than
    _DRIFT_REL_TOL relative. Missing metrics never flag.
    """
    for row in rows:
        row["drift"] = []
    if len(rows) < 2:
        return
    maes = [r["metrics"]["mae"] for r in rows if r["metrics"]["mae"] is not None]
    elpds = [r["metrics"]["elpd_per_obs"] for r in rows if r["metrics"]["elpd_per_obs"] is not None]
    best_mae = min(maes) if maes else None
    best_elpd = max(elpds) if elpds else None
    ref = (
        min(
            (r for r in rows if r["metrics"]["mae"] is not None),
            key=lambda r: r["metrics"]["mae"],
        )
        if best_mae is not None
        else rows[0]
    )
    tolerance = ref["metrics"].get("coverage_tolerance") or _COVERAGE_TOLERANCE_DEFAULT
    for row in rows:
        m = row["metrics"]
        if best_mae is not None and m["mae"] is not None:
            if m["mae"] > best_mae + _DRIFT_REL_TOL * abs(best_mae):
                row["drift"].append("mae")
        if best_elpd is not None and m["elpd_per_obs"] is not None:
            if m["elpd_per_obs"] < best_elpd - _DRIFT_REL_TOL * abs(best_elpd):
                row["drift"].append("elpd_per_obs")
        if row is not ref:
            for level, ref_cov in ref["metrics"]["coverage"].items():
                cov = m["coverage"].get(level)
                if cov is not None and ref_cov is not None and abs(cov - ref_cov) > tolerance:
                    row["drift"].append(f"coverage@{level}")


@runs_app.command("history")
def runs_history(
    output_dir: str = typer.Option(
        "outputs", "--output-dir", help="Directory holding the pipeline run directories."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of the table."
    ),
) -> None:
    """Cross-run metrics history and drift monitor, grouped by feature epoch.

    Walks the run directories (tolerating corrupt manifests, skipping dry
    runs) and prints one row per successful run that completed evaluate:
    version, tag, sampler settings, headline metrics, and wall-clock. Rows are
    grouped by the feature stamp the metrics were computed against; a stamp
    change is an explicit epoch break, and drift is only ever flagged within
    an epoch. A run is flagged (*) when a coverage level moves more than the
    coverage tolerance vs the epoch's best-MAE run, or when MAE /
    elpd-per-obs regress more than 2% vs the epoch best.

    Examples:
        panelcast runs history
        panelcast runs history --json
    """
    import json
    from pathlib import Path

    base = Path(output_dir)
    if not base.is_dir():
        typer.echo(f"No runs found: {base} does not exist.")
        raise typer.Exit(code=0)

    try:
        run_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        run_dirs = []

    rows: list[dict] = []
    for run_dir in run_dirs:
        if run_dir.name in ("latest", "failed"):
            continue
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "evaluation" / "metrics.json"
        if not (manifest_path.exists() and metrics_path.exists()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or not isinstance(metrics_payload, dict):
            continue
        flags = manifest.get("flags") or {}
        if flags.get("dry_run") or not manifest.get("success"):
            continue
        stamp = metrics_payload.get("feature_stamp")
        rows.append(
            {
                "run_id": run_dir.name,
                "created_at": manifest.get("created_at"),
                "version": manifest.get("version"),
                "tag": manifest.get("tag"),
                "num_chains": flags.get("num_chains"),
                "num_samples": flags.get("num_samples"),
                "duration_seconds": manifest.get("duration_seconds"),
                "feature_stamp": stamp if isinstance(stamp, dict) else None,
                "metrics": _history_metrics(metrics_payload),
            }
        )

    if not rows:
        typer.echo(f"No evaluated runs found under {base}.")
        raise typer.Exit(code=0)

    # Group by feature stamp; epoch order follows each stamp's first run.
    groups: dict[str | None, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["feature_stamp"] or {}).get("input_hash"), []).append(row)
    for group_rows in groups.values():
        _flag_history_drift(group_rows)

    if as_json:
        payload = [
            {
                "feature_stamp": group_rows[0]["feature_stamp"],
                "runs": [{k: v for k, v in r.items() if k != "feature_stamp"} for r in group_rows],
            }
            for group_rows in groups.values()
        ]
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=0)

    def cell(value: object, decimals: int = 3) -> str:
        if value is None:
            return "?"
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)

    header = (
        f"  {'run_id':<30} {'created_at':<20} {'version':<9} {'tag':<12} {'cfg':<9} "
        f"{'mae':<9} {'rmse':<9} {'r2':<8} {'crps':<8} {'cov@80':<8} {'cov@95':<8} "
        f"{'wis':<8} {'elpd/obs':<10} wall_s"
    )
    any_flagged = False
    for idx, (stamp_hash, group_rows) in enumerate(groups.items()):
        if idx:
            typer.echo("")
        stamp = group_rows[0]["feature_stamp"] or {}
        source = f" (features from run {stamp['run_id']})" if stamp.get("run_id") else ""
        typer.echo(f"= epoch {idx + 1}: feature stamp {(stamp_hash or 'unstamped')[:12]}{source} =")
        typer.echo(header)
        for row in group_rows:
            m = row["metrics"]
            drift = set(row["drift"])
            any_flagged = any_flagged or bool(drift)

            def metric(name: str, decimals: int = 3, *, d: set = drift, m: dict = m) -> str:
                return cell(m[name], decimals) + ("*" if name in d else "")

            def coverage(level: str, *, d: set = drift, m: dict = m) -> str:
                return cell(m["coverage"].get(level)) + ("*" if f"coverage@{level}" in d else "")

            created = str(row["created_at"] or "?")[:19]
            tag = str(row["tag"] or "")[:12]
            cfg = (
                f"{row['num_chains']}x{row['num_samples']}"
                if row["num_chains"] is not None and row["num_samples"] is not None
                else "?"
            )
            wall = cell(float(row["duration_seconds"]), 0) if row["duration_seconds"] else "?"
            typer.echo(
                f"  {row['run_id']:<30} {created:<20} {cell(row['version']):<9} {tag:<12} "
                f"{cfg:<9} {metric('mae'):<9} {metric('rmse'):<9} {metric('r2'):<8} "
                f"{metric('crps'):<8} {coverage('0.80'):<8} {coverage('0.95'):<8} "
                f"{metric('wis'):<8} {metric('elpd_per_obs'):<10} {wall}"
            )
    if any_flagged:
        typer.echo(
            "\n* drift within epoch vs the best-MAE run: coverage shifted beyond "
            f"tolerance (default {_COVERAGE_TOLERANCE_DEFAULT:g}), or mae/elpd_per_obs "
            f"worse than the epoch best by >{_DRIFT_REL_TOL:.0%}."
        )
