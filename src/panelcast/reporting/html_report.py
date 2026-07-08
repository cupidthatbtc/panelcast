"""One self-contained HTML dashboard per run.

A run scatters its story across ~15 artifacts; this composes them into a
single portable ``reports/index.html`` — header, verdict strip, per-split
metrics, stage durations, interactive Plotly figures (or embedded PNGs with
``interactive=False``), and the coefficient table. Strictly read-only over
existing run artifacts; renders offline (plotly.js inlined, PNGs base64).
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import structlog

log = structlog.get_logger()

_STYLE = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0;
       color: #1a1a1a; background: #fafafa; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: 0.5rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.meta { color: #555; font-size: 0.85rem; line-height: 1.6; }
.verdicts { display: flex; gap: 0.6rem; margin: 1rem 0; flex-wrap: wrap; }
.verdict { padding: 0.35rem 0.8rem; border-radius: 4px; font-size: 0.85rem;
           font-weight: 600; color: #fff; }
.pass { background: #2e7d32; } .fail { background: #c62828; }
.unknown { background: #757575; }
.bar { height: 14px; background: #4c78a8; display: inline-block; }
.barrow td { border: none; padding: 0.15rem 0.6rem; }
figure { margin: 1rem 0; } figure img { max-width: 100%; }
"""


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _verdict(label: str, ok: bool | None) -> str:
    cls = "unknown" if ok is None else ("pass" if ok else "fail")
    text = "n/a" if ok is None else ("PASS" if ok else "FAIL")
    return f'<span class="verdict {cls}">{html.escape(label)}: {text}</span>'


def _header_block(manifest: dict | None, run_dir: Path) -> str:
    if manifest is None:
        return f'<p class="meta">run dir: {html.escape(str(run_dir))} (no manifest.json)</p>'
    git = manifest.get("git") or {}
    dirty = " (dirty)" if git.get("dirty") else ""
    fields = [
        ("run", manifest.get("run_id")),
        ("command", manifest.get("command")),
        ("git", f"{(git.get('commit') or '?')[:12]}{dirty}"),
        ("seed", manifest.get("seed")),
        ("version", manifest.get("version")),
        ("duration", f"{(manifest.get('duration_seconds') or 0):.0f}s"),
        ("tag", manifest.get("tag")),
    ]
    rows = " · ".join(
        f"<b>{k}</b>: {html.escape(str(v))}" for k, v in fields if v not in (None, "")
    )
    return f'<p class="meta">{rows}</p>'


def _metrics_table(metrics: dict | None) -> str:
    if not metrics:
        return "<p>No metrics.json found.</p>"
    splits = metrics.get("splits") or {}
    if not splits:
        return "<p>metrics.json has no splits block.</p>"
    header = (
        "<tr><th>split</th><th>n</th><th>MAE</th><th>RMSE</th><th>R2</th>"
        "<th>CRPS</th><th>cov@80</th><th>cov@95</th></tr>"
    )
    rows = []
    for name, split in splits.items():
        pm = split.get("point_metrics") or {}
        cov = ((split.get("calibration") or {}).get("coverages")) or {}

        def fmt(value, places=3):
            return f"{value:.{places}f}" if isinstance(value, (int, float)) else "—"

        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{split.get('n_test', '—')}</td>"
            f"<td>{fmt(pm.get('mae'))}</td><td>{fmt(pm.get('rmse'))}</td>"
            f"<td>{fmt(pm.get('r2'))}</td>"
            f"<td>{fmt((split.get('crps') or {}).get('mean_crps'))}</td>"
            f"<td>{fmt((cov.get('0.80') or {}).get('empirical'))}</td>"
            f"<td>{fmt((cov.get('0.95') or {}).get('empirical'))}</td></tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def _durations_block(manifest: dict | None) -> str:
    durations = (manifest or {}).get("stage_durations") or {}
    if not durations:
        return ""
    longest = max(durations.values()) or 1.0
    rows = "".join(
        f'<tr class="barrow"><td>{html.escape(stage)}</td>'
        f'<td style="text-align:left; width: 420px;">'
        f'<span class="bar" style="width:{max(2, int(400 * secs / longest))}px"></span> '
        f"{secs:.0f}s</td></tr>"
        for stage, secs in durations.items()
    )
    return f"<h2>Stage durations</h2><table>{rows}</table>"


def _figures_block(run_dir: Path, interactive: bool) -> str:
    if interactive:
        try:
            from panelcast.visualization.dashboard import (
                create_coefficients_table,
                create_dashboard_figures,
                load_dashboard_data,
            )

            data = load_dashboard_data(run_dir)
            fragments = create_dashboard_figures(data)
            parts = [
                f"<h2>{html.escape(view.title())}</h2>{fragment}"
                for view, fragment in fragments.items()
            ]
            if data.coefficients is not None:
                parts.append("<h2>Coefficients</h2>" + create_coefficients_table(data.coefficients))
            if parts:
                return "".join(parts)
        except Exception as e:
            log.warning("report_interactive_figures_failed", error=str(e)[:300])
    # Fallback (and --no-interactive): embed the publication PNGs.
    parts = []
    for png in sorted((run_dir / "reports" / "figures").glob("*.png")):
        encoded = base64.b64encode(png.read_bytes()).decode("ascii")
        parts.append(
            f"<figure><img src='data:image/png;base64,{encoded}' "
            f"alt='{html.escape(png.stem)}'/>"
            f"<figcaption>{html.escape(png.stem)}</figcaption></figure>"
        )
    if not parts:
        return "<p>No figures available.</p>"
    return "<h2>Figures</h2>" + "".join(parts)


def build_run_report(run_dir: Path | None = None, interactive: bool = True) -> str:
    """Compose the run's artifacts into one portable HTML page."""
    if run_dir is None:
        from panelcast.paths import resolve_latest

        run_dir = resolve_latest()
        if run_dir is None:
            raise FileNotFoundError("no latest run found under outputs/")
    run_dir = Path(run_dir)

    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "evaluation" / "metrics.json")
    diagnostics = _read_json(run_dir / "evaluation" / "diagnostics.json")
    readiness = _read_json(run_dir / "reports" / "publication_readiness.json")

    calibration_ok = None
    if metrics:
        calibration_ok = (metrics.get("calibration") or {}).get("within_tolerance")
    readiness_ok = None
    if readiness is not None:
        readiness_ok = bool(readiness.get("ready", readiness.get("publication_ready", False)))
    verdicts = "".join(
        [
            _verdict("convergence", (diagnostics or {}).get("passed")),
            _verdict("calibration", calibration_ok),
            _verdict("readiness", readiness_ok),
            _verdict("run", (manifest or {}).get("success")),
        ]
    )

    title = (manifest or {}).get("run_id") or run_dir.name
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>panelcast run {html.escape(str(title))}</title>
<style>{_STYLE}</style></head>
<body><main>
<h1>panelcast run report — {html.escape(str(title))}</h1>
{_header_block(manifest, run_dir)}
<div class="verdicts">{verdicts}</div>
<h2>Metrics</h2>
{_metrics_table(metrics)}
{_durations_block(manifest)}
{_figures_block(run_dir, interactive)}
</main></body></html>
"""


def write_run_report(
    run_dir: Path | None = None,
    output_path: Path | None = None,
    interactive: bool = True,
) -> Path:
    """Build the report and write it into the run's reports dir."""
    if run_dir is None:
        from panelcast.paths import resolve_latest

        run_dir = resolve_latest()
        if run_dir is None:
            raise FileNotFoundError("no latest run found under outputs/")
    run_dir = Path(run_dir)
    content = build_run_report(run_dir, interactive=interactive)
    if output_path is None:
        output_path = run_dir / "reports" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    log.info("run_report_written", path=str(output_path), bytes=len(content))
    return output_path
