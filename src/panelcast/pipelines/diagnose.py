"""Convergence + PPC report over an existing evaluation run.

Reads the artifacts the evaluate stage already wrote (``diagnostics.json``
and ``metrics.json`` under the run's evaluation dir) and renders a
focused convergence + posterior-predictive-check report — the two things the
review flagged (a failing convergence gate, PPC p-values pinned at the
extremes). No model refit; this just re-presents what the run produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from panelcast.paths import resolve_evaluation_dir

# A PPC p-value at/near 0 or 1 is the signature of a misspecified statistic
# (e.g. the symmetric-likelihood / left-skewed-target mismatch).
_EXTREME_LO = 0.01
_EXTREME_HI = 0.99
_WARN_LO = 0.05
_WARN_HI = 0.95


@dataclass
class DiagnoseReport:
    convergence: dict
    ppc: list[dict]
    extreme_ppc: list[str]
    verdict: str
    artifacts: list[Path] = field(default_factory=list)


def _flag(p: float) -> str:
    if math.isnan(p):
        return "missing"
    if p <= _EXTREME_LO or p >= _EXTREME_HI:
        return "pinned"
    if p <= _WARN_LO or p >= _WARN_HI:
        return "warn"
    return "ok"


def _json_safe(obj: object) -> object:
    """Recursively replace non-finite floats with None so json.dumps emits valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_report(eval_dir: Path) -> DiagnoseReport:
    """Assemble the diagnose report from evaluation artifacts."""
    diag_path = eval_dir / "diagnostics.json"
    metrics_path = eval_dir / "metrics.json"
    if not diag_path.exists() and not metrics_path.exists():
        raise FileNotFoundError(
            f"No evaluation artifacts under {eval_dir}. Run the evaluate stage first "
            "(panelcast run / panelcast stage evaluate)."
        )

    convergence: dict = {}
    if diag_path.exists():
        with open(diag_path, encoding="utf-8") as f:
            convergence = json.load(f)

    ppc_rows: list[dict] = []
    extreme: list[str] = []
    missing: list[str] = []
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        summary = (metrics.get("ppc") or {}).get("summary") or {}
        for name, stat in summary.items():
            p = float(stat.get("p_value", float("nan")))
            flag = _flag(p)
            if flag == "pinned":
                extreme.append(name)
            elif flag == "missing":
                missing.append(name)
            ppc_rows.append(
                {
                    "statistic": name,
                    "observed": stat.get("observed"),
                    "p_value": p,
                    "flag": flag,
                }
            )

    passed = convergence.get("passed")
    missing_note = (
        f" {len(missing)} PPC statistic(s) had no p-value ({', '.join(missing)})."
        if missing
        else ""
    )
    if passed is True and not extreme:
        verdict = "Converged and no PPC statistics pinned at the extremes." + missing_note
    elif passed is True:
        verdict = (
            f"Converged, but {len(extreme)} PPC statistic(s) pinned at the extremes "
            f"({', '.join(extreme)}) — likely likelihood misspecification." + missing_note
        )
    elif passed is False:
        tail = (
            f" and {len(extreme)} pinned PPC statistic(s) ({', '.join(extreme)})"
            if extreme
            else ""
        )
        verdict = f"Convergence gate FAILED{tail}." + missing_note
    else:
        verdict = "Convergence status unavailable; see PPC flags below." + missing_note

    return DiagnoseReport(
        convergence=convergence,
        ppc=ppc_rows,
        extreme_ppc=extreme,
        verdict=verdict,
    )


def render_markdown(report: DiagnoseReport) -> str:
    lines = ["# Diagnostics report", "", f"**Verdict:** {report.verdict}", "", "## Convergence", ""]
    c = report.convergence
    if c:
        rhat = c.get("rhat_max")
        lines += [
            f"- Status: {'PASS' if c.get('passed') else 'FAIL'}",
            f"- R-hat (max): {rhat if rhat is not None else 'n/a (single chain)'} "
            f"(threshold < {c.get('rhat_threshold', '?')})",
            f"- ESS bulk (min): {c.get('ess_bulk_min', '?')} "
            f"(threshold {c.get('ess_threshold', '?')})",
            f"- Divergences: {c.get('divergences', '?')}",
        ]
    else:
        lines.append("- _No diagnostics.json found._")
    lines += ["", "## Posterior predictive checks", ""]
    if report.ppc:
        lines += ["| Statistic | Observed | p-value | Flag |", "| --- | --- | --- | --- |"]
        for row in report.ppc:
            obs = row["observed"]
            obs_s = f"{obs:.3f}" if isinstance(obs, (int, float)) else str(obs)
            lines.append(
                f"| {row['statistic']} | {obs_s} | {row['p_value']:.3f} | {row['flag']} |"
            )
    else:
        lines.append("- _No PPC summary found._")
    lines.append("")
    return "\n".join(lines)


def _top_rows_markdown(rows, split: str, n: int = 25) -> str:
    """Worst-|residual| rows as a hand-rendered pipe table (no tabulate dep)."""
    top = rows.head(n)
    cols = [c for c in top.columns if c != "sq_error_share"]
    lines = [
        f"# Worst {len(top)} predictions — {split}",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in top.iterrows():
        cells = [
            f"{v:.3f}" if isinstance(v, float) and math.isfinite(v) else str(v)
            for v in (row[c] for c in cols)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def run_error_decomposition(
    eval_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Decompose per-row errors for every split with an identified predictions.json.

    Read-only over evaluate artifacts. Writes, per split, the full per-row
    CSV, one CSV per rollup, and a worst-25 Markdown table. Payloads that
    predate the identified schema raise ValueError with the re-run hint.
    """
    from panelcast.evaluation.decomposition import decompose_errors
    from panelcast.paths import resolve_reports_dir
    from panelcast.reporting.tables import export_table

    if eval_dir is None:
        eval_dir = resolve_evaluation_dir()
    if output_dir is None:
        output_dir = resolve_reports_dir() / "diagnostics"

    payloads: dict[str, Path] = {
        p.parent.name: p for p in sorted(eval_dir.glob("*/predictions.json"))
    }
    if not payloads:
        flat = eval_dir / "predictions.json"
        if flat.exists():
            payloads["primary"] = flat
    if not payloads:
        raise FileNotFoundError(
            f"No predictions.json under {eval_dir}. Run the evaluate stage first."
        )

    artifacts: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in payloads.items():
        with open(path, encoding="utf-8") as f:
            predictions = json.load(f)
        decomp = decompose_errors(predictions)
        artifacts += export_table(
            decomp.rows, output_dir / f"error_decomposition_{split}", formats=("csv",)
        )
        for name, rollup in decomp.rollups.items():
            artifacts += export_table(
                rollup, output_dir / f"error_rollup_{name}_{split}", formats=("csv",)
            )
        md_path = output_dir / f"error_top25_{split}.md"
        md_path.write_text(_top_rows_markdown(decomp.rows, split), encoding="utf-8")
        artifacts.append(md_path)
    return artifacts


def run_diagnose(
    eval_dir: Path | None = None,
    output_dir: Path = Path("reports/diagnostics"),
) -> DiagnoseReport:
    """Build the report and write it as Markdown + JSON.

    eval_dir=None resolves to the latest run's evaluation dir, falling back
    to the legacy flat location when no latest pointer exists.
    """
    if eval_dir is None:
        eval_dir = resolve_evaluation_dir()
    report = build_report(eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "diagnostics_report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = output_dir / "diagnostics_report.json"
    json_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "verdict": report.verdict,
                    "convergence": report.convergence,
                    "ppc": report.ppc,
                    "extreme_ppc": report.extreme_ppc,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    report.artifacts = [md_path, json_path]
    return report
