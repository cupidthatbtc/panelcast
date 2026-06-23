"""Convergence + PPC report over an existing evaluation run.

Reads the artifacts the evaluate stage already wrote
(``outputs/evaluation/diagnostics.json`` and ``metrics.json``) and renders a
focused convergence + posterior-predictive-check report — the two things the
review flagged (a failing convergence gate, PPC p-values pinned at the
extremes). No model refit; this just re-presents what the run produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    if p <= _EXTREME_LO or p >= _EXTREME_HI:
        return "pinned"
    if p <= _WARN_LO or p >= _WARN_HI:
        return "warn"
    return "ok"


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
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        summary = (metrics.get("ppc") or {}).get("summary") or {}
        for name, stat in summary.items():
            p = float(stat.get("p_value", float("nan")))
            flag = _flag(p)
            if flag == "pinned":
                extreme.append(name)
            ppc_rows.append(
                {
                    "statistic": name,
                    "observed": stat.get("observed"),
                    "p_value": p,
                    "flag": flag,
                }
            )

    passed = convergence.get("passed")
    if passed is True and not extreme:
        verdict = "Converged and no PPC statistics pinned at the extremes."
    elif passed is True:
        verdict = (
            f"Converged, but {len(extreme)} PPC statistic(s) pinned at the extremes "
            f"({', '.join(extreme)}) — likely likelihood misspecification."
        )
    elif passed is False:
        tail = (
            f" and {len(extreme)} pinned PPC statistic(s) ({', '.join(extreme)})"
            if extreme
            else ""
        )
        verdict = f"Convergence gate FAILED{tail}."
    else:
        verdict = "Convergence status unavailable; see PPC flags below."

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


def run_diagnose(
    eval_dir: Path = Path("outputs/evaluation"),
    output_dir: Path = Path("reports/diagnostics"),
) -> DiagnoseReport:
    """Build the report and write it as Markdown + JSON."""
    report = build_report(eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "diagnostics_report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = output_dir / "diagnostics_report.json"
    json_path.write_text(
        json.dumps(
            {
                "verdict": report.verdict,
                "convergence": report.convergence,
                "ppc": report.ppc,
                "extreme_ppc": report.extreme_ppc,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    report.artifacts = [md_path, json_path]
    return report
