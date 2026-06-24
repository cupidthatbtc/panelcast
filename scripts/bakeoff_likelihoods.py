"""Real-subset likelihood bake-off: run train+evaluate per family (x discretize)
and emit one comparison table (convergence, PPC pins, point/calibration, LOO).

Each combo runs the diagnostic preset end of the existing pipeline against the
on-disk splits/features (so run the data/splits/features stages once first), with
``AOTY_DATASET_PATH`` pointing at the subset. Each run's ``outputs/evaluation/``
is copied aside before the next combo overwrites it.

GPU usage: this only shells out to ``panelcast run`` — point it at the GPU venv
(``~/aoty-gpu``) by running it with that interpreter / on a CUDA host. CPU works
too, just slower.

Examples
--------
    AOTY_DATASET_PATH=data/raw/aoty_subset.csv \
        python scripts/bakeoff_likelihoods.py \
        --combos studentt,studentt+discretize,skew_normal,skew_normal+discretize,split_normal

Writes ``outputs/bakeoff/comparison.{json,md}`` and per-combo evaluation copies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_COMBOS = (
    "studentt",
    "studentt+discretize",
    "skew_normal",
    "skew_normal+discretize",
    "split_normal",
)


def _json_safe(obj):
    """Recursively replace non-finite floats with None so json.dumps is valid JSON."""
    import math

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _parse_combo(combo: str) -> tuple[str, bool]:
    """'skew_normal+discretize' -> ('skew_normal', True)."""
    parts = [p.strip() for p in combo.split("+")]
    family = parts[0]
    discretize = "discretize" in parts[1:]
    return family, discretize


def _run_combo(
    family: str,
    discretize: bool,
    preset: str,
    eval_dir: Path,
    extra_args: list[str],
) -> int:
    """Invoke `panelcast run` for one combo; return its exit code."""
    cmd = [
        "panelcast",
        "run",
        "--preset",
        preset,
        "--likelihood-family",
        family,
        "--stages",
        "train,evaluate",
        *extra_args,
    ]
    if discretize:
        cmd.append("--discretize-observation")
    print(f"\n=== {family}{'+discretize' if discretize else ''} ===\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _read_metrics(eval_dir: Path) -> dict:
    """Pull the comparison fields from one evaluation run (best-effort)."""
    out: dict = {}
    diag_path = eval_dir / "diagnostics.json"
    metrics_path = eval_dir / "metrics.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        out.update(
            passed=diag.get("passed"),
            rhat_max=diag.get("rhat_max"),
            ess_bulk_min=diag.get("ess_bulk_min"),
            divergences=diag.get("divergences"),
        )
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        point = m.get("point_metrics", {})
        cov95 = (m.get("calibration", {}).get("coverages", {}) or {}).get("0.95", {})
        loo = (m.get("info_criteria", {}) or {}).get("loo", {})
        ppc = m.get("ppc", {})
        out.update(
            mae=point.get("mae"),
            rmse=point.get("rmse"),
            r2=point.get("r2"),
            cov95=cov95.get("empirical"),
            width95=cov95.get("interval_width"),
            crps=(m.get("crps", {}) or {}).get("mean_crps"),
            ppc_pinned=len(ppc.get("extreme_statistics", []) or []),
            ppc_pinned_names=",".join(ppc.get("extreme_statistics", []) or []) or "-",
            loo_elpd=loo.get("elpd"),
            pareto_k_max=loo.get("pareto_k_max"),
            pareto_k_gt07=loo.get("pareto_k_gt_0_7"),
        )
    return out


def _fmt(v, spec: str = ".3g") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


_COLUMNS = [
    ("combo", "combo", "{}"),
    ("passed", "conv", None),
    ("rhat_max", "rhat", ".3f"),
    ("ess_bulk_min", "ess", ".0f"),
    ("divergences", "div", None),
    ("ppc_pinned", "ppc_pin", None),
    ("ppc_pinned_names", "pinned", "{}"),
    ("mae", "mae", ".2f"),
    ("rmse", "rmse", ".2f"),
    ("cov95", "cov95", ".3f"),
    ("width95", "w95", ".2f"),
    ("crps", "crps", ".2f"),
    ("loo_elpd", "loo", ".0f"),
    ("pareto_k_max", "k_max", ".2f"),
]


def _render_markdown(rows: list[dict]) -> str:
    head = "| " + " | ".join(label for _, label, _ in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = ["# Likelihood bake-off (real subset)", "", head, sep]
    for r in rows:
        cells = []
        for key, _, spec in _COLUMNS:
            v = r.get(key)
            cells.append(str(v) if spec == "{}" else _fmt(v, spec or ".3g"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combos",
        default=",".join(DEFAULT_COMBOS),
        help="Comma list of 'family' or 'family+discretize'.",
    )
    parser.add_argument("--preset", default="diagnostic")
    parser.add_argument("--eval-dir", default="outputs/evaluation")
    parser.add_argument("--output-dir", default="outputs/bakeoff")
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra args appended to every `panelcast run` (space-separated).",
    )
    args = parser.parse_args()

    if "AOTY_DATASET_PATH" not in os.environ:
        print(
            "warning: AOTY_DATASET_PATH not set; the run will use the default raw "
            "dataset. Set it to data/raw/aoty_subset.csv for the subset bake-off.",
            file=sys.stderr,
        )

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = args.extra_args.split() if args.extra_args.strip() else []

    rows: list[dict] = []
    for combo in [c.strip() for c in args.combos.split(",") if c.strip()]:
        family, discretize = _parse_combo(combo)
        label = combo.replace("+", "_")
        code = _run_combo(family, discretize, args.preset, eval_dir, extra)
        row: dict = {"combo": combo}
        if code != 0:
            row["error"] = f"panelcast run exited {code}"
            print(f"  ! {combo} failed (exit {code}); recording and continuing")
            rows.append(row)
            continue
        dest = out_dir / label
        if dest.exists():
            shutil.rmtree(dest)
        if eval_dir.exists():
            shutil.copytree(eval_dir, dest)
        row.update(_read_metrics(eval_dir))
        rows.append(row)

    (out_dir / "comparison.json").write_text(json.dumps(_json_safe(rows), indent=2), encoding="utf-8")
    md = _render_markdown(rows)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {out_dir / 'comparison.json'} and {out_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
