"""Transform x latent-process 2x2 bake-off (likelihood fixed = studentt).

Closes the one untested cell the model-spec review named: every prior
`offset_logit` / `ar1` verdict was recorded against a single axis at a time, so
the `offset_logit x ar1` combination had never actually been run. This fits the
full 2x2 grid (identity|offset_logit x rw|ar1) end-to-end on the on-disk
splits/features and emits one comparison table.

Each cell runs `panelcast run --preset diagnostic --stages train,evaluate
--likelihood-family studentt` with the cell's transform/latent, so the
data/splits/features stages must already exist on disk and `AOTY_DATASET_PATH`
must point at the subset. Each run's `outputs/evaluation/` is snapshotted into
`.audit/transform_latent_bakeoff/<cell>/` before the next cell overwrites it.

GPU: invoke with the GPU venv interpreter (`~/aoty-gpu/bin/python`) so the
sibling `panelcast` is the GPU build. Do NOT set `JAX_PLATFORMS=cuda` — evaluate
needs the cpu backend.

Example
-------
    AOTY_DATASET_PATH=data/raw/aoty_subset.csv \
        ~/aoty-gpu/bin/python scripts/bakeoff_transform_latent.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

CELLS = (
    {"name": "identity_rw", "transform": "identity", "latent": "rw"},
    {"name": "identity_ar1", "transform": "identity", "latent": "ar1"},
    {"name": "offset_logit_rw", "transform": "offset_logit", "latent": "rw"},
    {"name": "offset_logit_ar1", "transform": "offset_logit", "latent": "ar1"},
)


def _json_safe(obj):
    """Recursively replace non-finite floats with None so json.dumps stays valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _run_cell(panelcast_bin: str, cell: dict, preset: str) -> int:
    """Invoke `panelcast run` for one transform/latent cell; return its exit code."""
    cmd = [
        panelcast_bin,
        "run",
        "--preset",
        preset,
        "--stages",
        "train,evaluate",
        "--likelihood-family",
        "studentt",
        "--target-transform",
        cell["transform"],
        "--latent-process",
        cell["latent"],
    ]
    print(f"\n=== {cell['name']} ({cell['transform']} x {cell['latent']}) ===\n  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


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
        pit = (m.get("calibration", {}) or {}).get("pit", {}) or {}
        loo = (m.get("info_criteria", {}) or {}).get("loo", {})
        pinned = m.get("ppc", {}).get("extreme_statistics", []) or []
        out.update(
            mae=point.get("mae"),
            rmse=point.get("rmse"),
            r2=point.get("r2"),
            cov95=cov95.get("empirical"),
            pit_dev=pit.get("max_abs_dev_from_uniform"),
            crps=(m.get("crps", {}) or {}).get("mean_crps"),
            ppc_pinned=len(pinned),
            ppc_pinned_names=",".join(pinned) or "-",
            loo_elpd=loo.get("elpd"),
            pareto_k_max=loo.get("pareto_k_max"),
            pareto_k_gt07=loo.get("pareto_k_gt_0_7"),
        )
    return out


def _fmt(v, spec: str = ".3g") -> str:
    """Format one metric for the markdown table ('-' for missing)."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


_COLUMNS = [
    ("name", "cell", "{}"),
    ("transform", "transform", "{}"),
    ("latent", "latent", "{}"),
    ("passed", "conv", None),
    ("rhat_max", "rhat", ".3f"),
    ("ess_bulk_min", "ess", ".0f"),
    ("divergences", "div", None),
    ("ppc_pinned", "ppc_pin", None),
    ("ppc_pinned_names", "pinned", "{}"),
    ("mae", "mae", ".2f"),
    ("rmse", "rmse", ".2f"),
    ("cov95", "cov95", ".3f"),
    ("pit_dev", "pit_dev", ".3f"),
    ("crps", "crps", ".2f"),
    ("pareto_k_max", "k_max", ".2f"),
    ("pareto_k_gt07", "k>0.7", None),
]


def _render_markdown(rows: list[dict]) -> str:
    """Render the per-cell rows as the comparison markdown table."""
    head = "| " + " | ".join(label for _, label, _ in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = ["# Transform x latent-process bake-off (real subset, likelihood=studentt)", "", head, sep]
    for r in rows:
        cells = []
        for key, _, spec in _COLUMNS:
            v = r.get(key)
            cells.append(str(v) if spec == "{}" else _fmt(v, spec or ".3g"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the 2x2 grid, snapshot each cell, and write the comparison artifacts."""
    default_bin = str(Path(sys.executable).parent / "panelcast")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="diagnostic")
    parser.add_argument("--eval-dir", default="outputs/evaluation")
    parser.add_argument("--out-dir", default=".audit/transform_latent_bakeoff")
    parser.add_argument(
        "--panelcast-bin",
        default=default_bin,
        help="panelcast executable (default: sibling of the running interpreter).",
    )
    args = parser.parse_args()

    if "AOTY_DATASET_PATH" not in os.environ:
        print(
            "warning: AOTY_DATASET_PATH not set; set it to data/raw/aoty_subset.csv "
            "for the subset bake-off.",
            file=sys.stderr,
        )

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cell in CELLS:
        row = {"name": cell["name"], "transform": cell["transform"], "latent": cell["latent"]}
        code = _run_cell(args.panelcast_bin, cell, args.preset)
        if code != 0:
            row["error"] = f"panelcast run exited {code}"
            print(f"  ! {cell['name']} failed (exit {code}); recording and continuing")
            rows.append(row)
            continue
        dest = out_dir / cell["name"]
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ("metrics.json", "diagnostics.json"):
            src = eval_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
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
