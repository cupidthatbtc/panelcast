"""Transform x latent-process 2x2 bake-off (likelihood fixed = studentt).

Closes the one untested cell the model-spec review named: every prior
`offset_logit` / `ar1` verdict was recorded against a single axis at a time, so
the `offset_logit x ar1` combination had never actually been run. This fits the
full 2x2 grid (identity|offset_logit x rw|ar1) end-to-end on the on-disk
splits/features and emits one comparison table.

Each cell runs `panelcast run --preset diagnostic --stages train,evaluate
--likelihood-family studentt` with the cell's transform/latent, so the
data/splits/features stages must already exist on disk and `AOTY_DATASET_PATH`
must point at the subset. `PANELCAST_SAVE_LOG_LIKELIHOOD=1` is set for each run so
the evaluate stage persists `outputs/evaluation/log_likelihood.nc` (pointwise,
score-scale). Each run's `outputs/evaluation/` is snapshotted into
`.audit/transform_latent_bakeoff/<cell>/` before the next cell overwrites it; the
snapshotted log-likelihoods feed the direct held-out elpd estimator (#63) so the
table reports test lppd + SE and the paired per-point `elpd_diff +/- dse` vs the
kept default.

GPU: invoke with the GPU venv interpreter (`~/aoty-gpu/bin/python`) so the
sibling `panelcast` is the GPU build. Do NOT set `JAX_PLATFORMS=cuda` — evaluate
needs the cpu backend.

Example
-------
    AOTY_DATASET_PATH=data/raw/aoty_subset.csv \
        ~/aoty-gpu/bin/python scripts/bakeoff_transform_latent.py

Pass `--reassemble` to rebuild the comparison artifacts from existing snapshots
without re-fitting. The elpd columns and pairwise rows are recomputed from the
per-cell `log_likelihood.nc` snapshots; cells without one (offset_logit_ar1)
render "-" rather than resurfacing pre-#63 PSIS-LOO numbers.
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

DEFAULT_CELL = "identity_rw"  # the kept default; pairwise diffs are measured against it
_SNAPSHOT_FILES = ("metrics.json", "diagnostics.json", "log_likelihood.nc")


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
    # Opt the evaluate stage into persisting the pointwise log-likelihood so the
    # pairwise az.compare below has a common-test-set idata per cell.
    env = {**os.environ, "PANELCAST_SAVE_LOG_LIKELIHOOD": "1"}
    print(f"\n=== {cell['name']} ({cell['transform']} x {cell['latent']}) ===\n  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, env=env).returncode


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
        info = m.get("info_criteria", {}) or {}
        # Direct held-out lppd (#63). No legacy `loo` fallback: pre-#63
        # snapshots recompute from their log_likelihood.nc instead (see
        # _collect_rows_from_snapshots) — the PSIS-LOO-on-test numbers those
        # metrics.json carry are the artifact #63 retired.
        elpd = info.get("heldout_elpd", {}) or {}
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
            elpd=elpd.get("elpd"),
            elpd_se=elpd.get("se"),
        )
    return out


def _pairwise_elpd(rows: list[dict], out_dir: Path, default_cell: str = DEFAULT_CELL) -> list[dict]:
    """Paired per-point held-out elpd_diff +/- dse of each cell vs the kept default.

    The per-point estimator and the pairing math live in
    `panelcast.select.scoring` (#101); this wraps them in the bake-off's cell
    layout. The diff is signed cell-minus-default (positive = beats the
    default). Returns [] when the default cell's snapshot is absent.
    """
    from panelcast.select.scoring import paired_elpd

    default_nc = out_dir / default_cell / "log_likelihood.nc"
    if not default_nc.exists():
        return []

    pairwise: list[dict] = []
    for row in rows:
        name = row["name"]
        cell_nc = out_dir / name / "log_likelihood.nc"
        if name == default_cell or not cell_nc.exists():
            continue
        try:
            pair = paired_elpd(cell_nc, default_nc)
        except Exception as exc:  # noqa: BLE001 — audit script: degrade, don't abort
            print(f"  ! pairwise diff failed for {name}: {type(exc).__name__}: {exc}")
            continue
        pairwise.append({"name": name, "elpd_diff": pair.diff, "dse": pair.dse, "z": pair.z})
    return pairwise


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
    ("elpd", "elpd", ".1f"),
    ("elpd_se", "se", ".1f"),
]


def _render_markdown(
    rows: list[dict], pairwise: list[dict] | None = None, default_cell: str = DEFAULT_CELL
) -> str:
    """Render the per-cell rows (and optional pairwise LOO table) as markdown."""
    head = "| " + " | ".join(label for _, label, _ in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [
        "# Transform x latent-process bake-off (real subset, likelihood=studentt)",
        "",
        head,
        sep,
    ]
    for r in rows:
        cells = []
        for key, _, spec in _COLUMNS:
            v = r.get(key)
            cells.append(str(v) if spec == "{}" else _fmt(v, spec or ".3g"))
        lines.append("| " + " | ".join(cells) + " |")

    if pairwise:
        lines += [
            "",
            f"## Pairwise held-out elpd vs kept default ({default_cell})",
            "",
            "elpd_diff is cell minus default (positive = beats the default); dse is the",
            "*paired* difference SE from per-point elpd diffs on identical test data (#63).",
            "",
            "| cell | elpd_diff | dse | z (diff/dse) |",
            "| --- | --- | --- | --- |",
        ]
        for p in pairwise:
            lines.append(
                f"| {p['name']} | {_fmt(p.get('elpd_diff'), '+.1f')} | "
                f"{_fmt(p.get('dse'), '.1f')} | {_fmt(p.get('z'), '+.2f')} |"
            )
        scored = {p["name"] for p in pairwise} | {default_cell}
        omitted = [
            r["name"] for r in rows if r.get("elpd") is not None and r["name"] not in scored
        ]
        if omitted:
            lines += [
                "",
                f"_No current pointwise log-likelihood snapshot for {', '.join(omitted)}, "
                "so the pairwise table omits them._",
            ]
    return "\n".join(lines) + "\n"


def _collect_rows_from_snapshots(out_dir: Path) -> list[dict]:
    """Rebuild per-cell rows from existing snapshots (no re-fitting).

    elpd/se are recomputed from the snapshotted pointwise `log_likelihood.nc`
    (the #63 estimator), overriding whatever metrics.json recorded; cells
    without one render "-".
    """
    import numpy as np

    from panelcast.select.scoring import pointwise_elpd

    rows: list[dict] = []
    for cell in CELLS:
        row = {"name": cell["name"], "transform": cell["transform"], "latent": cell["latent"]}
        dest = out_dir / cell["name"]
        if (dest / "metrics.json").exists():
            row.update(_read_metrics(dest))
        else:
            row["error"] = "no snapshot"
        nc = dest / "log_likelihood.nc"
        if nc.exists():
            try:
                elpd_i = pointwise_elpd(nc)
                row["elpd"] = float(np.sum(elpd_i))
                row["elpd_se"] = float(np.sqrt(elpd_i.size * np.var(elpd_i, ddof=1)))
            except Exception as exc:  # noqa: BLE001 — audit script: degrade, don't abort
                print(f"  ! elpd recompute failed for {cell['name']}: {type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def _run_grid(panelcast_bin: str, preset: str, eval_dir: Path, out_dir: Path) -> list[dict]:
    """Fit each cell, snapshot its evaluation outputs, and collect the row."""
    rows: list[dict] = []
    for cell in CELLS:
        row = {"name": cell["name"], "transform": cell["transform"], "latent": cell["latent"]}
        code = _run_cell(panelcast_bin, cell, preset)
        if code != 0:
            row["error"] = f"panelcast run exited {code}"
            print(f"  ! {cell['name']} failed (exit {code}); recording and continuing")
            rows.append(row)
            continue
        dest = out_dir / cell["name"]
        dest.mkdir(parents=True, exist_ok=True)
        for fname in _SNAPSHOT_FILES:
            src = eval_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
        row.update(_read_metrics(eval_dir))
        rows.append(row)
    return rows


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
    parser.add_argument(
        "--reassemble",
        action="store_true",
        help="Skip fitting; rebuild comparison.{md,json} from existing --out-dir snapshots.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reassemble:
        rows = _collect_rows_from_snapshots(out_dir)
    else:
        if "AOTY_DATASET_PATH" not in os.environ:
            print(
                "warning: AOTY_DATASET_PATH not set; set it to data/raw/aoty_subset.csv "
                "for the subset bake-off.",
                file=sys.stderr,
            )
        rows = _run_grid(args.panelcast_bin, args.preset, eval_dir, out_dir)

    pairwise = _pairwise_elpd(rows, out_dir)

    payload = {"cells": _json_safe(rows), "pairwise_vs_default": _json_safe(pairwise)}
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = _render_markdown(rows, pairwise)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {out_dir / 'comparison.json'} and {out_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
