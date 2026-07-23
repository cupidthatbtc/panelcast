from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

SPLITS = ("within_entity_temporal", "entity_disjoint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_metrics(data: dict, name: str) -> dict:
    split = data["splits"][name]
    point = split["point_metrics"]
    calibration = split["calibration"]
    result = {
        "n": point["n_observations"],
        "mae": point["mae"],
        "rmse": point["rmse"],
        "r2": point["r2"],
        "mean_bias": point["mean_bias"],
        "crps": split["crps"]["mean_crps"],
        "coverage_80": calibration["coverages"]["0.80"]["empirical"],
        "coverage_95": calibration["coverages"]["0.95"]["empirical"],
        "width_80": calibration["coverages"]["0.80"]["interval_width"],
        "width_95": calibration["coverages"]["0.95"]["interval_width"],
        "wis": calibration["wis"],
    }
    info = split.get("info_criteria", {}).get("heldout_elpd")
    if info:
        result["heldout_elpd"] = info["elpd"]
        result["heldout_elpd_se"] = info["se"]
    return result


def metric_delta(old: dict, fixed: dict) -> dict:
    return {
        key: fixed[key] - old[key]
        for key in fixed
        if key != "n"
        and key in old
        and isinstance(old[key], (int, float))
        and isinstance(fixed[key], (int, float))
    }


def row_identity(path: Path) -> dict:
    frame = pd.read_parquet(path)
    columns = [
        column
        for column in ("original_row_id", "Artist", "Album", "User_Score")
        if column in frame.columns
    ]
    records = frame[columns].where(pd.notna(frame[columns]), None).to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "rows": len(frame),
        "columns": columns,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def pointwise_elpd(path: Path) -> np.ndarray:
    values = np.asarray(az.from_netcdf(path).log_likelihood["y"])
    samples = values.reshape(-1, values.shape[-1])
    return np.asarray(logsumexp(samples, axis=0) - math.log(samples.shape[0]), dtype=float)


def paired_elpd(entity_path: Path, incumbent_path: Path) -> dict:
    entity = pointwise_elpd(entity_path)
    incumbent = pointwise_elpd(incumbent_path)
    if entity.shape != incumbent.shape:
        raise ValueError("pointwise log-likelihood arrays must have identical shapes")
    difference = entity - incumbent
    paired_se = float(math.sqrt(difference.size * np.var(difference, ddof=1)))
    total = float(difference.sum())
    return {
        "definition": "entity_obs_minus_incumbent",
        "n": int(difference.size),
        "difference": total,
        "paired_se": paired_se,
        "z": total / paired_se,
        "pointwise_difference": difference.tolist(),
    }


def validate_pairing(
    entity_fixed: Path,
    incumbent_fixed: Path,
    canonical_split: Path,
) -> None:
    relative = Path("evaluation/within_entity_temporal/predictions.json")
    canonical = pd.read_parquet(canonical_split)
    expected = {
        "y_true": canonical["User_Score"].to_numpy(dtype=float),
        "entity": canonical["Artist"].astype(str).to_numpy(),
        "event": canonical["Album"].astype(str).to_numpy(),
    }
    for label, fixed in (("entity_obs", entity_fixed), ("incumbent", incumbent_fixed)):
        predictions = load_json(fixed / relative)
        actual = {
            "y_true": np.asarray(predictions["y_true"], dtype=float),
            "entity": np.asarray(predictions["entity"], dtype=str),
            "event": np.asarray(predictions["event"], dtype=str),
        }
        for field, expected_values in expected.items():
            if actual[field].shape != expected_values.shape or not np.array_equal(
                actual[field], expected_values
            ):
                raise ValueError(f"{label} {field} does not match the canonical ordered split")


def write_baseline_csv(rows: list[dict], path: Path) -> None:
    table = []
    for row in rows:
        runtime = row["runtime_s"]
        table.append(
            {
                "Model": row["model"],
                "Split": row["split"],
                "N": row["n_obs"],
                "MAE": f"{row['mae']:.2f}",
                "RMSE": f"{row['rmse']:.2f}",
                "R²": f"{row['r2']:.3f}",
                "CRPS": f"{row['crps']:.2f}",
                "80% Cov": f"{row['cov80']:.3f}",
                "95% Cov": f"{row['cov95']:.3f}",
                "95% Width": f"{row['width95']:.2f}",
                "PPC skew p": f"{row['ppc_skew_p']:.3f}",
                "Runtime (s)": "—" if runtime is None else f"{runtime:.2f}",
            }
        )
    pd.DataFrame(table).to_csv(path)


def arm_record(source: Path, fixed: Path, model_file: str) -> dict:
    source_metrics_path = source / "evaluation/metrics.json"
    fixed_metrics_path = fixed / "evaluation/metrics.json"
    source_metrics = load_json(source_metrics_path)
    fixed_metrics = load_json(fixed_metrics_path)
    old = {split: split_metrics(source_metrics, split) for split in SPLITS}
    new = {split: split_metrics(fixed_metrics, split) for split in SPLITS}
    return {
        "source_run": source.name,
        "fixed_output_run": fixed.name,
        "model_file": model_file,
        "model_sha256": sha256(source / "models" / model_file),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "training_summary_sha256": sha256(source / "models/training_summary.json"),
        "resolved_config_sha256": sha256(source / "resolved_config.yaml"),
        "resolved_config": yaml.safe_load(
            (source / "resolved_config.yaml").read_text(encoding="utf-8")
        ),
        "manifest_input_hashes": load_json(source / "manifest.json").get("input_hashes", {}),
        "archived_metrics_sha256": sha256(source_metrics_path),
        "fixed_metrics_sha256": sha256(fixed_metrics_path),
        "pointwise_log_likelihood_sha256": sha256(fixed / "evaluation/log_likelihood.nc"),
        "archived_metrics": old,
        "fixed_metrics": new,
        "fixed_minus_archived": {
            split: metric_delta(old[split], new[split]) for split in SPLITS
        },
    }


def builder_argv(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        ".audit/fair_eval_0131/build_evidence.py",
        "--entity-source",
        str(args.entity_source.resolve()),
        "--entity-fixed",
        str(args.entity_fixed.resolve()),
        "--entity-model",
        args.entity_model,
        "--incumbent-source",
        str(args.incumbent_source.resolve()),
        "--incumbent-fixed",
        str(args.incumbent_fixed.resolve()),
        "--incumbent-model",
        args.incumbent_model,
        "--data-root",
        str(args.data_root.resolve()),
        "--output",
        ".audit/fair_eval_0131",
        "--evaluated-at",
        args.evaluated_at,
        "--evaluator-revision",
        args.evaluator_revision,
        "--code-base-revision",
        args.code_base_revision,
    ]


def build(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = {}
    identities = {}
    for split in SPLITS:
        feature_path = data_root / "features" / split / "test_features.parquet"
        split_path = data_root / "splits" / split / "test.parquet"
        files[f"features/{split}/test_features.parquet"] = sha256(feature_path)
        files[f"splits/{split}/test.parquet"] = sha256(split_path)
        identities[split] = row_identity(split_path)

    validate_pairing(
        args.entity_fixed,
        args.incumbent_fixed,
        data_root / "splits" / SPLITS[0] / "test.parquet",
    )
    paired = paired_elpd(
        args.entity_fixed / "evaluation/log_likelihood.nc",
        args.incumbent_fixed / "evaluation/log_likelihood.nc",
    )
    if paired["n"] != identities[SPLITS[0]]["rows"]:
        raise ValueError("pointwise log likelihood does not match the paired split")

    baseline_dir = args.entity_fixed / "reports" / "baselines"
    baseline_json = baseline_dir / "baseline_comparison.json"
    baseline_csv = output / "baseline_comparison.csv"
    baseline_rows = load_json(baseline_json)
    if len(baseline_rows) != 13 or not any(row.get("model") == "ridge" for row in baseline_rows):
        raise ValueError("baseline comparison is incomplete")
    shutil.copy2(baseline_json, output / baseline_json.name)
    write_baseline_csv(baseline_rows, baseline_csv)

    stamp = load_json(data_root / "features/.stamp.json")
    return {
        "schema_version": 1,
        "issue": 247,
        "evaluated_at": args.evaluated_at,
        "evaluator_revision": args.evaluator_revision,
        "code_base_revision": args.code_base_revision,
        "environment": {
            "platform": "WSL2 Linux",
            "device": "CPU",
            "jax_platforms": "cpu",
            "seed": 42,
            "save_log_likelihood": True,
        },
        "commands": [
            "JAX_PLATFORMS=cpu "
            + shlex.join(
                [
                    "python",
                    ".audit/fair_eval_0131/reproduce.py",
                    str(args.entity_source.resolve()),
                    str(args.entity_fixed.resolve()),
                ]
            ),
            "JAX_PLATFORMS=cpu "
            + shlex.join(
                [
                    "python",
                    ".audit/fair_eval_0131/reproduce.py",
                    str(args.incumbent_source.resolve()),
                    str(args.incumbent_fixed.resolve()),
                ]
            ),
            shlex.join(
                [
                    "panelcast",
                    "compare",
                    "--baselines",
                    "--metrics",
                    str(args.entity_fixed.resolve() / "evaluation/metrics.json"),
                    "--output",
                    str(args.entity_fixed.resolve() / "reports/baselines"),
                ]
            ),
            shlex.join(builder_argv(args)),
        ],
        "builder_argv": builder_argv(args),
        "data": {
            "feature_input_hash": stamp["input_hash"],
            "files": files,
            "row_identity": identities,
        },
        "arms": {
            "entity_obs": arm_record(
                args.entity_source, args.entity_fixed, args.entity_model
            ),
            "incumbent": arm_record(
                args.incumbent_source, args.incumbent_fixed, args.incumbent_model
            ),
        },
        "paired_elpd": paired,
        "baseline_comparison_sha256": sha256(output / baseline_json.name),
        "baseline_comparison_csv_sha256": sha256(baseline_csv),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-source", type=Path, required=True)
    parser.add_argument("--entity-fixed", type=Path, required=True)
    parser.add_argument("--entity-model", required=True)
    parser.add_argument("--incumbent-source", type=Path, required=True)
    parser.add_argument("--incumbent-fixed", type=Path, required=True)
    parser.add_argument("--incumbent-model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--code-base-revision", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    record = build(args)
    (args.output / "fair_eval.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
