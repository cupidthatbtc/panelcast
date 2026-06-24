"""Validate preflight memory extrapolation against measured GPU peaks.

Motivation: the quick formula over-estimated the 4x5000 publication run by
4.3x (96.9 GB vs the calibrated 22.8 GB projection), and the calibrated
projection itself is a linear extrapolation from one tiny two-point
calibration (10 & 50 samples) that was never validated against a real
measured peak.

This script runs a ladder of real mini-MCMC subprocesses on the actual
training data (one GPU job at a time), records measured peaks, fits linear
models on the lower rungs against several candidate scaling variables, and
reports prediction error at the held-out top rung. Two probe rungs isolate
the warmup and chain-count contributions. The existing 10/50 calibration and
the quick formula are scored against the same measurements.

Run from the repo root with nothing else on the GPU (takes ~30-40 min):
    .pixi/envs/default/bin/python scripts/experiment_preflight_validation.py

Writes outputs/experiments/preflight_validation.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from panelcast.config.descriptor import load_descriptor
from panelcast.gpu_memory.estimate import estimate_memory_gb
from panelcast.pipelines.train_bayes import _apply_max_albums_cap, load_training_data
from panelcast.preflight.calibrate import calculate_calibration
from panelcast.preflight.full_check import (
    _derive_dimensions_from_model_args,
    _run_mini_mcmc_subprocess,
    serialize_model_args,
)

# Ladder rungs: the first four share warmup=samples at 2 chains (fit/holdout),
# the last two probe warmup and chain-count contributions independently.
RUNGS: list[dict[str, Any]] = [
    {"name": "2x50", "num_chains": 2, "num_warmup": 50, "num_samples": 50},
    {"name": "2x100", "num_chains": 2, "num_warmup": 100, "num_samples": 100},
    {"name": "2x250", "num_chains": 2, "num_warmup": 250, "num_samples": 250},
    {"name": "2x500", "num_chains": 2, "num_warmup": 500, "num_samples": 500},
    {"name": "2x250_w50", "num_chains": 2, "num_warmup": 50, "num_samples": 250},
    {"name": "1x250", "num_chains": 1, "num_warmup": 250, "num_samples": 250},
    # The production calibration's own two points (1 chain, 10 warmup).
    {"name": "cal_10", "num_chains": 1, "num_warmup": 10, "num_samples": 10},
    {"name": "cal_50", "num_chains": 1, "num_warmup": 10, "num_samples": 50},
]

FIT_RUNGS = ("2x50", "2x100", "2x250")
HOLDOUT_RUNG = "2x500"

# Candidate scaling variables for the linear fit.
PREDICTORS = {
    "samples_per_chain": lambda r: r["num_samples"],
    "total_kept_draws": lambda r: r["num_chains"] * r["num_samples"],
    "warmup_plus_samples": lambda r: r["num_warmup"] + r["num_samples"],
}


def prepare_mini_run_args(descriptor) -> dict:
    """Real training data shaped exactly like the production fit."""
    model_args, _feature_cols, _train_df = load_training_data(
        features_path=Path("data/features/train_features.parquet"),
        splits_path=Path("data/splits/within_entity_temporal/train.parquet"),
        descriptor=descriptor,
    )
    artist_album_counts = model_args.pop("artist_album_counts")
    model_args = _apply_max_albums_cap(model_args, 50, artist_album_counts)

    X = model_args["X"]
    std = X.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    X_std = ((X - X.mean(axis=0)) / std_safe).astype(np.float32)

    # Only the keys the mini_run subprocess consumes.
    return {
        "artist_idx": model_args["artist_idx"],
        "album_seq": model_args["album_seq"],
        "prev_score": model_args["prev_score"],
        "X": X_std,
        "y": model_args["y"],
        "n_artists": model_args["n_artists"],
        "max_seq": model_args["max_seq"],
        "n_reviews": model_args["n_reviews"],
        "n_exponent": 0.0,
        "learn_n_exponent": False,
    }


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares (intercept, slope)."""
    coeffs = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
    return float(coeffs[1]), float(coeffs[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Dataset descriptor (default AOTY)")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=2400,
        help="Per-rung subprocess timeout (the 2x500 rung takes ~15 min).",
    )
    cli_args = parser.parse_args()

    # The parent does CPU-only data prep; the measurement subprocesses must
    # see the GPU. Run the parent with JAX_PLATFORMS=cpu if you want to be
    # extra safe — it is stripped from the child environment here either way.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.pop("JAX_PLATFORMS", None)

    descriptor = load_descriptor(cli_args.dataset)
    mini_args = prepare_mini_run_args(descriptor)
    n_obs, n_artists, n_features, max_seq = _derive_dimensions_from_model_args(mini_args)
    print(
        f"data: n_obs={n_obs} n_artists={n_artists} " f"n_features={n_features} max_seq={max_seq}",
        flush=True,
    )

    args_path = serialize_model_args(mini_args)
    measurements: dict[str, dict] = {}
    try:
        for rung in RUNGS:
            rung_name = str(rung["name"])
            print(f"=== {rung_name} ===", flush=True)
            result = _run_mini_mcmc_subprocess(
                args_path,
                timeout_seconds=cli_args.timeout_seconds,
                num_warmup=int(rung["num_warmup"]),
                num_samples=int(rung["num_samples"]),
                num_chains=int(rung["num_chains"]),
                prefix=descriptor.model_prefix,
            )
            if not result.get("success", False):
                raise SystemExit(f"Rung {rung_name} failed: {result.get('error')}")
            peak_gb = result["peak_memory_bytes"] / (1024**3)
            measurements[rung_name] = {
                **rung,
                "peak_gb": peak_gb,
                "runtime_seconds": result["runtime_seconds"],
            }
            print(
                f"    peak={peak_gb:.2f} GiB t={result['runtime_seconds']:.0f}s",
                flush=True,
            )
    finally:
        args_path.unlink(missing_ok=True)

    holdout = measurements[HOLDOUT_RUNG]

    # Fit each candidate predictor on the three lower fit rungs; score on the
    # held-out 2x500 rung.
    fits: dict[str, dict] = {}
    for pred_name, pred_fn in PREDICTORS.items():
        xs = [float(pred_fn(measurements[name])) for name in FIT_RUNGS]
        ys = [measurements[name]["peak_gb"] for name in FIT_RUNGS]
        intercept, slope = fit_line(xs, ys)
        projected = intercept + slope * float(pred_fn(holdout))
        error_pct = 100.0 * (projected - holdout["peak_gb"]) / holdout["peak_gb"]
        fits[pred_name] = {
            "intercept_gb": intercept,
            "per_unit_gb": slope,
            "projected_holdout_gb": projected,
            "measured_holdout_gb": holdout["peak_gb"],
            "error_percent": error_pct,
        }

    # Production calibration baseline: 10/50-point fit, extrapolated on
    # warmup+samples exactly as run_extrapolated_preflight_check does.
    cal_fixed, cal_per_sample = calculate_calibration(
        (10, measurements["cal_10"]["peak_gb"]),
        (50, measurements["cal_50"]["peak_gb"]),
    )
    cal_target = holdout["num_warmup"] + holdout["num_samples"]
    cal_projected = cal_fixed + cal_per_sample * cal_target
    cal_error_pct = 100.0 * (cal_projected - holdout["peak_gb"]) / holdout["peak_gb"]

    # Quick-formula baseline at each rung.
    quick = {}
    for name, m in measurements.items():
        est = estimate_memory_gb(
            n_observations=n_obs,
            n_features=n_features,
            n_artists=n_artists,
            max_seq=max_seq,
            num_chains=m["num_chains"],
            num_samples=m["num_samples"],
            num_warmup=m["num_warmup"],
        )
        quick[name] = {
            "estimate_gb": est.total_gb,
            "measured_gb": m["peak_gb"],
            "ratio": est.total_gb / m["peak_gb"] if m["peak_gb"] > 0 else None,
        }

    out = {
        "dataset": descriptor.name,
        "dimensions": {
            "n_observations": n_obs,
            "n_artists": n_artists,
            "n_features": n_features,
            "max_seq": max_seq,
        },
        "measurements": measurements,
        "ladder_fits": fits,
        "production_calibration_baseline": {
            "fixed_overhead_gb": cal_fixed,
            "per_sample_gb": cal_per_sample,
            "target_samples": cal_target,
            "projected_holdout_gb": cal_projected,
            "measured_holdout_gb": holdout["peak_gb"],
            "error_percent": cal_error_pct,
        },
        "quick_formula": quick,
    }

    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preflight_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    print(f"\n{'rung':12} {'chains':>6} {'warmup':>6} {'samples':>7} {'peak GiB':>9}")
    for name, m in measurements.items():
        print(
            f"{name:12} {m['num_chains']:>6d} {m['num_warmup']:>6d} "
            f"{m['num_samples']:>7d} {m['peak_gb']:>9.2f}"
        )
    print(f"\n{'predictor':22} {'projected 2x500':>15} {'measured':>9} {'error %':>8}")
    for pred_name, fit in fits.items():
        print(
            f"{pred_name:22} {fit['projected_holdout_gb']:>15.2f} "
            f"{fit['measured_holdout_gb']:>9.2f} {fit['error_percent']:>8.1f}"
        )
    print(
        f"{'cal(10/50)+w+s':22} {cal_projected:>15.2f} "
        f"{holdout['peak_gb']:>9.2f} {cal_error_pct:>8.1f}"
    )


if __name__ == "__main__":
    main()
