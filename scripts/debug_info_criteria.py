"""Reproduce the silently-swallowed LOO/WAIC failure on the saved model.

Both cheap validation runs recorded info_criteria as status="unavailable"
with an EMPTY reason (str(e) was empty — MemoryError signature). This script
replays the exact evaluate-stage call path against the current saved model
and prints the full exception, so the root cause is diagnosable.

Run from the repo root (GPU, nothing else running):
    .pixi/envs/default/bin/python scripts/debug_info_criteria.py
Add --cpu to force JAX onto the CPU (distinguishes GPU-OOM from logic bugs).
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Force JAX_PLATFORMS=cpu")
    parser.add_argument(
        "--max-draws",
        type=int,
        default=None,
        help="Optionally truncate posterior draws to this many (memory probe).",
    )
    args = parser.parse_args()

    if args.cpu:
        import os

        os.environ["JAX_PLATFORMS"] = "cpu"

    import arviz as az
    import pandas as pd

    from panelcast.pipelines.evaluate import (
        _compute_info_criteria,
        _extract_posterior_samples,
        _prepare_test_model_args,
    )
    from panelcast.pipelines.training_summary import load_training_summary

    models_dir = Path("models")
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    current = manifest["current"]
    model_name = next(iter(current.values())) if isinstance(current, dict) else current
    print(f"model: {model_name}")

    model_path = models_dir / model_name
    if model_path.suffix != ".nc":
        model_path = model_path.with_suffix(".nc")
    idata = az.from_netcdf(model_path)
    summary = load_training_summary(models_dir / "training_summary.json").to_json_dict()
    prefix = summary["dataset"]["model_prefix"]

    split_dir = Path("data/splits/within_artist_temporal")
    test_df = pd.read_parquet(split_dir / "test.parquet")
    train_df = pd.read_parquet(split_dir / "train.parquet")
    test_features = pd.read_parquet(
        Path("data/features/within_artist_temporal/test_features.parquet")
    )
    val_df = None
    val_path = split_dir / "validation.parquet"
    if val_path.exists():
        val_df = pd.read_parquet(val_path)

    model_args, y_true = _prepare_test_model_args(
        test_df, test_features, summary, train_df=train_df, val_df=val_df, strict=False
    )
    posterior_samples = _extract_posterior_samples(idata)
    if args.max_draws is not None:
        posterior_samples = {k: v[: args.max_draws] for k, v in posterior_samples.items()}
    first_var = next(iter(idata.posterior.data_vars))
    n_chains = int(idata.posterior[first_var].shape[0])
    n_draws = int(idata.posterior[first_var].shape[1])
    n_total = next(iter(posterior_samples.values())).shape[0]
    print(
        f"posterior: {n_chains} chains x {n_draws} draws (using {n_total}), "
        f"n_test={len(y_true)}"
    )

    try:
        result = _compute_info_criteria(
            posterior_samples=posterior_samples,
            model_args=model_args,
            y_true=y_true,
            n_chains=n_chains,
            n_draws=n_draws,
            prefix=prefix,
        )
        print("SUCCESS:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e!r}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
