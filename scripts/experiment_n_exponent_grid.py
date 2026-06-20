"""n_exponent grid: fixed-value fits + test-set LOO table (Phase-8 stretch).

Fits the production model at a grid of fixed heteroscedastic exponents
(0.0 baseline plus 0.15-0.6; iid rating aggregation would imply 0.5, the
learned posterior says ~0.002) and scores each on held-out test ELPD via the
same latent-marginalized pointwise log-likelihood the evaluate stage uses.

GPU-budget item: each grid point is a full cheap fit (~10-15 min at the
default 2x500). Run sequentially from the repo root with nothing else on
the GPU:

    .pixi/envs/default/bin/python scripts/experiment_n_exponent_grid.py \
        [--target-transform identity|offset_logit] [--values 0.0,0.15,...]

Writes outputs/experiments/n_exponent_grid.json. Does NOT touch models/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panelcast.config.descriptor import load_descriptor
from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import priors_for_transform
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.evaluate import _compute_info_criteria, _prepare_test_model_args
from panelcast.pipelines.train_bayes import (
    _apply_max_albums_cap,
    load_training_data,
    locate_level_prior,
)

DEFAULT_VALUES = (0.0, 0.15, 0.25, 0.33, 0.4, 0.5, 0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Dataset descriptor (default AOTY)")
    parser.add_argument(
        "--target-transform",
        default="identity",
        choices=("identity", "offset_logit"),
        help="Model arm to run the grid on (identity unless logit was revived).",
    )
    parser.add_argument(
        "--values",
        default=",".join(str(v) for v in DEFAULT_VALUES),
        help="Comma-separated fixed n_exponent values.",
    )
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    values = tuple(float(v) for v in args.values.split(",") if v.strip())
    descriptor = load_descriptor(args.dataset)
    prefix = descriptor.model_prefix
    bounds = tuple(descriptor.target_bounds)
    transform_name = args.target_transform

    import pandas as pd

    model_args, feature_cols, train_df = load_training_data(
        features_path=Path("data/features/train_features.parquet"),
        splits_path=Path("data/splits/within_artist_temporal/train.parquet"),
        descriptor=descriptor,
        target_transform=transform_name,
        ar_center="global",
    )
    artist_album_counts = model_args.pop("artist_album_counts")
    artist_to_idx = model_args.pop("artist_to_idx")
    global_mean_score = float(model_args.pop("global_mean_score"))
    ar_center_value = float(model_args.pop("ar_center_value"))
    model_args = _apply_max_albums_cap(model_args, 50, artist_album_counts)

    X = np.asarray(model_args["X"])
    std = X.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    feature_means = X.mean(axis=0)
    model_args["X"] = ((X - feature_means) / std_safe).astype(np.float32)

    n_reviews = np.asarray(model_args["n_reviews"], dtype=float)
    n_ref = float(np.median(n_reviews))
    model_args["learn_n_exponent"] = False
    model_args["likelihood_df"] = 4.0
    model_args["target_bounds"] = bounds

    mcmc_config = MCMCConfig(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        target_accept_prob=args.target_accept,
        max_tree_depth=10,
    )
    base_priors = locate_level_prior(
        priors_for_transform(transform_name, ar_center="global"),
        ar_center_value=ar_center_value,
        target_transform=transform_name,
        target_bounds=bounds,
    )
    model = make_score_model(prefix)

    # Test-set inputs (mirrors the evaluate stage's primary-split path).
    split_dir = Path("data/splits/within_artist_temporal")
    test_df = pd.read_parquet(split_dir / "test.parquet")
    test_features = pd.read_parquet(
        Path("data/features/within_artist_temporal/test_features.parquet")
    )
    val_df = None
    val_path = split_dir / "validation.parquet"
    if val_path.exists():
        val_df = pd.read_parquet(val_path)

    transform = get_transform(transform_name, target_bounds=bounds, offset=0.5)
    results: dict[str, dict] = {}

    for value in values:
        name = f"n_exp_{value:g}"
        print(f"=== {name} ===", flush=True)
        grid_args = dict(model_args)
        grid_args["n_exponent"] = value
        grid_args["n_ref"] = n_ref if value != 0.0 else None
        grid_args["priors"] = base_priors

        fit_result = fit_model(
            model=model,
            model_args=grid_args,
            config=mcmc_config,
            progress_bar=True,
            exclude_from_idata=(f"{prefix}_rw_raw",),
            exclude_from_collection=(f"{prefix}_rw_raw",),
        )

        # Build the per-fit summary the evaluate helpers expect.
        summary = {
            "dataset": descriptor.to_summary_block(),
            "artist_to_idx": artist_to_idx,
            "n_artists": grid_args["n_artists"],
            "max_seq": grid_args["max_seq"],
            "max_albums": 50,
            "min_albums_filter": 2,
            "global_mean_score": global_mean_score,
            "likelihood_df": 4.0,
            "n_exponent": value,
            "learn_n_exponent": False,
            "n_ref": grid_args["n_ref"],
            "priors": base_priors.__dict__,
            "target_transform": transform_name,
            "logit_offset": 0.5,
            "ar_center_value": ar_center_value,
            "feature_cols": feature_cols,
            "feature_scaler": {"mean": feature_means.tolist(), "std": std_safe.tolist()},
        }
        test_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, val_df=val_df
        )
        test_args["n_exponent"] = value
        test_args["n_ref"] = grid_args["n_ref"]
        test_args["learn_n_exponent"] = False
        test_args["priors"] = base_priors

        posterior_samples = {
            site: np.asarray(fit_result.idata.posterior[site]).reshape(
                -1, *fit_result.idata.posterior[site].shape[2:]
            )
            for site in fit_result.idata.posterior.data_vars
        }
        if transform_name != "identity":
            y_for_loglik = np.asarray(transform.forward(y_true), dtype=np.float32)
        else:
            y_for_loglik = y_true
        try:
            info = _compute_info_criteria(
                posterior_samples=posterior_samples,
                model_args=test_args,
                y_true=y_for_loglik,
                n_chains=args.num_chains,
                n_draws=args.num_samples,
                prefix=prefix,
                transform=transform,
                y_raw=y_true,
                seed=args.seed,
            )
        except Exception as e:  # keep the grid going; record the failure
            info = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}

        results[name] = {
            "n_exponent": value,
            "divergences": fit_result.divergences,
            "runtime_seconds": fit_result.runtime_seconds,
            "peak_gpu_memory_bytes": fit_result.peak_gpu_memory_bytes,
            "info_criteria": info,
        }
        elpd = info.get("loo", {}).get("elpd") if isinstance(info, dict) else None
        print(
            f"    elpd={elpd} div={fit_result.divergences} " f"t={fit_result.runtime_seconds:.0f}s",
            flush=True,
        )

    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "n_exponent_grid.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "arm": transform_name,
                "mcmc": f"{args.num_chains}x{args.num_samples}/{args.num_warmup}",
                "n_ref": n_ref,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")

    print(f"\n{'value':>8} {'elpd':>12} {'se':>8} {'div':>5} {'sec':>6}")
    for name, row in results.items():
        loo = row["info_criteria"].get("loo", {})
        elpd = loo.get("elpd")
        se = loo.get("se")
        print(
            f"{row['n_exponent']:>8g} "
            f"{elpd if elpd is not None else float('nan'):>12.1f} "
            f"{se if se is not None else float('nan'):>8.1f} "
            f"{row['divergences']:>5d} {row['runtime_seconds']:>6.0f}"
        )


if __name__ == "__main__":
    main()
