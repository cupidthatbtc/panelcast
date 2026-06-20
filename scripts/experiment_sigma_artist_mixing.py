"""Phase-6 bake-off: sigma_artist mixing under parameterization/prior gates.

Compares ESS and R-hat for the four combinations of
artist_effect_param x sigma_artist_prior_type on the real training data at
reduced MCMC settings (2 chains x 300/300). Motivated by the published
sigma_artist ESS deficit (561 < 800 at production settings).

Run sequentially (one MCMC at a time) from the repo root:
    .pixi/envs/default/bin/python scripts/experiment_sigma_artist_mixing.py

Pass --dataset to run against a non-AOTY descriptor (paths/prefix/bounds are
derived from it; note the bundled aero fixture is ~44 synthetic rows — it
exercises the mechanics, not the statistics).

Writes results to outputs/experiments/sigma_artist_mixing.json. Does NOT
touch models/ (no save_model, no manifest update).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np

from panelcast.config.descriptor import DatasetDescriptor, load_descriptor
from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.priors import priors_for_transform
from panelcast.pipelines.train_bayes import (
    _apply_max_albums_cap,
    load_training_data,
    locate_level_prior,
)

VARIANTS: dict[str, dict] = {
    "noncentered_halfnormal": {},
    "zerosum_halfnormal": {"artist_effect_param": "zerosum"},
    "noncentered_lognormal": {"sigma_artist_prior_type": "lognormal"},
    "zerosum_lognormal": {
        "artist_effect_param": "zerosum",
        "sigma_artist_prior_type": "lognormal",
    },
}


def watch_sites(prefix: str) -> list[str]:
    return [
        f"{prefix}_sigma_artist",
        f"{prefix}_mu_artist",
        f"{prefix}_rho",
        f"{prefix}_sigma_rw",
    ]


def prepare_args(descriptor: DatasetDescriptor) -> tuple[dict, float]:
    """Replicate train_models data prep at defaults (identity, centered)."""
    model_args, feature_cols, _train_df = load_training_data(
        features_path=Path("data/features/train_features.parquet"),
        splits_path=Path("data/splits/within_artist_temporal/train.parquet"),
        descriptor=descriptor,
    )
    artist_album_counts = model_args.pop("artist_album_counts")
    model_args.pop("artist_to_idx")
    model_args.pop("global_mean_score")
    ar_center_value = model_args.pop("ar_center_value")
    model_args = _apply_max_albums_cap(model_args, 50, artist_album_counts)

    X = model_args["X"]
    std = X.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    model_args["X"] = ((X - X.mean(axis=0)) / std_safe).astype(np.float32)

    model_args["n_exponent"] = 0.0
    model_args["learn_n_exponent"] = False
    model_args["n_ref"] = None
    model_args["likelihood_df"] = 4.0
    model_args["target_bounds"] = tuple(descriptor.target_bounds)
    return model_args, float(ar_center_value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset descriptor (bare name or YAML path); omit for AOTY defaults.",
    )
    cli_args = parser.parse_args()
    descriptor = load_descriptor(cli_args.dataset)
    prefix = descriptor.model_prefix
    sites = watch_sites(prefix)

    model_args, ar_center_value = prepare_args(descriptor)
    config = MCMCConfig(
        num_warmup=300,
        num_samples=300,
        num_chains=2,
        seed=42,
        target_accept_prob=0.9,
        max_tree_depth=10,
        chain_method="sequential",
    )

    results: dict[str, dict] = {}
    for name, overrides in VARIANTS.items():
        priors = locate_level_prior(
            priors_for_transform("identity", ar_center="global", **overrides),
            ar_center_value=ar_center_value,
        )
        args = dict(model_args)
        args["priors"] = priors
        print(f"=== {name} ===", flush=True)
        fit_result = fit_model(
            model=make_score_model(prefix),
            model_args=args,
            config=config,
            progress_bar=True,
            exclude_from_idata=(f"{prefix}_rw_raw",),
        )
        summ = az.summary(fit_result.idata, var_names=sites, kind="diagnostics")
        # az.summary indexes by site name (flattened); collect per-site rows.
        site_stats = {
            str(idx): {
                "ess_bulk": float(row["ess_bulk"]),
                "ess_tail": float(row["ess_tail"]),
                "r_hat": float(row["r_hat"]),
            }
            for idx, row in summ.iterrows()
        }
        results[name] = {
            "sites": site_stats,
            "divergences": int(fit_result.divergences),
            "runtime_seconds": float(fit_result.runtime_seconds),
        }
        sig = site_stats.get(f"{prefix}_sigma_artist", {})
        print(
            f"    sigma_artist ess_bulk={sig.get('ess_bulk')} "
            f"r_hat={sig.get('r_hat')} div={fit_result.divergences} "
            f"t={fit_result.runtime_seconds:.0f}s",
            flush=True,
        )

    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sigma_artist_mixing.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": "2x300/300 identity centered",
                "dataset": descriptor.name,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")

    print(f"\n{'variant':26} {'ess_bulk':>9} {'ess_tail':>9} {'r_hat':>6} {'div':>4} {'sec':>5}")
    for name, r in results.items():
        sig = r["sites"].get(f"{prefix}_sigma_artist", {})
        print(
            f"{name:26} {sig.get('ess_bulk', float('nan')):>9.0f} "
            f"{sig.get('ess_tail', float('nan')):>9.0f} "
            f"{sig.get('r_hat', float('nan')):>6.3f} "
            f"{r['divergences']:>4d} {r['runtime_seconds']:>5.0f}"
        )


if __name__ == "__main__":
    main()
