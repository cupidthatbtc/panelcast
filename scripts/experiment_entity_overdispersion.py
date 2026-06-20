"""Entity-overdispersion / lognormal-sigma_obs bake-off (gated upgrade family).

Evaluates the two default-off observation-noise gates as one experiment
family on held-out data, mirroring ``experiment_sigma_artist_mixing.py`` (the
ESS harness) and ``experiment_n_exponent_grid.py`` (the test-set LOO/ELPD
harness). For each variant it records, at cheap MCMC settings:

  - convergence: per-watch-site ESS / R-hat, divergence count
  - generalization: held-out LOO ELPD (latent-marginalized, same path as the
    evaluate stage) on the score scale
  - calibration: held-out 80/95 posterior-predictive coverage

Variants (the gates share the variance budget, so they are compared jointly):
    off            baseline (published behavior, bit-identical)
    c1_0.25        heteroscedastic_entity_obs, tau_entity_scale=0.25
    c1_0.5         heteroscedastic_entity_obs, tau_entity_scale=0.5
    c2_lognormal   sigma_obs_prior_type=lognormal
    c1c2           both (tau_entity_scale=0.25 + lognormal sigma_obs)

Run sequentially (one MCMC at a time) from a domain's working directory so
``data/features`` and ``data/splits`` resolve to that domain, e.g.:

    cd <domain_dir>   # imdb_episodes/, science_impact_econ/, science_impact/
    <repo>/.pixi/envs/default/bin/python \
        <repo>/scripts/experiment_entity_overdispersion.py \
        --dataset descriptor.yaml --data-root .

Writes ``outputs/experiments/entity_overdispersion.json`` (under the cwd).
Does NOT touch models/ (no save_model, no manifest update).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from jax import random
from numpyro.infer import Predictive

from panelcast.config.descriptor import load_descriptor
from panelcast.evaluation.calibration import compute_coverage
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

# Each variant is a set of PriorConfig overrides forwarded through
# priors_for_transform; "off" is the published default.
VARIANTS: dict[str, dict] = {
    "off": {},
    "c1_0.25": {"heteroscedastic_entity_obs": True, "tau_entity_scale": 0.25},
    "c1_0.5": {"heteroscedastic_entity_obs": True, "tau_entity_scale": 0.5},
    "c2_lognormal": {"sigma_obs_prior_type": "lognormal"},
    "c1c2": {
        "heteroscedastic_entity_obs": True,
        "tau_entity_scale": 0.25,
        "sigma_obs_prior_type": "lognormal",
    },
}


def watch_sites(prefix: str) -> list[str]:
    # tau_entity is only present on the C1 variants; az.summary skips absent
    # names, so listing it unconditionally is safe.
    return [
        f"{prefix}_sigma_obs",
        f"{prefix}_sigma_artist",
        f"{prefix}_sigma_rw",
        f"{prefix}_tau_entity",
    ]


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    return ((X - mean) / std_safe).astype(np.float32), mean, std_safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Dataset descriptor (default AOTY).")
    parser.add_argument(
        "--data-root",
        default=".",
        help="Directory holding data/features and data/splits (default cwd).",
    )
    parser.add_argument(
        "--target-transform",
        default="identity",
        choices=("identity", "offset_logit"),
    )
    parser.add_argument("--max-albums", type=int, default=50)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated subset of variant names to run.",
    )
    parser.add_argument(
        "--entity-obs-keep-max",
        type=int,
        default=20000,
        help=(
            "Keep entity_obs_raw in the saved fit (fair forward eval) when "
            "n_artists <= this; exclude it above (memory). Default 20000."
        ),
    )
    cli = parser.parse_args()

    descriptor = load_descriptor(cli.dataset)
    prefix = descriptor.model_prefix
    bounds = tuple(descriptor.target_bounds)
    transform_name = cli.target_transform
    transform = get_transform(transform_name, target_bounds=bounds, offset=0.5)
    data_root = Path(cli.data_root)
    selected = [v.strip() for v in cli.variants.split(",") if v.strip()]

    # --- Training data (mirrors train_models prep at defaults) ---
    model_args, feature_cols, train_df = load_training_data(
        features_path=data_root / "data/features/train_features.parquet",
        splits_path=data_root / "data/splits/within_artist_temporal/train.parquet",
        descriptor=descriptor,
        target_transform=transform_name,
        ar_center="global",
    )
    artist_album_counts = model_args.pop("artist_album_counts")
    artist_to_idx = model_args.pop("artist_to_idx")
    global_mean_score = float(model_args.pop("global_mean_score"))
    ar_center_value = float(model_args.pop("ar_center_value"))
    model_args = _apply_max_albums_cap(model_args, cli.max_albums, artist_album_counts)

    model_args["X"], feature_means, feature_std = _standardize(np.asarray(model_args["X"]))
    model_args["learn_n_exponent"] = False
    model_args["n_exponent"] = 0.0
    model_args["n_ref"] = None
    model_args["likelihood_df"] = 4.0
    model_args["target_bounds"] = bounds

    # --- Held-out test inputs (mirrors evaluate's primary-split path) ---
    split_dir = data_root / "data/splits/within_artist_temporal"
    test_df = pd.read_parquet(split_dir / "test.parquet")
    test_features = pd.read_parquet(
        data_root / "data/features/within_artist_temporal/test_features.parquet"
    )
    val_path = split_dir / "validation.parquet"
    val_df = pd.read_parquet(val_path) if val_path.exists() else None

    mcmc_config = MCMCConfig(
        num_warmup=cli.num_warmup,
        num_samples=cli.num_samples,
        num_chains=cli.num_chains,
        seed=cli.seed,
        target_accept_prob=cli.target_accept,
        max_tree_depth=10,
    )
    model = make_score_model(prefix)
    sites = watch_sites(prefix)
    results: dict[str, dict] = {}

    for name in selected:
        overrides = VARIANTS[name]
        print(f"=== {name} ({overrides or 'baseline'}) ===", flush=True)
        priors = locate_level_prior(
            priors_for_transform(transform_name, ar_center="global", **overrides),
            ar_center_value=ar_center_value,
            target_transform=transform_name,
            target_bounds=bounds,
        )
        gate_on = bool(overrides.get("heteroscedastic_entity_obs", False))
        fit_args = dict(model_args)
        fit_args["priors"] = priors

        # Always drop the big rw_raw tensor. For entity_obs_raw, KEEP it when
        # the entity cardinality is small (so the held-out forward LOO/coverage
        # condition on each seen entity's *fitted* overdispersion -- the correct
        # warm-split treatment); only drop it above the cap, where memory forces
        # the cold-start prior-marginalization. This makes the C1 evaluation
        # fair on forward splits without touching the model or its parity.
        n_artists = int(fit_args["n_artists"])
        keep_entity = gate_on and n_artists <= cli.entity_obs_keep_max
        idata_excludes = [f"{prefix}_rw_raw"]
        collection_excludes = [f"{prefix}_rw_raw"]
        if gate_on and not keep_entity:
            idata_excludes.append(f"{prefix}_entity_obs_raw")
            collection_excludes.append(f"{prefix}_entity_obs_raw")

        fit_result = fit_model(
            model=model,
            model_args=fit_args,
            config=mcmc_config,
            progress_bar=True,
            exclude_from_idata=tuple(idata_excludes),
            exclude_from_collection=tuple(collection_excludes),
        )

        # --- Convergence: per-site ESS / R-hat ---
        present = [s for s in sites if s in fit_result.idata.posterior.data_vars]
        summ = az.summary(fit_result.idata, var_names=present, kind="diagnostics")
        site_stats = {
            str(idx): {
                "ess_bulk": float(row["ess_bulk"]),
                "ess_tail": float(row["ess_tail"]),
                "r_hat": float(row["r_hat"]),
            }
            for idx, row in summ.iterrows()
        }

        # --- Flatten posterior for the evaluate helpers ---
        posterior_samples = {
            site: np.asarray(fit_result.idata.posterior[site]).reshape(
                -1, *fit_result.idata.posterior[site].shape[2:]
            )
            for site in fit_result.idata.posterior.data_vars
        }

        # --- Held-out LOO ELPD (score scale) ---
        summary = {
            "dataset": descriptor.to_summary_block(),
            "artist_to_idx": artist_to_idx,
            "n_artists": fit_args["n_artists"],
            "max_seq": fit_args["max_seq"],
            "max_albums": cli.max_albums,
            "min_albums_filter": 2,
            "global_mean_score": global_mean_score,
            "likelihood_df": 4.0,
            "n_exponent": 0.0,
            "learn_n_exponent": False,
            "n_ref": None,
            "priors": priors.__dict__,
            "target_transform": transform_name,
            "logit_offset": 0.5,
            "ar_center_value": ar_center_value,
            "feature_cols": feature_cols,
            "feature_scaler": {"mean": feature_means.tolist(), "std": feature_std.tolist()},
        }
        test_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, val_df=val_df
        )
        test_args["n_exponent"] = 0.0
        test_args["n_ref"] = None
        test_args["learn_n_exponent"] = False
        test_args["priors"] = priors

        y_for_loglik = (
            np.asarray(transform.forward(y_true), dtype=np.float32)
            if transform_name != "identity"
            else y_true
        )
        try:
            info = _compute_info_criteria(
                posterior_samples=posterior_samples,
                model_args=test_args,
                y_true=y_for_loglik,
                n_chains=cli.num_chains,
                n_draws=cli.num_samples,
                prefix=prefix,
                transform=transform,
                y_raw=y_true,
                seed=cli.seed,
            )
        except Exception as e:  # keep the family going; record the failure
            info = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}

        # --- Held-out 80/95 coverage (Predictive marginalizes excluded latents) ---
        try:
            predictive = Predictive(model, posterior_samples=posterior_samples, batch_ndims=1)
            pred = predictive(random.key(cli.seed), **{**test_args, "y": None})
            y_pred = np.asarray(pred[f"{prefix}_y"])  # (n_samples, n_obs_test)
            if transform_name != "identity":
                y_pred = np.asarray(transform.backward(y_pred))
            coverage: dict[str, Any] = {}
            for prob in (0.80, 0.95):
                cov = compute_coverage(y_true, y_pred, prob=prob)
                coverage[f"{prob:.2f}"] = {
                    "empirical": float(cov.empirical),
                    "interval_width": float(cov.interval_width),
                }
        except Exception as e:
            coverage = {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}

        results[name] = {
            "overrides": overrides,
            "sites": site_stats,
            "divergences": int(fit_result.divergences),
            "runtime_seconds": float(fit_result.runtime_seconds),
            "info_criteria": info,
            "coverage": coverage,
        }

        elpd = info.get("loo", {}).get("elpd") if isinstance(info, dict) else None
        cov80 = coverage.get("0.80", {}).get("empirical") if isinstance(coverage, dict) else None
        cov95 = coverage.get("0.95", {}).get("empirical") if isinstance(coverage, dict) else None
        print(
            f"    elpd={elpd} cov80={cov80} cov95={cov95} "
            f"div={fit_result.divergences} t={fit_result.runtime_seconds:.0f}s",
            flush=True,
        )

    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entity_overdispersion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": descriptor.name,
                "arm": transform_name,
                "mcmc": f"{cli.num_chains}x{cli.num_samples}/{cli.num_warmup}",
                "max_albums": cli.max_albums,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")

    # Compact comparison table.
    print(
        f"\n{'variant':14} {'elpd':>10} {'se':>7} {'cov80':>6} {'cov95':>6} {'div':>4} {'sec':>5}"
    )
    for name, r in results.items():
        loo = r["info_criteria"].get("loo", {}) if isinstance(r["info_criteria"], dict) else {}
        cov = r["coverage"] if isinstance(r["coverage"], dict) else {}
        elpd = loo.get("elpd", float("nan"))
        se = loo.get("se", float("nan"))
        c80 = cov.get("0.80", {}).get("empirical", float("nan"))
        c95 = cov.get("0.95", {}).get("empirical", float("nan"))
        print(
            f"{name:14} {elpd:>10.1f} {se:>7.1f} {c80:>6.2f} {c95:>6.2f} "
            f"{r['divergences']:>4d} {r['runtime_seconds']:>5.0f}"
        )


if __name__ == "__main__":
    main()
