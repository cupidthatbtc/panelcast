"""Next-album prediction pipeline.

Generates predictions for:
- Known artists (3 scenarios): next album using trained artist effects
- New/hypothetical artists (2 scenarios): using population distribution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import structlog
from jax import random
from numpyro.infer import Predictive

from panelcast.data.alignment import join_splits_with_features
from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.predict import extract_posterior_samples, predict_new_artist
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.training_summary import (
    ar_center_on_model_scale,
    load_training_summary,
)

if TYPE_CHECKING:
    from panelcast.pipelines.stages import StageContext

log = structlog.get_logger()

# Scenario names
SCENARIOS_KNOWN = ["same", "population_mean", "artist_mean"]
SCENARIOS_NEW = ["population", "debut_defaults"]

# Generic (entity/event) artifact schema. The pipeline writes generic-named
# artifacts as the canonical output and keeps the legacy AOTY-flavored names as
# byte-identical copies for one release (dual-write, then deprecate). These maps
# translate the internal legacy column/scenario names to the generic ones.
_ENTITY_COLUMN_RENAME = {"artist": "entity", "n_training_albums": "n_training_events"}
_ENTITY_SCENARIO_RENAME = {"artist_mean": "entity_mean"}


def _extract_posterior_samples(idata: object) -> dict[str, jnp.ndarray]:
    """Backward-compatible wrapper for public posterior extraction helper."""
    return extract_posterior_samples(idata)


def _predict_known_artists(
    posterior_samples: dict[str, jnp.ndarray],
    summary: dict,
    last_album_info: pd.DataFrame,
    artist_mean_features: pd.DataFrame,
    seed: int = 42,
    strict: bool = False,
    batch_size: int = 500,
    artist_batch_size: int = 50,
) -> pd.DataFrame:
    """Generate next-album predictions for all known artists under 3 scenarios.

    Scenarios:
    - "same": Use the artist's last album's feature values
    - "population_mean": Use population mean features (zeros after z-scoring)
    - "artist_mean": Use the artist's mean feature values

    Args:
        posterior_samples: Flattened posterior samples dict.
        summary: Training summary dict.
        last_album_info: DataFrame with last album info per artist.
        artist_mean_features: DataFrame with mean feature values per artist.

    Returns:
        DataFrame with columns: artist, scenario, pred_mean, pred_std,
        pred_q05, pred_q25, pred_q50, pred_q75, pred_q95,
        last_score, n_training_albums.
    """
    artist_to_idx = summary["artist_to_idx"]
    max_seq = summary["max_seq"]
    min_albums_filter = int(summary.get("min_albums_filter", 2))
    feature_cols = summary["feature_cols"]
    ds_block = summary.get("dataset") or {}
    target_col = ds_block.get("target_col", "User_Score")
    prefix = ds_block.get("model_prefix", "user")
    target_bounds = tuple(ds_block.get("target_bounds", (0.0, 100.0)))
    transform = get_transform(
        summary.get("target_transform") or "identity",
        target_bounds=target_bounds,
        offset=float(summary.get("logit_offset") or 0.5),
    )
    ar_center = ar_center_on_model_scale(summary)
    scaler = summary.get("feature_scaler")
    if scaler is None:
        raise ValueError(
            "Training summary missing 'feature_scaler' key. "
            "Re-run the train stage to regenerate training_summary.json."
        )
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)

    n_total_samples = next(iter(posterior_samples.values())).shape[0]

    # Prepare per-artist metadata
    artists = list(artist_to_idx.keys())
    n_artists_total = len(artists)

    horizon_clamped_artists = []
    for artist in artists:
        if artist not in last_album_info.index:
            continue
        if int(last_album_info.loc[artist, "album_seq"]) + 1 > max_seq:
            horizon_clamped_artists.append(artist)

    if horizon_clamped_artists:
        msg = (
            "Next-album prediction requires extrapolation beyond training sequence horizon "
            f"for {len(horizon_clamped_artists)} artists (max_seq={max_seq}). "
            "Increase --max-albums during training or disable --strict for exploratory runs."
        )
        if strict:
            raise ValueError(msg)
        log.warning(
            "predict_horizon_clamped",
            n_artists=len(horizon_clamped_artists),
            max_seq=max_seq,
            artists_sample=horizon_clamped_artists[:5],
        )

    # Precompute standardized scenario feature matrices once. The batch loop
    # previously re-checked column membership and re-standardized the same
    # rows for every artist in every batch and scenario.
    missing_mean_cols = [c for c in feature_cols if c not in artist_mean_features.columns]
    if missing_mean_cols:
        log.warning("artist_mean_missing_features", missing=missing_mean_cols)
        artist_mean_scaled = None
    else:
        artist_mean_scaled = (
            artist_mean_features[feature_cols].astype(np.float32) - X_mean
        ) / X_std
    last_album_scaled = (last_album_info[feature_cols].astype(np.float32) - X_mean) / X_std

    results = []

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Process artists in batches
        for batch_start in range(0, n_artists_total, artist_batch_size):
            batch_end = min(batch_start + artist_batch_size, n_artists_total)
            batch_artists = artists[batch_start:batch_end]

            for scenario in SCENARIOS_KNOWN:
                # Build model_args for this batch of artists (one obs per artist)
                artist_idxs = []
                album_seqs = []
                prev_scores = []
                X_list = []
                n_reviews_list = []
                valid_artists = []
                last_scores = []
                n_training_albums_list = []
                horizon_clamped_flags = []

                for artist in batch_artists:
                    idx = artist_to_idx[artist]

                    if artist not in last_album_info.index:
                        continue

                    info = last_album_info.loc[artist]
                    last_seq = int(info["album_seq"])
                    next_seq = min(last_seq + 1, max_seq)
                    horizon_clamped = (last_seq + 1) > max_seq
                    last_score = float(info[target_col])
                    median_n_reviews = int(info["median_n_reviews"])
                    n_albums = int(info["n_albums"])
                    if n_albums < min_albums_filter:
                        # Match training behavior: artists below threshold use static effect.
                        next_seq = 1

                    artist_idxs.append(idx)
                    album_seqs.append(next_seq)
                    prev_scores.append(last_score)
                    n_reviews_list.append(median_n_reviews)
                    valid_artists.append(artist)
                    last_scores.append(last_score)
                    n_training_albums_list.append(n_albums)
                    horizon_clamped_flags.append(horizon_clamped)

                    # Feature vector depends on scenario
                    if scenario == "same":
                        X_list.append(last_album_scaled.loc[artist].values.astype(np.float32))
                    elif scenario == "population_mean":
                        X_list.append(np.zeros(len(feature_cols), dtype=np.float32))
                    elif scenario == "artist_mean":
                        if artist_mean_scaled is not None and artist in artist_mean_scaled.index:
                            X_list.append(artist_mean_scaled.loc[artist].values.astype(np.float32))
                        else:
                            X_list.append(np.zeros(len(feature_cols), dtype=np.float32))

                if not valid_artists:
                    continue

                artist_idx_arr = np.array(artist_idxs, dtype=np.int32)
                album_seq_arr = np.array(album_seqs, dtype=np.int32)
                prev_score_arr = np.array(prev_scores, dtype=np.float32)
                if transform.name != "identity":
                    prev_score_arr = np.asarray(transform.forward(prev_score_arr), dtype=np.float32)
                X_arr = np.stack(X_list).astype(np.float32)
                n_reviews_arr = np.array(n_reviews_list, dtype=np.int32)

                model_args = {
                    "artist_idx": artist_idx_arr,
                    "album_seq": album_seq_arr,
                    "prev_score": prev_score_arr,
                    "X": X_arr,
                    "y": None,
                    "n_reviews": n_reviews_arr,
                    "n_artists": summary["n_artists"],
                    "max_seq": max_seq,
                    "n_exponent": summary.get("n_exponent", 0.0),
                    "learn_n_exponent": summary.get("learn_n_exponent", False),
                    "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
                    "n_ref": summary.get("n_ref"),
                    "likelihood_df": summary.get("likelihood_df", 4.0),
                    "priors": PriorConfig(**summary["priors"]),
                    "target_bounds": target_bounds,
                    "ar_center": ar_center,
                }

                # Run Predictive in chunks -- create once, replace posterior_samples
                # per batch to preserve function identity and avoid JAX recompilation
                y_chunks: list[np.ndarray] = []
                first_batch_ps = {k: v[:batch_size] for k, v in posterior_samples.items()}
                predictive = Predictive(
                    make_score_model(prefix),
                    posterior_samples=first_batch_ps,
                    batch_ndims=1,
                )
                for start in range(0, n_total_samples, batch_size):
                    end = min(start + batch_size, n_total_samples)
                    batch_ps = {k: v[start:end] for k, v in posterior_samples.items()}
                    predictive.posterior_samples = batch_ps

                    rng_key = random.key(seed + start + batch_start * 1000)
                    preds = predictive(rng_key, **model_args)
                    y_key = next(k for k in preds if k.endswith("_y"))
                    y_chunks.append(np.asarray(preds[y_key]))

                # shape: (n_samples, n_artists_in_batch)
                y_pred = np.concatenate(y_chunks, axis=0)
                if transform.name != "identity":
                    y_pred = np.asarray(transform.inverse(y_pred))

                # Compute summary stats per artist
                for i, artist in enumerate(valid_artists):
                    samples = np.clip(y_pred[:, i], target_bounds[0], target_bounds[1])
                    results.append(
                        {
                            "artist": artist,
                            "scenario": scenario,
                            "pred_mean": float(np.mean(samples)),
                            "pred_std": float(np.std(samples)),
                            "pred_q05": float(np.percentile(samples, 5)),
                            "pred_q25": float(np.percentile(samples, 25)),
                            "pred_q50": float(np.percentile(samples, 50)),
                            "pred_q75": float(np.percentile(samples, 75)),
                            "pred_q95": float(np.percentile(samples, 95)),
                            "last_score": last_scores[i],
                            "n_training_albums": n_training_albums_list[i],
                            "horizon_clamped": horizon_clamped_flags[i],
                        }
                    )

            if batch_end % 200 == 0 or batch_end == n_artists_total:
                log.info(
                    "known_artist_progress",
                    processed=batch_end,
                    total=n_artists_total,
                )

    return pd.DataFrame(results)


def _predict_new_artists(
    posterior_samples: dict[str, jnp.ndarray],
    summary: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate predictions for hypothetical new artists under 2 scenarios.

    Scenarios:
    - "population": Population mean features, median n_reviews
    - "debut_defaults": Population mean features, minimum n_reviews (debut-like)

    Both scenarios use training global-mean `prev_score` as the neutral
    cold-start baseline (matching evaluation protocol and train-time debut
    handling), then differ only by assumed review count.

    Args:
        posterior_samples: Flattened posterior samples dict (numpy-compatible).
        summary: Training summary dict.

    Returns:
        DataFrame with columns: scenario, pred_mean, pred_std,
        pred_q05, pred_q25, pred_q50, pred_q75, pred_q95.
    """
    n_features = len(summary["feature_cols"])
    n_reviews_median = summary["n_reviews_stats"]["median"]
    n_reviews_min = summary["n_reviews_stats"]["min"]
    prev_score_default = float(summary.get("global_mean_score", 0.0))
    ds_block = summary.get("dataset") or {}
    prefix = ds_block.get("model_prefix", "user")
    target_bounds = tuple(ds_block.get("target_bounds", (0.0, 100.0)))
    target_transform = summary.get("target_transform") or "identity"
    logit_offset = float(summary.get("logit_offset") or 0.5)
    if target_transform != "identity":
        transform = get_transform(target_transform, target_bounds, logit_offset)
        prev_score_default = float(np.asarray(transform.forward(prev_score_default)))

    # Determine if model uses heteroscedastic noise
    learn_n_exponent = summary.get("learn_n_exponent", False)
    n_exponent = summary.get("n_exponent", 0.0)
    has_hetero = learn_n_exponent or n_exponent != 0.0

    results = []

    scenarios = [
        ("population", n_reviews_median),
        ("debut_defaults", n_reviews_min),
    ]

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        for scenario_name, n_reviews_val in scenarios:
            X_new = jnp.zeros(n_features)

            kwargs: dict = {
                "posterior_samples": {
                    k: jnp.array(np.asarray(v)) for k, v in posterior_samples.items()
                },
                "X_new": X_new,
                "prev_score": prev_score_default,
                "prefix": f"{prefix}_",
                "seed": seed,
                "target_bounds": target_bounds,
                "likelihood_df": float(summary.get("likelihood_df", 4.0)),
                "target_transform": target_transform,
                "logit_offset": logit_offset,
                "ar_center": ar_center_on_model_scale(summary),
                # Cold-start must use the trained likelihood family, not the
                # studentt default — otherwise a beta/skew/discretized model
                # silently predicts new entities under Student-t.
                "likelihood_family": summary.get("likelihood_family") or "studentt",
                "discretize_observation": bool(summary.get("discretize_observation")),
            }

            if has_hetero:
                kwargs["n_reviews_new"] = jnp.array([n_reviews_val])
                if not learn_n_exponent and n_exponent != 0.0:
                    kwargs["fixed_n_exponent"] = n_exponent

            pred = predict_new_artist(**kwargs)

            y_samples = np.asarray(pred["y"])
            results.append(
                {
                    "scenario": scenario_name,
                    "pred_mean": float(np.mean(y_samples)),
                    "pred_std": float(np.std(y_samples)),
                    "pred_q05": float(np.percentile(y_samples, 5)),
                    "pred_q25": float(np.percentile(y_samples, 25)),
                    "pred_q50": float(np.percentile(y_samples, 50)),
                    "pred_q75": float(np.percentile(y_samples, 75)),
                    "pred_q95": float(np.percentile(y_samples, 95)),
                }
            )

    return pd.DataFrame(results)


def predict_next_albums(ctx: StageContext) -> dict:
    """Generate next-album predictions for known and new artists.

    Known artists get 3 scenarios (same features, population mean, artist mean).
    New artists get 2 scenarios (population, debut defaults).

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with prediction summary and output paths.
    """
    log.info("predict_next_start")
    seed = ctx.seed

    # Load the typed training summary first: it records the dataset the
    # model was trained on, which drives the model key and site prefix.
    model_dir = Path("models")
    summary_path = model_dir / "training_summary.json"
    summary = load_training_summary(summary_path).to_json_dict()
    ds_block = summary.get("dataset") or {}
    entity_col = ds_block.get("entity_col", "Artist")
    n_obs_col = ds_block.get("n_obs_col", "User_Ratings")
    prefix = ds_block.get("model_prefix", "user")
    model_key = f"{prefix}_score"

    manifest = load_manifest(model_dir)

    if manifest is None or model_key not in manifest.current:
        raise ValueError(f"No trained {model_key} model found in models/manifest.json")

    model_filename = manifest.current[model_key]
    model_path = model_dir / model_filename

    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    # Guard: posterior sites must match the expected prefix; a mismatch means
    # the fitted model belongs to a different dataset descriptor.
    site_prefix = f"{prefix}_"
    if not any(str(v).startswith(site_prefix) for v in idata.posterior.data_vars):
        found = sorted(str(v) for v in idata.posterior.data_vars)[:8]
        raise ValueError(
            f"Posterior has no sites with expected prefix '{site_prefix}'. "
            f"Found sites: {found}. Re-run the train stage with the matching "
            "dataset descriptor."
        )

    # Extract posterior samples
    posterior_samples = _extract_posterior_samples(idata)
    n_total_samples = next(iter(posterior_samples.values())).shape[0]
    log.info("posterior_samples_extracted", n_total_samples=n_total_samples)

    # Load training data to get per-artist last album info
    train_df = pd.read_parquet(
        resolve_split_dir(Path("data/splits"), SplitType.WITHIN_ENTITY_TEMPORAL) / "train.parquet"
    )
    train_features = pd.read_parquet("data/features/train_features.parquet")

    # Join on the stable original_row_id key (legacy parquets fall back to
    # positional index alignment inside the helper).
    train_df = join_splits_with_features(train_df, train_features, name="predict_train")

    feature_cols = summary["feature_cols"]
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    # Compute album sequence within artist
    train_df = train_df.copy()
    train_df["album_seq"] = train_df.groupby(entity_col).cumcount() + 1

    # Compute n_reviews column
    if "n_reviews" in train_df.columns:
        n_reviews_col = "n_reviews"
    elif n_obs_col in train_df.columns:
        n_reviews_col = n_obs_col
    else:
        n_reviews_col = None

    # Get last album info per artist (sort by album_seq, take last)
    train_df = train_df.sort_values([entity_col, "album_seq"])
    last_album_info = train_df.groupby(entity_col).last()

    # Add n_albums and median n_reviews per artist
    artist_stats = train_df.groupby(entity_col).agg(
        n_albums=("album_seq", "max"),
    )
    if n_reviews_col:
        artist_n_reviews = train_df.groupby(entity_col)[n_reviews_col].median()
        artist_stats["median_n_reviews"] = artist_n_reviews
    else:
        artist_stats["median_n_reviews"] = summary["n_reviews_stats"]["median"]

    last_album_info = last_album_info.join(artist_stats[["n_albums", "median_n_reviews"]])

    # Compute artist mean features
    artist_mean_features = train_df.groupby(entity_col)[feature_cols].mean()

    # Generate known artist predictions
    log.info("predicting_known_artists", n_artists=len(summary["artist_to_idx"]))
    known_df = _predict_known_artists(
        posterior_samples,
        summary,
        last_album_info,
        artist_mean_features,
        seed=seed,
        strict=ctx.strict,
        batch_size=int(getattr(ctx, "predictive_batch_size", 500)),
        artist_batch_size=int(getattr(ctx, "predict_artist_batch_size", 50)),
    )
    log.info("known_predictions_complete", n_rows=len(known_df))

    # Generate new artist predictions
    log.info("predicting_new_artists")
    new_df = _predict_new_artists(posterior_samples, summary, seed=seed)
    log.info("new_predictions_complete", n_rows=len(new_df))

    # Save outputs
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Canonical generic-named artifacts (entity/event schema).
    known_entity_df = known_df.rename(columns=_ENTITY_COLUMN_RENAME)
    if "scenario" in known_entity_df.columns:
        known_entity_df["scenario"] = known_entity_df["scenario"].replace(
            _ENTITY_SCENARIO_RENAME
        )
    new_entity_df = new_df.rename(columns=_ENTITY_COLUMN_RENAME)
    known_entity_df.to_csv(output_dir / "next_event_known_entities.csv", index=False)
    new_entity_df.to_csv(output_dir / "next_event_new_entity.csv", index=False)

    # Legacy AOTY-named copies, kept byte-identical for one release so existing
    # consumers (publication tables, fan charts) keep working. Deprecate later.
    known_df.to_csv(output_dir / "next_album_known_artists.csv", index=False)
    new_df.to_csv(output_dir / "next_album_new_artist.csv", index=False)

    # Validate prediction bounds — log warnings but do NOT clip (hiding model issues)
    for label, df_check in [("known", known_df), ("new", new_df)]:
        if "pred_mean" in df_check.columns and len(df_check) > 0:
            means = df_check["pred_mean"]
            n_below = int((means < 0).sum())
            n_above = int((means > 100).sum())
            if n_below > 0 or n_above > 0:
                log.warning(
                    "predictions_out_of_bounds",
                    split=label,
                    n_below_zero=n_below,
                    n_above_100=n_above,
                    min_pred=float(means.min()),
                    max_pred=float(means.max()),
                )
        if "pred_q95" in df_check.columns and "pred_q05" in df_check.columns:
            widths = df_check["pred_q95"] - df_check["pred_q05"]
            n_wide = int((widths > 80).sum())
            if n_wide > 0:
                log.warning(
                    "predictions_interval_too_wide",
                    split=label,
                    n_90ci_wider_than_80=n_wide,
                    max_width=float(widths.max()),
                )

    n_horizon_clamped_artists = 0
    if "horizon_clamped" in known_df.columns and len(known_df) > 0:
        n_horizon_clamped_artists = int(
            known_df.loc[known_df["horizon_clamped"], "artist"].nunique()
        )

    # Collect prediction-level stats for monitoring
    pred_stats = {}
    if "pred_mean" in known_df.columns and len(known_df) > 0:
        km = known_df["pred_mean"]
        pred_stats["known"] = {
            "min": float(km.min()),
            "max": float(km.max()),
            "mean": float(km.mean()),
            "n_below_zero": int((km < 0).sum()),
            "n_above_100": int((km > 100).sum()),
        }

    pred_summary = {
        "n_known_artists": len(summary["artist_to_idx"]),
        "scenarios_known": SCENARIOS_KNOWN,
        "scenarios_new": SCENARIOS_NEW,
        "n_posterior_samples": int(n_total_samples),
        "batch_size": 500,
        "n_horizon_clamped_artists": n_horizon_clamped_artists,
        "prediction_stats": pred_stats,
    }
    with open(output_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(pred_summary, f, indent=2)

    log.info(
        "predict_next_complete",
        known_artists=len(summary["artist_to_idx"]),
        known_rows=len(known_df),
        new_rows=len(new_df),
    )

    return {
        # Canonical generic-named artifacts.
        "known_predictions_path": str(output_dir / "next_event_known_entities.csv"),
        "new_predictions_path": str(output_dir / "next_event_new_entity.csv"),
        # Legacy AOTY-named copies (dual-written for one release).
        "known_predictions_legacy_path": str(output_dir / "next_album_known_artists.csv"),
        "new_predictions_legacy_path": str(output_dir / "next_album_new_artist.csv"),
        "summary_path": str(output_dir / "prediction_summary.json"),
        "pred_summary": pred_summary,
    }


def predict_artist_next(
    artist: str,
    seed: int = 42,
    batch_size: int = 500,
    models_dir: str | Path = "models",
    splits_path: str | Path | None = None,
    features_path: str | Path = "data/features/train_features.parquet",
) -> pd.DataFrame:
    """Next-album predictions for one known artist (library convenience).

    Wraps the predict stage's known-artist path for a single entity: loads
    the current model and training summary, rebuilds the artist's last-album
    and mean-feature metadata from the training split, and returns the
    per-scenario prediction rows (same columns as
    outputs/predictions/next_album_known_artists.csv).

    Args:
        artist: Entity name exactly as it appears in the training data.
        seed: Predictive rng seed.
        batch_size: Posterior-draw batch size for the predictive.
        models_dir: Directory holding manifest.json / training_summary.json.
        splits_path: Training split parquet. Defaults to the alias-resolved
            within-entity-temporal train split (canonical or legacy directory).
        features_path: Training features parquet.

    Returns:
        DataFrame with one row per scenario for the requested artist.

    Raises:
        KeyError: If the artist was not part of the trained model.
    """
    if splits_path is None:
        splits_path = (
            resolve_split_dir(Path("data/splits"), SplitType.WITHIN_ENTITY_TEMPORAL)
            / "train.parquet"
        )
    model_dir = Path(models_dir)
    summary = load_training_summary(model_dir / "training_summary.json").to_json_dict()
    ds_block = summary.get("dataset") or {}
    entity_col = ds_block.get("entity_col", "Artist")
    n_obs_col = ds_block.get("n_obs_col", "User_Ratings")
    prefix = ds_block.get("model_prefix", "user")

    artist_to_idx = summary["artist_to_idx"]
    if artist not in artist_to_idx:
        raise KeyError(
            f"{entity_col} {artist!r} is not part of the trained model "
            f"({len(artist_to_idx)} known entities). Use predict_new_artist "
            "for cold-start predictions."
        )

    manifest = load_manifest(model_dir)
    model_key = f"{prefix}_score"
    if manifest is None or model_key not in manifest.current:
        raise ValueError(f"No trained {model_key} model found in {model_dir}/manifest.json")
    idata = load_model(model_dir / manifest.current[model_key])
    posterior_samples = _extract_posterior_samples(idata)

    train_df = pd.read_parquet(splits_path)
    train_features = pd.read_parquet(features_path)
    train_df = join_splits_with_features(train_df, train_features, name="predict_one_train")
    train_df = train_df[train_df[entity_col] == artist].copy()
    if train_df.empty:
        raise KeyError(
            f"{entity_col} {artist!r} has no rows in the training split at {splits_path}."
        )

    feature_cols = summary["feature_cols"]
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    train_df["album_seq"] = train_df.groupby(entity_col).cumcount() + 1
    train_df = train_df.sort_values([entity_col, "album_seq"])

    last_album_info = train_df.groupby(entity_col).last()
    artist_stats = train_df.groupby(entity_col).agg(n_albums=("album_seq", "max"))
    if "n_reviews" in train_df.columns:
        artist_stats["median_n_reviews"] = train_df.groupby(entity_col)["n_reviews"].median()
    elif n_obs_col in train_df.columns:
        artist_stats["median_n_reviews"] = train_df.groupby(entity_col)[n_obs_col].median()
    else:
        artist_stats["median_n_reviews"] = summary["n_reviews_stats"]["median"]
    last_album_info = last_album_info.join(artist_stats[["n_albums", "median_n_reviews"]])
    artist_mean_features = train_df.groupby(entity_col)[feature_cols].mean()

    summary_single = {**summary, "artist_to_idx": {artist: artist_to_idx[artist]}}
    return _predict_known_artists(
        posterior_samples,
        summary_single,
        last_album_info,
        artist_mean_features,
        seed=seed,
        strict=False,
        batch_size=batch_size,
        artist_batch_size=1,
    )
