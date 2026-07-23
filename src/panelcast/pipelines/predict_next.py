"""Next-event prediction pipeline.

Generates predictions for:
- Known entities (3 scenarios): next event using trained entity effects
- New/hypothetical entities (2 scenarios): using population distribution
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from panelcast.data.chronology import normalize_chronology
from panelcast.data.imputation import apply_imputation
from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.predict import extract_posterior_samples, predict_new_entity
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.transforms import get_transform
from panelcast.paths import ArtifactPaths
from panelcast.pipelines.training_summary import (
    ar_center_on_model_scale,
    load_training_summary,
)

if TYPE_CHECKING:
    from panelcast.pipelines.stages import StageContext

log = structlog.get_logger()

# Scenario names
SCENARIOS_KNOWN = ["same", "population_mean", "entity_mean"]
SCENARIOS_NEW = ["population", "debut_defaults"]


def _extract_posterior_samples(idata: object) -> dict[str, jnp.ndarray]:
    """Backward-compatible wrapper for public posterior extraction helper."""
    return extract_posterior_samples(idata)


@dataclass(frozen=True)
class _BatchScenarioArrays:
    """Per-batch, per-scenario model inputs plus the bookkeeping the caller needs
    to summarize each artist's predictive draws."""

    artist_idx: np.ndarray
    album_seq: np.ndarray
    prev_score: np.ndarray
    X: np.ndarray
    n_reviews: np.ndarray
    prev_meas_sigma: np.ndarray | None
    valid_artists: list[str]
    last_scores: list[float]
    n_training_events: list[int]
    horizon_clamped_flags: list[bool]


def _scenario_feature_vector(
    scenario: str,
    artist: str,
    *,
    last_album_scaled: pd.DataFrame,
    artist_mean_scaled: pd.DataFrame | None,
    n_features: int,
) -> np.ndarray:
    """Standardized feature vector for one artist under one known-entity scenario.

    "same" uses the entity's last-event features; "population_mean" is the
    z-scored origin; "entity_mean" uses the entity's mean features when available
    and otherwise falls back to the origin.
    """
    if scenario == "same":
        return last_album_scaled.loc[artist].values.astype(np.float32)
    if scenario == "population_mean":
        return np.zeros(n_features, dtype=np.float32)
    if scenario == "entity_mean":
        if artist_mean_scaled is not None and artist in artist_mean_scaled.index:
            return artist_mean_scaled.loc[artist].values.astype(np.float32)
        return np.zeros(n_features, dtype=np.float32)
    raise ValueError(f"Unknown known-entity scenario: {scenario!r}")


def _build_batch_scenario_args(
    batch_artists: list[str],
    scenario: str,
    *,
    artist_to_idx: dict,
    last_album_info: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    last_album_scaled: pd.DataFrame,
    artist_mean_scaled: pd.DataFrame | None,
    transform,
    propagate_rw: bool,
    max_seq: int,
    min_albums_filter: int,
    eiv_on: bool,
    global_std: float,
) -> _BatchScenarioArrays | None:
    """Accumulate one batch's per-artist model inputs under one scenario.

    Returns None when no artist in the batch has a last-event row to predict
    from. Mirrors the legacy inline loop exactly: the same column reads and dtype
    casts, the sub-threshold static-effect ``next_seq = 1`` override, the prev
    score target-transform, and (with errors-in-variables on) the data-derived
    prev_meas_sigma with zero/non-finite review counts pinned to 0.
    """
    artist_idxs = []
    album_seqs = []
    prev_scores = []
    prev_nrevs = []
    X_list = []
    n_reviews_list = []
    valid_artists = []
    last_scores = []
    n_training_events_list = []
    horizon_clamped_flags = []

    for artist in batch_artists:
        idx = artist_to_idx[artist]
        if artist not in last_album_info.index:
            continue

        info = last_album_info.loc[artist]
        last_seq = int(info["album_seq"])
        last_score = float(info[target_col])
        median_n_reviews = int(info["median_n_reviews"])
        n_albums = int(info["n_albums"])
        below_threshold = n_albums < min_albums_filter
        next_seq = (last_seq + 1) if propagate_rw else min(last_seq + 1, max_seq)
        if below_threshold:
            # Match training behavior: artists below threshold use static effect.
            next_seq = 1
        # Sub-threshold entities are pinned to seq 1, so they never extrapolate
        # past the horizon and must not count as clamped.
        horizon_clamped = (not propagate_rw) and not below_threshold and (last_seq + 1) > max_seq

        artist_idxs.append(idx)
        album_seqs.append(next_seq)
        prev_scores.append(last_score)
        # Measurement error of prev_score uses the last album's own review count;
        # debut-free here (every known entity has >=1).
        prev_nrevs.append(
            float(info["n_reviews"])
            if "n_reviews" in last_album_info.columns
            else float(median_n_reviews)
        )
        n_reviews_list.append(median_n_reviews)
        valid_artists.append(artist)
        last_scores.append(last_score)
        n_training_events_list.append(n_albums)
        horizon_clamped_flags.append(horizon_clamped)

        X_list.append(
            _scenario_feature_vector(
                scenario,
                artist,
                last_album_scaled=last_album_scaled,
                artist_mean_scaled=artist_mean_scaled,
                n_features=len(feature_cols),
            )
        )

    if not valid_artists:
        return None

    prev_score_arr = np.array(prev_scores, dtype=np.float32)
    if transform.name != "identity":
        prev_score_arr = np.asarray(transform.forward(prev_score_arr), dtype=np.float32)

    prev_meas_sigma = None
    if eiv_on:
        prev_nrev_arr = np.array(prev_nrevs, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            prev_meas_sigma = np.where(
                ~np.isfinite(prev_nrev_arr) | (prev_nrev_arr <= 0),
                0.0,
                global_std / np.sqrt(np.maximum(prev_nrev_arr, 1.0)),
            ).astype(np.float32)

    return _BatchScenarioArrays(
        artist_idx=np.array(artist_idxs, dtype=np.int32),
        album_seq=np.array(album_seqs, dtype=np.int32),
        prev_score=prev_score_arr,
        X=np.stack(X_list).astype(np.float32),
        n_reviews=np.array(n_reviews_list, dtype=np.int32),
        prev_meas_sigma=prev_meas_sigma,
        valid_artists=valid_artists,
        last_scores=last_scores,
        n_training_events=n_training_events_list,
        horizon_clamped_flags=horizon_clamped_flags,
    )


def _group_pooling_args(priors_obj: PriorConfig, summary: dict) -> dict:
    """Group-pooling model args from the training summary, or empty when off.

    The gated model reads ``group_offset[group_idx_by_artist]`` at both train and
    predict time; evaluate.py resolves these the same way.
    """
    if not priors_obj.entity_group_pooling:
        return {}
    group_idx_by_artist = summary.get("group_idx_by_artist")
    n_groups = summary.get("n_groups")
    if group_idx_by_artist is None or n_groups is None:
        raise ValueError(
            "entity_group_pooling is on but the training summary lacks "
            "group_idx_by_artist/n_groups — re-run the train stage."
        )
    return {
        "group_idx_by_artist": np.asarray(group_idx_by_artist, dtype=np.int32),
        "n_groups": int(n_groups),
    }


def _predict_known_entities(
    posterior_samples: dict[str, jnp.ndarray],
    summary: dict,
    last_album_info: pd.DataFrame,
    artist_mean_features: pd.DataFrame,
    seed: int = 42,
    strict: bool = False,
    batch_size: int = 500,
    artist_batch_size: int = 50,
    conformal_levels: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Generate next-event predictions for all known entities under 3 scenarios.

    Scenarios:
    - "same": Use the entity's last event's feature values
    - "population_mean": Use population mean features (zeros after z-scoring)
    - "entity_mean": Use the entity's mean feature values

    Args:
        posterior_samples: Flattened posterior samples dict.
        summary: Training summary dict.
        last_album_info: DataFrame with last event info per entity.
        artist_mean_features: DataFrame with mean feature values per entity.

    Returns:
        DataFrame with columns: entity, scenario, pred_mean, pred_std,
        pred_q05, pred_q25, pred_q50, pred_q75, pred_q95,
        last_score, n_training_events.
    """
    artist_to_idx = summary["artist_to_idx"]
    max_seq = summary["max_seq"]
    min_albums_filter = int(summary.get("min_albums_filter", 2))
    priors_obj = PriorConfig(**summary["priors"])
    propagate_rw = priors_obj.propagate_rw_horizon
    eiv_on = priors_obj.errors_in_variables
    global_std = float(summary.get("global_std_score") or 0.0)
    if eiv_on and global_std <= 0.0:
        log.warning("eiv_sigma_zero_legacy_summary", context="predict_next")
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
        info = last_album_info.loc[artist]
        if int(info["n_albums"]) < min_albums_filter:
            continue  # below threshold -> static effect (seq 1), never extrapolates
        if int(info["album_seq"]) + 1 > max_seq:
            horizon_clamped_artists.append(artist)

    # propagate_rw_horizon grows the trajectory to cover the deepest next_seq so
    # re-sampled rw_raw accumulates the full innovations (no clamp); otherwise
    # the legacy clamp at max_seq applies and deep extrapolation is flagged.
    if propagate_rw:
        deepest_next_seq = max_seq
        for artist in horizon_clamped_artists:
            deepest_next_seq = max(
                deepest_next_seq, int(last_album_info.loc[artist, "album_seq"]) + 1
            )
        model_max_seq = deepest_next_seq
        if horizon_clamped_artists:
            log.info(
                "predict_horizon_propagated",
                n_artists=len(horizon_clamped_artists),
                max_seq_train=max_seq,
                model_max_seq=model_max_seq,
            )
    else:
        model_max_seq = max_seq
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

    # Genre/group pooling: the model needs the per-entity group indices at
    # predict time too (same array the train stage saved and evaluate.py reads).
    group_pooling_args = _group_pooling_args(priors_obj, summary)

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Process artists in batches
        for batch_start in range(0, n_artists_total, artist_batch_size):
            batch_end = min(batch_start + artist_batch_size, n_artists_total)
            batch_artists = artists[batch_start:batch_end]

            for scenario in SCENARIOS_KNOWN:
                built = _build_batch_scenario_args(
                    batch_artists,
                    scenario,
                    artist_to_idx=artist_to_idx,
                    last_album_info=last_album_info,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    last_album_scaled=last_album_scaled,
                    artist_mean_scaled=artist_mean_scaled,
                    transform=transform,
                    propagate_rw=propagate_rw,
                    max_seq=max_seq,
                    min_albums_filter=min_albums_filter,
                    eiv_on=eiv_on,
                    global_std=global_std,
                )
                if built is None:
                    continue

                model_args = {
                    "artist_idx": built.artist_idx,
                    "album_seq": built.album_seq,
                    "prev_score": built.prev_score,
                    "X": built.X,
                    "y": None,
                    "n_reviews": built.n_reviews,
                    "n_artists": summary["n_artists"],
                    "max_seq": model_max_seq,
                    "n_exponent": summary.get("n_exponent", 0.0),
                    "learn_n_exponent": summary.get("learn_n_exponent", False),
                    "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
                    "n_ref": summary.get("n_ref"),
                    "likelihood_df": summary.get("likelihood_df", 4.0),
                    "priors": priors_obj,
                    "target_bounds": target_bounds,
                    "ar_center": ar_center,
                }
                if eiv_on:
                    model_args["prev_meas_sigma"] = built.prev_meas_sigma
                model_args.update(group_pooling_args)

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
                for i, artist in enumerate(built.valid_artists):
                    samples = np.clip(y_pred[:, i], target_bounds[0], target_bounds[1])
                    row = {
                        "entity": artist,
                        "scenario": scenario,
                        "pred_mean": float(np.mean(samples)),
                        "pred_std": float(np.std(samples)),
                        "pred_q05": float(np.percentile(samples, 5)),
                        "pred_q25": float(np.percentile(samples, 25)),
                        "pred_q50": float(np.percentile(samples, 50)),
                        "pred_q75": float(np.percentile(samples, 75)),
                        "pred_q95": float(np.percentile(samples, 95)),
                        "last_score": built.last_scores[i],
                        "n_training_events": built.n_training_events[i],
                        "horizon_clamped": built.horizon_clamped_flags[i],
                    }
                    if conformal_levels is not None:
                        row["conformal_q05"] = float(
                            np.quantile(samples, conformal_levels[0])
                        )
                        row["conformal_q95"] = float(
                            np.quantile(samples, conformal_levels[1])
                        )
                    results.append(row)

            if batch_end % 200 == 0 or batch_end == n_artists_total:
                log.info(
                    "known_artist_progress",
                    processed=batch_end,
                    total=n_artists_total,
                )

    return pd.DataFrame(results)


def _predict_new_entities(
    posterior_samples: dict[str, jnp.ndarray],
    summary: dict,
    seed: int = 42,
    conformal_levels: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Generate predictions for hypothetical new entities under 2 scenarios.

    Scenarios:
    - "population": Population mean features, median n_reviews
    - "debut_defaults": Population mean features, minimum n_reviews (debut-like)

    Both scenarios use training global-mean `prev_score` as the neutral
    cold-start baseline (matching evaluation protocol and train-time debut
    handling), then differ only by assumed observation count.

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

    priors_obj = PriorConfig(**summary["priors"])

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
                # silently predicts new entities under Student-t. All three from
                # one PriorConfig, mirroring the rollout path.
                "likelihood_family": priors_obj.likelihood_family,
                "skew_tailweight": priors_obj.skew_tailweight,
                "discretize_observation": priors_obj.discretize_observation,
            }

            if has_hetero or priors_obj.likelihood_family == "beta_binomial":
                kwargs["n_reviews_new"] = jnp.array([n_reviews_val])
                if not learn_n_exponent and n_exponent != 0.0:
                    kwargs["fixed_n_exponent"] = n_exponent

            pred = predict_new_entity(**kwargs)

            y_samples = np.asarray(pred["y"])
            row = {
                "scenario": scenario_name,
                "pred_mean": float(np.mean(y_samples)),
                "pred_std": float(np.std(y_samples)),
                "pred_q05": float(np.percentile(y_samples, 5)),
                "pred_q25": float(np.percentile(y_samples, 25)),
                "pred_q50": float(np.percentile(y_samples, 50)),
                "pred_q75": float(np.percentile(y_samples, 75)),
                "pred_q95": float(np.percentile(y_samples, 95)),
            }
            if conformal_levels is not None:
                row["conformal_q05"] = float(np.quantile(y_samples, conformal_levels[0]))
                row["conformal_q95"] = float(np.quantile(y_samples, conformal_levels[1]))
            results.append(row)

    return pd.DataFrame(results)


def _load_conformal_levels(
    evaluation_dir: Path, lo: float = 0.05, hi: float = 0.95
) -> tuple[float, float] | None:
    """Recalibrated (lo, hi) quantile levels from the evaluate stage's conformal block.

    The evaluate stage persists the calibration PIT quantile grid in
    metrics.json (#156); interpolating it remaps any nominal level without
    touching the samples. None (with a warning) when the block is absent —
    the flag is on but evaluate ran without it or without a validation split.
    """
    metrics_path = evaluation_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        grid = metrics["calibration"]["conformal"]["pit_quantile_grid"]
        levels = np.asarray(grid["levels"], dtype=float)
        values = np.asarray(grid["values"], dtype=float)
        return float(np.interp(lo, levels, values)), float(np.interp(hi, levels, values))
    except (OSError, ValueError, KeyError, TypeError):
        log.warning(
            "conformal_levels_unavailable",
            path=str(metrics_path),
            hint="run evaluate with conformal_calibration and val_albums >= 1 first",
        )
        return None


def _normalize_training_chronology(
    frame: pd.DataFrame, entity_col: str, dataset: dict
) -> pd.DataFrame:
    date_candidates = (
        dataset.get("parsed_date_col"),
        dataset.get("date_col"),
        "Release_Date_Parsed",
        "Release_Date",
    )
    date_col = next((column for column in date_candidates if column in frame.columns), None)
    if date_col is None:
        return frame
    event_col = dataset.get("event_col", "Album")
    return normalize_chronology(
        frame, entity_col=entity_col, date_col=date_col, event_col=event_col
    )


def predict_next_events(ctx: StageContext) -> dict:
    """Generate next-event predictions for known and new entities.

    Known entities get 3 scenarios (same features, population mean, entity mean).
    New entities get 2 scenarios (population, debut defaults).

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with prediction summary and output paths.
    """
    log.info("predict_next_start")
    seed = ctx.seed

    # Load the typed training summary first: it records the dataset the
    # model was trained on, which drives the model key and site prefix.
    # Roots come from ctx.paths but are rebuilt through the module-local Path
    # (with the original string form) so test patches keep applying.
    paths = ArtifactPaths.from_ctx(ctx)
    model_dir = Path(paths.models)
    summary_path = model_dir / "training_summary.json"
    summary = load_training_summary(summary_path).to_json_dict()
    ds_block = summary.get("dataset") or {}
    entity_col = ds_block.get("entity_col", "Artist")
    n_obs_col = ds_block.get("n_obs_col", "User_Ratings")
    prefix = ds_block.get("model_prefix", "user")
    model_key = f"{prefix}_score"

    manifest = load_manifest(model_dir)

    if manifest is None or model_key not in manifest.current:
        raise ValueError(
            f"No trained {model_key} model found in {model_dir / 'manifest.json'}. "
            "Run `panelcast stage train` first."
        )

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
        resolve_split_dir(Path(paths.splits), SplitType.WITHIN_ENTITY_TEMPORAL) / "train.parquet"
    )
    train_features = pd.read_parquet(paths.features / "train_features.parquet")

    # Join on the stable original_row_id key (legacy parquets fall back to
    # positional index alignment inside the helper).
    train_df = join_splits_with_features(train_df, train_features, name="predict_train")
    train_df = _normalize_training_chronology(train_df, entity_col, ds_block)

    feature_cols = summary["feature_cols"]
    train_df = apply_imputation(
        train_df, feature_cols, (summary.get("feature_scaler") or {}).get("imputation")
    )

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

    conformal_levels = None
    if getattr(ctx, "conformal_calibration", False):
        conformal_levels = _load_conformal_levels(Path(paths.evaluation))

    # Generate known entity predictions
    log.info("predicting_known_artists", n_artists=len(summary["artist_to_idx"]))
    known_df = _predict_known_entities(
        posterior_samples,
        summary,
        last_album_info,
        artist_mean_features,
        seed=seed,
        strict=ctx.strict,
        batch_size=int(getattr(ctx, "predictive_batch_size", 500)),
        artist_batch_size=int(getattr(ctx, "predict_artist_batch_size", 50)),
        conformal_levels=conformal_levels,
    )
    log.info("known_predictions_complete", n_rows=len(known_df))

    # Generate new entity predictions
    log.info("predicting_new_artists")
    new_df = _predict_new_entities(
        posterior_samples, summary, seed=seed, conformal_levels=conformal_levels
    )
    log.info("new_predictions_complete", n_rows=len(new_df))

    # Save outputs
    output_dir = Path(paths.predictions.as_posix())
    output_dir.mkdir(parents=True, exist_ok=True)

    known_df.to_csv(output_dir / "next_event_known_entities.csv", index=False)
    new_df.to_csv(output_dir / "next_event_new_entity.csv", index=False)

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
            known_df.loc[known_df["horizon_clamped"], "entity"].nunique()
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
        "known_predictions_path": str(output_dir / "next_event_known_entities.csv"),
        "new_predictions_path": str(output_dir / "next_event_new_entity.csv"),
        "summary_path": str(output_dir / "prediction_summary.json"),
        "pred_summary": pred_summary,
    }


def predict_entity_next(
    entity: str,
    seed: int = 42,
    batch_size: int = 500,
    models_dir: str | Path = "models",
    splits_path: str | Path | None = None,
    features_path: str | Path = "data/features/train_features.parquet",
) -> pd.DataFrame:
    """Next-event predictions for one known entity (library convenience).

    Wraps the predict stage's known-entity path for a single entity: loads
    the current model and training summary, rebuilds the entity's last-event
    and mean-feature metadata from the training split, and returns the
    per-scenario prediction rows (same columns as
    outputs/predictions/next_event_known_entities.csv).

    Args:
        entity: Entity name exactly as it appears in the training data.
        seed: Predictive rng seed.
        batch_size: Posterior-draw batch size for the predictive.
        models_dir: Directory holding manifest.json / training_summary.json.
        splits_path: Training split parquet. Defaults to the alias-resolved
            within-entity-temporal train split (canonical or legacy directory).
        features_path: Training features parquet.

    Returns:
        DataFrame with one row per scenario for the requested entity.

    Raises:
        KeyError: If the entity was not part of the trained model.
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
    if entity not in artist_to_idx:
        raise KeyError(
            f"{entity_col} {entity!r} is not part of the trained model "
            f"({len(artist_to_idx)} known entities). Use predict_new_entity "
            "for cold-start predictions."
        )

    manifest = load_manifest(model_dir)
    model_key = f"{prefix}_score"
    if manifest is None or model_key not in manifest.current:
        raise ValueError(
            f"No trained {model_key} model found in {model_dir / 'manifest.json'}. "
            "Run `panelcast stage train` first."
        )
    idata = load_model(model_dir / manifest.current[model_key])
    posterior_samples = _extract_posterior_samples(idata)

    train_df = pd.read_parquet(splits_path)
    train_features = pd.read_parquet(features_path)
    train_df = join_splits_with_features(train_df, train_features, name="predict_one_train")
    train_df = _normalize_training_chronology(train_df, entity_col, ds_block)
    train_df = train_df[train_df[entity_col] == entity].copy()
    if train_df.empty:
        raise KeyError(
            f"{entity_col} {entity!r} has no rows in the training split at {splits_path}."
        )

    feature_cols = summary["feature_cols"]
    train_df = apply_imputation(
        train_df, feature_cols, (summary.get("feature_scaler") or {}).get("imputation")
    )
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

    summary_single = {**summary, "artist_to_idx": {entity: artist_to_idx[entity]}}
    return _predict_known_entities(
        posterior_samples,
        summary_single,
        last_album_info,
        artist_mean_features,
        seed=seed,
        strict=False,
        batch_size=batch_size,
        artist_batch_size=1,
    )
