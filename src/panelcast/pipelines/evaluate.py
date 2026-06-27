"""Model evaluation pipeline.

Runs split-aware evaluation and diagnostics, generating JSON artifacts
for publication and dashboard workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import structlog
import xarray as xr
from jax import random
from numpyro.infer import Predictive, log_likelihood

from panelcast.data.alignment import join_splits_with_features
from panelcast.data.split_types import (
    SplitType,
    legacy_split_name,
    resolve_split_dir,
    resolve_split_type,
)
from panelcast.evaluation.calibration import (
    compute_coverage,
    compute_interval_score,
    compute_pit_values,
    compute_reliability_data,
    compute_weighted_interval_score,
)
from panelcast.evaluation.metrics import compute_crps, compute_point_metrics
from panelcast.evaluation.ppc import compute_ppc_statistics
from panelcast.models.bayes.diagnostics import (
    check_convergence,
    compute_residual_autocorrelation,
    get_divergence_info,
)
from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.predict import extract_posterior_samples, predict_new_entity
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.train_bayes import _apply_max_albums_cap
from panelcast.pipelines.training_summary import (
    ar_center_on_model_scale,
    load_training_summary,
)

if TYPE_CHECKING:
    from panelcast.pipelines.stages import StageContext

log = structlog.get_logger()

PRIMARY_SPLIT = str(SplitType.WITHIN_ENTITY_TEMPORAL.value)
SECONDARY_SPLIT = str(SplitType.ENTITY_DISJOINT.value)


def _json_safe(value: Any) -> Any:
    """Convert payloads to strict-JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.tolist())
        except TypeError:
            pass
    return value


def _write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    """Write strict JSON, replacing NaN/inf values with null."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=indent, allow_nan=False)


def _extract_posterior_samples(idata: object) -> dict[str, jnp.ndarray]:
    """Backward-compatible wrapper for posterior extraction helper."""
    return extract_posterior_samples(idata)


def _summary_dataset(summary: dict) -> dict:
    """Domain names recorded at train time (AOTY defaults for legacy summaries)."""
    block = summary.get("dataset") or {}
    return {
        "entity_col": block.get("entity_col", "Artist"),
        "target_col": block.get("target_col", "User_Score"),
        "n_obs_col": block.get("n_obs_col", "User_Ratings"),
        "prefix": block.get("model_prefix", "user"),
        "target_bounds": tuple(block.get("target_bounds", (0.0, 100.0))),
    }


def _transform_from_summary(summary: dict):
    """Resolve the target transform the model was trained under."""
    ds = _summary_dataset(summary)
    return get_transform(
        summary.get("target_transform") or "identity",
        target_bounds=ds["target_bounds"],
        offset=float(summary.get("logit_offset") or 0.5),
    )


def _ar_center_from_summary(summary: dict) -> float:
    """Model-scale AR(1) center the model was trained with."""
    return ar_center_on_model_scale(summary)


_HISTORY_BINS: tuple[tuple[int, int | None], ...] = ((1, 2), (3, 5), (6, 10), (11, None))


def _stratify_metrics_by_history(
    artist_idx: np.ndarray,
    train_df: pd.DataFrame,
    summary: dict,
    y_true: np.ndarray,
    y_samples: np.ndarray,
    interval: float,
) -> list[dict]:
    """Per-history-bin accuracy/coverage/interval-width diagnostics.

    Rows are binned by the artist's TRAINING album count (the information the
    model had), using the same entity->index mapping as the model args so the
    bins line up with the (internally sorted) prediction rows.
    """
    ds = _summary_dataset(summary)
    entity_col = ds["entity_col"]
    artist_to_idx = summary["artist_to_idx"]

    counts_by_idx = np.zeros(int(summary["n_artists"]), dtype=np.int64)
    for entity, count in train_df.groupby(entity_col).size().items():
        idx = artist_to_idx.get(entity)
        if idx is not None:
            counts_by_idx[int(idx)] = int(count)
    row_counts = counts_by_idx[np.asarray(artist_idx, dtype=np.int64)]

    lo_q = 100.0 * (1.0 - interval) / 2.0
    hi_q = 100.0 - lo_q
    lo = np.percentile(y_samples, lo_q, axis=0)
    hi = np.percentile(y_samples, hi_q, axis=0)
    pred_mean = y_samples.mean(axis=0)

    rows: list[dict] = []
    for low, high in _HISTORY_BINS:
        upper = np.inf if high is None else high
        mask = (row_counts >= low) & (row_counts <= upper)
        n = int(mask.sum())
        if n == 0:
            continue
        yt = y_true[mask]
        residuals = yt - pred_mean[mask]
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        rows.append(
            {
                "train_albums_bin": f"{low}+" if high is None else f"{low}-{high}",
                "n": n,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
                "coverage": float(np.mean((yt >= lo[mask]) & (yt <= hi[mask]))),
                "mean_interval_width": float(np.mean(hi[mask] - lo[mask])),
                "interval": interval,
            }
        )
    return rows


def _prepare_test_model_args(
    test_df: pd.DataFrame,
    test_features: pd.DataFrame,
    summary: dict,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    strict: bool = False,
) -> tuple[dict, np.ndarray]:
    """Build model_args for test data using training summary metadata."""
    ds = _summary_dataset(summary)
    entity_col = ds["entity_col"]
    target_col = ds["target_col"]
    n_obs_col = ds["n_obs_col"]
    test_df = join_splits_with_features(test_df, test_features, name="primary_test")

    sort_cols = [entity_col]
    if "Release_Date_Parsed" in test_df.columns:
        sort_cols.append("Release_Date_Parsed")
    if "Album" in test_df.columns:
        sort_cols.append("Album")
    test_df = test_df.sort_values(sort_cols, na_position="first").copy()

    artist_to_idx = summary["artist_to_idx"]
    test_df = test_df.copy()
    test_df["_artist_idx"] = test_df[entity_col].map(artist_to_idx)

    unknown_mask = test_df["_artist_idx"].isna()
    if unknown_mask.any():
        unknown_artists = sorted(
            test_df.loc[unknown_mask, entity_col].astype(str).unique().tolist()
        )
        preview = unknown_artists[:5]
        raise ValueError(
            "Unknown artists found in primary split test data. "
            "This indicates train/test mismatch and would invalidate evaluation. "
            f"n_unknown_rows={int(unknown_mask.sum())}, "
            f"unknown_artists_sample={preview}."
        )

    artist_idx_raw = test_df["_artist_idx"].values
    if np.isnan(artist_idx_raw).any():
        raise ValueError("NaN values remain in artist_idx after dropna filtering.")
    artist_idx = artist_idx_raw.astype(np.int32)

    if train_df is not None:
        train_artist_last_seq = (
            train_df.groupby(entity_col).cumcount().groupby(train_df[entity_col]).last() + 1
        )
        train_artist_last_score = train_df.groupby(entity_col)[target_col].last()
    else:
        train_artist_last_seq = pd.Series(dtype=int)
        train_artist_last_score = pd.Series(dtype=float)

    raw_seq = test_df.groupby(entity_col).cumcount() + 1
    train_offset = test_df[entity_col].map(train_artist_last_seq).fillna(0).astype(int)
    album_seq = (raw_seq + train_offset).values

    min_albums_filter = summary.get("min_albums_filter", 2)
    # Apply the same dynamic-effects eligibility rule used in training:
    # threshold is based on training history per artist, not test fold counts.
    if train_df is not None:
        artist_counts_for_filter = train_df.groupby(entity_col).size()
    else:
        artist_counts_for_filter = pd.Series(dtype=int)
    artist_train_counts = test_df[entity_col].map(artist_counts_for_filter).fillna(0)
    below_threshold = (artist_train_counts < min_albums_filter).values
    album_seq = np.where(below_threshold, 1, album_seq)

    max_albums = summary.get("max_albums", 50)
    max_seq_train = summary["max_seq"]
    priors_obj = PriorConfig(**summary["priors"])
    propagate_rw = priors_obj.propagate_rw_horizon
    eiv_on = priors_obj.errors_in_variables

    test_counts = pd.Series(artist_idx).value_counts().sort_index()
    test_counts = test_counts.reindex(range(summary["n_artists"]), fill_value=0)
    if train_df is not None:
        train_idx = train_df[entity_col].map(summary["artist_to_idx"]).dropna().astype(int)
        train_counts = train_idx.value_counts().sort_index()
        train_counts = train_counts.reindex(range(summary["n_artists"]), fill_value=0)
        artist_album_counts = train_counts + test_counts
    else:
        artist_album_counts = test_counts

    temp_args = {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
    }
    temp_args = _apply_max_albums_cap(temp_args, max_albums, artist_album_counts)
    album_seq = temp_args["album_seq"]
    n_horizon_clamped = int(np.sum(album_seq > max_seq_train))
    if propagate_rw:
        # The variant the strict-mode guard advertises: keep the deep-horizon
        # album_seq and grow the trajectory so re-sampled rw_raw accumulates the
        # full innovations (deep intervals widen ~sqrt(h-max_seq)*sigma_rw).
        album_seq = album_seq.astype(np.int32)
        max_seq_eval = int(album_seq.max()) if len(album_seq) else max_seq_train
        if n_horizon_clamped > 0:
            log.info(
                "primary_eval_horizon_propagated",
                n_rows_over_horizon=n_horizon_clamped,
                max_seq_train=max_seq_train,
                max_seq_eval=max_seq_eval,
            )
    else:
        if n_horizon_clamped > 0:
            msg = (
                "Primary split evaluation requires extrapolation beyond training horizon. "
                f"n_rows_over_horizon={n_horizon_clamped}, max_seq_train={max_seq_train}. "
                "Increase --max-albums or use a model variant that explicitly samples future "
                "random-walk innovations (propagate_rw_horizon)."
            )
            if strict:
                raise ValueError(msg)
            log.warning(
                "primary_eval_horizon_clamped",
                n_rows_over_horizon=n_horizon_clamped,
                max_seq_train=max_seq_train,
            )
        album_seq = np.minimum(album_seq, max_seq_train).astype(np.int32)
        max_seq_eval = max_seq_train

    global_mean = summary["global_mean_score"]
    # Sequential prev_score: use the most recent known score before each test album.
    # If a validation album exists between training and test (within-artist temporal
    # split), use the val album's actual score as prev_score — it's an observed value
    # that chronologically precedes the test album.
    if val_df is not None and not val_df.empty:
        val_last_score = val_df.groupby(entity_col)[target_col].last()
        # Use val score where available, fall back to train last
        base_prev = test_df[entity_col].map(val_last_score)
        base_prev = base_prev.fillna(test_df[entity_col].map(train_artist_last_score))
        base_prev = base_prev.fillna(global_mean)
        n_val_used = int(
            base_prev.index.isin(
                test_df[test_df[entity_col].isin(val_last_score.index)].index
            ).sum()
        )
        log.info(
            "primary_prev_score_mode",
            mode="sequential_with_val",
            n_val_used=n_val_used,
            message="Using validation album scores as prev_score where available.",
        )
    else:
        base_prev = test_df[entity_col].map(train_artist_last_score).fillna(global_mean)

    # Errors-in-variables: the de-noised AR regressor needs the review count of
    # the album that SUPPLIED each prev_score (its measurement error). The
    # boundary prev comes from the val/train last album; subsequent test albums
    # carry the preceding test album's count. NaN where prev fell back to the
    # global mean (no prior album) -> pinned to a zero measurement error below.
    nrev_col_test: str | None = None
    if eiv_on:
        if "n_reviews" in test_df.columns:
            nrev_col_test = "n_reviews"
        elif n_obs_col in test_df.columns:
            nrev_col_test = n_obs_col

        def _last_nrev(df: pd.DataFrame | None) -> pd.Series:
            if df is None or df.empty:
                return pd.Series(dtype=float)
            col = "n_reviews" if "n_reviews" in df.columns else n_obs_col
            if col not in df.columns:
                return pd.Series(dtype=float)
            return df.groupby(entity_col)[col].last()

        base_prev_nrev = test_df[entity_col].map(_last_nrev(val_df))
        base_prev_nrev = base_prev_nrev.fillna(test_df[entity_col].map(_last_nrev(train_df)))

    prev_scores: list[float] = []
    prev_nrevs: list[float] = []
    for _, group in test_df.groupby(entity_col, sort=False):
        first_idx = group.index[0]
        group_prev = [float(base_prev.loc[first_idx])]
        if eiv_on:
            group_nrev = [float(base_prev_nrev.loc[first_idx])]
        for j in range(1, len(group)):
            group_prev.append(float(group.iloc[j - 1][target_col]))
            if eiv_on:
                prev_row_nrev = (
                    float(group.iloc[j - 1][nrev_col_test])
                    if nrev_col_test is not None
                    else float("nan")
                )
                group_nrev.append(prev_row_nrev)
        prev_scores.extend(group_prev)
        if eiv_on:
            prev_nrevs.extend(group_nrev)
    test_df["_prev_score"] = prev_scores
    if eiv_on:
        test_df["_prev_nrev"] = prev_nrevs
    n_multi = int((test_df.groupby(entity_col).size() > 1).sum())
    if n_multi > 0:
        log.info(
            "primary_prev_score_sequential",
            n_multi_album_artists=n_multi,
        )
    prev_score = test_df["_prev_score"].values.astype(np.float32)
    # The model consumes prev_score on its training scale.
    transform = _transform_from_summary(summary)
    if transform.name != "identity":
        prev_score = np.asarray(transform.forward(prev_score), dtype=np.float32)

    feature_cols = summary["feature_cols"]
    test_df[feature_cols] = test_df[feature_cols].fillna(0)
    X = test_df[feature_cols].values.astype(np.float32)

    scaler = summary.get("feature_scaler")
    if scaler is None:
        raise ValueError(
            "Training summary missing 'feature_scaler' key. "
            "Re-run the train stage to regenerate training_summary.json."
        )
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)
    X = (X - X_mean) / X_std

    if "n_reviews" in test_features.columns:
        n_reviews_raw = test_df["n_reviews"].values
    elif n_obs_col in test_df.columns:
        n_reviews_raw = test_df[n_obs_col].values
    else:
        raise ValueError(f"No n_reviews or {n_obs_col} column found in test data")

    invalid_mask = pd.isna(n_reviews_raw) | (n_reviews_raw <= 0)
    valid_mask = ~invalid_mask

    if invalid_mask.sum() > 0:
        log.info("test_invalid_n_reviews_dropped", n_dropped=int(invalid_mask.sum()))
        artist_idx = artist_idx[valid_mask]
        album_seq = album_seq[valid_mask]
        prev_score = prev_score[valid_mask]
        X = X[valid_mask]
        n_reviews_raw = n_reviews_raw[valid_mask]
        test_df = test_df[valid_mask].reset_index(drop=True)

    n_reviews = n_reviews_raw.astype(np.int32)
    y_true = test_df[target_col].values.astype(np.float32)

    test_model_args = {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
        "prev_score": prev_score,
        "X": X,
        "y": None,
        "n_reviews": n_reviews,
        "n_artists": summary["n_artists"],
        "max_seq": max_seq_eval,
        "n_exponent": summary.get("n_exponent", 0.0),
        "learn_n_exponent": summary.get("learn_n_exponent", False),
        "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
        "n_ref": summary.get("n_ref"),
        "likelihood_df": summary.get("likelihood_df", 4.0),
        "priors": priors_obj,
        "target_bounds": ds["target_bounds"],
        "ar_center": _ar_center_from_summary(summary),
    }

    if eiv_on:
        global_std = float(summary.get("global_std_score") or 0.0)
        if global_std <= 0.0:
            log.warning("eiv_sigma_zero_legacy_summary", context="primary_eval")
        prev_nrev = test_df["_prev_nrev"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            prev_meas_sigma = np.where(
                np.isnan(prev_nrev) | (prev_nrev <= 0),
                0.0,
                global_std / np.sqrt(np.maximum(prev_nrev, 1.0)),
            ).astype(np.float32)
        test_model_args["prev_meas_sigma"] = prev_meas_sigma

    return test_model_args, y_true


def _resolve_feature_split_dir(split_name: str) -> Path:
    """Resolve feature directory with backward compatibility fallback.

    Prefers the canonical directory, falls back to a legacy-named directory if
    only that exists, and finally to the flat ``data/features`` root for the
    primary split (the legacy layout that mirrored primary features there).
    Unknown split names (not a SplitType) are returned as-is so callers can
    point at arbitrary directories.
    """
    candidate = Path("data/features") / split_name
    if candidate.exists():
        return candidate
    try:
        split_type: SplitType | None = resolve_split_type(split_name)
    except ValueError:
        split_type = None
    if split_type is not None:
        legacy = legacy_split_name(split_type)
        if legacy is not None:
            legacy_path = Path("data/features") / legacy
            if legacy_path.exists():
                return legacy_path
        if split_type is SplitType.WITHIN_ENTITY_TEMPORAL:
            return Path("data/features")
    return candidate


def _run_known_artist_predictive(
    posterior_samples: dict[str, Any],
    model_args: dict[str, Any],
    seed_offset: int = 0,
    prefix: str = "user",
    batch_size: int = 500,
) -> np.ndarray:
    """Run Predictive on known artists using chunked posterior batches."""
    n_total_samples = next(iter(posterior_samples.values())).shape[0]
    y_pred_chunks: list[np.ndarray] = []

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Predictive freezes its batch shape at construction, so a ragged
        # final chunk (n_total % batch_size != 0) needs its own instance;
        # reassigning posterior_samples is only safe between chunks of the
        # same length.
        predictives: dict[int, Predictive] = {}
        for start in range(0, n_total_samples, batch_size):
            end = min(start + batch_size, n_total_samples)
            batch_samples = {k: v[start:end] for k, v in posterior_samples.items()}
            predictive = predictives.get(end - start)
            if predictive is None:
                predictive = Predictive(
                    make_score_model(prefix),
                    posterior_samples=batch_samples,
                    batch_ndims=1,
                )
                predictives[end - start] = predictive
            else:
                predictive.posterior_samples = batch_samples

            rng_key = random.key(seed_offset + start)
            preds = predictive(rng_key, **model_args)
            y_key = next(k for k in preds if k.endswith("_y"))
            y_pred_chunks.append(np.asarray(preds[y_key]))

    return np.concatenate(y_pred_chunks, axis=0)


def _prepare_disjoint_inputs(
    test_df: pd.DataFrame,
    test_features: pd.DataFrame,
    summary: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare inputs for artist-disjoint cold-start evaluation."""
    ds = _summary_dataset(summary)
    # Cold-start protocol: do NOT derive prev_score from held-out labels.
    # Use training global mean as neutral prior for all unseen-artist rows.
    df = join_splits_with_features(test_df, test_features, name="disjoint_test").copy()
    df["_prev_score"] = float(summary["global_mean_score"])

    artist_counts = df.groupby(ds["entity_col"]).size()
    n_multi_album_artists = int((artist_counts > 1).sum())
    if n_multi_album_artists > 0:
        log.info(
            "disjoint_prev_score_cold_start_mode",
            mode="global_mean_for_all_rows",
            n_multi_album_artists=n_multi_album_artists,
        )

    feature_cols = summary["feature_cols"]
    df[feature_cols] = df[feature_cols].fillna(0)
    X = df[feature_cols].values.astype(np.float32)
    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)
    X = (X - X_mean) / X_std

    if "n_reviews" in df.columns:
        n_reviews_raw = df["n_reviews"].values
    elif ds["n_obs_col"] in df.columns:
        n_reviews_raw = df[ds["n_obs_col"]].values
    else:
        raise ValueError(f"No n_reviews or {ds['n_obs_col']} column found in test data")

    invalid_mask = pd.isna(n_reviews_raw) | (n_reviews_raw <= 0)
    if invalid_mask.sum() > 0:
        log.info("disjoint_invalid_n_reviews_dropped", n_dropped=int(invalid_mask.sum()))
        valid_mask = ~invalid_mask
        df = df[valid_mask].reset_index(drop=True)
        X = X[valid_mask]
        n_reviews_raw = n_reviews_raw[valid_mask]

    y_true = df[ds["target_col"]].values.astype(np.float32)
    prev_score = df["_prev_score"].values.astype(np.float32)
    transform = _transform_from_summary(summary)
    if transform.name != "identity":
        prev_score = np.asarray(transform.forward(prev_score), dtype=np.float32)
    n_reviews = n_reviews_raw.astype(np.int32)
    return X, prev_score, n_reviews, y_true


def _run_new_artist_predictive(
    posterior_samples: dict[str, Any],
    summary: dict,
    X: np.ndarray,
    prev_score: np.ndarray,
    n_reviews: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate predictions for unseen artists via population distribution."""
    kwargs: dict[str, Any] = {
        "posterior_samples": {k: jnp.asarray(v) for k, v in posterior_samples.items()},
        "X_new": jnp.asarray(X, dtype=jnp.float32),
        "prev_score": jnp.asarray(prev_score, dtype=jnp.float32),
        "prefix": f"{_summary_dataset(summary)['prefix']}_",
        "seed": seed,
        "target_bounds": _summary_dataset(summary)["target_bounds"],
        "likelihood_df": float(summary.get("likelihood_df", 4.0)),
        "likelihood_family": summary.get("likelihood_family") or "studentt",
        "discretize_observation": bool(summary.get("discretize_observation", False)),
        "target_transform": summary.get("target_transform") or "identity",
        "logit_offset": float(summary.get("logit_offset") or 0.5),
        "ar_center": _ar_center_from_summary(summary),
    }

    learn_n_exponent = bool(summary.get("learn_n_exponent", False))
    fixed_n_exponent = float(summary.get("n_exponent", 0.0))
    if learn_n_exponent or fixed_n_exponent != 0.0:
        kwargs["n_reviews_new"] = jnp.asarray(n_reviews, dtype=jnp.float32)
        if not learn_n_exponent and fixed_n_exponent != 0.0:
            kwargs["fixed_n_exponent"] = fixed_n_exponent

    pred = predict_new_entity(**kwargs)
    y_pred = np.asarray(pred["y"])
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    return y_pred


def _compute_info_criteria(
    posterior_samples: dict[str, Any],
    model_args: dict[str, Any],
    y_true: np.ndarray,
    n_chains: int,
    n_draws: int,
    prefix: str = "user",
    transform: Any = None,
    y_raw: np.ndarray | None = None,
    seed: int = 0,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Compute WAIC/LOO from pointwise log-likelihood.

    The saved posterior deliberately excludes the huge ``{prefix}_rw_raw``
    tensor (and ``{prefix}_entity_obs_raw`` when the entity-overdispersion
    gate is on), so the per-draw career trajectories / per-entity noise
    factors cannot be conditioned on. Missing latent sites are therefore
    marginalized exactly the way the rest of the test-set evaluation does:
    sampled per posterior draw via ``Predictive`` (anchored at the posterior
    init effects and scaled by the posterior sigma_rw / tau_entity), in
    draw-batches to bound device memory. Without
    this, ``log_likelihood`` hits the unseeded sample site and dies with a
    bare AssertionError — the silent "unavailable" failure on both cheap
    validation runs.

    When a target transform is active, the per-observation log-Jacobian
    ``log|dz/dy|`` evaluated at the RAW-scale y is added to the model-scale
    pointwise log-likelihood, putting ELPDs on the score scale so they are
    comparable across transforms (identity adds zeros — numbers unchanged).
    """
    model = make_score_model(prefix)
    args_with_y = dict(model_args)
    args_with_y["y"] = y_true
    args_predictive = dict(model_args)
    args_predictive["y"] = None

    rw_site = f"{prefix}_rw_raw"
    entity_site = f"{prefix}_entity_obs_raw"
    # High-cardinality latents excluded from the saved posterior are
    # marginalized by re-sampling from their (data-independent unit-normal)
    # priors, anchored at the posterior draws of every other site. rw_raw is
    # always excluded; entity_obs_raw exists (and is excluded) only when the
    # entity-overdispersion gate is on -- detected by tau_entity being present.
    gate_on = f"{prefix}_tau_entity" in posterior_samples
    excluded_latents = [s for s in (rw_site,) if s not in posterior_samples]
    if gate_on and entity_site not in posterior_samples:
        excluded_latents.append(entity_site)
    # The errors-in-variables regressor latent has no scalar companion site, so
    # detect it from the priors carried in model_args. When on it is excluded
    # from the saved fit (n_obs cardinality) -> marginalize by re-sampling its
    # unit-normal prior, exactly as for rw_raw.
    eiv_site = f"{prefix}_prev_latent_raw"
    eiv_on = bool(getattr(model_args.get("priors"), "errors_in_variables", False))
    if eiv_on and eiv_site not in posterior_samples:
        excluded_latents.append(eiv_site)
    needs_latents = bool(excluded_latents)
    n_total = int(next(iter(posterior_samples.values())).shape[0])

    log_lik_chunks: list[np.ndarray] = []
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk = {k: v[start:end] for k, v in posterior_samples.items()}
        if needs_latents:
            latent_pred = Predictive(
                model,
                posterior_samples=chunk,
                batch_ndims=1,
                return_sites=excluded_latents,
            )
            latents = latent_pred(random.key(seed + start), **args_predictive)
            chunk = {**chunk, **{s: latents[s] for s in excluded_latents}}
        log_lik_dict = log_likelihood(
            model,
            chunk,
            batch_ndims=1,
            **args_with_y,
        )
        y_key = next((k for k in log_lik_dict if k.endswith("_y")), None)
        if y_key is None:
            raise ValueError("Unable to locate observed site in log_likelihood output.")
        log_lik_chunks.append(np.asarray(log_lik_dict[y_key]))

    log_lik = np.concatenate(log_lik_chunks, axis=0)

    # Score-scale comparability across transforms: add the per-observation
    # log-Jacobian at the raw-scale y (constant across draws).
    if transform is not None and getattr(transform, "name", "identity") != "identity":
        if y_raw is None:
            raise ValueError("y_raw is required to apply the transform Jacobian.")
        jacobian = np.asarray(transform.log_jacobian(y_raw))
        log_lik = log_lik + jacobian[None, :]

    n_samples_total, n_obs = log_lik.shape
    if n_chains * n_draws != n_samples_total:
        n_chains_use = 1
        n_draws_use = n_samples_total
    else:
        n_chains_use = n_chains
        n_draws_use = n_draws

    da = xr.DataArray(
        log_lik.reshape(n_chains_use, n_draws_use, n_obs),
        dims=["chain", "draw", "obs"],
        coords={
            "chain": range(n_chains_use),
            "draw": range(n_draws_use),
            "obs": range(n_obs),
        },
    )
    groups: dict[str, xr.Dataset] = {"log_likelihood": xr.Dataset({"y": da})}
    # az.loo needs posterior draws to estimate relative efficiency (reff);
    # rebuild the (chain, draw, ...) posterior group from the flat samples.
    if n_chains_use * n_draws_use == n_samples_total:
        post_vars: dict[str, xr.DataArray] = {}
        for site, values in posterior_samples.items():
            arr = np.asarray(values)
            site_shape = arr.shape[1:]
            dims = ["chain", "draw"] + [f"{site}_dim_{i}" for i in range(len(site_shape))]
            post_vars[site] = xr.DataArray(
                arr.reshape(n_chains_use, n_draws_use, *site_shape), dims=dims
            )
        groups["posterior"] = xr.Dataset(post_vars)
    idata_ll = az.InferenceData(**groups)
    loo_kwargs: dict[str, Any] = {} if "posterior" in groups else {"reff": 1.0}
    loo = az.loo(idata_ll, var_name="y", pointwise=True, **loo_kwargs)
    waic = az.waic(idata_ll, var_name="y", pointwise=True)

    pareto_k = np.asarray(loo.pareto_k)
    return {
        "loo": {
            "elpd": float(loo.elpd_loo),
            "se": float(loo.se),
            "p": float(loo.p_loo),
            "pareto_k_max": float(np.max(pareto_k)),
            "pareto_k_gt_0_7": int(np.sum(pareto_k > 0.7)),
        },
        "waic": {
            "elpd": float(waic.elpd_waic),
            "se": float(waic.se),
            "p": float(waic.p_waic),
        },
        "scale": "score" if transform is not None else "model",
        "latents_marginalized": bool(needs_latents),
    }


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred_samples: np.ndarray,
    calibration_intervals: tuple[float, ...],
    coverage_tolerance: float,
    prediction_interval: float,
    discretize: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute metrics and export payloads for one split."""
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    point_metrics = compute_point_metrics(y_true, y_pred_mean)
    crps_result = compute_crps(y_true, y_pred_samples)

    coverages: dict[str, Any] = {}
    within_tolerance = True
    for prob in calibration_intervals:
        coverage = compute_coverage(y_true, y_pred_samples, prob=prob)
        delta = abs(float(coverage.empirical) - prob)
        ok = delta <= coverage_tolerance
        within_tolerance = within_tolerance and ok
        key = f"{prob:.2f}"
        coverages[key] = {
            "nominal": prob,
            "empirical": float(coverage.empirical),
            "interval_width": float(coverage.interval_width),
            "abs_error": float(delta),
            "within_tolerance": ok,
        }

    # Interval scores per calibration level
    interval_scores: dict[str, dict[str, float]] = {}
    for prob in calibration_intervals:
        is_result = compute_interval_score(y_true, y_pred_samples, prob=prob)
        key = f"{prob:.2f}"
        interval_scores[key] = {
            "mean_score": float(is_result.mean_score),
            "sharpness": float(is_result.sharpness_component),
            "penalty": float(is_result.calibration_penalty),
        }

    # Weighted interval score (Bracher et al. 2021)
    wis_result = compute_weighted_interval_score(
        y_true, y_pred_samples, probs=calibration_intervals
    )
    wis_value = float(wis_result.wis)

    # Round y_true to match the integer y_rep so the PPC compares on one grid.
    ppc_y_true = np.round(y_true) if discretize else y_true
    ppc_result = compute_ppc_statistics(ppc_y_true, y_pred_samples)
    ppc_payload = {
        "summary": ppc_result.summary,
        "n_samples": ppc_result.n_samples,
        "extreme_statistics": ppc_result.check_extreme(),
    }

    reliability = compute_reliability_data(y_true, y_pred_samples)
    pit = compute_pit_values(y_true, y_pred_samples)

    alpha = (1.0 - prediction_interval) / 2.0
    lower_pct = 100.0 * alpha
    upper_pct = 100.0 * (1.0 - alpha)
    y_pred_lower = np.percentile(y_pred_samples, lower_pct, axis=0)
    y_pred_upper = np.percentile(y_pred_samples, upper_pct, axis=0)

    split_metrics = {
        "point_metrics": point_metrics.to_summary_dict(),
        "calibration": {
            "coverages": coverages,
            "coverage_tolerance": float(coverage_tolerance),
            "within_tolerance": bool(within_tolerance),
            "interval_scores": interval_scores,
            "wis": wis_value,
            "pit": pit,
        },
        "crps": crps_result.to_summary_dict(),
        "ppc": ppc_payload,
        "prediction_interval": {
            "level": float(prediction_interval),
            "lower_percentile": float(lower_pct),
            "upper_percentile": float(upper_pct),
        },
    }

    residuals = (y_true - y_pred_mean).tolist()

    predictions_payload = {
        "y_true": y_true.tolist(),
        "y_pred_mean": y_pred_mean.tolist(),
        "y_pred_lower": y_pred_lower.tolist(),
        "y_pred_upper": y_pred_upper.tolist(),
        "residuals": residuals,
        "interval_level": float(prediction_interval),
    }
    calibration_payload = {
        "predicted_probs": reliability.predicted_probs.tolist(),
        "observed_freq": reliability.observed_freq.tolist(),
        "counts": reliability.counts.tolist(),
        "bin_edges": reliability.bin_edges.tolist(),
    }
    return split_metrics, predictions_payload, calibration_payload


def evaluate_models(ctx: StageContext) -> dict:
    """Evaluate fitted models on primary and secondary test splits."""
    log.info("evaluation_pipeline_start")

    model_dir = Path("models")

    # A missing manifest means no trained model at all -- report that before
    # touching the training summary so the error names the actual gap.
    manifest = load_manifest(model_dir)
    if manifest is None:
        raise ValueError("No trained user_score model found in models/manifest.json")

    # The typed training summary records the dataset the model was trained
    # on, which drives the model key and posterior-site prefix.
    summary_path = model_dir / "training_summary.json"
    summary = load_training_summary(summary_path).to_json_dict()
    ds = _summary_dataset(summary)
    prefix = ds["prefix"]
    model_key = f"{prefix}_score"

    if model_key not in manifest.current:
        raise ValueError(f"No trained {model_key} model found in models/manifest.json")

    model_filename = manifest.current[model_key]
    model_path = model_dir / model_filename
    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    # Guard: the fitted posterior must carry sites for the expected prefix.
    # A mismatch means models/ holds a model trained under a different
    # dataset descriptor — fail with a named error instead of a deep KeyError.
    site_prefix = f"{prefix}_"
    if not any(str(v).startswith(site_prefix) for v in idata.posterior.data_vars):
        found = sorted(str(v) for v in idata.posterior.data_vars)[:8]
        raise ValueError(
            f"Posterior has no sites with expected prefix '{site_prefix}'. "
            f"Found sites: {found}. The fitted model in models/ was trained "
            "with a different dataset descriptor; re-run the train stage."
        )

    diagnostics = check_convergence(
        idata,
        rhat_threshold=float(getattr(ctx, "rhat_threshold", 1.01)),
        ess_threshold=int(getattr(ctx, "ess_threshold", 400)),
        allow_divergences=bool(getattr(ctx, "allow_divergences", False)),
    )
    _ = get_divergence_info(idata)
    diagnostics_result = {
        "passed": diagnostics.passed,
        "rhat_max": float(diagnostics.rhat_max),
        "ess_bulk_min": float(diagnostics.ess_bulk_min),
        "divergences": int(diagnostics.divergences),
        "rhat_threshold": float(diagnostics.rhat_threshold),
        "ess_threshold": int(diagnostics.ess_threshold),
    }

    # Warn if loading old summary without sigma_rw_prior_type
    priors_dict = summary.get("priors", {})
    if isinstance(priors_dict, dict) and "sigma_rw_prior_type" not in priors_dict:
        log.warning(
            "prior_config_compat",
            message=(
                "Training summary missing sigma_rw_prior_type — "
                "defaulting to 'lognormal'. Original training used 'halfnormal'. "
                "This only affects retraining, not prediction."
            ),
        )

    # Prior predictive check on training data structure
    prior_predictive_result = None
    try:
        from panelcast.evaluation.prior_predictive import run_prior_predictive

        # Build training model_args from training data
        primary_split_dir_pp = resolve_split_dir(Path("data/splits"), PRIMARY_SPLIT)
        primary_features_dir_pp = _resolve_feature_split_dir(PRIMARY_SPLIT)
        train_df_pp = pd.read_parquet(primary_split_dir_pp / "train.parquet")
        train_features_pp = pd.read_parquet(primary_features_dir_pp / "train_features.parquet")

        train_df_pp = join_splits_with_features(
            train_df_pp, train_features_pp, name="prior_predictive_train"
        )

        feature_cols = summary["feature_cols"]
        train_df_pp[feature_cols] = train_df_pp[feature_cols].fillna(0)
        X_train = train_df_pp[feature_cols].values.astype(np.float32)
        scaler = summary.get("feature_scaler", {})
        if scaler:
            X_mean = np.array(scaler["mean"], dtype=np.float32)
            X_std = np.array(scaler["std"], dtype=np.float32)
            X_train = (X_train - X_mean) / X_std

        artist_to_idx = summary["artist_to_idx"]
        artist_idx_pp = train_df_pp[ds["entity_col"]].map(artist_to_idx)
        unknown_artist_mask_pp = artist_idx_pp.isna()
        if unknown_artist_mask_pp.any():
            unknown_artists = sorted(
                train_df_pp.loc[unknown_artist_mask_pp, ds["entity_col"]]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "Unknown artists found while preparing training prior-predictive inputs. "
                "This indicates train/summary mismatch. "
                f"n_unknown_rows={int(unknown_artist_mask_pp.sum())}, "
                f"unknown_artists_sample={unknown_artists[:5]}."
            )
        train_df_pp["_artist_idx"] = artist_idx_pp.astype(np.int32)
        train_df_pp["_album_seq"] = (
            train_df_pp.groupby(ds["entity_col"]).cumcount().astype(np.int32) + 1
        )

        global_mean = summary["global_mean_score"]
        prev_scores = (
            train_df_pp.groupby(ds["entity_col"])[ds["target_col"]]
            .shift(1)
            .fillna(global_mean)
            .values.astype(np.float32)
        )
        transform_pp = _transform_from_summary(summary)
        if transform_pp.name != "identity":
            prev_scores = np.asarray(transform_pp.forward(prev_scores), dtype=np.float32)

        if "n_reviews" in train_df_pp.columns:
            n_reviews_pp = train_df_pp["n_reviews"].fillna(1).values.astype(np.int32)
        elif ds["n_obs_col"] in train_df_pp.columns:
            n_reviews_pp = train_df_pp[ds["n_obs_col"]].fillna(1).values.astype(np.int32)
        else:
            n_reviews_pp = np.ones(len(train_df_pp), dtype=np.int32)

        train_model_args = {
            "artist_idx": train_df_pp["_artist_idx"].values,
            "album_seq": train_df_pp["_album_seq"].values,
            "prev_score": prev_scores,
            "X": X_train,
            "y": None,
            "n_reviews": n_reviews_pp,
            "n_artists": summary["n_artists"],
            "max_seq": summary["max_seq"],
            "n_exponent": summary.get("n_exponent", 0.0),
            "learn_n_exponent": summary.get("learn_n_exponent", False),
            "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
            "n_ref": summary.get("n_ref"),
            "likelihood_df": summary.get("likelihood_df", 4.0),
            "priors": PriorConfig(**summary["priors"]),
            "target_bounds": ds["target_bounds"],
            "ar_center": _ar_center_from_summary(summary),
        }

        if bool(getattr(train_model_args["priors"], "errors_in_variables", False)):
            global_std_pp = float(summary.get("global_std_score") or 0.0)
            if global_std_pp <= 0.0:
                log.warning("eiv_sigma_zero_legacy_summary", context="prior_predictive")
            prev_nrev_pp = (
                train_df_pp.groupby(ds["entity_col"])[
                    "n_reviews" if "n_reviews" in train_df_pp.columns else ds["n_obs_col"]
                ]
                .shift(1)
                .to_numpy(dtype=float)
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                train_model_args["prev_meas_sigma"] = np.where(
                    np.isnan(prev_nrev_pp) | (prev_nrev_pp <= 0),
                    0.0,
                    global_std_pp / np.sqrt(np.maximum(prev_nrev_pp, 1.0)),
                ).astype(np.float32)

        prior_predictive_result = run_prior_predictive(
            make_score_model(prefix),
            train_model_args,
            n_samples=500,
            max_obs=2000,
            seed=ctx.seed,
            score_bounds=ds["target_bounds"],
            transform=_transform_from_summary(summary),
        )
        log.info(
            "prior_predictive_complete",
            reasonable=prior_predictive_result.reasonable,
            fraction_in_bounds=prior_predictive_result.fraction_in_bounds,
            checks_passed=prior_predictive_result.checks_passed,
            flags=prior_predictive_result.informational_flags,
        )
    except Exception as e:
        log.warning("prior_predictive_failed", error=str(e))

    # Plausibility flags are informational by default but gate strict runs.
    if (
        ctx.strict
        and prior_predictive_result is not None
        and not prior_predictive_result.checks_passed
    ):
        raise ValueError(
            "Prior predictive plausibility checks failed under --strict: "
            + "; ".join(prior_predictive_result.informational_flags or [])
        )

    posterior_samples = _extract_posterior_samples(idata)
    first_var = next(iter(idata.posterior.data_vars))
    n_chains = int(idata.posterior[first_var].shape[0])
    n_draws = int(idata.posterior[first_var].shape[1])

    intervals = tuple(sorted(set(ctx.calibration_intervals)))
    split_results: dict[str, Any] = {}
    split_artifacts: dict[str, dict[str, Any]] = {}

    # Primary split: known-artist posterior predictive
    primary_split_dir = resolve_split_dir(Path("data/splits"), PRIMARY_SPLIT)
    primary_features_dir = _resolve_feature_split_dir(PRIMARY_SPLIT)
    primary_test_df = pd.read_parquet(primary_split_dir / "test.parquet")
    primary_train_df = pd.read_parquet(primary_split_dir / "train.parquet")
    primary_test_features = pd.read_parquet(primary_features_dir / "test_features.parquet")

    # Load validation split if it exists — its scores bridge the gap between
    # training and test for sequential prev_score computation.
    primary_val_df = None
    for val_name in ("validation.parquet", "val.parquet"):
        val_path = primary_split_dir / val_name
        if val_path.exists():
            primary_val_df = pd.read_parquet(val_path)
            log.info("validation_split_loaded", path=str(val_path), n_albums=len(primary_val_df))
            break

    primary_model_args, primary_y_true = _prepare_test_model_args(
        primary_test_df,
        primary_test_features,
        summary,
        train_df=primary_train_df,
        val_df=primary_val_df,
        strict=ctx.strict,
    )
    primary_y_samples = _run_known_artist_predictive(
        posterior_samples,
        primary_model_args,
        seed_offset=ctx.seed,
        prefix=prefix,
        batch_size=int(getattr(ctx, "predictive_batch_size", 500)),
    )
    # Predictive draws come out on the model scale; metrics/PPC/calibration
    # operate on the score scale.
    transform = _transform_from_summary(summary)
    if transform.name != "identity":
        primary_y_samples = np.asarray(transform.inverse(primary_y_samples))
    discretize_obs = bool(summary.get("discretize_observation", False))
    primary_metrics, primary_predictions, primary_calibration = _evaluate_predictions(
        primary_y_true,
        primary_y_samples,
        calibration_intervals=intervals,
        coverage_tolerance=ctx.coverage_tolerance,
        prediction_interval=ctx.prediction_interval,
        discretize=discretize_obs,
    )
    try:
        # Log-likelihood must be evaluated on the model scale; the ELPDs are
        # internally consistent per transform (cross-transform comparison
        # adds the Jacobian — see _compute_info_criteria docs).
        if transform.name != "identity":
            y_for_loglik = np.asarray(transform.forward(primary_y_true), dtype=np.float32)
        else:
            y_for_loglik = primary_y_true
        primary_info_criteria = _compute_info_criteria(
            posterior_samples=posterior_samples,
            model_args=primary_model_args,
            y_true=y_for_loglik,
            n_chains=n_chains,
            n_draws=n_draws,
            prefix=prefix,
            transform=transform,
            y_raw=primary_y_true,
            seed=ctx.seed,
            batch_size=int(getattr(ctx, "predictive_batch_size", 500)),
        )
    except Exception as e:
        if ctx.strict:
            raise
        # str(e) alone can be empty (e.g. MemoryError); always record the
        # exception type and log the traceback so failures are diagnosable.
        log.warning(
            "info_criteria_failed",
            error_type=type(e).__name__,
            error=str(e)[:2000],
            exc_info=True,
        )
        primary_info_criteria = {
            "status": "unavailable",
            "reason": f"{type(e).__name__}: {e}",
        }

    # Stratified diagnostics: how do accuracy, coverage and interval width
    # vary with the amount of artist history available at training time?
    # Informational only — never fail the stage over it.
    try:
        stratified_by_history = _stratify_metrics_by_history(
            artist_idx=np.asarray(primary_model_args["artist_idx"]),
            train_df=primary_train_df,
            summary=summary,
            y_true=primary_y_true,
            y_samples=primary_y_samples,
            interval=ctx.prediction_interval,
        )
    except Exception as e:
        log.warning(
            "stratified_metrics_failed",
            error_type=type(e).__name__,
            error=str(e)[:500],
        )
        stratified_by_history = []

    # Informational AR(1) adequacy check: within-artist lag-1 autocorrelation
    # of posterior-mean residuals (rows are artist/date sorted upstream).
    residual_acf = compute_residual_autocorrelation(
        primary_y_true - primary_y_samples.mean(axis=0),
        primary_model_args["artist_idx"],
    )
    log.info(
        "residual_autocorrelation",
        lag1_acf=residual_acf["lag1_acf"],
        n_pairs=residual_acf["n_pairs"],
    )

    split_results[PRIMARY_SPLIT] = {
        **primary_metrics,
        "n_test": int(len(primary_y_true)),
        "info_criteria": primary_info_criteria,
        "residual_autocorrelation": residual_acf,
        "stratified_by_history": stratified_by_history,
    }
    split_artifacts[PRIMARY_SPLIT] = {
        "predictions": primary_predictions,
        "calibration": primary_calibration,
    }

    # Secondary split: artist-disjoint cold-start predictive path
    if ctx.evaluate_secondary_split:
        secondary_split_dir = resolve_split_dir(Path("data/splits"), SECONDARY_SPLIT)
        secondary_features_dir = _resolve_feature_split_dir(SECONDARY_SPLIT)
        secondary_test_path = secondary_split_dir / "test.parquet"
        secondary_feat_path = secondary_features_dir / "test_features.parquet"

        if secondary_test_path.exists() and secondary_feat_path.exists():
            secondary_test_df = pd.read_parquet(secondary_test_path)
            secondary_test_features = pd.read_parquet(secondary_feat_path)
            X, prev_score, n_reviews, secondary_y_true = _prepare_disjoint_inputs(
                secondary_test_df,
                secondary_test_features,
                summary,
            )
            secondary_y_samples = _run_new_artist_predictive(
                posterior_samples=posterior_samples,
                summary=summary,
                X=X,
                prev_score=prev_score,
                n_reviews=n_reviews,
                seed=ctx.seed + 1000,
            )
            secondary_metrics, secondary_predictions, secondary_calibration = _evaluate_predictions(
                secondary_y_true,
                secondary_y_samples,
                calibration_intervals=intervals,
                coverage_tolerance=ctx.coverage_tolerance,
                prediction_interval=ctx.prediction_interval,
                discretize=discretize_obs,
            )
            split_results[SECONDARY_SPLIT] = {
                **secondary_metrics,
                "n_test": int(len(secondary_y_true)),
                "info_criteria": {
                    "status": "unavailable",
                    "reason": "entity-disjoint evaluation uses new-entity predictive path",
                },
            }
            split_artifacts[SECONDARY_SPLIT] = {
                "predictions": secondary_predictions,
                "calibration": secondary_calibration,
            }
        else:
            message = (
                "Secondary split evaluation enabled but required artifacts are missing: "
                f"test_exists={secondary_test_path.exists()}, "
                f"features_exists={secondary_feat_path.exists()}."
            )
            if ctx.strict:
                raise FileNotFoundError(message)
            log.warning(
                "secondary_split_missing_artifacts",
                split=SECONDARY_SPLIT,
                test_exists=secondary_test_path.exists(),
                features_exists=secondary_feat_path.exists(),
            )

    # Enforce calibration tolerance when strict mode is enabled.
    for split_name, result in split_results.items():
        if not result["calibration"]["within_tolerance"]:
            log.warning(
                "calibration_out_of_tolerance",
                split=split_name,
                tolerance=ctx.coverage_tolerance,
                coverages=result["calibration"]["coverages"],
            )
            if ctx.strict:
                raise ValueError(
                    f"Calibration coverage outside tolerance on split '{split_name}' "
                    f"(tolerance={ctx.coverage_tolerance})."
                )

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = output_dir / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics_result, indent=2)

    for split_name, artifacts in split_artifacts.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_json(split_dir / "predictions.json", artifacts["predictions"])
        _write_json(split_dir / "calibration.json", artifacts["calibration"])

    # Backward compatibility with existing dashboards/reporting paths.
    if PRIMARY_SPLIT in split_artifacts:
        _write_json(output_dir / "predictions.json", split_artifacts[PRIMARY_SPLIT]["predictions"])
        _write_json(output_dir / "calibration.json", split_artifacts[PRIMARY_SPLIT]["calibration"])

    primary = split_results[PRIMARY_SPLIT]
    metrics_full = {
        "schema_version": 2,
        "model": f"{prefix}_score",
        "model_path": str(model_path),
        "primary_split": PRIMARY_SPLIT,
        "splits": split_results,
        # Legacy top-level fields (primary split)
        "n_test": primary["n_test"],
        "point_metrics": primary["point_metrics"],
        "calibration": primary["calibration"],
        "crps": primary["crps"],
        "ppc": primary.get("ppc"),
        "info_criteria": primary.get("info_criteria"),
    }

    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics_full, indent=2)

    # Save prior predictive results
    if prior_predictive_result is not None:
        pp_payload = {
            "summary": prior_predictive_result.summary,
            "reasonable": prior_predictive_result.reasonable,
            "bounds": list(prior_predictive_result.bounds),
            "fraction_in_bounds": prior_predictive_result.fraction_in_bounds,
            "checks": prior_predictive_result.checks,
            "checks_passed": prior_predictive_result.checks_passed,
            "informational_flags": prior_predictive_result.informational_flags,
            "n_samples": prior_predictive_result.n_samples,
            "seed": prior_predictive_result.seed,
            "n_obs_original": prior_predictive_result.n_obs_original,
            "max_obs": prior_predictive_result.max_obs,
            "sampled_indices": (
                prior_predictive_result.sampled_indices.tolist()
                if prior_predictive_result.sampled_indices is not None
                else None
            ),
        }
        _write_json(output_dir / "prior_predictive.json", pp_payload, indent=2)

    log.info(
        "evaluation_pipeline_complete",
        diagnostics_path=str(diagnostics_path),
        metrics_path=str(metrics_path),
        splits=list(split_results.keys()),
    )

    return {
        "diagnostics": diagnostics_result,
        "metrics": metrics_full,
    }
