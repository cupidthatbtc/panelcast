"""Bayesian model training pipeline.

Fits NumPyro models on training data with configured MCMC parameters,
saves model artifacts, and handles convergence checking.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
import structlog

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.data.alignment import ROW_ID_COL, join_splits_with_features
from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.models.bayes.diagnostics import check_convergence
from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.io import save_model
from panelcast.models.bayes.model import compute_sigma_scaled, make_score_model
from panelcast.models.bayes.priors import priors_for_transform
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.errors import ConvergenceError
from panelcast.pipelines.training_summary import TrainingSummary
from panelcast.utils.hashing import hash_dataframe

if TYPE_CHECKING:
    from panelcast.pipelines.stages import StageContext

log = structlog.get_logger()

# Entity cardinality above which the gate-on {prefix}_entity_obs_raw plate is
# dropped from the saved fit (memory). At or below it the plate is kept so the
# evaluate stage conditions on each seen entity's fitted overdispersion instead
# of marginalizing it from the prior (the correct forward-split treatment).
# Only ever affects gate-on (heteroscedastic_entity_obs) fits; gate-off fits
# never create the site, so this is a no-op for every published number.
_ENTITY_OBS_KEEP_MAX = 20000


def _validate_strict_sampling_config(
    *,
    strict: bool,
    num_chains: int,
    num_samples: int,
    ess_threshold: int,
) -> None:
    """Fail fast when strict diagnostics are impossible by configuration."""
    if not strict:
        return

    if num_chains < 2:
        raise ConvergenceError(
            "Strict mode requires at least 2 chains for R-hat diagnostics. "
            f"Got num_chains={num_chains}. Increase --num-chains or disable --strict.",
            stage="train",
        )

    if num_samples < ess_threshold:
        raise ConvergenceError(
            "Strict mode requires num_samples >= ess_threshold per chain to make "
            f"ESS achievable. Got num_samples={num_samples}, ess_threshold={ess_threshold}. "
            "Increase --num-samples or lower --ess-threshold.",
            stage="train",
        )


def locate_level_prior(
    priors,
    ar_center_value: float,
    target_transform: str = "identity",
    logit_offset: float = 0.5,
    target_bounds: tuple[float, float] = (0.0, 100.0),
):
    """Locate the mu_artist prior at the score level when AR centering is on.

    With AR centering active the AR term no longer absorbs the score level,
    so mu_artist becomes the level parameter. The legacy Normal(0, scale)
    prior would sit tens of SDs from the posterior on the raw score scale.
    The location is the model-scale centering value (forward-transformed
    under a non-identity target transform).

    No-ops when centering is off (priors.ar_center == "none", value 0.0) or
    when the user explicitly configured a non-zero mu_artist_loc.
    """
    if priors.ar_center == "none" or priors.mu_artist_loc != 0.0:
        return priors
    level = float(ar_center_value)
    if target_transform != "identity":
        level = float(
            get_transform(
                target_transform,
                target_bounds=target_bounds,
                offset=logit_offset,
            ).forward(level)
        )
    log.info(
        "mu_artist_prior_located",
        mu_artist_loc=level,
        reason="ar_center frees mu_artist to carry the score level",
    )
    return dataclasses.replace(priors, mu_artist_loc=level)


def load_training_data(
    features_path: Path,
    splits_path: Path,
    min_albums_filter: int = 2,
    descriptor: DatasetDescriptor | None = None,
    debut_prev_score_source: str = "train_mean",
    target_transform: str = "identity",
    logit_offset: float = 0.5,
    ar_center: str = "global",
    entity_group_pooling: bool = False,
) -> tuple[dict, list[str], pd.DataFrame]:
    """Load training data and prepare model arguments.

    Loads feature and split parquet files, merges them, fills NaN values,
    and prepares the model_args dictionary for MCMC fitting.

    Args:
        features_path: Path to train_features.parquet.
        splits_path: Path to train.parquet (splits).
        min_albums_filter: Minimum albums for dynamic effects.
        descriptor: Dataset descriptor (None = AOTY defaults).

    Returns:
        Tuple of (model_args dict, feature_cols list, merged train_df).
    """
    train_features = pd.read_parquet(features_path)
    train_df = pd.read_parquet(splits_path)

    # Join on the stable original_row_id key (falls back to positional index
    # alignment for legacy feature parquets without the key column).
    train_df = join_splits_with_features(train_df, train_features, name="train")

    # Keep n_reviews for heteroscedastic uncertainty modeling only.
    # It should not be included in X (mean model predictors).
    # original_row_id is join metadata, never a predictor.
    feature_cols = [c for c in train_features.columns if c not in ("n_reviews", ROW_ID_COL)]
    if "n_reviews" in train_features.columns:
        log.info(
            "excluding_n_reviews_from_predictors",
            reason="n_reviews_used_only_for_noise_scaling",
        )
    if not feature_cols:
        raise ValueError(
            "No predictor features available after excluding n_reviews. "
            "Feature parquet must include at least one predictor column."
        )

    # Handle NaN values in predictor features (fill with 0 for numeric stability)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    # Prepare model data
    model_args, valid_mask = prepare_model_data(
        train_df,
        feature_cols,
        min_albums_filter=min_albums_filter,
        descriptor=descriptor,
        debut_prev_score_source=debut_prev_score_source,
        target_transform=target_transform,
        logit_offset=logit_offset,
        ar_center=ar_center,
        entity_group_pooling=entity_group_pooling,
    )

    # Apply valid_mask to train_df so it matches the filtered model arrays
    train_df = train_df[valid_mask].copy()

    return model_args, feature_cols, train_df


def _build_entity_groups(
    train_df: pd.DataFrame,
    artists: list,
    entity_col: str,
    group_col: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Per-entity group indices for the entity_group_pooling gate.

    Each entity's group is its modal ``group_col`` value over its training
    rows (pandas mode sorts, so ties break deterministically). Entities with
    no observed group and groups holding fewer than two entities collapse
    into the ``__rest__`` bucket at index 0, so every learned offset is
    informed by at least two entities.
    """
    modal = train_df.groupby(entity_col)[group_col].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None
    )
    counts = modal.value_counts()
    small = set(counts[counts < 2].index)
    bucketed = {
        a: ("__rest__" if (g is None or pd.isna(g) or g in small) else g)
        for a, g in modal.items()
    }
    groups_sorted = sorted({g for g in bucketed.values() if g != "__rest__"})
    group_to_idx = {"__rest__": 0, **{g: i + 1 for i, g in enumerate(groups_sorted)}}
    group_idx_by_artist = np.array([group_to_idx[bucketed[a]] for a in artists], dtype=np.int32)
    return group_idx_by_artist, group_to_idx


def prepare_model_data(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    min_albums_filter: int = 2,
    descriptor: DatasetDescriptor | None = None,
    debut_prev_score_source: str = "train_mean",
    target_transform: str = "identity",
    logit_offset: float = 0.5,
    ar_center: str = "global",
    entity_group_pooling: bool = False,
) -> tuple[dict, np.ndarray]:
    """Prepare data for NumPyro model fitting.

    Creates the arrays needed by the Bayesian model including artist indices,
    album sequences, and feature matrix.

    Args:
        train_df: Training data with features and target.
        feature_cols: List of feature column names.
        min_albums_filter: Minimum albums for dynamic effects. Artists with
            fewer albums have all their albums treated as sequence 1 (static effect only).
        descriptor: Dataset descriptor providing entity/target/n_obs column
            names (None = AOTY defaults).
        debut_prev_score_source: Where the debut-album prev_score fill comes
            from. "train_mean" (default) uses the training-split mean — no
            information from held-out rows. "dataset_stats" reproduces the
            legacy behavior of reading the pre-split mean from
            data/processed/dataset_stats.json (mild test-set leakage).
        target_transform: "identity" (default) keeps scores on their natural
            scale; "offset_logit" trains the model on the Smithson-Verkuilen
            logit scale (y and prev_score are forward-transformed AFTER the
            debut fill happens on the raw scale).
        logit_offset: Half-count continuity offset for offset_logit.
        ar_center: AR(1) centering mode. "global" (default) centers
            prev_score on the same value used for the debut fill, so debut
            AR terms are exactly zero; "none" reproduces the legacy
            uncentered form (ar_center model arg = 0.0); "artist_running"
            centers each observation on the artist's running mean of
            previous training scores (sensitivity analysis only).
        entity_group_pooling: When True, build per-entity group indices from
            the descriptor's entity_group_col (modal value over training
            rows; sparse/missing groups bucket to "__rest__") and add
            group_idx_by_artist / n_groups / group_to_idx to model_args.

    Returns:
        Tuple of (model_args dict, valid_mask boolean array indicating retained rows).
    """
    descriptor = descriptor or DatasetDescriptor()
    entity_col = descriptor.entity_col
    target_col = descriptor.target_col
    n_obs_col = descriptor.n_obs_col

    # Create artist index mapping (sorted for deterministic ordering)
    artists = sorted(train_df[entity_col].unique())
    artist_to_idx = {a: i for i, a in enumerate(artists)}
    artist_idx = train_df[entity_col].map(artist_to_idx).values

    # Album sequence (within artist, 1-indexed to match model expectations)
    album_seq = (train_df.groupby(entity_col).cumcount() + 1).values

    # Apply min_albums_filter: artists below threshold get static effect only
    # by clamping their album_seq to 1
    artist_counts = train_df.groupby(entity_col).size()
    below_threshold = train_df[entity_col].map(artist_counts < min_albums_filter).values
    album_seq = np.where(below_threshold, 1, album_seq)

    # Previous score (shifted within artist, using a global mean for debut
    # albums). Default source is the training-split mean: it carries no
    # information from held-out rows. The legacy "dataset_stats" source reads
    # the PRE-SPLIT mean from data/processed/dataset_stats.json, which leaks
    # a small amount of test-set information into training.
    train_df = train_df.copy()
    train_df["prev_score"] = train_df.groupby(entity_col)[target_col].shift(1)
    if debut_prev_score_source == "dataset_stats":
        stats_path = Path("data/processed/dataset_stats.json")
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                dataset_stats = json.load(f)
            global_mean = float(dataset_stats["global_mean_score"])
            log.info("debut_prev_score_source", source="dataset_stats.json", value=global_mean)
        else:
            global_mean = float(train_df[target_col].mean())
            log.error(
                "debut_prev_score_fallback",
                source="training_mean",
                value=global_mean,
                reason=(
                    "data/processed/dataset_stats.json not found — using training-set "
                    "mean despite debut_prev_score_source='dataset_stats'."
                ),
            )
    elif debut_prev_score_source == "train_mean":
        global_mean = float(train_df[target_col].mean())
        log.info("debut_prev_score_source", source="train_split_mean", value=global_mean)
    else:
        raise ValueError(
            f"Unknown debut_prev_score_source: '{debut_prev_score_source}'. "
            "Expected 'train_mean' or 'dataset_stats'."
        )
    train_df["prev_score"] = train_df["prev_score"].fillna(global_mean)
    prev_score = train_df["prev_score"].values

    # Lagged review count for the errors-in-variables measurement-error scale
    # (prev_meas_sigma = global_std_score / sqrt(prev_n_reviews), built below).
    # Debuts (no prior album) lag to NaN and are pinned to 0 so their de-noised
    # regressor stays at the debut fill and debut AR terms remain exactly zero.
    if "n_reviews" in train_df.columns:
        _nrev_lag_col: str | None = "n_reviews"
    elif n_obs_col in train_df.columns:
        _nrev_lag_col = n_obs_col
    else:
        _nrev_lag_col = None
    if _nrev_lag_col is not None:
        prev_n_reviews = train_df.groupby(entity_col)[_nrev_lag_col].shift(1).to_numpy(dtype=float)
    else:
        prev_n_reviews = np.full(len(train_df), np.nan)

    # AR(1) centering: ar_term = rho * (prev_score - center). "global" shares
    # the debut-fill value, so debut AR terms vanish exactly and rho stops
    # absorbing the overall score level.
    ar_center_arr: np.ndarray | np.floating
    if ar_center == "none":
        ar_center_arr = np.float32(0.0)
        ar_center_value = 0.0
    elif ar_center == "global":
        ar_center_arr = np.float32(global_mean)
        ar_center_value = float(global_mean)
        log.info("ar_center", mode="global", value=ar_center_value)
    elif ar_center == "artist_running":
        # Running mean of the artist's PREVIOUS training scores; debuts fall
        # back to the debut-fill value so their AR terms still vanish.
        # Sensitivity analysis only: overlaps with the artist effect.
        running = train_df.groupby(entity_col)[target_col].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        ar_center_arr = running.fillna(global_mean).values.astype(np.float32)
        ar_center_value = float(global_mean)
        log.info("ar_center", mode="artist_running", fallback_value=ar_center_value)
    else:
        raise ValueError(
            f"Unknown ar_center: '{ar_center}'. Expected 'none', 'global', or 'artist_running'."
        )

    # Feature matrix
    X = train_df[feature_cols].values.astype(np.float32)

    # Target
    y = train_df[target_col].values.astype(np.float32)

    # Target transform: debut fill above happened on the raw scale; under
    # offset_logit both the target and the AR input move to the logit scale.
    if target_transform != "identity":
        transform = get_transform(
            target_transform,
            target_bounds=tuple(descriptor.target_bounds),
            offset=logit_offset,
        )
        y = np.asarray(transform.forward(y), dtype=np.float32)
        prev_score = np.asarray(transform.forward(prev_score), dtype=np.float32)
        # The center is subtracted on the model scale; "none" keeps the
        # legacy uncentered form (0.0 stays 0.0, not forward(0.0)).
        if ar_center != "none":
            ar_center_arr = np.asarray(transform.forward(ar_center_arr), dtype=np.float32)

    # Errors-in-variables measurement-error scale, on the model scale (matches
    # prev_score after any transform). Fixed and data-derived -> no funnel.
    global_std_score = float(np.std(y)) if len(y) else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        prev_meas_sigma = np.where(
            np.isnan(prev_n_reviews) | (prev_n_reviews <= 0),
            0.0,
            global_std_score / np.sqrt(np.maximum(prev_n_reviews, 1.0)),
        ).astype(np.float32)

    # Extract n_reviews for heteroscedastic noise
    # Keep as raw values (may be float with NaN) for proper NaN detection before int cast
    if "n_reviews" in train_df.columns:
        n_reviews_raw = train_df["n_reviews"].values
    else:
        # Fallback: if n_reviews not in features, try the raw count column
        if n_obs_col in train_df.columns:
            n_reviews_raw = train_df[n_obs_col].values
        else:
            raise ValueError(
                "n_reviews column not found. Feature parquet must include n_reviews "
                f"or source data must include {n_obs_col}."
            )

    # Validate n_reviews: identify missing or invalid values BEFORE int cast
    # NaN cannot be represented in int32, so detection must happen on raw values
    invalid_mask = pd.isna(n_reviews_raw) | (n_reviews_raw <= 0)
    n_invalid = invalid_mask.sum()

    # Track which rows are valid (returned to caller for DataFrame filtering)
    valid_mask = ~invalid_mask

    if n_invalid > 0:
        invalid_pct = n_invalid / len(n_reviews_raw) * 100
        if invalid_pct > 50:
            raise ValueError(
                f"Too many invalid n_reviews: {n_invalid}/{len(n_reviews_raw)} "
                f"({invalid_pct:.1f}%). Check source data for missing observation counts."
            )
        # Log warning about rows that will be dropped
        log.warning(
            "n_reviews_invalid_rows",
            n_invalid=n_invalid,
            pct_invalid=round(invalid_pct, 1),
            action="dropping_invalid_rows",
        )
        # Filter out invalid rows from all arrays
        n_reviews_raw = n_reviews_raw[valid_mask]
        y = y[valid_mask]
        X = X[valid_mask]
        artist_idx = artist_idx[valid_mask]
        album_seq = album_seq[valid_mask]
        prev_score = prev_score[valid_mask]
        prev_meas_sigma = prev_meas_sigma[valid_mask]
        if np.ndim(ar_center_arr) > 0:
            # np.asarray is a no-op for the array branch (ndim > 0 here) but tells
            # the type checker the value is indexable (mypy can't narrow on np.ndim).
            ar_center_arr = np.asarray(ar_center_arr)[valid_mask]

    # Cast to int32 AFTER filtering (NaN-free at this point)
    n_reviews = n_reviews_raw.astype(np.int32)

    # Compute album counts per artist (indexed by artist_idx, not artist name)
    artist_album_counts = pd.Series(artist_idx).value_counts().sort_index()
    # Reindex to full range so _apply_max_albums_cap doesn't get IndexError
    artist_album_counts = artist_album_counts.reindex(range(len(artists)), fill_value=0)

    # Validate that all artist indices in the data are within the expected range
    max_idx = artist_idx.max() if len(artist_idx) > 0 else 0
    if max_idx >= len(artists):
        raise ValueError(
            f"artist_idx contains index {max_idx} but only {len(artists)} artists exist. "
            "This indicates a mismatch between artist mapping and filtered data."
        )

    model_args = {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
        "prev_score": prev_score,
        "prev_meas_sigma": prev_meas_sigma,
        "X": X,
        "y": y,
        "n_reviews": n_reviews,
        "n_artists": len(artists),
        "artist_album_counts": artist_album_counts,
        "artist_to_idx": artist_to_idx,
        "global_mean_score": global_mean,
        "global_std_score": global_std_score,
        "ar_center": ar_center_arr,
        "ar_center_value": ar_center_value,
    }

    if entity_group_pooling:
        group_col = descriptor.entity_group_col
        if group_col is None:
            raise ValueError(
                "entity_group_pooling=True but the dataset descriptor has no "
                "entity_group_col — the gate is unusable for this domain."
            )
        if group_col not in train_df.columns:
            raise ValueError(
                f"entity_group_pooling=True but column '{group_col}' is missing "
                "from the training split."
            )
        group_idx_by_artist, group_to_idx = _build_entity_groups(
            train_df, artists, entity_col, group_col
        )
        model_args["group_idx_by_artist"] = group_idx_by_artist
        model_args["n_groups"] = len(group_to_idx)
        model_args["group_to_idx"] = group_to_idx
        log.info(
            "entity_group_pooling",
            group_col=group_col,
            n_groups=len(group_to_idx),
            n_rest=int(np.sum(group_idx_by_artist == 0)),
        )

    return model_args, valid_mask


def _apply_max_albums_cap(
    model_args: dict,
    max_albums_cap: int,
    artist_album_counts: pd.Series,
) -> dict:
    """Apply max_albums cap to model arguments, keeping most recent albums.

    CAP BEHAVIOR (max-events cap; domain-agnostic): an entity's events beyond
    the most recent ``max_albums_cap`` are NOT dropped — they are collapsed onto
    sequence position 1 (the initial entity effect). Every row still contributes
    to the likelihood; only the time-varying latent it indexes changes. So the
    cap bounds the random-walk trajectory length (and peak memory), it does not
    subsample data. For AOTY the default is 50 albums/artist; a domain with
    longer histories should raise ``--max-albums`` (see docs/EVALUATION_PROTOCOL.md).

    For artists with more than max_albums_cap albums, renumbers so that the
    most recent albums get distinct positions (1 to max_albums_cap) and
    older albums share position 1.

    The hard cap is motivated by the random walk model structure:
    - Recent K albums get distinct positions, tracking career trajectory
      via the random walk innovations (sigma_rw per step).
    - Older albums share position 1, using only the initial artist effect
      (mu_artist + sigma_artist * z).  Over many random walk steps the
      cumulative variance grows, so distant positions carry little signal
      about the current state — grouping them at the initial effect is a
      principled simplification, not an approximation artifact.
    - No leakage since album_seq is calculated on training data only.

    Args:
        model_args: Dictionary from prepare_model_data.
        max_albums_cap: Maximum albums per artist (from ctx.max_albums).
        artist_album_counts: Series mapping artist index to album count.

    Returns:
        Updated model_args with adjusted album_seq and max_seq.
    """
    # Guard against non-positive max_albums_cap to ensure valid shapes
    max_albums_cap = max(1, int(max_albums_cap))

    album_seq = model_args["album_seq"]
    artist_idx = model_args["artist_idx"]

    # For each artist, compute offset to shift album_seq so most recent albums
    # get positions 1 to max_albums_cap, and older albums share position 1
    # offset = max(0, n_albums - max_albums_cap)
    offsets = np.maximum(0, artist_album_counts.values - max_albums_cap)

    # Apply per-artist offset: new_seq = max(1, original_seq - offset[artist])
    artist_offsets = offsets[artist_idx]
    album_seq = np.maximum(1, album_seq - artist_offsets).astype(np.int32)

    # Compute max_seq from actual capped album_seq values for consistency.
    # Since album_seq is 1-indexed and model converts to 0-indexed, max_seq = album_seq.max().
    max_seq = int(album_seq.max())

    n_capped_artists = (artist_album_counts > max_albums_cap).sum()
    if n_capped_artists > 0:
        log.info(
            "max_albums_applied",
            max_albums=max_albums_cap,
            artists_capped=int(n_capped_artists),
            message=f"Using {max_albums_cap} most recent albums per artist",
        )

    model_args["album_seq"] = album_seq
    model_args["max_seq"] = max_seq
    return model_args


def _build_heteroscedastic_summary(
    ctx: StageContext,
    fit_result,
    model_args: dict,
    prefix: str,
    n_ref,
) -> dict:
    """Summarize the heteroscedastic noise mode from the fitted posterior.

    Pure read of ``fit_result.idata.posterior`` plus numpy/ArviZ; returns the
    ``summary["heteroscedastic_mode"]`` payload for the learned, fixed, or
    homoscedastic case and emits the same ``heteroscedastic_summary`` log line.
    """
    if ctx.learn_n_exponent:
        # Extract n_exponent posterior
        n_exp_samples = fit_result.idata.posterior[f"{prefix}_n_exponent"].values.flatten()
        n_exp_mean = float(np.mean(n_exp_samples))
        n_exp_std = float(np.std(n_exp_samples))

        # Compute 94% HDI for n_exponent
        hdi = az.hdi(fit_result.idata, var_names=[f"{prefix}_n_exponent"], hdi_prob=0.94)
        hdi_low = float(hdi[f"{prefix}_n_exponent"].values[0])
        hdi_high = float(hdi[f"{prefix}_n_exponent"].values[1])

        # Get ESS and R-hat for n_exponent
        n_exp_summary = az.summary(
            fit_result.idata, var_names=[f"{prefix}_n_exponent"], kind="diagnostics"
        )

        # Check if sigma_ref mode is active (n_ref was passed to model)
        use_sigma_ref = model_args.get("n_ref") is not None

        # Compute effective sigma range using posterior mean exponent
        n_reviews = model_args["n_reviews"]
        sigma_obs_mean = float(fit_result.idata.posterior[f"{prefix}_sigma_obs"].mean())
        sigma_at_max_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.max(n_reviews)), jnp.array(n_exp_mean)
            )
        )
        sigma_at_min_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.min(n_reviews)), jnp.array(n_exp_mean)
            )
        )

        # Reference scaling values for interpretation
        ref_sqrt = 0.5  # Square-root scaling
        ref_cube_root = 0.33  # Cube-root scaling
        interpretation = (
            "closer to cube-root scaling (0.33)"
            if abs(n_exp_mean - ref_cube_root) < abs(n_exp_mean - ref_sqrt)
            else "closer to square-root scaling (0.5)"
        )

        # Build heteroscedastic_mode dict (common fields)
        hetero_dict = {
            "mode": "learned",
            "n_exponent_mean": n_exp_mean,
            "n_exponent_std": n_exp_std,
            "n_exponent_hdi_94": [hdi_low, hdi_high],
            "n_exponent_ess_bulk": int(n_exp_summary["ess_bulk"].values[0]),
            "n_exponent_r_hat": float(n_exp_summary["r_hat"].values[0]),
            "interpretation": interpretation,
            "reference_sqrt": ref_sqrt,
            "reference_cube_root": ref_cube_root,
            "sigma_scaled_range": {
                "min": sigma_at_max_n,
                "max": sigma_at_min_n,
                "at_n_reviews_max": int(np.max(n_reviews)),
                "at_n_reviews_min": int(np.min(n_reviews)),
                "base_sigma_obs": sigma_obs_mean,
            },
        }

        if use_sigma_ref:
            # Sigma-ref mode: add sigma_ref stats and sigma_obs derived stats
            sigma_ref_samples = fit_result.idata.posterior[f"{prefix}_sigma_ref"].values.flatten()
            sigma_ref_hdi = az.hdi(
                fit_result.idata,
                var_names=[f"{prefix}_sigma_ref"],
                hdi_prob=0.94,
            )
            sigma_ref_hdi_low = float(sigma_ref_hdi[f"{prefix}_sigma_ref"].values[0])
            sigma_ref_hdi_high = float(sigma_ref_hdi[f"{prefix}_sigma_ref"].values[1])

            sigma_obs_samples = fit_result.idata.posterior[f"{prefix}_sigma_obs"].values.flatten()

            hetero_dict["parameterization"] = "sigma_ref"
            hetero_dict["n_ref"] = n_ref
            hetero_dict["n_ref_method"] = "median"
            hetero_dict["sigma_ref"] = {
                "mean": float(np.mean(sigma_ref_samples)),
                "std": float(np.std(sigma_ref_samples)),
                "hdi_94": [sigma_ref_hdi_low, sigma_ref_hdi_high],
            }
            hetero_dict["sigma_obs_derived"] = {
                "mean": float(np.mean(sigma_obs_samples)),
                "std": float(np.std(sigma_obs_samples)),
            }
        else:
            hetero_dict["parameterization"] = "sigma_obs"

        log.info(
            "heteroscedastic_summary",
            mode="learned",
            parameterization="sigma_ref" if use_sigma_ref else "sigma_obs",
            n_exponent_mean=round(n_exp_mean, 4),
            n_exponent_hdi_94=[round(hdi_low, 4), round(hdi_high, 4)],
            interpretation=interpretation,
            sigma_range=[round(sigma_at_max_n, 4), round(sigma_at_min_n, 4)],
        )
        return hetero_dict

    if ctx.n_exponent != 0.0:
        # Fixed heteroscedastic mode
        n_reviews = model_args["n_reviews"]
        sigma_obs_mean = float(fit_result.idata.posterior[f"{prefix}_sigma_obs"].mean())
        # Wrap numpy scalars in JAX arrays for compute_sigma_scaled compatibility
        sigma_at_max_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.max(n_reviews)), jnp.array(ctx.n_exponent)
            )
        )
        sigma_at_min_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.min(n_reviews)), jnp.array(ctx.n_exponent)
            )
        )

        log.info(
            "heteroscedastic_summary",
            mode="fixed",
            n_exponent=ctx.n_exponent,
            sigma_range=[round(sigma_at_max_n, 4), round(sigma_at_min_n, 4)],
        )
        return {
            "mode": "fixed",
            "n_exponent": ctx.n_exponent,
            "sigma_scaled_range": {
                "min": sigma_at_max_n,
                "max": sigma_at_min_n,
                "at_n_reviews_max": int(np.max(n_reviews)),
                "at_n_reviews_min": int(np.min(n_reviews)),
                "base_sigma_obs": sigma_obs_mean,
            },
        }

    # Homoscedastic mode (default)
    return {
        "mode": "homoscedastic",
    }


def train_models(
    ctx: StageContext,
    features_path: Path | None = None,
    splits_path: Path | None = None,
) -> dict:
    """Train Bayesian models on feature data.

    Fits the user score model using MCMC, checks convergence,
    and saves model artifacts.

    Args:
        ctx: Stage context with run configuration.
        features_path: Optional path to features parquet. Defaults to
            data/features/train_features.parquet.
        splits_path: Optional path to splits parquet. Defaults to
            data/splits/within_entity_temporal/train.parquet (resolving a
            legacy-named directory if only that exists).

    Returns:
        Dictionary with training results and paths.

    Raises:
        ConvergenceError: If strict mode and divergences > 0.
    """
    log.info(
        "train_pipeline_start",
        seed=ctx.seed,
        strict=ctx.strict,
        max_albums=ctx.max_albums,
        min_albums_filter=ctx.min_albums_filter,
        num_chains=ctx.num_chains,
        num_samples=ctx.num_samples,
        num_warmup=ctx.num_warmup,
        target_accept=ctx.target_accept,
    )

    _validate_strict_sampling_config(
        strict=ctx.strict,
        num_chains=ctx.num_chains,
        num_samples=ctx.num_samples,
        ess_threshold=ctx.ess_threshold,
    )

    # Resolve dataset descriptor once; all column names and posterior-site
    # prefixes below derive from it (defaults reproduce AOTY behavior).
    descriptor = getattr(ctx, "descriptor", None) or DatasetDescriptor()
    prefix = descriptor.model_prefix

    # Load training data using shared function
    features_path = features_path or Path("data/features/train_features.parquet")
    splits_path = splits_path or (
        resolve_split_dir(Path("data/splits"), SplitType.WITHIN_ENTITY_TEMPORAL) / "train.parquet"
    )

    debut_prev_score_source = str(getattr(ctx, "debut_prev_score_source", "train_mean"))
    target_transform = str(getattr(ctx, "target_transform", "identity"))
    logit_offset = float(getattr(ctx, "logit_offset", 0.5))
    ar_center = str(getattr(ctx, "ar_center", "global"))
    entity_group_pooling = bool(getattr(ctx, "entity_group_pooling", False))
    model_args, feature_cols, train_df = load_training_data(
        features_path=features_path,
        splits_path=splits_path,
        min_albums_filter=ctx.min_albums_filter,
        descriptor=descriptor,
        debut_prev_score_source=debut_prev_score_source,
        target_transform=target_transform,
        logit_offset=logit_offset,
        ar_center=ar_center,
        entity_group_pooling=entity_group_pooling,
    )

    # Compute artists below threshold for metadata
    artist_counts = train_df.groupby(descriptor.entity_col).size()
    n_below_threshold = (artist_counts < ctx.min_albums_filter).sum()

    log.info(
        "data_loaded",
        train_rows=len(train_df),
        n_features=len(feature_cols),
    )

    # Pop metadata fields before passing to NumPyro model
    artist_album_counts = model_args.pop("artist_album_counts")
    artist_to_idx = model_args.pop("artist_to_idx")
    group_to_idx = model_args.pop("group_to_idx", None)
    global_mean_score = model_args.pop("global_mean_score")
    # prepare_model_data always supplies this; fall back to the score std for
    # hand-built model_args (test fixtures that bypass prepare_model_data).
    global_std_score = model_args.pop(
        "global_std_score",
        float(np.std(model_args["y"])) if model_args.get("y") is not None else 0.0,
    )
    ar_center_value = model_args.pop("ar_center_value")

    # Apply max_albums cap from CLI/config (uses most recent albums per artist)
    model_args = _apply_max_albums_cap(model_args, ctx.max_albums, artist_album_counts)

    # Log n_reviews statistics for diagnostics
    n_reviews = model_args["n_reviews"]
    log.info(
        "n_reviews_distribution",
        min=int(np.min(n_reviews)),
        max=int(np.max(n_reviews)),
        median=int(np.median(n_reviews)),
        mean=float(np.mean(n_reviews)),
    )

    # Compute reference review count for sigma-ref reparameterization
    n_ref = float(np.median(n_reviews))
    log.info("sigma_ref_mode", n_ref=n_ref, n_ref_method="median")

    # Standardize feature matrix X (z-score per column) so that
    # beta ~ N(0, 1) prior is appropriate regardless of feature scale.
    # n_reviews is NOT standardized -- it lives outside X and is used
    # for heteroscedastic noise scaling on its natural scale.
    X_raw = model_args["X"]
    if np.isnan(X_raw).any():
        n_nan = int(np.isnan(X_raw).sum())
        raise ValueError(
            f"Feature matrix X contains {n_nan} NaN values after fillna(0). "
            "Check feature engineering pipeline for columns producing NaN."
        )
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    # Guard against constant features (std=0): leave them unscaled
    X_std_safe = np.where(X_std == 0.0, 1.0, X_std)
    model_args["X"] = ((X_raw - X_mean) / X_std_safe).astype(np.float32)
    std_range_val = [float(X_std.min()), float(X_std.max())] if len(X_std) > 0 else []
    log.info(
        "features_standardized",
        n_features=len(X_mean),
        n_constant=int((X_std == 0.0).sum()),
        std_range=std_range_val,
    )
    # Store scaler params for prediction-time use
    feature_scaler = {
        "mean": X_mean.tolist(),
        "std": X_std_safe.tolist(),
        "feature_cols": feature_cols,
    }

    log.info(
        "model_data_prepared",
        n_artists=model_args["n_artists"],
        n_observations=len(model_args["y"]),
        n_features=model_args["X"].shape[1],
        max_seq=model_args["max_seq"],
        n_reviews_shape=model_args["n_reviews"].shape,
    )

    # Add heteroscedastic noise configuration to model_args
    model_args["n_exponent"] = ctx.n_exponent
    model_args["learn_n_exponent"] = ctx.learn_n_exponent
    model_args["n_exponent_prior"] = ctx.n_exponent_prior
    model_args["likelihood_df"] = getattr(ctx, "likelihood_df", 4.0)

    # Add n_ref for sigma-ref reparameterization (model accepts n_ref=None for homoscedastic)
    model_args["n_ref"] = n_ref if (ctx.learn_n_exponent or ctx.n_exponent != 0.0) else None
    model_args["n_ref_method"] = "median"

    # Log heteroscedastic mode
    if ctx.learn_n_exponent:
        if ctx.n_exponent_prior == "beta":
            log.info(
                "heteroscedastic_mode",
                mode="learned",
                prior_type="beta",
                prior_alpha=ctx.n_exponent_alpha,
                prior_beta=ctx.n_exponent_beta,
            )
        else:
            log.info(
                "heteroscedastic_mode",
                mode="learned",
                prior_type=ctx.n_exponent_prior,
            )
    elif ctx.n_exponent != 0.0:
        log.info("heteroscedastic_mode", mode="fixed", exponent=ctx.n_exponent)
    else:
        log.info("heteroscedastic_mode", mode="homoscedastic")

    # Configure MCMC from CLI args
    mcmc_config = MCMCConfig(
        num_warmup=ctx.num_warmup,
        num_samples=ctx.num_samples,
        num_chains=ctx.num_chains,
        seed=ctx.seed,
        target_accept_prob=ctx.target_accept,
        max_tree_depth=ctx.max_tree_depth,
        chain_method=ctx.chain_method,
    )

    # Get priors with heteroscedastic config from CLI; the transform factory
    # right-sizes noise scales when training on the logit scale.
    priors = priors_for_transform(
        target_transform,
        logit_offset=logit_offset,
        n_exponent_alpha=ctx.n_exponent_alpha,
        n_exponent_beta=ctx.n_exponent_beta,
        ar_center=ar_center,
        latent_process=str(getattr(ctx, "latent_process", "rw")),
        sigma_obs_prior_type=str(getattr(ctx, "sigma_obs_prior_type", "halfnormal")),
        heteroscedastic_entity_obs=bool(getattr(ctx, "heteroscedastic_entity_obs", False)),
        tau_entity_scale=float(getattr(ctx, "tau_entity_scale", 0.25)),
        likelihood_family=str(getattr(ctx, "likelihood_family", "studentt")),
        discretize_observation=bool(getattr(ctx, "discretize_observation", False)),
        errors_in_variables=bool(getattr(ctx, "errors_in_variables", False)),
        propagate_rw_horizon=bool(getattr(ctx, "propagate_rw_horizon", False)),
        entity_group_pooling=entity_group_pooling,
    )

    priors = locate_level_prior(
        priors,
        ar_center_value=ar_center_value,
        target_transform=target_transform,
        logit_offset=logit_offset,
        target_bounds=tuple(descriptor.target_bounds),
    )
    model_args["priors"] = priors
    model_args["target_bounds"] = tuple(descriptor.target_bounds)

    # Fit model
    log.info("fitting_model", model=f"{prefix}_score")
    exclude_rw_raw_from_collection = getattr(ctx, "exclude_rw_raw_from_collection", False)
    entity_obs_on = bool(getattr(ctx, "heteroscedastic_entity_obs", False))
    # rw_raw is the always-excluded large tensor. When the entity gate is on,
    # entity_obs_raw is the per-entity unit-normal plate. KEEP it when the entity
    # cardinality is small (so the evaluate stage conditions on each seen
    # entity's *fitted* overdispersion on forward splits -- the correct warm
    # treatment); only DROP it above the cap, where memory forces the cold-start
    # prior-marginalization (e.g. ~50k-director domains). The interpretable
    # deterministic {prefix}_entity_log_scale is kept either way.
    n_artists_fit = int(model_args["n_artists"])
    drop_entity_obs = entity_obs_on and n_artists_fit > _ENTITY_OBS_KEEP_MAX
    idata_excludes = [f"{prefix}_rw_raw"]
    collection_excludes = [f"{prefix}_rw_raw"] if exclude_rw_raw_from_collection else []
    if drop_entity_obs:
        idata_excludes.append(f"{prefix}_entity_obs_raw")
        if exclude_rw_raw_from_collection:
            collection_excludes.append(f"{prefix}_entity_obs_raw")
    # The errors-in-variables regressor latent is an n_obs-cardinality unit
    # normal like rw_raw: always drop it from the saved fit (re-sampled from its
    # prior when marginalized in evaluate). Created only on the gate-on branch.
    if bool(getattr(ctx, "errors_in_variables", False)):
        idata_excludes.append(f"{prefix}_prev_latent_raw")
        if exclude_rw_raw_from_collection:
            collection_excludes.append(f"{prefix}_prev_latent_raw")
    if entity_obs_on:
        log.info(
            "entity_obs_raw_storage",
            n_artists=n_artists_fit,
            kept=not drop_entity_obs,
            keep_max=_ENTITY_OBS_KEEP_MAX,
        )
    fit_result = fit_model(
        model=make_score_model(prefix),
        model_args=model_args,
        config=mcmc_config,
        progress_bar=True,  # Always show MCMC progress for real-time feedback
        # Large tensors, exclude to prevent OOM
        exclude_from_idata=tuple(idata_excludes),
        # Opt-in in-sampler exclusion: never store these draws on device
        # (~96% peak-GPU cut at production settings; posterior parity for all
        # other sites is guarded by tests).
        exclude_from_collection=(tuple(collection_excludes) or None),
    )

    log.info(
        "model_fitted",
        divergences=fit_result.divergences,
        runtime_seconds=fit_result.runtime_seconds,
        gpu_info=fit_result.gpu_info,
        peak_gpu_memory_bytes=fit_result.peak_gpu_memory_bytes,
        tree_depth_saturation=fit_result.tree_depth_saturation,
    )

    # Check convergence using CLI-provided thresholds
    diagnostics = check_convergence(
        fit_result.idata,
        rhat_threshold=ctx.rhat_threshold,
        ess_threshold=ctx.ess_threshold,
        allow_divergences=ctx.allow_divergences,
    )

    log.info(
        "convergence_check",
        passed=diagnostics.passed,
        rhat_max=diagnostics.rhat_max,
        rhat_threshold=ctx.rhat_threshold,
        ess_bulk_min=diagnostics.ess_bulk_min,
        ess_threshold=ctx.ess_threshold,
        divergences=diagnostics.divergences,
        allow_divergences=ctx.allow_divergences,
    )

    # Handle strict mode
    # Note: allow_divergences is already passed to check_convergence above,
    # so diagnostics.passed accounts for it. But we need to check divergences
    # separately when strict=True and allow_divergences=False.
    if ctx.strict and fit_result.divergences > 0 and not ctx.allow_divergences:
        raise ConvergenceError(
            f"Model had {fit_result.divergences} divergent transitions. "
            "Re-run without --strict, use --allow-divergences, or increase --target-accept.",
            stage="train",
        )

    if ctx.strict and not diagnostics.passed:
        ess_thresh = ctx.ess_threshold * ctx.num_chains
        raise ConvergenceError(
            f"Convergence failed: rhat_max={diagnostics.rhat_max:.4f} "
            f"(thresh {ctx.rhat_threshold}), ess_min={diagnostics.ess_bulk_min:.0f} "
            f"(thresh {ess_thresh})",
            stage="train",
        )

    # Compute data hash for reproducibility
    data_hash = hash_dataframe(train_df)

    # Save model
    model_dir = Path("models")
    model_path, manifest = save_model(
        fit_result=fit_result,
        model_type=f"{prefix}_score",
        priors=priors,
        data_hash=data_hash,
        output_dir=model_dir,
    )

    log.info("model_saved", path=str(model_path))

    # Save training summary
    summary = {
        "model_type": f"{prefix}_score",
        "model_path": str(model_path),
        "mcmc_config": mcmc_config.to_dict(),
        "convergence_thresholds": {
            "rhat_threshold": ctx.rhat_threshold,
            "ess_threshold": ctx.ess_threshold,
            "allow_divergences": ctx.allow_divergences,
        },
        "min_albums_filter": ctx.min_albums_filter,
        "n_artists_below_threshold": int(n_below_threshold),
        "priors": asdict(priors),
        "data_hash": data_hash,
        "n_observations": len(model_args["y"]),
        "n_artists": model_args["n_artists"],
        "n_features": model_args["X"].shape[1],
        "feature_scaler": feature_scaler,
        "artist_to_idx": artist_to_idx,
        "max_seq": model_args["max_seq"],
        "max_albums": ctx.max_albums,
        "global_mean_score": float(global_mean_score),
        "global_std_score": float(global_std_score),
        "feature_cols": feature_cols,
        "n_exponent": ctx.n_exponent,
        "learn_n_exponent": ctx.learn_n_exponent,
        "n_exponent_prior": ctx.n_exponent_prior,
        "likelihood_df": getattr(ctx, "likelihood_df", 4.0),
        "debut_prev_score_source": debut_prev_score_source,
        "n_ref": model_args.get("n_ref"),
        "n_reviews_stats": {
            "min": int(np.min(model_args["n_reviews"])),
            "max": int(np.max(model_args["n_reviews"])),
            "median": int(np.median(model_args["n_reviews"])),
            "mean": float(np.mean(model_args["n_reviews"])),
        },
        "divergences": fit_result.divergences,
        "divergence_rate": float(
            fit_result.divergences / (mcmc_config.num_samples * mcmc_config.num_chains)
        ),
        "runtime_seconds": fit_result.runtime_seconds,
        # Preflight-validation telemetry: same counter the calibration
        # mini-runs measure, so projections are directly comparable.
        "peak_gpu_memory_bytes": fit_result.peak_gpu_memory_bytes,
        "gpu_info": fit_result.gpu_info,
        "exclude_rw_raw_from_collection": exclude_rw_raw_from_collection,
        "sigma_obs_prior_type": getattr(ctx, "sigma_obs_prior_type", "halfnormal"),
        "heteroscedastic_entity_obs": entity_obs_on,
        "tau_entity_scale": float(getattr(ctx, "tau_entity_scale", 0.25)),
        "diagnostics": {
            "passed": diagnostics.passed,
            "rhat_max": float(diagnostics.rhat_max),
            "ess_bulk_min": float(diagnostics.ess_bulk_min),
            "rhat_threshold": float(diagnostics.rhat_threshold),
            "ess_threshold": int(diagnostics.ess_threshold),
        },
    }

    # Log high divergence rate recommendation
    divergence_rate = fit_result.divergences / (mcmc_config.num_samples * mcmc_config.num_chains)
    if divergence_rate > 0.10:
        log.warning(
            "high_divergence_rate",
            divergence_rate=round(divergence_rate, 4),
            recommendation="Consider running grid search fallback with fixed n_exponent values",
        )

    summary["heteroscedastic_mode"] = _build_heteroscedastic_summary(
        ctx, fit_result, model_args, prefix, n_ref
    )

    # Validate through the typed contract: declared fields serialize in the
    # historical key order, with schema_version + dataset appended at the end.
    summary["dataset"] = descriptor.to_summary_block()
    summary["target_transform"] = target_transform
    summary["logit_offset"] = logit_offset
    # Raw-scale centering value; consumers re-apply the target transform.
    # The mode that produced it lives in priors.ar_center.
    summary["ar_center_value"] = ar_center_value
    summary["likelihood_family"] = str(getattr(ctx, "likelihood_family", "studentt"))
    summary["discretize_observation"] = bool(getattr(ctx, "discretize_observation", False))
    summary["entity_group_pooling"] = entity_group_pooling
    if entity_group_pooling:
        summary["entity_group_col"] = descriptor.entity_group_col
        summary["group_to_idx"] = group_to_idx
        summary["group_idx_by_artist"] = [int(g) for g in model_args["group_idx_by_artist"]]
        summary["n_groups"] = int(model_args["n_groups"])
    summary = TrainingSummary(**summary).to_json_dict()

    summary_path = model_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("train_pipeline_complete", summary_path=str(summary_path))

    return summary
