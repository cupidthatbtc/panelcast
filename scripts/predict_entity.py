#!/usr/bin/env python3
"""Look up any entity and predict its next event score.

Column names, the model prefix, and the grouping entity are read from
``models/training_summary.json`` (the trained dataset descriptor), so this runs
against whatever domain the model was trained on. The examples below use the
AOTY domain, where an entity is an artist and an event is an album.

Usage:
    pixi run -- python scripts/predict_entity.py "Kendrick Lamar"
    pixi run -- python scripts/predict_entity.py "Radiohead"
    pixi run -- python scripts/predict_entity.py  # interactive search
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
from numpyro.infer import Predictive

from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.models.bayes.model import make_score_model
from panelcast.models.bayes.predict import extract_posterior_samples, predict_new_entity
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.training_summary import ar_center_on_model_scale

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def descriptor_cols(summary: dict) -> dict:
    """Read the dataset descriptor's column names and model prefix from summary."""
    ds = summary.get("dataset") or {}
    return {
        "entity": ds.get("entity_col", "Artist"),
        "event": ds.get("event_col", "Album"),
        "target": ds.get("target_col", "User_Score"),
        "n_obs": ds.get("n_obs_col", "User_Ratings"),
        "prefix": ds.get("model_prefix", "user"),
    }


def _pick_col(df: pd.DataFrame, *names: str) -> str | None:
    """First of ``names`` present in ``df``; raw dumps use spaced display names."""
    for name in names:
        if name and name in df.columns:
            return name
    return None


def load_everything():
    """Load model, summary, and data once."""
    manifest = load_manifest(Path("models"))
    summary_path = Path("models") / "training_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    cols = descriptor_cols(summary)
    idata = load_model(Path("models") / manifest.current[f"{cols['prefix']}_score"])
    posterior_samples = extract_posterior_samples(idata)

    split_dir = resolve_split_dir(Path("data/splits"), SplitType.WITHIN_ENTITY_TEMPORAL)
    train = pd.read_parquet(split_dir / "train.parquet")
    train_feat = pd.read_parquet("data/features/train_features.parquet")
    overlap = list(set(train.columns) & set(train_feat.columns))
    if overlap:
        train = train.drop(columns=overlap)
    train = train.join(train_feat, how="left")
    feature_cols = summary["feature_cols"]
    train[feature_cols] = train[feature_cols].fillna(0)

    test = pd.read_parquet(split_dir / "test.parquet")

    # Load raw data for events not in train/test
    raw = pd.read_csv("data/raw/all_albums_full.csv", encoding="utf-8-sig")

    # Evaluation predictions for accuracy check
    eval_preds = None
    eval_path = Path("outputs/evaluation/within_entity_temporal/predictions.json")
    if eval_path.exists():
        with open(eval_path) as f:
            eval_preds = json.load(f)

    return summary, posterior_samples, train, test, raw, eval_preds, feature_cols


def find_entity(query: str, raw: pd.DataFrame, entity_col: str) -> str | None:
    """Fuzzy-find an entity name. Returns exact name or None."""
    # Exact match first
    all_entities = set(raw[entity_col].dropna().unique())
    if query in all_entities:
        return query

    # Case-insensitive
    lower_map = {a.lower(): a for a in all_entities}
    if query.lower() in lower_map:
        return lower_map[query.lower()]

    # Substring match
    matches = [a for a in all_entities if query.lower() in a.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"\nMultiple matches for '{query}':")
        for i, m in enumerate(sorted(matches)[:20], 1):
            print(f"  {i}. {m}")
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more")
        try:
            choice = input("\nPick a number (or press Enter to cancel): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                return sorted(matches)[int(choice) - 1]
        except (EOFError, KeyboardInterrupt):
            pass
        return None

    print(f"No {entity_col} found matching '{query}'")
    return None


def predict_known_entity(
    entity: str,
    summary: dict,
    posterior_samples: dict,
    train: pd.DataFrame,
    feature_cols: list[str],
    seq: int,
    prev_score: float,
) -> np.ndarray:
    """Run prediction for a known entity at given sequence/prev_score."""
    cols = descriptor_cols(summary)
    ds = summary.get("dataset") or {}
    target_bounds = tuple(ds.get("target_bounds", (0.0, 100.0)))
    ar_center = ar_center_on_model_scale(summary)
    transform = get_transform(
        summary.get("target_transform") or "identity",
        target_bounds=target_bounds,
        offset=float(summary.get("logit_offset") or 0.5),
    )
    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)
    priors = PriorConfig(**summary["priors"])
    trained_df = summary.get("likelihood_df", 1000.0)

    entity_train = train[train[cols["entity"]] == entity]
    last_features = entity_train.iloc[-1][feature_cols].values.astype(np.float32)
    X_scaled = ((last_features - X_mean) / X_std).astype(np.float32)

    n_reviews_col = _pick_col(entity_train, cols["n_obs"])
    n_reviews = int(entity_train[n_reviews_col].median()) if n_reviews_col else 100

    # prev_score enters the AR(1) term on the model scale, same as training.
    prev = float(prev_score)
    if transform.name != "identity":
        prev = float(np.asarray(transform.forward(prev)))

    model_args = {
        "artist_idx": np.array([summary["artist_to_idx"][entity]], dtype=np.int32),
        "album_seq": np.array([min(seq, summary["max_seq"])], dtype=np.int32),
        "prev_score": np.array([prev], dtype=np.float32),
        "X": X_scaled.reshape(1, -1),
        "y": None,
        "n_reviews": np.array([n_reviews], dtype=np.int32),
        "n_artists": summary["n_artists"],
        "max_seq": summary["max_seq"],
        "n_exponent": summary.get("n_exponent", 0.0),
        "learn_n_exponent": summary.get("learn_n_exponent", False),
        "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
        "n_ref": summary.get("n_ref"),
        "likelihood_df": trained_df,
        "priors": priors,
        "target_bounds": target_bounds,
        "ar_center": ar_center,
    }

    n_samples = next(iter(posterior_samples.values())).shape[0]
    batch_size = 500
    y_chunks: list[np.ndarray] = []

    model = make_score_model(cols["prefix"])
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        first_ps = {k: v[:batch_size] for k, v in posterior_samples.items()}
        predictive = Predictive(model, posterior_samples=first_ps, batch_ndims=1)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            predictive.posterior_samples = {k: v[start:end] for k, v in posterior_samples.items()}
            preds = predictive(random.key(42 + start), **model_args)
            y_key = next(k for k in preds if k.endswith("_y"))
            y_chunks.append(np.asarray(preds[y_key]).ravel())

    y = np.concatenate(y_chunks)
    if transform.name != "identity":
        y = np.asarray(transform.inverse(y))
    return y


def predict_new_entity_score(
    summary: dict,
    posterior_samples: dict,
    prev_score: float,
) -> np.ndarray:
    """Predict for an entity not in training data."""
    cols = descriptor_cols(summary)
    ds = summary.get("dataset") or {}
    target_bounds = tuple(ds.get("target_bounds", (0.0, 100.0)))
    target_transform = summary.get("target_transform") or "identity"
    logit_offset = float(summary.get("logit_offset") or 0.5)
    n_features = len(summary["feature_cols"])
    X_new = jnp.zeros(n_features)

    learn_n = summary.get("learn_n_exponent", False)
    n_exp = summary.get("n_exponent", 0.0)
    has_hetero = learn_n or n_exp != 0.0
    likelihood_family = (
        summary.get("likelihood_family")
        or summary.get("priors", {}).get("likelihood_family")
        or "studentt"
    )

    prev = float(prev_score)
    if target_transform != "identity":
        prev = float(np.asarray(get_transform(target_transform, target_bounds, logit_offset).forward(prev)))

    kwargs: dict = {
        "posterior_samples": {k: jnp.asarray(v) for k, v in posterior_samples.items()},
        "X_new": X_new,
        "prev_score": prev,
        "prefix": f"{cols['prefix']}_",
        "seed": 42,
        "target_bounds": target_bounds,
        "likelihood_df": float(summary.get("likelihood_df", 4.0)),
        "target_transform": target_transform,
        "logit_offset": logit_offset,
        "ar_center": ar_center_on_model_scale(summary),
        "likelihood_family": likelihood_family,
        "discretize_observation": bool(summary.get("discretize_observation")),
    }
    if has_hetero or likelihood_family == "beta_binomial":
        kwargs["n_reviews_new"] = jnp.array([summary["n_reviews_stats"]["median"]])
        if not learn_n and n_exp != 0.0:
            kwargs["fixed_n_exponent"] = n_exp

    pred = predict_new_entity(**kwargs)
    return np.asarray(pred["y"])


def format_prediction(samples: np.ndarray) -> str:
    """Format prediction samples into a summary string."""
    return (
        f"  Mean:    {np.mean(samples):.1f}\n"
        f"  Median:  {np.median(samples):.1f}\n"
        f"  90% PI:  [{np.percentile(samples, 5):.1f}, {np.percentile(samples, 95):.1f}]\n"
        f"  IQR:     [{np.percentile(samples, 25):.1f}, {np.percentile(samples, 75):.1f}]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def winsorize_prev_score(
    prev_score: float,
    entity_scores: list[float],
    clip_sigma: float = 1.5,
) -> tuple[float, bool]:
    """Winsorize prev_score to within clip_sigma std of the entity mean.

    Returns (clipped_score, was_clipped).
    """
    if len(entity_scores) < 2 or clip_sigma <= 0:
        return prev_score, False
    mean = float(np.mean(entity_scores))
    std = float(np.std(entity_scores))
    if std < 1.0:
        std = 10.0  # conservative default for very consistent entities
    lower = mean - clip_sigma * std
    upper = mean + clip_sigma * std
    clipped = float(np.clip(prev_score, lower, upper))
    return clipped, abs(clipped - prev_score) > 0.1


def predict_for_entity(
    entity: str,
    summary,
    posterior_samples,
    train,
    test,
    raw,
    eval_preds,
    feature_cols,
    robust: float = 0.0,
):
    """Run full analysis for one entity.

    Args:
        robust: If > 0, winsorize prev_score at this many std from the entity mean.
            E.g. robust=1.5 clips outlier prev_scores to within 1.5σ.
    """
    cols = descriptor_cols(summary)
    entity_col = cols["entity"]
    event_col = cols["event"]
    is_known = entity in summary["artist_to_idx"]

    # Raw display columns: processed parquets use canonical (underscored) names,
    # the raw dump uses spaced display names — accept either.
    raw_target = _pick_col(raw, cols["target"], cols["target"].replace("_", " "))
    raw_nobs = _pick_col(raw, cols["n_obs"], cols["n_obs"].replace("_", " "))

    # Gather all events from raw data
    entity_raw = raw[raw[entity_col] == entity].copy()
    if "Year" in entity_raw.columns:
        entity_raw = entity_raw.sort_values("Year")

    entity_train = (
        train[train[entity_col] == entity].sort_values("Year")
        if "Year" in train.columns
        else train[train[entity_col] == entity]
    )
    entity_test = (
        test[test[entity_col] == entity].sort_values("Year")
        if "Year" in test.columns
        else test[test[entity_col] == entity]
    )

    # Build event timeline
    print(f"\n{'=' * 60}")
    print(f"  {entity}")
    print(f"{'=' * 60}")
    print(f"\n  Known to model: {'Yes' if is_known else 'No (new entity)'}")
    print(f"  Training events: {len(entity_train)}")
    print(f"  Test events: {len(entity_test)}")
    print(f"  Total in dataset: {len(entity_raw)}")

    # Show history
    print("\n  HISTORY")
    print(f"  {'Year':<6} {'Score':>5}  {'Ratings':>8}  {'Split':<8}  {event_col}")
    print(f"  {'-' * 70}")

    for _, r in entity_raw.iterrows():
        year = f"{r['Year']:.0f}" if pd.notna(r.get("Year")) else "????"
        score = f"{r[raw_target]:.0f}" if raw_target and pd.notna(r.get(raw_target)) else "  --"
        ratings = f"{int(r[raw_nobs]):,}" if raw_nobs and pd.notna(r.get(raw_nobs)) else "--"
        event = r.get(event_col, "?")

        # Determine split
        in_train = (
            len(entity_train[entity_train[event_col] == event]) > 0
            if event_col in entity_train.columns
            else False
        )
        in_test = (
            len(entity_test[entity_test[event_col] == event]) > 0
            if event_col in entity_test.columns
            else False
        )
        split = "train" if in_train else ("test" if in_test else "---")

        print(f"  {year:<6} {score:>5}  {ratings:>8}  {split:<8}  {event}")

    if not is_known:
        # New entity prediction
        print("\n  PREDICTION (new entity — population distribution)")
        global_mean = summary["global_mean_score"]
        samples = predict_new_entity_score(summary, posterior_samples, prev_score=global_mean)
        print(format_prediction(samples))
        return

    # Sequential predictions through test events and beyond
    robust_label = f" [robust {robust:.1f}σ]" if robust > 0 else ""
    print(f"\n  SEQUENTIAL PREDICTIONS (within-entity){robust_label}")

    # Build the score chain: training scores, then test scores, then predict next
    train_scores = entity_train[cols["target"]].tolist()
    all_known_scores = list(train_scores)  # accumulates for winsorization
    test_events = []
    for _, r in entity_test.iterrows():
        test_events.append(
            {
                "event": r.get(event_col, "?"),
                "year": r.get("Year", None),
                "actual": float(r[cols["target"]]),
            }
        )

    n_train = len(train_scores)
    prev_score = train_scores[-1] if train_scores else summary["global_mean_score"]

    # Predict each test event sequentially
    errors = []
    for i, te in enumerate(test_events):
        seq = n_train + i + 1

        # Optionally winsorize prev_score
        use_prev = prev_score
        clipped_note = ""
        if robust > 0:
            use_prev, was_clipped = winsorize_prev_score(prev_score, all_known_scores, robust)
            if was_clipped:
                clipped_note = f" (clipped from {prev_score:.0f})"

        samples = predict_known_entity(
            entity,
            summary,
            posterior_samples,
            train,
            feature_cols,
            seq=seq,
            prev_score=use_prev,
        )
        pred_mean = float(np.mean(samples))
        error = te["actual"] - pred_mean
        errors.append(abs(error))

        year_str = f"({int(te['year'])})" if te["year"] else ""
        print(f"\n  seq {seq}: {te['event']} {year_str}")
        print(f"    prev_score = {use_prev:.0f}{clipped_note}")
        print(format_prediction(samples))
        print(f"    Actual:  {te['actual']:.0f}")
        sign = "+" if error > 0 else ""
        print(f"    Error:   {sign}{error:.1f}")

        # Feed actual score forward (sequential)
        prev_score = te["actual"]
        all_known_scores.append(te["actual"])

    # Predict NEXT event (not in dataset)
    next_seq = n_train + len(test_events) + 1

    # Winsorize for next prediction too
    use_prev = prev_score
    clipped_note = ""
    if robust > 0:
        use_prev, was_clipped = winsorize_prev_score(prev_score, all_known_scores, robust)
        if was_clipped:
            clipped_note = f" (clipped from {prev_score:.0f})"

    samples = predict_known_entity(
        entity,
        summary,
        posterior_samples,
        train,
        feature_cols,
        seq=next_seq,
        prev_score=use_prev,
    )

    # Find events in raw that aren't in train or test (future/excluded)
    train_events_set = (
        set(entity_train[event_col].tolist()) if event_col in entity_train.columns else set()
    )
    test_events_set = (
        set(entity_test[event_col].tolist()) if event_col in entity_test.columns else set()
    )
    future_events = entity_raw[
        ~entity_raw[event_col].isin(train_events_set | test_events_set)
        & (entity_raw[raw_target].isna() if raw_target else True)
    ]
    if not future_events.empty:
        next_name = future_events.iloc[0][event_col]
    elif raw_target:
        # Check for scored events not in splits
        excluded = entity_raw[
            ~entity_raw[event_col].isin(train_events_set | test_events_set)
            & entity_raw[raw_target].notna()
        ]
        next_name = excluded.iloc[0][event_col] if not excluded.empty else f"Next {event_col}"
    else:
        next_name = f"Next {event_col}"

    print(f"\n  seq {next_seq}: {next_name} (PREDICTION)")
    print(f"    prev_score = {use_prev:.0f}{clipped_note}")
    print(format_prediction(samples))

    # Accuracy summary
    if errors:
        print("\n  ACCURACY ON HELD-OUT EVENTS")
        print(f"    MAE:  {np.mean(errors):.1f}")
        if len(errors) > 1:
            print(f"    RMSE: {np.sqrt(np.mean(np.square(errors))):.1f}")
        print("    (model overall MAE: 5.9, RMSE: 8.6)")


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def main():
    # Parse --robust flag
    robust = 0.0
    args = list(sys.argv[1:])
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--robust":
            # Sigma is optional; only consume the next token if it's numeric, so
            # `--robust "Some Entity"` doesn't crash on float("Some Entity").
            if i + 1 < len(args) and _is_float(args[i + 1]):
                robust = float(args[i + 1])
                i += 2
            else:
                robust = 1.5  # default clip sigma
                i += 1
        else:
            filtered_args.append(args[i])
            i += 1

    print("Loading model and data...", end="", flush=True)
    summary, posterior_samples, train, test, raw, eval_preds, feature_cols = load_everything()
    print(" done.")
    cols = descriptor_cols(summary)

    mcmc = summary.get("mcmc_config", {})
    print(
        f"\nModel: {mcmc.get('num_chains', 1)} chain, {mcmc.get('num_warmup', 500)} warmup, "
        f"{mcmc.get('num_samples', 500)} samples"
    )
    print(
        f"Trained on {summary['n_observations']:,} events from "
        f"{summary['n_artists']:,} {cols['entity']}s"
    )
    if robust > 0:
        print(f"Robust mode: winsorizing prev_score at {robust:.1f}σ from the entity mean")

    if filtered_args:
        query = " ".join(filtered_args)
        entity = find_entity(query, raw, cols["entity"])
        if entity:
            predict_for_entity(
                entity,
                summary,
                posterior_samples,
                train,
                test,
                raw,
                eval_preds,
                feature_cols,
                robust=robust,
            )
    else:
        # Interactive mode
        print(f"\nType a {cols['entity']} name (or 'q' to quit):\n")
        while True:
            try:
                query = input(f"{cols['entity']}> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() == "q":
                break
            entity = find_entity(query, raw, cols["entity"])
            if entity:
                predict_for_entity(
                    entity,
                    summary,
                    posterior_samples,
                    train,
                    test,
                    raw,
                    eval_preds,
                    feature_cols,
                    robust=robust,
                )
            print()


if __name__ == "__main__":
    main()
