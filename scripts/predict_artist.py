#!/usr/bin/env python3
"""Look up any artist and predict their next album score.

Usage:
    pixi run -- python scripts/predict_artist.py "Kendrick Lamar"
    pixi run -- python scripts/predict_artist.py "Radiohead"
    pixi run -- python scripts/predict_artist.py  # interactive search
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

from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.predict import extract_posterior_samples, predict_new_artist
from panelcast.models.bayes.priors import PriorConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_everything():
    """Load model, summary, and data once."""
    manifest = load_manifest(Path("models"))
    idata = load_model(Path("models") / manifest.current["user_score"])
    with open("models/training_summary.json") as f:
        summary = json.load(f)
    posterior_samples = extract_posterior_samples(idata)

    train = pd.read_parquet("data/splits/within_artist_temporal/train.parquet")
    train_feat = pd.read_parquet("data/features/train_features.parquet")
    overlap = list(set(train.columns) & set(train_feat.columns))
    if overlap:
        train = train.drop(columns=overlap)
    train = train.join(train_feat, how="left")
    feature_cols = summary["feature_cols"]
    train[feature_cols] = train[feature_cols].fillna(0)

    test = pd.read_parquet("data/splits/within_artist_temporal/test.parquet")

    # Load raw data for albums not in train/test
    raw = pd.read_csv("data/raw/all_albums_full.csv", encoding="utf-8-sig")

    # Evaluation predictions for accuracy check
    eval_preds = None
    eval_path = Path("outputs/evaluation/within_artist_temporal/predictions.json")
    if eval_path.exists():
        with open(eval_path) as f:
            eval_preds = json.load(f)

    return summary, posterior_samples, train, test, raw, eval_preds, feature_cols


def find_artist(query: str, train: pd.DataFrame, raw: pd.DataFrame) -> str | None:
    """Fuzzy-find an artist name. Returns exact name or None."""
    # Exact match first
    all_artists = set(raw["Artist"].dropna().unique())
    if query in all_artists:
        return query

    # Case-insensitive
    lower_map = {a.lower(): a for a in all_artists}
    if query.lower() in lower_map:
        return lower_map[query.lower()]

    # Substring match
    matches = [a for a in all_artists if query.lower() in a.lower()]
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

    print(f"No artist found matching '{query}'")
    return None


def predict_known_artist(
    artist: str,
    summary: dict,
    posterior_samples: dict,
    train: pd.DataFrame,
    feature_cols: list[str],
    seq: int,
    prev_score: float,
) -> np.ndarray:
    """Run prediction for a known artist at given sequence/prev_score."""
    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)
    priors = PriorConfig(**summary["priors"])
    trained_df = summary.get("likelihood_df", 1000.0)

    artist_train = train[train["Artist"] == artist]
    last_features = artist_train.iloc[-1][feature_cols].values.astype(np.float32)
    X_scaled = ((last_features - X_mean) / X_std).astype(np.float32)

    n_reviews_col = "User_Ratings" if "User_Ratings" in artist_train.columns else None
    n_reviews = int(artist_train[n_reviews_col].median()) if n_reviews_col else 100

    model_args = {
        "artist_idx": np.array([summary["artist_to_idx"][artist]], dtype=np.int32),
        "album_seq": np.array([min(seq, summary["max_seq"])], dtype=np.int32),
        "prev_score": np.array([prev_score], dtype=np.float32),
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
    }

    n_samples = next(iter(posterior_samples.values())).shape[0]
    batch_size = 500
    y_chunks: list[np.ndarray] = []

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        first_ps = {k: v[:batch_size] for k, v in posterior_samples.items()}
        predictive = Predictive(user_score_model, posterior_samples=first_ps, batch_ndims=1)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            predictive.posterior_samples = {k: v[start:end] for k, v in posterior_samples.items()}
            preds = predictive(random.key(42 + start), **model_args)
            y_key = next(k for k in preds if k.endswith("_y"))
            y_chunks.append(np.asarray(preds[y_key]).ravel())

    return np.concatenate(y_chunks)


def predict_new_artist_score(
    summary: dict,
    posterior_samples: dict,
    prev_score: float,
) -> np.ndarray:
    """Predict for an artist not in training data."""
    n_features = len(summary["feature_cols"])
    X_new = jnp.zeros(n_features)

    learn_n = summary.get("learn_n_exponent", False)
    n_exp = summary.get("n_exponent", 0.0)
    has_hetero = learn_n or n_exp != 0.0

    kwargs: dict = {
        "posterior_samples": {k: jnp.asarray(v) for k, v in posterior_samples.items()},
        "X_new": X_new,
        "prev_score": prev_score,
        "prefix": "user_",
        "seed": 42,
    }
    if has_hetero:
        kwargs["n_reviews_new"] = jnp.array([summary["n_reviews_stats"]["median"]])
        if not learn_n and n_exp != 0.0:
            kwargs["fixed_n_exponent"] = n_exp

    pred = predict_new_artist(**kwargs)
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
    artist_scores: list[float],
    clip_sigma: float = 1.5,
) -> tuple[float, bool]:
    """Winsorize prev_score to within clip_sigma std of artist mean.

    Returns (clipped_score, was_clipped).
    """
    if len(artist_scores) < 2 or clip_sigma <= 0:
        return prev_score, False
    mean = float(np.mean(artist_scores))
    std = float(np.std(artist_scores))
    if std < 1.0:
        std = 10.0  # conservative default for very consistent artists
    lower = mean - clip_sigma * std
    upper = mean + clip_sigma * std
    clipped = float(np.clip(prev_score, lower, upper))
    return clipped, abs(clipped - prev_score) > 0.1


def predict_for_artist(
    artist: str,
    summary,
    posterior_samples,
    train,
    test,
    raw,
    eval_preds,
    feature_cols,
    robust: float = 0.0,
):
    """Run full analysis for one artist.

    Args:
        robust: If > 0, winsorize prev_score at this many std from artist mean.
            E.g. robust=1.5 clips outlier prev_scores to within 1.5σ.
    """
    is_known = artist in summary["artist_to_idx"]

    # Gather all albums from raw data
    artist_raw = raw[raw["Artist"] == artist].copy()
    if "Year" in artist_raw.columns:
        artist_raw = artist_raw.sort_values("Year")

    artist_train = (
        train[train["Artist"] == artist].sort_values("Year")
        if "Year" in train.columns
        else train[train["Artist"] == artist]
    )
    artist_test = (
        test[test["Artist"] == artist].sort_values("Year")
        if "Year" in test.columns
        else test[test["Artist"] == artist]
    )

    # Build album timeline
    print(f"\n{'=' * 60}")
    print(f"  {artist}")
    print(f"{'=' * 60}")
    print(f"\n  Known to model: {'Yes' if is_known else 'No (new artist)'}")
    print(f"  Training albums: {len(artist_train)}")
    print(f"  Test albums: {len(artist_test)}")
    print(f"  Total in dataset: {len(artist_raw)}")

    # Show discography
    print("\n  DISCOGRAPHY")
    print(f"  {'Year':<6} {'Score':>5}  {'Ratings':>8}  {'Split':<8}  Album")
    print(f"  {'-' * 70}")

    for _, r in artist_raw.iterrows():
        year = f"{r['Year']:.0f}" if pd.notna(r.get("Year")) else "????"
        score = f"{r['User Score']:.0f}" if pd.notna(r.get("User Score")) else "  --"
        ratings = f"{int(r['User Ratings']):,}" if pd.notna(r.get("User Ratings")) else "--"
        album = r.get("Album", "?")

        # Determine split
        in_train = (
            len(artist_train[artist_train["Album"] == album]) > 0
            if "Album" in artist_train.columns
            else False
        )
        in_test = (
            len(artist_test[artist_test["Album"] == album]) > 0
            if "Album" in artist_test.columns
            else False
        )
        split = "train" if in_train else ("test" if in_test else "---")

        print(f"  {year:<6} {score:>5}  {ratings:>8}  {split:<8}  {album}")

    if not is_known:
        # New artist prediction
        print("\n  PREDICTION (new artist — population distribution)")
        global_mean = summary["global_mean_score"]
        samples = predict_new_artist_score(summary, posterior_samples, prev_score=global_mean)
        print(format_prediction(samples))
        return

    # Sequential predictions through test albums and beyond
    robust_label = f" [robust {robust:.1f}σ]" if robust > 0 else ""
    print(f"\n  SEQUENTIAL PREDICTIONS (within-artist){robust_label}")

    # Build the score chain: training scores, then test scores, then predict next
    train_scores = artist_train["User_Score"].tolist()
    all_known_scores = list(train_scores)  # accumulates for winsorization
    test_albums = []
    for _, r in artist_test.iterrows():
        test_albums.append(
            {
                "album": r.get("Album", "?"),
                "year": r.get("Year", None),
                "actual": float(r["User_Score"]),
            }
        )

    n_train = len(train_scores)
    prev_score = train_scores[-1] if train_scores else summary["global_mean_score"]

    # Predict each test album sequentially
    errors = []
    for i, ta in enumerate(test_albums):
        seq = n_train + i + 1

        # Optionally winsorize prev_score
        use_prev = prev_score
        clipped_note = ""
        if robust > 0:
            use_prev, was_clipped = winsorize_prev_score(prev_score, all_known_scores, robust)
            if was_clipped:
                clipped_note = f" (clipped from {prev_score:.0f})"

        samples = predict_known_artist(
            artist,
            summary,
            posterior_samples,
            train,
            feature_cols,
            seq=seq,
            prev_score=use_prev,
        )
        pred_mean = float(np.mean(samples))
        error = ta["actual"] - pred_mean
        errors.append(abs(error))

        year_str = f"({int(ta['year'])})" if ta["year"] else ""
        print(f"\n  seq {seq}: {ta['album']} {year_str}")
        print(f"    prev_score = {use_prev:.0f}{clipped_note}")
        print(format_prediction(samples))
        print(f"    Actual:  {ta['actual']:.0f}")
        sign = "+" if error > 0 else ""
        print(f"    Error:   {sign}{error:.1f}")

        # Feed actual score forward (sequential)
        prev_score = ta["actual"]
        all_known_scores.append(ta["actual"])

    # Predict NEXT album (not in dataset)
    next_seq = n_train + len(test_albums) + 1

    # Winsorize for next prediction too
    use_prev = prev_score
    clipped_note = ""
    if robust > 0:
        use_prev, was_clipped = winsorize_prev_score(prev_score, all_known_scores, robust)
        if was_clipped:
            clipped_note = f" (clipped from {prev_score:.0f})"

    samples = predict_known_artist(
        artist,
        summary,
        posterior_samples,
        train,
        feature_cols,
        seq=next_seq,
        prev_score=use_prev,
    )

    # Find albums in raw that aren't in train or test (future/excluded)
    train_albums_set = (
        set(artist_train["Album"].tolist()) if "Album" in artist_train.columns else set()
    )
    test_albums_set = (
        set(artist_test["Album"].tolist()) if "Album" in artist_test.columns else set()
    )
    future_albums = artist_raw[
        ~artist_raw["Album"].isin(train_albums_set | test_albums_set)
        & artist_raw["User Score"].isna()
    ]
    if not future_albums.empty:
        next_name = future_albums.iloc[0]["Album"]
    else:
        # Check for scored albums not in splits
        excluded = artist_raw[
            ~artist_raw["Album"].isin(train_albums_set | test_albums_set)
            & artist_raw["User Score"].notna()
        ]
        if not excluded.empty:
            next_name = excluded.iloc[0]["Album"]
        else:
            next_name = "Next Album"

    print(f"\n  seq {next_seq}: {next_name} (PREDICTION)")
    print(f"    prev_score = {use_prev:.0f}{clipped_note}")
    print(format_prediction(samples))

    # Accuracy summary
    if errors:
        print("\n  ACCURACY ON HELD-OUT ALBUMS")
        print(f"    MAE:  {np.mean(errors):.1f}")
        if len(errors) > 1:
            print(f"    RMSE: {np.sqrt(np.mean(np.square(errors))):.1f}")
        print("    (model overall MAE: 5.9, RMSE: 8.6)")


def main():
    # Parse --robust flag
    robust = 0.0
    args = list(sys.argv[1:])
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--robust":
            if i + 1 < len(args):
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

    mcmc = summary.get("mcmc_config", {})
    print(
        f"\nModel: {mcmc.get('num_chains', 1)} chain, {mcmc.get('num_warmup', 500)} warmup, "
        f"{mcmc.get('num_samples', 500)} samples"
    )
    print(f"Trained on {summary['n_observations']:,} albums from {summary['n_artists']:,} artists")
    if robust > 0:
        print(f"Robust mode: winsorizing prev_score at {robust:.1f}σ from artist mean")

    if filtered_args:
        query = " ".join(filtered_args)
        artist = find_artist(query, train, raw)
        if artist:
            predict_for_artist(
                artist,
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
        print("\nType an artist name (or 'q' to quit):\n")
        while True:
            try:
                query = input("Artist> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() == "q":
                break
            artist = find_artist(query, train, raw)
            if artist:
                predict_for_artist(
                    artist,
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
