#!/usr/bin/env python3
"""Generate publication-quality artist trajectory plots.

Usage:
    pixi run -- python scripts/generate_trajectories.py "Björk"
    pixi run -- python scripts/generate_trajectories.py "Kendrick Lamar" "Kanye West" "Radiohead"
    pixi run -- python scripts/generate_trajectories.py --top 20  # top 20 most prolific
    pixi run -- python scripts/generate_trajectories.py "Björk" --critic  # include critic scores
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import random
from numpyro.infer import Predictive
from sklearn.linear_model import Ridge

from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.predict import extract_posterior_samples
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.training_summary import ar_center_on_model_scale

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {
    "train": "#1565C0",
    "test_actual": "#2E7D32",
    "prediction": "#E65100",
    "ci_fill": "#FF8A65",
    "smooth": "#90CAF9",
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
    "text": "#212121",
    "annotation": "#616161",
}


def _setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": "white",
            "axes.edgecolor": "#BDBDBD",
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )


# ---------------------------------------------------------------------------
# Data loading (shared across all artists)
# ---------------------------------------------------------------------------

_CACHE = {}


def _find_artist(query: str, train: pd.DataFrame, raw: pd.DataFrame) -> str | None:
    """Find an artist by name (exact, case-insensitive, or substring)."""
    all_artists = set(raw["Artist"].dropna().unique())
    if query in all_artists:
        return query
    lower_map = {a.lower(): a for a in all_artists}
    if query.lower() in lower_map:
        return lower_map[query.lower()]
    matches = [a for a in all_artists if query.lower() in a.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Pick the one with the most albums in training data
        best = None
        best_count = -1
        for m in matches:
            count = len(train[train["Artist"] == m])
            if count > best_count:
                best_count = count
                best = m
        if best:
            print(f"  Matched '{query}' → '{best}' ({best_count} training albums)")
            return best
    return None


def _load_all():
    if _CACHE:
        return _CACHE

    manifest = load_manifest(Path("models"))
    idata = load_model(Path("models") / manifest.current["user_score"])
    with open("models/training_summary.json") as f:
        summary = json.load(f)
    posterior_samples = extract_posterior_samples(idata)

    train = pd.read_parquet("data/splits/within_entity_temporal/train.parquet")
    train_feat = pd.read_parquet("data/features/train_features.parquet")
    overlap = list(set(train.columns) & set(train_feat.columns))
    if overlap:
        train = train.drop(columns=overlap)
    train = train.join(train_feat, how="left")
    feature_cols = summary["feature_cols"]
    train[feature_cols] = train[feature_cols].fillna(0)

    test = pd.read_parquet("data/splits/within_entity_temporal/test.parquet")
    raw = pd.read_csv("data/raw/all_albums_full.csv", encoding="utf-8-sig")

    # Load validation split
    val = pd.DataFrame()
    for val_name in ("validation.parquet", "val.parquet"):
        val_path = Path("data/splits/within_entity_temporal") / val_name
        if val_path.exists():
            val = pd.read_parquet(val_path)
            break

    _CACHE.update(
        {
            "summary": summary,
            "posterior_samples": posterior_samples,
            "train": train,
            "val": val,
            "test": test,
            "raw": raw,
            "feature_cols": feature_cols,
        }
    )
    return _CACHE


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------


def _predict_at(
    artist: str,
    seq: int,
    prev_score: float,
    data: dict,
) -> np.ndarray:
    """Run prediction for one album position, return samples."""
    summary = data["summary"]
    posterior_samples = data["posterior_samples"]
    train = data["train"]
    feature_cols = data["feature_cols"]

    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)
    priors = PriorConfig(**summary["priors"])

    artist_train = train[train["Artist"] == artist]
    last_features = artist_train.iloc[-1][feature_cols].values.astype(np.float32)
    X_scaled = ((last_features - X_mean) / X_std).astype(np.float32)
    n_reviews = (
        int(artist_train["User_Ratings"].median())
        if "User_Ratings" in artist_train.columns
        else 100
    )

    # The shipped model runs on the target's transform scale (offset_logit), so
    # prev_score must be forwarded onto that scale, ar_center/target_bounds must
    # be passed, and the sampled draws inverse-transformed back to 0-100 — same
    # as predict_entity.predict_known_entity. (Was: raw prev, ar_center=0, and a
    # bare clip of logit-scale draws, which made every PI band nonsense.)
    ds = summary.get("dataset") or {}
    target_bounds = tuple(ds.get("target_bounds", (0.0, 100.0)))
    target_transform = summary.get("target_transform") or "identity"
    logit_offset = float(summary.get("logit_offset") or 0.5)
    transform = get_transform(target_transform, target_bounds, logit_offset)

    prev = float(prev_score)
    if target_transform != "identity":
        prev = float(np.asarray(transform.forward(prev)))

    model_args = {
        "artist_idx": np.array([summary["artist_to_idx"][artist]], dtype=np.int32),
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
        "likelihood_df": summary.get("likelihood_df", 1000.0),
        "priors": priors,
        "target_bounds": target_bounds,
        "ar_center": ar_center_on_model_scale(summary),
    }

    n_samples = next(iter(posterior_samples.values())).shape[0]
    batch = min(500, n_samples)
    first_ps = {k: v[:batch] for k, v in posterior_samples.items()}
    predictive = Predictive(user_score_model, posterior_samples=first_ps, batch_ndims=1)
    preds = predictive(random.key(42), **model_args)
    y_key = next(k for k in preds if k.endswith("_y"))
    y = np.asarray(preds[y_key]).ravel()
    if target_transform != "identity":
        y = np.asarray(transform.inverse(y))
    return y


# ---------------------------------------------------------------------------
# Frequentist prediction engine
# ---------------------------------------------------------------------------

_FREQ_CACHE = {}


def _fit_frequentist(data: dict) -> dict:
    """Fit a Ridge regression with artist dummies + prev_score for frequentist PIs."""
    if _FREQ_CACHE:
        return _FREQ_CACHE

    train = data["train"]
    feature_cols = data["feature_cols"]
    summary = data["summary"]
    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)

    # Sort by artist+year so prev_score shift is correct
    train_sorted = train.sort_values(["Artist", "Year"]).reset_index(drop=True)

    # Build design matrix: scaled features + prev_score
    X_features = train_sorted[feature_cols].fillna(0).values.astype(np.float32)
    X_scaled = (X_features - X_mean) / X_std

    # Compute prev_score per artist (shift User_Score within each artist group)
    global_mean = float(train_sorted["User_Score"].mean())
    prev_scores = train_sorted.groupby("Artist")["User_Score"].shift(1)
    prev_scores = prev_scores.fillna(global_mean).values.astype(np.float32).reshape(-1, 1)
    prev_scaled = (prev_scores - 70.0) / 15.0

    X_full = np.hstack([X_scaled, prev_scaled])
    y = train_sorted["User_Score"].values.astype(np.float32)

    model = Ridge(alpha=1.0)
    model.fit(X_full, y)

    residuals = y - model.predict(X_full)
    sigma_resid = float(np.std(residuals))

    _FREQ_CACHE.update(
        {
            "model": model,
            "residuals": residuals,
            "sigma_resid": sigma_resid,
            "X_mean": X_mean,
            "X_std": X_std,
        }
    )
    return _FREQ_CACHE


def _predict_freq_at(
    artist: str,
    prev_score: float,
    data: dict,
    n_bootstrap: int = 1000,
) -> np.ndarray:
    """Frequentist prediction with bootstrap PI for one album."""
    freq = _fit_frequentist(data)
    model = freq["model"]
    residuals = freq["residuals"]
    X_mean = freq["X_mean"]
    X_std = freq["X_std"]
    train = data["train"]
    feature_cols = data["feature_cols"]

    artist_train = train[train["Artist"] == artist]
    last_features = artist_train.iloc[-1][feature_cols].values.astype(np.float32)
    X_scaled = (last_features - X_mean) / X_std
    prev_scaled = (prev_score - 70.0) / 15.0

    x_row = np.hstack([X_scaled, [prev_scaled]]).reshape(1, -1)
    y_hat = model.predict(x_row)[0]

    # Bootstrap PI: y_hat + resampled residuals
    rng = np.random.default_rng(42)
    boot_residuals = rng.choice(residuals, size=n_bootstrap, replace=True)
    samples = np.clip(y_hat + boot_residuals, 0, 100)
    return samples


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------


def build_trajectory(
    artist: str, data: dict, freq: bool = False, albums_only: bool = False
) -> dict | None:
    """Build full trajectory data for an artist."""
    summary = data["summary"]
    train = data["train"]
    val = data.get("val", pd.DataFrame())
    test = data["test"]
    raw = data["raw"]

    if artist not in summary["artist_to_idx"]:
        return None

    # Always use full unfiltered data for prediction sequence
    artist_train_full = train[train["Artist"] == artist].sort_values("Year")
    artist_val_full = (
        val[val["Artist"] == artist].sort_values("Year") if not val.empty else pd.DataFrame()
    )
    artist_test_full = test[test["Artist"] == artist].sort_values("Year")
    artist_raw = raw[raw["Artist"] == artist].sort_values("Year")

    # For display: optionally filter to studio albums only
    if albums_only:

        def _filter(df):
            return df[df["Album_Type"] == "Album"] if "Album_Type" in df.columns else df

        artist_train_disp = _filter(artist_train_full)
        artist_val_disp = _filter(artist_val_full) if not artist_val_full.empty else artist_val_full
        artist_test_disp = _filter(artist_test_full)
    else:
        artist_train_disp = artist_train_full
        artist_val_disp = artist_val_full
        artist_test_disp = artist_test_full

    train_years = artist_train_disp["Year"].tolist()
    train_scores = artist_train_disp["User_Score"].tolist()
    train_albums = (
        artist_train_disp["Album"].tolist() if "Album" in artist_train_disp.columns else []
    )

    val_years = artist_val_disp["Year"].tolist() if not artist_val_disp.empty else []
    val_scores = artist_val_disp["User_Score"].tolist() if not artist_val_disp.empty else []
    val_albums = (
        artist_val_disp["Album"].tolist()
        if not artist_val_disp.empty and "Album" in artist_val_disp.columns
        else []
    )

    test_years = artist_test_disp["Year"].tolist()
    test_scores = artist_test_disp["User_Score"].tolist()
    test_albums = artist_test_disp["Album"].tolist() if "Album" in artist_test_disp.columns else []

    # Use FULL unfiltered counts for prediction sequence
    n_train = len(artist_train_full)
    full_train_scores = artist_train_full["User_Score"].tolist()
    full_val_scores = artist_val_full["User_Score"].tolist() if not artist_val_full.empty else []
    full_test_years = artist_test_full["Year"].tolist()
    full_test_scores = artist_test_full["User_Score"].tolist()
    prev_score = full_train_scores[-1] if full_train_scores else 70.0

    pred_years = []
    pred_means = []
    pred_q05 = []
    pred_q25 = []
    pred_q75 = []
    pred_q95 = []

    # Frequentist PI accumulators
    freq_means = []
    freq_q05 = []
    freq_q25 = []
    freq_q75 = []
    freq_q95 = []

    # Predict val albums (use val score as prev_score, matching eval pipeline)
    if full_val_scores:
        prev_score = full_val_scores[-1]

    freq_prev = prev_score

    # Predict test albums (always use full unfiltered test sequence)
    for i, (year, actual) in enumerate(zip(full_test_years, full_test_scores)):
        seq = n_train + len(full_val_scores) + i + 1
        samples = _predict_at(artist, seq, prev_score, data)
        pred_years.append(year)
        pred_means.append(float(np.mean(samples)))
        pred_q05.append(float(np.percentile(samples, 5)))
        pred_q25.append(float(np.percentile(samples, 25)))
        pred_q75.append(float(np.percentile(samples, 75)))
        pred_q95.append(float(np.percentile(samples, 95)))

        if freq:
            fsamp = _predict_freq_at(artist, freq_prev, data)
            freq_means.append(float(np.mean(fsamp)))
            freq_q05.append(float(np.percentile(fsamp, 5)))
            freq_q25.append(float(np.percentile(fsamp, 25)))
            freq_q75.append(float(np.percentile(fsamp, 75)))
            freq_q95.append(float(np.percentile(fsamp, 95)))

        prev_score = actual
        freq_prev = actual

    # Predict next album
    all_years = train_years + val_years + full_test_years
    next_year = max(all_years) + 2 if all_years else 2027
    seq = n_train + len(full_val_scores) + len(full_test_scores) + 1
    samples = _predict_at(artist, seq, prev_score, data)
    pred_years.append(next_year)
    pred_means.append(float(np.mean(samples)))
    pred_q05.append(float(np.percentile(samples, 5)))
    pred_q25.append(float(np.percentile(samples, 25)))
    pred_q75.append(float(np.percentile(samples, 75)))
    pred_q95.append(float(np.percentile(samples, 95)))

    if freq:
        fsamp = _predict_freq_at(artist, freq_prev, data)
        freq_means.append(float(np.mean(fsamp)))
        freq_q05.append(float(np.percentile(fsamp, 5)))
        freq_q25.append(float(np.percentile(fsamp, 25)))
        freq_q75.append(float(np.percentile(fsamp, 75)))
        freq_q95.append(float(np.percentile(fsamp, 95)))

    # Find next album name
    known_set = set(train_albums) | set(val_albums) | set(test_albums)
    future = artist_raw[~artist_raw["Album"].isin(known_set) & artist_raw["User Score"].isna()]
    next_name = future.iloc[0]["Album"] if not future.empty else "Next Album"

    # Critic scores from raw data (optional)
    critic_years = []
    critic_scores = []
    if "Critic Score" in artist_raw.columns:
        for _, r in artist_raw.iterrows():
            if pd.notna(r.get("Critic Score")) and pd.notna(r.get("Year")):
                critic_years.append(r["Year"])
                critic_scores.append(r["Critic Score"])

    return {
        "artist": artist,
        "train_years": train_years,
        "train_scores": train_scores,
        "train_albums": train_albums,
        "val_years": val_years,
        "val_scores": val_scores,
        "val_albums": val_albums,
        "critic_years": critic_years,
        "critic_scores": critic_scores,
        "test_years": test_years,
        "test_scores": test_scores,
        "test_albums": test_albums,
        "pred_years": pred_years,
        "pred_means": pred_means,
        "pred_q05": pred_q05,
        "pred_q25": pred_q25,
        "pred_q75": pred_q75,
        "pred_q95": pred_q95,
        "freq_means": freq_means,
        "freq_q05": freq_q05,
        "freq_q25": freq_q25,
        "freq_q75": freq_q75,
        "freq_q95": freq_q95,
        "next_name": next_name,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _trend_line(x, y, num_points=200):
    """Compute a LOWESS-style trend line (not interpolation)."""
    if len(x) < 4:
        return None, None
    try:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        # Use a low-degree polynomial as a simple trend
        # Degree scales with data: 2 for <8 points, 3 for 8-15, 4 for 15+
        deg = min(2 + len(x_arr) // 8, 4)
        coeffs = np.polyfit(x_arr, y_arr, deg)
        x_smooth = np.linspace(x_arr.min(), x_arr.max(), num_points)
        y_smooth = np.clip(np.polyval(coeffs, x_smooth), 0, 100)
        return x_smooth, y_smooth
    except Exception:
        return None, None


def plot_trajectory(
    traj: dict,
    output_dir: Path,
    show_critic: bool = False,
    show_freq: bool = False,
    compare: bool = False,
    guesses: list[dict] | None = None,
) -> Path:
    """Generate a publication-quality trajectory plot."""
    _setup_style()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Gentle trend line across all known albums (train + val + test)
    trend_years = traj["train_years"] + traj.get("val_years", []) + traj["test_years"]
    trend_scores = traj["train_scores"] + traj.get("val_scores", []) + traj["test_scores"]
    if len(trend_years) >= 4:
        xs, ys = _trend_line(trend_years, trend_scores)
        if xs is not None:
            ax.plot(
                xs,
                ys,
                color=COLORS["smooth"],
                linewidth=2.5,
                alpha=0.4,
                zorder=1,
            )

    # Training albums (dots + connecting line)
    ax.plot(
        traj["train_years"],
        traj["train_scores"],
        "o-",
        color=COLORS["train"],
        markersize=7,
        linewidth=1.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Training albums",
        zorder=3,
    )

    # Critic scores (optional)
    if show_critic and traj.get("critic_years") and len(traj["critic_years"]) >= 2:
        ax.plot(
            traj["critic_years"],
            traj["critic_scores"],
            "x-",
            color="#78909C",
            markersize=5,
            linewidth=1.0,
            alpha=0.45,
            label="Critic score",
            zorder=2,
        )

    # Validation albums (known scores, not trained on)
    if traj.get("val_years"):
        ax.plot(
            traj["val_years"],
            traj["val_scores"],
            "^",
            color="#9C27B0",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Validation (actual)",
            zorder=5,
        )

    # Test albums (actual scores)
    if traj["test_years"]:
        ax.plot(
            traj["test_years"],
            traj["test_scores"],
            "s",
            color=COLORS["test_actual"],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Held-out (actual)",
            zorder=5,
        )

    # Predictions with nested PI bands
    py = traj["pred_years"]
    has_freq = (compare or show_freq) and traj.get("freq_means")

    # Bayesian bands (skip if freq-only mode)
    if not show_freq:
        bayes_label = "Bayesian" if has_freq else "Prediction"
        pi_label_90 = "90% PI (Bayesian)" if has_freq else "90% PI"
        pi_label_50 = "50% PI (Bayesian)" if has_freq else "50% PI"
        ax.fill_between(
            py,
            traj["pred_q05"],
            traj["pred_q95"],
            alpha=0.10,
            color=COLORS["ci_fill"],
            label=pi_label_90,
            zorder=1,
        )
        ax.fill_between(
            py,
            traj["pred_q25"],
            traj["pred_q75"],
            alpha=0.20,
            color=COLORS["ci_fill"],
            label=pi_label_50,
            zorder=1,
        )
        ax.plot(
            py,
            traj["pred_means"],
            "D--",
            color=COLORS["prediction"],
            markersize=8,
            linewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=bayes_label,
            zorder=4,
        )

    # Frequentist bands
    if has_freq:
        freq_color = "#7B1FA2"  # purple
        freq_fill = "#CE93D8"
        pi_label_90f = "90% PI (Frequentist)"
        ax.fill_between(
            py,
            traj["freq_q05"],
            traj["freq_q95"],
            alpha=0.10,
            color=freq_fill,
            label=pi_label_90f,
            zorder=1,
        )
        ax.fill_between(
            py,
            traj["freq_q25"],
            traj["freq_q75"],
            alpha=0.20,
            color=freq_fill,
            label="50% PI (Frequentist)",
            zorder=1,
        )
        ax.plot(
            py,
            traj["freq_means"],
            "v--",
            color=freq_color,
            markersize=8,
            linewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label="Frequentist" if not show_freq else "Prediction",
            zorder=4,
        )

    # Album name annotations
    all_years = traj["train_years"] + traj.get("val_years", []) + traj["test_years"]
    all_scores = traj["train_scores"] + traj.get("val_scores", []) + traj["test_scores"]
    all_albums = traj["train_albums"] + traj.get("val_albums", []) + traj["test_albums"]

    if all_albums:
        # Alternate annotation position to avoid overlap
        for i, (yr, sc, alb) in enumerate(zip(all_years, all_scores, all_albums)):
            offset_y = 10 if i % 2 == 0 else -14
            ha = "left"
            alb_short = alb[:22] + ("…" if len(alb) > 22 else "")
            ax.annotate(
                alb_short,
                (yr, sc),
                textcoords="offset points",
                xytext=(4, offset_y),
                fontsize=6.5,
                color=COLORS["annotation"],
                alpha=0.85,
                ha=ha,
            )

    # Next album label
    next_label = traj["next_name"][:25]
    ax.annotate(
        f"→ {next_label}\n   pred: {traj['pred_means'][-1]:.0f}",
        (traj["pred_years"][-1], traj["pred_means"][-1]),
        textcoords="offset points",
        xytext=(8, -5),
        fontsize=8,
        fontweight="bold",
        color=COLORS["prediction"],
    )

    # Human guesses overlay
    guess_colors = ["#D32F2F", "#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#00838F"]
    if guesses:
        next_year = traj["pred_years"][-1]
        for gi, g in enumerate(guesses):
            color = guess_colors[gi % len(guess_colors)]
            name = g["Name"]
            score = float(g["Predicted Score"])
            lo = float(g["Low"])
            hi = float(g["High"])
            gtype = g.get("Type", "").strip()
            # Offset each guess slightly on the x-axis so they don't overlap
            x_offset = 0.3 * (gi - (len(guesses) - 1) / 2)
            x = next_year + x_offset
            ax.plot(
                x,
                score,
                "★",
                color=color,
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=f"{name}: {score:.0f} [{lo:.0f}–{hi:.0f}]" + (f" ({gtype})" if gtype else ""),
                zorder=6,
            )
            ax.vlines(x, lo, hi, color=color, linewidth=2.5, alpha=0.6, zorder=5)
            ax.hlines([lo, hi], x - 0.15, x + 0.15, color=color, linewidth=1.5, alpha=0.6, zorder=5)

    # Axes
    ax.set_ylim(0, 100)
    ax.set_ylabel("User Score")
    ax.set_xlabel("Year")

    ax.set_title(traj["artist"], fontsize=16, fontweight="bold", pad=15)

    # Grid
    ax.grid(True, alpha=0.3, color=COLORS["grid"], linestyle="-")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc="lower left",
        framealpha=0.9,
        edgecolor="#BDBDBD",
        fancybox=True,
    )

    # Stats box
    n_train = len(traj["train_years"])
    test_errors = []
    for i, (actual, pred) in enumerate(zip(traj["test_scores"], traj["pred_means"])):
        test_errors.append(abs(actual - pred))
    mae_text = f"MAE: {np.mean(test_errors):.1f}" if test_errors else "No test data"

    stats_text = f"Train: {n_train} albums | {mae_text}"
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["annotation"],
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "#BDBDBD",
        },
    )

    plt.tight_layout()

    safe_name = (
        traj["artist"].replace("/", "_").replace(" ", "_").replace("$", "S").replace(".", "")
    )[:50]
    if guesses:
        guesser_names = "_".join(g["Name"].replace(" ", "").replace("/", "") for g in guesses)[:30]
        output_path = output_dir / f"trajectory_{safe_name}_{guesser_names}.png"
    else:
        output_path = output_dir / f"trajectory_{safe_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = sys.argv[1:]

    # Parse flags
    top_n = 0
    show_critic = False
    show_freq = False
    compare = False
    albums_only = False
    guess_file = None
    artist_queries = []
    i = 0
    while i < len(args):
        if args[i] == "--top":
            top_n = int(args[i + 1]) if i + 1 < len(args) else 20
            i += 2
        elif args[i] == "--critic":
            show_critic = True
            i += 1
        elif args[i] == "--freq":
            show_freq = True
            i += 1
        elif args[i] == "--compare":
            compare = True
            i += 1
        elif args[i] == "--albums-only":
            albums_only = True
            i += 1
        elif args[i] == "--guess":
            guess_file = args[i + 1] if i + 1 < len(args) else "data/guesses/responses.csv"
            i += 2
        else:
            artist_queries.append(args[i])
            i += 1

    print("Loading model and data...", end="", flush=True)
    data = _load_all()
    print(" done.")

    # Load human guesses if provided
    guesses = {}
    if guess_file:
        gf = Path(guess_file)
        if gf.exists():
            gdf = pd.read_csv(gf)
            for col in ["Name", "Artist", "Predicted Score", "Low", "High"]:
                if col not in gdf.columns:
                    print(f"  Warning: guess file missing column '{col}'")
                    guess_file = None
                    break
            if guess_file:
                for artist_name, group in gdf.groupby("Artist"):
                    matched = _find_artist(str(artist_name), data["train"], data["raw"])
                    if matched:
                        guesses[matched] = group.to_dict("records")
                print(f"Loaded {len(gdf)} guesses for {len(guesses)} artists.")
        else:
            print(f"  Warning: guess file not found: {gf}")

    output_dir = Path("reports/trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build artist list
    if top_n > 0:
        train = data["train"]
        counts = train.groupby("Artist").size().sort_values(ascending=False)
        artists = counts.head(top_n).index.tolist()
        print(f"Generating trajectories for top {top_n} most prolific artists")
    elif artist_queries:
        artists = []
        for q in artist_queries:
            a = _find_artist(q, data["train"], data["raw"])
            if a:
                artists.append(a)
    elif guesses:
        artists = list(guesses.keys())
        print(f"Using artists from guess file: {', '.join(artists)}")
    else:
        print("Usage:")
        print('  python scripts/generate_trajectories.py "Artist Name"')
        print("  python scripts/generate_trajectories.py --top 20")
        return

    use_freq = show_freq or compare

    if use_freq:
        print("Fitting frequentist model...", end="", flush=True)
        _fit_frequentist(data)
        print(" done.")

    for artist in artists:
        print(f"  {artist}...", end="", flush=True)
        # If guesses specify a Type, use it to override albums_only per artist
        artist_guesses = guesses.get(artist, [])
        artist_albums_only = albums_only
        if artist_guesses:
            types = [g.get("Type", "").strip().lower() for g in artist_guesses]
            if any(t == "albums only" for t in types):
                artist_albums_only = True
        traj = build_trajectory(artist, data, freq=use_freq, albums_only=artist_albums_only)
        if traj is None:
            print(" (not in model, skipped)")
            continue
        path = plot_trajectory(
            traj,
            output_dir,
            show_critic=show_critic,
            show_freq=show_freq,
            compare=compare,
            guesses=guesses.get(artist, []),
        )
        print(f" saved to {path}")

    print(f"\nDone. {len(artists)} trajectories in {output_dir}/")


if __name__ == "__main__":
    main()
