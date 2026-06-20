"""
Build a single comprehensive CSV merging all available data sources.

Sources:
  1. data/raw/all_albums_full.csv          — base album data (18 cols)
  2. data/processed/cleaned_all.csv        — cleaning flags & derived cols (+11)
  3. data/processed/critic_score.csv        — membership flag
  4. data/processed/user_score_minratings_* — membership flags
  5. data/splits/within_artist_temporal/    — train/test split assignment
  6. outputs/predictions/next_album_*.csv  — model predictions (pivoted)
  7. models/training_summary.json          — artist model index, global mean
  8. models/*.nc                           — posterior artist effects
  9. Computed: genre stats, artist career stats, label stats, decade

Output: data/comprehensive_albums.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_raw():
    df = pd.read_csv(ROOT / "data/raw/all_albums_full.csv", low_memory=False)
    print(f"  Raw: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_cleaned_extras():
    df = pd.read_csv(ROOT / "data/processed/cleaned_all.csv", low_memory=False)
    print(f"  Cleaned: {len(df):,} rows, {len(df.columns)} cols")
    new_cols = [
        "original_row_id",
        "Release_Date_Parsed",
        "date_risk",
        "date_imputation_type",
        "flag_future_year",
        "flag_sparse_era",
        "num_artists",
        "is_collaboration",
        "collab_type",
        "primary_genre",
        "is_unknown_artist",
    ]
    return df[["Album_URL"] + new_cols]


def load_subset_flags():
    url_col = "Album_URL"
    critic = pd.read_csv(
        ROOT / "data/processed/critic_score.csv", usecols=[url_col], low_memory=False
    )
    user5 = pd.read_csv(
        ROOT / "data/processed/user_score_minratings_5.csv", usecols=[url_col], low_memory=False
    )
    user10 = pd.read_csv(
        ROOT / "data/processed/user_score_minratings_10.csv", usecols=[url_col], low_memory=False
    )
    user25 = pd.read_csv(
        ROOT / "data/processed/user_score_minratings_25.csv", usecols=[url_col], low_memory=False
    )
    print(
        f"  Subsets: critic={len(critic):,}, user5={len(user5):,}, user10={len(user10):,}, user25={len(user25):,}"
    )

    flags = pd.DataFrame({url_col: pd.concat([critic, user5, user10, user25])[url_col].unique()})
    flags["in_critic_set"] = flags[url_col].isin(critic[url_col])
    flags["in_user_5"] = flags[url_col].isin(user5[url_col])
    flags["in_user_10"] = flags[url_col].isin(user10[url_col])
    flags["in_user_25"] = flags[url_col].isin(user25[url_col])
    return flags


def load_split_assignment():
    splits_dir = ROOT / "data/splits/within_artist_temporal"
    url_col = "Album_URL"
    parts = []
    for name in ["train", "test", "validation"]:
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if url_col in df.columns and len(df) > 0:
                df = df[[url_col]].copy()
                df["split_set"] = name
                parts.append(df)
                print(f"  Split {name}: {len(df):,} rows")
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=[url_col, "split_set"])


def load_predictions():
    pred_path = ROOT / "outputs/predictions/next_album_known_artists.csv"
    if not pred_path.exists():
        print("  Predictions: not found, skipping")
        return pd.DataFrame()

    preds = pd.read_csv(pred_path)
    print(f"  Predictions: {len(preds):,} rows, {preds['artist'].nunique():,} artists")

    scenario_map = {
        "same": "pred_same",
        "population_mean": "pred_popmean",
        "artist_mean": "pred_artmean",
    }
    value_cols = [
        "pred_mean",
        "pred_std",
        "pred_q05",
        "pred_q25",
        "pred_q50",
        "pred_q75",
        "pred_q95",
    ]

    pivoted_parts = []
    for scenario, prefix in scenario_map.items():
        subset = preds[preds["scenario"] == scenario].copy()
        rename = {col: f"{prefix}_{col.replace('pred_', '')}" for col in value_cols}
        subset = subset.rename(columns=rename)
        subset = subset[["artist"] + list(rename.values())]
        pivoted_parts.append(subset)

    result = pivoted_parts[0]
    for part in pivoted_parts[1:]:
        result = result.merge(part, on="artist", how="outer")

    meta = (
        preds.groupby("artist")
        .first()[["last_score", "n_training_albums", "horizon_clamped"]]
        .reset_index()
    )
    meta = meta.rename(
        columns={
            "last_score": "pred_last_score",
            "n_training_albums": "pred_n_training_albums",
            "horizon_clamped": "pred_horizon_clamped",
        }
    )
    result = result.merge(meta, on="artist", how="left")
    return result


def load_artist_posteriors():
    """Extract per-artist posterior effects from the Bayesian model."""
    ts_path = ROOT / "models/training_summary.json"
    if not ts_path.exists():
        print("  Artist posteriors: training_summary.json not found, skipping")
        return pd.DataFrame()

    with open(ts_path) as f:
        ts = json.load(f)

    artist_to_idx = ts.get("artist_to_idx", {})
    global_mean = ts.get("global_mean_score", None)
    print(f"  Training summary: {len(artist_to_idx):,} artists, global_mean={global_mean:.2f}")

    # Build artist index mapping
    artist_df = pd.DataFrame(
        [{"artist": name, "model_artist_idx": idx} for name, idx in artist_to_idx.items()]
    )
    artist_df["model_global_mean"] = global_mean

    # Try to load posterior artist effects from NetCDF
    nc_files = sorted(ROOT.glob("models/*.nc"))
    if nc_files:
        try:
            import arviz as az

            idata = az.from_netcdf(nc_files[-1])
            post = idata.posterior

            if "user_init_artist_effect" in post:
                ae = post["user_init_artist_effect"]
                # Posterior mean, std, and 90% credible interval per artist
                ae_mean = ae.mean(dim=["chain", "draw"]).values
                ae_std = ae.std(dim=["chain", "draw"]).values
                ae_q05 = ae.quantile(0.05, dim=["chain", "draw"]).values
                ae_q95 = ae.quantile(0.95, dim=["chain", "draw"]).values

                artist_df["artist_effect_mean"] = artist_df["model_artist_idx"].map(
                    dict(enumerate(ae_mean))
                )
                artist_df["artist_effect_std"] = artist_df["model_artist_idx"].map(
                    dict(enumerate(ae_std))
                )
                artist_df["artist_effect_q05"] = artist_df["model_artist_idx"].map(
                    dict(enumerate(ae_q05))
                )
                artist_df["artist_effect_q95"] = artist_df["model_artist_idx"].map(
                    dict(enumerate(ae_q95))
                )
                print(f"  Posterior: loaded artist effects for {len(ae_mean)} artists")

            # Key model hyperparameters (scalar summaries)
            for var, col in [
                ("user_sigma_artist", "model_sigma_artist"),
                ("user_rho", "model_ar1_rho"),
            ]:
                if var in post:
                    artist_df[col] = float(post[var].mean())

        except Exception as e:
            print(f"  Posterior: could not load NetCDF ({e})")

    return artist_df


def compute_artist_career_stats(raw):
    """Compute per-artist career statistics from raw album data."""
    df = raw.copy()
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["Critic_Score_num"] = pd.to_numeric(df["Critic Score"], errors="coerce")
    df["Avg_Track_num"] = pd.to_numeric(df["Avg Track Score"], errors="coerce")

    stats = (
        df.groupby("Artist")
        .agg(
            artist_total_albums=("Album", "count"),
            artist_first_year=("Year_num", "min"),
            artist_latest_year=("Year_num", "max"),
            artist_mean_user_score=("User_Score_num", "mean"),
            artist_mean_critic_score=("Critic_Score_num", "mean"),
            artist_mean_track_score=("Avg_Track_num", "mean"),
            artist_total_user_ratings=(
                "User Ratings",
                lambda x: pd.to_numeric(x, errors="coerce").sum(),
            ),
            artist_max_user_ratings=(
                "User Ratings",
                lambda x: pd.to_numeric(x, errors="coerce").max(),
            ),
        )
        .reset_index()
    )

    stats["artist_career_span"] = stats["artist_latest_year"] - stats["artist_first_year"]

    # Round for readability
    for col in ["artist_mean_user_score", "artist_mean_critic_score", "artist_mean_track_score"]:
        stats[col] = stats[col].round(2)

    print(f"  Artist career stats: {len(stats):,} artists")
    return stats


def compute_genre_stats(raw, cleaned_extras):
    """Compute per-genre aggregate statistics using primary_genre from cleaned data."""
    # Join primary_genre onto raw via Album URL
    df = raw[["Album URL", "User Score", "Critic Score", "Avg Track Score", "User Ratings"]].copy()
    genre_map = cleaned_extras[["Album_URL", "primary_genre"]].copy()
    df = df.merge(genre_map, left_on="Album URL", right_on="Album_URL", how="left")

    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["Critic_Score_num"] = pd.to_numeric(df["Critic Score"], errors="coerce")
    df["Avg_Track_num"] = pd.to_numeric(df["Avg Track Score"], errors="coerce")
    df["User_Ratings_num"] = pd.to_numeric(df["User Ratings"], errors="coerce")

    stats = (
        df.groupby("primary_genre")
        .agg(
            genre_n_albums=("Album URL", "count"),
            genre_mean_user_score=("User_Score_num", "mean"),
            genre_mean_critic_score=("Critic_Score_num", "mean"),
            genre_mean_track_score=("Avg_Track_num", "mean"),
            genre_median_user_ratings=("User_Ratings_num", "median"),
        )
        .reset_index()
    )

    # Rank genres by album count
    stats["genre_rank_by_count"] = (
        stats["genre_n_albums"].rank(ascending=False, method="min").astype(int)
    )

    for col in [
        "genre_mean_user_score",
        "genre_mean_critic_score",
        "genre_mean_track_score",
        "genre_median_user_ratings",
    ]:
        stats[col] = stats[col].round(2)

    print(f"  Genre stats: {len(stats):,} unique primary genres")
    return stats


def compute_label_stats(raw):
    """Compute per-label aggregate statistics."""
    df = raw[["Label", "User Score", "Critic Score"]].copy()
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["Critic_Score_num"] = pd.to_numeric(df["Critic Score"], errors="coerce")

    stats = (
        df.groupby("Label")
        .agg(
            label_n_albums=("Label", "count"),
            label_mean_user_score=("User_Score_num", "mean"),
            label_mean_critic_score=("Critic_Score_num", "mean"),
        )
        .reset_index()
    )

    stats["label_mean_user_score"] = stats["label_mean_user_score"].round(2)
    stats["label_mean_critic_score"] = stats["label_mean_critic_score"].round(2)

    print(f"  Label stats: {len(stats):,} unique labels")
    return stats


def compute_decade_stats(raw):
    """Compute per-decade aggregate statistics."""
    df = raw[["Year", "User Score", "Critic Score"]].copy()
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    df["decade"] = (df["Year_num"] // 10 * 10).astype("Int64")
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["Critic_Score_num"] = pd.to_numeric(df["Critic Score"], errors="coerce")

    stats = (
        df.groupby("decade")
        .agg(
            decade_n_albums=("Year", "count"),
            decade_mean_user_score=("User_Score_num", "mean"),
            decade_mean_critic_score=("Critic_Score_num", "mean"),
        )
        .reset_index()
    )

    stats["decade_mean_user_score"] = stats["decade_mean_user_score"].round(2)
    stats["decade_mean_critic_score"] = stats["decade_mean_critic_score"].round(2)

    print(f"  Decade stats: {len(stats):,} decades")
    return stats, df[["Year", "decade"]].drop_duplicates()


def compute_per_album_features(raw):
    """Compute per-album features that require sorting/parsing."""
    df = raw[
        [
            "Artist",
            "Album",
            "Year",
            "Release Date",
            "Album URL",
            "Genres",
            "Descriptors",
            "User Score",
            "Critic Score",
        ]
    ].copy()

    # --- Album ID from URL ---
    df["album_id"] = df["Album URL"].str.extract(r"/album/(\d+)-", expand=False).astype("Int64")

    # --- Genre and descriptor counts ---
    df["n_genres"] = (
        df["Genres"].str.count(",").add(1).where(df["Genres"].notna(), other=pd.NA).astype("Int64")
    )
    df["n_descriptors"] = (
        df["Descriptors"]
        .str.count(",")
        .add(1)
        .where(df["Descriptors"].notna(), other=pd.NA)
        .astype("Int64")
    )

    # --- Release month ---
    parsed = pd.to_datetime(df["Release Date"], format="mixed", errors="coerce")
    df["release_month"] = parsed.dt.month.astype("Int64")
    df["release_day_of_week"] = parsed.dt.dayofweek.astype("Int64")  # 0=Mon, 6=Sun

    # --- Album sequence within artist (chronological order) ---
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    df["_parsed_date"] = parsed
    df = df.sort_values(["Artist", "_parsed_date", "Year_num", "Album"])
    df["album_sequence"] = df.groupby("Artist").cumcount() + 1

    # --- Days since previous album ---
    df["_prev_date"] = df.groupby("Artist")["_parsed_date"].shift(1)
    df["days_since_prev_album"] = (df["_parsed_date"] - df["_prev_date"]).dt.days.astype("Int64")

    # --- Previous album score ---
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["prev_user_score"] = df.groupby("Artist")["User_Score_num"].shift(1)

    # --- Debut / latest flags ---
    artist_max_seq = df.groupby("Artist")["album_sequence"].transform("max")
    df["is_debut_album"] = df["album_sequence"] == 1
    df["is_latest_album"] = df["album_sequence"] == artist_max_seq

    # --- Score percentile ranks ---
    df["user_score_percentile"] = df["User_Score_num"].rank(pct=True).mul(100).round(1)
    critic_num = pd.to_numeric(df["Critic Score"], errors="coerce")
    df["critic_score_percentile"] = critic_num.rank(pct=True).mul(100).round(1)

    keep = [
        "Album URL",
        "album_id",
        "n_genres",
        "n_descriptors",
        "release_month",
        "release_day_of_week",
        "album_sequence",
        "days_since_prev_album",
        "prev_user_score",
        "is_debut_album",
        "is_latest_album",
        "user_score_percentile",
        "critic_score_percentile",
    ]

    print(f"  Per-album features: {len(keep) - 1} new columns")
    return df[keep]


def compute_artist_extra_stats(raw):
    """Artist consistency and extremes — not derivable from existing career stats."""
    df = raw[["Artist", "User Score"]].copy()
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")

    stats = (
        df.groupby("Artist")
        .agg(
            artist_score_std=("User_Score_num", "std"),
            artist_best_user_score=("User_Score_num", "max"),
            artist_worst_user_score=("User_Score_num", "min"),
        )
        .reset_index()
    )

    stats["artist_score_std"] = stats["artist_score_std"].round(2)

    print(f"  Artist extra stats: {len(stats):,} artists")
    return stats


def load_model_training_features():
    """Load per-album features from the model's constant_data (training set only)."""
    nc_files = sorted(ROOT.glob("models/*.nc"))
    if not nc_files:
        print("  Model features: no NetCDF found, skipping")
        return pd.DataFrame()

    try:
        import arviz as az

        idata = az.from_netcdf(nc_files[-1])

        if not hasattr(idata, "constant_data"):
            print("  Model features: no constant_data group, skipping")
            return pd.DataFrame()

        cd = idata.constant_data

        # Load the training split to get Album URLs for joining
        train_path = ROOT / "data/splits/within_artist_temporal/train.parquet"
        if not train_path.exists():
            print("  Model features: train split not found, skipping")
            return pd.DataFrame()

        train_split = pd.read_parquet(train_path)

        # Extract per-observation scalars
        result = train_split[["Album_URL"]].copy()

        if "album_seq" in cd:
            result["model_album_seq"] = cd["album_seq"].values
        if "prev_score" in cd:
            result["model_prev_score"] = np.round(cd["prev_score"].values, 2)
        if "n_reviews" in cd:
            result["model_n_reviews"] = cd["n_reviews"].values.astype(int)

        # Extract the standardized feature matrix
        if "X" in cd:
            with open(ROOT / "models/training_summary.json") as f:
                ts = json.load(f)
            feat_cols = ts.get("feature_cols", [])
            X = cd["X"].values
            if len(feat_cols) == X.shape[1]:
                for i, col in enumerate(feat_cols):
                    result[f"feat_{col}"] = np.round(X[:, i], 4)

        print(
            f"  Model features: {len(result):,} training albums, {len(result.columns) - 1} feature cols"
        )
        return result

    except Exception as e:
        print(f"  Model features: error ({e})")
        return pd.DataFrame()


def load_old_run_consistency():
    """Load artist consistency rankings from old model run."""
    path = Path(
        "/mnt/c/Users/jcwen/runs/run_20260110_012055_b61e4dab/artist_consistency_rankings.csv"
    )
    if not path.exists():
        print("  Old consistency: not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(path)
    keep = {
        "Artist": "Artist",
        "Consistency_Score": "old_consistency_score",
        "Consistency_CI_Low": "old_consistency_ci_low",
        "Consistency_CI_High": "old_consistency_ci_high",
        "Composite_Score": "old_composite_score",
        "Rank": "old_consistency_rank",
        "Score_Z": "old_score_z",
        "Consistency_Z": "old_consistency_z",
    }
    df = df[list(keep.keys())].rename(columns=keep)
    print(f"  Old consistency: {len(df):,} artists")
    return df


def load_old_run_trajectories():
    """Load artist trajectory predictions from old model run."""
    path = Path(
        "/mnt/c/Users/jcwen/runs/run_20260110_012055_b61e4dab/artist_next_album_predictions.csv"
    )
    if not path.exists():
        print("  Old trajectories: not found, skipping")
        return pd.DataFrame()

    df = pd.read_csv(path)
    keep = {
        "Artist": "Artist",
        "Slope_Per_Year": "old_slope_per_year",
        "Slope_CI_Low": "old_slope_ci_low",
        "Slope_CI_High": "old_slope_ci_high",
        "Mean_Gap": "old_mean_gap_days",
        "Prob_Increase": "old_prob_increase",
        "Direction": "old_direction",
    }
    df = df[list(keep.keys())].rename(columns=keep)
    print(f"  Old trajectories: {len(df):,} artists")
    return df


def load_old_run_effects():
    """Load genre, label, and descriptor effects from old model run."""
    path = Path("/mnt/c/Users/jcwen/runs/run_20260110_012055_b61e4dab/effects_summary.json")
    if not path.exists():
        print("  Old effects: not found, skipping")
        return {}, {}, {}

    with open(path) as f:
        es = json.load(f)

    # Genre effects → joinable on primary_genre
    genre_rows = []
    for genre, vals in es.get("exploratory_genre", {}).items():
        genre_rows.append(
            {
                "primary_genre": genre,
                "genre_effect_coef": round(vals["coef"], 3),
                "genre_effect_ci_low": round(vals["ci_low"], 3),
                "genre_effect_ci_high": round(vals["ci_high"], 3),
                "genre_effect_p_positive": round(vals["p_gt_zero"], 4),
            }
        )
    genre_effects = pd.DataFrame(genre_rows) if genre_rows else pd.DataFrame()

    # Label effects → joinable on Label
    label_rows = []
    for label, vals in es.get("exploratory_label", {}).items():
        if label == "__OTHER__":
            continue
        label_rows.append(
            {
                "Label": label,
                "label_effect_coef": round(vals["coef"], 3),
                "label_effect_ci_low": round(vals["ci_low"], 3),
                "label_effect_ci_high": round(vals["ci_high"], 3),
            }
        )
    label_effects = pd.DataFrame(label_rows) if label_rows else pd.DataFrame()

    # Descriptor effects → build lookup for per-album matching
    desc_effects = es.get("exploratory_descriptor", {})

    print(
        f"  Old effects: {len(genre_rows)} genre, {len(label_rows)} label, {len(desc_effects)} descriptor effects"
    )
    return genre_effects, label_effects, desc_effects


def compute_descriptor_features(raw, desc_effects):
    """Compute per-album descriptor-based features using effect coefficients."""
    if not desc_effects:
        return pd.DataFrame()

    rows = []
    for _, album in raw[["Album URL", "Descriptors"]].iterrows():
        url = album["Album URL"]
        descs_str = album["Descriptors"]

        if pd.isna(descs_str) or not descs_str.strip():
            rows.append({"Album URL": url})
            continue

        descs = [d.strip().lower() for d in str(descs_str).split(",")]
        matched_effects = []
        for d in descs:
            # Try exact match and common normalizations
            for key in [d, d.replace("-", " "), d.replace("_", " ")]:
                if key in desc_effects:
                    matched_effects.append(desc_effects[key]["coef"])
                    break

        pos = [e for e in matched_effects if e > 0]
        neg = [e for e in matched_effects if e < 0]

        rows.append(
            {
                "Album URL": url,
                "desc_effects_sum": round(sum(matched_effects), 3) if matched_effects else None,
                "desc_effects_mean": round(sum(matched_effects) / len(matched_effects), 3)
                if matched_effects
                else None,
                "desc_n_positive_effects": len(pos),
                "desc_n_negative_effects": len(neg),
                "desc_strongest_effect": round(max(matched_effects, key=abs), 3)
                if matched_effects
                else None,
            }
        )

    result = pd.DataFrame(rows)
    populated = result["desc_effects_sum"].notna().sum()
    print(f"  Descriptor features: {populated:,} albums with matched descriptors")
    return result


def multi_hot_encode_genres(raw):
    """One-hot encode all genres into individual binary columns."""
    genre_sets = raw["Genres"].dropna().str.split(r"\s*,\s*")
    all_genres = sorted({g for gs in genre_sets for g in gs if g})
    print(f"  Genre one-hot: {len(all_genres)} unique genres")

    result = pd.DataFrame(
        0,
        index=raw.index,
        columns=[
            f"genre_{g.replace(' ', '_').replace('-', '_').replace('&', 'and')}" for g in all_genres
        ],
        dtype=np.int8,
    )
    for idx, genres_str in raw["Genres"].items():
        if pd.isna(genres_str):
            continue
        for g in str(genres_str).split(","):
            g = g.strip()
            col = f"genre_{g.replace(' ', '_').replace('-', '_').replace('&', 'and')}"
            if col in result.columns:
                result.at[idx, col] = 1
    return result


def multi_hot_encode_descriptors(raw):
    """One-hot encode all descriptors into individual binary columns."""
    desc_sets = raw["Descriptors"].dropna().str.split(r"\s*,\s*")
    all_descs = sorted({d for ds in desc_sets for d in ds if d})
    print(f"  Descriptor one-hot: {len(all_descs)} unique descriptors")

    result = pd.DataFrame(
        0,
        index=raw.index,
        columns=[f"desc_{d.replace(' ', '_').replace('-', '_')}" for d in all_descs],
        dtype=np.int8,
    )
    for idx, desc_str in raw["Descriptors"].items():
        if pd.isna(desc_str):
            continue
        for d in str(desc_str).split(","):
            d = d.strip()
            col = f"desc_{d.replace(' ', '_').replace('-', '_')}"
            if col in result.columns:
                result.at[idx, col] = 1
    return result


def multi_hot_encode_labels(raw, top_n=500):
    """One-hot encode top N labels into individual binary columns."""
    label_counts = raw["Label"].value_counts()
    top_labels = label_counts.head(top_n).index.tolist()
    print(
        f"  Label one-hot: top {len(top_labels)} labels (min count: {label_counts.iloc[min(top_n - 1, len(label_counts) - 1)]})"
    )

    result = pd.DataFrame(
        0,
        index=raw.index,
        columns=[
            f"label_{lbl.replace(' ', '_').replace('.', '').replace(',', '')[:40]}"
            for lbl in top_labels
        ],
        dtype=np.int8,
    )
    for idx, label in raw["Label"].items():
        if pd.isna(label):
            continue
        label = str(label).strip()
        if label in top_labels:
            col = f"label_{label.replace(' ', '_').replace('.', '').replace(',', '')[:40]}"
            if col in result.columns:
                result.at[idx, col] = 1
    return result


def compute_per_year_stats(raw):
    """Compute per-year aggregate statistics."""
    df = raw[["Year", "User Score", "Critic Score", "User Ratings"]].copy()
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["Critic_Score_num"] = pd.to_numeric(df["Critic Score"], errors="coerce")
    df["User_Ratings_num"] = pd.to_numeric(df["User Ratings"], errors="coerce")

    stats = (
        df.groupby("Year_num")
        .agg(
            year_n_albums=("Year", "count"),
            year_mean_user_score=("User_Score_num", "mean"),
            year_mean_critic_score=("Critic_Score_num", "mean"),
            year_median_user_ratings=("User_Ratings_num", "median"),
            year_std_user_score=("User_Score_num", "std"),
        )
        .reset_index()
        .rename(columns={"Year_num": "Year_join"})
    )

    for col in [
        "year_mean_user_score",
        "year_mean_critic_score",
        "year_median_user_ratings",
        "year_std_user_score",
    ]:
        stats[col] = stats[col].round(2)

    print(f"  Per-year stats: {len(stats)} years")
    return stats


def compute_rolling_artist_stats(raw):
    """Compute rolling window statistics for each artist's discography."""
    df = raw[["Artist", "Album", "Year", "Release Date", "User Score", "Album URL"]].copy()
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["_date"] = pd.to_datetime(df["Release Date"], format="mixed", errors="coerce")
    df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.sort_values(["Artist", "_date", "Year_num", "Album"])

    # Rolling 3-album and 5-album statistics
    grouped = df.groupby("Artist")["User_Score_num"]
    df["rolling_3_mean"] = grouped.transform(lambda x: x.rolling(3, min_periods=2).mean()).round(2)
    df["rolling_3_std"] = grouped.transform(lambda x: x.rolling(3, min_periods=2).std()).round(2)
    df["rolling_5_mean"] = grouped.transform(lambda x: x.rolling(5, min_periods=3).mean()).round(2)
    df["rolling_5_std"] = grouped.transform(lambda x: x.rolling(5, min_periods=3).std()).round(2)

    # Cumulative mean (expanding window)
    df["cumulative_mean"] = grouped.transform(lambda x: x.expanding().mean()).round(2)

    # Score momentum (current - rolling 3 mean), shifted to avoid leakage
    df["score_momentum"] = (df["User_Score_num"] - df["rolling_3_mean"]).round(2)

    keep = [
        "Album URL",
        "rolling_3_mean",
        "rolling_3_std",
        "rolling_5_mean",
        "rolling_5_std",
        "cumulative_mean",
        "score_momentum",
    ]
    print(f"  Rolling artist stats: {len(keep) - 1} columns")
    return df[keep]


def compute_rankings_and_tiers(raw):
    """Compute various rankings and categorical tiers."""
    df = raw[["Album URL", "User Score", "User Ratings", "Critic Score"]].copy()
    df["User_Score_num"] = pd.to_numeric(df["User Score"], errors="coerce")
    df["User_Ratings_num"] = pd.to_numeric(df["User Ratings"], errors="coerce")
    df["Critic_Score_num"] = pd.to_numeric(df["Critic Score"], errors="coerce")

    # Popularity tier based on user ratings
    conditions = [
        df["User_Ratings_num"] >= 5000,
        df["User_Ratings_num"] >= 1000,
        df["User_Ratings_num"] >= 100,
        df["User_Ratings_num"] >= 10,
        df["User_Ratings_num"] >= 1,
    ]
    choices = ["mainstream", "popular", "known", "niche", "obscure"]
    df["popularity_tier"] = np.select(conditions, choices, default=pd.NA)
    df.loc[df["User_Ratings_num"].isna(), "popularity_tier"] = pd.NA

    # Score tier
    conditions_score = [
        df["User_Score_num"] >= 90,
        df["User_Score_num"] >= 80,
        df["User_Score_num"] >= 70,
        df["User_Score_num"] >= 60,
        df["User_Score_num"] >= 50,
        df["User_Score_num"] >= 0,
    ]
    choices_score = ["exceptional", "great", "good", "mixed", "poor", "bad"]
    df["score_tier"] = np.select(conditions_score, choices_score, default=pd.NA)
    df.loc[df["User_Score_num"].isna(), "score_tier"] = pd.NA

    # Critic-user agreement
    df["critic_user_gap"] = (df["User_Score_num"] - df["Critic_Score_num"]).round(1)
    conditions_agree = [
        df["critic_user_gap"].abs() <= 3,
        df["critic_user_gap"].abs() <= 8,
        df["critic_user_gap"] > 8,
        df["critic_user_gap"] < -8,
    ]
    choices_agree = ["consensus", "slight_diff", "user_favored", "critic_favored"]
    df["critic_user_agreement"] = np.select(conditions_agree, choices_agree, default=pd.NA)
    df.loc[df["critic_user_gap"].isna(), "critic_user_agreement"] = pd.NA

    keep = [
        "Album URL",
        "popularity_tier",
        "score_tier",
        "critic_user_gap",
        "critic_user_agreement",
    ]
    print(f"  Rankings/tiers: {len(keep) - 1} columns")
    return df[keep]


def main():
    print("Loading data sources...")
    raw = load_raw()
    cleaned_extras = load_cleaned_extras()
    subset_flags = load_subset_flags()
    split_assign = load_split_assignment()
    predictions = load_predictions()
    artist_posteriors = load_artist_posteriors()
    artist_career = compute_artist_career_stats(raw)
    artist_extra = compute_artist_extra_stats(raw)
    genre_stats = compute_genre_stats(raw, cleaned_extras)
    label_stats = compute_label_stats(raw)
    decade_stats, decade_map = compute_decade_stats(raw)
    per_album = compute_per_album_features(raw)
    model_features = load_model_training_features()
    old_consistency = load_old_run_consistency()
    old_trajectories = load_old_run_trajectories()
    genre_effects, label_effects, desc_effects_lookup = load_old_run_effects()
    desc_features = compute_descriptor_features(raw, desc_effects_lookup)
    genre_onehot = multi_hot_encode_genres(raw)
    desc_onehot = multi_hot_encode_descriptors(raw)
    label_onehot = multi_hot_encode_labels(raw, top_n=2500)
    year_stats = compute_per_year_stats(raw)
    rolling_stats = compute_rolling_artist_stats(raw)
    tier_features = compute_rankings_and_tiers(raw)

    print("\nMerging...")

    # 1) Start with raw
    merged = raw.copy()

    # 2) Add decade column
    merged["Year_num"] = pd.to_numeric(merged["Year"], errors="coerce")
    merged["decade"] = (merged["Year_num"] // 10 * 10).astype("Int64")
    merged.drop(columns=["Year_num"], inplace=True)

    # 3) Join cleaned extras
    merged = merged.merge(cleaned_extras, left_on="Album URL", right_on="Album_URL", how="left")
    merged.drop(columns=["Album_URL"], inplace=True)
    print(f"  After cleaned extras: {len(merged):,} rows, {len(merged.columns)} cols")

    # 4) Join subset flags
    merged = merged.merge(subset_flags, left_on="Album URL", right_on="Album_URL", how="left")
    if "Album_URL" in merged.columns:
        merged.drop(columns=["Album_URL"], inplace=True)
    for col in ["in_critic_set", "in_user_5", "in_user_10", "in_user_25"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False)
            merged[col] = merged[col].astype(bool)
    print(f"  After subset flags: {len(merged):,} rows, {len(merged.columns)} cols")

    # 5) Join split assignment
    merged = merged.merge(split_assign, left_on="Album URL", right_on="Album_URL", how="left")
    if "Album_URL" in merged.columns:
        merged.drop(columns=["Album_URL"], inplace=True)
    print(f"  After split assign: {len(merged):,} rows, {len(merged.columns)} cols")

    # 6) Join per-album computed features
    merged = merged.merge(per_album, on="Album URL", how="left")
    print(f"  After per-album features: {len(merged):,} rows, {len(merged.columns)} cols")

    # 7) Join predictions
    if not predictions.empty:
        merged = merged.merge(predictions, left_on="Artist", right_on="artist", how="left")
        merged.drop(columns=["artist"], inplace=True)
        print(f"  After predictions: {len(merged):,} rows, {len(merged.columns)} cols")

    # 8) Join artist posteriors
    if not artist_posteriors.empty:
        merged = merged.merge(artist_posteriors, left_on="Artist", right_on="artist", how="left")
        merged.drop(columns=["artist"], inplace=True)
        print(f"  After artist posteriors: {len(merged):,} rows, {len(merged.columns)} cols")

    # 9) Join artist career stats
    merged = merged.merge(artist_career, on="Artist", how="left")
    print(f"  After artist career: {len(merged):,} rows, {len(merged.columns)} cols")

    # 10) Join artist extra stats
    merged = merged.merge(artist_extra, on="Artist", how="left")
    print(f"  After artist extra: {len(merged):,} rows, {len(merged.columns)} cols")

    # 11) Join genre stats via primary_genre
    merged = merged.merge(genre_stats, on="primary_genre", how="left")
    print(f"  After genre stats: {len(merged):,} rows, {len(merged.columns)} cols")

    # 12) Join label stats
    merged = merged.merge(label_stats, on="Label", how="left")
    print(f"  After label stats: {len(merged):,} rows, {len(merged.columns)} cols")

    # 13) Join decade stats
    merged = merged.merge(decade_stats, on="decade", how="left")
    print(f"  After decade stats: {len(merged):,} rows, {len(merged.columns)} cols")

    # 14) Join model training features (training set only)
    if not model_features.empty:
        merged = merged.merge(model_features, left_on="Album URL", right_on="Album_URL", how="left")
        if "Album_URL" in merged.columns:
            merged.drop(columns=["Album_URL"], inplace=True)
        print(f"  After model features: {len(merged):,} rows, {len(merged.columns)} cols")

    # 15) Join old-run consistency rankings
    if not old_consistency.empty:
        merged = merged.merge(old_consistency, on="Artist", how="left")
        print(f"  After old consistency: {len(merged):,} rows, {len(merged.columns)} cols")

    # 16) Join old-run trajectory predictions
    if not old_trajectories.empty:
        merged = merged.merge(old_trajectories, on="Artist", how="left")
        print(f"  After old trajectories: {len(merged):,} rows, {len(merged.columns)} cols")

    # 17) Join genre effects from old model
    if not genre_effects.empty:
        merged = merged.merge(genre_effects, on="primary_genre", how="left")
        print(f"  After genre effects: {len(merged):,} rows, {len(merged.columns)} cols")

    # 18) Join label effects from old model
    if not label_effects.empty:
        merged = merged.merge(label_effects, on="Label", how="left")
        print(f"  After label effects: {len(merged):,} rows, {len(merged.columns)} cols")

    # 19) Join descriptor features
    if not desc_features.empty:
        merged = merged.merge(desc_features, on="Album URL", how="left")
        print(f"  After descriptor features: {len(merged):,} rows, {len(merged.columns)} cols")

    # 20) Join per-year stats
    merged["_year_join"] = pd.to_numeric(merged["Year"], errors="coerce")
    merged = merged.merge(year_stats, left_on="_year_join", right_on="Year_join", how="left")
    merged.drop(columns=["_year_join", "Year_join"], inplace=True)
    print(f"  After year stats: {len(merged):,} rows, {len(merged.columns)} cols")

    # 21) Join rolling artist stats
    merged = merged.merge(rolling_stats, on="Album URL", how="left")
    print(f"  After rolling stats: {len(merged):,} rows, {len(merged.columns)} cols")

    # 22) Join tier features
    merged = merged.merge(tier_features, on="Album URL", how="left")
    print(f"  After tiers: {len(merged):,} rows, {len(merged.columns)} cols")

    # 23) Concat one-hot encoded columns (same index, no join needed)
    merged = pd.concat([merged, genre_onehot, desc_onehot, label_onehot], axis=1)
    print(f"  After one-hot encoding: {len(merged):,} rows, {len(merged.columns)} cols")

    # --- Column ordering ---
    identity_cols = [
        "Artist",
        "Album",
        "Year",
        "decade",
        "Release Date",
        "release_month",
        "release_day_of_week",
        "Album URL",
        "album_id",
        "All Artists",
        "Album Type",
    ]
    score_cols = [
        "Critic Score",
        "User Score",
        "Avg Track Score",
        "User Ratings",
        "Critic Reviews",
        "user_score_percentile",
        "critic_score_percentile",
    ]
    track_cols = ["Tracks", "Runtime (min)", "Avg Track Runtime (min)"]
    metadata_cols = ["Genres", "n_genres", "Label", "Descriptors", "n_descriptors"]
    cleaning_cols = [
        "original_row_id",
        "Release_Date_Parsed",
        "date_risk",
        "date_imputation_type",
        "flag_future_year",
        "flag_sparse_era",
        "primary_genre",
        "is_unknown_artist",
    ]
    collab_cols = ["num_artists", "is_collaboration", "collab_type"]
    sequence_cols = [
        "album_sequence",
        "is_debut_album",
        "is_latest_album",
        "days_since_prev_album",
        "prev_user_score",
    ]
    membership_cols = ["in_critic_set", "in_user_5", "in_user_10", "in_user_25", "split_set"]

    artist_career_cols = [
        "artist_total_albums",
        "artist_first_year",
        "artist_latest_year",
        "artist_career_span",
        "artist_mean_user_score",
        "artist_mean_critic_score",
        "artist_mean_track_score",
        "artist_total_user_ratings",
        "artist_max_user_ratings",
        "artist_score_std",
        "artist_best_user_score",
        "artist_worst_user_score",
    ]
    genre_stat_cols = [
        "genre_n_albums",
        "genre_rank_by_count",
        "genre_mean_user_score",
        "genre_mean_critic_score",
        "genre_mean_track_score",
        "genre_median_user_ratings",
    ]
    label_stat_cols = ["label_n_albums", "label_mean_user_score", "label_mean_critic_score"]
    decade_stat_cols = ["decade_n_albums", "decade_mean_user_score", "decade_mean_critic_score"]
    year_stat_cols = [
        "year_n_albums",
        "year_mean_user_score",
        "year_mean_critic_score",
        "year_median_user_ratings",
        "year_std_user_score",
    ]
    rolling_cols = [
        "rolling_3_mean",
        "rolling_3_std",
        "rolling_5_mean",
        "rolling_5_std",
        "cumulative_mean",
        "score_momentum",
    ]
    tier_cols = ["popularity_tier", "score_tier", "critic_user_gap", "critic_user_agreement"]

    genre_effect_cols = [
        "genre_effect_coef",
        "genre_effect_ci_low",
        "genre_effect_ci_high",
        "genre_effect_p_positive",
    ]
    label_effect_cols = [
        "label_effect_coef",
        "label_effect_ci_low",
        "label_effect_ci_high",
    ]
    desc_feat_cols = [
        "desc_effects_sum",
        "desc_effects_mean",
        "desc_n_positive_effects",
        "desc_n_negative_effects",
        "desc_strongest_effect",
    ]

    old_consistency_cols = [
        "old_consistency_score",
        "old_consistency_ci_low",
        "old_consistency_ci_high",
        "old_composite_score",
        "old_consistency_rank",
        "old_score_z",
        "old_consistency_z",
    ]
    old_trajectory_cols = [
        "old_slope_per_year",
        "old_slope_ci_low",
        "old_slope_ci_high",
        "old_mean_gap_days",
        "old_prob_increase",
        "old_direction",
    ]

    model_cols = [
        "model_artist_idx",
        "model_global_mean",
        "artist_effect_mean",
        "artist_effect_std",
        "artist_effect_q05",
        "artist_effect_q95",
        "model_sigma_artist",
        "model_ar1_rho",
        "model_album_seq",
        "model_prev_score",
        "model_n_reviews",
    ]
    feat_cols = sorted([c for c in merged.columns if c.startswith("feat_")])
    pred_cols = [c for c in merged.columns if c.startswith("pred_")]

    ordered = (
        identity_cols
        + score_cols
        + track_cols
        + metadata_cols
        + cleaning_cols
        + collab_cols
        + sequence_cols
        + artist_career_cols
        + old_consistency_cols
        + old_trajectory_cols
        + rolling_cols
        + tier_cols
        + genre_stat_cols
        + genre_effect_cols
        + label_stat_cols
        + label_effect_cols
        + decade_stat_cols
        + year_stat_cols
        + desc_feat_cols
        + membership_cols
        + model_cols
        + feat_cols
        + pred_cols
    )
    # One-hot columns go at the very end
    genre_oh = sorted(
        [c for c in merged.columns if c.startswith("genre_") and c not in set(ordered)]
    )
    desc_oh = sorted([c for c in merged.columns if c.startswith("desc_") and c not in set(ordered)])
    label_oh = sorted(
        [c for c in merged.columns if c.startswith("label_") and c not in set(ordered)]
    )
    ordered = list(ordered) + genre_oh + desc_oh + label_oh
    ordered = tuple(ordered)
    remaining = [c for c in merged.columns if c not in ordered]
    final_order = [c for c in ordered if c in merged.columns] + remaining
    merged = merged[final_order]

    # Save
    out_path = ROOT / "data" / "comprehensive_albums_v3.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  {len(merged):,} rows × {len(merged.columns)} columns")
    print(f"  Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nColumns ({len(merged.columns)}):")
    for i, col in enumerate(merged.columns):
        non_null = merged[col].notna().sum()
        pct = non_null / len(merged) * 100
        print(f"  {i + 1:3d}. {col:<40s}  ({pct:5.1f}% populated)")


if __name__ == "__main__":
    main()
