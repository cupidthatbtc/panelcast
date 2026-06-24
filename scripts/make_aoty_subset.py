"""Build a representative artist-subset of the full AOTY corpus.

The full corpus (~130k rows, ~62k albums with >=10 user ratings across ~11.6k
artists) is too large to fit a 4x5000 publication run on a single 24 GB GPU in
reasonable wall-clock. This script samples whole artists -- not albums -- so the
subset keeps multi-album discographies intact (needed for the within-entity
temporal split and the per-artist random walk) and preserves the strong left
skew of the score distribution (skewness ~= -2.06).

Sampling is by entity: we pick ``--n-artists`` artists that have at least
``--min-albums`` *qualifying* albums (a user score present and at least
``--min-ratings`` ratings), then emit every raw row for those artists -- their
full discographies, including albums below the rating threshold -- so the
cleaning/feature stages see the real entity histories and apply their own
filtering downstream.

    python scripts/make_aoty_subset.py \
        --source ../aoty_pred_pub/data/raw/all_albums_full.csv \
        --output data/raw/aoty_subset.csv

Point the pipeline at the result with AOTY_DATASET_PATH=data/raw/aoty_subset.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

DEFAULT_SOURCE = "../aoty_pred_pub/data/raw/all_albums_full.csv"
DEFAULT_OUTPUT = "data/raw/aoty_subset.csv"


def build_subset(
    source: Path,
    output: Path,
    n_artists: int,
    min_albums: int,
    min_ratings: int,
    seed: int,
) -> None:
    df = pd.read_csv(source, encoding="utf-8-sig")

    user_score = pd.to_numeric(df["User Score"], errors="coerce")
    user_ratings = pd.to_numeric(df["User Ratings"], errors="coerce")
    qualifies = user_score.notna() & (user_ratings >= min_ratings)

    qual_counts = df.loc[qualifies].groupby("Artist").size()
    eligible = qual_counts[qual_counts >= min_albums].index.to_numpy()
    if len(eligible) < n_artists:
        raise SystemExit(
            f"only {len(eligible)} artists have >= {min_albums} qualifying albums; "
            f"cannot sample {n_artists}."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n_artists, replace=False)
    chosen_set = set(chosen.tolist())

    subset = df[df["Artist"].isin(chosen_set)].copy()

    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output, index=False)

    # Report on the qualifying slice -- that is what the model actually fits.
    sub_qual = subset.loc[qualifies.reindex(subset.index, fill_value=False)]
    sub_scores = pd.to_numeric(sub_qual["User Score"], errors="coerce").dropna()
    per_artist = sub_qual.groupby("Artist").size()
    print(f"wrote {output}")
    print(f"  artists sampled:      {len(chosen_set)} (seed {seed})")
    print(f"  raw rows (full discog): {len(subset)}")
    print(f"  qualifying albums:    {len(sub_qual)} (>= {min_ratings} ratings)")
    print(f"  qualifying user-score skewness: {float(skew(sub_scores)):.3f}")
    print(f"  albums/artist among qualifying: min {int(per_artist.min())}, "
          f"median {per_artist.median():.0f}, max {int(per_artist.max())}, "
          f"mean {per_artist.mean():.2f}")
    print(f"  artists with >= 3 qualifying (usable for temporal split): "
          f"{int((per_artist >= 3).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--n-artists", type=int, default=800)
    parser.add_argument("--min-albums", type=int, default=2,
                        help="Minimum qualifying albums for an artist to be eligible.")
    parser.add_argument("--min-ratings", type=int, default=10,
                        help="Rating floor for an album to count as qualifying.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_subset(
        source=args.source,
        output=args.output,
        n_artists=args.n_artists,
        min_albums=args.min_albums,
        min_ratings=args.min_ratings,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
