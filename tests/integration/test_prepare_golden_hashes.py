"""Golden-hash regression guard for the data-preparation pipeline.

Freezes ``hash_dataframe`` digests of every ``prepare_datasets`` output on a
small fixture (the e2e minimal dataset plus rows that exercise each filter and
date tier). Any behavioral change to cleaning or filtering — column renames,
date parsing, collaboration/genre extraction, exclusion filters, dataset
naming — shows up as a hash mismatch here.

Captured on the pre-descriptor implementation; the descriptor refactor must
keep these outputs byte-identical (default-equals-AOTY contract).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from panelcast.pipelines.prepare_dataset import PrepareConfig, prepare_datasets
from panelcast.utils.hashing import hash_dataframe
from tests.e2e.conftest import MINIMAL_TEST_DATA

# Rows that exercise every cleaning/filter path so each output dataset gets a
# distinct hash (the plain minimal data survives all filters untouched).
EXTRA_ROWS = [
    # Passes min_ratings 5 only (7 ratings).
    {
        "Artist": "Threshold Artist",
        "Album": "Barely Rated",
        "Year": 2019,
        "Release Date": "March 03, 2019",
        "Genres": "Rock, Lo-Fi",
        "Critic Score": 70,
        "User Score": 71,
        "Avg Track Score": 70,
        "User Ratings": 7,
        "Critic Reviews": 3,
        "Tracks": 9,
        "Runtime (min)": 36.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "Small Label",
        "Descriptors": "raw",
        "Album URL": "https://example.com/23",
        "All Artists": "Threshold Artist",
        "Album Type": "Album",
    },
    # Passes 5 and 10, fails 25 (15 ratings).
    {
        "Artist": "Threshold Artist",
        "Album": "Mid Ratings",
        "Year": 2021,
        "Release Date": "June 18, 2021",
        "Genres": "Rock, Lo-Fi",
        "Critic Score": 73,
        "User Score": 74,
        "Avg Track Score": 72,
        "User Ratings": 15,
        "Critic Reviews": 4,
        "Tracks": 10,
        "Runtime (min)": 38.0,
        "Avg Track Runtime (min)": 3.8,
        "Label": "Small Label",
        "Descriptors": "warm",
        "Album URL": "https://example.com/24",
        "All Artists": "Threshold Artist",
        "Album Type": "Album",
    },
    # Missing user score; stays in critic dataset only.
    {
        "Artist": "Critic Only",
        "Album": "No User Score",
        "Year": 2020,
        "Release Date": "May 09, 2020",
        "Genres": "Jazz",
        "Critic Score": 81,
        "User Score": "",
        "Avg Track Score": "",
        "User Ratings": "",
        "Critic Reviews": 6,
        "Tracks": 8,
        "Runtime (min)": 41.0,
        "Avg Track Runtime (min)": 5.1,
        "Label": "Jazz Label",
        "Descriptors": "smooth",
        "Album URL": "https://example.com/25",
        "All Artists": "Critic Only",
        "Album Type": "Album",
    },
    # Zero critic reviews; drops from critic dataset (min_reviews=1).
    {
        "Artist": "User Only",
        "Album": "No Critics",
        "Year": 2018,
        "Release Date": "September 14, 2018",
        "Genres": "Pop",
        "Critic Score": "",
        "User Score": 77,
        "Avg Track Score": 76,
        "User Ratings": 90,
        "Critic Reviews": 0,
        "Tracks": 11,
        "Runtime (min)": 39.0,
        "Avg Track Runtime (min)": 3.5,
        "Label": "Pop Label",
        "Descriptors": "bright",
        "Album URL": "https://example.com/26",
        "All Artists": "User Only",
        "Album Type": "Album",
    },
    # Empty Artist identifier; only survives in cleaned_all.
    {
        "Artist": "",
        "Album": "Orphan Album",
        "Year": 2017,
        "Release Date": "July 07, 2017",
        "Genres": "Ambient",
        "Critic Score": 60,
        "User Score": 65,
        "Avg Track Score": 63,
        "User Ratings": 50,
        "Critic Reviews": 2,
        "Tracks": 6,
        "Runtime (min)": 30.0,
        "Avg Track Runtime (min)": 5.0,
        "Label": "",
        "Descriptors": "",
        "Album URL": "https://example.com/27",
        "All Artists": "",
        "Album Type": "Album",
    },
    # Tier-2 date: no release date, year present (jan1 imputation).
    {
        "Artist": "Hip Hop Artist",
        "Album": "Date Unknown",
        "Year": 2013,
        "Release Date": "",
        "Genres": "Hip Hop",
        "Critic Score": 68,
        "User Score": 70,
        "Avg Track Score": 69,
        "User Ratings": 120,
        "Critic Reviews": 5,
        "Tracks": 12,
        "Runtime (min)": 44.0,
        "Avg Track Runtime (min)": 3.7,
        "Label": "Hip Hop Records",
        "Descriptors": "early",
        "Album URL": "https://example.com/28",
        "All Artists": "Hip Hop Artist",
        "Album Type": "Mixtape",
    },
    # Tier-3 date: no release date and no year (unimputed, date_missing=True).
    {
        "Artist": "Metal Band",
        "Album": "Lost Demo",
        "Year": "",
        "Release Date": "",
        "Genres": "Metal",
        "Critic Score": "",
        "User Score": 66,
        "Avg Track Score": "",
        "User Ratings": 30,
        "Critic Reviews": "",
        "Tracks": 4,
        "Runtime (min)": 15.0,
        "Avg Track Runtime (min)": 3.75,
        "Label": "",
        "Descriptors": "",
        "Album URL": "https://example.com/29",
        "All Artists": "Metal Band",
        "Album Type": "EP",
    },
    # Five-artist ensemble collaboration.
    {
        "Artist": "Pop Singer",
        "Album": "Festival Anthem",
        "Year": 2022,
        "Release Date": "August 12, 2022",
        "Genres": "Pop, Dance Pop",
        "Critic Score": 71,
        "User Score": 74,
        "Avg Track Score": 72,
        "User Ratings": 800,
        "Critic Reviews": 9,
        "Tracks": 1,
        "Runtime (min)": 4.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "Major Pop",
        "Descriptors": "anthemic",
        "Album URL": "https://example.com/30",
        "All Artists": (
            "Pop Singer | Hip Hop Artist | Indie Artist | Metal Band | Electronic Producer"
        ),
        "Album Type": "Single",
    },
    # Unknown-artist sentinel row.
    {
        "Artist": "[unknown artist]",
        "Album": "Mystery Tape",
        "Year": 2016,
        "Release Date": "October 01, 2016",
        "Genres": "Experimental",
        "Critic Score": "",
        "User Score": 62,
        "Avg Track Score": "",
        "User Ratings": 40,
        "Critic Reviews": "",
        "Tracks": 7,
        "Runtime (min)": 28.0,
        "Avg Track Runtime (min)": 4.0,
        "Label": "",
        "Descriptors": "",
        "Album URL": "https://example.com/31",
        "All Artists": "[unknown artist]",
        "Album Type": "Album",
    },
]

FIELDNAMES = [
    "Artist",
    "Album",
    "Year",
    "Release Date",
    "Genres",
    "Critic Score",
    "User Score",
    "Avg Track Score",
    "User Ratings",
    "Critic Reviews",
    "Tracks",
    "Runtime (min)",
    "Avg Track Runtime (min)",
    "Label",
    "Descriptors",
    "Album URL",
    "All Artists",
    "Album Type",
]

# Frozen on the pre-descriptor implementation (2026-06-10).
GOLDEN_HASHES = {
    "cleaned_all": "37fc73ade545b44d46b8c3430c48709a3a8f7ce8e3496e3fc3e14c86f744b1fe",
    "critic_score": "1bf97dd627d62fc6da5d12393ff023d99f58a39332a956d0f7069ec56caba50c",
    "user_score_minratings_5": "281e98f626f83be82f84d7932c65f33e7a2422aebe07d691c164ff878fff5a3b",
    "user_score_minratings_10": "da98f6bdbb707cfc133f6e9ce8b8c75e4391b8083c706404ba3f2cff601c2608",
    "user_score_minratings_25": "d74ba133bfd56d312304af73838efd2edafff08a93661e35df31887176b718b9",
}

GOLDEN_ROWS = {
    "cleaned_all": 31,
    "critic_score": 27,
    "user_score_minratings_5": 29,
    "user_score_minratings_10": 28,
    "user_score_minratings_25": 27,
}

GOLDEN_STATS = {
    "global_mean_score": 77.89285714285714,
    "global_std_score": 6.795092377832726,
    "n_albums": 28,
    "n_artists": 9,
    "source_dataset": "user_score_minratings_10",
}

GOLDEN_EXCLUSIONS = {
    "missing_artist_or_album_identifier": 4,
    "missing_user_score": 3,
    "missing_critic_score": 3,
    "below_min_ratings_25": 2,
    "below_min_ratings_10": 1,
}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory):
    """Run prepare_datasets once on the augmented fixture."""
    tmp = tmp_path_factory.mktemp("golden_prepare")
    csv_path = tmp / "raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(MINIMAL_TEST_DATA)
        writer.writerows(EXTRA_ROWS)

    result = prepare_datasets(
        PrepareConfig(
            raw_path=str(csv_path),
            output_dir=str(tmp / "processed"),
            audit_dir=str(tmp / "audit"),
        )
    )
    return result, tmp


class TestPrepareGoldenHashes:
    def test_dataset_row_counts(self, prepared):
        result, _ = prepared
        rows = {name: result.summary["datasets"][name]["rows"] for name in GOLDEN_ROWS}
        assert rows == GOLDEN_ROWS

    def test_dataset_hashes(self, prepared):
        result, _ = prepared
        mismatches = {}
        for name, expected in GOLDEN_HASHES.items():
            df = pd.read_parquet(result.datasets_created[name])
            actual = hash_dataframe(df)
            if actual != expected:
                mismatches[name] = actual
        assert not mismatches, (
            "Output dataset content changed (behavior-changing edit to the "
            f"cleaning/filtering layer?): {mismatches}"
        )

    def test_dataset_stats(self, prepared):
        _, tmp = prepared
        stats = json.loads(
            Path(tmp / "processed" / "dataset_stats.json").read_text(encoding="utf-8")
        )
        assert stats == GOLDEN_STATS

    def test_exclusion_reasons(self, prepared):
        result, _ = prepared
        assert result.summary["exclusions"]["exclusions_by_reason"] == GOLDEN_EXCLUSIONS
