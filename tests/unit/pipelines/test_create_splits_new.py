"""Additional coverage tests for create_splits pipeline.

Targets uncovered lines/branches including:
- main() CLI entry point full output verification
- create_splits with explicit SplitConfig source_path
- SplitManifest content_hash computation
- hash_dataframe integration in create_splits
- Summary JSON artist_excluded computation
- SplitResult field types
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from panelcast.pipelines.create_splits import (
    SplitConfig,
    SplitResult,
    create_splits,
    main,
    save_split_parquet,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def artist_df():
    """DataFrame with diverse artists for split testing."""
    rows = []
    for artist_id in range(12):
        n_albums = 4 + (artist_id % 3)  # 4-6 albums per artist
        for album_idx in range(n_albums):
            rows.append(
                {
                    "Artist": f"Artist_{artist_id}",
                    "Album": f"Album_{artist_id}_{album_idx}",
                    "Release_Date_Parsed": pd.Timestamp("2017-01-01")
                    + pd.DateOffset(months=album_idx * 6),
                    "User_Score": 60.0 + album_idx * 3 + artist_id * 0.5,
                    "User_Ratings": 20 + album_idx * 15,
                }
            )
    return pd.DataFrame(rows)


# ============================================================================
# Tests: main() CLI full output
# ============================================================================


class TestMainCliOutput:
    def test_main_prints_all_sections(self, artist_df, capsys):
        """main() should print source, temporal, disjoint, and output directory."""
        n = len(artist_df)
        train = artist_df.iloc[: n // 2].copy()
        val = artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = artist_df.iloc[n * 3 // 4 :].copy()

        with (
            patch(
                "panelcast.pipelines.create_splits.pd.read_parquet",
                return_value=artist_df,
            ),
            patch(
                "panelcast.pipelines.create_splits.within_artist_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.artist_disjoint_split",
                return_value=(train, val, test),
            ),
            patch("panelcast.pipelines.create_splits.validate_temporal_split"),
            patch("panelcast.pipelines.create_splits.assert_no_artist_overlap"),
            patch(
                "panelcast.pipelines.create_splits.create_split_assignments",
                return_value=[],
            ),
            patch(
                "panelcast.pipelines.create_splits.save_manifest",
                return_value=Path("manifest.json"),
            ),
            patch(
                "panelcast.pipelines.create_splits.save_split_parquet",
            ),
            patch(
                "panelcast.pipelines.create_splits.hash_dataframe",
                return_value="b" * 64,
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "SPLIT PIPELINE COMPLETE" in out
        assert "Source:" in out
        assert "Rows:" in out
        assert "Artists:" in out
        assert "Within-Artist Temporal Split:" in out
        assert "Train:" in out
        assert "Validation:" in out
        assert "Test:" in out
        assert "Artists included:" in out
        assert "Artists excluded:" in out
        assert "Artist-Disjoint Split:" in out
        assert "Output directory:" in out

    def test_main_shows_insufficient_albums_message(self, artist_df, capsys):
        """main() output should show artists excluded count."""
        n = len(artist_df)
        # Use a subset of artists for train to simulate exclusions
        included_artists = artist_df["Artist"].unique()[:8]
        train = artist_df[artist_df["Artist"].isin(included_artists)].iloc[:20].copy()
        val = artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = artist_df.iloc[n * 3 // 4 :].copy()

        with (
            patch(
                "panelcast.pipelines.create_splits.pd.read_parquet",
                return_value=artist_df,
            ),
            patch(
                "panelcast.pipelines.create_splits.within_artist_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.artist_disjoint_split",
                return_value=(train, val, test),
            ),
            patch("panelcast.pipelines.create_splits.validate_temporal_split"),
            patch("panelcast.pipelines.create_splits.assert_no_artist_overlap"),
            patch(
                "panelcast.pipelines.create_splits.create_split_assignments",
                return_value=[],
            ),
            patch(
                "panelcast.pipelines.create_splits.save_manifest",
                return_value=Path("manifest.json"),
            ),
            patch(
                "panelcast.pipelines.create_splits.save_split_parquet",
            ),
            patch(
                "panelcast.pipelines.create_splits.hash_dataframe",
                return_value="c" * 64,
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "insufficient albums" in out


# ============================================================================
# Tests: summary JSON artists_excluded computation
# ============================================================================


class TestSummaryArtistsExcluded:
    def test_artists_excluded_counts_correctly(self, tmp_path, artist_df):
        """artists_excluded = total artists - temporal train artists."""
        total_artists = artist_df["Artist"].nunique()
        # Simulate temporal split that only includes half the artists
        included = artist_df["Artist"].unique()[:6]
        train = artist_df[artist_df["Artist"].isin(included)].copy()
        val = artist_df[~artist_df["Artist"].isin(included)].iloc[:5].copy()
        test = artist_df[~artist_df["Artist"].isin(included)].iloc[5:10].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_artist_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.artist_disjoint_split",
                return_value=(train, val, test),
            ),
            patch("panelcast.pipelines.create_splits.validate_temporal_split"),
            patch("panelcast.pipelines.create_splits.assert_no_artist_overlap"),
            patch(
                "panelcast.pipelines.create_splits.create_split_assignments",
                return_value=[],
            ),
            patch(
                "panelcast.pipelines.create_splits.save_manifest",
                return_value=Path("manifest.json"),
            ),
        ):
            result = create_splits(config)

        summary = result.summary
        expected_excluded = total_artists - train["Artist"].nunique()
        assert summary["within_artist_temporal"]["artists_excluded"] == expected_excluded


# ============================================================================
# Tests: SplitResult field types
# ============================================================================


class TestSplitResultFields:
    def test_result_summary_is_serializable(self, tmp_path, artist_df):
        """Summary dict should be JSON-serializable."""
        n = len(artist_df)
        train = artist_df.iloc[: n // 2].copy()
        val = artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_artist_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.artist_disjoint_split",
                return_value=(train, val, test),
            ),
            patch("panelcast.pipelines.create_splits.validate_temporal_split"),
            patch("panelcast.pipelines.create_splits.assert_no_artist_overlap"),
            patch(
                "panelcast.pipelines.create_splits.create_split_assignments",
                return_value=[],
            ),
            patch(
                "panelcast.pipelines.create_splits.save_manifest",
                return_value=Path("manifest.json"),
            ),
        ):
            result = create_splits(config)

        # Summary should be JSON-serializable (no numpy types etc.)
        json_str = json.dumps(result.summary)
        assert len(json_str) > 0


# ============================================================================
# Tests: save_split_parquet with various dtypes
# ============================================================================


class TestSaveSplitParquetDtypes:
    def test_mixed_dtypes_roundtrip(self, tmp_path):
        """Save DataFrame with mixed dtypes: int, float, str, datetime."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "dt_col": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            }
        )
        path = tmp_path / "mixed.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        assert loaded.shape == (3, 4)
        assert loaded["str_col"].tolist() == ["a", "b", "c"]

    def test_large_dataframe(self, tmp_path):
        """Save larger DataFrame to exercise compression."""
        df = pd.DataFrame(
            {
                "Artist": [f"Artist_{i % 50}" for i in range(1000)],
                "Score": list(range(1000)),
            }
        )
        path = tmp_path / "large.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 1000
