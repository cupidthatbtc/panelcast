"""Coverage-targeted tests for pipelines/create_splits.py.

Tests target missed lines/branches:
- create_splits with None config (default config __post_init__ branch)
- main() CLI entry point
- SplitConfig __post_init__ edge cases
- Manifest creation / content hash computation
- End-to-end split parquet file saving
- Summary JSON integrity
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from panelcast.pipelines.create_splits import (
    SplitConfig,
    SplitResult,
    create_splits,
    main,
    save_split_parquet,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multi_artist_df():
    """DataFrame with multiple artists, each having enough albums for splitting."""
    rows = []
    for artist_id in range(15):
        artist_name = f"Artist_{artist_id}"
        for album_idx in range(6):
            rows.append(
                {
                    "Artist": artist_name,
                    "Album": f"Album_{artist_id}_{album_idx}",
                    "Release_Date_Parsed": pd.Timestamp("2018-01-01")
                    + pd.DateOffset(months=album_idx * 6),
                    "User_Score": 55.0 + album_idx * 4 + artist_id * 0.5,
                    "User_Ratings": 30 + album_idx * 10,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# TestSplitConfigPostInit
# =============================================================================


class TestSplitConfigPostInit:
    """Tests for SplitConfig __post_init__ branch coverage."""

    def test_source_path_none_triggers_computation(self):
        """When source_path is None (default), __post_init__ computes it."""
        config = SplitConfig(min_ratings=50)
        assert config.source_path == Path("data/processed/user_score_minratings_50.parquet")

    def test_source_path_explicit_skips_computation(self):
        """When source_path is explicitly set, __post_init__ does not override."""
        explicit = Path("/custom/data.parquet")
        config = SplitConfig(min_ratings=50, source_path=explicit)
        assert config.source_path == explicit

    def test_default_min_ratings_source_path(self):
        """Default min_ratings=10 produces expected path."""
        config = SplitConfig()
        assert "minratings_10" in str(config.source_path)


# =============================================================================
# TestCreateSplitsManifestAndHashing
# =============================================================================


class TestCreateSplitsManifestAndHashing:
    """Tests for manifest creation and content hash computation in create_splits."""

    def test_temporal_and_disjoint_manifests_created(self, tmp_path, multi_artist_df):
        """Both temporal and disjoint manifests are created via save_manifest."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        multi_artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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
            ) as mock_save_manifest,
        ):
            result = create_splits(config)

        # save_manifest called twice: once for temporal, once for disjoint
        assert mock_save_manifest.call_count == 2

    def test_summary_json_has_all_fields(self, tmp_path, multi_artist_df):
        """Pipeline summary JSON contains all expected sections."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        multi_artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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

        summary_path = config.output_dir / "pipeline_summary.json"
        assert summary_path.exists()

        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        # Check all required keys
        assert "run_timestamp" in data
        assert "source" in data
        assert data["source"]["rows"] == len(multi_artist_df)
        assert data["source"]["artists"] == multi_artist_df["Artist"].nunique()
        assert "within_entity_temporal" in data
        assert "entity_disjoint" in data
        assert "train_rows" in data["within_entity_temporal"]
        assert "val_rows" in data["within_entity_temporal"]
        assert "test_rows" in data["within_entity_temporal"]
        assert "artists_included" in data["within_entity_temporal"]
        assert "artists_excluded" in data["within_entity_temporal"]
        assert "train_artists" in data["entity_disjoint"]
        assert "val_artists" in data["entity_disjoint"]
        assert "test_artists" in data["entity_disjoint"]


# =============================================================================
# TestCreateSplitsSavesParquetFiles
# =============================================================================


class TestCreateSplitsSavesParquetFiles:
    """Tests that create_splits saves actual parquet files."""

    def test_temporal_parquet_files_created(self, tmp_path, multi_artist_df):
        """Temporal split parquet files are written to disk."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        multi_artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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

        # Temporal split files
        temporal_dir = config.output_dir / "within_entity_temporal"
        assert (temporal_dir / "train.parquet").exists()
        assert (temporal_dir / "validation.parquet").exists()
        assert (temporal_dir / "test.parquet").exists()

    def test_disjoint_parquet_files_created(self, tmp_path, multi_artist_df):
        """Disjoint split parquet files are written to disk."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        multi_artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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

        disjoint_dir = config.output_dir / "entity_disjoint"
        assert (disjoint_dir / "train.parquet").exists()
        assert (disjoint_dir / "validation.parquet").exists()
        assert (disjoint_dir / "test.parquet").exists()


# =============================================================================
# TestCreateSplitsResultFields
# =============================================================================


class TestCreateSplitsResultFields:
    """Tests for SplitResult field population."""

    def test_result_has_correct_source_path(self, tmp_path, multi_artist_df):
        """SplitResult.source_path matches config."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        multi_artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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

        assert result.source_path == config.source_path

    def test_result_temporal_splits_paths(self, tmp_path, multi_artist_df):
        """SplitResult.temporal_splits contains correct path keys."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        multi_artist_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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

        assert set(result.temporal_splits.keys()) == {"train", "validation", "test"}
        assert set(result.disjoint_splits.keys()) == {"train", "validation", "test"}


# =============================================================================
# TestMainCliEntryPoint
# =============================================================================


class TestMainCliEntryPoint:
    """Tests for main() CLI entry point."""

    def test_main_calls_create_splits_and_prints(self, tmp_path, multi_artist_df, capsys):
        """main() invokes create_splits with defaults and prints summary."""
        n = len(multi_artist_df)
        train = multi_artist_df.iloc[: n // 2].copy()
        val = multi_artist_df.iloc[n // 2 : n * 3 // 4].copy()
        test = multi_artist_df.iloc[n * 3 // 4 :].copy()

        source_path = tmp_path / "data" / "processed" / "user_score_minratings_10.parquet"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        multi_artist_df.to_parquet(source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.pd.read_parquet",
                return_value=multi_artist_df,
            ),
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
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
                return_value="a" * 64,
            ),
        ):
            main()

        captured = capsys.readouterr()
        assert "SPLIT PIPELINE COMPLETE" in captured.out
        assert "Within-Entity Temporal Split" in captured.out
        assert "Entity-Disjoint Split" in captured.out


# =============================================================================
# TestSaveSplitParquetEdgeCases
# =============================================================================


class TestSaveSplitParquetEdgeCases:
    """Edge case tests for save_split_parquet."""

    def test_single_row_dataframe(self, tmp_path):
        """Save and read back a single-row DataFrame."""
        df = pd.DataFrame({"Artist": ["A"], "Score": [75.0]})
        path = tmp_path / "single.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["Artist"] == "A"

    def test_wide_dataframe(self, tmp_path):
        """Save DataFrame with many columns."""
        df = pd.DataFrame({f"col_{i}": range(5) for i in range(50)})
        path = tmp_path / "wide.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        assert loaded.shape == (5, 50)
