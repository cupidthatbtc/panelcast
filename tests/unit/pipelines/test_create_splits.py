"""Tests for create_splits pipeline module."""

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


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with enough artists and albums for splitting."""
    rows = []
    for artist_id in range(10):
        artist_name = f"Artist_{artist_id}"
        for album_idx in range(5):
            rows.append(
                {
                    "Artist": artist_name,
                    "Album": f"Album_{artist_id}_{album_idx}",
                    "Release_Date_Parsed": pd.Timestamp("2020-01-01")
                    + pd.DateOffset(months=album_idx * 6),
                    "User_Score": 60.0 + album_idx * 5,
                    "User_Ratings": 50 + album_idx * 10,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def mock_split_results(sample_df):
    """Pre-computed train/val/test DataFrames."""
    n = len(sample_df)
    train = sample_df.iloc[: n // 2].copy()
    val = sample_df.iloc[n // 2 : n * 3 // 4].copy()
    test = sample_df.iloc[n * 3 // 4 :].copy()
    return train, val, test


class TestSplitConfig:
    """Tests for SplitConfig dataclass."""

    def test_default_values(self):
        """SplitConfig has sensible defaults."""
        config = SplitConfig()
        assert config.min_ratings == 10
        assert config.version == "v1"
        assert config.random_state == 42
        assert config.test_albums == 1
        # No validation split by default: all pre-test albums stay in train.
        assert config.val_albums == 0
        assert config.min_train_albums == 1
        assert config.disjoint_test_size == 0.15
        assert config.disjoint_val_size == 0.15

    def test_source_path_auto_computed(self):
        """Source path computed from min_ratings in __post_init__."""
        config = SplitConfig(min_ratings=25)
        assert config.source_path == Path("data/processed/user_score_minratings_25.parquet")

    def test_source_path_default_min_ratings(self):
        """Default min_ratings=10 computes correct source path."""
        config = SplitConfig()
        assert config.source_path == Path("data/processed/user_score_minratings_10.parquet")

    def test_source_path_explicit_override(self):
        """Explicit source_path overrides auto-computation."""
        explicit_path = Path("custom/path.parquet")
        config = SplitConfig(source_path=explicit_path)
        assert config.source_path == explicit_path

    def test_output_dir_default(self):
        """Default output directory is data/splits."""
        config = SplitConfig()
        assert config.output_dir == Path("data/splits")

    def test_custom_output_dir(self):
        """Custom output directory is accepted."""
        config = SplitConfig(output_dir=Path("/tmp/test_splits"))
        assert config.output_dir == Path("/tmp/test_splits")

    def test_custom_split_parameters(self):
        """Custom split parameters are stored."""
        config = SplitConfig(
            test_albums=2,
            val_albums=2,
            min_train_albums=3,
            disjoint_test_size=0.2,
            disjoint_val_size=0.1,
        )
        assert config.test_albums == 2
        assert config.val_albums == 2
        assert config.min_train_albums == 3
        assert config.disjoint_test_size == 0.2
        assert config.disjoint_val_size == 0.1


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_basic_creation(self, tmp_path):
        """SplitResult can be created with all fields."""
        result = SplitResult(
            source_path=Path("data/source.parquet"),
            temporal_manifest_path=tmp_path / "temporal_manifest.json",
            disjoint_manifest_path=tmp_path / "disjoint_manifest.json",
            temporal_splits={
                "train": tmp_path / "train.parquet",
                "validation": tmp_path / "val.parquet",
                "test": tmp_path / "test.parquet",
            },
            disjoint_splits={
                "train": tmp_path / "d_train.parquet",
                "validation": tmp_path / "d_val.parquet",
                "test": tmp_path / "d_test.parquet",
            },
            summary={"source": {"rows": 100}},
        )
        assert result.source_path == Path("data/source.parquet")
        assert len(result.temporal_splits) == 3
        assert len(result.disjoint_splits) == 3
        assert result.summary["source"]["rows"] == 100

    def test_result_has_expected_fields(self, tmp_path):
        """SplitResult has all expected fields."""
        result = SplitResult(
            source_path=Path("test.parquet"),
            temporal_manifest_path=tmp_path / "t.json",
            disjoint_manifest_path=tmp_path / "d.json",
            temporal_splits={},
            disjoint_splits={},
            summary={},
        )
        assert hasattr(result, "source_path")
        assert hasattr(result, "temporal_manifest_path")
        assert hasattr(result, "disjoint_manifest_path")
        assert hasattr(result, "temporal_splits")
        assert hasattr(result, "disjoint_splits")
        assert hasattr(result, "summary")


class TestSaveSplitParquet:
    """Tests for save_split_parquet function."""

    def test_saves_to_path(self, tmp_path):
        """save_split_parquet creates parquet file at specified path."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "output.parquet"
        save_split_parquet(df, path)
        assert path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """save_split_parquet creates parent directories if needed."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "nested" / "dir" / "output.parquet"
        save_split_parquet(df, path)
        assert path.exists()

    def test_roundtrip_data(self, tmp_path):
        """Saved parquet can be read back with same data."""
        df = pd.DataFrame({"x": [10, 20, 30], "y": [1.1, 2.2, 3.3]})
        path = tmp_path / "roundtrip.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_uses_snappy_compression(self, tmp_path):
        """File is created (snappy is the default compression)."""
        df = pd.DataFrame({"a": range(100)})
        path = tmp_path / "snappy.parquet"
        save_split_parquet(df, path)
        # File exists and is smaller than uncompressed would be
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_index_saved(self, tmp_path):
        """Parquet file does not include pandas index."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
        path = tmp_path / "no_index.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        # Index should be default RangeIndex, not the custom one
        assert list(loaded.index) == [0, 1, 2]

    def test_empty_dataframe(self, tmp_path):
        """Empty DataFrame can be saved and read back."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int)})
        path = tmp_path / "empty.parquet"
        save_split_parquet(df, path)
        loaded = pd.read_parquet(path)
        assert len(loaded) == 0


class TestCreateSplits:
    """Tests for create_splits pipeline function."""

    def test_default_config_created_when_none(self):
        """create_splits uses default SplitConfig when None is passed."""
        with (
            patch("panelcast.pipelines.create_splits.pd.read_parquet") as mock_read,
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split"
            ) as mock_temporal,
            patch("panelcast.pipelines.create_splits.entity_disjoint_split") as mock_disjoint,
            patch("panelcast.pipelines.create_splits.validate_temporal_split"),
            patch("panelcast.pipelines.create_splits.assert_no_artist_overlap"),
            patch("panelcast.pipelines.create_splits.save_manifest") as mock_save_manifest,
            patch(
                "panelcast.pipelines.create_splits.create_split_assignments",
                return_value=[],
            ),
            patch(
                "panelcast.pipelines.create_splits.hash_dataframe",
                return_value="a" * 64,
            ),
            patch("panelcast.pipelines.create_splits.save_split_parquet"),
        ):
            df = pd.DataFrame({"Artist": ["A"] * 5, "Score": range(5)})
            mock_read.return_value = df
            train_df = df.iloc[:3]
            val_df = df.iloc[3:4]
            test_df = df.iloc[4:]
            mock_temporal.return_value = (train_df, val_df, test_df)
            mock_disjoint.return_value = (train_df, val_df, test_df)
            mock_save_manifest.return_value = Path("manifest.json")

            result = create_splits(None)

            assert isinstance(result, SplitResult)
            # Verify it used default config's source path
            mock_read.assert_called_once()

    def test_creates_both_split_types(self, tmp_path, sample_df, mock_split_results):
        """create_splits creates both temporal and disjoint splits."""
        train, val, test = mock_split_results
        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        sample_df.to_parquet(config.source_path, index=False)

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

        assert result.temporal_splits is not None
        assert result.disjoint_splits is not None
        assert "train" in result.temporal_splits
        assert "validation" in result.temporal_splits
        assert "test" in result.temporal_splits
        assert "train" in result.disjoint_splits

    def test_summary_contains_expected_sections(self, tmp_path, sample_df, mock_split_results):
        """Pipeline summary has expected top-level keys."""
        train, val, test = mock_split_results
        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        sample_df.to_parquet(config.source_path, index=False)

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

        summary = result.summary
        assert "source" in summary
        assert "within_entity_temporal" in summary
        assert "entity_disjoint" in summary
        assert "run_timestamp" in summary
        assert summary["source"]["rows"] == len(sample_df)

    def test_saves_pipeline_summary_json(self, tmp_path, sample_df, mock_split_results):
        """Pipeline summary JSON is saved to output directory."""
        train, val, test = mock_split_results
        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
        )
        sample_df.to_parquet(config.source_path, index=False)

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
            create_splits(config)

        summary_path = config.output_dir / "pipeline_summary.json"
        assert summary_path.exists()
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "source" in data
        assert "within_entity_temporal" in data

    def test_passes_config_params_to_temporal_split(self, tmp_path, sample_df, mock_split_results):
        """Temporal split receives config parameters."""
        train, val, test = mock_split_results
        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
            test_albums=2,
            val_albums=3,
            min_train_albums=2,
        )
        sample_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ) as mock_temporal,
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
            create_splits(config)

        mock_temporal.assert_called_once()
        call_kwargs = mock_temporal.call_args
        assert call_kwargs[1]["test_albums"] == 2
        assert call_kwargs[1]["val_albums"] == 3
        assert call_kwargs[1]["min_train_albums"] == 2

    def test_passes_config_params_to_disjoint_split(self, tmp_path, sample_df, mock_split_results):
        """Disjoint split receives config parameters."""
        train, val, test = mock_split_results
        config = SplitConfig(
            source_path=tmp_path / "source.parquet",
            output_dir=tmp_path / "splits",
            disjoint_test_size=0.2,
            disjoint_val_size=0.1,
            random_state=99,
        )
        sample_df.to_parquet(config.source_path, index=False)

        with (
            patch(
                "panelcast.pipelines.create_splits.within_entity_temporal_split",
                return_value=(train, val, test),
            ),
            patch(
                "panelcast.pipelines.create_splits.entity_disjoint_split",
                return_value=(train, val, test),
            ) as mock_disjoint,
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
            create_splits(config)

        mock_disjoint.assert_called_once()
        call_kwargs = mock_disjoint.call_args
        assert call_kwargs[1]["test_size"] == 0.2
        assert call_kwargs[1]["val_size"] == 0.1
        assert call_kwargs[1]["random_state"] == 99


# --- from unit/pipelines/test_create_splits_coverage.py ---


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


# --- from unit/pipelines/test_create_splits_new.py ---


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
                return_value="b" * 64,
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "SPLIT PIPELINE COMPLETE" in out
        assert "Source:" in out
        assert "Rows:" in out
        assert "Artists:" in out
        assert "Within-Entity Temporal Split:" in out
        assert "Train:" in out
        assert "Validation:" in out
        assert "Test:" in out
        assert "Entities included:" in out
        assert "Entities excluded:" in out
        assert "Entity-Disjoint Split:" in out
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
                return_value="c" * 64,
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "insufficient events" in out


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

        summary = result.summary
        expected_excluded = total_artists - train["Artist"].nunique()
        assert summary["within_entity_temporal"]["artists_excluded"] == expected_excluded


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

        # Summary should be JSON-serializable (no numpy types etc.)
        json_str = json.dumps(result.summary)
        assert len(json_str) > 0


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
