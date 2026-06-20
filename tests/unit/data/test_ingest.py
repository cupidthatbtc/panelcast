"""Unit tests for data dimension extraction and ingestion."""

import dataclasses

import pandas as pd
import pytest

from panelcast.data.ingest import (
    DataDimensions,
    LoadMetadata,
    extract_data_dimensions,
    load_raw_albums,
    load_raw_dataset,
)

# =============================================================================
# DataDimensions tests
# =============================================================================


class TestDataDimensions:
    """Tests for DataDimensions dataclass."""

    def test_from_defaults_returns_conservative_estimates(self):
        dims = DataDimensions.from_defaults()
        assert dims.n_observations == 1000
        assert dims.n_artists == 100

    def test_from_defaults_source_indicates_fallback(self):
        dims = DataDimensions.from_defaults()
        assert "defaults" in dims.source

    def test_dataclass_frozen(self):
        dims = DataDimensions.from_defaults()
        with pytest.raises(dataclasses.FrozenInstanceError):
            dims.n_observations = 2000  # type: ignore[misc]

    def test_custom_creation(self):
        dims = DataDimensions(n_observations=500, n_artists=50, source="test")
        assert dims.n_observations == 500
        assert dims.n_artists == 50
        assert dims.source == "test"

    def test_equality(self):
        d1 = DataDimensions(n_observations=100, n_artists=10, source="x")
        d2 = DataDimensions(n_observations=100, n_artists=10, source="x")
        assert d1 == d2


# =============================================================================
# ExtractDataDimensions tests
# =============================================================================


class TestExtractDataDimensions:
    """Tests for extract_data_dimensions function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        csv_path = tmp_path / "test_albums.csv"
        csv_path.write_text(
            "Artist,User Ratings,Album\n"
            "Artist A,15,Album 1\n"
            "Artist A,20,Album 2\n"
            "Artist B,5,Album 3\n"
            "Artist B,12,Album 4\n"
            "Artist C,25,Album 5\n"
        )
        return csv_path

    def test_extract_from_nonexistent_file_returns_defaults(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.csv"
        dims = extract_data_dimensions(nonexistent)
        assert dims.n_observations == 1000
        assert dims.n_artists == 100
        assert "defaults" in dims.source

    def test_extract_applies_min_ratings_filter(self, sample_csv):
        dims = extract_data_dimensions(sample_csv, min_ratings=10)
        assert dims.n_observations == 4

    def test_extract_counts_unique_artists(self, sample_csv):
        dims = extract_data_dimensions(sample_csv, min_ratings=10)
        assert dims.n_artists == 3

    def test_extract_source_contains_filename(self, sample_csv):
        dims = extract_data_dimensions(sample_csv, min_ratings=10)
        assert "test_albums.csv" in dims.source

    def test_extract_handles_bom_encoding(self, tmp_path):
        csv_path = tmp_path / "bom_test.csv"
        csv_path.write_bytes(b"\xef\xbb\xbfArtist,User Ratings,Album\nArtist A,15,Album 1\n")
        dims = extract_data_dimensions(csv_path, min_ratings=10)
        assert dims.n_observations == 1
        assert dims.n_artists == 1

    def test_extract_returns_defaults_on_malformed_csv(self, tmp_path):
        csv_path = tmp_path / "malformed.csv"
        csv_path.write_text("This is not,a valid,CSV\nwith wrong,columns")
        dims = extract_data_dimensions(csv_path, min_ratings=10)
        assert dims.n_observations == 1000
        assert "defaults" in dims.source

    def test_high_min_ratings_filters_all(self, sample_csv):
        dims = extract_data_dimensions(sample_csv, min_ratings=1000)
        assert dims.n_observations == 0

    def test_zero_min_ratings_keeps_all(self, sample_csv):
        dims = extract_data_dimensions(sample_csv, min_ratings=0)
        assert dims.n_observations == 5
        assert dims.n_artists == 3

    def test_extract_uses_descriptor_raw_columns(self, tmp_path):
        """A non-AOTY descriptor selects raw header columns via the inverse map.

        The descriptor stores canonical names ("Sensor_Samples") plus a
        raw->canonical map; extraction must read the raw header
        ("Sensor Samples") so a domain with remapped columns still counts.
        """
        from panelcast.config.descriptor import DatasetDescriptor

        csv_path = tmp_path / "flights.csv"
        csv_path.write_text(
            "Airframe,Sensor Samples,Flight ID\nAF-1,30,F1\nAF-1,40,F2\nAF-2,3,F3\nAF-3,50,F4\n"
        )
        descriptor = DatasetDescriptor(
            name="aero",
            entity_col="Airframe",
            n_obs_col="Sensor_Samples",
            raw_column_map={"Sensor Samples": "Sensor_Samples"},
            secondary_target_col=None,
            secondary_prefix=None,
            secondary_n_obs_col=None,
        )
        dims = extract_data_dimensions(csv_path, min_ratings=10, descriptor=descriptor)
        # Kept rows (Sensor Samples >= 10): AF-1/30, AF-1/40, AF-3/50.
        assert dims.n_observations == 3
        assert dims.n_artists == 2  # AF-1, AF-3


# =============================================================================
# LoadMetadata tests
# =============================================================================


class TestLoadMetadata:
    """Tests for LoadMetadata dataclass."""

    def test_creation(self):
        meta = LoadMetadata(
            file_path="/data/test.csv",
            file_hash="abc123",
            load_timestamp="2024-01-01T00:00:00",
            row_count=100,
            column_count=10,
        )
        assert meta.file_path == "/data/test.csv"
        assert meta.file_hash == "abc123"
        assert meta.row_count == 100
        assert meta.column_count == 10


# =============================================================================
# load_raw_albums tests
# =============================================================================


class TestLoadRawAlbums:
    """Tests for load_raw_albums function."""

    def _make_csv(self, tmp_path, name="albums.csv", content=None):
        if content is None:
            content = (
                "Artist,Album,Year,Release Date,Genres,User Score,User Ratings,"
                "Tracks,Runtime (min),Avg Track Runtime (min),Album Type,All Artists\n"
                "A,Album1,2020,January 01 2020,Rock,80,100,10,40,4.0,LP,A\n"
            )
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_returns_dataframe_and_metadata(self, tmp_path):
        csv = self._make_csv(tmp_path)
        df, meta = load_raw_albums(csv, validate=False)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(meta, LoadMetadata)

    def test_adds_original_row_id(self, tmp_path):
        csv = self._make_csv(tmp_path)
        df, _ = load_raw_albums(csv, validate=False)
        assert "original_row_id" in df.columns

    def test_metadata_has_hash(self, tmp_path):
        csv = self._make_csv(tmp_path)
        _, meta = load_raw_albums(csv, validate=False)
        assert len(meta.file_hash) > 0

    def test_metadata_has_row_count(self, tmp_path):
        csv = self._make_csv(tmp_path)
        _, meta = load_raw_albums(csv, validate=False)
        assert meta.row_count == 1

    def test_metadata_has_timestamp(self, tmp_path):
        csv = self._make_csv(tmp_path)
        _, meta = load_raw_albums(csv, validate=False)
        assert len(meta.load_timestamp) > 0

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_albums(tmp_path / "nonexistent.csv")

    def test_metadata_file_path_is_absolute(self, tmp_path):
        csv = self._make_csv(tmp_path)
        _, meta = load_raw_albums(csv, validate=False)
        from pathlib import Path

        assert Path(meta.file_path).is_absolute()


# =============================================================================
# load_raw_dataset tests (legacy)
# =============================================================================


class TestLoadRawDataset:
    """Tests for legacy load_raw_dataset helper."""

    def test_load_raw_dataset_honors_encoding(self, tmp_path):
        csv_path = tmp_path / "latin1.csv"
        csv_path.write_bytes("Artist,Album\nBjörk,Debut\n".encode("latin-1"))
        df = load_raw_dataset(str(csv_path), encoding="latin-1")
        assert df.loc[0, "Artist"] == "Björk"
        assert "original_row_id" in df.columns

    def test_adds_original_row_id(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Artist,Album\nA,X\nB,Y\n")
        df = load_raw_dataset(str(csv_path))
        assert "original_row_id" in df.columns
        assert df["original_row_id"].tolist() == [0, 1]

    def test_preserves_all_columns(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Artist,Album,Extra\nA,X,1\n")
        df = load_raw_dataset(str(csv_path))
        assert "Extra" in df.columns
