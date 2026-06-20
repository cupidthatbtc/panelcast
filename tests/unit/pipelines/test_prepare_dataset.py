"""Tests for prepare_dataset configuration behavior."""

from pathlib import Path

import pandas as pd
import pytest

from panelcast.pipelines.prepare_dataset import (
    PrepareConfig,
    PrepareResult,
    save_dataset,
)

# ============================================================================
# PrepareConfig Tests
# ============================================================================


class TestPrepareConfig:
    """Tests for PrepareConfig dataclass."""

    def test_reads_dataset_env_at_instantiation(self, monkeypatch):
        """raw_path default should reflect current env value for each new config."""
        monkeypatch.setenv("AOTY_DATASET_PATH", "first.csv")
        first = PrepareConfig()

        monkeypatch.setenv("AOTY_DATASET_PATH", "second.csv")
        second = PrepareConfig()

        assert first.raw_path == "first.csv"
        assert second.raw_path == "second.csv"

    def test_uses_default_when_env_unset(self, monkeypatch):
        """raw_path should fall back to repo default when env var is absent."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        cfg = PrepareConfig()
        assert cfg.raw_path == "data/raw/all_albums_full.csv"

    def test_default_output_dir(self):
        """Default output_dir is data/processed."""
        cfg = PrepareConfig()
        assert cfg.output_dir == "data/processed"

    def test_default_audit_dir(self):
        """Default audit_dir is data/audit."""
        cfg = PrepareConfig()
        assert cfg.audit_dir == "data/audit"

    def test_default_min_ratings_thresholds(self):
        """Default thresholds are [5, 10, 25]."""
        cfg = PrepareConfig()
        assert cfg.min_ratings_thresholds == [5, 10, 25]

    def test_default_min_critic_reviews(self):
        """Default min_critic_reviews is 1."""
        cfg = PrepareConfig()
        assert cfg.min_critic_reviews == 1

    def test_default_primary_min_ratings(self):
        """Default primary threshold is 10 and validated against the list."""
        cfg = PrepareConfig()
        assert cfg.primary_min_ratings == 10

    def test_primary_min_ratings_must_be_in_thresholds(self):
        """primary_min_ratings outside min_ratings_thresholds raises."""
        with pytest.raises(ValueError, match="primary_min_ratings"):
            PrepareConfig(min_ratings_thresholds=[5, 25], primary_min_ratings=10)

    def test_primary_min_ratings_custom_valid(self):
        """A custom primary threshold present in the list is accepted."""
        cfg = PrepareConfig(min_ratings_thresholds=[5, 25], primary_min_ratings=25)
        assert cfg.primary_min_ratings == 25

    def test_default_validate_raw_schema(self):
        """Default validate_raw_schema is False."""
        cfg = PrepareConfig()
        assert cfg.validate_raw_schema is False

    def test_default_dataset_hash_output_is_none(self):
        """Default dataset_hash_output is None."""
        cfg = PrepareConfig()
        assert cfg.dataset_hash_output is None

    def test_custom_values(self, monkeypatch):
        """PrepareConfig accepts custom values."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        cfg = PrepareConfig(
            raw_path="custom.csv",
            output_dir="custom_output",
            audit_dir="custom_audit",
            min_ratings_thresholds=[10, 50],
            min_critic_reviews=5,
            validate_raw_schema=True,
            dataset_hash_output="hash.txt",
        )
        assert cfg.raw_path == "custom.csv"
        assert cfg.output_dir == "custom_output"
        assert cfg.audit_dir == "custom_audit"
        assert cfg.min_ratings_thresholds == [10, 50]
        assert cfg.min_critic_reviews == 5
        assert cfg.validate_raw_schema is True
        assert cfg.dataset_hash_output == "hash.txt"

    def test_thresholds_are_independent_instances(self):
        """Each config instance gets its own threshold list."""
        cfg1 = PrepareConfig()
        cfg2 = PrepareConfig()
        cfg1.min_ratings_thresholds.append(100)
        assert 100 not in cfg2.min_ratings_thresholds


# ============================================================================
# PrepareResult Tests
# ============================================================================


class TestPrepareResult:
    """Tests for PrepareResult dataclass."""

    def test_fields_accessible(self):
        """PrepareResult fields are accessible."""
        from panelcast.data.ingest import LoadMetadata

        meta = LoadMetadata(
            file_path="test.csv",
            file_hash="abc123",
            load_timestamp="2026-01-01T00:00:00",
            row_count=100,
            column_count=10,
        )
        result = PrepareResult(
            load_metadata=meta,
            datasets_created={"test": Path("test.parquet")},
            audit_paths={"summary": Path("audit.json")},
            summary={"raw_rows": 100},
        )
        assert result.load_metadata.row_count == 100
        assert "test" in result.datasets_created
        assert "summary" in result.audit_paths
        assert result.summary["raw_rows"] == 100


# ============================================================================
# save_dataset Tests
# ============================================================================


class TestSaveDataset:
    """Tests for save_dataset function."""

    def test_saves_parquet_and_csv(self, tmp_path):
        """save_dataset creates both parquet and CSV files."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        paths = save_dataset(df, tmp_path, "test_data")

        assert paths["parquet"].exists()
        assert paths["csv"].exists()
        assert paths["parquet"].name == "test_data.parquet"
        assert paths["csv"].name == "test_data.csv"

    def test_creates_output_directory(self, tmp_path):
        """save_dataset creates output directory if not existing."""
        nested_dir = tmp_path / "nested" / "dir"
        df = pd.DataFrame({"col1": [1]})
        save_dataset(df, nested_dir, "test")
        assert nested_dir.exists()

    def test_parquet_roundtrip(self, tmp_path):
        """Saved parquet file can be read back."""
        df = pd.DataFrame({"num": [1, 2, 3], "text": ["a", "b", "c"]})
        paths = save_dataset(df, tmp_path, "roundtrip")
        loaded = pd.read_parquet(paths["parquet"])
        pd.testing.assert_frame_equal(loaded, df)

    def test_csv_roundtrip(self, tmp_path):
        """Saved CSV file can be read back."""
        df = pd.DataFrame({"num": [1, 2, 3], "text": ["a", "b", "c"]})
        paths = save_dataset(df, tmp_path, "roundtrip")
        loaded = pd.read_csv(paths["csv"])
        pd.testing.assert_frame_equal(loaded, df)

    def test_empty_dataframe(self, tmp_path):
        """save_dataset handles empty DataFrame."""
        df = pd.DataFrame({"col1": pd.Series([], dtype="int64")})
        paths = save_dataset(df, tmp_path, "empty")
        assert paths["parquet"].exists()
        loaded = pd.read_parquet(paths["parquet"])
        assert len(loaded) == 0

    def test_returns_path_dict(self, tmp_path):
        """save_dataset returns dict with 'parquet' and 'csv' keys."""
        df = pd.DataFrame({"col1": [1]})
        paths = save_dataset(df, tmp_path, "test")
        assert set(paths.keys()) == {"parquet", "csv"}
