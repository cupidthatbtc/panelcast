"""Tests for prepare_dataset configuration behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from panelcast.pipelines.prepare_dataset import (
    PrepareConfig,
    PrepareResult,
    _default_raw_dataset_path,
    main,
    prepare_datasets,
    save_dataset,
)


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


# --- from unit/pipelines/test_prepare_dataset_new.py ---


def _make_raw_df(n=50):
    """Create a mock raw DataFrame with expected columns."""
    return pd.DataFrame(
        {
            "Artist": [f"Artist_{i % 10}" for i in range(n)],
            "Album": [f"Album_{i}" for i in range(n)],
            "Release_Date_Parsed": pd.date_range("2018-01-01", periods=n, freq="ME"),
            "User_Score": [60.0 + i * 0.5 for i in range(n)],
            "Critic_Score": [55.0 + i * 0.3 for i in range(n)],
            "User_Ratings": [20 + i * 5 for i in range(n)],
            "Critic_Reviews": [3 + i % 5 for i in range(n)],
        }
    )


def _make_load_metadata():
    """Create a mock LoadMetadata."""
    from panelcast.data.ingest import LoadMetadata

    return LoadMetadata(
        file_path="data/raw/all_albums_full.csv",
        file_hash="abc123def456" * 5,
        load_timestamp="2026-01-01T00:00:00",
        row_count=50,
        column_count=7,
    )


def _make_audit_logger_mock():
    """Create a mock AuditLogger."""
    logger = MagicMock()
    logger.save.return_value = {"summary": Path("audit/summary.json")}
    logger.get_summary.return_value = {
        "total_exclusions": 5,
        "exclusions_by_reason": {"missing_score": 3, "low_ratings": 2},
    }
    return logger


class TestDefaultRawDatasetPath:
    def test_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("AOTY_DATASET_PATH", "/custom/path.csv")
        assert _default_raw_dataset_path() == "/custom/path.csv"

    def test_fallback_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        assert _default_raw_dataset_path() == "data/raw/all_albums_full.csv"


class TestPrepareDatasets:
    def test_creates_all_user_datasets(self, tmp_path, monkeypatch):
        """prepare_datasets should create user score datasets for each threshold."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        config = PrepareConfig(
            raw_path=str(tmp_path / "raw.csv"),
            output_dir=str(tmp_path / "processed"),
            audit_dir=str(tmp_path / "audit"),
            min_ratings_thresholds=[5, 10],
            min_critic_reviews=1,
        )

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                side_effect=lambda df, descriptor, min_obs, target="primary", logger=None: (
                    raw_df.iloc[:30].copy() if target == "primary" else raw_df.iloc[:20].copy()
                ),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            result = prepare_datasets(config)

        assert isinstance(result, PrepareResult)
        assert result.load_metadata.row_count == 50
        # Should have user_score datasets for each threshold + critic + cleaned_all
        assert "user_score_minratings_5" in result.datasets_created
        assert "user_score_minratings_10" in result.datasets_created
        assert "critic_score" in result.datasets_created
        assert "cleaned_all" in result.datasets_created

    def test_default_config_when_none(self, tmp_path, monkeypatch):
        """prepare_datasets should use default PrepareConfig when None."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            result = prepare_datasets(None)

        # Default thresholds are [5, 10, 25]
        assert "user_score_minratings_5" in result.datasets_created
        assert "user_score_minratings_10" in result.datasets_created
        assert "user_score_minratings_25" in result.datasets_created

    def test_dataset_hash_output_written(self, tmp_path, monkeypatch):
        """dataset_hash_output should create a hash file when set."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        hash_path = tmp_path / "hashes" / "dataset.hash"
        config = PrepareConfig(
            raw_path=str(tmp_path / "raw.csv"),
            output_dir=str(tmp_path / "processed"),
            audit_dir=str(tmp_path / "audit"),
            min_ratings_thresholds=[10],
            dataset_hash_output=str(hash_path),
        )

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            prepare_datasets(config)

        assert hash_path.exists()
        content = hash_path.read_text(encoding="utf-8").strip()
        assert content == load_meta.file_hash

    def test_summary_has_expected_keys(self, tmp_path, monkeypatch):
        """Summary should contain raw_rows, cleaned_rows, datasets, exclusions."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        config = PrepareConfig(
            raw_path=str(tmp_path / "raw.csv"),
            output_dir=str(tmp_path / "processed"),
            audit_dir=str(tmp_path / "audit"),
            min_ratings_thresholds=[10],
        )

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            result = prepare_datasets(config)

        summary = result.summary
        assert "raw_rows" in summary
        assert "raw_hash" in summary
        assert "cleaned_rows" in summary
        assert "datasets" in summary
        assert "exclusions" in summary
        assert summary["raw_rows"] == 50
        assert summary["cleaned_rows"] == 50

    def test_audit_paths_returned(self, tmp_path, monkeypatch):
        """audit_paths should come from AuditLogger.save()."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        config = PrepareConfig(
            raw_path=str(tmp_path / "raw.csv"),
            output_dir=str(tmp_path / "processed"),
            audit_dir=str(tmp_path / "audit"),
            min_ratings_thresholds=[10],
        )

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            result = prepare_datasets(config)

        assert "summary" in result.audit_paths
        audit_logger.save.assert_called_once()


class TestMainCli:
    def test_main_prints_summary(self, tmp_path, capsys, monkeypatch):
        """main() should print formatted summary output."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "DATASET PREPARATION COMPLETE" in out
        assert "Raw data:" in out
        assert "File hash:" in out
        assert "Datasets created:" in out
        assert "Audit log:" in out
        assert "Total exclusions:" in out
        assert "Exclusions by reason:" in out

    def test_main_shows_exclusion_reasons(self, tmp_path, capsys, monkeypatch):
        """main() should list exclusion reasons."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        raw_df = _make_raw_df()
        load_meta = _make_load_metadata()
        audit_logger = _make_audit_logger_mock()

        with (
            patch(
                "panelcast.pipelines.prepare_dataset.load_raw_albums",
                return_value=(raw_df, load_meta),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.clean_albums",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.filter_for_target_model",
                return_value=raw_df.copy(),
            ),
            patch(
                "panelcast.pipelines.prepare_dataset.AuditLogger",
                return_value=audit_logger,
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "missing_score" in out
        assert "low_ratings" in out


class TestSaveDatasetNew:
    def test_returns_absolute_paths(self, tmp_path):
        """Returned paths should be Path objects."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        paths = save_dataset(df, tmp_path, "abs_test")
        assert isinstance(paths["parquet"], Path)
        assert isinstance(paths["csv"], Path)

    def test_large_dataframe_saved(self, tmp_path):
        """Large DataFrame should be saved without error."""
        df = pd.DataFrame(
            {
                "col": range(10000),
                "text": [f"row_{i}" for i in range(10000)],
            }
        )
        paths = save_dataset(df, tmp_path, "large")
        loaded = pd.read_parquet(paths["parquet"])
        assert len(loaded) == 10000
