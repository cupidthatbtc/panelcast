"""Additional coverage tests for prepare_dataset pipeline.

Targets uncovered lines/branches including:
- prepare_datasets() orchestration function
- main() CLI entry point
- _default_raw_dataset_path() runtime resolution
- dataset_hash_output artifact creation
- Integration with cleaning, ingest, and lineage modules
- Summary structure validation
"""

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

# ============================================================================
# Helpers
# ============================================================================


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


# ============================================================================
# Tests: _default_raw_dataset_path
# ============================================================================


class TestDefaultRawDatasetPath:
    def test_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("AOTY_DATASET_PATH", "/custom/path.csv")
        assert _default_raw_dataset_path() == "/custom/path.csv"

    def test_fallback_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        assert _default_raw_dataset_path() == "data/raw/all_albums_full.csv"


# ============================================================================
# Tests: prepare_datasets
# ============================================================================


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


# ============================================================================
# Tests: main() CLI entry point
# ============================================================================


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


# ============================================================================
# Tests: save_dataset additional cases
# ============================================================================


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
