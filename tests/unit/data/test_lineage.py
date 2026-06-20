"""Unit tests for data lineage and audit logging."""

import json
from dataclasses import asdict

import pandas as pd
import pytest

from panelcast.data.lineage import (
    AuditLogger,
    ExclusionRecord,
    FilterStats,
    record_lineage,
)

# =============================================================================
# ExclusionRecord tests
# =============================================================================


class TestExclusionRecord:
    """Tests for ExclusionRecord dataclass."""

    def test_basic_creation(self):
        record = ExclusionRecord(
            original_row_id=1,
            artist="Artist A",
            album="Album 1",
            reason="missing_score",
        )
        assert record.original_row_id == 1
        assert record.artist == "Artist A"
        assert record.album == "Album 1"
        assert record.reason == "missing_score"
        assert record.value is None

    def test_with_value(self):
        record = ExclusionRecord(
            original_row_id=2,
            artist="B",
            album="Album 2",
            reason="low_ratings",
            value=3,
        )
        assert record.value == 3

    def test_timestamp_auto_populated(self):
        record = ExclusionRecord(original_row_id=0, artist="A", album="X", reason="test")
        assert record.timestamp is not None
        assert len(record.timestamp) > 0

    def test_asdict_roundtrip(self):
        record = ExclusionRecord(
            original_row_id=5,
            artist="Artist",
            album="Album",
            reason="reason",
            value=42,
        )
        d = asdict(record)
        assert d["original_row_id"] == 5
        assert d["artist"] == "Artist"
        assert d["value"] == 42

    def test_custom_timestamp(self):
        record = ExclusionRecord(
            original_row_id=0,
            artist="A",
            album="X",
            reason="test",
            timestamp="2024-01-01T00:00:00",
        )
        assert record.timestamp == "2024-01-01T00:00:00"


# =============================================================================
# FilterStats tests
# =============================================================================


class TestFilterStats:
    """Tests for FilterStats dataclass."""

    def test_basic_creation(self):
        stats = FilterStats(
            filter_name="missing_score",
            rows_before=100,
            rows_excluded=10,
            rows_after=90,
        )
        assert stats.filter_name == "missing_score"
        assert stats.rows_before == 100
        assert stats.rows_excluded == 10
        assert stats.rows_after == 90

    def test_exclusion_rate(self):
        stats = FilterStats(filter_name="test", rows_before=200, rows_excluded=50, rows_after=150)
        assert stats.exclusion_rate == pytest.approx(0.25)

    def test_exclusion_rate_zero_before(self):
        stats = FilterStats(filter_name="test", rows_before=0, rows_excluded=0, rows_after=0)
        assert stats.exclusion_rate == 0.0

    def test_exclusion_rate_all_excluded(self):
        stats = FilterStats(filter_name="test", rows_before=50, rows_excluded=50, rows_after=0)
        assert stats.exclusion_rate == pytest.approx(1.0)

    def test_exclusion_rate_none_excluded(self):
        stats = FilterStats(filter_name="test", rows_before=100, rows_excluded=0, rows_after=100)
        assert stats.exclusion_rate == pytest.approx(0.0)


# =============================================================================
# AuditLogger tests
# =============================================================================


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_init_creates_output_dir(self, tmp_path):
        subdir = tmp_path / "audit_logs"
        logger = AuditLogger(output_dir=subdir)
        assert subdir.exists()
        assert subdir.is_dir()

    def test_init_custom_run_id(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test_run_123")
        assert logger.run_id == "test_run_123"

    def test_init_auto_run_id(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path)
        assert len(logger.run_id) > 0

    def test_log_exclusion_adds_record(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        logger.log_exclusion(row_id=1, artist="Artist A", album="Album 1", reason="test_reason")
        assert len(logger.exclusions) == 1
        assert logger.exclusions[0].original_row_id == 1
        assert logger.exclusions[0].reason == "test_reason"

    def test_log_exclusion_truncates_long_names(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        long_name = "A" * 200
        logger.log_exclusion(row_id=1, artist=long_name, album=long_name, reason="test")
        assert len(logger.exclusions[0].artist) == 100
        assert len(logger.exclusions[0].album) == 100

    def test_log_exclusion_with_value(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        logger.log_exclusion(row_id=1, artist="A", album="B", reason="low_score", value=5.0)
        assert logger.exclusions[0].value == 5.0

    def test_log_exclusions_bulk(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        df = pd.DataFrame(
            {
                "original_row_id": [0, 1, 2],
                "Artist": ["A", "B", "C"],
                "Album": ["X", "Y", "Z"],
                "User_Score": [10.0, 20.0, 30.0],
            }
        )
        logger.log_exclusions_bulk(df, reason="bulk_reason", value_col="User_Score")
        assert len(logger.exclusions) == 3
        assert logger.exclusions[0].value == 10.0
        assert logger.exclusions[2].artist == "C"

    def test_log_exclusions_bulk_no_value_col(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        df = pd.DataFrame(
            {
                "original_row_id": [0],
                "Artist": ["A"],
                "Album": ["X"],
            }
        )
        logger.log_exclusions_bulk(df, reason="no_val")
        assert logger.exclusions[0].value is None

    def test_log_filter_stats(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        logger.log_filter_stats("test_filter", rows_before=100, rows_excluded=10, rows_after=90)
        assert len(logger.filter_stats) == 1
        assert logger.filter_stats[0].filter_name == "test_filter"
        assert logger.filter_stats[0].rows_before == 100

    def test_save_creates_exclusions_jsonl(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test_run")
        logger.log_exclusion(row_id=1, artist="A", album="B", reason="r1")
        logger.log_exclusion(row_id=2, artist="C", album="D", reason="r2")
        paths = logger.save()

        assert "exclusions" in paths
        assert paths["exclusions"].exists()
        lines = paths["exclusions"].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["original_row_id"] == 1
        assert record["artist"] == "A"

    def test_save_creates_summary_json(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test_run")
        logger.log_exclusion(row_id=1, artist="A", album="B", reason="r1")
        logger.log_filter_stats("f1", rows_before=10, rows_excluded=1, rows_after=9)
        paths = logger.save()

        assert "summary" in paths
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        assert summary["run_id"] == "test_run"
        assert summary["total_exclusions"] == 1
        assert summary["exclusions_by_reason"]["r1"] == 1
        assert len(summary["filter_stats"]) == 1

    def test_save_empty_logger(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="empty")
        paths = logger.save()
        assert paths["exclusions"].exists()
        assert paths["summary"].exists()
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        assert summary["total_exclusions"] == 0

    def test_count_by_reason(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        logger.log_exclusion(row_id=1, artist="A", album="B", reason="r1")
        logger.log_exclusion(row_id=2, artist="C", album="D", reason="r1")
        logger.log_exclusion(row_id=3, artist="E", album="F", reason="r2")
        counts = logger._count_by_reason()
        assert counts["r1"] == 2
        assert counts["r2"] == 1

    def test_count_by_reason_sorted_descending(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        for i in range(5):
            logger.log_exclusion(row_id=i, artist="A", album="B", reason="common")
        logger.log_exclusion(row_id=99, artist="A", album="B", reason="rare")
        counts = logger._count_by_reason()
        keys = list(counts.keys())
        assert keys[0] == "common"
        assert keys[1] == "rare"

    def test_get_summary_without_saving(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        logger.log_exclusion(row_id=1, artist="A", album="B", reason="r1")
        logger.log_filter_stats("f1", rows_before=10, rows_excluded=1, rows_after=9)
        summary = logger.get_summary()
        assert summary["total_exclusions"] == 1
        assert "r1" in summary["exclusions_by_reason"]
        assert len(summary["filter_stats"]) == 1

    def test_get_summary_empty(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        summary = logger.get_summary()
        assert summary["total_exclusions"] == 0
        assert summary["exclusions_by_reason"] == {}
        assert summary["filter_stats"] == []

    def test_multiple_filter_stats(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="test")
        logger.log_filter_stats("f1", 100, 10, 90)
        logger.log_filter_stats("f2", 90, 5, 85)
        assert len(logger.filter_stats) == 2
        assert logger.filter_stats[0].filter_name == "f1"
        assert logger.filter_stats[1].filter_name == "f2"

    def test_save_file_naming(self, tmp_path):
        logger = AuditLogger(output_dir=tmp_path, run_id="my_run")
        paths = logger.save()
        assert "my_run" in paths["exclusions"].name
        assert "my_run" in paths["summary"].name

    def test_nested_output_dir_creation(self, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        logger = AuditLogger(output_dir=deep_dir)
        assert deep_dir.exists()


# =============================================================================
# record_lineage tests
# =============================================================================


class TestRecordLineage:
    """Tests for record_lineage function."""

    def test_record_lineage_does_not_raise(self):
        record_lineage(
            step_name="test_step",
            inputs={"file": "input.csv"},
            outputs={"file": "output.csv"},
        )

    def test_record_lineage_with_empty_dicts(self):
        record_lineage(step_name="empty", inputs={}, outputs={})
