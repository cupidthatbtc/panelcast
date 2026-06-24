"""Tests for split manifest I/O behavior."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from panelcast.data.manifests import (
    SplitAssignment,
    SplitManifest,
    SplitStats,
    create_split_assignments,
    generate_manifest_filename,
    load_manifest,
    save_manifest,
)


def _sample_manifest(**overrides) -> SplitManifest:
    """Create minimal split manifest for tests."""
    defaults = dict(
        version="v1",
        created_at="2026-01-01T00:00:00Z",
        split_type="within_entity_temporal",
        parameters={"test_albums": 1, "val_albums": 1},
        source_dataset={"path": "data/processed/user_score_minratings_10.parquet", "sha256": "abc"},
        splits={
            "train": SplitStats(row_count=10, unique_artists=3, sha256="train_hash"),
            "validation": SplitStats(row_count=3, unique_artists=3, sha256="val_hash"),
            "test": SplitStats(row_count=3, unique_artists=3, sha256="test_hash"),
        },
        assignments=[],
        content_hash="12345678deadbeef",
    )
    defaults.update(overrides)
    return SplitManifest(**defaults)


# =============================================================================
# SplitAssignment tests
# =============================================================================


class TestSplitAssignment:
    """Tests for SplitAssignment dataclass."""

    def test_basic_creation(self):
        a = SplitAssignment(original_row_id=1, split="train", reason="earlier_album")
        assert a.original_row_id == 1
        assert a.split == "train"
        assert a.reason == "earlier_album"

    def test_test_split(self):
        a = SplitAssignment(original_row_id=5, split="test", reason="last_album")
        assert a.split == "test"

    def test_validation_split(self):
        a = SplitAssignment(original_row_id=3, split="validation", reason="second_last")
        assert a.split == "validation"


# =============================================================================
# SplitStats tests
# =============================================================================


class TestSplitStats:
    """Tests for SplitStats dataclass."""

    def test_basic_creation(self):
        s = SplitStats(row_count=100, unique_artists=20, sha256="abcdef")
        assert s.row_count == 100
        assert s.unique_artists == 20
        assert s.sha256 == "abcdef"

    def test_zero_values(self):
        s = SplitStats(row_count=0, unique_artists=0, sha256="")
        assert s.row_count == 0


# =============================================================================
# SplitManifest tests
# =============================================================================


class TestSplitManifest:
    """Tests for SplitManifest dataclass."""

    def test_to_dict(self):
        m = _sample_manifest()
        d = m.to_dict()
        assert d["version"] == "v1"
        assert d["split_type"] == "within_entity_temporal"
        assert "train" in d["splits"]
        assert d["splits"]["train"]["row_count"] == 10

    def test_from_dict_roundtrip(self):
        m = _sample_manifest()
        d = m.to_dict()
        m2 = SplitManifest.from_dict(d)
        assert m2.version == m.version
        assert m2.split_type == m.split_type
        assert m2.splits["train"].row_count == 10

    def test_from_dict_with_assignments(self):
        m = _sample_manifest(
            assignments=[
                SplitAssignment(original_row_id=1, split="train", reason="earlier"),
                SplitAssignment(original_row_id=2, split="test", reason="last"),
            ]
        )
        d = m.to_dict()
        m2 = SplitManifest.from_dict(d)
        assert len(m2.assignments) == 2
        assert m2.assignments[0].original_row_id == 1
        assert m2.assignments[1].split == "test"

    def test_from_dict_missing_assignments_defaults_empty(self):
        d = _sample_manifest().to_dict()
        del d["assignments"]
        m = SplitManifest.from_dict(d)
        assert m.assignments == []

    def test_content_hash_default_empty(self):
        m = SplitManifest(
            version="v1",
            created_at="now",
            split_type="test",
            parameters={},
            source_dataset={},
            splits={},
        )
        assert m.content_hash == ""


# =============================================================================
# generate_manifest_filename tests
# =============================================================================


class TestGenerateManifestFilename:
    """Tests for generate_manifest_filename."""

    def test_format_with_datetime_and_hash(self):
        filename = generate_manifest_filename("v1", "12345678deadbeef")
        assert re.match(r"^split_v1_\d{8}_\d{6}_12345678\.json$", filename)

    def test_different_versions(self):
        f1 = generate_manifest_filename("v2", "abcdef1234567890")
        assert f1.startswith("split_v2_")

    def test_hash_prefix_is_8_chars(self):
        filename = generate_manifest_filename("v1", "aabbccdd11223344")
        # Extract hash prefix between last _ and .json
        parts = filename.replace(".json", "").split("_")
        assert len(parts[-1]) == 8
        assert parts[-1] == "aabbccdd"

    def test_short_hash_still_works(self):
        filename = generate_manifest_filename("v1", "abc")
        assert "abc" in filename


# =============================================================================
# save_manifest tests
# =============================================================================


class TestSaveManifest:
    """Tests for save_manifest."""

    def test_writes_canonical_and_versioned_files(self, tmp_path):
        manifest = _sample_manifest()
        returned_path = save_manifest(manifest, tmp_path)

        assert returned_path == tmp_path / "manifest.json"
        assert returned_path.exists()

        versioned_paths = list(tmp_path.glob("split_*.json"))
        assert len(versioned_paths) == 1
        assert versioned_paths[0].exists()

    def test_canonical_and_versioned_have_same_content(self, tmp_path):
        manifest = _sample_manifest()
        save_manifest(manifest, tmp_path)

        canonical_payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        versioned_paths = list(tmp_path.glob("split_*.json"))
        versioned_payload = json.loads(versioned_paths[0].read_text(encoding="utf-8"))
        assert canonical_payload == versioned_payload

    def test_creates_output_dir(self, tmp_path):
        subdir = tmp_path / "new_dir"
        manifest = _sample_manifest()
        save_manifest(manifest, subdir)
        assert subdir.exists()

    def test_creates_nested_output_dir(self, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        manifest = _sample_manifest()
        save_manifest(manifest, deep_dir)
        assert deep_dir.exists()

    def test_manifest_json_is_valid(self, tmp_path):
        manifest = _sample_manifest()
        save_manifest(manifest, tmp_path)
        data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert data["version"] == "v1"
        assert data["split_type"] == "within_entity_temporal"


# =============================================================================
# load_manifest tests
# =============================================================================


class TestLoadManifest:
    """Tests for load_manifest."""

    def test_roundtrip_save_load(self, tmp_path):
        manifest = _sample_manifest()
        save_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path / "manifest.json")
        assert loaded.version == manifest.version
        assert loaded.split_type == manifest.split_type
        assert loaded.splits["train"].row_count == 10

    def test_load_preserves_assignments(self, tmp_path):
        manifest = _sample_manifest(
            assignments=[
                SplitAssignment(original_row_id=1, split="test", reason="last"),
            ]
        )
        save_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path / "manifest.json")
        assert len(loaded.assignments) == 1
        assert loaded.assignments[0].split == "test"

    def test_load_preserves_parameters(self, tmp_path):
        manifest = _sample_manifest(parameters={"key": "value", "num": 42})
        save_manifest(manifest, tmp_path)
        loaded = load_manifest(tmp_path / "manifest.json")
        assert loaded.parameters["key"] == "value"
        assert loaded.parameters["num"] == 42


# =============================================================================
# create_split_assignments tests
# =============================================================================


class TestCreateSplitAssignments:
    """Tests for create_split_assignments."""

    def _make_df(self, n, artist="A"):
        return pd.DataFrame(
            {
                "original_row_id": list(range(n)),
                "Artist": [artist] * n,
                "Album": [f"album_{i}" for i in range(n)],
            }
        )

    def test_temporal_split_assigns_correct_reasons(self):
        train = self._make_df(2)
        val = self._make_df(1)
        test = self._make_df(1)
        assignments = create_split_assignments(
            train, val, test, split_type="within_entity_temporal"
        )
        splits = {a.split for a in assignments}
        assert splits == {"train", "validation", "test"}

    def test_temporal_split_total_count(self):
        train = self._make_df(3)
        val = self._make_df(1)
        test = self._make_df(1)
        assignments = create_split_assignments(
            train, val, test, split_type="within_entity_temporal"
        )
        assert len(assignments) == 5

    def test_temporal_split_test_reason_contains_artist(self):
        test = pd.DataFrame(
            {"original_row_id": [0], "Artist": ["Radiohead"], "Album": ["OK Computer"]}
        )
        train = self._make_df(1)
        val = self._make_df(1)
        assignments = create_split_assignments(
            train, val, test, split_type="within_entity_temporal"
        )
        test_assignments = [a for a in assignments if a.split == "test"]
        assert "Radiohead" in test_assignments[0].reason

    def test_entity_disjoint_split_assigns_reasons(self):
        train = self._make_df(2, artist="A")
        val = self._make_df(1, artist="B")
        test = self._make_df(1, artist="C")
        assignments = create_split_assignments(train, val, test, split_type="entity_disjoint")
        test_assignments = [a for a in assignments if a.split == "test"]
        assert test_assignments[0].reason == "artist_in_test_group"
        val_assignments = [a for a in assignments if a.split == "validation"]
        assert val_assignments[0].reason == "artist_in_validation_group"
        train_assignments = [a for a in assignments if a.split == "train"]
        assert train_assignments[0].reason == "artist_in_train_group"

    def test_temporal_truncates_long_artist_name(self):
        long_artist = "A" * 100
        test = pd.DataFrame({"original_row_id": [0], "Artist": [long_artist], "Album": ["X"]})
        train = self._make_df(1)
        val = self._make_df(1)
        assignments = create_split_assignments(
            train, val, test, split_type="within_entity_temporal"
        )
        test_a = [a for a in assignments if a.split == "test"]
        # Artist name truncated to 50 chars in reason
        assert len(test_a[0].reason) <= len("last_album_for_") + 50

    def test_empty_dataframes(self):
        empty = pd.DataFrame(columns=["original_row_id", "Artist", "Album"])
        assignments = create_split_assignments(
            empty, empty, empty, split_type="within_entity_temporal"
        )
        assert assignments == []
