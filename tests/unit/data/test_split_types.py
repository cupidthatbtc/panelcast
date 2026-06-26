"""Backward-compatible split-name aliasing (Track E).

The split strategies were renamed artist -> entity for domain portability.
These tests pin the contract that old, AOTY-flavored identifiers still resolve:
the manifest ``split_type`` normalization and on-disk directory fallback. (The
deprecated function aliases were removed in 0.3.0.)
"""

from __future__ import annotations

import pandas as pd
import pytest

from panelcast.data.manifests import (
    SplitManifest,
    SplitStats,
    create_split_assignments,
)
from panelcast.data.split import (
    entity_disjoint_split,
    within_entity_temporal_split,
)
from panelcast.data.split_types import (
    SplitType,
    legacy_split_name,
    resolve_split_dir,
    resolve_split_type,
    split_dir_name,
)


class TestResolveSplitType:
    def test_legacy_literals_map_to_canonical(self):
        assert resolve_split_type("within_artist_temporal") is SplitType.WITHIN_ENTITY_TEMPORAL
        assert resolve_split_type("artist_disjoint") is SplitType.ENTITY_DISJOINT

    def test_canonical_literals_pass_through(self):
        assert resolve_split_type("within_entity_temporal") is SplitType.WITHIN_ENTITY_TEMPORAL
        assert resolve_split_type("entity_disjoint") is SplitType.ENTITY_DISJOINT

    def test_enum_passthrough(self):
        assert resolve_split_type(SplitType.ENTITY_DISJOINT) is SplitType.ENTITY_DISJOINT

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_split_type("not_a_split")

    def test_dir_name_helpers(self):
        assert split_dir_name("within_artist_temporal") == "within_entity_temporal"
        assert legacy_split_name(SplitType.ENTITY_DISJOINT) == "artist_disjoint"


class TestManifestAliasResolution:
    def _legacy_manifest_dict(self, split_type: str) -> dict:
        return {
            "version": "v1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "split_type": split_type,
            "parameters": {"test_albums": 1},
            "source_dataset": {"path": "x", "sha256": "y", "row_count": 1, "unique_artists": 1},
            "splits": {"train": {"row_count": 1, "unique_artists": 1, "sha256": "z"}},
            "assignments": [],
            "content_hash": "h",
        }

    def test_old_manifest_split_type_resolves_to_canonical(self):
        manifest = SplitManifest.from_dict(self._legacy_manifest_dict("within_artist_temporal"))
        assert manifest.split_type == "within_entity_temporal"

    def test_old_disjoint_manifest_resolves(self):
        manifest = SplitManifest.from_dict(self._legacy_manifest_dict("artist_disjoint"))
        assert manifest.split_type == "entity_disjoint"
        assert isinstance(manifest.splits["train"], SplitStats)

    def test_create_assignments_accepts_legacy_literal(self):
        df = pd.DataFrame(
            {"original_row_id": [0, 1], "Artist": ["A", "B"]}
        )
        assignments = create_split_assignments(
            df.iloc[:1], df.iloc[1:2], df.iloc[:0], "within_artist_temporal", entity_col="Artist"
        )
        assert assignments  # non-empty; legacy literal accepted without error


class TestResolveSplitDir:
    def test_prefers_canonical_then_falls_back_to_legacy(self, tmp_path):
        # Only a legacy-named directory exists on disk.
        (tmp_path / "artist_disjoint").mkdir()
        resolved = resolve_split_dir(tmp_path, SplitType.ENTITY_DISJOINT)
        assert resolved == tmp_path / "artist_disjoint"

    def test_canonical_wins_when_present(self, tmp_path):
        (tmp_path / "entity_disjoint").mkdir()
        (tmp_path / "artist_disjoint").mkdir()
        resolved = resolve_split_dir(tmp_path, "artist_disjoint")
        assert resolved == tmp_path / "entity_disjoint"

    def test_canonical_target_when_nothing_exists(self, tmp_path):
        resolved = resolve_split_dir(tmp_path, SplitType.WITHIN_ENTITY_TEMPORAL)
        assert resolved == tmp_path / "within_entity_temporal"
