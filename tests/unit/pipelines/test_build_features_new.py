"""Additional coverage tests for build_features pipeline.

Targets uncovered lines including:
- build_features main orchestration function
- _transform_with_train_history score column masking branches
- get_feature_blocks with various combinations
- _assign_n_reviews edge cases
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.build_features import (
    _assign_n_reviews,
    _transform_with_train_history,
    build_features,
    get_default_feature_blocks,
    get_feature_blocks,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_ctx(**overrides):
    defaults = {
        "seed": 42,
        "enable_genre": True,
        "enable_artist": True,
        "enable_temporal": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_split_dfs(n_train=10, n_val=3, n_test=3):
    """Create DataFrames with all required columns for feature blocks."""
    rng = np.random.default_rng(42)

    def _make(n, start_year):
        dates = pd.date_range(f"{start_year}-01-01", periods=n, freq="ME")
        return pd.DataFrame(
            {
                "Artist": [f"artist_{i % 5}" for i in range(n)],
                "Album": [f"album_{start_year}_{i}" for i in range(n)],
                "Release_Date_Parsed": dates,
                "Year": [d.year for d in dates],
                "date_risk": ["low"] * n,
                "User_Score": rng.uniform(60, 95, n).astype(np.float32),
                "Critic_Score": rng.uniform(50, 90, n).astype(np.float32),
                "User_Ratings": rng.integers(10, 200, n),
                "Album_Type": ["LP"] * n,
                "Genres": ["Rock; Indie"] * n,
                "is_collaboration": [0] * n,
                "collab_type": ["solo"] * n,
                "num_artists": [1] * n,
            }
        )

    train = _make(n_train, 2018)
    val = _make(n_val, 2020)
    test = _make(n_test, 2021)
    return train, val, test


# ============================================================================
# Tests: build_features main function
# ============================================================================


class TestBuildFeatures:
    """Tests for the build_features orchestration function."""

    def test_build_features_creates_manifest(self, tmp_path, monkeypatch):
        """build_features should create a manifest.json with proper structure."""
        train, val, test = _make_split_dfs()

        # Set up directories
        splits_root = tmp_path / "data" / "splits"
        features_dir = tmp_path / "data" / "features"

        for split_name in ["within_entity_temporal", "entity_disjoint"]:
            split_dir = splits_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            train.to_parquet(split_dir / "train.parquet")
            val.to_parquet(split_dir / "validation.parquet")
            test.to_parquet(split_dir / "test.parquet")

        # Patch Path calls to use tmp_path
        monkeypatch.setattr(
            "panelcast.pipelines.build_features.Path",
            lambda p: tmp_path / p,
        )

        ctx = _make_ctx(enable_genre=False, enable_artist=True, enable_temporal=True)

        manifest = build_features(ctx)

        assert "seed" in manifest
        assert manifest["seed"] == 42
        assert "blocks" in manifest
        assert "feature_names" in manifest
        assert "split_features" in manifest
        assert "within_entity_temporal" in manifest["split_features"]
        assert "entity_disjoint" in manifest["split_features"]

        # Check that manifest was written to disk
        manifest_path = features_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["seed"] == 42

    def test_build_features_ablation_flags(self, tmp_path, monkeypatch):
        """Feature ablation flags should be recorded in manifest."""
        train, val, test = _make_split_dfs()

        splits_root = tmp_path / "data" / "splits"
        for split_name in ["within_entity_temporal", "entity_disjoint"]:
            split_dir = splits_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            train.to_parquet(split_dir / "train.parquet")
            val.to_parquet(split_dir / "validation.parquet")
            test.to_parquet(split_dir / "test.parquet")

        monkeypatch.setattr(
            "panelcast.pipelines.build_features.Path",
            lambda p: tmp_path / p,
        )

        ctx = _make_ctx(enable_genre=False, enable_artist=False, enable_temporal=False)
        manifest = build_features(ctx)

        assert manifest["feature_ablation"]["enable_genre"] is False
        assert manifest["feature_ablation"]["enable_artist"] is False
        assert manifest["feature_ablation"]["enable_temporal"] is False

    def test_build_features_creates_backward_compat_files(self, tmp_path, monkeypatch):
        """Root-level feature parquet files are created for backward compatibility."""
        train, val, test = _make_split_dfs()

        splits_root = tmp_path / "data" / "splits"
        features_dir = tmp_path / "data" / "features"
        for split_name in ["within_entity_temporal", "entity_disjoint"]:
            split_dir = splits_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            train.to_parquet(split_dir / "train.parquet")
            val.to_parquet(split_dir / "validation.parquet")
            test.to_parquet(split_dir / "test.parquet")

        monkeypatch.setattr(
            "panelcast.pipelines.build_features.Path",
            lambda p: tmp_path / p,
        )

        ctx = _make_ctx(enable_genre=False, enable_artist=True, enable_temporal=False)
        build_features(ctx)

        # Root-level files for backward compatibility
        assert (features_dir / "train_features.parquet").exists()
        assert (features_dir / "validation_features.parquet").exists()
        assert (features_dir / "test_features.parquet").exists()

    def test_build_features_split_manifests_have_n_reviews(self, tmp_path, monkeypatch):
        """Split manifests should include n_reviews statistics."""
        train, val, test = _make_split_dfs()

        splits_root = tmp_path / "data" / "splits"
        for split_name in ["within_entity_temporal", "entity_disjoint"]:
            split_dir = splits_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            train.to_parquet(split_dir / "train.parquet")
            val.to_parquet(split_dir / "validation.parquet")
            test.to_parquet(split_dir / "test.parquet")

        monkeypatch.setattr(
            "panelcast.pipelines.build_features.Path",
            lambda p: tmp_path / p,
        )

        ctx = _make_ctx(enable_genre=False)
        manifest = build_features(ctx)

        for split_name in ["within_entity_temporal", "entity_disjoint"]:
            for fold in ["train", "validation", "test"]:
                fold_info = manifest["split_features"][split_name][fold]
                assert "n_reviews_min" in fold_info
                assert "n_reviews_max" in fold_info
                assert "n_reviews_median" in fold_info
                assert fold_info["rows"] > 0
                assert fold_info["cols"] > 0

    def test_build_features_leakage_prevention_metadata(self, tmp_path, monkeypatch):
        """Manifest should document target label leakage prevention."""
        train, val, test = _make_split_dfs()

        splits_root = tmp_path / "data" / "splits"
        for split_name in ["within_entity_temporal", "entity_disjoint"]:
            split_dir = splits_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            train.to_parquet(split_dir / "train.parquet")
            val.to_parquet(split_dir / "validation.parquet")
            test.to_parquet(split_dir / "test.parquet")

        monkeypatch.setattr(
            "panelcast.pipelines.build_features.Path",
            lambda p: tmp_path / p,
        )

        ctx = _make_ctx(enable_genre=False)
        manifest = build_features(ctx)

        assert manifest["n_reviews_included"] is True
        assert "target_label_leakage_prevention" in manifest
        lp = manifest["target_label_leakage_prevention"]
        assert "User_Score" in lp["masked_score_columns"]
        assert "Critic_Score" in lp["masked_score_columns"]
        assert "validation" in lp["applies_to_splits"]
        assert "test" in lp["applies_to_splits"]


# ============================================================================
# Tests: _transform_with_train_history edge cases
# ============================================================================


class TestTransformWithTrainHistoryNew:
    """Additional tests for _transform_with_train_history."""

    def test_duplicate_target_index_preserved(self):
        """Target rows with duplicate index labels must be recovered positionally."""
        from panelcast.features.base import FeatureContext
        from panelcast.features.pipeline import FeaturePipeline
        from panelcast.features.temporal import TemporalBlock

        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B"],
                "Album": ["a1", "a2", "b1"],
                "Release_Date_Parsed": pd.to_datetime(["2019-01-01", "2020-01-01", "2019-06-01"]),
                "Year": [2019.0, 2020.0, 2019.0],
                "date_risk": ["low", "low", "low"],
                "User_Score": [70.0, 75.0, 80.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "Album": ["a3", "b2"],
                "Release_Date_Parsed": pd.to_datetime(["2021-01-01", "2021-06-01"]),
                "Year": [2021.0, 2021.0],
                "date_risk": ["low", "low"],
                "User_Score": [90.0, 85.0],
            },
            # Duplicate, overlapping-with-train index labels.
            index=pd.Index([0, 0]),
        )

        pipeline = FeaturePipeline([TemporalBlock({})])
        feature_ctx = FeatureContext(config={}, random_state=42)
        pipeline.fit(train_df, feature_ctx)

        result = _transform_with_train_history(pipeline, train_df, target_df, feature_ctx)
        assert len(result) == 2
        assert list(result.index) == [0, 0]
        # Row order must match target_df: A's third album then B's second.
        assert list(result["album_sequence"]) == [3, 2]

    def test_row_count_change_raises(self):
        """A pipeline that drops rows during transform must fail loudly."""
        from panelcast.features.base import FeatureContext

        class _DroppingPipeline:
            def transform(self, df, ctx):
                class _Out:
                    data = df.iloc[:-1]

                return _Out()

        train_df = pd.DataFrame({"Artist": ["A"], "User_Score": [70.0]})
        target_df = pd.DataFrame({"Artist": ["A"], "User_Score": [80.0]})
        feature_ctx = FeatureContext(config={}, random_state=42)

        with pytest.raises(ValueError, match="row count"):
            _transform_with_train_history(_DroppingPipeline(), train_df, target_df, feature_ctx)

    def test_masks_critic_score_column(self):
        """Critic_Score should be masked in target split."""
        from panelcast.features.artist import ArtistHistoryBlock
        from panelcast.features.base import FeatureContext
        from panelcast.features.pipeline import FeaturePipeline

        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "Album": ["a1", "a2"],
                "Release_Date_Parsed": pd.to_datetime(["2019-01-01", "2020-01-01"]),
                "User_Score": [70.0, 75.0],
                "Critic_Score": [60.0, 65.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["a3"],
                "Release_Date_Parsed": pd.to_datetime(["2021-01-01"]),
                "User_Score": [90.0],
                "Critic_Score": [85.0],
            }
        )

        pipeline = FeaturePipeline([ArtistHistoryBlock({})])
        feature_ctx = FeatureContext(config={}, random_state=42)
        pipeline.fit(train_df, feature_ctx)

        result = _transform_with_train_history(
            pipeline,
            train_df,
            target_df,
            feature_ctx,
        )
        # Target's User_Score (90) and Critic_Score (85) should not appear
        # in the history features. Only the 2 train records should be counted.
        assert result["user_prior_count"].iloc[0] == 2

    def test_handles_missing_score_columns(self):
        """If target lacks score columns, no error should occur."""
        from panelcast.features.artist import ArtistHistoryBlock
        from panelcast.features.base import FeatureContext
        from panelcast.features.pipeline import FeaturePipeline

        train_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["a1"],
                "Release_Date_Parsed": pd.to_datetime(["2019-01-01"]),
                "User_Score": [70.0],
                "Critic_Score": [60.0],
            }
        )
        # Target without Critic_Score column
        target_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["a2"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
                "User_Score": [80.0],
            }
        )

        pipeline = FeaturePipeline([ArtistHistoryBlock({})])
        feature_ctx = FeatureContext(config={}, random_state=42)
        pipeline.fit(train_df, feature_ctx)

        # Should not raise
        result = _transform_with_train_history(
            pipeline,
            train_df,
            target_df,
            feature_ctx,
        )
        assert len(result) == 1

    def test_output_preserves_original_index(self):
        """Output should have same index as target_df."""
        from panelcast.features.artist import ArtistHistoryBlock
        from panelcast.features.base import FeatureContext
        from panelcast.features.pipeline import FeaturePipeline

        train_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["a1"],
                "Release_Date_Parsed": pd.to_datetime(["2019-01-01"]),
                "User_Score": [70.0],
                "Critic_Score": [60.0],
            }
        )
        target_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "Album": ["a2", "a3"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01"]),
                "User_Score": [80.0, 85.0],
                "Critic_Score": [70.0, 75.0],
            },
            index=[10, 20],  # Non-standard index
        )

        pipeline = FeaturePipeline([ArtistHistoryBlock({})])
        feature_ctx = FeatureContext(config={}, random_state=42)
        pipeline.fit(train_df, feature_ctx)

        result = _transform_with_train_history(
            pipeline,
            train_df,
            target_df,
            feature_ctx,
        )
        assert result.index.tolist() == [10, 20]


# ============================================================================
# Tests: _assign_n_reviews edge cases
# ============================================================================


class TestAssignNReviewsNew:
    """Additional edge cases for _assign_n_reviews."""

    def test_empty_dataframes(self):
        """Should work with empty DataFrames."""
        features = pd.DataFrame({"feat1": pd.Series(dtype=float)})
        n_reviews = pd.Series(dtype=int, name="n_reviews")
        result = _assign_n_reviews(features, n_reviews, "test")
        assert "n_reviews" in result.columns
        assert len(result) == 0

    def test_large_n_reviews_values(self):
        """Should handle very large n_reviews values."""
        features = pd.DataFrame({"feat1": [1.0]}, index=[0])
        n_reviews = pd.Series([1000000], index=[0], name="n_reviews")
        result = _assign_n_reviews(features, n_reviews, "test")
        assert result["n_reviews"].iloc[0] == 1000000


# ============================================================================
# Tests: get_feature_blocks additional combos
# ============================================================================


class TestGetFeatureBlocksNew:
    """Additional tests for get_feature_blocks combinations."""

    def test_genre_only_disabled(self):
        """Genre disabled, others enabled."""
        blocks = get_feature_blocks(enable_genre=False, enable_artist=True, enable_temporal=True)
        names = [b.name for b in blocks]
        assert "genre" not in names
        assert "artist_history" in names
        assert "temporal" in names

    def test_artist_and_temporal_disabled(self):
        """Only genre is optional enabled block."""
        blocks = get_feature_blocks(enable_genre=True, enable_artist=False, enable_temporal=False)
        names = [b.name for b in blocks]
        assert "genre" in names
        assert "artist_history" not in names
        assert "temporal" not in names
        # Should have album_type, genre, collaboration = 3
        assert len(blocks) == 3

    def test_default_blocks_returns_list(self):
        """get_default_feature_blocks returns a list."""
        blocks = get_default_feature_blocks()
        assert isinstance(blocks, list)
        assert len(blocks) > 0
