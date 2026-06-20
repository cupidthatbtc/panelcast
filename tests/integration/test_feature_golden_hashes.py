"""Golden-hash regression guard for the feature-building pipeline.

Runs data -> splits -> features on the augmented minimal dataset (same
fixture as test_prepare_golden_hashes) and freezes ``hash_dataframe``
digests of every feature parquet. Any behavioral change to the feature
blocks or their descriptor-driven construction shows up here.

Captured on the pre-descriptor get_feature_blocks implementation; the
registry/descriptor refactor must keep outputs byte-identical.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.build_features import build_features
from panelcast.pipelines.create_splits import SplitConfig, create_splits
from panelcast.pipelines.prepare_dataset import PrepareConfig, prepare_datasets
from panelcast.utils.hashing import hash_dataframe
from tests.e2e.conftest import MINIMAL_TEST_DATA
from tests.integration.test_prepare_golden_hashes import EXTRA_ROWS, FIELDNAMES

# Frozen on the pre-descriptor implementation (2026-06-10).
GOLDEN_FEATURE_HASHES = {
    "within_artist_temporal/train": (
        "e6b4ea8b864ff2e35587c03a2a1ff2116f5a89e1e3708690c55e28af705f5aad"
    ),
    "within_artist_temporal/validation": (
        "255cd169ed4ad1e5f0a58f82ad206b823d1282605569cae05d35b96ff00f84c0"
    ),
    "within_artist_temporal/test": (
        "57f5e07b02830bec2aab4c12bddd2a6ed1148e6bcf321e79ee1bc5e7a3486c97"
    ),
    "artist_disjoint/train": ("52dc781661bf32e98322162ebf580291e9037753907de447d66d14e01ebf74c9"),
    "artist_disjoint/validation": (
        "109aece9bc3ea35655b576f3dab062d73487d7ac1574ffea3f942f9e72be8d6f"
    ),
    "artist_disjoint/test": ("faf7ed362e54fc65ff32eb664a5f111782762962eb45aa5ae774580cbbbaaaaf"),
}

GOLDEN_FEATURE_NAMES = [
    "album_sequence",
    "career_years",
    "release_gap_days",
    "release_year",
    "date_risk_ordinal",
    "date_missing",
    "is_album",
    "is_ep",
    "user_prior_mean",
    "user_prior_std",
    "user_prior_count",
    "user_trajectory",
    "critic_prior_mean",
    "critic_prior_std",
    "critic_prior_count",
    "critic_trajectory",
    "is_debut",
    "is_collaboration",
    "num_artists",
    "collab_type_ordinal",
    "n_reviews",
]

GOLDEN_BLOCKS = ["temporal", "album_type", "artist_history", "genre", "collaboration"]


@pytest.fixture(scope="module")
def built_features(tmp_path_factory: pytest.TempPathFactory):
    """Run data -> splits -> features once on the augmented fixture."""
    tmp = tmp_path_factory.mktemp("golden_features")
    csv_path = tmp / "raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(MINIMAL_TEST_DATA)
        writer.writerows(EXTRA_ROWS)

    import os

    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        prepare_datasets(
            PrepareConfig(
                raw_path=str(csv_path),
                output_dir="data/processed",
                audit_dir="data/audit",
            )
        )
        create_splits(SplitConfig(random_state=42, min_ratings=10))
        ctx = SimpleNamespace(
            seed=42,
            enable_genre=True,
            enable_artist=True,
            enable_temporal=True,
            descriptor=DatasetDescriptor(),
        )
        manifest = build_features(ctx)
    finally:
        os.chdir(cwd)
    return manifest, tmp


class TestFeatureGoldenHashes:
    def test_feature_parquet_hashes(self, built_features):
        _, tmp = built_features
        mismatches = {}
        for key, expected in GOLDEN_FEATURE_HASHES.items():
            split, part = key.split("/")
            path = Path(tmp) / "data" / "features" / split / f"{part}_features.parquet"
            actual = hash_dataframe(pd.read_parquet(path))
            if actual != expected:
                mismatches[key] = actual
        assert not mismatches, (
            "Feature matrix content changed (behavior-changing edit to the "
            f"feature blocks or their construction?): {mismatches}"
        )

    def test_feature_names(self, built_features):
        manifest, _ = built_features
        assert manifest["feature_names"] == GOLDEN_FEATURE_NAMES

    def test_block_names(self, built_features):
        manifest, _ = built_features
        assert manifest["blocks"] == GOLDEN_BLOCKS

    def test_masked_score_columns(self, built_features):
        manifest, _ = built_features
        assert manifest["target_label_leakage_prevention"]["masked_score_columns"] == [
            "User_Score",
            "Critic_Score",
        ]
