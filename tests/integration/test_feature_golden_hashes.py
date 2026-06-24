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

# Float columns are rounded before hashing so the guard is robust to the
# platform-level float noise (different BLAS/LAPACK builds) that otherwise makes
# byte-exact %.17g hashes differ between the dev machine and CI — while still
# catching real feature changes, which move values by far more than this.
_HASH_DECIMALS = 4


def _stable_feature_hash(df: pd.DataFrame) -> str:
    """hash_dataframe of the frame with float columns rounded to _HASH_DECIMALS."""
    rounded = df.copy()
    float_cols = rounded.select_dtypes(include="floating").columns
    rounded[float_cols] = rounded[float_cols].round(_HASH_DECIMALS)
    return hash_dataframe(rounded)


# Frozen on the pre-descriptor implementation (2026-06-10); hashes are over the
# rounded feature matrices (see _stable_feature_hash).
GOLDEN_FEATURE_HASHES = {
    "within_entity_temporal/train": (
        "1848368d2d0bfef815dfbd006ff51abdfdbe6fd1488b9797b5462383615a0d42"
    ),
    "within_entity_temporal/validation": (
        "255cd169ed4ad1e5f0a58f82ad206b823d1282605569cae05d35b96ff00f84c0"
    ),
    "within_entity_temporal/test": (
        "cd3bf6c1fbc7e790086ea40e10292166154dfe2f72a0920fd1a66c767604ca09"
    ),
    "entity_disjoint/train": ("34b097f3cd48d20af50ebe847a077b276309b675401e81676f71380cc975bfdf"),
    "entity_disjoint/validation": (
        "4f5b2d5ae99d8e9f4846eb1345171eecedf7f5ae81909b060876a5dd30f082e4"
    ),
    "entity_disjoint/test": ("180845ce1ef30915ef8ae7da2cef02e8e7dc0d4b73bd447b504b60a283934049"),
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
            actual = _stable_feature_hash(pd.read_parquet(path))
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
