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


# Deliberately regenerated for the 0.6.0 default roster (gbm_offset joined the
# shipped default blocks; every parquet gains its column). Hashes are over the
# rounded feature matrices (see _stable_feature_hash); the pre-0.6.0 values
# were frozen on the pre-descriptor implementation (2026-06-10).
#
# The two ``/train`` digests were regenerated for #293 when gbm_offset OOF
# moved from GroupKFold to the entity-aware temporal/cold-start protocol. Only
# train rows carry OOF offsets; held-out validation/test rows still use the
# same full-train deployment refit, so their four digests remain unchanged.
GOLDEN_FEATURE_HASHES = {
    "within_entity_temporal/train": (
        "b6762fc0c26ce30a03b2257b78886e5c2d9bce469d78b9a6e6ef85557f921575"
    ),
    "within_entity_temporal/validation": (
        "6cd73a3fb0ce81ab75630ddd8ddcee40471554a98c48eee4a496ed3881dbee7d"
    ),
    "within_entity_temporal/test": (
        "59ad6ec23443101aaf4da9f33e7bb651a0b83bbb64793e6a0af5e7493beac539"
    ),
    "entity_disjoint/train": ("c600f28eb6a04e043132d15268530817e5ea63b559a0f160d97d417900047b55"),
    "entity_disjoint/validation": (
        "6981f543faaa714f1d4c8e389cdba2034aa26cf76f7323ecd56cf732591d9ae5"
    ),
    "entity_disjoint/test": ("f79929a90ee7329627d8d093eae5eb725abe70f1f130b1b2ae7fff207cc104ab"),
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
    "gbm_offset",
    "n_reviews",
]

GOLDEN_BLOCKS = [
    "temporal",
    "album_type",
    "artist_history",
    "genre",
    "collaboration",
    "gbm_offset",
]


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
            # Mirrors the shipped default (on since 0.6.0).
            gbm_offset=True,
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
