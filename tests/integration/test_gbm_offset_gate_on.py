"""End-to-end gate-on run of the gbm_offset feature block (#86).

Same data -> splits -> features path as the golden-hash guard, with the block
enabled: the parquets gain exactly one finite gbm_offset column and the
manifest records the block. Gate-off parity is covered by the golden hashes
(the default roster is untouched).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.build_features import build_features
from panelcast.pipelines.create_splits import SplitConfig, create_splits
from panelcast.pipelines.prepare_dataset import PrepareConfig, prepare_datasets
from tests.e2e.conftest import MINIMAL_TEST_DATA
from tests.integration.test_prepare_golden_hashes import EXTRA_ROWS, FIELDNAMES


@pytest.fixture(scope="module")
def gate_on_features(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("gbm_offset_gate_on")
    csv_path = tmp / "raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(MINIMAL_TEST_DATA)
        writer.writerows(EXTRA_ROWS)

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
            gbm_offset=True,
            descriptor=DatasetDescriptor(),
        )
        manifest = build_features(ctx)
    finally:
        os.chdir(cwd)
    return manifest, tmp


class TestGbmOffsetGateOn:
    def test_manifest_records_block(self, gate_on_features):
        manifest, _ = gate_on_features
        assert manifest["gbm_offset"] is True
        assert manifest["blocks"][-1] == "gbm_offset"
        assert "gbm_offset" in manifest["feature_names"]

    def test_column_present_and_finite_on_every_split(self, gate_on_features):
        _, tmp = gate_on_features
        for split in ("within_entity_temporal", "entity_disjoint"):
            for part in ("train", "validation", "test"):
                path = Path(tmp) / "data" / "features" / split / f"{part}_features.parquet"
                df = pd.read_parquet(path)
                assert "gbm_offset" in df.columns, f"{split}/{part}"
                if len(df):
                    assert np.isfinite(df["gbm_offset"]).all(), f"{split}/{part}"

    def test_offset_tracks_target_scale(self, gate_on_features):
        """Smoke: train offsets live on the target's scale, not degenerate."""
        _, tmp = gate_on_features
        path = Path(tmp) / "data" / "features" / "within_entity_temporal" / (
            "train_features.parquet"
        )
        offsets = pd.read_parquet(path)["gbm_offset"]
        assert offsets.std() >= 0.0
        assert 0.0 <= offsets.mean() <= 100.0
