"""Period-effects pipeline wiring (#269): index building and config plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.pipelines.train_bayes import prepare_model_data


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "B", "B", "C"],
            "User_Score": [70.0, 75.0, 80.0, 72.0, 78.0, 75.0],
            "Release_Year": [2018, 2019, 2020, 2018, 2020, 2019],
            "feature_1": np.zeros(6, dtype=np.float32),
            "n_reviews": np.full(6, 50, dtype=np.int32),
        }
    )


class TestPrepareModelData:
    def test_gate_builds_sorted_period_indices(self):
        descriptor = DatasetDescriptor(period_col="Release_Year")
        model_args, _ = prepare_model_data(
            _train_df(), ["feature_1"], descriptor=descriptor, period_effects=True
        )
        assert model_args["n_periods"] == 3
        assert model_args["period_to_idx"] == {"2018": 0, "2019": 1, "2020": 2}
        # Chronology normalization may reorder rows; the index must align
        # row-for-row with the normalized frame's own period values.
        from panelcast.pipelines.train_bayes import _normalize_model_frame

        normalized = _normalize_model_frame(_train_df(), descriptor)
        expected = normalized["Release_Year"].map({2018: 0, 2019: 1, 2020: 2}).to_numpy()
        idx = np.asarray(model_args["period_idx"])
        assert idx.dtype == np.int32
        np.testing.assert_array_equal(idx, expected)

    def test_gate_without_descriptor_period_col_raises(self):
        with pytest.raises(ValueError, match="period_col"):
            prepare_model_data(_train_df(), ["feature_1"], period_effects=True)

    def test_gate_off_adds_no_period_keys(self):
        model_args, _ = prepare_model_data(_train_df(), ["feature_1"])
        assert "period_idx" not in model_args
        assert "n_periods" not in model_args
        assert "period_to_idx" not in model_args


class TestConfigPlumbing:
    def test_invalid_constraint_rejected(self):
        with pytest.raises(ValueError, match="period_constraint"):
            PipelineConfig(period_constraint="nope")

    def test_gate_requires_descriptor_period_col(self, tmp_path):
        dataset = tmp_path / "d.yaml"
        dataset.write_text("name: nop\n", encoding="utf-8")
        config = PipelineConfig(dataset=str(dataset), period_effects=True)
        with pytest.raises(ValueError, match="period_col"):
            PipelineOrchestrator(config, output_base=tmp_path)

    def test_gate_with_declared_period_col_resolves(self, tmp_path):
        dataset = tmp_path / "d.yaml"
        dataset.write_text("name: ok\nperiod_col: Release_Year\n", encoding="utf-8")
        config = PipelineConfig(dataset=str(dataset), period_effects=True)
        PipelineOrchestrator(config, output_base=tmp_path)
        assert config.period_effects is True

    def test_descriptor_hash_stable_when_period_col_unset(self):
        assert (
            DatasetDescriptor().descriptor_hash()
            == "a9e3e20540b1dcb5d6253bd342cff6fd73ed823597428f4e94abd51f8b67b8ec"
        )
        assert (
            DatasetDescriptor(period_col="Release_Year").descriptor_hash()
            != DatasetDescriptor().descriptor_hash()
        )
