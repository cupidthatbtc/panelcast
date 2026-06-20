"""Tests for the debut prev_score source gate (leakage control)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.train_bayes import prepare_model_data


def _train_df() -> pd.DataFrame:
    """Three artists; artists B and C have debuts; train mean = 75."""
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "B", "B", "C"],
            "User_Score": [70.0, 75.0, 80.0, 72.0, 78.0, 75.0],
            "feature_1": np.zeros(6, dtype=np.float32),
            "n_reviews": np.full(6, 50, dtype=np.int32),
        }
    )


class TestDebutPrevScoreSource:
    def test_default_uses_train_split_mean(self, tmp_path, monkeypatch):
        """Default must ignore dataset_stats.json even when it exists."""
        monkeypatch.chdir(tmp_path)
        stats_dir = tmp_path / "data" / "processed"
        stats_dir.mkdir(parents=True)
        # Pre-split mean deliberately different from the train mean.
        (stats_dir / "dataset_stats.json").write_text(
            json.dumps({"global_mean_score": 99.0}), encoding="utf-8"
        )

        model_args, _ = prepare_model_data(_train_df(), ["feature_1"])
        train_mean = _train_df()["User_Score"].mean()
        assert model_args["global_mean_score"] == pytest.approx(train_mean)
        # Debut rows (A1, B1, C1) carry the train mean, not 99.
        prev = np.asarray(model_args["prev_score"], dtype=float)
        debut_positions = [0, 3, 5]
        for pos in debut_positions:
            assert prev[pos] == pytest.approx(train_mean)

    def test_dataset_stats_source_reproduces_legacy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        stats_dir = tmp_path / "data" / "processed"
        stats_dir.mkdir(parents=True)
        (stats_dir / "dataset_stats.json").write_text(
            json.dumps({"global_mean_score": 99.0}), encoding="utf-8"
        )

        model_args, _ = prepare_model_data(
            _train_df(), ["feature_1"], debut_prev_score_source="dataset_stats"
        )
        assert model_args["global_mean_score"] == pytest.approx(99.0)

    def test_dataset_stats_missing_falls_back_to_train_mean(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model_args, _ = prepare_model_data(
            _train_df(), ["feature_1"], debut_prev_score_source="dataset_stats"
        )
        assert model_args["global_mean_score"] == pytest.approx(75.0)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="debut_prev_score_source"):
            prepare_model_data(_train_df(), ["feature_1"], debut_prev_score_source="bogus")

    def test_non_debut_rows_keep_shifted_scores(self):
        model_args, _ = prepare_model_data(_train_df(), ["feature_1"])
        prev = np.asarray(model_args["prev_score"], dtype=float)
        # A's 2nd/3rd albums and B's 2nd album use the actual previous scores.
        assert prev[1] == pytest.approx(70.0)
        assert prev[2] == pytest.approx(75.0)
        assert prev[4] == pytest.approx(72.0)
