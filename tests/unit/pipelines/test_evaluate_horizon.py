"""Horizon-rollout evaluation panel (#157): ragged masking and state seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.evaluate import _build_horizon_panel


@pytest.fixture
def summary():
    return {
        "artist_to_idx": {"A": 0, "B": 1},
        "feature_cols": ["f1"],
        "feature_scaler": {"mean": [1.0], "std": [2.0], "feature_cols": ["f1"]},
        "max_seq": 5,
        "min_albums_filter": 2,
        "global_mean_score": 70.0,
        "target_transform": "identity",
        "logit_offset": 0.5,
        "dataset": {
            "entity_col": "Artist",
            "event_col": "Album",
            "target_col": "User_Score",
            "n_obs_col": "User_Ratings",
            "model_prefix": "user",
            "target_bounds": (0.0, 100.0),
        },
    }


@pytest.fixture
def frames():
    test_df = pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "B"],
            "User_Score": [71.0, 72.0, 73.0, 60.0],
        }
    )
    test_features = pd.DataFrame(
        {
            "f1": [3.0, 5.0, 7.0, 9.0],
            "n_reviews": [10.0, 0.0, 20.0, 30.0],  # invalid count on A's 2nd event
        }
    )
    train_df = pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "A", "B"],
            "User_Score": [65.0, 66.0, 67.0, 68.0, 55.0],
        }
    )
    return test_df, test_features, train_df


class TestBuildHorizonPanel:
    def test_shapes_and_entity_order(self, summary, frames):
        test_df, test_features, train_df = frames
        panel = _build_horizon_panel(
            test_df, test_features, summary, 2, train_df=train_df, val_df=None
        )
        assert panel["entities"] == ["A", "B"]
        assert panel["artist_idx"].tolist() == [0, 1]
        assert panel["X_panel"].shape == (2, 2, 1)
        assert panel["y_panel"].shape == (2, 2)

    def test_ragged_and_invalid_counts_masked(self, summary, frames):
        test_df, test_features, train_df = frames
        panel = _build_horizon_panel(
            test_df, test_features, summary, 2, train_df=train_df, val_df=None
        )
        # h=1: both entities valid; h=2: A's event has an invalid count and B
        # has no second test event.
        assert panel["valid"].tolist() == [[True, True], [False, False]]
        assert panel["y_panel"][0].tolist() == [71.0, 60.0]

    def test_covariates_standardized_with_train_scaler(self, summary, frames):
        test_df, test_features, train_df = frames
        panel = _build_horizon_panel(
            test_df, test_features, summary, 1, train_df=train_df, val_df=None
        )
        np.testing.assert_allclose(panel["X_panel"][0, :, 0], [(3.0 - 1.0) / 2.0, (9.0 - 1.0) / 2.0])

    def test_state_seeds_from_train_history(self, summary, frames):
        test_df, test_features, train_df = frames
        panel = _build_horizon_panel(
            test_df, test_features, summary, 1, train_df=train_df, val_df=None
        )
        # AR lag seeds from each entity's last training score; B (1 train
        # event) is below min_albums_filter and must stay static.
        assert panel["y_last"].tolist() == [68.0, 55.0]
        assert panel["n_train_events"].tolist() == [4, 1]
        assert panel["dynamic_mask"].tolist() == [True, False]

    def test_val_score_takes_precedence_for_the_lag(self, summary, frames):
        test_df, test_features, train_df = frames
        val_df = pd.DataFrame({"Artist": ["A"], "User_Score": [69.5]})
        panel = _build_horizon_panel(
            test_df, test_features, summary, 1, train_df=train_df, val_df=val_df
        )
        assert panel["y_last"].tolist() == [69.5, 55.0]

    def test_unknown_entity_rejected(self, summary, frames):
        test_df, test_features, train_df = frames
        test_df.loc[3, "Artist"] = "C"
        with pytest.raises(ValueError, match="Unknown entities"):
            _build_horizon_panel(
                test_df, test_features, summary, 1, train_df=train_df, val_df=None
            )
