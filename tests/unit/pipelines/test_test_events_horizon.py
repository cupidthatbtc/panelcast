"""The split reserves the events the configured eval_horizon has to score.

``eval_horizon`` is user-configurable, but the only standard split path held out
one event per entity and ``test_events`` reached neither ``PipelineConfig`` nor
the YAML mapping, so a run with ``eval_horizon: 3`` scored h=1 and emitted
``{"h": 2, "n": 0}``-style empty masks for every deeper horizon and every
entity. The advertised h=1..H evaluation was unreachable without hand-replacing
split artifacts (#429).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.config.pipeline_yaml import PIPELINE_YAML_MAPPING, apply_yaml_overrides
from panelcast.data.split import within_entity_temporal_split
from panelcast.pipelines.create_splits import SplitConfig
from panelcast.pipelines.evaluate import _build_horizon_panel
from panelcast.pipelines.pipeline_config import PipelineConfig


class TestConfigResolution:
    def test_the_default_run_is_unchanged(self):
        """eval_horizon defaults to 0 (rollout off), so the shipped split still
        holds out exactly one event per entity."""
        config = PipelineConfig(run_id="default")
        assert config.eval_horizon == 0
        assert config.test_events == 1

    @pytest.mark.parametrize("horizon", [1, 2, 3, 5])
    def test_the_test_era_is_sized_from_the_horizon(self, horizon):
        config = PipelineConfig(run_id="horizon", eval_horizon=horizon)
        assert config.test_events == horizon

    def test_an_explicit_value_wins(self):
        config = PipelineConfig(run_id="explicit", eval_horizon=2, test_events=4)
        assert config.test_events == 4

    def test_an_explicit_value_that_cannot_cover_the_horizon_is_rejected(self):
        """Previously this produced empty masks for h=2..3 and a green stage."""
        with pytest.raises(ValueError, match="eval_horizon"):
            PipelineConfig(run_id="short", eval_horizon=3, test_events=1)

    def test_zero_test_events_is_rejected(self):
        with pytest.raises(ValueError, match="test_events"):
            PipelineConfig(run_id="none", test_events=0)

    def test_it_reaches_the_yaml_surface(self):
        assert "test_events" in PIPELINE_YAML_MAPPING
        kwargs = apply_yaml_overrides({}, {"eval_horizon": 3, "test_events": 4})
        assert PipelineConfig(run_id="yaml", **kwargs).test_events == 4

    def test_the_split_config_takes_it(self):
        assert SplitConfig(test_events=3).test_events == 3


@pytest.fixture
def panel_summary() -> dict:
    return {
        "artist_to_idx": {"A": 0, "B": 1},
        "feature_cols": ["f1"],
        "feature_scaler": {"mean": [0.0], "std": [1.0], "feature_cols": ["f1"]},
        "max_seq": 10,
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


def _panel_frames(n_test_events: int):
    """A panel whose split held out ``n_test_events`` events per entity."""
    rows = []
    for entity, base in (("A", 70.0), ("B", 60.0)):
        for k in range(6):
            rows.append({"Artist": entity, "User_Score": base + k, "seq": k})
    df = pd.DataFrame(rows)
    train_parts, test_parts = [], []
    for _, group in df.groupby("Artist", sort=False):
        train_parts.append(group.iloc[:-n_test_events])
        test_parts.append(group.iloc[-n_test_events:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    test_features = pd.DataFrame(
        {
            "f1": np.arange(len(test_df), dtype=float),
            "n_reviews": np.full(len(test_df), 25.0),
        }
    )
    return train_df, test_df, test_features


class TestHorizonsHaveRowsToScore:
    def test_a_one_event_test_era_can_only_fill_h1(self, panel_summary):
        """The behavior #429 reported: h=2 and h=3 come back empty."""
        train_df, test_df, test_features = _panel_frames(1)
        panel = _build_horizon_panel(
            test_df, test_features, panel_summary, 3, train_df=train_df, val_df=None
        )
        assert panel["valid"][0].tolist() == [True, True]
        assert not panel["valid"][1].any()
        assert not panel["valid"][2].any()

    def test_a_horizon_sized_test_era_fills_every_step(self, panel_summary):
        """With test_events == eval_horizon each h=1..H has scored rows."""
        train_df, test_df, test_features = _panel_frames(3)
        panel = _build_horizon_panel(
            test_df, test_features, panel_summary, 3, train_df=train_df, val_df=None
        )
        assert panel["valid"].all()
        assert panel["y_panel"][2].tolist() == [75.0, 65.0]


class TestSplitReservesTheEvents:
    def test_the_test_era_holds_the_configured_number_of_events(self):
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 6 + ["B"] * 6,
                "Album": [f"e{i}" for i in range(12)],
                "Release_Date_Parsed": list(pd.date_range("2010", periods=6, freq="YS")) * 2,
            }
        )
        train_df, _, test_df = within_entity_temporal_split(
            df, test_events=3, val_events=0, min_train_events=1
        )
        assert test_df.groupby("Artist").size().tolist() == [3, 3]
        assert train_df.groupby("Artist").size().tolist() == [3, 3]

    def test_entities_without_enough_history_drop_out(self):
        """A deeper test era needs more history per entity; the eligible set
        shrinks, which is why horizon numbers are not cross-comparable."""
        df = pd.DataFrame(
            {
                "Artist": ["A"] * 6 + ["B"] * 4,
                "Album": [f"e{i}" for i in range(10)],
                "Release_Date_Parsed": (
                    list(pd.date_range("2010", periods=6, freq="YS"))
                    + list(pd.date_range("2010", periods=4, freq="YS"))
                ),
            }
        )
        # B has exactly test_events + min_train_events events, so it survives.
        _, _, shallow = within_entity_temporal_split(
            df, test_events=3, val_events=0, min_train_events=1
        )
        assert set(shallow["Artist"]) == {"A", "B"}
        # One more reserved event and B no longer has a training history.
        _, _, deep = within_entity_temporal_split(
            df, test_events=4, val_events=0, min_train_events=1
        )
        assert set(deep["Artist"]) == {"A"}
