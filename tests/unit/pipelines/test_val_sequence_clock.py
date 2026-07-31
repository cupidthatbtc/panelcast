"""Held-out validation events advance the latent clock, not just the AR lag.

With ``val_events >= 1`` the test AR lag already takes ``prev_score`` from the
last validation event, but ``album_seq`` was offset by a train-only count: for
an entity with two train events, one validation event and one test event the
test latent effect was indexed at sequence 3 while the lag was the score at
position 3, so the latent clock ran one step behind the observation it scored.
``_build_horizon_panel`` split the same way, seeding the terminal state from
train counts while ``y_last`` started after validation, so the ancestral rollout
compounded one fewer innovation than the horizon it reported (#428).

Both clocks now come off :func:`_preceding_event_counts`, and the ``max_events``
cap offset stays train-only, which is the separate thing #247 fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.evaluate import (
    _build_horizon_panel,
    _preceding_event_counts,
    _prepare_test_model_args,
)


@pytest.fixture
def summary() -> dict:
    return {
        "artist_to_idx": {"A": 0, "B": 1},
        "max_seq": 10,
        "max_albums": 50,
        "min_albums_filter": 2,
        "global_mean_score": 75.0,
        "feature_cols": ["f1"],
        "feature_scaler": {"mean": [0.0], "std": [1.0]},
        "n_artists": 2,
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "n_ref": None,
        "target_transform": "identity",
        "logit_offset": 0.5,
        "priors": {
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_scale": 0.3,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
        },
    }


@pytest.fixture
def frames():
    """Two entities with two train events each; A also has a validation event."""
    train_df = pd.DataFrame(
        {
            "Artist": ["A", "A", "B", "B"],
            "User_Score": [70.0, 71.0, 60.0, 61.0],
        }
    )
    val_df = pd.DataFrame({"Artist": ["A"], "User_Score": [72.0]})
    test_df = pd.DataFrame(
        {
            "Artist": ["A", "B"],
            "User_Score": [80.0, 65.0],
            "Album": ["a3", "b3"],
        }
    )
    test_features = pd.DataFrame({"f1": [1.0, 2.0], "n_reviews": [10, 20]})
    return train_df, val_df, test_df, test_features


class TestPrecedingEventCounts:
    def test_train_only(self, frames):
        train_df, _, _, _ = frames
        counts = _preceding_event_counts(train_df, None, entity_col="Artist")
        assert counts.to_dict() == {"A": 2, "B": 2}

    def test_validation_events_count(self, frames):
        train_df, val_df, _, _ = frames
        counts = _preceding_event_counts(train_df, val_df, entity_col="Artist")
        assert counts.to_dict() == {"A": 3, "B": 2}

    def test_no_history_is_empty(self):
        assert _preceding_event_counts(None, None, entity_col="Artist").empty

    def test_validation_only_history(self, frames):
        _, val_df, _, _ = frames
        counts = _preceding_event_counts(None, val_df, entity_col="Artist")
        assert counts.to_dict() == {"A": 1}


class TestTestSequencePosition:
    def test_the_test_row_sits_after_the_validation_event(self, summary, frames):
        """A: 2 train + 1 val + this test event -> sequence position 4."""
        train_df, val_df, test_df, test_features = frames
        model_args, _, _ = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, val_df=val_df
        )
        assert model_args["album_seq"].tolist() == [4, 3]

    def test_without_validation_the_position_is_unchanged(self, summary, frames):
        """The standard split reserves no validation events, so the shipped
        default keeps every number it had."""
        train_df, _, test_df, test_features = frames
        model_args, _, _ = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, val_df=None
        )
        assert model_args["album_seq"].tolist() == [3, 3]

    def test_the_latent_clock_matches_the_ar_clock(self, summary, frames):
        """The lag is the validation score, so the latent position must be the
        one that follows it -- the two clocks come off one computation."""
        train_df, val_df, test_df, test_features = frames
        model_args, _, _ = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, val_df=val_df
        )
        counts = _preceding_event_counts(train_df, val_df, entity_col="Artist")
        # Row 0 is entity A: its lag is the validation score, so its latent
        # position must be the one that follows that event.
        assert model_args["prev_score"][0] == pytest.approx(72.0)
        assert model_args["album_seq"][0] == counts["A"] + 1

    def test_the_max_events_cap_offset_stays_train_only(self, summary, frames):
        """#247 pinned the cap to training counts so test events keep the frame
        training assigned them; this fix must not fold validation into it."""
        train_df, val_df, test_df, test_features = frames
        summary["max_albums"] = 2
        capped, _, _ = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df, val_df=val_df
        )
        uncapped, _, _ = _prepare_test_model_args(
            test_df, test_features, summary | {"max_albums": 50}, train_df=train_df, val_df=val_df
        )
        # Both entities have 2 train events, so the cap shifts them identically
        # regardless of A's extra validation event.
        shift = np.asarray(uncapped["album_seq"]) - np.asarray(capped["album_seq"])
        assert shift.tolist() == [shift[0], shift[0]]


class TestHorizonPanelClock:
    def _panel(self, summary, frames, val_df):
        train_df, _, test_df, test_features = frames
        return _build_horizon_panel(
            test_df, test_features, summary, 1, train_df=train_df, val_df=val_df
        )

    def test_terminal_step_count_includes_validation(self, summary, frames):
        """y_last already starts after validation; the innovation count must
        too, or the h=1 forecast is one step short of its own AR lag."""
        _, val_df, _, _ = frames
        panel = self._panel(summary, frames, val_df)
        assert panel["y_last"].tolist() == [72.0, 61.0]
        assert panel["n_train_events"].tolist() == [3, 2]

    def test_without_validation_the_counts_are_unchanged(self, summary, frames):
        panel = self._panel(summary, frames, None)
        assert panel["y_last"].tolist() == [71.0, 61.0]
        assert panel["n_train_events"].tolist() == [2, 2]

    def test_eligibility_still_reads_training_history(self, summary, frames):
        """dynamic_mask mirrors the training-time filter, not the clock: a
        validation event never made an entity eligible for dynamic effects."""
        train_df, val_df, test_df, test_features = frames
        summary["min_albums_filter"] = 3
        panel = _build_horizon_panel(
            test_df, test_features, summary, 1, train_df=train_df, val_df=val_df
        )
        assert panel["n_train_events"].tolist() == [3, 2]
        assert panel["dynamic_mask"].tolist() == [False, False]

    def test_the_clock_is_capped_at_the_trained_horizon(self, summary, frames):
        train_df, val_df, test_df, test_features = frames
        summary["max_seq"] = 2
        panel = _build_horizon_panel(
            test_df, test_features, summary, 1, train_df=train_df, val_df=val_df
        )
        assert panel["n_train_events"].tolist() == [2, 2]
