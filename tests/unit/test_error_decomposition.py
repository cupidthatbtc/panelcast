"""Error decomposition over the identified predictions payload (#180)."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.evaluation.decomposition import decompose_errors


@pytest.fixture()
def payload():
    return {
        "y_true": [70.0, 80.0, 60.0, 90.0],
        "y_pred_mean": [72.0, 75.0, 65.0, 90.0],
        "residuals": [-2.0, 5.0, -5.0, 0.0],
        "interval_level": 0.8,
        "entity": ["A", "A", "B", "C"],
        "event": ["a1", "a2", "b1", "c1"],
        "group": ["rock", "rock", "pop", "pop"],
        "n_reviews": [10, 20, 30, 40],
        "train_history": [3, 3, 0, 1],
        "y_pred_sd": [2.0, 5.0, 5.0, 1.0],
        "pit": [0.2, 0.01, 0.99, 0.5],
        "covered": {"0.80": [True, False, False, True], "0.95": [True, False, True, True]},
    }


def test_rows_ranked_and_shares_sum_to_one(payload):
    decomp = decompose_errors(payload)
    rows = decomp.rows
    assert len(rows) == 4
    assert rows["abs_residual"].is_monotonic_decreasing
    assert rows["sq_error_share"].sum() == pytest.approx(1.0)
    # Identity stays attached to its own error after the sort.
    a2 = rows[rows["event"] == "a2"].iloc[0]
    assert a2["entity"] == "A"
    assert a2["residual"] == pytest.approx(5.0)
    assert a2["std_residual"] == pytest.approx(1.0)


def test_miscalibration_flags_far_tails(payload):
    rows = decompose_errors(payload).rows.set_index("event")
    assert bool(rows.loc["a2", "miscalibrated"])
    assert bool(rows.loc["b1", "miscalibrated"])
    assert not bool(rows.loc["a1", "miscalibrated"])
    assert not bool(rows.loc["c1", "miscalibrated"])


def test_rollups_by_entity_group_and_decile(payload):
    rollups = decompose_errors(payload).rollups
    entity = rollups["entity"]
    assert set(entity.index) == {"A", "B", "C"}
    assert entity.loc["A", "n"] == 2
    assert entity.loc["A", "mae"] == pytest.approx(3.5)
    # A carries (4+25)/54 of squared error and must rank first.
    assert entity.index[0] == "A"
    assert entity.loc["A", "sq_error_share"] == pytest.approx(29.0 / 54.0)
    assert entity.loc["C", "coverage_0.80"] == pytest.approx(1.0)
    group = rollups["group"]
    assert set(group.index) == {"rock", "pop"}
    assert "n_reviews_decile" in rollups


def test_unidentified_payload_raises_clear_error():
    with pytest.raises(ValueError, match="identity fields"):
        decompose_errors({"y_true": [1.0], "y_pred_mean": [1.0]})


def test_zero_sd_yields_nan_std_residual(payload):
    payload["y_pred_sd"] = [0.0, 5.0, 5.0, 1.0]
    rows = decompose_errors(payload).rows.set_index("event")
    assert np.isnan(rows.loc["a1", "std_residual"])
