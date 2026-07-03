"""Prior-predictive pre-sweep screen: per-transform checks and suggestions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.evaluation.prior_predictive import PriorPredictiveResult
from panelcast.select.prior_screen import (
    TransformScreen,
    _suggestions_from_result,
    render_prior_block,
    screen_transforms,
)

N_ARTISTS, ALBUMS_EACH = 6, 5


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for a in range(N_ARTISTS):
        base = rng.normal(72, 6)
        for _ in range(ALBUMS_EACH):
            rows.append(
                {
                    "Artist": f"artist_{a}",
                    "User_Score": float(np.clip(base + rng.normal(0, 4), 0, 100)),
                    "User_Ratings": int(rng.integers(20, 400)),
                    "f_one": rng.normal(),
                    "f_two": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


FEATURES = ["f_one", "f_two"]


class TestScreen:
    def test_screens_every_candidate_transform(self, train_df):
        screens = screen_transforms(
            train_df, FEATURES, DatasetDescriptor(), n_samples=50, seed=0
        )
        assert [s.transform for s in screens] == ["offset_logit", "identity"]

    def test_offset_logit_is_near_bounds_by_construction(self, train_df):
        # The back-transform lands in (low - offset, high + offset), so all
        # mass sits within the half-point margin of the bounds.
        (screen,) = screen_transforms(
            train_df,
            FEATURES,
            DatasetDescriptor(),
            transforms=("offset_logit",),
            n_samples=50,
            seed=0,
        )
        assert screen.fraction_in_bounds >= 0.90
        assert screen.reasonable
        assert screen.summary["min"] >= -0.5
        assert screen.summary["max"] <= 100.5

    def test_summary_and_checks_populated(self, train_df):
        (screen,) = screen_transforms(
            train_df,
            FEATURES,
            DatasetDescriptor(),
            transforms=("identity",),
            n_samples=50,
            seed=0,
        )
        assert set(screen.summary) >= {"mean", "sd", "skewness", "min", "max"}
        assert set(screen.checks) == {"mean", "sd", "skewness"}


class TestSuggestions:
    def _result(self, **overrides) -> PriorPredictiveResult:
        base = dict(
            y_samples=np.zeros((2, 2)),
            summary={"mean": 70.0, "sd": 10.0},
            reasonable=True,
            bounds=(0.0, 100.0),
            fraction_in_bounds=1.0,
            checks={
                "mean": {"value": 70.0, "low": 60.0, "high": 90.0, "passed": True},
                "sd": {"value": 10.0, "low": 5.0, "high": 20.0, "passed": True},
            },
        )
        base.update(overrides)
        return PriorPredictiveResult(**base)

    def test_clean_result_yields_no_suggestions(self):
        assert _suggestions_from_result(self._result(), 72.0, 8.0) == []

    def test_out_of_bounds_mass_flagged(self):
        suggestions = _suggestions_from_result(
            self._result(reasonable=False, fraction_in_bounds=0.4), 72.0, 8.0
        )
        assert any("40%" in s for s in suggestions)

    def test_failed_mean_check_suggests_shift(self):
        result = self._result(
            checks={
                "mean": {"value": 30.0, "low": 60.0, "high": 90.0, "passed": False},
                "sd": {"value": 10.0, "low": 5.0, "high": 20.0, "passed": True},
            }
        )
        suggestions = _suggestions_from_result(result, 72.0, 8.0)
        assert any("+42.0" in s for s in suggestions)

    def test_failed_sd_check_suggests_scaling(self):
        result = self._result(
            checks={
                "mean": {"value": 70.0, "low": 60.0, "high": 90.0, "passed": True},
                "sd": {"value": 40.0, "low": 5.0, "high": 20.0, "passed": False},
            }
        )
        suggestions = _suggestions_from_result(result, 72.0, 8.0)
        assert any("0.20" in s for s in suggestions)


class TestRender:
    def _screen(self, **overrides) -> TransformScreen:
        base = dict(
            transform="identity",
            fraction_in_bounds=0.97,
            reasonable=True,
            summary={"mean": 71.2, "sd": 9.4},
            checks={},
        )
        base.update(overrides)
        return TransformScreen(**base)

    def test_block_has_row_per_transform(self):
        md, payload = render_prior_block(
            [self._screen(), self._screen(transform="offset_logit")]
        )
        assert md.count("| identity |") == 1
        assert md.count("| offset_logit |") == 1
        assert len(payload["screens"]) == 2

    def test_suggestions_rendered_when_present(self):
        md, payload = render_prior_block(
            [self._screen(suggestions=["scale sigma_obs by ~0.5"])]
        )
        assert "nothing is pruned" in md
        assert "scale sigma_obs by ~0.5" in md
        assert payload["screens"][0]["suggestions"] == ["scale sigma_obs by ~0.5"]

    def test_no_suggestion_section_when_clean(self):
        md, _ = render_prior_block([self._screen()])
        assert "Suggestions" not in md
