"""Parametrized target-transform axis for evaluate-pipeline helpers.

Checks that _transform_from_summary resolves the trained transform and that
_prepare_test_model_args feeds prev_score to the model on the training scale
while keeping y_true on the raw score scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.evaluate import (
    _prepare_test_model_args,
    _transform_from_summary,
)

TRANSFORMS = ["identity", "offset_logit"]


def _make_summary(target_transform: str | None = None) -> dict:
    priors: dict = {
        "mu_artist_scale": 1.0,
        "sigma_artist_scale": 0.5,
        "sigma_rw_scale": 0.1,
        "rho_scale": 0.3,
        "beta_scale": 1.0,
        "sigma_obs_scale": 1.0,
        "n_exponent_alpha": 2.0,
        "n_exponent_beta": 4.0,
    }
    summary: dict = {
        "artist_to_idx": {"A": 0, "B": 1},
        "max_seq": 5,
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
        "priors": priors,
    }
    if target_transform is not None:
        summary["target_transform"] = target_transform
        summary["logit_offset"] = 0.5
        priors["target_transform"] = target_transform
    return summary


class TestTransformFromSummary:
    def test_legacy_summary_resolves_to_identity(self):
        """Summaries written before the transform existed must keep the
        legacy identity behaviour."""
        assert _transform_from_summary(_make_summary()).name == "identity"

    def test_explicit_none_resolves_to_identity(self):
        summary = _make_summary()
        summary["target_transform"] = None
        assert _transform_from_summary(summary).name == "identity"

    @pytest.mark.parametrize("transform_name", TRANSFORMS)
    def test_recorded_transform_resolved(self, transform_name):
        summary = _make_summary(transform_name)
        assert _transform_from_summary(summary).name == transform_name

    def test_bounds_come_from_dataset_block(self):
        summary = _make_summary("offset_logit")
        summary["dataset"] = {"target_bounds": [0.0, 10.0]}
        t = _transform_from_summary(summary)
        # forward(10) on (0,10) bounds maps the upper bound to logit(10.5/11)
        expected = float(np.log(10.5 / 11.0) - np.log(0.5 / 11.0))
        assert float(t.forward(np.float32(10.0))) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("transform_name", TRANSFORMS)
class TestPrepareTestModelArgsTransformAxis:
    """prev_score enters the model on the training scale; y_true stays raw."""

    def _prepare(self, transform_name):
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame({"f1": [1.0], "n_reviews": [10]})
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )
        summary = _make_summary(transform_name)
        return _prepare_test_model_args(test_df, test_features, summary, train_df=train_df)

    def test_prev_score_on_model_scale(self, transform_name):
        model_args, _, _ = self._prepare(transform_name)
        # Artist A's only test row takes the last train score (75) as
        # prev_score, forward-transformed onto the training scale.
        t = get_transform(transform_name, (0.0, 100.0), offset=0.5)
        expected = float(t.forward(np.float32(75.0)))
        assert float(model_args["prev_score"][0]) == pytest.approx(expected, rel=1e-5)

    def test_y_true_stays_on_raw_scale(self, transform_name):
        _, y_true, _ = self._prepare(transform_name)
        assert y_true[0] == pytest.approx(80.0)

    def test_priors_carry_transform(self, transform_name):
        model_args, _, _ = self._prepare(transform_name)
        assert model_args["priors"].target_transform == transform_name
