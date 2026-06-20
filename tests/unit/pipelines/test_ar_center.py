"""Tests for the AR(1) centering gate (rho/mu_artist deconfound)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.train_bayes import prepare_model_data
from panelcast.pipelines.training_summary import ar_center_on_model_scale


def _train_df() -> pd.DataFrame:
    """Three artists; A has three albums, B two, C one; train mean = 75."""
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "B", "B", "C"],
            "User_Score": [70.0, 75.0, 80.0, 72.0, 78.0, 75.0],
            "feature_1": np.zeros(6, dtype=np.float32),
            "n_reviews": np.full(6, 50, dtype=np.int32),
        }
    )


class TestPrepareModelDataArCenter:
    def test_global_center_equals_debut_fill(self):
        """Default centering shares the debut-fill value, so debut rows have
        prev_score - center == 0 exactly."""
        model_args, _ = prepare_model_data(_train_df(), ["feature_1"])
        train_mean = _train_df()["User_Score"].mean()
        assert float(model_args["ar_center"]) == pytest.approx(train_mean)
        assert model_args["ar_center_value"] == pytest.approx(train_mean)
        prev = np.asarray(model_args["prev_score"], dtype=float)
        for debut_pos in [0, 3, 5]:
            assert prev[debut_pos] - float(model_args["ar_center"]) == pytest.approx(0.0)

    def test_none_keeps_legacy_uncentered_form(self):
        model_args, _ = prepare_model_data(_train_df(), ["feature_1"], ar_center="none")
        assert float(model_args["ar_center"]) == 0.0
        assert model_args["ar_center_value"] == 0.0

    def test_artist_running_uses_running_mean_of_previous_scores(self):
        model_args, _ = prepare_model_data(_train_df(), ["feature_1"], ar_center="artist_running")
        center = np.asarray(model_args["ar_center"], dtype=float)
        assert center.shape == (6,)
        train_mean = _train_df()["User_Score"].mean()
        # Debuts fall back to the debut-fill value.
        for debut_pos in [0, 3, 5]:
            assert center[debut_pos] == pytest.approx(train_mean)
        # A's 2nd album: mean(70); A's 3rd: mean(70, 75); B's 2nd: mean(72).
        assert center[1] == pytest.approx(70.0)
        assert center[2] == pytest.approx(72.5)
        assert center[4] == pytest.approx(72.0)
        # The summary value stores the global fallback.
        assert model_args["ar_center_value"] == pytest.approx(train_mean)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="ar_center"):
            prepare_model_data(_train_df(), ["feature_1"], ar_center="bogus")

    def test_global_center_transforms_with_offset_logit(self):
        """Under offset_logit the center moves to the logit scale together
        with prev_score, so debut AR terms stay exactly zero."""
        model_args, _ = prepare_model_data(
            _train_df(), ["feature_1"], target_transform="offset_logit"
        )
        t = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        train_mean = _train_df()["User_Score"].mean()
        expected = float(t.forward(np.float32(train_mean)))
        assert float(model_args["ar_center"]) == pytest.approx(expected, rel=1e-5)
        # Raw-scale value recorded for consumers.
        assert model_args["ar_center_value"] == pytest.approx(train_mean)
        prev = np.asarray(model_args["prev_score"], dtype=float)
        for debut_pos in [0, 3, 5]:
            assert prev[debut_pos] - float(model_args["ar_center"]) == pytest.approx(0.0, abs=1e-6)

    def test_none_center_not_transformed(self):
        """ar_center='none' must stay 0.0 on the model scale, not forward(0)."""
        model_args, _ = prepare_model_data(
            _train_df(), ["feature_1"], target_transform="offset_logit", ar_center="none"
        )
        assert float(model_args["ar_center"]) == 0.0


class TestArCenterOnModelScale:
    def _summary(self, **extra) -> dict:
        base = {
            "priors": {"ar_center": "global"},
            "ar_center_value": 75.0,
        }
        base.update(extra)
        return base

    def test_legacy_summary_resolves_to_zero(self):
        assert ar_center_on_model_scale({"priors": {}}) == 0.0

    def test_none_mode_resolves_to_zero(self):
        summary = self._summary(priors={"ar_center": "none"}, ar_center_value=0.0)
        assert ar_center_on_model_scale(summary) == 0.0

    def test_global_identity_returns_raw_value(self):
        assert ar_center_on_model_scale(self._summary()) == pytest.approx(75.0)

    def test_global_offset_logit_forward_transforms(self):
        summary = self._summary(target_transform="offset_logit", logit_offset=0.5)
        t = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        expected = float(t.forward(np.float32(75.0)))
        assert ar_center_on_model_scale(summary) == pytest.approx(expected, rel=1e-5)

    def test_bounds_come_from_dataset_block(self):
        summary = self._summary(
            target_transform="offset_logit",
            logit_offset=0.5,
            ar_center_value=7.5,
            dataset={"target_bounds": [0.0, 10.0]},
        )
        t = get_transform("offset_logit", (0.0, 10.0), offset=0.5)
        expected = float(t.forward(np.float32(7.5)))
        assert ar_center_on_model_scale(summary) == pytest.approx(expected, rel=1e-5)


class TestLocateLevelPrior:
    def test_centered_identity_moves_loc_to_train_mean(self):
        from panelcast.models.bayes.priors import PriorConfig
        from panelcast.pipelines.train_bayes import locate_level_prior

        priors = locate_level_prior(PriorConfig(ar_center="global"), ar_center_value=75.0)
        assert priors.mu_artist_loc == pytest.approx(75.0)

    def test_centered_logit_moves_loc_to_logit_scale(self):
        from panelcast.models.bayes.priors import PriorConfig
        from panelcast.pipelines.train_bayes import locate_level_prior

        priors = locate_level_prior(
            PriorConfig(ar_center="global", target_transform="offset_logit"),
            ar_center_value=75.0,
            target_transform="offset_logit",
        )
        t = get_transform("offset_logit", (0.0, 100.0), offset=0.5)
        assert priors.mu_artist_loc == pytest.approx(float(t.forward(75.0)), rel=1e-5)

    def test_uncentered_keeps_zero_loc(self):
        from panelcast.models.bayes.priors import PriorConfig
        from panelcast.pipelines.train_bayes import locate_level_prior

        priors = locate_level_prior(PriorConfig(ar_center="none"), ar_center_value=0.0)
        assert priors.mu_artist_loc == 0.0

    def test_explicit_loc_respected(self):
        from panelcast.models.bayes.priors import PriorConfig
        from panelcast.pipelines.train_bayes import locate_level_prior

        priors = locate_level_prior(
            PriorConfig(ar_center="global", mu_artist_loc=12.0), ar_center_value=75.0
        )
        assert priors.mu_artist_loc == 12.0


class TestPipelineConfigArCenter:
    def test_default_is_global(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        assert PipelineConfig().ar_center == "global"

    def test_invalid_mode_raises(self):
        from panelcast.pipelines.orchestrator import PipelineConfig

        with pytest.raises(ValueError, match="ar_center"):
            PipelineConfig(ar_center="bogus")

    def test_resume_restores_ar_center(self):
        from panelcast.pipelines.orchestrator import PipelineOrchestrator

        assert "ar_center" in PipelineOrchestrator.RESUME_CONFIG_KEYS
