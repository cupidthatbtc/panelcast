"""Tests for the opt-in sensitivity stage wiring and suite helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.pipelines.sensitivity import (
    SensitivityResult,
    _feature_groups_from_names,
    run_feature_ablation,
)
from panelcast.pipelines.stages import (
    build_optional_stages,
    build_pipeline_stages,
    get_execution_order,
    get_stage,
)

runner = CliRunner()


class TestSensitivityStageRegistry:
    def test_not_in_default_stage_list(self):
        names = [s.name for s in build_pipeline_stages()]
        assert "sensitivity" not in names

    def test_default_execution_order_excludes_sensitivity(self):
        names = [s.name for s in get_execution_order(None)]
        assert "sensitivity" not in names

    def test_named_selection_includes_sensitivity(self):
        order = get_execution_order(["sensitivity"])
        assert [s.name for s in order] == ["sensitivity"]

    def test_named_selection_with_dependency_orders_after_evaluate(self):
        order = get_execution_order(["sensitivity", "evaluate"])
        names = [s.name for s in order]
        assert names.index("evaluate") < names.index("sensitivity")

    def test_get_stage_finds_sensitivity(self):
        stage = get_stage("sensitivity")
        assert stage.name == "sensitivity"
        assert stage.depends_on == ["evaluate"]

    def test_optional_stages_registry(self):
        assert [s.name for s in build_optional_stages()] == ["sensitivity"]

    def test_unknown_stage_error_lists_sensitivity(self):
        with pytest.raises(KeyError, match="sensitivity"):
            get_execution_order(["bogus_stage"])


class TestFeatureGroupsFromNames:
    def test_aoty_feature_names_map_to_groups(self):
        feature_cols = [
            "album_sequence",
            "career_years",
            "release_gap_days",
            "release_year",
            "date_risk_ordinal",
            "date_missing",
            "is_album",
            "is_ep",
            "user_prior_mean",
            "user_prior_std",
            "user_prior_count",
            "user_trajectory",
            "critic_prior_mean",
            "critic_prior_std",
            "critic_prior_count",
            "critic_trajectory",
            "is_debut",
            "genre_pc_0",
            "genre_pc_1",
        ]
        summary = {"dataset": {"model_prefix": "user", "secondary_prefix": "critic"}}
        groups = _feature_groups_from_names(feature_cols, summary)
        assert groups["genre"] == [17, 18]
        assert set(groups["temporal"]) == {0, 1, 2, 3, 4, 5}
        assert 8 in groups["artist_history"]  # user_prior_mean
        assert 16 in groups["artist_history"]  # is_debut
        assert 6 not in groups["artist_history"]  # is_album stays out

    def test_empty_groups_dropped(self):
        groups = _feature_groups_from_names(
            ["album_sequence"], {"dataset": {"model_prefix": "perf"}}
        )
        assert "genre" not in groups
        assert "artist_history" not in groups
        assert groups["temporal"] == [0]


class TestFeatureAblationBaselineReuse:
    def _model_args(self, n_obs: int = 8, n_features: int = 4) -> dict:
        return {"X": np.zeros((n_obs, n_features), dtype=np.float32)}

    @patch("panelcast.pipelines.sensitivity.extract_coefficient_summary")
    @patch("panelcast.pipelines.sensitivity.check_convergence")
    @patch("panelcast.pipelines.sensitivity.fit_model")
    def test_baseline_reused_skips_refit(self, mock_fit, mock_conv, mock_coeff):
        mock_fit.return_value = MagicMock(idata=MagicMock(), mcmc=MagicMock())
        mock_conv.return_value = MagicMock(passed=True, rhat_max=1.0, divergences=0)
        mock_coeff.return_value = pd.DataFrame()

        baseline = SensitivityResult(
            name="full",
            config={},
            convergence=mock_conv.return_value,
            loo=None,
        )
        results = run_feature_ablation(
            model=lambda: None,
            model_args=self._model_args(),
            feature_groups={"genre": [0, 1]},
            compute_loo_cv=False,
            baseline=baseline,
        )
        # One fit for the ablated variant only — no baseline refit (Q6).
        assert mock_fit.call_count == 1
        assert results["full"] is baseline
        assert "no_genre" in results

    @patch("panelcast.pipelines.sensitivity.extract_coefficient_summary")
    @patch("panelcast.pipelines.sensitivity.check_convergence")
    @patch("panelcast.pipelines.sensitivity.fit_model")
    def test_without_baseline_fits_full_model(self, mock_fit, mock_conv, mock_coeff):
        mock_fit.return_value = MagicMock(idata=MagicMock(), mcmc=MagicMock())
        mock_conv.return_value = MagicMock(passed=True, rhat_max=1.0, divergences=0)
        mock_coeff.return_value = pd.DataFrame()

        run_feature_ablation(
            model=lambda: None,
            model_args=self._model_args(),
            feature_groups={"genre": [0, 1]},
            compute_loo_cv=False,
        )
        assert mock_fit.call_count == 2  # full + no_genre


class TestSensitivitySuiteRealFitSmoke:
    """Non-mocked smoke: the suite's model_args must survive a REAL fit.

    Regression: run_sensitivity_suite left global_std_score/effective_ceiling
    (emitted by prepare_model_data) in model_args; the model function has no
    **kwargs, so the first real refit died with a TypeError. Suite tests that
    stub both load_training_data and fit_model could never see it.
    """

    @pytest.mark.timeout(300)
    def test_ablation_axis_runs_real_fit(self, tmp_path, monkeypatch):
        import json
        from types import SimpleNamespace

        from panelcast.config.descriptor import DatasetDescriptor
        from panelcast.pipelines.sensitivity import run_sensitivity_suite

        monkeypatch.chdir(tmp_path)

        n_artists, n_per = 3, 3
        n = n_artists * n_per
        rng = np.random.default_rng(0)
        artists = [f"artist_{i}" for i in range(n_artists) for _ in range(n_per)]
        splits_dir = tmp_path / "data" / "splits" / "within_entity_temporal"
        splits_dir.mkdir(parents=True)
        pd.DataFrame(
            {"Artist": artists, "User_Score": rng.uniform(60, 90, n).astype(np.float32)}
        ).to_parquet(splits_dir / "train.parquet")

        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)
        # Names deliberately match no ablation group -> only the "full"
        # baseline fits, keeping the smoke to a single tiny MCMC run.
        pd.DataFrame(
            {
                "feature_0": rng.standard_normal(n).astype(np.float32),
                "feature_1": rng.standard_normal(n).astype(np.float32),
                "n_reviews": rng.integers(5, 200, n),
            }
        ).to_parquet(features_dir / "train_features.parquet")

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        summary = {
            "debut_prev_score_source": "train_mean",
            "target_transform": "identity",
            "logit_offset": 0.5,
            "priors": {"ar_center": "global"},
            "n_ref": None,
            "likelihood_df": 4.0,
        }
        (models_dir / "training_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        ctx = SimpleNamespace(
            descriptor=DatasetDescriptor(),
            sensitivity_axes=("ablation",),
            min_albums_filter=2,
            max_albums=50,
            num_warmup=15,
            num_samples=15,
            num_chains=1,
            seed=0,
            target_accept=0.9,
            max_tree_depth=10,
            chain_method="sequential",
            n_exponent=0.0,
            learn_n_exponent=False,
            progress_bar=False,
        )

        result = run_sensitivity_suite(ctx)

        assert "sensitivity_results" in result
        payload = json.loads(
            (tmp_path / "reports" / "sensitivity" / "sensitivity_results.json").read_text(
                encoding="utf-8"
            )
        )
        assert "ablation" in payload
        assert payload["ablation"], "no ablation rows recorded"


class TestStageSensitivityCli:
    def test_stage_sensitivity_dispatch(self, monkeypatch):
        captured = {}

        def fake_run_pipeline(config):
            captured["config"] = config
            return 0

        monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", fake_run_pipeline)
        result = runner.invoke(
            app,
            ["stage", "sensitivity", "--num-samples", "300", "--num-chains", "1"],
        )
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.stages == ["sensitivity"]
        assert config.num_samples == 300
        assert config.num_chains == 1
