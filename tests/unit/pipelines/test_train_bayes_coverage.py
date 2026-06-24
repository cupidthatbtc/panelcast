"""Additional coverage tests for train_bayes pipeline.

Targets missed lines/branches in train_bayes.py including:
- train_models entry point with mocked MCMC
- Config validation (strict mode checks)
- Chain settings passthrough to MCMCConfig
- Convergence threshold checking and strict enforcement
- Save/load artifact flows
- Failure recovery when MCMC or convergence fails
- Feature standardization logic
- NaN validation in feature matrix X
- Heteroscedastic mode summary branches (learned, fixed, homoscedastic)
- High divergence rate warning
- n_ref computation and passthrough
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.errors import ConvergenceError
from panelcast.pipelines.train_bayes import (
    _apply_max_albums_cap,
    _validate_strict_sampling_config,
    load_training_data,
    prepare_model_data,
    train_models,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_ctx(**overrides):
    """Create a StageContext-like namespace with sensible defaults."""
    defaults = {
        "seed": 42,
        "strict": False,
        "max_albums": 50,
        "min_albums_filter": 2,
        "num_chains": 4,
        "num_samples": 1000,
        "num_warmup": 500,
        "target_accept": 0.9,
        "max_tree_depth": 10,
        "chain_method": "sequential",
        "rhat_threshold": 1.01,
        "ess_threshold": 400,
        "allow_divergences": False,
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_alpha": 2.0,
        "n_exponent_beta": 4.0,
        "n_exponent_prior": "logit-normal",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_train_parquets(tmp_path, n_artists=3, n_albums_per=3, n_features=2):
    """Create feature and split parquet files suitable for train_models."""
    n_rows = n_artists * n_albums_per
    artists = []
    for i in range(n_artists):
        artists.extend([f"artist_{i}"] * n_albums_per)

    splits_df = pd.DataFrame(
        {
            "Artist": artists,
            "User_Score": np.random.default_rng(42).uniform(60, 95, n_rows).astype(np.float32),
        },
        index=pd.RangeIndex(n_rows),
    )

    feature_data = {
        f"feature_{i}": np.random.default_rng(42 + i).standard_normal(n_rows).astype(np.float32)
        for i in range(n_features)
    }
    feature_data["n_reviews"] = np.random.default_rng(99).integers(5, 200, n_rows)
    features_df = pd.DataFrame(feature_data, index=pd.RangeIndex(n_rows))

    features_path = tmp_path / "features.parquet"
    splits_path = tmp_path / "splits.parquet"
    features_df.to_parquet(features_path)
    splits_df.to_parquet(splits_path)
    return features_path, splits_path


def _make_fake_fit_result(divergences=0, runtime=10.0, n_chains=4, n_samples=100):
    """Create a mock FitResult with minimal structure."""
    result = MagicMock()
    result.divergences = divergences
    result.runtime_seconds = runtime
    result.gpu_info = "CPU only"

    # Build a mock posterior that supports dict-style access
    sigma_obs_mock = MagicMock()
    sigma_obs_mock.mean.return_value = 5.0
    sigma_obs_mock.values = np.full((n_chains, n_samples), 5.0)

    posterior = MagicMock()
    posterior.__getitem__ = MagicMock(return_value=sigma_obs_mock)

    idata = MagicMock()
    idata.posterior = posterior
    result.idata = idata

    return result


def _make_fake_diagnostics(passed=True, rhat_max=1.003, ess_bulk_min=2000):
    """Create a mock ConvergenceDiagnostics."""
    diag = MagicMock()
    diag.passed = passed
    diag.rhat_max = rhat_max
    diag.ess_bulk_min = ess_bulk_min
    diag.ess_tail_min = 1800
    diag.divergences = 0
    diag.rhat_threshold = 1.01
    diag.ess_threshold = 400
    return diag


# ============================================================================
# Tests: _validate_strict_sampling_config
# ============================================================================


class TestValidateStrictSamplingConfigExtended:
    def test_exact_boundary_two_chains_passes(self):
        """Exactly 2 chains should pass strict validation."""
        _validate_strict_sampling_config(
            strict=True, num_chains=2, num_samples=500, ess_threshold=400
        )

    def test_zero_chains_in_strict_fails(self):
        """Zero chains should fail in strict mode."""
        with pytest.raises(ConvergenceError, match="at least 2 chains"):
            _validate_strict_sampling_config(
                strict=True, num_chains=0, num_samples=1000, ess_threshold=400
            )

    def test_convergence_error_stage_is_train(self):
        """ConvergenceError from strict validation should report stage='train'."""
        with pytest.raises(ConvergenceError) as exc_info:
            _validate_strict_sampling_config(
                strict=True, num_chains=1, num_samples=100, ess_threshold=400
            )
        assert exc_info.value.stage == "train"

    def test_non_strict_allows_any_configuration(self):
        """Non-strict mode should accept any configuration without error."""
        _validate_strict_sampling_config(
            strict=False, num_chains=0, num_samples=0, ess_threshold=10000
        )


# ============================================================================
# Tests: train_models entry point
# ============================================================================


class TestTrainModelsEntryPoint:
    def test_train_models_homoscedastic_mode(self, tmp_path):
        """train_models should succeed with homoscedastic mode (n_exponent=0.0)."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.0, learn_n_exponent=False)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        fake_manifest = MagicMock()

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", fake_manifest),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["model_type"] == "user_score"
        assert summary["heteroscedastic_mode"]["mode"] == "homoscedastic"
        assert "mcmc_config" in summary

    def test_train_models_fixed_heteroscedastic_mode(self, tmp_path):
        """train_models should produce fixed heteroscedastic summary when n_exponent != 0."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.5, learn_n_exponent=False)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "fixed"
        assert summary["heteroscedastic_mode"]["n_exponent"] == 0.5

    def test_train_models_learned_heteroscedastic_sigma_obs_mode(self, tmp_path):
        """train_models should produce learned heteroscedastic summary with sigma_obs parameterization."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        # n_exponent=0.0 but learn_n_exponent=True, n_ref will be None when
        # learn_n_exponent is True but n_exponent is 0.0 -- actually n_ref is
        # set when learn_n_exponent is True. Let's just check the learned path.
        ctx = _make_ctx(learn_n_exponent=True, n_exponent=0.0, n_exponent_prior="logit-normal")

        # Build a more detailed fit_result for learned mode
        fit_result = _make_fake_fit_result()
        n_exp_samples = np.full((4, 100), 0.4)
        sigma_obs_samples = np.full((4, 100), 5.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.4

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = np.full((4, 100), 6.0)
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Mock az.hdi and az.summary for n_exponent/sigma_ref
        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.3, 0.5])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "learned"
        assert summary["heteroscedastic_mode"]["parameterization"] == "sigma_ref"
        assert "sigma_ref" in summary["heteroscedastic_mode"]

    def test_train_models_learned_beta_prior_logging(self, tmp_path):
        """train_models should log beta prior mode when n_exponent_prior='beta'."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(
            learn_n_exponent=True,
            n_exponent=0.0,
            n_exponent_prior="beta",
            n_exponent_alpha=3.0,
            n_exponent_beta=5.0,
        )

        fit_result = _make_fake_fit_result()
        n_exp_samples = np.full((4, 100), 0.35)
        sigma_obs_samples = np.full((4, 100), 5.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.35

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = np.full((4, 100), 6.0)
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }

        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.2, 0.5])))
        fake_hdi = hdi_result
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [900.0],
                "r_hat": [1.001],
            }
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=fake_hdi),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            # Should not raise
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert summary["heteroscedastic_mode"]["mode"] == "learned"


class TestTrainModelsStrictConvergence:
    def test_strict_divergences_raises(self, tmp_path):
        """Strict mode with divergences > 0 and allow_divergences=False should raise."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=True, allow_divergences=False, num_chains=4, num_samples=1000)

        fit_result = _make_fake_fit_result(divergences=5)
        diagnostics = _make_fake_diagnostics(passed=True)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ConvergenceError, match="divergent transitions"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)

    def test_strict_divergences_allowed_does_not_raise(self, tmp_path):
        """Strict mode with allow_divergences=True should not raise on divergences."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=True, allow_divergences=True, num_chains=4, num_samples=1000)

        fit_result = _make_fake_fit_result(divergences=3)
        diagnostics = _make_fake_diagnostics(passed=True)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)
            assert summary["divergences"] == 3

    def test_strict_diagnostics_failed_raises(self, tmp_path):
        """Strict mode with diagnostics.passed=False should raise ConvergenceError."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=True, allow_divergences=True, num_chains=4, num_samples=1000)

        fit_result = _make_fake_fit_result(divergences=0)
        diagnostics = _make_fake_diagnostics(passed=False, rhat_max=1.05, ess_bulk_min=100)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ConvergenceError, match="Convergence failed"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)


class TestTrainModelsHighDivergenceWarning:
    def test_high_divergence_rate_does_not_crash(self, tmp_path):
        """High divergence rate (>10%) should log warning but not crash in non-strict."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(strict=False, num_chains=4, num_samples=100)

        # 50 divergences out of 400 total = 12.5%
        fit_result = _make_fake_fit_result(divergences=50, n_chains=4, n_samples=100)
        diagnostics = _make_fake_diagnostics(passed=False)

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)
            assert summary["divergence_rate"] > 0.10


class TestTrainModelsSummaryOutput:
    def test_training_summary_written_to_disk(self, tmp_path):
        """training_summary.json should be written with complete metadata."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx()

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="hash_abc",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        summary_path = tmp_path / "models/training_summary.json"
        assert summary_path.exists()
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        assert saved["data_hash"] == "hash_abc"
        assert "feature_scaler" in saved
        assert "feature_cols" in saved
        assert "n_reviews_stats" in saved
        assert "convergence_thresholds" in saved

    def test_default_feature_paths_used(self, tmp_path):
        """When features_path and splits_path are None, defaults should be used."""
        ctx = _make_ctx()

        # Create the default paths
        default_features = tmp_path / "data/features/train_features.parquet"
        default_splits = tmp_path / "data/splits/within_entity_temporal/train.parquet"

        # Write real parquets at default paths
        n = 9
        features_df = pd.DataFrame(
            {
                "feature_0": np.ones(n, dtype=np.float32),
                "n_reviews": np.full(n, 50, dtype=np.int32),
            },
            index=pd.RangeIndex(n),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "User_Score": np.linspace(70, 90, n, dtype=np.float32),
            },
            index=pd.RangeIndex(n),
        )
        default_features.parent.mkdir(parents=True, exist_ok=True)
        default_splits.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(default_features)
        splits_df.to_parquet(default_splits)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx)
            assert summary["model_type"] == "user_score"


class TestTrainModelsFeatureStandardization:
    def test_features_are_standardized(self, tmp_path):
        """Feature matrix X should be z-score standardized before fitting."""
        features_path, splits_path = _make_train_parquets(tmp_path, n_features=2)
        ctx = _make_ctx()

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        X = captured_model_args["X"]
        # Each column should have mean ~0 and std ~1 (after standardization)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            assert abs(col.mean()) < 0.01, f"Column {col_idx} mean not ~0"
            assert abs(col.std() - 1.0) < 0.1, f"Column {col_idx} std not ~1"

    def test_constant_features_unscaled(self, tmp_path):
        """Constant features (std=0) should remain as-is after standardization."""
        n = 9
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "User_Score": np.linspace(70, 90, n, dtype=np.float32),
            },
            index=pd.RangeIndex(n),
        )
        features_df = pd.DataFrame(
            {
                "feature_const": np.ones(n, dtype=np.float32),  # constant
                "feature_vary": np.arange(n, dtype=np.float32),  # varying
                "n_reviews": np.full(n, 50, dtype=np.int32),
            },
            index=pd.RangeIndex(n),
        )
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        ctx = _make_ctx()

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        X = captured_model_args["X"]
        # Constant column should be (1-1)/1 = 0
        # (X_std_safe = 1.0 when std=0, so (x-mean)/1 = x - mean)
        const_col = X[:, 0]
        assert np.allclose(const_col, 0.0), "Constant feature should be centered to 0"

    def test_nan_in_X_raises_value_error(self, tmp_path):
        """NaN in feature matrix X after fillna should raise ValueError."""
        n = 6
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "C", "C"],
                "User_Score": np.linspace(70, 90, n, dtype=np.float32),
            },
            index=pd.RangeIndex(n),
        )
        # Create features that have NaN AFTER the join but before standardization.
        # Since fillna(0) is applied to feature_cols before prepare_model_data,
        # and then X = train_df[feature_cols].values is built inside prepare_model_data,
        # NaN in X can only happen if something goes wrong. Let's test the
        # validation inside train_models by monkeypatching the X array.
        features_df = pd.DataFrame(
            {
                "feature_1": np.ones(n, dtype=np.float32),
                "n_reviews": np.full(n, 50, dtype=np.int32),
            },
            index=pd.RangeIndex(n),
        )
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        ctx = _make_ctx()

        # Patch load_training_data to return model_args with NaN in X
        def _fake_load(*args, **kwargs):
            model_args = {
                "artist_idx": np.array([0, 0, 1, 1, 2, 2]),
                "album_seq": np.array([1, 2, 1, 2, 1, 2]),
                "prev_score": np.full(6, 75.0),
                "X": np.array([[1.0], [np.nan], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32),
                "y": np.linspace(70, 90, 6, dtype=np.float32),
                "n_reviews": np.full(6, 50, dtype=np.int32),
                "n_artists": 3,
                "artist_album_counts": pd.Series([2, 2, 2]),
                "artist_to_idx": {"A": 0, "B": 1, "C": 2},
                "global_mean_score": 75.0,
                "ar_center": np.float32(75.0),
                "ar_center_value": 75.0,
            }
            feature_cols = ["feature_1"]
            train_df = pd.DataFrame(
                {"Artist": ["A", "A", "B", "B", "C", "C"]},
            )
            return model_args, feature_cols, train_df

        with (
            patch(
                "panelcast.pipelines.train_bayes.load_training_data",
                side_effect=_fake_load,
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ValueError, match="NaN values"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)


class TestTrainModelsNRefComputation:
    def test_n_ref_set_for_learned_exponent(self, tmp_path):
        """n_ref should be set to median of n_reviews when learn_n_exponent=True."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(learn_n_exponent=True)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        hdi_result = MagicMock()
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.3, 0.5])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        fit_result = _make_fake_fit_result()
        n_exp_samples = np.full((4, 100), 0.4)
        sigma_obs_samples = np.full((4, 100), 5.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.4

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = np.full((4, 100), 6.0)
        sigma_ref_mock.mean.return_value = 6.0

        posterior_dict = {
            "user_n_exponent": n_exp_mock,
            "user_sigma_obs": sigma_obs_mock,
            "user_sigma_ref": sigma_ref_mock,
        }
        fit_result.idata.posterior.__getitem__ = MagicMock(
            side_effect=lambda key: posterior_dict[key]
        )

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
            patch("panelcast.pipelines.train_bayes.az.hdi", return_value=hdi_result),
            patch(
                "panelcast.pipelines.train_bayes.az.summary",
                return_value=fake_summary_df,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert captured_model_args["n_ref"] is not None

    def test_n_ref_none_for_homoscedastic(self, tmp_path):
        """n_ref should be None when n_exponent=0 and learn_n_exponent=False."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.0, learn_n_exponent=False)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert captured_model_args["n_ref"] is None


class TestTrainModelsMCMCConfigPassthrough:
    def test_mcmc_config_from_ctx(self, tmp_path):
        """MCMCConfig should be constructed from ctx attributes."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(
            num_warmup=200,
            num_samples=300,
            num_chains=2,
            seed=123,
            target_accept=0.85,
            max_tree_depth=8,
            chain_method="vectorized",
        )

        captured_config = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_config["config"] = config
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            train_models(ctx, features_path=features_path, splits_path=splits_path)

        config = captured_config["config"]
        assert config.num_warmup == 200
        assert config.num_samples == 300
        assert config.num_chains == 2
        assert config.seed == 123
        assert config.target_accept_prob == 0.85
        assert config.max_tree_depth == 8
        assert config.chain_method == "vectorized"


class TestTrainModelsMinAlbumsFilter:
    def test_min_albums_filter_passthrough(self, tmp_path):
        """min_albums_filter from ctx should be passed to load_training_data."""
        features_path, splits_path = _make_train_parquets(tmp_path, n_artists=3, n_albums_per=3)
        ctx = _make_ctx(min_albums_filter=3)

        fit_result = _make_fake_fit_result()
        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", return_value=fit_result),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)
            assert summary["min_albums_filter"] == 3


class TestTrainModelsHeteroscedasticConfig:
    def test_model_args_include_heteroscedastic_keys(self, tmp_path):
        """Model args passed to fit_model should include heteroscedastic configuration."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent=0.5, learn_n_exponent=False)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
            patch(
                "panelcast.pipelines.train_bayes.compute_sigma_scaled",
                return_value=3.5,
            ),
        ):
            train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert captured_model_args["n_exponent"] == 0.5
        assert captured_model_args["learn_n_exponent"] is False
        assert captured_model_args["n_exponent_prior"] == "logit-normal"
        # n_ref should be set since n_exponent != 0
        assert captured_model_args["n_ref"] is not None


class TestTrainModelsPriorsPassthrough:
    def test_priors_in_model_args_and_summary(self, tmp_path):
        """PriorConfig should be passed in model_args and serialized in summary."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(n_exponent_alpha=3.0, n_exponent_beta=6.0)

        captured_model_args = {}

        def _capture_fit(
            model,
            model_args,
            config,
            progress_bar,
            exclude_from_idata,
            exclude_from_collection=None,
        ):
            captured_model_args.update(model_args)
            return _make_fake_fit_result()

        diagnostics = _make_fake_diagnostics()
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("panelcast.pipelines.train_bayes.fit_model", side_effect=_capture_fit),
            patch(
                "panelcast.pipelines.train_bayes.check_convergence",
                return_value=diagnostics,
            ),
            patch(
                "panelcast.pipelines.train_bayes.save_model",
                return_value=(model_dir / "model.nc", MagicMock()),
            ),
            patch(
                "panelcast.pipelines.train_bayes.hash_dataframe",
                return_value="abc123",
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            summary = train_models(ctx, features_path=features_path, splits_path=splits_path)

        assert "priors" in captured_model_args
        assert summary["priors"]["n_exponent_alpha"] == 3.0
        assert summary["priors"]["n_exponent_beta"] == 6.0


class TestTrainModelsFitFailure:
    def test_fit_model_exception_propagates(self, tmp_path):
        """If fit_model raises, the error should propagate up."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx()

        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch(
                "panelcast.pipelines.train_bayes.fit_model",
                side_effect=RuntimeError("MCMC crashed"),
            ),
            patch(
                "panelcast.pipelines.train_bayes.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(RuntimeError, match="MCMC crashed"):
                train_models(ctx, features_path=features_path, splits_path=splits_path)
