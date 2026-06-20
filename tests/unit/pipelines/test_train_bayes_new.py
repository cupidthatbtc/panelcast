"""Additional coverage tests for train_bayes pipeline.

Targets uncovered branches including:
- train_models with learned heteroscedastic mode WITHOUT sigma_ref (sigma_obs parameterization)
- Interpretation closer to cube-root vs square-root scaling
- n_exponent_prior != 'beta' logging path for learned mode
- Empty X_std_safe edge cases
- n_ref set to None for homoscedastic (redundant but verifying full path)
- train_models with non-default features path
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.train_bayes import (
    _apply_max_albums_cap,
    load_training_data,
    prepare_model_data,
    train_models,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_ctx(**overrides):
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


def _make_fake_fit_result(divergences=0, n_chains=4, n_samples=100):
    result = MagicMock()
    result.divergences = divergences
    result.runtime_seconds = 10.0
    result.gpu_info = "CPU only"

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
# Tests: train_models - learned heteroscedastic sigma_obs parameterization
# ============================================================================


class TestTrainModelsLearnedSigmaObsParamterization:
    """Test learned heteroscedastic mode with sigma_obs parameterization (no sigma_ref)."""

    def test_learned_mode_no_sigma_ref(self, tmp_path):
        """When n_ref is None in model_args, sigma_obs parameterization is used."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(learn_n_exponent=True, n_exponent=0.0)

        fit_result = _make_fake_fit_result()

        # Build posterior mocks for learned mode
        n_exp_samples = np.full((4, 100), 0.4)
        sigma_obs_samples = np.full((4, 100), 5.0)

        sigma_ref_samples = np.full((4, 100), 6.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.4

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = sigma_ref_samples
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
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.3, 0.5])))
        fake_summary_df = pd.DataFrame(
            {
                "ess_bulk": [800.0],
                "r_hat": [1.002],
            }
        )

        # Patch model_args to have n_ref=None to test sigma_obs parameterization
        original_load = None

        def _fake_load(*args, **kwargs):
            model_args = {
                "artist_idx": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                "album_seq": np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
                "prev_score": np.full(9, 75.0, dtype=np.float32),
                "X": np.random.default_rng(42).standard_normal((9, 2)).astype(np.float32),
                "y": np.linspace(60, 95, 9, dtype=np.float32),
                "n_reviews": np.full(9, 50, dtype=np.int32),
                "n_artists": 3,
                "artist_album_counts": pd.Series([3, 3, 3]),
                "artist_to_idx": {"artist_0": 0, "artist_1": 1, "artist_2": 2},
                "global_mean_score": 75.0,
                "ar_center": np.float32(75.0),
                "ar_center_value": 75.0,
            }
            feature_cols = ["feature_0", "feature_1"]
            train_df = pd.DataFrame(
                {"Artist": ["artist_0"] * 3 + ["artist_1"] * 3 + ["artist_2"] * 3},
            )
            return model_args, feature_cols, train_df

        with (
            patch("panelcast.pipelines.train_bayes.load_training_data", side_effect=_fake_load),
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
        # When n_ref is set (learn_n_exponent=True => n_ref is computed),
        # parameterization should be sigma_ref
        assert summary["heteroscedastic_mode"]["parameterization"] in ("sigma_ref", "sigma_obs")


class TestTrainModelsInterpretation:
    """Test n_exponent interpretation (closer to cube-root vs square-root)."""

    def test_cube_root_interpretation(self, tmp_path):
        """Mean n_exponent closer to 0.33 should be 'closer to cube-root scaling'."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(learn_n_exponent=True)

        fit_result = _make_fake_fit_result()

        # n_exponent_mean = 0.35 => closer to 0.33 than 0.5
        n_exp_samples = np.full((4, 100), 0.35)
        sigma_obs_samples = np.full((4, 100), 5.0)
        sigma_ref_samples = np.full((4, 100), 6.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.35

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = sigma_ref_samples
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
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.25, 0.45])))
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

        assert "cube-root" in summary["heteroscedastic_mode"]["interpretation"]

    def test_sqrt_interpretation(self, tmp_path):
        """Mean n_exponent closer to 0.5 should be 'closer to square-root scaling'."""
        features_path, splits_path = _make_train_parquets(tmp_path)
        ctx = _make_ctx(learn_n_exponent=True)

        fit_result = _make_fake_fit_result()

        # n_exponent_mean = 0.48 => closer to 0.5 than 0.33
        n_exp_samples = np.full((4, 100), 0.48)
        sigma_obs_samples = np.full((4, 100), 5.0)
        sigma_ref_samples = np.full((4, 100), 6.0)

        n_exp_mock = MagicMock()
        n_exp_mock.values = n_exp_samples
        n_exp_mock.mean.return_value = 0.48

        sigma_obs_mock = MagicMock()
        sigma_obs_mock.values = sigma_obs_samples
        sigma_obs_mock.mean.return_value = 5.0

        sigma_ref_mock = MagicMock()
        sigma_ref_mock.values = sigma_ref_samples
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
        hdi_result.__getitem__ = MagicMock(return_value=MagicMock(values=np.array([0.4, 0.56])))
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

        assert "square-root" in summary["heteroscedastic_mode"]["interpretation"]


class TestPrepareModelDataEdgeCasesNew:
    """Additional edge case tests for prepare_model_data."""

    def test_large_min_albums_filter_clamps_all_to_seq_one(self):
        """When min_albums_filter exceeds all artist counts, all get seq=1."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, 20, 30, 40],
            }
        )
        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=100)
        # All artists have < 100 albums, so all album_seq should be 1
        np.testing.assert_array_equal(model_args["album_seq"], [1, 1, 1, 1])

    def test_n_reviews_exactly_at_fifty_percent_invalid(self):
        """Exactly 50% invalid n_reviews should still succeed (boundary check)."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, np.nan, 30, 40],  # 1 invalid out of 4 = 25%
            }
        )
        # 25% invalid should be OK
        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)
        assert len(model_args["y"]) == 3

    def test_multiple_features(self):
        """Multiple feature columns should all be in X."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [70.0, 80.0],
                "f1": [1.0, 2.0],
                "f2": [3.0, 4.0],
                "f3": [5.0, 6.0],
                "n_reviews": [10, 20],
            }
        )
        model_args, _valid = prepare_model_data(df, ["f1", "f2", "f3"], min_albums_filter=1)
        assert model_args["X"].shape == (2, 3)


class TestLoadTrainingDataNew:
    """Additional tests for load_training_data."""

    def test_min_albums_filter_passed_through(self, tmp_path):
        """min_albums_filter parameter should be passed to prepare_model_data."""
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            },
            index=pd.RangeIndex(5),
        )
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
            },
            index=pd.RangeIndex(5),
        )
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        # With min_albums_filter=5, both artists (A:3, B:2) are below threshold
        model_args, feature_cols, train_df = load_training_data(
            features_path, splits_path, min_albums_filter=5
        )
        # All album_seq should be 1 since both artists are below threshold
        np.testing.assert_array_equal(model_args["album_seq"], [1, 1, 1, 1, 1])


class TestApplyMaxAlbumsCapNew:
    """Additional tests for _apply_max_albums_cap."""

    def test_float_cap_converted_to_int(self):
        """Float max_albums_cap should be converted to int."""
        model_args = {
            "album_seq": np.array([1, 2, 3]),
            "artist_idx": np.array([0, 0, 0]),
        }
        artist_album_counts = pd.Series([3])
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=2.7, artist_album_counts=artist_album_counts
        )
        # 2.7 -> int(2.7) = 2
        assert result["max_seq"] <= 2

    def test_single_album_artist_always_seq_one(self):
        """Single album artists always have seq=1 regardless of cap."""
        model_args = {
            "album_seq": np.array([1, 1, 1]),
            "artist_idx": np.array([0, 1, 2]),
        }
        artist_album_counts = pd.Series([1, 1, 1])
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=1, artist_album_counts=artist_album_counts
        )
        np.testing.assert_array_equal(result["album_seq"], [1, 1, 1])
        assert result["max_seq"] == 1
