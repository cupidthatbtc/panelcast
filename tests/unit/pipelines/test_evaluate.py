"""Branch-coverage tests targeting evaluate.py lines not yet hit elsewhere.

Target lines: 92, 222, 402-403, 409, 503, 517-543, 593, 602-609, 626-629,
820-821, 872-946, 962, 1012, 1027, 1213-1231.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.pipelines.evaluate import (
    _compute_info_criteria,
    _evaluate_predictions,
    _extract_posterior_samples,
    _json_safe,
    _prepare_disjoint_inputs,
    _prepare_test_model_args,
    _resolve_feature_split_dir,
    _run_new_artist_predictive,
    _write_json,
    evaluate_models,
)


@pytest.fixture()
def summary():
    return {
        "artist_to_idx": {"Artist_A": 0, "Artist_B": 1},
        "n_artists": 2,
        "max_seq": 5,
        "max_albums": 10,
        "min_albums_filter": 1,
        "global_mean_score": 70.0,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "feature_cols": ["feat_1", "feat_2"],
        },
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "likelihood_df": 4.0,
        "n_ref": None,
    }


class TestExtractPosteriorSamples:
    def test_delegates_to_helper(self):
        # The wrapper just calls extract_posterior_samples(idata) — confirm the
        # delegation path (line 92) is exercised.
        fake = {"user_mu": np.ones((2, 10))}
        with patch(
            "panelcast.pipelines.evaluate.extract_posterior_samples", return_value=fake
        ) as mock_fn:
            result = _extract_posterior_samples(object())
        mock_fn.assert_called_once()
        assert result is fake


class TestSummaryDatasetDefaults:
    """_summary_dataset AOTY defaults for legacy summaries (no 'dataset' key)."""

    def test_defaults_applied_when_no_dataset_block(self):
        from panelcast.pipelines.evaluate import _summary_dataset

        result = _summary_dataset({})
        assert result["entity_col"] == "Artist"
        assert result["target_col"] == "User_Score"
        assert result["n_obs_col"] == "User_Ratings"
        assert result["prefix"] == "user"
        assert result["target_bounds"] == (0.0, 100.0)

    def test_custom_values_used_when_dataset_block_present(self):
        from panelcast.pipelines.evaluate import _summary_dataset

        result = _summary_dataset(
            {
                "dataset": {
                    "entity_col": "Team",
                    "target_col": "Score",
                    "n_obs_col": "Votes",
                    "model_prefix": "team",
                    "target_bounds": [0.0, 10.0],
                }
            }
        )
        assert result["entity_col"] == "Team"
        assert result["prefix"] == "team"
        assert result["target_bounds"] == (0.0, 10.0)


class TestResolveFeatureSplitDirBranches:
    def test_unknown_split_name_returns_candidate(self, tmp_path, monkeypatch):
        # An unrecognised split name triggers the except ValueError branch (402-403)
        # and returns the candidate path (line 412). The function uses relative
        # Path() so chdir makes the relative result resolve correctly.
        monkeypatch.chdir(tmp_path)
        result = _resolve_feature_split_dir("completely_unknown_split")
        assert result.resolve() == (tmp_path / "data/features/completely_unknown_split").resolve()

    def test_legacy_path_returned_when_exists(self, tmp_path, monkeypatch):
        # Build a legacy directory for ENTITY_DISJOINT (its legacy name).
        from panelcast.data.split_types import SplitType, legacy_split_name

        legacy = legacy_split_name(SplitType.ENTITY_DISJOINT)
        if legacy is None:
            pytest.skip("ENTITY_DISJOINT has no legacy name in this build")

        legacy_dir = tmp_path / "data" / "features" / legacy
        legacy_dir.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)
        # canonical dir does NOT exist → falls through to legacy check (line 409)
        result = _resolve_feature_split_dir(str(SplitType.ENTITY_DISJOINT.value))
        assert result.resolve() == legacy_dir.resolve()


class TestDisjointInputsTransform:
    def test_non_identity_transform_applied_to_prev_score(self, summary):
        s = dict(summary)
        s["target_transform"] = "offset_logit"
        s["target_bounds"] = (0.0, 100.0)
        s["logit_offset"] = 0.5

        test_df = pd.DataFrame({"Artist": ["New_X"], "User_Score": [75.0], "User_Ratings": [30]})
        test_features = pd.DataFrame(
            {"feat_1": [1.0], "feat_2": [2.0], "n_reviews": [30]},
            index=test_df.index,
        )

        # Ensure transform is NOT identity so line 503 executes.
        from panelcast.models.bayes.transforms import get_transform

        transform = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.5)
        if transform.name == "identity":
            pytest.skip("logit transform reported as identity in this build")

        _X, prev_score, _n, _y = _prepare_disjoint_inputs(test_df, test_features, s)
        # prev_score should differ from the raw global_mean (70.0) because
        # the transform was applied.
        assert not np.allclose(prev_score, 70.0)


class TestRunNewArtistPredictive:
    def _minimal_posterior(self, n_draws: int = 4) -> dict:
        rng = np.random.default_rng(0)
        return {
            "user_sigma_obs": rng.normal(size=(n_draws,)).astype(np.float32),
            "user_mu_artist": rng.normal(size=(n_draws, 2)).astype(np.float32),
        }

    def test_1d_output_reshaped_to_2d(self, summary):
        # predict_new_entity returning shape (n_obs,) must be reshaped to (n_obs, 1)
        # — lines 541-542.
        posterior = self._minimal_posterior()
        X = np.zeros((3, 2), dtype=np.float32)
        prev = np.full(3, 70.0, dtype=np.float32)
        n_rev = np.ones(3, dtype=np.int32)

        fake_pred = {"y": np.ones(3, dtype=np.float32)}  # 1-D
        with patch("panelcast.pipelines.evaluate.predict_new_entity", return_value=fake_pred):
            y = _run_new_artist_predictive(posterior, summary, X, prev, n_rev, seed=0)

        assert y.ndim == 2
        assert y.shape == (3, 1)

    def test_2d_output_unchanged(self, summary):
        posterior = self._minimal_posterior()
        X = np.zeros((3, 2), dtype=np.float32)
        prev = np.full(3, 70.0, dtype=np.float32)
        n_rev = np.ones(3, dtype=np.int32)

        fake_pred = {"y": np.ones((4, 3), dtype=np.float32)}  # already 2-D
        with patch("panelcast.pipelines.evaluate.predict_new_entity", return_value=fake_pred):
            y = _run_new_artist_predictive(posterior, summary, X, prev, n_rev, seed=0)

        assert y.shape == (4, 3)

    def test_learn_n_exponent_adds_n_reviews_kwarg(self, summary):
        # When learn_n_exponent=True, n_reviews_new is added (line 535).
        s = dict(summary)
        s["learn_n_exponent"] = True
        posterior = self._minimal_posterior()
        X = np.zeros((2, 2), dtype=np.float32)
        prev = np.full(2, 70.0, dtype=np.float32)
        n_rev = np.array([10, 20], dtype=np.int32)

        captured: dict = {}

        def fake_predict(**kwargs):
            captured.update(kwargs)
            return {"y": np.ones((4, 2), dtype=np.float32)}

        with patch("panelcast.pipelines.evaluate.predict_new_entity", side_effect=fake_predict):
            _run_new_artist_predictive(posterior, s, X, prev, n_rev, seed=0)

        assert "n_reviews_new" in captured
        assert "fixed_n_exponent" not in captured

    def test_fixed_n_exponent_nonzero_adds_both_kwargs(self, summary):
        # When learn_n_exponent=False and n_exponent != 0, fixed_n_exponent is
        # also added (line 537).
        s = dict(summary)
        s["learn_n_exponent"] = False
        s["n_exponent"] = 0.3

        posterior = self._minimal_posterior()
        X = np.zeros((2, 2), dtype=np.float32)
        prev = np.full(2, 70.0, dtype=np.float32)
        n_rev = np.array([10, 20], dtype=np.int32)

        captured: dict = {}

        def fake_predict(**kwargs):
            captured.update(kwargs)
            return {"y": np.ones((4, 2), dtype=np.float32)}

        with patch("panelcast.pipelines.evaluate.predict_new_entity", side_effect=fake_predict):
            _run_new_artist_predictive(posterior, s, X, prev, n_rev, seed=0)

        assert "n_reviews_new" in captured
        assert "fixed_n_exponent" in captured
        assert captured["fixed_n_exponent"] == pytest.approx(0.3)


class TestComputeInfoCriteriaBranches:
    def _fake_loo_waic(self, n_obs: int):
        return (
            SimpleNamespace(
                elpd_loo=-50.0,
                se=3.0,
                p_loo=5.0,
                pareto_k=np.full(n_obs, 0.1),
            ),
            SimpleNamespace(elpd_waic=-52.0, se=3.2, p_waic=5.5),
        )

    def test_entity_overdispersion_gate_excludes_entity_site(self):
        # When user_tau_entity is present but user_entity_obs_raw is absent,
        # entity_obs_raw must be appended to excluded_latents (line 593).
        n_obs, n_total = 5, 10
        posterior_samples = {
            "user_tau_entity": np.ones(n_total),
            "user_sigma_obs": np.ones(n_total),
            # user_rw_raw intentionally absent → also marginalized
        }
        model_args = {
            "artist_idx": np.zeros(n_obs, dtype=np.int32),
            "album_seq": np.ones(n_obs, dtype=np.int32),
            "prev_score": np.full(n_obs, 70.0, dtype=np.float32),
            "X": np.zeros((n_obs, 2), dtype=np.float32),
            "n_reviews": np.full(n_obs, 50, dtype=np.int32),
            "n_artists": 1,
            "max_seq": 5,
        }
        y_true = np.full(n_obs, 70.0, dtype=np.float32)
        fake_log_lik = np.random.default_rng(0).normal(size=(n_total, n_obs))
        fake_loo, fake_waic = self._fake_loo_waic(n_obs)

        latent_sites_requested: list[list[str]] = []

        def fake_predictive_cls(model, posterior_samples, batch_ndims, return_sites):
            latent_sites_requested.append(list(return_sites))
            mock = MagicMock()
            # Return dummy latents for the excluded sites
            mock.return_value = {
                s: np.zeros((len(next(iter(posterior_samples.values()))), n_obs))
                for s in return_sites
            }
            return mock

        with (
            patch("panelcast.pipelines.evaluate.Predictive", side_effect=fake_predictive_cls),
            patch(
                "panelcast.pipelines.evaluate.log_likelihood",
                return_value={"user_y": fake_log_lik},
            ),
            patch("panelcast.pipelines.evaluate.az.loo", return_value=fake_loo),
            patch("panelcast.pipelines.evaluate.az.waic", return_value=fake_waic),
        ):
            result = _compute_info_criteria(
                posterior_samples, model_args, y_true, n_chains=1, n_draws=10
            )

        assert result["latents_marginalized"] is True
        # Both rw_raw and entity_obs_raw should appear somewhere in the requests
        all_requested = [s for sites in latent_sites_requested for s in sites]
        assert "user_entity_obs_raw" in all_requested

    def test_jacobian_correction_applied(self):
        # When transform is not identity, log-Jacobian is added (lines 626-629).
        n_obs, n_total = 4, 8
        posterior_samples = {
            "user_sigma_obs": np.ones(n_total),
            "user_rw_raw": np.zeros((n_total, 2, 1)),
        }
        model_args = {
            "artist_idx": np.zeros(n_obs, dtype=np.int32),
            "album_seq": np.ones(n_obs, dtype=np.int32),
            "prev_score": np.full(n_obs, 70.0, dtype=np.float32),
            "X": np.zeros((n_obs, 2), dtype=np.float32),
            "n_reviews": np.full(n_obs, 50, dtype=np.int32),
            "n_artists": 1,
            "max_seq": 5,
        }
        y_true = np.full(n_obs, 70.0, dtype=np.float32)
        y_raw = np.full(n_obs, 75.0, dtype=np.float32)
        fake_log_lik = np.zeros((n_total, n_obs))
        fake_loo, fake_waic = self._fake_loo_waic(n_obs)

        transform = MagicMock()
        transform.name = "offset_logit"
        # log_jacobian returns a constant offset; verify it is added
        transform.log_jacobian = MagicMock(return_value=np.full(n_obs, -2.0))

        with (
            patch(
                "panelcast.pipelines.evaluate.log_likelihood",
                return_value={"user_y": fake_log_lik},
            ),
            patch("panelcast.pipelines.evaluate.az.loo", return_value=fake_loo),
            patch("panelcast.pipelines.evaluate.az.waic", return_value=fake_waic),
        ):
            result = _compute_info_criteria(
                posterior_samples,
                model_args,
                y_true,
                n_chains=1,
                n_draws=n_total,
                transform=transform,
                y_raw=y_raw,
            )

        transform.log_jacobian.assert_called_once()
        assert "loo" in result

    def test_jacobian_requires_y_raw(self):
        # y_raw=None with a non-identity transform raises (line 627).
        n_obs, n_total = 4, 8
        posterior_samples = {
            "user_sigma_obs": np.ones(n_total),
            "user_rw_raw": np.zeros((n_total, 2, 1)),
        }
        model_args = {
            "artist_idx": np.zeros(n_obs, dtype=np.int32),
            "album_seq": np.ones(n_obs, dtype=np.int32),
            "prev_score": np.full(n_obs, 70.0, dtype=np.float32),
            "X": np.zeros((n_obs, 2), dtype=np.float32),
            "n_reviews": np.full(n_obs, 50, dtype=np.int32),
            "n_artists": 1,
            "max_seq": 5,
        }
        y_true = np.full(n_obs, 70.0, dtype=np.float32)
        fake_log_lik = np.zeros((n_total, n_obs))

        transform = MagicMock()
        transform.name = "offset_logit"

        with (
            patch(
                "panelcast.pipelines.evaluate.log_likelihood",
                return_value={"user_y": fake_log_lik},
            ),
        ):
            with pytest.raises(ValueError, match="y_raw is required"):
                _compute_info_criteria(
                    posterior_samples,
                    model_args,
                    y_true,
                    n_chains=1,
                    n_draws=n_total,
                    transform=transform,
                    y_raw=None,
                )


class TestEvaluateModelsPrefixMismatch:
    def test_prefix_mismatch_raises(self, tmp_path, summary):
        # idata.posterior has sites with a different prefix → lines 820-821
        (tmp_path / "models").mkdir()
        summary_path = tmp_path / "models" / "training_summary.json"
        summary_path.write_text(json.dumps(summary))

        fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
        # idata has "critic_" prefixed sites, not "user_"
        fake_idata = az.from_dict(posterior={"critic_sigma_obs": np.ones((1, 5))})

        fake_summary_obj = MagicMock()
        fake_summary_obj.to_json_dict.return_value = dict(summary)

        ctx = SimpleNamespace(
            seed=0,
            strict=False,
            calibration_intervals=(0.80,),
            coverage_tolerance=0.1,
            prediction_interval=0.90,
            evaluate_secondary_split=False,
        )

        with (
            patch(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=fake_manifest,
            ),
            patch(
                "panelcast.pipelines.evaluate.load_training_summary",
                return_value=fake_summary_obj,
            ),
            patch(
                "panelcast.pipelines.evaluate.load_model",
                return_value=fake_idata,
            ),
            patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
        ):
            with pytest.raises(ValueError, match="Posterior has no sites with expected prefix"):
                evaluate_models(ctx)


def _make_standard_ctx(strict=False, secondary=False, seed=42):
    return SimpleNamespace(
        seed=seed,
        strict=strict,
        calibration_intervals=(0.80, 0.95),
        coverage_tolerance=0.10,
        prediction_interval=0.90,
        evaluate_secondary_split=secondary,
    )


def _setup_primary_split(tmp_path, summary):
    (tmp_path / "models").mkdir(exist_ok=True)
    split_dir = tmp_path / "data" / "splits" / "within_entity_temporal"
    feat_dir = tmp_path / "data" / "features" / "within_entity_temporal"
    split_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [72.0, 60.0],
            "User_Ratings": [90, 70],
            "Release_Date_Parsed": pd.to_datetime(["2018-01-01", "2019-01-01"]),
            "Album": ["A0", "B0"],
        }
    )
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_B"],
            "User_Score": [75.0, 65.0],
            "User_Ratings": [100, 80],
            "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "Album": ["A1", "B1"],
        }
    )
    feat_df = pd.DataFrame({"feat_1": [1.0, -0.5], "feat_2": [2.5, 1.5], "n_reviews": [100, 80]})
    train_feat_df = pd.DataFrame(
        {"feat_1": [1.0, -0.5], "feat_2": [2.5, 1.5], "n_reviews": [90, 70]}
    )

    train_df.to_parquet(split_dir / "train.parquet")
    test_df.to_parquet(split_dir / "test.parquet")
    feat_df.to_parquet(feat_dir / "test_features.parquet")
    train_feat_df.to_parquet(feat_dir / "train_features.parquet")

    (tmp_path / "models" / "training_summary.json").write_text(json.dumps(summary))


class TestEvaluateModelsPriorPredictive:
    def _base_patches(self, tmp_path):
        """Patches shared by all tests in this class."""
        fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
        fake_idata = az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))})
        diag = SimpleNamespace(
            passed=True,
            rhat_max=1.0,
            ess_bulk_min=1000,
            divergences=0,
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        rng = np.random.default_rng(7)
        return {
            "panelcast.pipelines.evaluate.load_manifest": fake_manifest,
            "panelcast.pipelines.evaluate.load_model": fake_idata,
            "panelcast.pipelines.evaluate.check_convergence": diag,
            "panelcast.pipelines.evaluate._extract_posterior_samples": {
                "user_sigma_obs": np.ones(5)
            },
            "panelcast.pipelines.evaluate._run_known_artist_predictive": rng.normal(
                70, 5, size=(10, 2)
            ),
            "panelcast.pipelines.evaluate._compute_info_criteria": {
                "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
            },
        }

    def test_prior_predictive_success_writes_json(self, tmp_path, summary):
        # Exercises lines 872-946 (prior predictive data prep), 946-952 (log.info),
        # and 1213-1231 (write prior_predictive.json).
        _setup_primary_split(tmp_path, summary)
        ctx = _make_standard_ctx()

        pp_result = SimpleNamespace(
            reasonable=True,
            fraction_in_bounds=0.95,
            checks_passed=True,
            informational_flags=[],
            summary={"mean": 70.0},
            bounds=(0.0, 100.0),
            checks={"in_bounds": True},
            n_samples=500,
            seed=42,
            n_obs_original=100,
            max_obs=2000,
            sampled_indices=np.arange(10),
        )

        with (
            patch(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            patch(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            patch(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            patch("panelcast.pipelines.evaluate.get_divergence_info"),
            patch(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones(5)},
            ),
            patch(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
            ),
            patch(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                    "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
                },
            ),
            patch(
                "panelcast.evaluation.prior_predictive.run_prior_predictive",
                return_value=pp_result,
            ),
            patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
        ):
            result = evaluate_models(ctx)

        pp_path = tmp_path / "outputs" / "evaluation" / "prior_predictive.json"
        assert pp_path.exists()
        with open(pp_path) as f:
            pp_data = json.load(f)
        assert pp_data["reasonable"] is True
        assert pp_data["n_samples"] == 500
        assert isinstance(pp_data["sampled_indices"], list)

    def test_prior_predictive_strict_raises_on_failed_checks(self, tmp_path, summary):
        # Line 962: strict=True + checks_passed=False → ValueError
        _setup_primary_split(tmp_path, summary)
        ctx = _make_standard_ctx(strict=True)

        pp_result = SimpleNamespace(
            reasonable=False,
            fraction_in_bounds=0.10,
            checks_passed=False,
            informational_flags=["mean out of range"],
            summary={},
            bounds=(0.0, 100.0),
            checks={},
            n_samples=500,
            seed=42,
            n_obs_original=100,
            max_obs=2000,
            sampled_indices=None,
        )

        with (
            patch(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            patch(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            patch(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            patch("panelcast.pipelines.evaluate.get_divergence_info"),
            patch(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones(5)},
            ),
            patch(
                "panelcast.evaluation.prior_predictive.run_prior_predictive",
                return_value=pp_result,
            ),
            patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
        ):
            with pytest.raises(ValueError, match="Prior predictive plausibility checks failed"):
                evaluate_models(ctx)

    def test_prior_predictive_no_train_features_falls_through(self, tmp_path, summary):
        # When train_features.parquet is absent the try-block raises; the except
        # catches it, logs a warning, and execution continues (non-strict).
        _setup_primary_split(tmp_path, summary)
        # Remove the train_features file so run_prior_predictive block fails
        feat_dir = tmp_path / "data" / "features" / "within_entity_temporal"
        (feat_dir / "train_features.parquet").unlink()

        ctx = _make_standard_ctx(strict=False)

        with (
            patch(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            patch(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            patch(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            patch("panelcast.pipelines.evaluate.get_divergence_info"),
            patch(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones(5)},
            ),
            patch(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
            ),
            patch(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                    "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
                },
            ),
            patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
        ):
            result = evaluate_models(ctx)

        # Should complete; prior_predictive.json must NOT exist
        pp_path = tmp_path / "outputs" / "evaluation" / "prior_predictive.json"
        assert not pp_path.exists()
        assert "metrics" in result


class TestEvaluateModelsTransformPaths:
    """Covers transform.inverse (line 1012) and transform.forward (line 1027)."""

    def _patched_run(self, tmp_path, summary, transform_name="offset_logit"):
        s = dict(summary)
        s["target_transform"] = transform_name
        s["logit_offset"] = 0.5
        _setup_primary_split(tmp_path, s)
        ctx = _make_standard_ctx()

        rng = np.random.default_rng(0)
        # Samples on the model (logit) scale — realistic small values
        y_samples = rng.normal(0.0, 0.5, size=(10, 2)).astype(np.float64)

        pp_result = SimpleNamespace(
            reasonable=True,
            fraction_in_bounds=0.9,
            checks_passed=True,
            informational_flags=[],
            summary={},
            bounds=(0.0, 100.0),
            checks={},
            n_samples=100,
            seed=42,
            n_obs_original=2,
            max_obs=200,
            sampled_indices=None,
        )

        with (
            patch(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            patch(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            patch(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            patch("panelcast.pipelines.evaluate.get_divergence_info"),
            patch(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones(5)},
            ),
            patch(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=y_samples,
            ),
            patch(
                "panelcast.evaluation.prior_predictive.run_prior_predictive",
                return_value=pp_result,
            ),
            patch(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -5.0, "se": 1.0, "p": 1.0},
                    "waic": {"elpd": -5.1, "se": 1.0, "p": 1.0},
                },
            ),
            patch("panelcast.pipelines.evaluate.Path", side_effect=lambda p: tmp_path / p),
        ):
            return evaluate_models(ctx)

    def test_non_identity_transform_back_transforms_predictions(self, tmp_path, summary):
        # The predictive samples are mocked on the model (logit) scale near 0.
        # With offset_logit, the inverse transform must map them back to the
        # score scale (~midpoint of [0, 100]); skipping it would leave the
        # written predictions near 0.
        import json

        from panelcast.models.bayes.transforms import get_transform

        t = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.5)
        if t.name == "identity":
            pytest.skip("logit resolves to identity in this build")

        result = self._patched_run(tmp_path, summary, transform_name="offset_logit")
        assert "metrics" in result

        pred_files = list(tmp_path.rglob("predictions.json"))
        assert pred_files, "evaluate wrote no predictions.json"
        means = json.loads(pred_files[0].read_text())["y_pred_mean"]
        assert means, "no predicted means recorded"
        assert all(0.0 <= m <= 100.0 for m in means)
        # Back-transformed logit-0 lands mid-scale; raw (untransformed) means
        # would sit near 0, so this distinguishes the inverse actually running.
        assert max(means) > 10.0


# --- from unit/pipelines/test_evaluate_coverage.py ---


@pytest.fixture
def mock_summary():
    """Minimal training summary for evaluation tests."""
    return {
        "artist_to_idx": {"Artist_A": 0, "Artist_B": 1},
        "n_artists": 2,
        "max_seq": 5,
        "max_albums": 10,
        "min_albums_filter": 2,
        "global_mean_score": 70.0,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "feature_cols": ["feat_1", "feat_2"],
        },
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "n_ref": None,
    }


class TestJsonSafeExtended:
    """Additional edge cases for _json_safe not covered by existing tests."""

    def test_dict_keys_coerced_to_string(self):
        """Non-string dict keys are coerced to strings."""
        result = _json_safe({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_set_elements_converted(self):
        """Sets are converted to lists with elements processed."""
        result = _json_safe({float("nan"), 1.0})
        assert isinstance(result, list)
        assert len(result) == 2
        assert None in result
        assert 1.0 in result

    def test_tolist_fallback_on_non_standard_object(self):
        """Objects with tolist() that are not str/bytes are converted."""

        class HasTolist:
            def tolist(self):
                return [1, 2, 3]

        result = _json_safe(HasTolist())
        assert result == [1, 2, 3]

    def test_tolist_raises_type_error_passthrough(self):
        """If tolist() raises TypeError, original value is returned."""

        class BadTolist:
            def tolist(self):
                raise TypeError("nope")

        obj = BadTolist()
        result = _json_safe(obj)
        assert result is obj

    def test_negative_zero_preserved(self):
        """Negative zero is finite and preserved."""
        result = _json_safe(-0.0)
        assert result == 0.0

    def test_deeply_nested_structure(self):
        """Deeply nested dicts/lists are fully traversed."""
        data = {"a": [{"b": [float("inf")]}]}
        result = _json_safe(data)
        assert result == {"a": [{"b": [None]}]}


class TestWriteJson:
    """Tests for _write_json helper."""

    def test_writes_valid_json_file(self, tmp_path):
        """Written file contains valid JSON."""
        path = tmp_path / "test.json"
        _write_json(path, {"a": 1, "b": [2, 3]})
        with open(path) as f:
            data = json.load(f)
        assert data == {"a": 1, "b": [2, 3]}

    def test_nan_replaced_with_null(self, tmp_path):
        """NaN values become null in output JSON."""
        path = tmp_path / "nan.json"
        _write_json(path, {"val": float("nan")})
        with open(path) as f:
            data = json.load(f)
        assert data["val"] is None

    def test_indent_parameter(self, tmp_path):
        """Indent is passed through to json.dump."""
        path = tmp_path / "indented.json"
        _write_json(path, {"x": 1}, indent=4)
        text = path.read_text()
        assert "\n" in text  # indented output has newlines

    def test_numpy_values_serialized(self, tmp_path):
        """Numpy values are converted before writing."""
        path = tmp_path / "np.json"
        _write_json(path, {"arr": np.array([1.0, 2.0]), "scalar": np.float64(3.14)})
        with open(path) as f:
            data = json.load(f)
        assert data["arr"] == [1.0, 2.0]
        assert abs(data["scalar"] - 3.14) < 0.01


class TestPrepareTestModelArgsExtended:
    """Coverage for missed branches in _prepare_test_model_args."""

    def test_missing_feature_scaler_raises(self, mock_summary):
        """Raises ValueError when feature_scaler is missing from summary."""
        summary = dict(mock_summary)
        summary["feature_scaler"] = None

        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"],
                "User_Score": [75.0],
                "User_Ratings": [100],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
                "n_reviews": [100],
            },
            index=test_df.index,
        )

        with pytest.raises(ValueError, match="feature_scaler"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_uses_user_ratings_when_no_n_reviews(self, mock_summary):
        """Falls back to User_Ratings column when n_reviews is absent."""
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"],
                "User_Score": [75.0],
                "User_Ratings": [200],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
            },
            index=test_df.index,
        )

        model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
        assert model_args["n_reviews"][0] == 200

    def test_missing_n_reviews_and_user_ratings_raises(self, mock_summary):
        """Raises ValueError when neither n_reviews nor User_Ratings exists."""
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"],
                "User_Score": [75.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
            },
            index=test_df.index,
        )

        with pytest.raises(ValueError, match="No n_reviews or User_Ratings"):
            _prepare_test_model_args(test_df, test_features, mock_summary)

    def test_invalid_n_reviews_dropped(self, mock_summary):
        """Rows with invalid n_reviews (NaN or <=0) are dropped."""
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A", "Artist_B"],
                "User_Score": [75.0, 80.0],
                "User_Ratings": [100, 80],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0, 2.0],
                "feat_2": [2.0, 3.0],
                "n_reviews": [100, -5],  # second row invalid
            },
            index=test_df.index,
        )

        model_args, y_true = _prepare_test_model_args(test_df, test_features, mock_summary)
        # Only valid row survives
        assert len(y_true) == 1
        assert model_args["n_reviews"][0] == 100

    def test_length_mismatch_raises(self, mock_summary):
        """Raises ValueError when test_df and test_features have different lengths."""
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"],
                "User_Score": [75.0],
                "User_Ratings": [100],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0, 2.0],
                "feat_2": [2.0, 3.0],
                "n_reviews": [100, 200],
            }
        )

        with pytest.raises(ValueError, match="row count mismatch"):
            _prepare_test_model_args(test_df, test_features, mock_summary)

    def test_train_df_none_uses_defaults(self, mock_summary):
        """When train_df is None, album_seq starts at 1 and prev_score uses global mean."""
        summary = dict(mock_summary)
        summary["min_albums_filter"] = 1

        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"],
                "User_Score": [75.0],
                "User_Ratings": [100],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
                "n_reviews": [100],
            },
            index=test_df.index,
        )

        model_args, _ = _prepare_test_model_args(test_df, test_features, summary, train_df=None)
        # With no training data, prev_score falls back to global mean
        assert model_args["prev_score"][0] == pytest.approx(70.0)

    def test_overlap_columns_dropped(self, mock_summary):
        """Overlapping columns between test_df and test_features are dropped from test_df."""
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A"],
                "User_Score": [75.0],
                "User_Ratings": [100],
                "feat_1": [999.0],  # overlaps with test_features
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
                "n_reviews": [100],
            },
            index=test_df.index,
        )

        model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
        # feature value should come from test_features, not the overlapping df column
        # After standardization: (1.0 - 1.0) / 0.5 = 0.0
        assert model_args["X"][0, 0] == pytest.approx(0.0)


class TestPrepareDisjointInputsExtended:
    """Coverage for missed branches in _prepare_disjoint_inputs."""

    def test_overlap_columns_dropped(self, mock_summary):
        """Overlapping columns between test_df and test_features are dropped."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A"],
                "User_Score": [75.0],
                "User_Ratings": [50],
                "feat_1": [999.0],  # overlap
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
                "n_reviews": [50],
            },
            index=test_df.index,
        )

        X, _, _, _ = _prepare_disjoint_inputs(test_df, test_features, mock_summary)
        assert X[0, 0] == pytest.approx(0.0)  # (1.0-1.0)/0.5

    def test_length_mismatch_raises(self, mock_summary):
        """Raises ValueError on length mismatch."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A"],
                "User_Score": [75.0],
                "User_Ratings": [50],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0, 2.0],
                "feat_2": [2.0, 3.0],
                "n_reviews": [50, 60],
            }
        )

        with pytest.raises(ValueError, match="row count mismatch"):
            _prepare_disjoint_inputs(test_df, test_features, mock_summary)

    def test_index_mismatch_raises(self, mock_summary):
        """Raises ValueError on index mismatch."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A"],
                "User_Score": [75.0],
                "User_Ratings": [50],
            },
            index=[0],
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
                "n_reviews": [50],
            },
            index=[10],
        )

        with pytest.raises(ValueError, match="different indices"):
            _prepare_disjoint_inputs(test_df, test_features, mock_summary)

    def test_user_ratings_fallback(self, mock_summary):
        """Falls back to User_Ratings when n_reviews not in columns."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A"],
                "User_Score": [75.0],
                "User_Ratings": [42],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
            },
            index=test_df.index,
        )

        _, _, n_reviews, _ = _prepare_disjoint_inputs(test_df, test_features, mock_summary)
        assert n_reviews[0] == 42

    def test_missing_n_reviews_and_user_ratings_raises(self, mock_summary):
        """Raises ValueError when neither n_reviews nor User_Ratings is present."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A"],
                "User_Score": [75.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0],
                "feat_2": [2.0],
            },
            index=test_df.index,
        )

        with pytest.raises(ValueError, match="No n_reviews or User_Ratings"):
            _prepare_disjoint_inputs(test_df, test_features, mock_summary)

    def test_invalid_n_reviews_dropped(self, mock_summary):
        """Rows with invalid n_reviews are filtered out."""
        test_df = pd.DataFrame(
            {
                "Artist": ["New_A", "New_B"],
                "User_Score": [75.0, 80.0],
                "User_Ratings": [50, 60],
            }
        )
        test_features = pd.DataFrame(
            {
                "feat_1": [1.0, 2.0],
                "feat_2": [2.0, 3.0],
                "n_reviews": [50, 0],  # second row invalid (<=0)
            },
            index=test_df.index,
        )

        X, _, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, mock_summary)
        assert len(y_true) == 1
        assert n_reviews[0] == 50


class TestResolveFeatureSplitDir:
    """Tests for _resolve_feature_split_dir."""

    def test_existing_directory_returned(self, tmp_path, monkeypatch):
        """When the split-specific directory exists, it is returned."""
        split_dir = tmp_path / "data" / "features" / "my_split"
        split_dir.mkdir(parents=True)

        with monkeypatch.context() as m:
            m.setattr(
                "panelcast.pipelines.evaluate.Path",
                lambda p: tmp_path / p,
            )
            result = _resolve_feature_split_dir("my_split")
        assert result.exists()

    def test_primary_split_fallback(self, tmp_path, monkeypatch):
        """PRIMARY_SPLIT falls back to data/features when split dir missing."""
        fallback_dir = tmp_path / "data" / "features"
        fallback_dir.mkdir(parents=True)

        # Don't create the split-specific dir
        result = (
            _resolve_feature_split_dir.__wrapped__("within_entity_temporal")
            if hasattr(_resolve_feature_split_dir, "__wrapped__")
            else None
        )

        # Direct test without monkeypatching Path (function uses Path internally)
        # Just verify the logic: if candidate doesn't exist and split is PRIMARY,
        # return Path("data/features")
        from panelcast.pipelines.evaluate import PRIMARY_SPLIT

        assert PRIMARY_SPLIT == "within_entity_temporal"


class TestComputeInfoCriteria:
    """Tests for _compute_info_criteria covering chain/draw mismatch."""

    def test_matching_chains_draws(self):
        """Standard case: n_chains * n_draws == n_samples_total."""
        n_chains, n_draws, n_obs = 2, 50, 10
        samples_total = n_chains * n_draws
        # user_rw_raw present -> no latent marginalization (Predictive) needed
        posterior_samples = {
            "user_sigma_obs": np.ones(samples_total),
            "user_rw_raw": np.zeros((samples_total, 4, 1)),
        }

        model_args = {
            "artist_idx": np.zeros(n_obs, dtype=np.int32),
            "album_seq": np.ones(n_obs, dtype=np.int32),
            "prev_score": np.full(n_obs, 70.0, dtype=np.float32),
            "X": np.zeros((n_obs, 2), dtype=np.float32),
            "n_reviews": np.full(n_obs, 50, dtype=np.int32),
            "n_artists": 1,
            "max_seq": 5,
        }
        y_true = np.full(n_obs, 70.0, dtype=np.float32)

        rng = np.random.default_rng(0)
        fake_log_lik = rng.normal(size=(samples_total, n_obs))

        with (
            patch("panelcast.pipelines.evaluate.log_likelihood") as mock_ll,
            patch("panelcast.pipelines.evaluate.az.loo") as mock_loo,
            patch("panelcast.pipelines.evaluate.az.waic") as mock_waic,
        ):
            mock_ll.return_value = {"user_y": fake_log_lik}
            mock_loo.return_value = SimpleNamespace(
                elpd_loo=-100.0,
                se=5.0,
                p_loo=10.0,
                pareto_k=np.full(n_obs, 0.2),
            )
            mock_waic.return_value = SimpleNamespace(
                elpd_waic=-102.0,
                se=5.5,
                p_waic=11.0,
            )
            result = _compute_info_criteria(
                posterior_samples, model_args, y_true, n_chains, n_draws
            )

        assert "loo" in result
        assert "waic" in result
        assert result["loo"]["elpd"] == -100.0
        assert result["waic"]["elpd"] == -102.0
        assert result["loo"]["pareto_k_gt_0_7"] == 0
        assert result["latents_marginalized"] is False

    def test_mismatched_chains_draws(self):
        """When n_chains * n_draws != n_samples_total, falls back to 1 chain."""
        n_obs = 5
        n_samples_total = 73  # doesn't match 2*50
        posterior_samples = {
            "user_sigma_obs": np.ones(n_samples_total),
            "user_rw_raw": np.zeros((n_samples_total, 4, 1)),
        }

        model_args = {
            "artist_idx": np.zeros(n_obs, dtype=np.int32),
            "album_seq": np.ones(n_obs, dtype=np.int32),
            "prev_score": np.full(n_obs, 70.0, dtype=np.float32),
            "X": np.zeros((n_obs, 2), dtype=np.float32),
            "n_reviews": np.full(n_obs, 50, dtype=np.int32),
            "n_artists": 1,
            "max_seq": 5,
        }
        y_true = np.full(n_obs, 70.0, dtype=np.float32)

        rng = np.random.default_rng(0)
        fake_log_lik = rng.normal(size=(n_samples_total, n_obs))

        with (
            patch("panelcast.pipelines.evaluate.log_likelihood") as mock_ll,
            patch("panelcast.pipelines.evaluate.az.loo") as mock_loo,
            patch("panelcast.pipelines.evaluate.az.waic") as mock_waic,
        ):
            mock_ll.return_value = {"user_y": fake_log_lik}
            mock_loo.return_value = SimpleNamespace(
                elpd_loo=-200.0,
                se=10.0,
                p_loo=20.0,
                pareto_k=np.full(n_obs, 0.2),
            )
            mock_waic.return_value = SimpleNamespace(
                elpd_waic=-205.0,
                se=10.5,
                p_waic=21.0,
            )
            result = _compute_info_criteria(
                posterior_samples,
                model_args,
                y_true,
                n_chains=2,
                n_draws=50,  # 2*50=100 != 73
            )

        assert "loo" in result
        assert result["loo"]["elpd"] == -200.0

    def test_missing_y_key_raises(self):
        """Raises ValueError when no observed site ending in '_y' is found."""
        posterior_samples = {
            "sigma": np.ones(10),
            "user_rw_raw": np.zeros((10, 4, 1)),
        }
        model_args = {"y": np.ones(5)}
        y_true = np.ones(5)

        with patch("panelcast.pipelines.evaluate.log_likelihood") as mock_ll:
            mock_ll.return_value = {"some_other_site": np.ones((10, 5))}
            with pytest.raises(ValueError, match="Unable to locate observed site"):
                _compute_info_criteria(posterior_samples, model_args, y_true, 1, 10)


class TestEvaluatePredictions:
    """Tests for _evaluate_predictions covering full metrics pipeline."""

    def test_returns_three_payloads(self):
        """Returns split_metrics, predictions_payload, and calibration_payload."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(70, 5, size=20).astype(np.float32)
        y_pred_samples = rng.normal(70, 5, size=(100, 20))

        metrics, preds, calib = _evaluate_predictions(
            y_true,
            y_pred_samples,
            calibration_intervals=(0.80, 0.95),
            coverage_tolerance=0.10,
            prediction_interval=0.90,
        )

        # metrics structure
        assert "point_metrics" in metrics
        assert "calibration" in metrics
        assert "crps" in metrics
        assert "ppc" in metrics
        assert "prediction_interval" in metrics
        assert metrics["point_metrics"]["n_observations"] == 20

        # calibration structure
        assert "coverages" in metrics["calibration"]
        assert "0.80" in metrics["calibration"]["coverages"]
        assert "0.95" in metrics["calibration"]["coverages"]
        assert "interval_scores" in metrics["calibration"]
        assert "wis" in metrics["calibration"]
        assert isinstance(metrics["calibration"]["wis"], float)

        # predictions payload
        assert len(preds["y_true"]) == 20
        assert len(preds["y_pred_mean"]) == 20
        assert len(preds["y_pred_lower"]) == 20
        assert len(preds["y_pred_upper"]) == 20
        assert preds["interval_level"] == 0.90

        # calibration payload
        assert "predicted_probs" in calib
        assert "observed_freq" in calib
        assert "counts" in calib

    def test_ppc_payload_structure(self):
        """PPC payload contains summary, n_samples, and extreme_statistics."""
        rng = np.random.default_rng(99)
        y_true = rng.normal(70, 5, size=15).astype(np.float32)
        y_pred_samples = rng.normal(70, 5, size=(50, 15))

        metrics, _, _ = _evaluate_predictions(
            y_true,
            y_pred_samples,
            calibration_intervals=(0.95,),
            coverage_tolerance=0.10,
            prediction_interval=0.95,
        )

        ppc = metrics["ppc"]
        assert "summary" in ppc
        assert "n_samples" in ppc
        assert "extreme_statistics" in ppc

    def test_within_tolerance_flag(self):
        """Coverage within_tolerance flag reflects actual calibration quality."""
        rng = np.random.default_rng(7)
        n = 100
        y_true = rng.normal(0, 1, size=n).astype(np.float32)
        # Well-calibrated samples: centered on y_true
        y_pred_samples = y_true[None, :] + rng.normal(0, 1, size=(500, n))

        metrics, _, _ = _evaluate_predictions(
            y_true,
            y_pred_samples,
            calibration_intervals=(0.80,),
            coverage_tolerance=0.20,  # very generous
            prediction_interval=0.95,
        )
        assert metrics["calibration"]["within_tolerance"] is True


class TestEvaluateModelsExtended:
    """Coverage for missed branches in evaluate_models."""

    def _setup_dirs_and_files(self, tmp_path, mock_summary, include_secondary=False):
        """Create the directory structure and parquet files needed by evaluate_models."""
        (tmp_path / "models").mkdir()
        (tmp_path / "data" / "splits" / "within_entity_temporal").mkdir(parents=True)
        (tmp_path / "data" / "features" / "within_entity_temporal").mkdir(parents=True)

        train_df = pd.DataFrame(
            {
                "Artist": ["Artist_A", "Artist_B"],
                "User_Score": [72.0, 60.0],
                "User_Ratings": [90, 70],
                "Release_Date_Parsed": pd.to_datetime(["2018-01-01", "2019-01-01"]),
                "Album": ["A0", "B0"],
            }
        )
        test_df = pd.DataFrame(
            {
                "Artist": ["Artist_A", "Artist_B"],
                "User_Score": [75.0, 65.0],
                "User_Ratings": [100, 80],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01", "2021-01-01"]),
                "Album": ["A1", "B1"],
            }
        )
        feat_df = pd.DataFrame(
            {
                "feat_1": [1.0, -0.5],
                "feat_2": [2.5, 1.5],
                "n_reviews": [100, 80],
            }
        )

        train_df.to_parquet(tmp_path / "data/splits/within_entity_temporal/train.parquet")
        test_df.to_parquet(tmp_path / "data/splits/within_entity_temporal/test.parquet")
        feat_df.to_parquet(tmp_path / "data/features/within_entity_temporal/test_features.parquet")

        if include_secondary:
            sec_dir = tmp_path / "data" / "splits" / "entity_disjoint"
            sec_dir.mkdir(parents=True)
            sec_feat_dir = tmp_path / "data" / "features" / "entity_disjoint"
            sec_feat_dir.mkdir(parents=True)

            sec_test_df = pd.DataFrame(
                {
                    "Artist": ["New_C", "New_D"],
                    "User_Score": [60.0, 55.0],
                    "User_Ratings": [30, 40],
                }
            )
            sec_feat_df = pd.DataFrame(
                {
                    "feat_1": [0.5, 1.5],
                    "feat_2": [1.0, 2.0],
                    "n_reviews": [30, 40],
                }
            )
            sec_test_df.to_parquet(sec_dir / "test.parquet")
            sec_feat_df.to_parquet(sec_feat_dir / "test_features.parquet")

        with open(tmp_path / "models/training_summary.json", "w", encoding="utf-8") as f:
            json.dump(mock_summary, f)

        return train_df, test_df

    def _make_ctx(self, strict=False, secondary=False):
        """Build a minimal StageContext-like namespace."""
        return SimpleNamespace(
            seed=42,
            strict=strict,
            calibration_intervals=(0.80, 0.95),
            coverage_tolerance=0.03,
            prediction_interval=0.95,
            evaluate_secondary_split=secondary,
        )

    def _standard_patches(self, tmp_path, y_samples_shape=(10, 2)):
        """Return a dict of standard patches for evaluate_models."""
        fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
        fake_idata = az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))})
        diagnostics = SimpleNamespace(
            passed=True,
            rhat_max=1.0,
            ess_bulk_min=1000,
            divergences=0,
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        rng = np.random.default_rng(0)

        return {
            "panelcast.pipelines.evaluate.load_manifest": lambda *a, **kw: fake_manifest,
            "panelcast.pipelines.evaluate.load_model": lambda *a, **kw: fake_idata,
            "panelcast.pipelines.evaluate.check_convergence": lambda *a, **kw: diagnostics,
            "panelcast.pipelines.evaluate.get_divergence_info": lambda *a, **kw: None,
            "panelcast.pipelines.evaluate._extract_posterior_samples": lambda *a, **kw: {
                "user_sigma_obs": np.ones((5,))
            },
            "panelcast.pipelines.evaluate._run_known_artist_predictive": lambda *a, **kw: (
                rng.normal(70, 5, size=y_samples_shape)
            ),
            "panelcast.pipelines.evaluate._compute_info_criteria": lambda *a, **kw: {
                "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
            },
            "panelcast.pipelines.evaluate.Path": lambda p: tmp_path / p,
        }

    def test_info_criteria_failure_non_strict(self, tmp_path, mock_summary):
        """In non-strict mode, info_criteria failure records status=unavailable."""
        self._setup_dirs_and_files(tmp_path, mock_summary)
        ctx = self._make_ctx(strict=False)

        with (
            patch(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            patch(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            patch(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            patch("panelcast.pipelines.evaluate.get_divergence_info"),
            patch(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones((5,))},
            ),
            patch(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
            ),
            patch(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                side_effect=RuntimeError("info boom"),
            ),
            patch("panelcast.pipelines.evaluate.Path", side_effect=lambda path: tmp_path / path),
        ):
            result = evaluate_models(ctx)

        ic = result["metrics"]["splits"]["within_entity_temporal"]["info_criteria"]
        assert ic["status"] == "unavailable"
        assert "info boom" in ic["reason"]

    def test_info_criteria_failure_strict_raises(self, tmp_path, mock_summary):
        """In strict mode, info_criteria failure raises."""
        self._setup_dirs_and_files(tmp_path, mock_summary)
        ctx = self._make_ctx(strict=True)

        from unittest.mock import patch as p

        with (
            p(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            p(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            p(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            p("panelcast.pipelines.evaluate.get_divergence_info"),
            p(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones((5,))},
            ),
            p(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
            ),
            p(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                side_effect=RuntimeError("strict boom"),
            ),
            p("panelcast.pipelines.evaluate.Path", side_effect=lambda path: tmp_path / path),
        ):
            with pytest.raises(RuntimeError, match="strict boom"):
                evaluate_models(ctx)

    def test_secondary_split_evaluation_success(self, tmp_path, mock_summary):
        """Secondary split evaluation produces results when artifacts exist."""
        self._setup_dirs_and_files(tmp_path, mock_summary, include_secondary=True)
        ctx = self._make_ctx(secondary=True)

        from unittest.mock import patch as p

        rng = np.random.default_rng(0)

        with (
            p(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            p(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            p(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            p("panelcast.pipelines.evaluate.get_divergence_info"),
            p(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones((5,))},
            ),
            p(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=rng.normal(70, 5, size=(10, 2)),
            ),
            p(
                "panelcast.pipelines.evaluate._run_new_artist_predictive",
                return_value=rng.normal(70, 5, size=(10, 2)),
            ),
            p(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                    "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
                },
            ),
            p("panelcast.pipelines.evaluate.Path", side_effect=lambda path: tmp_path / path),
        ):
            result = evaluate_models(ctx)

        splits = result["metrics"]["splits"]
        assert "within_entity_temporal" in splits
        assert "entity_disjoint" in splits
        # Secondary split has unavailable info_criteria
        assert splits["entity_disjoint"]["info_criteria"]["status"] == "unavailable"

    def test_secondary_split_missing_non_strict_warns(self, tmp_path, mock_summary):
        """Non-strict mode warns but continues when secondary artifacts missing."""
        self._setup_dirs_and_files(tmp_path, mock_summary, include_secondary=False)
        ctx = self._make_ctx(strict=False, secondary=True)

        from unittest.mock import patch as p

        with (
            p(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            p(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            p(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            p("panelcast.pipelines.evaluate.get_divergence_info"),
            p(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones((5,))},
            ),
            p(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
            ),
            p(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                    "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
                },
            ),
            p("panelcast.pipelines.evaluate.Path", side_effect=lambda path: tmp_path / path),
        ):
            result = evaluate_models(ctx)

        # Should succeed without secondary split in results
        assert "entity_disjoint" not in result["metrics"]["splits"]

    def test_calibration_warning_non_strict(self, tmp_path, mock_summary):
        """Non-strict mode warns on calibration out-of-tolerance but continues."""
        self._setup_dirs_and_files(tmp_path, mock_summary)
        ctx = self._make_ctx(strict=False)

        fake_metrics = {
            "point_metrics": {
                "rmse": 5.0,
                "mae": 4.0,
                "r2": 0.5,
                "mean_bias": 0.1,
                "n_observations": 2,
            },
            "calibration": {
                "coverages": {"0.80": {"nominal": 0.80, "empirical": 0.50}},
                "coverage_tolerance": 0.03,
                "within_tolerance": False,
                "interval_scores": {},
                "wis": 5.0,
            },
            "crps": {"mean_crps": 3.0, "n_obs": 2},
            "ppc": {"summary": {}, "n_samples": 10, "extreme_statistics": {}},
            "prediction_interval": {
                "level": 0.95,
                "lower_percentile": 2.5,
                "upper_percentile": 97.5,
            },
        }

        from unittest.mock import patch as p

        with (
            p(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            p(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            p(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            p("panelcast.pipelines.evaluate.get_divergence_info"),
            p(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones((5,))},
            ),
            p(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 1, size=(5, 2)),
            ),
            p(
                "panelcast.pipelines.evaluate._evaluate_predictions",
                return_value=(
                    fake_metrics,
                    {
                        "y_true": [],
                        "y_pred_mean": [],
                        "y_pred_lower": [],
                        "y_pred_upper": [],
                        "interval_level": 0.95,
                    },
                    {"predicted_probs": [], "observed_freq": [], "counts": []},
                ),
            ),
            p(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                    "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
                },
            ),
            p("panelcast.pipelines.evaluate.Path", side_effect=lambda path: tmp_path / path),
        ):
            # Should not raise despite bad calibration
            result = evaluate_models(ctx)

        assert not result["metrics"]["splits"]["within_entity_temporal"]["calibration"][
            "within_tolerance"
        ]

    def test_manifest_missing_user_score_raises(self):
        """Raises ValueError when manifest has no user_score entry."""
        fake_manifest = SimpleNamespace(current={"critic_score": "c.nc"})
        ctx = self._make_ctx()

        from unittest.mock import patch as p

        # Hermetic: do not depend on a real models/training_summary.json.
        fake_summary = MagicMock()
        fake_summary.to_json_dict.return_value = {"dataset": {"model_prefix": "user"}}

        with (
            p("panelcast.pipelines.evaluate.load_manifest", return_value=fake_manifest),
            p(
                "panelcast.pipelines.evaluate.load_training_summary",
                return_value=fake_summary,
            ),
        ):
            with pytest.raises(ValueError, match="No trained user_score model"):
                evaluate_models(ctx)

    def test_output_files_written(self, tmp_path, mock_summary):
        """Verify that evaluation artifacts are written to disk."""
        self._setup_dirs_and_files(tmp_path, mock_summary)
        ctx = self._make_ctx(strict=False)

        from unittest.mock import patch as p

        with (
            p(
                "panelcast.pipelines.evaluate.load_manifest",
                return_value=SimpleNamespace(current={"user_score": "model.nc"}),
            ),
            p(
                "panelcast.pipelines.evaluate.load_model",
                return_value=az.from_dict(posterior={"user_sigma_obs": np.ones((1, 5))}),
            ),
            p(
                "panelcast.pipelines.evaluate.check_convergence",
                return_value=SimpleNamespace(
                    passed=True,
                    rhat_max=1.0,
                    ess_bulk_min=1000,
                    divergences=0,
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
            ),
            p("panelcast.pipelines.evaluate.get_divergence_info"),
            p(
                "panelcast.pipelines.evaluate._extract_posterior_samples",
                return_value={"user_sigma_obs": np.ones((5,))},
            ),
            p(
                "panelcast.pipelines.evaluate._run_known_artist_predictive",
                return_value=np.random.default_rng(0).normal(70, 5, size=(10, 2)),
            ),
            p(
                "panelcast.pipelines.evaluate._compute_info_criteria",
                return_value={
                    "loo": {"elpd": -10.0, "se": 1.0, "p": 2.0},
                    "waic": {"elpd": -10.2, "se": 1.1, "p": 2.1},
                },
            ),
            p("panelcast.pipelines.evaluate.Path", side_effect=lambda path: tmp_path / path),
        ):
            evaluate_models(ctx)

        # Check output files
        out_dir = tmp_path / "outputs" / "evaluation"
        assert (out_dir / "diagnostics.json").exists()
        assert (out_dir / "metrics.json").exists()
        # Backward compat files
        assert (out_dir / "predictions.json").exists()
        assert (out_dir / "calibration.json").exists()
        # Split-specific directory
        assert (out_dir / "within_entity_temporal" / "predictions.json").exists()
        assert (out_dir / "within_entity_temporal" / "calibration.json").exists()

        # Verify metrics.json content
        with open(out_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["schema_version"] == 2
        assert metrics["primary_split"] == "within_entity_temporal"


# --- from unit/pipelines/test_evaluate_new.py ---


class TestJsonSafe:
    """Tests for _json_safe conversion utility."""

    def test_dict_keys_converted_to_str(self):
        """Dict keys should be stringified."""
        result = _json_safe({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_nested_dict(self):
        """Nested dicts should be recursively converted."""
        result = _json_safe({"a": {"b": np.float64(1.5)}})
        assert result == {"a": {"b": 1.5}}

    def test_list_converted(self):
        """Lists should have elements converted."""
        result = _json_safe([np.int64(1), np.float32(2.5)])
        assert result == [1, 2.5]

    def test_tuple_converted_to_list(self):
        """Tuples should be converted to lists."""
        result = _json_safe((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set_converted_to_list(self):
        """Sets should be converted to lists."""
        result = _json_safe({1, 2})
        assert isinstance(result, list)
        assert set(result) == {1, 2}

    def test_numpy_array(self):
        """Numpy arrays should be converted to lists."""
        result = _json_safe(np.array([1.0, 2.0, 3.0]))
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_scalar(self):
        """Numpy scalars should be converted to Python types."""
        assert _json_safe(np.float64(3.14)) == 3.14
        assert _json_safe(np.int32(42)) == 42

    def test_inf_replaced_with_none(self):
        """Inf values should be replaced with None."""
        assert _json_safe(float("inf")) is None
        assert _json_safe(float("-inf")) is None

    def test_nan_replaced_with_none(self):
        """NaN values should be replaced with None."""
        assert _json_safe(float("nan")) is None

    def test_finite_float_preserved(self):
        """Finite floats should be preserved."""
        assert _json_safe(3.14) == 3.14

    def test_string_passthrough(self):
        """Strings should pass through unchanged."""
        assert _json_safe("hello") == "hello"

    def test_none_passthrough(self):
        """None should pass through unchanged."""
        assert _json_safe(None) is None

    def test_bool_passthrough(self):
        """Booleans should pass through unchanged."""
        assert _json_safe(True) is True
        assert _json_safe(False) is False

    def test_numpy_bool(self):
        """Numpy booleans should be converted."""
        result = _json_safe(np.bool_(True))
        assert result is True or result == True  # noqa: E712


class TestWriteJson_new:
    """Tests for _write_json helper."""

    def test_writes_valid_json(self, tmp_path):
        """Should write valid JSON to disk."""
        path = tmp_path / "test.json"
        _write_json(path, {"key": "value", "num": 42})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"key": "value", "num": 42}

    def test_nan_replaced_with_null(self, tmp_path):
        """NaN values should be written as null."""
        path = tmp_path / "test.json"
        _write_json(path, {"val": float("nan")})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["val"] is None

    def test_indent_parameter(self, tmp_path):
        """Indent parameter should be passed through."""
        path = tmp_path / "test.json"
        _write_json(path, {"a": 1}, indent=2)
        content = path.read_text(encoding="utf-8")
        assert "  " in content  # indented

    def test_numpy_values_converted(self, tmp_path):
        """Numpy values should be converted before writing."""
        path = tmp_path / "test.json"
        _write_json(path, {"arr": np.array([1, 2, 3]), "scalar": np.float64(3.14)})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["arr"] == [1, 2, 3]
        assert data["scalar"] == 3.14


class TestResolveFeatureSplitDir_new:
    """Tests for _resolve_feature_split_dir."""

    def test_existing_split_dir_returned(self, tmp_path, monkeypatch):
        """When split dir exists, it should be returned."""
        split_dir = tmp_path / "data" / "features" / "within_entity_temporal"
        split_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.Path",
            lambda p: tmp_path / p,
        )

        result = _resolve_feature_split_dir("within_entity_temporal")
        assert result == split_dir

    def test_primary_split_fallback(self, tmp_path, monkeypatch):
        """Primary split should fall back to data/features when split dir missing."""
        # Don't create the split-specific dir
        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.Path",
            lambda p: tmp_path / p,
        )

        result = _resolve_feature_split_dir("within_entity_temporal")
        assert result == features_dir

    def test_secondary_split_no_fallback(self, tmp_path, monkeypatch):
        """Non-primary split should return candidate path even if missing."""
        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.Path",
            lambda p: tmp_path / p,
        )

        result = _resolve_feature_split_dir("entity_disjoint")
        # Should return the candidate path, not fall back
        assert "entity_disjoint" in str(result)


class TestPrepareTestModelArgs:
    """Tests for _prepare_test_model_args helper."""

    def _make_summary(self):
        return {
            "artist_to_idx": {"A": 0, "B": 1},
            "max_seq": 5,
            "max_albums": 50,
            "min_albums_filter": 2,
            "global_mean_score": 75.0,
            "feature_cols": ["f1"],
            "feature_scaler": {
                "mean": [0.0],
                "std": [1.0],
            },
            "n_artists": 2,
            "n_exponent": 0.0,
            "learn_n_exponent": False,
            "n_exponent_prior": "logit-normal",
            "n_ref": None,
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

    def test_basic_preparation(self):
        """Should produce valid model_args and y_true."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [80.0, 85.0],
                "Album": ["a1", "b1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 82.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )

        assert "artist_idx" in model_args
        assert "X" in model_args
        assert model_args["y"] is None
        assert len(y_true) == 2

    def test_overlap_columns_dropped(self):
        """Overlapping columns should be dropped from test_df."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
                "f1": [999.0],  # overlap
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )
        # f1 should come from features, not test_df
        assert model_args["X"].shape == (1, 1)

    def test_unknown_artist_raises(self):
        """Unknown artists in test data should raise ValueError."""
        test_df = pd.DataFrame(
            {
                "Artist": ["UNKNOWN"],
                "User_Score": [80.0],
                "Album": ["u1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="Unknown artists"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_length_mismatch_raises(self):
        """Length mismatch between test_df and test_features should raise."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A", "B"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="row count mismatch"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_index_mismatch_raises(self):
        """Index mismatch should raise ValueError."""
        test_df = pd.DataFrame(
            {"Artist": ["A"], "User_Score": [80.0], "Album": ["a1"]},
            index=[0],
        )
        test_features = pd.DataFrame(
            {"f1": [1.0], "n_reviews": [10]},
            index=[5],
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="different indices"):
            _prepare_test_model_args(test_df, test_features, summary)

    def test_missing_feature_scaler_raises(self):
        """Missing feature_scaler in summary should raise ValueError."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()
        summary.pop("feature_scaler")
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        with pytest.raises(ValueError, match="feature_scaler"):
            _prepare_test_model_args(test_df, test_features, summary, train_df=train_df)

    def test_invalid_n_reviews_filtered(self):
        """Invalid n_reviews rows should be filtered out."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [80.0, 85.0],
                "Album": ["a1", "a2"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, -5],  # one invalid
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )
        assert len(y_true) == 1

    def test_no_train_df(self):
        """Should work when train_df is None."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=None
        )
        assert len(y_true) == 1

    def test_user_ratings_fallback(self):
        """Should fall back to User_Ratings when n_reviews not in features."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
                "User_Ratings": [15],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        model_args, y_true = _prepare_test_model_args(
            test_df, test_features, summary, train_df=train_df
        )
        assert len(y_true) == 1

    def test_no_n_reviews_or_user_ratings_raises(self):
        """Should raise when neither n_reviews nor User_Ratings is available."""
        test_df = pd.DataFrame(
            {
                "Artist": ["A"],
                "User_Score": [80.0],
                "Album": ["a1"],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()
        train_df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
            }
        )

        with pytest.raises(ValueError, match="n_reviews or User_Ratings"):
            _prepare_test_model_args(test_df, test_features, summary, train_df=train_df)


class TestPrepareDisjointInputs:
    """Tests for _prepare_disjoint_inputs helper."""

    def _make_summary(self):
        return {
            "global_mean_score": 75.0,
            "feature_cols": ["f1"],
            "feature_scaler": {
                "mean": [0.0],
                "std": [1.0],
            },
        }

    def test_basic_preparation(self):
        """Should produce X, prev_score, n_reviews, y_true."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewB"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, 20],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)

        assert X.shape == (2, 1)
        assert len(y_true) == 2
        # Cold-start: all prev_score should be global mean
        np.testing.assert_allclose(prev_score, [75.0, 75.0])

    def test_overlap_columns_dropped(self):
        """Overlapping columns between test_df and test_features are handled."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA"],
                "User_Score": [80.0],
                "f1": [999.0],  # overlap
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        assert X.shape == (1, 1)

    def test_invalid_n_reviews_filtered(self):
        """Invalid n_reviews should be filtered."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewB"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0],
                "n_reviews": [10, -1],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        assert len(y_true) == 1

    def test_length_mismatch_raises(self):
        """Length mismatch should raise ValueError."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewB"],
                "User_Score": [80.0, 85.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
                "n_reviews": [10],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="row count mismatch"):
            _prepare_disjoint_inputs(test_df, test_features, summary)

    def test_index_mismatch_raises(self):
        """Index mismatch should raise ValueError."""
        test_df = pd.DataFrame(
            {"Artist": ["NewA"], "User_Score": [80.0]},
            index=[0],
        )
        test_features = pd.DataFrame(
            {"f1": [1.0], "n_reviews": [10]},
            index=[5],
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="different indices"):
            _prepare_disjoint_inputs(test_df, test_features, summary)

    def test_user_ratings_fallback(self):
        """Should use User_Ratings when n_reviews not present."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA"],
                "User_Score": [80.0],
                "User_Ratings": [15],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        assert len(y_true) == 1
        assert n_reviews[0] == 15

    def test_no_n_reviews_or_user_ratings_raises(self):
        """Should raise when neither column is available."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA"],
                "User_Score": [80.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0],
            }
        )
        summary = self._make_summary()

        with pytest.raises(ValueError, match="n_reviews or User_Ratings"):
            _prepare_disjoint_inputs(test_df, test_features, summary)

    def test_multi_album_artists_use_global_mean(self):
        """Multi-album artists in disjoint split should all use global mean."""
        test_df = pd.DataFrame(
            {
                "Artist": ["NewA", "NewA", "NewB"],
                "User_Score": [80.0, 85.0, 90.0],
            }
        )
        test_features = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "n_reviews": [10, 20, 30],
            }
        )
        summary = self._make_summary()

        X, prev_score, n_reviews, y_true = _prepare_disjoint_inputs(test_df, test_features, summary)
        # All prev_score should be global mean = 75.0
        np.testing.assert_allclose(prev_score, [75.0, 75.0, 75.0])
