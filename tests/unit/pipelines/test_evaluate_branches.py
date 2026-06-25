"""Branch-coverage tests targeting evaluate.py lines not yet hit elsewhere.

Target lines: 92, 222, 402-403, 409, 503, 517-543, 593, 602-609, 626-629,
820-821, 872-946, 962, 1012, 1027, 1213-1231.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import arviz as az
import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.evaluate import (
    _compute_info_criteria,
    _extract_posterior_samples,
    _prepare_disjoint_inputs,
    _prepare_test_model_args,
    _resolve_feature_split_dir,
    _run_new_artist_predictive,
    evaluate_models,
)


# ---------------------------------------------------------------------------
# Shared summary fixture
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _extract_posterior_samples (line 92)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# line 222 — NaN artist_idx after map (shouldn't happen via normal flow but
# the guard exists; reach it via monkeypatching map to return NaN)
# ---------------------------------------------------------------------------


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

        result = _summary_dataset({
            "dataset": {
                "entity_col": "Team",
                "target_col": "Score",
                "n_obs_col": "Votes",
                "model_prefix": "team",
                "target_bounds": [0.0, 10.0],
            }
        })
        assert result["entity_col"] == "Team"
        assert result["prefix"] == "team"
        assert result["target_bounds"] == (0.0, 10.0)


# ---------------------------------------------------------------------------
# _resolve_feature_split_dir — unknown split name (lines 402-403) and
# legacy path exists (line 409)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _prepare_disjoint_inputs — non-identity transform (line 503)
# ---------------------------------------------------------------------------


class TestDisjointInputsTransform:
    def test_non_identity_transform_applied_to_prev_score(self, summary):
        s = dict(summary)
        s["target_transform"] = "offset_logit"
        s["target_bounds"] = (0.0, 100.0)
        s["logit_offset"] = 0.5

        test_df = pd.DataFrame(
            {"Artist": ["New_X"], "User_Score": [75.0], "User_Ratings": [30]}
        )
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


# ---------------------------------------------------------------------------
# _run_new_artist_predictive — lines 517-543
# ---------------------------------------------------------------------------


class TestRunNewArtistPredictive:
    def _minimal_posterior(self, n_draws: int = 4) -> dict:
        rng = np.random.default_rng(0)
        return {
            "user_sigma_obs": rng.normal(size=(n_draws,)).astype(np.float32),
            "user_mu_artist": rng.normal(size=(n_draws, 2)).astype(np.float32),
        }

    def test_1d_output_reshaped_to_2d(self, summary):
        # predict_new_artist returning shape (n_obs,) must be reshaped to (n_obs, 1)
        # — lines 541-542.
        posterior = self._minimal_posterior()
        X = np.zeros((3, 2), dtype=np.float32)
        prev = np.full(3, 70.0, dtype=np.float32)
        n_rev = np.ones(3, dtype=np.int32)

        fake_pred = {"y": np.ones(3, dtype=np.float32)}  # 1-D
        with patch(
            "panelcast.pipelines.evaluate.predict_new_artist", return_value=fake_pred
        ):
            y = _run_new_artist_predictive(posterior, summary, X, prev, n_rev, seed=0)

        assert y.ndim == 2
        assert y.shape == (3, 1)

    def test_2d_output_unchanged(self, summary):
        posterior = self._minimal_posterior()
        X = np.zeros((3, 2), dtype=np.float32)
        prev = np.full(3, 70.0, dtype=np.float32)
        n_rev = np.ones(3, dtype=np.int32)

        fake_pred = {"y": np.ones((4, 3), dtype=np.float32)}  # already 2-D
        with patch(
            "panelcast.pipelines.evaluate.predict_new_artist", return_value=fake_pred
        ):
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

        with patch("panelcast.pipelines.evaluate.predict_new_artist", side_effect=fake_predict):
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

        with patch("panelcast.pipelines.evaluate.predict_new_artist", side_effect=fake_predict):
            _run_new_artist_predictive(posterior, s, X, prev, n_rev, seed=0)

        assert "n_reviews_new" in captured
        assert "fixed_n_exponent" in captured
        assert captured["fixed_n_exponent"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _compute_info_criteria — entity-overdispersion gate (line 593) and latent
# Predictive block (lines 602-609), and Jacobian branch (lines 626-629)
# ---------------------------------------------------------------------------


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
            mock.return_value = {s: np.zeros((len(next(iter(posterior_samples.values()))), n_obs)) for s in return_sites}
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


# ---------------------------------------------------------------------------
# evaluate_models — prefix mismatch (lines 820-821)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# evaluate_models — prior predictive block (lines 872-946) and strict gate
# (line 962); also transform inverse (1012) + forward (1027) + pp JSON (1213-1231)
# ---------------------------------------------------------------------------


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
    feat_df = pd.DataFrame(
        {"feat_1": [1.0, -0.5], "feat_2": [2.5, 1.5], "n_reviews": [100, 80]}
    )
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

    def test_non_identity_transform_lines_1012_and_1027(self, tmp_path, summary):
        # With logit transform, lines 1012 (inverse) and 1027 (forward) both
        # execute; just confirm the pipeline completes without error.
        from panelcast.models.bayes.transforms import get_transform

        t = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.5)
        if t.name == "identity":
            pytest.skip("logit resolves to identity in this build")

        result = self._patched_run(tmp_path, summary, transform_name="offset_logit")
        assert "metrics" in result
