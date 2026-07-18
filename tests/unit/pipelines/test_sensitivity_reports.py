"""Tests targeting uncovered lines in panelcast.pipelines.sensitivity.

Covers:
- line 184: re-raise KeyError when var_names is None
- lines 397-398: run_threshold_sensitivity LOO success path
- lines 886-887: TypeError/ValueError on base elpd in create_oat_summary_table
- lines 920-921: TypeError/ValueError on per-variant elpd
- lines 926-927: TypeError/ValueError on per-variant elpd_se
- lines 1021-1098: run_split_seed_sensitivity
- lines 1158-1285: run_sensitivity_suite
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import arviz as az
import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.cv import LOOResult
from panelcast.models.bayes.diagnostics import ConvergenceDiagnostics
from panelcast.models.bayes.fit import FitResult, MCMCConfig
from panelcast.models.bayes.priors import get_default_priors
from panelcast.pipelines.sensitivity import (
    SensitivityResult,
    create_oat_summary_table,
    extract_coefficient_summary,
    run_split_seed_sensitivity,
    run_threshold_sensitivity,
)

# ============================================================================
# Shared fixtures (mirrors test_sensitivity_coverage.py)
# ============================================================================


@pytest.fixture
def passing_convergence():
    return ConvergenceDiagnostics(
        rhat_max=1.001,
        ess_bulk_min=2500,
        ess_tail_min=2200,
        divergences=0,
        passed=True,
        failing_params=[],
        summary_df=pd.DataFrame(),
        rhat_threshold=1.01,
        ess_threshold=400,
    )


@pytest.fixture
def simple_idata():
    return az.from_dict(
        posterior={
            "user_beta": np.random.default_rng(0).normal(size=(2, 50, 3)),
            "user_rho": np.random.default_rng(1).normal(size=(2, 50)),
        }
    )


@pytest.fixture
def mock_loo():
    return LOOResult(
        loo=MagicMock(
            elpd_loo=-500.0, se=20.0, p_loo=50.0, pareto_k=np.array([0.1]), warning=None
        ),
        elpd_loo=-500.0,
        se_elpd=20.0,
        p_loo=50.0,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )


def _make_fit_result(idata):
    return FitResult(
        mcmc=MagicMock(),
        idata=idata,
        divergences=0,
        runtime_seconds=1.0,
        gpu_info="CPU only",
    )


# ============================================================================
# extract_coefficient_summary — line 184 (re-raise when var_names is None)
# ============================================================================


class TestExtractCoefficientSummaryReRaise:
    def test_reraise_keyerror_when_var_names_none(self, monkeypatch):
        """KeyError propagates when var_names=None and az.summary raises."""
        import xarray as xr

        import panelcast.pipelines.sensitivity as _mod

        def _raise_key(*a, **kw):
            raise KeyError("no variables")

        monkeypatch.setattr(_mod.az, "summary", _raise_key)

        idata = az.from_dict(posterior={"x": np.ones((2, 10))})
        with pytest.raises(KeyError):
            extract_coefficient_summary(idata, var_names=None)


# ============================================================================
# run_threshold_sensitivity — lines 397-398 (LOO success path)
# ============================================================================


class TestRunThresholdSensitivityLooSuccess:
    def test_loo_success_path_calls_add_and_compute(
        self, monkeypatch, simple_idata, passing_convergence, mock_loo
    ):
        """Lines 397-398: add_log_likelihood_to_idata and compute_loo are called."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.compute_log_likelihood",
            lambda *a, **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.add_log_likelihood_to_idata",
            lambda *a, **kw: simple_idata,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.compute_loo",
            lambda *a, **kw: mock_loo,
        )

        def loader(t):
            return pd.DataFrame({"x": range(5)}), {"y": np.zeros(5)}

        results = run_threshold_sensitivity(
            model=lambda: None,
            data_loader=loader,
            thresholds=(10,),
            compute_loo_cv=True,
        )
        assert results[10].loo is mock_loo


# ============================================================================
# create_oat_summary_table — lines 886-887, 920-921, 926-927
# (TypeError/ValueError exception handlers)
# ============================================================================


class TestCreateOatSummaryTableExceptions:
    def _bad_loo(self, elpd=None, se=None):
        """LOO whose numeric fields raise TypeError when converted to float."""
        loo_obj = LOOResult(
            loo=MagicMock(elpd_loo=elpd, se=se, p_loo=0, pareto_k=np.array([]), warning=None),
            elpd_loo=elpd,
            se_elpd=se,
            p_loo=0.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        return loo_obj

    def test_base_elpd_type_error_falls_back_to_none(self, passing_convergence):
        """Lines 886-887: TypeError on float(base elpd) sets base_elpd=None."""
        bad = self._bad_loo(elpd="not-a-float", se=1.0)
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=bad,
            ),
        }
        df = create_oat_summary_table(results)
        # base_elpd is None → not eligible, delta is None
        assert not df.iloc[0]["eligible_for_ranking"]

    def test_variant_elpd_type_error_falls_back_to_none(self, passing_convergence):
        """Lines 920-921: TypeError on float(variant elpd) sets elpd=None."""
        good_loo = LOOResult(
            loo=MagicMock(elpd_loo=-400.0, se=10.0, p_loo=5, pareto_k=np.array([]), warning=None),
            elpd_loo=-400.0,
            se_elpd=10.0,
            p_loo=5.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        bad_elpd = self._bad_loo(elpd="bad", se=5.0)
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=good_loo,
            ),
            "param_x2": SensitivityResult(
                name="param_x2",
                config={},
                convergence=passing_convergence,
                loo=bad_elpd,
            ),
        }
        df = create_oat_summary_table(results)
        variant_row = df[df["variant"] == "param_x2"].iloc[0]
        assert not variant_row["eligible_for_ranking"]
        assert variant_row["elpd"] is None or pd.isna(variant_row["elpd"])

    def test_variant_elpd_se_type_error_falls_back_to_none(self, passing_convergence):
        """Lines 926-927: TypeError on float(variant elpd_se) sets elpd_se=None."""
        good_loo = LOOResult(
            loo=MagicMock(elpd_loo=-400.0, se=10.0, p_loo=5, pareto_k=np.array([]), warning=None),
            elpd_loo=-400.0,
            se_elpd=10.0,
            p_loo=5.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        # valid elpd but bad se
        bad_se = LOOResult(
            loo=MagicMock(elpd_loo=-300.0, se="oops", p_loo=5, pareto_k=np.array([]), warning=None),
            elpd_loo=-300.0,
            se_elpd="oops",
            p_loo=5.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=good_loo,
            ),
            "param_x2": SensitivityResult(
                name="param_x2",
                config={},
                convergence=passing_convergence,
                loo=bad_se,
            ),
        }
        df = create_oat_summary_table(results)
        variant_row = df[df["variant"] == "param_x2"].iloc[0]
        # elpd is valid so eligible, but elpd_se should be None/NaN
        assert variant_row["elpd_se"] is None or pd.isna(variant_row["elpd_se"])


# ============================================================================
# run_split_seed_sensitivity — lines 1021-1098
# ============================================================================


class TestRunSplitSeedSensitivity:
    """Tests for run_split_seed_sensitivity (no real MCMC, stub all heavy deps)."""

    def _make_posterior_samples(self, n_features=2, prefix="user"):
        rng = np.random.default_rng(0)
        return {
            f"{prefix}_beta": rng.normal(size=(200, n_features)).astype(np.float32),
            f"{prefix}_rho": rng.normal(size=(200,)).astype(np.float32),
            f"{prefix}_mu": rng.normal(size=(200,)).astype(np.float32),
            f"{prefix}_sigma": np.abs(rng.normal(size=(200,))).astype(np.float32),
        }

    def _make_summary(self, prefix="user"):
        return {
            "dataset": {
                "entity_col": "Artist",
                "target_col": "Score",
                "model_prefix": prefix,
                "target_bounds": [0.0, 100.0],
            },
            "target_transform": "identity",
            "logit_offset": 0.5,
            "n_reviews_stats": {"median": 10.0},
            "global_mean_score": 70.0,
            "likelihood_df": 4.0,
            "priors": {},
        }

    def _make_source_df(self, n=60, prefix="user"):
        rng = np.random.default_rng(7)
        artists = [f"Artist_{i}" for i in range(20)]
        return pd.DataFrame(
            {
                "Artist": np.repeat(artists, 3),
                "Score": rng.uniform(50, 90, n).astype(float),
            }
        )

    def test_returns_seed_rows_and_spread(self, monkeypatch):
        """Full happy path: returns per-seed coverage dict and 'spread' key.

        run_split_seed_sensitivity imports its helpers inside the function body,
        so we patch them at the source modules they're imported from.
        """
        posterior_samples = self._make_posterior_samples()
        summary = self._make_summary()
        source_df = self._make_source_df()

        rng = np.random.default_rng(1)
        small_test = pd.DataFrame(
            {"Artist": ["A1", "A2", "A3"], "Score": rng.uniform(50, 90, 3).astype(float)}
        )
        n_test = len(small_test)

        monkeypatch.setattr(
            "panelcast.data.split.entity_disjoint_split",
            lambda df, entity_col, test_size, val_size, random_state: (
                df.iloc[:40], df.iloc[40:50], small_test
            ),
        )
        monkeypatch.setattr(
            "panelcast.models.bayes.predict.predict_new_entity",
            lambda *a, **kw: {
                "y": np.random.default_rng(2).uniform(50, 90, (200, n_test))
            },
        )
        monkeypatch.setattr(
            "panelcast.pipelines.training_summary.ar_center_on_model_scale",
            lambda s: 0.0,
        )

        result = run_split_seed_sensitivity(
            source_df,
            posterior_samples,
            summary,
            seeds=(42, 43),
        )

        assert "seed_42" in result
        assert "seed_43" in result
        assert "spread" in result
        assert "coverage" in result["seed_42"]
        assert "coverage_max_minus_min" in result["spread"]

    def test_single_seed_spread_is_zero(self, monkeypatch):
        """With one seed, spread (max-min) == 0."""
        posterior_samples = self._make_posterior_samples()
        summary = self._make_summary()
        source_df = self._make_source_df()

        small_test = pd.DataFrame({"Artist": ["A1", "A2"], "Score": [70.0, 80.0]})
        n_test = len(small_test)

        monkeypatch.setattr(
            "panelcast.data.split.entity_disjoint_split",
            lambda df, **kw: (df.iloc[:40], df.iloc[40:50], small_test),
        )
        monkeypatch.setattr(
            "panelcast.models.bayes.predict.predict_new_entity",
            lambda *a, **kw: {"y": np.full((200, n_test), 75.0)},
        )
        monkeypatch.setattr(
            "panelcast.pipelines.training_summary.ar_center_on_model_scale",
            lambda s: 0.0,
        )

        result = run_split_seed_sensitivity(
            source_df, posterior_samples, summary, seeds=(42,)
        )
        assert result["spread"]["coverage_max_minus_min"] == pytest.approx(0.0)

    def test_non_identity_transform_applies_forward(self, monkeypatch):
        """When target_transform != 'identity', prev is passed through get_transform."""
        posterior_samples = self._make_posterior_samples()
        summary = self._make_summary()
        summary["target_transform"] = "offset_logit"
        source_df = self._make_source_df()

        small_test = pd.DataFrame({"Artist": ["A1"], "Score": [70.0]})

        monkeypatch.setattr(
            "panelcast.data.split.entity_disjoint_split",
            lambda df, **kw: (df, df, small_test),
        )
        monkeypatch.setattr(
            "panelcast.models.bayes.predict.predict_new_entity",
            lambda *a, **kw: {"y": np.full((200, 1), 75.0)},
        )
        monkeypatch.setattr(
            "panelcast.pipelines.training_summary.ar_center_on_model_scale",
            lambda s: 0.0,
        )

        result = run_split_seed_sensitivity(
            source_df, posterior_samples, summary, seeds=(42,)
        )
        assert "seed_42" in result

    def test_threads_observation_model_from_priors(self, monkeypatch):
        """Coverage is scored under the trained observation model, sourced from
        the single PriorConfig (not a mix of top-level and priors lookups)."""
        posterior_samples = self._make_posterior_samples()
        summary = self._make_summary()
        # Production summaries carry the family both top-level and in priors;
        # the fix reads the priors copy, so a stale top-level default must lose.
        summary["likelihood_family"] = "studentt"
        summary["discretize_observation"] = False
        summary["priors"] = {
            "likelihood_family": "beta",
            "skew_tailweight": 1.7,
            "discretize_observation": True,
        }
        summary["learn_n_exponent"] = False
        summary["n_exponent"] = 0.5
        source_df = self._make_source_df()

        small_test = pd.DataFrame({"Artist": ["A1"], "Score": [70.0]})

        captured: list[dict] = []
        monkeypatch.setattr(
            "panelcast.data.split.entity_disjoint_split",
            lambda df, **kw: (df, df, small_test),
        )

        def _capture(*a, **kw):
            captured.append(kw)
            return {"y": np.full((200, 1), 75.0)}

        monkeypatch.setattr(
            "panelcast.models.bayes.predict.predict_new_entity", _capture
        )
        monkeypatch.setattr(
            "panelcast.pipelines.training_summary.ar_center_on_model_scale",
            lambda s: 0.0,
        )

        run_split_seed_sensitivity(source_df, posterior_samples, summary, seeds=(42,))

        assert captured
        call = captured[0]
        assert call["likelihood_family"] == "beta"
        assert call["skew_tailweight"] == pytest.approx(1.7)
        assert call["discretize_observation"] is True
        assert call["fixed_n_exponent"] == pytest.approx(0.5)


# ============================================================================
# run_sensitivity_suite — lines 1158-1285
# ============================================================================


class TestRunSensitivitySuite:
    """Tests for run_sensitivity_suite, fully mocked to avoid real I/O."""

    def _ctx(self, axes=("priors",)):
        descriptor = MagicMock()
        descriptor.model_prefix = "user"
        descriptor.target_bounds = (0.0, 100.0)
        descriptor.processed_name.return_value = "aoty_None"
        ctx = SimpleNamespace(
            descriptor=descriptor,
            sensitivity_axes=axes,
            min_albums_filter=2,
            max_albums=50,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=0,
            target_accept=0.9,
            max_tree_depth=10,
            chain_method="sequential",
            n_exponent=0.0,
            learn_n_exponent=False,
            sensitivity_split_seeds=(42,),
            min_ratings=None,
        )
        return ctx

    def _patch_suite_deps(self, monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo):
        """Patch every external dependency run_sensitivity_suite touches.

        run_sensitivity_suite imports several symbols locally (inside the function
        body), so we must patch them at their *source* module, not via the
        sensitivity namespace.
        """
        # load_training_summary — imported locally inside run_sensitivity_suite
        # as `from panelcast.pipelines.training_summary import load_training_summary`
        fake_summary_dict = {
            "debut_prev_score_source": "train_mean",
            "target_transform": "identity",
            "logit_offset": 0.5,
            "priors": {"ar_center": "global"},
            "global_mean_score": 70.0,
            "n_reviews_stats": {"median": 10.0},
            "likelihood_df": 4.0,
            "n_ref": 5,
            "dataset": {
                "entity_col": "Artist",
                "target_col": "Score",
                "model_prefix": "user",
                "target_bounds": [0.0, 100.0],
            },
        }
        fake_summary_obj = MagicMock()
        fake_summary_obj.to_json_dict.return_value = fake_summary_dict
        import panelcast.pipelines.training_summary as _ts_mod
        monkeypatch.setattr(_ts_mod, "load_training_summary", lambda p: fake_summary_obj)

        # load_training_data / _apply_max_albums_cap / locate_level_prior
        # imported from panelcast.pipelines.train_bayes
        n = 10
        X = np.ones((n, 3), dtype=np.float32)
        model_args_base = {
            "X": X,
            "y": np.zeros(n, dtype=np.float32),
            "artist_album_counts": np.ones(n, dtype=np.int32),
            "artist_to_idx": {},
            "global_mean_score": 70.0,
            "ar_center_value": 0.0,
        }
        feature_cols = ["genre_pc1", "user_prior_mean", "album_sequence"]
        train_df = pd.DataFrame({"Artist": [f"A{i}" for i in range(n)], "Score": np.zeros(n)})
        import panelcast.pipelines.train_bayes as _tb_mod
        monkeypatch.setattr(_tb_mod, "load_training_data", lambda **kw: (dict(model_args_base), feature_cols, train_df, None))
        monkeypatch.setattr(_tb_mod, "_apply_max_albums_cap", lambda args, max_albums, counts: args)
        monkeypatch.setattr(_tb_mod, "locate_level_prior", lambda config, **kw: config)

        # make_score_model — imported from panelcast.models.bayes.model
        import panelcast.models.bayes.model as _model_mod
        monkeypatch.setattr(_model_mod, "make_score_model", lambda prefix: (lambda: None))

        # fit_model / check_convergence / resolve_split_dir — top-level imports in sensitivity.py
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr("panelcast.pipelines.sensitivity.fit_model", lambda *a, **kw: fit_result)
        monkeypatch.setattr("panelcast.pipelines.sensitivity.check_convergence", lambda *a, **kw: passing_convergence)
        monkeypatch.setattr("panelcast.pipelines.sensitivity.resolve_split_dir", lambda p, split_type: tmp_path, raising=False)

        # LOO computation — also top-level imports; mock to avoid numpyro calls
        monkeypatch.setattr("panelcast.pipelines.sensitivity.compute_log_likelihood", lambda *a, **kw: MagicMock())
        monkeypatch.setattr("panelcast.pipelines.sensitivity.add_log_likelihood_to_idata", lambda *a, **kw: simple_idata)
        monkeypatch.setattr("panelcast.pipelines.sensitivity.compute_loo", lambda *a, **kw: mock_loo)

        return fake_summary_dict

    def test_priors_axis_writes_json(
        self, monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo
    ):
        """'priors' axis runs, writes sensitivity_results.json and oat_summary.csv."""
        from panelcast.pipelines.sensitivity import run_sensitivity_suite

        self._patch_suite_deps(monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo)

        # Redirect output dir by patching Path inside the function
        out_dir = tmp_path / "reports" / "sensitivity"

        import panelcast.pipelines.sensitivity as _mod

        original_path = _mod.__builtins__ if hasattr(_mod, "__builtins__") else None

        # Patch cwd-relative Path calls by monkeypatching json.dump output path
        # The suite does: out_dir = Path("reports/sensitivity"); out_dir.mkdir(...)
        # We need to override that. Patch the json open call or redirect via monkeypatch.chdir.
        monkeypatch.chdir(tmp_path)

        ctx = self._ctx(axes=("priors",))
        result = run_sensitivity_suite(ctx)

        assert "sensitivity_results" in result
        json_path = tmp_path / "reports" / "sensitivity" / "sensitivity_results.json"
        assert json_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert "priors" in payload
        assert payload["axes"] == ["priors"]

        oat_csv = tmp_path / "reports" / "sensitivity" / "oat_summary.csv"
        assert oat_csv.exists()

    def test_ablation_axis_runs(
        self, monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo
    ):
        """'ablation' axis runs feature ablation and records results."""
        from panelcast.pipelines.sensitivity import run_sensitivity_suite

        self._patch_suite_deps(monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo)
        monkeypatch.chdir(tmp_path)

        ctx = self._ctx(axes=("ablation",))
        result = run_sensitivity_suite(ctx)

        json_path = tmp_path / "reports" / "sensitivity" / "sensitivity_results.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert "ablation" in payload

    def test_split_seed_axis_runs(
        self, monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo
    ):
        """'split_seed' axis reads manifest/model, runs coverage sensitivity."""
        from panelcast.pipelines.sensitivity import run_sensitivity_suite

        self._patch_suite_deps(monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo)
        monkeypatch.chdir(tmp_path)

        # Write the manifest and a fake model file
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        manifest = {"current": {"user_score": "user_score_v1.nc"}}
        (models_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        simple_idata.to_netcdf(str(models_dir / "user_score_v1.nc"))

        # Write a fake processed parquet
        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        n = 60
        source_df = pd.DataFrame(
            {"Artist": [f"A{i % 20}" for i in range(n)], "Score": rng.uniform(50, 90, n)}
        )
        source_df.to_parquet(data_dir / "aoty_None.parquet")

        # Patch helpers used inside run_split_seed_sensitivity (local imports)
        small_test = source_df.iloc[:5].copy()
        import panelcast.data.split as _split_mod
        import panelcast.models.bayes.predict as _pred_mod
        import panelcast.pipelines.training_summary as _ts_mod2
        monkeypatch.setattr(_split_mod, "entity_disjoint_split", lambda df, **kw: (df, df, small_test))
        monkeypatch.setattr(_pred_mod, "predict_new_entity", lambda *a, **kw: {"y": np.full((200, len(small_test)), 70.0)})
        monkeypatch.setattr(_ts_mod2, "ar_center_on_model_scale", lambda s: 0.0)
        # extract_posterior_samples — imported locally in run_sensitivity_suite
        monkeypatch.setattr(
            _pred_mod, "extract_posterior_samples",
            lambda idata: {"user_beta": np.ones((200, 3), dtype=np.float32), "user_rho": np.zeros(200, dtype=np.float32)},
        )

        ctx = self._ctx(axes=("split_seed",))
        result = run_sensitivity_suite(ctx)

        json_path = tmp_path / "reports" / "sensitivity" / "sensitivity_results.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert "split_seed" in payload
        assert "seed_42" in payload["split_seed"]

    def test_all_axes_together(
        self, monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo
    ):
        """All three axes run without error when combined."""
        from panelcast.pipelines.sensitivity import run_sensitivity_suite

        self._patch_suite_deps(monkeypatch, tmp_path, simple_idata, passing_convergence, mock_loo)
        monkeypatch.chdir(tmp_path)

        # Minimal manifest + model for split_seed
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        manifest = {"current": {"user_score": "user_score_v1.nc"}}
        (models_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        simple_idata.to_netcdf(str(models_dir / "user_score_v1.nc"))

        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        n = 60
        source_df = pd.DataFrame(
            {"Artist": [f"A{i % 20}" for i in range(n)], "Score": rng.uniform(50, 90, n)}
        )
        source_df.to_parquet(data_dir / "aoty_None.parquet")

        small_test = source_df.iloc[:5].copy()
        import panelcast.data.split as _split_mod
        import panelcast.models.bayes.predict as _pred_mod
        import panelcast.pipelines.training_summary as _ts_mod2
        monkeypatch.setattr(_split_mod, "entity_disjoint_split", lambda df, **kw: (df, df, small_test))
        monkeypatch.setattr(_pred_mod, "predict_new_entity", lambda *a, **kw: {"y": np.full((200, len(small_test)), 70.0)})
        monkeypatch.setattr(_ts_mod2, "ar_center_on_model_scale", lambda s: 0.0)
        monkeypatch.setattr(
            _pred_mod, "extract_posterior_samples",
            lambda idata: {"user_beta": np.ones((200, 3), dtype=np.float32), "user_rho": np.zeros(200, dtype=np.float32)},
        )

        ctx = self._ctx(axes=("priors", "ablation", "split_seed"))
        result = run_sensitivity_suite(ctx)

        payload = json.loads(
            (tmp_path / "reports" / "sensitivity" / "sensitivity_results.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(payload["axes"]) == {"priors", "ablation", "split_seed"}
