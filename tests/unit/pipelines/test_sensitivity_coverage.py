"""Coverage-focused tests for panelcast.pipelines.sensitivity.

Targets missed lines in:
- run_prior_sensitivity (model fitting loop, LOO failure branch, logging)
- run_threshold_sensitivity (data loading loop, LOO failure branch)
- run_feature_ablation (baseline fit, per-group ablation loop, LOO failure)
- extract_coefficient_summary (prefix, KeyError fallback, empty valid_vars)
- aggregate_sensitivity_results (coefficients metric branch)
- create_oat_summary_table (name-parsing edge cases, non-finite ELPD)
- generate_oat_configs (unknown parameter warning)
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import arviz as az
import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.cv import LOOResult
from panelcast.evaluation.metrics import CRPSResult
from panelcast.models.bayes.diagnostics import ConvergenceDiagnostics
from panelcast.models.bayes.fit import FitResult, MCMCConfig
from panelcast.models.bayes.priors import PriorConfig, get_default_priors
from panelcast.pipelines.sensitivity import (
    PRIOR_CONFIGS,
    SensitivityResult,
    aggregate_sensitivity_results,
    create_coefficient_comparison_df,
    create_oat_summary_table,
    extract_coefficient_summary,
    generate_oat_configs,
    run_feature_ablation,
    run_prior_sensitivity,
    run_threshold_sensitivity,
)

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def passing_convergence():
    """ConvergenceDiagnostics that reports passed=True."""
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
def failing_convergence():
    """ConvergenceDiagnostics that reports passed=False."""
    return ConvergenceDiagnostics(
        rhat_max=1.05,
        ess_bulk_min=100,
        ess_tail_min=80,
        divergences=15,
        passed=False,
        failing_params=["param1"],
        summary_df=pd.DataFrame(),
        rhat_threshold=1.01,
        ess_threshold=400,
    )


@pytest.fixture
def simple_idata():
    """Minimal ArviZ InferenceData with two posterior variables."""
    return az.from_dict(
        posterior={
            "user_beta": np.random.default_rng(0).normal(size=(2, 50, 3)),
            "user_rho": np.random.default_rng(1).normal(size=(2, 50)),
        }
    )


@pytest.fixture
def mock_loo():
    """LOOResult with reasonable values."""
    return LOOResult(
        loo=MagicMock(
            elpd_loo=-500.0, se=20.0, p_loo=50.0, pareto_k=np.array([0.1, 0.3]), warning=None
        ),
        elpd_loo=-500.0,
        se_elpd=20.0,
        p_loo=50.0,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )


def _make_fit_result(idata):
    """Build a FitResult wrapping the given idata."""
    return FitResult(
        mcmc=MagicMock(),
        idata=idata,
        divergences=0,
        runtime_seconds=1.0,
        gpu_info="CPU only",
    )


# ============================================================================
# extract_coefficient_summary
# ============================================================================


class TestExtractCoefficientSummary:
    """Tests for extract_coefficient_summary covering prefix and error branches."""

    def test_basic_extraction_returns_dataframe(self, simple_idata):
        """Extracting all variables returns a non-empty DataFrame."""
        df = extract_coefficient_summary(simple_idata)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_var_names_filter(self, simple_idata):
        """Specifying var_names restricts the output."""
        df = extract_coefficient_summary(simple_idata, var_names=["user_rho"])
        assert "user_rho" in df.index or any("user_rho" in str(i) for i in df.index)

    def test_prefix_prepended_to_var_names(self, simple_idata):
        """Prefix is prepended to each name in var_names."""
        df = extract_coefficient_summary(simple_idata, var_names=["rho"], prefix="user_")
        assert not df.empty

    def test_prefix_ignored_when_var_names_none(self, simple_idata):
        """Prefix has no effect when var_names is None."""
        df = extract_coefficient_summary(simple_idata, var_names=None, prefix="user_")
        assert not df.empty

    def test_missing_var_names_returns_empty(self, simple_idata):
        """When all requested var_names are missing, returns empty DataFrame."""
        df = extract_coefficient_summary(simple_idata, var_names=["nonexistent_param"])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_partial_missing_var_names(self, simple_idata):
        """When some var_names are missing, returns summary for the valid ones."""
        df = extract_coefficient_summary(simple_idata, var_names=["user_rho", "missing_param"])
        assert not df.empty
        assert any("user_rho" in str(i) for i in df.index)

    def test_key_error_reraise_when_var_names_none(self):
        """KeyError is re-raised when var_names is None and summary fails."""
        # When var_names is None but supplied as explicit list with a bad name,
        # and the only names are bad, function returns empty df.
        # The re-raise branch is hit when az.summary itself fails with var_names=None.
        # We trigger it by passing a var_names=[bad] so the initial az.summary
        # raises, then var_names is not None so it falls through to per-var loop.
        # To test the re-raise: pass var_names=None but corrupt the idata.
        import xarray as xr

        # Create idata whose posterior group is empty (no data vars)
        empty_ds = xr.Dataset()
        idata = az.InferenceData(posterior=empty_ds)
        with pytest.raises((KeyError, TypeError, ValueError)):
            extract_coefficient_summary(idata, var_names=None)


# ============================================================================
# run_prior_sensitivity
# ============================================================================


class TestRunPriorSensitivity:
    """Tests for run_prior_sensitivity with mocked fitting."""

    def test_returns_results_for_each_config(self, monkeypatch, simple_idata, passing_convergence):
        """One SensitivityResult per prior config entry."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )

        configs = {
            "a": get_default_priors(),
            "b": get_default_priors(),
        }
        results = run_prior_sensitivity(
            model=lambda: None,
            model_args={"y": np.zeros(5)},
            configs=configs,
            compute_loo_cv=False,
        )
        assert set(results.keys()) == {"a", "b"}
        for r in results.values():
            assert isinstance(r, SensitivityResult)
            assert r.loo is None  # LOO disabled

    def test_loo_computed_when_enabled(
        self, monkeypatch, simple_idata, passing_convergence, mock_loo
    ):
        """LOO is populated when compute_loo_cv=True and computation succeeds."""
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

        results = run_prior_sensitivity(
            model=lambda: None,
            model_args={"y": np.zeros(5)},
            configs={"only": get_default_priors()},
            compute_loo_cv=True,
        )
        assert results["only"].loo is mock_loo

    def test_loo_failure_logged_not_raised(self, monkeypatch, simple_idata, passing_convergence):
        """When LOO computation raises, result.loo is None and no exception propagates."""
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
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("loo boom")),
        )

        results = run_prior_sensitivity(
            model=lambda: None,
            model_args={"y": np.zeros(5)},
            configs={"fail_loo": get_default_priors()},
            compute_loo_cv=True,
        )
        assert results["fail_loo"].loo is None

    def test_default_configs_and_mcmc_config(self, monkeypatch, simple_idata, passing_convergence):
        """When configs and mcmc_config are None, defaults are used."""
        call_log = []

        def fake_fit(model, args, config=None, progress_bar=True):
            call_log.append(config)
            return _make_fit_result(simple_idata)

        monkeypatch.setattr("panelcast.pipelines.sensitivity.fit_model", fake_fit)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )

        results = run_prior_sensitivity(
            model=lambda: None,
            model_args={"y": np.zeros(5)},
            configs=None,
            mcmc_config=None,
            compute_loo_cv=False,
        )
        # Should use PRIOR_CONFIGS (3 entries)
        assert len(results) == len(PRIOR_CONFIGS)
        # MCMCConfig should be the default
        assert all(isinstance(c, MCMCConfig) for c in call_log)

    def test_convergence_status_logged(self, monkeypatch, simple_idata, failing_convergence):
        """Result captures failing convergence diagnostics."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: failing_convergence,
        )

        results = run_prior_sensitivity(
            model=lambda: None,
            model_args={"y": np.zeros(5)},
            configs={"bad": get_default_priors()},
            compute_loo_cv=False,
        )
        assert results["bad"].convergence.passed is False
        assert results["bad"].convergence.divergences == 15


# ============================================================================
# run_threshold_sensitivity
# ============================================================================


class TestRunThresholdSensitivity:
    """Tests for run_threshold_sensitivity with mocked data loader."""

    def test_returns_results_per_threshold(self, monkeypatch, simple_idata, passing_convergence):
        """One SensitivityResult per threshold value."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )

        def loader(threshold):
            df = pd.DataFrame({"x": range(threshold)})
            return df, {"y": np.zeros(threshold)}

        results = run_threshold_sensitivity(
            model=lambda: None,
            data_loader=loader,
            thresholds=(5, 15),
            compute_loo_cv=False,
        )
        assert set(results.keys()) == {5, 15}
        assert results[5].name == "threshold_5"
        assert results[5].config["n_obs"] == 5
        assert results[15].config["threshold"] == 15

    def test_loo_failure_in_threshold(self, monkeypatch, simple_idata, passing_convergence):
        """LOO failure in a threshold variant results in loo=None."""
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
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("loo fail")),
        )

        def loader(t):
            return pd.DataFrame({"x": [1]}), {"y": np.zeros(1)}

        results = run_threshold_sensitivity(
            model=lambda: None,
            data_loader=loader,
            thresholds=(10,),
            compute_loo_cv=True,
        )
        assert results[10].loo is None

    def test_n_obs_from_model_args_when_df_has_no_len(
        self, monkeypatch, simple_idata, passing_convergence
    ):
        """n_obs falls back to model_args['y'].shape[0] when df has no __len__."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )

        class NoLenObj:
            pass

        def loader(t):
            return NoLenObj(), {"y": np.zeros(7)}

        results = run_threshold_sensitivity(
            model=lambda: None,
            data_loader=loader,
            thresholds=(10,),
            compute_loo_cv=False,
        )
        assert results[10].config["n_obs"] == 7


# ============================================================================
# run_feature_ablation
# ============================================================================


class TestRunFeatureAblation:
    """Tests for run_feature_ablation with mocked fitting."""

    def _run(self, monkeypatch, simple_idata, convergence, loo_raises=False, compute_loo=False):
        """Helper to run ablation with standard mocks."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: convergence,
        )
        if compute_loo and not loo_raises:
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
                lambda *a, **kw: LOOResult(
                    loo=MagicMock(
                        elpd_loo=-100, se=5, p_loo=10, pareto_k=np.array([0.1]), warning=None
                    ),
                    elpd_loo=-100.0,
                    se_elpd=5.0,
                    p_loo=10.0,
                    n_high_pareto_k=0,
                    high_pareto_k_indices=np.array([]),
                    warning=None,
                ),
            )
        elif compute_loo and loo_raises:
            monkeypatch.setattr(
                "panelcast.pipelines.sensitivity.compute_log_likelihood",
                lambda *a, **kw: (_ for _ in ()).throw(KeyError("boom")),
            )

        X = np.ones((10, 5), dtype=np.float32)
        model_args = {"X": X, "y": np.zeros(10)}
        feature_groups = {"group_a": [0, 1], "group_b": [3, 4]}

        return run_feature_ablation(
            model=lambda: None,
            model_args=model_args,
            feature_groups=feature_groups,
            compute_loo_cv=compute_loo,
        )

    def test_full_baseline_plus_ablations(self, monkeypatch, simple_idata, passing_convergence):
        """Returns 'full' baseline and one 'no_{group}' entry per group."""
        results = self._run(monkeypatch, simple_idata, passing_convergence)
        assert "full" in results
        assert "no_group_a" in results
        assert "no_group_b" in results
        assert len(results) == 3

    def test_ablation_config_records_group_info(
        self, monkeypatch, simple_idata, passing_convergence
    ):
        """Each ablation result records which group and columns were ablated."""
        results = self._run(monkeypatch, simple_idata, passing_convergence)
        assert results["full"].config["ablated_features"] is None
        assert results["no_group_a"].config["ablated_features"] == "group_a"
        assert results["no_group_a"].config["ablated_columns"] == [0, 1]

    def test_loo_computed_for_all_variants(self, monkeypatch, simple_idata, passing_convergence):
        """LOO is populated for baseline and all ablations when enabled."""
        results = self._run(monkeypatch, simple_idata, passing_convergence, compute_loo=True)
        for name, r in results.items():
            assert r.loo is not None, f"LOO missing for {name}"

    def test_loo_failure_in_baseline_captured(self, monkeypatch, simple_idata, passing_convergence):
        """LOO failure in the full baseline yields loo=None for that entry."""
        results = self._run(
            monkeypatch,
            simple_idata,
            passing_convergence,
            compute_loo=True,
            loo_raises=True,
        )
        assert results["full"].loo is None

    def test_failing_convergence_recorded(self, monkeypatch, simple_idata, failing_convergence):
        """Failing convergence is recorded on each result."""
        results = self._run(monkeypatch, simple_idata, failing_convergence)
        for r in results.values():
            assert r.convergence.passed is False


# ============================================================================
# aggregate_sensitivity_results - coefficients metric
# ============================================================================


class TestAggregateCoefficientsMetric:
    """Tests for the 'coefficients' metric branch in aggregate_sensitivity_results."""

    def test_coefficients_metric_populates_param_columns(self):
        """Aggregation with metric='coefficients' adds per-param mean/sd columns."""
        coef_df = pd.DataFrame(
            {"mean": [1.0, 2.0], "sd": [0.1, 0.2]},
            index=["alpha", "beta"],
        )
        results = {
            "v1": SensitivityResult(
                name="v1",
                config={},
                convergence=None,
                coefficients=coef_df,
            )
        }
        df = aggregate_sensitivity_results(results, metric="coefficients")
        assert "alpha_mean" in df.columns
        assert "alpha_sd" in df.columns
        assert "beta_mean" in df.columns

    def test_coefficients_metric_with_empty_coefficients(self):
        """Empty coefficients produce no extra columns."""
        results = {"empty": SensitivityResult(name="empty", config={}, coefficients=pd.DataFrame())}
        df = aggregate_sensitivity_results(results, metric="coefficients")
        assert len(df) == 1
        # Should still have convergence columns but no param columns
        assert "convergence_passed" in df.columns

    def test_coefficients_metric_no_sd_column(self):
        """When coefficients lack 'sd' column, only mean columns are added."""
        coef_df = pd.DataFrame({"mean": [1.0]}, index=["gamma"])
        results = {"v1": SensitivityResult(name="v1", config={}, coefficients=coef_df)}
        df = aggregate_sensitivity_results(results, metric="coefficients")
        assert "gamma_mean" in df.columns
        assert "gamma_sd" not in df.columns


# ============================================================================
# generate_oat_configs edge cases
# ============================================================================


class TestGenerateOatConfigsEdgeCases:
    """Edge cases for generate_oat_configs."""

    def test_custom_base(self):
        """Custom base config is respected and included as 'default'."""
        custom = PriorConfig(mu_artist_scale=10.0)
        configs = generate_oat_configs(
            base=custom, parameters=("mu_artist_scale",), multipliers=(2.0,)
        )
        assert configs["default"] is custom
        assert configs["mu_artist_scale_x2"].mu_artist_scale == 20.0

    def test_unknown_parameter_skipped(self):
        """Parameters not in PriorConfig are skipped with a warning."""
        configs = generate_oat_configs(parameters=("nonexistent_field",), multipliers=(2.0,))
        # Only the default config should be present
        assert list(configs.keys()) == ["default"]

    def test_custom_multipliers(self):
        """Custom multipliers are used instead of defaults."""
        configs = generate_oat_configs(parameters=("beta_scale",), multipliers=(0.1, 10.0))
        assert "beta_scale_x0.1" in configs
        assert "beta_scale_x10" in configs
        base_val = asdict(get_default_priors())["beta_scale"]
        assert configs["beta_scale_x10"].beta_scale == pytest.approx(base_val * 10.0)


# ============================================================================
# create_oat_summary_table edge cases
# ============================================================================


class TestCreateOatSummaryTableEdgeCases:
    """Edge cases for OAT summary table generation."""

    def test_non_oat_name_format(self, passing_convergence, mock_loo):
        """Variant names without '_x' are stored as-is with multiplier=None."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
            "custom_name": SensitivityResult(
                name="custom_name",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
        }
        df = create_oat_summary_table(results)
        custom_row = df[df["variant"] == "custom_name"].iloc[0]
        assert custom_row["parameter"] == "custom_name"
        assert pd.isna(custom_row["multiplier"]) or custom_row["multiplier"] is None

    def test_non_numeric_multiplier_in_name(self, passing_convergence, mock_loo):
        """Variant name with '_x' but non-numeric suffix falls back gracefully."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
            "param_xabc": SensitivityResult(
                name="param_xabc",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
        }
        df = create_oat_summary_table(results)
        row = df[df["variant"] == "param_xabc"].iloc[0]
        assert row["parameter"] == "param_xabc"
        assert pd.isna(row["multiplier"]) or row["multiplier"] is None

    def test_non_finite_elpd_treated_as_none(self, passing_convergence):
        """Non-finite ELPD (NaN/Inf) treated as missing; variant not eligible."""
        nan_loo = LOOResult(
            loo=MagicMock(
                elpd_loo=float("nan"), se=float("nan"), p_loo=0, pareto_k=np.array([]), warning=None
            ),
            elpd_loo=float("nan"),
            se_elpd=float("nan"),
            p_loo=0.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=nan_loo,
            ),
        }
        df = create_oat_summary_table(results)
        assert not df.iloc[0]["eligible_for_ranking"]

    def test_base_missing_from_results(self, passing_convergence, mock_loo):
        """When base_name is not present, all deltas are None."""
        results = {
            "sigma_rw_scale_x2": SensitivityResult(
                name="sigma_rw_scale_x2",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
        }
        df = create_oat_summary_table(results, base_name="missing_default")
        assert not df.iloc[0]["eligible_for_ranking"]

    def test_empty_results(self):
        """Empty results dict produces empty DataFrame."""
        df = create_oat_summary_table({})
        assert df.empty

    def test_base_convergence_failed(self, failing_convergence, mock_loo):
        """When baseline convergence failed, no variant is eligible."""
        ok_conv = ConvergenceDiagnostics(
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
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=failing_convergence,
                loo=mock_loo,
            ),
            "sigma_rw_scale_x2": SensitivityResult(
                name="sigma_rw_scale_x2",
                config={},
                convergence=ok_conv,
                loo=mock_loo,
            ),
        }
        df = create_oat_summary_table(results)
        # base_eligible is False => nobody eligible
        assert not df["eligible_for_ranking"].any()

    def test_base_loo_none(self, passing_convergence, mock_loo):
        """When baseline LOO is None, all deltas are None."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=None,
            ),
            "beta_scale_x2": SensitivityResult(
                name="beta_scale_x2",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
        }
        df = create_oat_summary_table(results)
        assert not df["eligible_for_ranking"].any()


# ============================================================================
# create_coefficient_comparison_df edge cases
# ============================================================================


class TestCoefficientComparisonEdgeCasesExtended:
    """Additional edge-case tests for create_coefficient_comparison_df."""

    def test_hdi_bounds_extraction(self):
        """Lower/upper bounds are extracted from hdi_X% columns."""
        coef = pd.DataFrame(
            {"mean": [0.5], "hdi_3%": [0.1], "hdi_97%": [0.9]},
            index=["param_a"],
        )
        results = {"v1": SensitivityResult(name="v1", config={}, coefficients=coef)}
        df = create_coefficient_comparison_df(results, ["param_a"])
        assert len(df) == 1
        assert df.iloc[0]["lower"] == pytest.approx(0.1)
        assert df.iloc[0]["upper"] == pytest.approx(0.9)

    def test_no_hdi_columns(self):
        """When coefficients lack HDI columns, lower/upper are None."""
        coef = pd.DataFrame({"mean": [0.5]}, index=["param_a"])
        results = {"v1": SensitivityResult(name="v1", config={}, coefficients=coef)}
        df = create_coefficient_comparison_df(results, ["param_a"])
        assert df.iloc[0]["lower"] is None
        assert df.iloc[0]["upper"] is None
