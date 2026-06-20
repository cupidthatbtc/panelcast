"""Additional coverage tests for sensitivity pipeline.

Targets uncovered branches including:
- aggregate_sensitivity_results: crps metric, convergence metric, empty results, elpd sorting
- create_coefficient_comparison_df: missing params, empty coefficients, multiple variants
- SensitivityResult fields
- run_prior_sensitivity with coefficient_vars specified
- run_threshold_sensitivity with default mcmc_config
- run_feature_ablation LOO failure per-group
- create_oat_summary_table sorting and eligibility logic
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.cv import LOOResult
from panelcast.evaluation.metrics import CRPSResult
from panelcast.models.bayes.diagnostics import ConvergenceDiagnostics
from panelcast.models.bayes.fit import FitResult, MCMCConfig
from panelcast.models.bayes.priors import PriorConfig, get_default_priors
from panelcast.pipelines.sensitivity import (
    OAT_MULTIPLIERS,
    OAT_PARAMETERS,
    SensitivityResult,
    aggregate_sensitivity_results,
    create_coefficient_comparison_df,
    create_oat_summary_table,
    generate_oat_configs,
    run_prior_sensitivity,
    run_threshold_sensitivity,
)

# ============================================================================
# Shared fixtures
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
def mock_loo():
    return LOOResult(
        loo=MagicMock(elpd_loo=-500.0, se=20.0, p_loo=50.0, pareto_k=np.array([0.1]), warning=None),
        elpd_loo=-500.0,
        se_elpd=20.0,
        p_loo=50.0,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )


@pytest.fixture
def mock_crps():
    return CRPSResult(
        mean_crps=3.5,
        crps_values=np.array([2.0, 3.0, 4.0, 5.0]),
        n_obs=4,
    )


@pytest.fixture
def simple_idata():
    import arviz as az

    return az.from_dict(
        posterior={
            "user_beta": np.random.default_rng(0).normal(size=(2, 50, 3)),
            "user_rho": np.random.default_rng(1).normal(size=(2, 50)),
        }
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
# Tests: aggregate_sensitivity_results
# ============================================================================


class TestAggregateSensitivityResultsCRPS:
    """Tests for aggregate with crps metric."""

    def test_crps_metric_populates_columns(self, mock_crps, passing_convergence):
        """CRPS metric aggregation should include mean_crps and n_obs."""
        results = {
            "v1": SensitivityResult(
                name="v1",
                config={},
                convergence=passing_convergence,
                crps=mock_crps,
            ),
        }
        df = aggregate_sensitivity_results(results, metric="crps")
        assert "mean_crps" in df.columns
        assert "n_obs" in df.columns
        assert df.loc["v1", "mean_crps"] == 3.5
        assert df.loc["v1", "n_obs"] == 4

    def test_crps_metric_none_crps(self, passing_convergence):
        """None crps should produce None values in columns."""
        results = {
            "v1": SensitivityResult(
                name="v1",
                config={},
                convergence=passing_convergence,
                crps=None,
            ),
        }
        df = aggregate_sensitivity_results(results, metric="crps")
        assert df.loc["v1", "mean_crps"] is None
        assert df.loc["v1", "n_obs"] is None


class TestAggregateSensitivityResultsConvergence:
    """Tests for aggregate with convergence metric."""

    def test_convergence_metric_includes_diag_fields(self, passing_convergence):
        """Convergence metric should include rhat_max, ess_bulk_min, etc."""
        results = {
            "v1": SensitivityResult(
                name="v1",
                config={},
                convergence=passing_convergence,
            ),
        }
        df = aggregate_sensitivity_results(results, metric="convergence")
        assert "convergence_passed" in df.columns
        assert "rhat_max" in df.columns
        assert "ess_bulk_min" in df.columns
        assert df.loc["v1", "convergence_passed"] == True  # noqa: E712

    def test_convergence_metric_none_convergence(self):
        """None convergence should produce None values."""
        results = {
            "v1": SensitivityResult(
                name="v1",
                config={},
                convergence=None,
            ),
        }
        df = aggregate_sensitivity_results(results, metric="convergence")
        assert df.loc["v1", "convergence_passed"] is None
        assert df.loc["v1", "rhat_max"] is None


class TestAggregateSensitivityResultsELPD:
    """Tests for ELPD metric sorting."""

    def test_elpd_sorted_descending(self, passing_convergence):
        """Results should be sorted by ELPD descending."""
        loo1 = LOOResult(
            loo=MagicMock(),
            elpd_loo=-600.0,
            se_elpd=25.0,
            p_loo=40.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        loo2 = LOOResult(
            loo=MagicMock(),
            elpd_loo=-400.0,
            se_elpd=20.0,
            p_loo=30.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        results = {
            "worse": SensitivityResult(
                name="worse",
                config={},
                convergence=passing_convergence,
                loo=loo1,
            ),
            "better": SensitivityResult(
                name="better",
                config={},
                convergence=passing_convergence,
                loo=loo2,
            ),
        }
        df = aggregate_sensitivity_results(results, metric="elpd")
        assert df.index.tolist() == ["better", "worse"]

    def test_empty_results_returns_empty_df(self):
        """Empty results dict should return empty DataFrame."""
        df = aggregate_sensitivity_results({}, metric="elpd")
        assert df.empty


# ============================================================================
# Tests: create_coefficient_comparison_df
# ============================================================================


class TestCoefficientComparisonNew:
    """Additional tests for create_coefficient_comparison_df."""

    def test_multiple_variants_and_params(self):
        """Multiple variants and params should generate rows for each pair."""
        coef1 = pd.DataFrame(
            {"mean": [0.5, 1.0], "hdi_3%": [0.1, 0.5], "hdi_97%": [0.9, 1.5]},
            index=["alpha", "beta"],
        )
        coef2 = pd.DataFrame(
            {"mean": [0.6, 1.1], "hdi_3%": [0.2, 0.6], "hdi_97%": [1.0, 1.6]},
            index=["alpha", "beta"],
        )
        results = {
            "v1": SensitivityResult(name="v1", config={}, coefficients=coef1),
            "v2": SensitivityResult(name="v2", config={}, coefficients=coef2),
        }
        df = create_coefficient_comparison_df(results, ["alpha", "beta"])
        assert len(df) == 4
        v1_alpha = df[(df["variant"] == "v1") & (df["param"] == "alpha")]
        assert v1_alpha.iloc[0]["mean"] == 0.5

    def test_missing_param_skipped(self):
        """Missing param in coefficients should be skipped."""
        coef = pd.DataFrame({"mean": [0.5]}, index=["alpha"])
        results = {
            "v1": SensitivityResult(name="v1", config={}, coefficients=coef),
        }
        df = create_coefficient_comparison_df(results, ["alpha", "missing"])
        assert len(df) == 1  # Only alpha

    def test_empty_coefficients_skipped(self):
        """Variant with empty coefficients should be skipped."""
        results = {
            "v1": SensitivityResult(name="v1", config={}, coefficients=pd.DataFrame()),
        }
        df = create_coefficient_comparison_df(results, ["alpha"])
        assert len(df) == 0


# ============================================================================
# Tests: run_prior_sensitivity with coefficient_vars
# ============================================================================


class TestRunPriorSensitivityWithCoefVars:
    """Test coefficient_vars parameter in run_prior_sensitivity."""

    def test_coefficient_vars_passed_to_extract(
        self, monkeypatch, simple_idata, passing_convergence
    ):
        """coefficient_vars should be passed to extract_coefficient_summary."""
        fit_result = _make_fit_result(simple_idata)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.fit_model",
            lambda *a, **kw: fit_result,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )

        results = run_prior_sensitivity(
            model=lambda: None,
            model_args={"y": np.zeros(5)},
            configs={"test": get_default_priors()},
            compute_loo_cv=False,
            coefficient_vars=["user_rho"],
        )
        # Should have coefficients with user_rho
        assert not results["test"].coefficients.empty


# ============================================================================
# Tests: run_threshold_sensitivity with defaults
# ============================================================================


class TestRunThresholdSensitivityDefaults:
    """Test default mcmc_config in threshold sensitivity."""

    def test_default_mcmc_config(self, monkeypatch, simple_idata, passing_convergence):
        """When mcmc_config is None, default MCMCConfig is used."""
        configs_used = []

        def fake_fit(model, args, config=None, progress_bar=True):
            configs_used.append(config)
            return _make_fit_result(simple_idata)

        monkeypatch.setattr("panelcast.pipelines.sensitivity.fit_model", fake_fit)
        monkeypatch.setattr(
            "panelcast.pipelines.sensitivity.check_convergence",
            lambda *a, **kw: passing_convergence,
        )

        def loader(t):
            return pd.DataFrame({"x": [1]}), {"y": np.zeros(1)}

        run_threshold_sensitivity(
            model=lambda: None,
            data_loader=loader,
            thresholds=(5,),
            mcmc_config=None,
            compute_loo_cv=False,
        )
        assert isinstance(configs_used[0], MCMCConfig)


# ============================================================================
# Tests: generate_oat_configs
# ============================================================================


class TestGenerateOatConfigsNew:
    """Additional tests for generate_oat_configs."""

    def test_default_parameters_and_multipliers(self):
        """Default call should use OAT_PARAMETERS and OAT_MULTIPLIERS."""
        configs = generate_oat_configs()
        expected_count = 1 + len(OAT_PARAMETERS) * len(OAT_MULTIPLIERS)
        assert len(configs) == expected_count
        assert "default" in configs

    def test_single_parameter_single_multiplier(self):
        """Single parameter with single multiplier should produce 2 configs."""
        configs = generate_oat_configs(parameters=("beta_scale",), multipliers=(2.0,))
        assert len(configs) == 2
        assert "default" in configs
        assert "beta_scale_x2" in configs

    def test_none_base_uses_defaults(self):
        """When base is None, get_default_priors() is used."""
        configs = generate_oat_configs(base=None, parameters=("beta_scale",), multipliers=(2.0,))
        default_val = get_default_priors().beta_scale
        assert configs["beta_scale_x2"].beta_scale == pytest.approx(default_val * 2.0)


# ============================================================================
# Tests: create_oat_summary_table
# ============================================================================


class TestCreateOatSummaryTableNew:
    """Additional tests for OAT summary table."""

    def test_eligible_sorted_by_abs_delta(self, passing_convergence):
        """Eligible variants should be sorted by |elpd_delta| descending."""
        base_loo = LOOResult(
            loo=MagicMock(),
            elpd_loo=-500.0,
            se_elpd=20.0,
            p_loo=50.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        big_delta_loo = LOOResult(
            loo=MagicMock(),
            elpd_loo=-600.0,
            se_elpd=25.0,
            p_loo=55.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        small_delta_loo = LOOResult(
            loo=MagicMock(),
            elpd_loo=-510.0,
            se_elpd=21.0,
            p_loo=51.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )

        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=base_loo,
            ),
            "beta_scale_x2": SensitivityResult(
                name="beta_scale_x2",
                config={},
                convergence=passing_convergence,
                loo=big_delta_loo,
            ),
            "beta_scale_x0.5": SensitivityResult(
                name="beta_scale_x0.5",
                config={},
                convergence=passing_convergence,
                loo=small_delta_loo,
            ),
        }
        df = create_oat_summary_table(results)

        eligible = df[df["eligible_for_ranking"]]
        assert len(eligible) == 3
        # First eligible by abs delta should be beta_scale_x2 (|delta|=100)
        # then beta_scale_x0.5 (|delta|=10)
        # then default (delta=0)
        first_variant = eligible.iloc[0]["variant"]
        assert first_variant == "beta_scale_x2"

    def test_no_loo_for_variant(self, passing_convergence, mock_loo):
        """Variant with no LOO should not be eligible."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=mock_loo,
            ),
            "beta_scale_x2": SensitivityResult(
                name="beta_scale_x2",
                config={},
                convergence=passing_convergence,
                loo=None,
            ),
        }
        df = create_oat_summary_table(results)
        beta_row = df[df["variant"] == "beta_scale_x2"]
        assert not beta_row.iloc[0]["eligible_for_ranking"]

    def test_convergence_flag_values(self, passing_convergence):
        """Convergence flag should be OK, FAILED, or MISSING."""
        failing = ConvergenceDiagnostics(
            rhat_max=1.05,
            ess_bulk_min=100,
            ess_tail_min=80,
            divergences=15,
            passed=False,
            failing_params=["p1"],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=passing_convergence,
                loo=None,
            ),
            "failing": SensitivityResult(
                name="failing",
                config={},
                convergence=failing,
                loo=None,
            ),
            "missing": SensitivityResult(
                name="missing",
                config={},
                convergence=None,
                loo=None,
            ),
        }
        df = create_oat_summary_table(results)
        default_row = df[df["variant"] == "default"]
        failing_row = df[df["variant"] == "failing"]
        missing_row = df[df["variant"] == "missing"]
        assert default_row.iloc[0]["convergence_flag"] == "OK"
        assert failing_row.iloc[0]["convergence_flag"] == "FAILED"
        assert missing_row.iloc[0]["convergence_flag"] == "MISSING"


# ============================================================================
# Tests: SensitivityResult dataclass
# ============================================================================


class TestSensitivityResult:
    """Tests for the SensitivityResult dataclass."""

    def test_default_fields(self):
        """Default field values should be None/empty."""
        result = SensitivityResult(name="test", config={})
        assert result.idata is None
        assert result.convergence is None
        assert result.loo is None
        assert result.crps is None
        assert result.coefficients.empty

    def test_all_fields_set(self, passing_convergence, mock_loo, mock_crps):
        """All fields should be settable."""
        coef = pd.DataFrame({"mean": [1.0]}, index=["alpha"])
        result = SensitivityResult(
            name="full",
            config={"key": "value"},
            idata=MagicMock(),
            convergence=passing_convergence,
            loo=mock_loo,
            crps=mock_crps,
            coefficients=coef,
        )
        assert result.name == "full"
        assert result.loo.elpd_loo == -500.0
        assert result.crps.mean_crps == 3.5
        assert not result.coefficients.empty
