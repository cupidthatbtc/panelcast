"""Unit tests for sensitivity analysis module.

Tests data structures, prior configurations, and aggregation functions.
Integration tests requiring actual model fitting are skipped by default.
Set RUN_SLOW_TESTS=1 to run them.
"""

import os

import numpy as np
import pandas as pd
import pytest

# Skip slow integration tests unless RUN_SLOW_TESTS is set
SKIP_SLOW = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow integration test - set RUN_SLOW_TESTS=1 to run",
)

from panelcast.evaluation.cv import LOOResult
from panelcast.evaluation.metrics import CRPSResult
from panelcast.models.bayes.diagnostics import ConvergenceDiagnostics
from panelcast.models.bayes.priors import PriorConfig, get_default_priors
from panelcast.pipelines.sensitivity import (
    OAT_MULTIPLIERS,
    OAT_PARAMETERS,
    PRIOR_CONFIGS,
    SensitivityResult,
    aggregate_sensitivity_results,
    create_coefficient_comparison_df,
    create_oat_summary_table,
    generate_oat_configs,
)
from panelcast.reporting.tables import create_sensitivity_summary_table

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_convergence():
    """Create a mock ConvergenceDiagnostics with passing values."""
    return ConvergenceDiagnostics(
        rhat_max=1.002,
        ess_bulk_min=2500,
        ess_tail_min=2200,
        divergences=0,
        passed=True,
        failing_params=[],
        summary_df=pd.DataFrame(
            {
                "mean": [0.5, 0.3],
                "sd": [0.1, 0.05],
                "r_hat": [1.001, 1.002],
                "ess_bulk": [2500, 2700],
                "ess_tail": [2200, 2400],
            },
            index=["param1", "param2"],
        ),
        rhat_threshold=1.01,
        ess_threshold=400,
    )


@pytest.fixture
def mock_loo_result():
    """Create a mock LOOResult with synthetic ELPD."""

    # Create a minimal mock that has the required attributes
    class MockELPDData:
        elpd_loo = -1234.5
        se = 45.2
        p_loo = 120.3
        pareto_k = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        warning = None

    return LOOResult(
        loo=MockELPDData(),
        elpd_loo=-1234.5,
        se_elpd=45.2,
        p_loo=120.3,
        n_high_pareto_k=1,  # One value > 0.7
        high_pareto_k_indices=np.array([5]),
        warning=None,
    )


@pytest.fixture
def mock_crps_result():
    """Create a mock CRPSResult."""
    return CRPSResult(
        mean_crps=5.8,
        crps_values=np.array([4.5, 5.2, 6.1, 5.8, 6.4]),
        n_obs=5,
    )


@pytest.fixture
def mock_coefficients():
    """Create a mock coefficient summary DataFrame."""
    return pd.DataFrame(
        {
            "mean": [0.15, 0.08, 12.5],
            "sd": [0.03, 0.02, 1.2],
            "hdi_3%": [0.09, 0.04, 10.1],
            "hdi_97%": [0.21, 0.12, 14.9],
        },
        index=["user_rho", "user_beta[0]", "sigma_obs"],
    )


@pytest.fixture
def mock_sensitivity_results(mock_convergence, mock_loo_result, mock_coefficients):
    """Create a set of mock SensitivityResults for testing aggregation."""
    results = {}

    # Default config result
    results["default"] = SensitivityResult(
        name="default",
        config={"priors": {"mu_artist_scale": 1.0}},
        idata=None,
        convergence=mock_convergence,
        loo=mock_loo_result,
        crps=None,
        coefficients=mock_coefficients,
    )

    # Diffuse config result - slightly different ELPD
    diffuse_loo = LOOResult(
        loo=type(
            "MockELPD",
            (),
            {
                "elpd_loo": -1256.3,
                "se": 48.1,
                "p_loo": 135.2,
                "pareto_k": np.array([0.3, 0.4, 0.5]),
                "warning": None,
            },
        )(),
        elpd_loo=-1256.3,
        se_elpd=48.1,
        p_loo=135.2,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )

    diffuse_coef = mock_coefficients.copy()
    diffuse_coef["mean"] = [0.142, 0.072, 13.1]
    diffuse_coef["hdi_3%"] = [0.078, 0.032, 10.7]
    diffuse_coef["hdi_97%"] = [0.206, 0.112, 15.5]

    results["diffuse"] = SensitivityResult(
        name="diffuse",
        config={"priors": {"mu_artist_scale": 5.0}},
        idata=None,
        convergence=mock_convergence,
        loo=diffuse_loo,
        crps=None,
        coefficients=diffuse_coef,
    )

    # Informative config result
    informative_loo = LOOResult(
        loo=type(
            "MockELPD",
            (),
            {
                "elpd_loo": -1240.1,
                "se": 44.8,
                "p_loo": 115.8,
                "pareto_k": np.array([0.2, 0.3]),
                "warning": None,
            },
        )(),
        elpd_loo=-1240.1,
        se_elpd=44.8,
        p_loo=115.8,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )

    informative_coef = mock_coefficients.copy()
    informative_coef["mean"] = [0.155, 0.085, 12.2]
    informative_coef["hdi_3%"] = [0.090, 0.045, 9.8]
    informative_coef["hdi_97%"] = [0.220, 0.125, 14.6]

    results["informative"] = SensitivityResult(
        name="informative",
        config={"priors": {"mu_artist_scale": 0.5}},
        idata=None,
        convergence=mock_convergence,
        loo=informative_loo,
        crps=None,
        coefficients=informative_coef,
    )

    return results


# ============================================================================
# Data Structure Tests
# ============================================================================


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_sensitivity_result_fields(self, mock_convergence, mock_loo_result, mock_coefficients):
        """Test that all fields are accessible."""
        result = SensitivityResult(
            name="test_variant",
            config={"threshold": 10},
            idata=None,
            convergence=mock_convergence,
            loo=mock_loo_result,
            crps=None,
            coefficients=mock_coefficients,
        )

        assert result.name == "test_variant"
        assert result.config == {"threshold": 10}
        assert result.idata is None
        assert result.convergence is mock_convergence
        assert result.loo is mock_loo_result
        assert result.crps is None
        assert result.coefficients is mock_coefficients

    def test_sensitivity_result_default_coefficients(self):
        """Test that coefficients default to empty DataFrame."""
        result = SensitivityResult(
            name="minimal",
            config={},
        )

        assert isinstance(result.coefficients, pd.DataFrame)
        assert result.coefficients.empty

    def test_sensitivity_result_all_none(self):
        """Test SensitivityResult with all optional fields as None."""
        result = SensitivityResult(
            name="none_test",
            config={"test": True},
            idata=None,
            convergence=None,
            loo=None,
            crps=None,
        )

        assert result.name == "none_test"
        assert result.convergence is None
        assert result.loo is None
        assert result.crps is None


class TestPriorConfigs:
    """Tests for prior configuration definitions."""

    def test_prior_configs_defined(self):
        """Test that PRIOR_CONFIGS has default, diffuse, and informative."""
        assert "default" in PRIOR_CONFIGS
        assert "diffuse" in PRIOR_CONFIGS
        assert "informative" in PRIOR_CONFIGS

    def test_prior_configs_count(self):
        """Test that we have exactly 3 prior configurations."""
        assert len(PRIOR_CONFIGS) == 3

    def test_diffuse_priors_wider(self):
        """Test that diffuse priors have larger scales than default."""
        default = PRIOR_CONFIGS["default"]
        diffuse = PRIOR_CONFIGS["diffuse"]

        assert diffuse.mu_artist_scale > default.mu_artist_scale
        assert diffuse.sigma_artist_scale > default.sigma_artist_scale
        assert diffuse.beta_scale > default.beta_scale

    def test_informative_priors_tighter(self):
        """Test that informative priors have smaller scales than default."""
        default = PRIOR_CONFIGS["default"]
        informative = PRIOR_CONFIGS["informative"]

        assert informative.mu_artist_scale < default.mu_artist_scale
        assert informative.sigma_artist_scale < default.sigma_artist_scale
        assert informative.beta_scale < default.beta_scale

    def test_prior_configs_all_positive_scales(self):
        """Test that all prior scales are positive."""
        for name, config in PRIOR_CONFIGS.items():
            assert config.mu_artist_scale > 0, f"{name}.mu_artist_scale"
            assert config.sigma_artist_scale > 0, f"{name}.sigma_artist_scale"
            assert config.beta_scale > 0, f"{name}.beta_scale"
            assert config.sigma_obs_scale > 0, f"{name}.sigma_obs_scale"


# ============================================================================
# Aggregation Function Tests
# ============================================================================


class TestAggregateSensitivityResults:
    """Tests for aggregate_sensitivity_results function."""

    def test_aggregate_sensitivity_results_returns_dataframe(self, mock_sensitivity_results):
        """Test that aggregate returns a DataFrame."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")
        assert isinstance(df, pd.DataFrame)

    def test_aggregate_sensitivity_results_columns_elpd(self, mock_sensitivity_results):
        """Test expected columns for ELPD aggregation."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")

        expected_columns = [
            "convergence_passed",
            "divergences",
            "rhat_max",
            "ess_bulk_min",
            "elpd",
            "elpd_se",
            "p_loo",
            "n_high_pareto_k",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_aggregate_sensitivity_results_rows(self, mock_sensitivity_results):
        """Test that all variants appear as rows."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")

        assert len(df) == 3
        assert "default" in df.index
        assert "diffuse" in df.index
        assert "informative" in df.index

    def test_aggregate_sensitivity_results_sorted_by_elpd(self, mock_sensitivity_results):
        """Test that results are sorted by ELPD (descending)."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")

        # Higher ELPD should come first
        elpd_values = df["elpd"].values
        assert elpd_values[0] >= elpd_values[1] >= elpd_values[2]

    def test_aggregate_sensitivity_results_convergence_metric(self, mock_sensitivity_results):
        """Test aggregation with convergence metric."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="convergence")

        assert "convergence_passed" in df.columns
        assert "divergences" in df.columns
        assert "rhat_max" in df.columns

    def test_aggregate_sensitivity_results_empty_dict(self):
        """Test aggregation with empty results dictionary."""
        df = aggregate_sensitivity_results({}, metric="elpd")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_aggregate_sensitivity_results_none_loo(self, mock_convergence, mock_coefficients):
        """Test aggregation when LOO is None."""
        results = {
            "no_loo": SensitivityResult(
                name="no_loo",
                config={},
                idata=None,
                convergence=mock_convergence,
                loo=None,
                crps=None,
                coefficients=mock_coefficients,
            )
        }

        df = aggregate_sensitivity_results(results, metric="elpd")
        assert df.loc["no_loo", "elpd"] is None


class TestCreateCoefficientComparisonDf:
    """Tests for create_coefficient_comparison_df function."""

    def test_create_coefficient_comparison_df_shape(self, mock_sensitivity_results):
        """Test correct shape for forest plot DataFrame."""
        params = ["user_rho", "user_beta[0]"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        # Should have 3 variants * 2 params = 6 rows
        assert len(df) == 6

    def test_create_coefficient_comparison_df_columns(self, mock_sensitivity_results):
        """Test expected columns for forest plot data."""
        params = ["user_rho"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        expected_columns = ["variant", "param", "mean", "lower", "upper"]
        assert list(df.columns) == expected_columns

    def test_create_coefficient_comparison_df_values(self, mock_sensitivity_results):
        """Test that values are correctly extracted."""
        params = ["user_rho"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        # Check default row
        default_row = df[df["variant"] == "default"]
        assert len(default_row) == 1
        assert default_row.iloc[0]["mean"] == 0.15
        assert default_row.iloc[0]["lower"] == 0.09
        assert default_row.iloc[0]["upper"] == 0.21

    def test_create_coefficient_comparison_df_missing_param(self, mock_sensitivity_results):
        """Test handling of missing parameter names."""
        params = ["nonexistent_param"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        # Should return empty DataFrame for missing params
        assert len(df) == 0

    def test_create_coefficient_comparison_df_empty_coefficients(
        self, mock_convergence, mock_loo_result
    ):
        """Test handling of empty coefficients DataFrame."""
        results = {
            "empty_coef": SensitivityResult(
                name="empty_coef",
                config={},
                idata=None,
                convergence=mock_convergence,
                loo=mock_loo_result,
                crps=None,
                coefficients=pd.DataFrame(),  # Empty
            )
        }

        params = ["user_rho"]
        df = create_coefficient_comparison_df(results, params)
        assert len(df) == 0


# ============================================================================
# OAT Config Generation Tests
# ============================================================================


class TestOATConfigs:
    def test_oat_configs_count(self):
        """Should produce 1 + len(params) * len(multipliers) configs."""
        configs = generate_oat_configs()
        expected = 1 + len(OAT_PARAMETERS) * len(OAT_MULTIPLIERS)
        assert len(configs) == expected

    def test_oat_configs_only_one_changed(self):
        """Each config should differ from default in exactly one field."""
        import dataclasses

        configs = generate_oat_configs()
        default = configs["default"]
        default_dict = dataclasses.asdict(default)
        for name, config in configs.items():
            if name == "default":
                continue
            config_dict = dataclasses.asdict(config)
            diffs = [k for k in default_dict if default_dict[k] != config_dict[k]]
            assert len(diffs) == 1, f"{name} differs in {diffs}"

    def test_oat_configs_multiplier_applied(self):
        """Verify value = default * multiplier."""
        import dataclasses

        configs = generate_oat_configs()
        default = configs["default"]
        default_dict = dataclasses.asdict(default)
        for name, config in configs.items():
            if name == "default":
                continue
            config_dict = dataclasses.asdict(config)
            # Parse multiplier from name
            parts = name.rsplit("_x", 1)
            mult = float(parts[1])
            param = parts[0]
            expected = default_dict[param] * mult
            np.testing.assert_allclose(config_dict[param], expected)


# ============================================================================
# OAT Summary Table Tests
# ============================================================================


class TestOATSummaryTable:
    @pytest.fixture
    def oat_results(self, mock_convergence, mock_loo_result, mock_coefficients):
        """Create mock OAT results."""
        results = {}
        results["default"] = SensitivityResult(
            name="default",
            config={},
            convergence=mock_convergence,
            loo=mock_loo_result,
            coefficients=mock_coefficients,
        )
        # A variant with slightly different ELPD
        better_loo = LOOResult(
            loo=type(
                "MockELPD",
                (),
                {
                    "elpd_loo": -1200.0,
                    "se": 40.0,
                    "p_loo": 100.0,
                    "pareto_k": np.array([0.2]),
                    "warning": None,
                },
            )(),
            elpd_loo=-1200.0,
            se_elpd=40.0,
            p_loo=100.0,
            n_high_pareto_k=0,
            high_pareto_k_indices=np.array([]),
            warning=None,
        )
        results["sigma_rw_scale_x2"] = SensitivityResult(
            name="sigma_rw_scale_x2",
            config={},
            convergence=mock_convergence,
            loo=better_loo,
            coefficients=mock_coefficients,
        )
        return results

    def test_oat_summary_delta(self, oat_results):
        """ELPD deltas should match expectations."""
        df = create_oat_summary_table(oat_results)
        variant_row = df[df["variant"] == "sigma_rw_scale_x2"].iloc[0]
        expected_delta = -1200.0 - (-1234.5)
        np.testing.assert_allclose(variant_row["elpd_delta"], expected_delta)

    def test_oat_summary_sorted(self, oat_results):
        """Eligible variants sorted by |delta|, ineligible at bottom."""
        df = create_oat_summary_table(oat_results)
        eligible = df[df["eligible_for_ranking"]]
        deltas = eligible["elpd_delta"].dropna().abs().values
        # Should be descending
        assert all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))

    def test_oat_summary_failed_convergence(self, mock_loo_result, mock_coefficients):
        """Failed variant gets convergence_flag='FAILED', eligible_for_ranking=False."""
        failed_conv = ConvergenceDiagnostics(
            rhat_max=1.05,
            ess_bulk_min=100,
            ess_tail_min=80,
            divergences=20,
            passed=False,
            failing_params=["param1"],
            summary_df=pd.DataFrame(),
            rhat_threshold=1.01,
            ess_threshold=400,
        )
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=ConvergenceDiagnostics(
                    rhat_max=1.002,
                    ess_bulk_min=2500,
                    ess_tail_min=2200,
                    divergences=0,
                    passed=True,
                    failing_params=[],
                    summary_df=pd.DataFrame(),
                    rhat_threshold=1.01,
                    ess_threshold=400,
                ),
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            ),
            "beta_scale_x5": SensitivityResult(
                name="beta_scale_x5",
                config={},
                convergence=failed_conv,
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            ),
        }
        df = create_oat_summary_table(results)
        failed_row = df[df["variant"] == "beta_scale_x5"].iloc[0]
        assert failed_row["convergence_flag"] == "FAILED"
        assert not failed_row["eligible_for_ranking"]
        assert pd.isna(failed_row["elpd_delta"]) or failed_row["elpd_delta"] is None

    def test_oat_summary_eligible_independent_of_flag(
        self, mock_convergence, mock_loo_result, mock_coefficients
    ):
        """Converged variants without comparable LOO should be ineligible."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=mock_convergence,
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            ),
            "mu_artist_scale_x2": SensitivityResult(
                name="mu_artist_scale_x2",
                config={},
                convergence=mock_convergence,
                loo=None,  # Missing LOO => not eligible for ranking
                coefficients=mock_coefficients,
            ),
        }
        df = create_oat_summary_table(results)
        variant_row = df[df["variant"] == "mu_artist_scale_x2"].iloc[0]
        assert variant_row["convergence_flag"] == "OK"
        assert not variant_row["eligible_for_ranking"]
        assert pd.isna(variant_row["elpd_delta"]) or variant_row["elpd_delta"] is None

    def test_oat_summary_missing_convergence_flag(self, mock_loo_result, mock_coefficients):
        """Missing convergence diagnostics should be marked MISSING."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=None,  # Missing diagnostics
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            ),
        }
        df = create_oat_summary_table(results)
        row = df.iloc[0]
        assert row["convergence_flag"] == "MISSING"
        assert not row["eligible_for_ranking"]
        assert pd.isna(row["elpd_delta"]) or row["elpd_delta"] is None


class TestSensitivitySummaryFormatting:
    def test_create_sensitivity_summary_table_tolerates_string_numbers(self):
        """Formatting should tolerate numeric values stored as strings."""
        raw = pd.DataFrame(
            {
                "variant": ["default"],
                "parameter": ["baseline"],
                "multiplier": [1.0],
                "elpd": ["-1234.5"],
                "elpd_delta": [None],
                "elpd_se": ["45.2"],
                "convergence_flag": ["OK"],
            }
        )
        formatted = create_sensitivity_summary_table(raw)
        assert formatted.loc[0, "ELPD"] == "-1234.5"
        assert formatted.loc[0, "SE"] == "45.2"
        assert formatted.loc[0, "ΔELPD"] == "—"


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestSensitivityResultDefaults:
    """Tests for SensitivityResult default values and edge cases."""

    def test_default_idata_is_none(self):
        """Default idata is None."""
        result = SensitivityResult(name="test", config={})
        assert result.idata is None

    def test_default_convergence_is_none(self):
        """Default convergence is None."""
        result = SensitivityResult(name="test", config={})
        assert result.convergence is None

    def test_default_loo_is_none(self):
        """Default loo is None."""
        result = SensitivityResult(name="test", config={})
        assert result.loo is None

    def test_default_crps_is_none(self):
        """Default crps is None."""
        result = SensitivityResult(name="test", config={})
        assert result.crps is None

    def test_config_can_be_nested_dict(self):
        """Config can be a nested dictionary."""
        config = {"priors": {"mu_artist_scale": 1.0, "beta_scale": 2.0}}
        result = SensitivityResult(name="nested", config=config)
        assert result.config["priors"]["mu_artist_scale"] == 1.0


class TestPriorConfigDetails:
    """Detailed tests for PRIOR_CONFIGS values."""

    def test_default_priors_match_get_default_priors(self):
        """PRIOR_CONFIGS['default'] matches get_default_priors()."""
        from dataclasses import asdict

        default_config = PRIOR_CONFIGS["default"]
        expected = get_default_priors()
        assert asdict(default_config) == asdict(expected)

    def test_all_configs_have_all_fields(self):
        """All prior configs have the same set of fields."""
        from dataclasses import fields

        expected_fields = {f.name for f in fields(PriorConfig)}
        for name, config in PRIOR_CONFIGS.items():
            config_fields = {f.name for f in fields(config)}
            assert config_fields == expected_fields, f"{name} missing fields"

    def test_diffuse_sigma_obs_wider(self):
        """Diffuse has wider sigma_obs_scale."""
        assert PRIOR_CONFIGS["diffuse"].sigma_obs_scale > PRIOR_CONFIGS["default"].sigma_obs_scale

    def test_informative_sigma_obs_tighter(self):
        """Informative has tighter sigma_obs_scale."""
        assert (
            PRIOR_CONFIGS["informative"].sigma_obs_scale < PRIOR_CONFIGS["default"].sigma_obs_scale
        )

    def test_diffuse_rho_scale_wider(self):
        """Diffuse has wider rho_scale."""
        assert PRIOR_CONFIGS["diffuse"].rho_scale > PRIOR_CONFIGS["default"].rho_scale

    def test_informative_rho_scale_tighter(self):
        """Informative has tighter rho_scale."""
        assert PRIOR_CONFIGS["informative"].rho_scale < PRIOR_CONFIGS["default"].rho_scale


class TestAggregateSensitivityResultsEdgeCases:
    """Edge cases for aggregate_sensitivity_results."""

    def test_aggregate_with_crps_metric(self, mock_sensitivity_results, mock_crps_result):
        """Aggregation with crps metric."""
        # Add CRPS to one result
        mock_sensitivity_results["default"].crps = mock_crps_result
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="crps")
        assert isinstance(df, pd.DataFrame)

    def test_aggregate_single_result(self, mock_convergence, mock_loo_result, mock_coefficients):
        """Aggregation with a single result."""
        results = {
            "only_one": SensitivityResult(
                name="only_one",
                config={},
                convergence=mock_convergence,
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            )
        }
        df = aggregate_sensitivity_results(results, metric="elpd")
        assert len(df) == 1
        assert "only_one" in df.index

    def test_aggregate_preserves_convergence_info(self, mock_sensitivity_results):
        """Aggregation preserves convergence diagnostics."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")
        assert all(df["convergence_passed"]), "All mocks should pass"
        assert all(df["divergences"] == 0), "All mocks have zero divergences"


class TestCoefficientComparisonEdgeCases:
    """Edge cases for create_coefficient_comparison_df."""

    def test_multiple_params_order(self, mock_sensitivity_results):
        """Multiple params maintain correct order."""
        params = ["user_rho", "sigma_obs"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)
        # Each variant has both params
        for variant in ["default", "diffuse", "informative"]:
            variant_rows = df[df["variant"] == variant]
            assert set(variant_rows["param"]) == {"user_rho", "sigma_obs"}

    def test_empty_results_dict(self):
        """Empty results dict produces empty DataFrame."""
        df = create_coefficient_comparison_df({}, ["user_rho"])
        assert len(df) == 0

    def test_single_param_single_variant(
        self, mock_convergence, mock_loo_result, mock_coefficients
    ):
        """Single param from single variant produces one row."""
        results = {
            "single": SensitivityResult(
                name="single",
                config={},
                convergence=mock_convergence,
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            )
        }
        df = create_coefficient_comparison_df(results, ["user_rho"])
        assert len(df) == 1
        assert df.iloc[0]["variant"] == "single"
        assert df.iloc[0]["param"] == "user_rho"


class TestOATConfigsEdgeCases:
    """Edge cases for OAT config generation."""

    def test_oat_default_matches_get_default_priors(self):
        """OAT default config matches get_default_priors()."""
        from dataclasses import asdict

        configs = generate_oat_configs()
        assert asdict(configs["default"]) == asdict(get_default_priors())

    def test_oat_config_names_contain_multiplier(self):
        """Non-default OAT names contain '_x' multiplier suffix."""
        configs = generate_oat_configs()
        for name in configs:
            if name == "default":
                continue
            assert "_x" in name, f"{name} should contain '_x' multiplier"

    def test_oat_multipliers_are_positive(self):
        """All OAT multipliers are positive."""
        for mult in OAT_MULTIPLIERS:
            assert mult > 0

    def test_oat_parameters_are_valid_fields(self):
        """All OAT parameters are valid PriorConfig fields."""
        from dataclasses import fields

        valid_fields = {f.name for f in fields(PriorConfig)}
        for param in OAT_PARAMETERS:
            assert param in valid_fields, f"{param} not a PriorConfig field"


class TestOATSummaryEdgeCases:
    """Edge cases for OAT summary table."""

    def test_single_default_only(self, mock_convergence, mock_loo_result, mock_coefficients):
        """OAT summary with only default result."""
        results = {
            "default": SensitivityResult(
                name="default",
                config={},
                convergence=mock_convergence,
                loo=mock_loo_result,
                coefficients=mock_coefficients,
            ),
        }
        df = create_oat_summary_table(results)
        assert len(df) == 1
        default_row = df[df["variant"] == "default"].iloc[0]
        # Default should have elpd_delta of 0 or NaN (it's the reference)
        assert pd.isna(default_row["elpd_delta"]) or default_row["elpd_delta"] == 0.0


# ============================================================================
# Integration Tests (Skip by default - require model fitting)
# ============================================================================
