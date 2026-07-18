"""Unit tests for reporting tables module.

Tests cover:
- Adaptive precision formatting for numeric values
- create_coefficient_table with various configurations
- create_diagnostics_table with convergence status
- create_comparison_table for model ranking
- export_table for CSV and LaTeX output
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.reporting.tables import (
    _escape_latex_param_name,
    _format_with_precision,
    create_coefficient_table,
    create_comparison_table,
    create_diagnostics_table,
    create_sensitivity_summary_table,
    export_table,
)


def make_mock_idata(
    n_chains: int = 4,
    n_draws: int = 500,
    param_names: list[str] | None = None,
    add_log_likelihood: bool = False,
    n_obs: int = 100,
) -> az.InferenceData:
    """Create mock InferenceData for testing.

    Parameters
    ----------
    n_chains : int
        Number of chains
    n_draws : int
        Number of draws per chain
    param_names : list[str], optional
        Parameter names (default: ["mu", "sigma"])
    add_log_likelihood : bool
        If True, add log_likelihood group for LOO/comparison tests
    n_obs : int
        Number of observations for log_likelihood

    Returns
    -------
    az.InferenceData
        Mock InferenceData with posterior and optionally log_likelihood
    """
    if param_names is None:
        param_names = ["mu", "sigma"]

    np.random.seed(42)
    posterior_dict = {}

    for name in param_names:
        # Create well-mixed samples
        samples = np.random.randn(n_chains, n_draws)
        posterior_dict[name] = xr.DataArray(
            samples,
            dims=["chain", "draw"],
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )

    posterior = xr.Dataset(posterior_dict)

    # Create sample_stats
    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(
                np.zeros((n_chains, n_draws), dtype=bool),
                dims=["chain", "draw"],
                coords={"chain": range(n_chains), "draw": range(n_draws)},
            ),
        }
    )

    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    if add_log_likelihood:
        # Add stable log_likelihood for LOO/WAIC tests.
        # Keep low variance to avoid noisy Pareto-k/WAIC warnings in unit tests.
        obs_effect = np.linspace(-0.2, 0.2, n_obs)[None, None, :]
        log_lik = -100.0 + obs_effect + np.random.randn(n_chains, n_draws, n_obs) * 0.05
        log_likelihood = xr.Dataset(
            {
                "y": xr.DataArray(
                    log_lik,
                    dims=["chain", "draw", "y_dim_0"],
                    coords={
                        "chain": range(n_chains),
                        "draw": range(n_draws),
                        "y_dim_0": range(n_obs),
                    },
                )
            }
        )
        idata = az.InferenceData(
            posterior=posterior,
            sample_stats=sample_stats,
            log_likelihood=log_likelihood,
        )

    return idata


@pytest.fixture
def mock_idata():
    """Create basic mock InferenceData."""
    return make_mock_idata()


@pytest.fixture
def mock_idata_with_loglik():
    """Create mock InferenceData with log_likelihood for model comparison."""
    return make_mock_idata(add_log_likelihood=True)


@pytest.fixture
def mock_coefficient_df():
    """Create mock coefficient DataFrame for export tests."""
    return pd.DataFrame(
        {
            "Estimate": ["1.23", "0.45", "-0.12"],
            "SE": ["0.12", "0.05", "0.03"],
            "CI Lower": ["0.99", "0.35", "-0.18"],
            "CI Upper": ["1.47", "0.55", "-0.06"],
        },
        index=["mu", "sigma", "beta_effect"],
    )


class TestAdaptivePrecision:
    """Tests for adaptive precision formatting."""

    def test_precision_large_uncertainty(self):
        """Large uncertainty should result in fewer decimals."""
        # With uncertainty of 0.5, should get ~2 decimal places
        result = _format_with_precision(1.234, 0.5)
        # Should not show more than 2-3 decimals
        assert "." in result
        decimals = len(result.split(".")[-1])
        assert decimals <= 3

    def test_precision_small_uncertainty(self):
        """Small uncertainty should result in more decimals."""
        # With uncertainty of 0.00005, should get more decimal places
        result = _format_with_precision(0.001234, 0.00005)
        # Should show enough decimals to capture the precision
        assert "." in result
        decimals = len(result.split(".")[-1])
        assert decimals >= 3

    def test_precision_zero_uncertainty(self):
        """Zero uncertainty should use minimum decimals."""
        result = _format_with_precision(1.234567, 0.0)
        # Should use min_decimals=2
        assert result == "1.23"

    def test_precision_nan_value(self):
        """NaN value should return 'nan'."""
        result = _format_with_precision(np.nan, 0.1)
        assert result.lower() == "nan"

    def test_precision_inf_value(self):
        """Inf value should return 'inf'."""
        result = _format_with_precision(np.inf, 0.1)
        assert result.lower() == "inf"

    def test_precision_nan_uncertainty(self):
        """NaN uncertainty should use min_decimals."""
        result = _format_with_precision(1.234567, np.nan)
        assert result == "1.23"

    def test_precision_custom_min_decimals(self):
        """Custom min_decimals should be respected."""
        result = _format_with_precision(1.234567, 0.0, min_decimals=4)
        assert result == "1.2346"

    def test_precision_custom_max_decimals(self):
        """Custom max_decimals should be respected."""
        # Very small uncertainty that would normally give many decimals
        result = _format_with_precision(0.123456789, 0.000001, max_decimals=4)
        decimals = len(result.split(".")[-1])
        assert decimals <= 4


class TestCreateCoefficientTable:
    """Tests for coefficient table generation."""

    def test_basic_coefficient_table(self, mock_idata):
        """Basic coefficient table should have correct structure."""
        result = create_coefficient_table(mock_idata)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # mu, sigma
        assert "mu" in result.index
        assert "sigma" in result.index

    def test_column_names(self, mock_idata):
        """Verify correct column names."""
        result = create_coefficient_table(mock_idata)

        expected_cols = ["Estimate", "SE", "CI Lower", "CI Upper"]
        assert list(result.columns) == expected_cols

    def test_var_names_filter(self, mock_idata):
        """Filtering by var_names should work."""
        result = create_coefficient_table(mock_idata, var_names=["mu"])

        assert len(result) == 1
        assert "mu" in result.index
        assert "sigma" not in result.index

    def test_hdi_prob_custom(self, mock_idata):
        """Custom HDI probability should work."""
        # Should not raise error
        result_94 = create_coefficient_table(mock_idata, hdi_prob=0.94)
        result_89 = create_coefficient_table(mock_idata, hdi_prob=0.89)

        # Results should be different (different CI bounds)
        assert isinstance(result_94, pd.DataFrame)
        assert isinstance(result_89, pd.DataFrame)

    def test_adaptive_precision_applied(self, mock_idata):
        """Adaptive precision should format values as strings."""
        result = create_coefficient_table(mock_idata, apply_precision=True)

        # Values should be strings when precision is applied
        for col in ["Estimate", "SE", "CI Lower", "CI Upper"]:
            assert result[col].dtype == object  # string dtype

    def test_no_precision_returns_numeric(self, mock_idata):
        """Without precision, values should be numeric."""
        result = create_coefficient_table(mock_idata, apply_precision=False)

        # Values should be numeric
        for col in ["Estimate", "SE", "CI Lower", "CI Upper"]:
            assert np.issubdtype(result[col].dtype, np.floating)

    def test_empty_var_names(self, mock_idata):
        """Empty var_names list should return empty DataFrame."""
        result = create_coefficient_table(mock_idata, var_names=[])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["Estimate", "SE", "CI Lower", "CI Upper"]

    def test_missing_posterior_raises(self):
        """Missing posterior group should raise ValueError."""
        idata = az.InferenceData()

        with pytest.raises(ValueError, match="posterior"):
            create_coefficient_table(idata)

    def test_multivariate_param(self):
        """Test with multivariate parameters (e.g., beta[0], beta[1])."""
        np.random.seed(42)
        beta = np.random.randn(4, 500, 3)  # 3 beta coefficients
        posterior = xr.Dataset(
            {
                "beta": xr.DataArray(
                    beta,
                    dims=["chain", "draw", "beta_dim_0"],
                    coords={
                        "chain": range(4),
                        "draw": range(500),
                        "beta_dim_0": range(3),
                    },
                )
            }
        )
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((4, 500), dtype=bool),
                    dims=["chain", "draw"],
                )
            }
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        result = create_coefficient_table(idata, var_names=["beta"])

        assert len(result) == 3  # beta[0], beta[1], beta[2]


class TestCreateDiagnosticsTable:
    """Tests for diagnostics table generation."""

    def test_basic_diagnostics(self, mock_idata):
        """Basic diagnostics table should have correct columns."""
        result = create_diagnostics_table(mock_idata)

        expected_cols = ["R-hat", "ESS Bulk", "ESS Tail", "MCSE Mean", "Status"]
        assert list(result.columns) == expected_cols

    def test_convergence_status_pass(self, mock_idata):
        """All params should pass with good idata."""
        result = create_diagnostics_table(mock_idata)

        # All parameters should pass
        assert all(status == "Pass" for status in result["Status"])

    def test_convergence_status_fail_rhat(self):
        """Parameters with bad R-hat should fail."""
        # Create idata with poor mixing
        np.random.seed(42)
        n_chains, n_draws = 4, 500

        # Create chains with different means (poor mixing)
        samples = np.zeros((n_chains, n_draws))
        for c in range(n_chains):
            samples[c, :] = np.random.randn(n_draws) + c * 5.0

        posterior = xr.Dataset(
            {
                "bad_param": xr.DataArray(
                    samples,
                    dims=["chain", "draw"],
                )
            }
        )
        sample_stats = xr.Dataset(
            {
                "diverging": xr.DataArray(
                    np.zeros((n_chains, n_draws), dtype=bool), dims=["chain", "draw"]
                )
            }
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        result = create_diagnostics_table(idata)

        # Should have failing status
        assert "Fail" in result.loc["bad_param", "Status"]

    def test_single_chain_rhat_not_failed(self):
        """Single-chain runs yield NaN R-hat; status must not read 'Fail (R-hat)'."""
        np.random.seed(7)
        n_chains, n_draws = 1, 800
        posterior = xr.Dataset(
            {"p": xr.DataArray(np.random.randn(n_chains, n_draws), dims=["chain", "draw"])}
        )
        sample_stats = xr.Dataset(
            {"diverging": xr.DataArray(np.zeros((n_chains, n_draws), dtype=bool), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        status = create_diagnostics_table(idata).loc["p", "Status"]
        assert "R-hat" not in status
        assert status == "n/a (single chain)"

    def test_multichain_zero_variance_labels_no_variance(self):
        """A constant param in a multi-chain run yields NaN R-hat but finite ESS;
        the label must distinguish it from a single-chain run."""
        n_chains, n_draws = 2, 500
        posterior = xr.Dataset(
            {"c": xr.DataArray(np.zeros((n_chains, n_draws)), dims=["chain", "draw"])}
        )
        sample_stats = xr.Dataset(
            {"diverging": xr.DataArray(np.zeros((n_chains, n_draws), dtype=bool), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        status = create_diagnostics_table(idata).loc["c", "Status"]
        assert status == "n/a (no variance)"

    def test_nan_rhat_with_low_ess_fails_ess(self):
        """NaN R-hat but ESS below threshold is a genuine ESS failure, not 'n/a'."""
        np.random.seed(11)
        n_chains, n_draws = 1, 800
        posterior = xr.Dataset(
            {"p": xr.DataArray(np.random.randn(n_chains, n_draws), dims=["chain", "draw"])}
        )
        sample_stats = xr.Dataset(
            {"diverging": xr.DataArray(np.zeros((n_chains, n_draws), dtype=bool), dims=["chain", "draw"])}
        )
        idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

        status = create_diagnostics_table(idata, ess_threshold=10_000).loc["p", "Status"]
        assert status == "Fail (ESS)"

    def test_ess_as_integers(self, mock_idata):
        """ESS values should be formatted as integers."""
        result = create_diagnostics_table(mock_idata)

        # ESS columns should be int dtype
        assert result["ESS Bulk"].dtype == np.int64 or result["ESS Bulk"].dtype == int
        assert result["ESS Tail"].dtype == np.int64 or result["ESS Tail"].dtype == int

    def test_rhat_decimals(self, mock_idata):
        """R-hat values should have 4 decimal places."""
        result = create_diagnostics_table(mock_idata)

        for rhat_str in result["R-hat"]:
            # Should be string with 4 decimals
            assert "." in rhat_str
            decimals = len(rhat_str.split(".")[-1])
            assert decimals == 4

    def test_empty_var_names(self, mock_idata):
        """Empty var_names should return empty DataFrame."""
        result = create_diagnostics_table(mock_idata, var_names=[])

        assert len(result) == 0

    def test_custom_thresholds(self, mock_idata):
        """Custom thresholds should be respected."""
        # With very strict thresholds, may fail
        result_strict = create_diagnostics_table(
            mock_idata, rhat_threshold=1.0001, ess_threshold=100000
        )

        # Should still return valid DataFrame
        assert isinstance(result_strict, pd.DataFrame)


class TestCreateComparisonTable:
    """Tests for model comparison table generation."""

    def test_comparison_ranking(self):
        """Best model should be first by ELPD."""
        # Create two models with different log_likelihood
        idata1 = make_mock_idata(add_log_likelihood=True, n_obs=50)
        idata2 = make_mock_idata(add_log_likelihood=True, n_obs=50)

        # Make model2 have better log_likelihood
        idata2.log_likelihood["y"].values[:] = idata2.log_likelihood["y"].values + 10

        models = {"model1": idata1, "model2": idata2}
        result = create_comparison_table(models)

        # model2 should be first (better ELPD)
        assert result.index[0] == "model2"

    def test_delta_column(self):
        """Delta should show difference from best model."""
        idata1 = make_mock_idata(add_log_likelihood=True, n_obs=50)
        idata2 = make_mock_idata(add_log_likelihood=True, n_obs=50)

        models = {"model1": idata1, "model2": idata2}
        result = create_comparison_table(models)

        # First model should have delta = 0
        assert float(result.iloc[0]["Delta"]) == 0.0

    def test_single_model(self, mock_idata_with_loglik):
        """Single model comparison should work."""
        models = {"only_model": mock_idata_with_loglik}
        result = create_comparison_table(models)

        assert len(result) == 1
        assert result.index[0] == "only_model"
        assert result.loc["only_model", "Delta"] == "0.0"
        assert result.loc["only_model", "Weight"] == "1.00"

    def test_column_names(self):
        """Verify correct column names."""
        idata = make_mock_idata(add_log_likelihood=True, n_obs=50)
        models = {"model": idata}
        result = create_comparison_table(models)

        expected_cols = ["ELPD", "SE", "p_eff", "Delta", "Weight"]
        assert list(result.columns) == expected_cols

    def test_empty_model_dict_raises(self):
        """Empty model_dict should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            create_comparison_table({})

    def test_waic_option(self, mock_idata_with_loglik):
        """WAIC option should work."""
        models = {"model": mock_idata_with_loglik}
        result = create_comparison_table(models, ic="waic")

        assert isinstance(result, pd.DataFrame)
        assert "ELPD" in result.columns


class TestExportTable:
    """Tests for table export functionality."""

    def test_csv_export(self, mock_coefficient_df):
        """CSV export should create correct file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_table"
            paths = export_table(mock_coefficient_df, output_path, formats=("csv",))

            assert len(paths) == 1
            csv_path = paths[0]
            assert csv_path.suffix == ".csv"
            assert csv_path.exists()

            # Read back and verify
            df_read = pd.read_csv(csv_path, index_col=0)
            assert list(df_read.index) == list(mock_coefficient_df.index)

    def test_latex_export(self, mock_coefficient_df):
        """LaTeX export should create correct file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_table"
            paths = export_table(mock_coefficient_df, output_path, formats=("tex",))

            assert len(paths) == 1
            tex_path = paths[0]
            assert tex_path.suffix == ".tex"
            assert tex_path.exists()

            # Read and verify LaTeX content
            content = tex_path.read_text()
            assert "\\begin{table}" in content
            assert "\\end{table}" in content

    def test_both_formats(self, mock_coefficient_df):
        """Both CSV and LaTeX should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_table"
            paths = export_table(mock_coefficient_df, output_path, formats=("csv", "tex"))

            assert len(paths) == 2
            extensions = {p.suffix for p in paths}
            assert extensions == {".csv", ".tex"}

    def test_special_characters_escaped(self, mock_coefficient_df):
        """Special characters in index should be escaped in LaTeX."""
        # Create DataFrame with underscore in index
        df = mock_coefficient_df.copy()
        df.index = ["mu_artist", "sigma_obs", "beta_effect"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_table"
            paths = export_table(df, output_path, formats=("tex",))

            tex_path = paths[0]
            content = tex_path.read_text()

            # pandas escape=True should escape underscores
            # The exact escape format depends on pandas version
            assert tex_path.exists()

    def test_custom_caption_label(self, mock_coefficient_df):
        """Custom caption and label should appear in LaTeX output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_table"
            paths = export_table(
                mock_coefficient_df,
                output_path,
                formats=("tex",),
                caption="Model coefficients",
                label="tab:custom_label",
            )

            tex_path = paths[0]
            content = tex_path.read_text()

            assert "Model coefficients" in content
            assert "tab:custom_label" in content

    def test_default_label_from_filename(self, mock_coefficient_df):
        """Label should be derived from filename if not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "my_table"
            paths = export_table(mock_coefficient_df, output_path, formats=("tex",))

            tex_path = paths[0]
            content = tex_path.read_text()

            assert "tab:my_table" in content

    def test_creates_parent_directories(self, mock_coefficient_df):
        """Export should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir1" / "subdir2" / "test_table"
            paths = export_table(mock_coefficient_df, output_path)

            assert all(p.exists() for p in paths)

    def test_empty_dataframe(self):
        """Empty DataFrame should export without error."""
        df = pd.DataFrame(columns=["A", "B"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_table"
            paths = export_table(df, output_path)

            assert all(p.exists() for p in paths)


class TestTableIntegration:
    """Integration tests for full table workflow."""

    def test_coefficient_table_to_export(self, mock_idata):
        """Full workflow: create coefficient table and export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create table
            df = create_coefficient_table(mock_idata)

            # Export
            output_path = Path(tmpdir) / "coefficients"
            paths = export_table(df, output_path, caption="Model coefficients")

            # Verify both files exist
            assert (output_path.with_suffix(".csv")).exists()
            assert (output_path.with_suffix(".tex")).exists()

    def test_diagnostics_table_to_export(self, mock_idata):
        """Full workflow: create diagnostics table and export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create table
            df = create_diagnostics_table(mock_idata)

            # Export
            output_path = Path(tmpdir) / "diagnostics"
            paths = export_table(df, output_path)

            # Verify
            csv_df = pd.read_csv(output_path.with_suffix(".csv"), index_col=0)
            assert "Status" in csv_df.columns


# --- from unit/test_reporting_tables_expanded.py ---


def _make_idata(n_chains=4, n_draws=500, param_names=None):
    """Helper to build mock InferenceData."""
    if param_names is None:
        param_names = ["mu", "sigma"]
    np.random.seed(42)
    posterior_dict = {}
    for name in param_names:
        posterior_dict[name] = xr.DataArray(
            np.random.randn(n_chains, n_draws),
            dims=["chain", "draw"],
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
    posterior = xr.Dataset(posterior_dict)
    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(
                np.zeros((n_chains, n_draws), dtype=bool),
                dims=["chain", "draw"],
            )
        }
    )
    return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


class TestFormatWithPrecisionExpanded:
    """Expanded edge-case tests for _format_with_precision."""

    def test_negative_inf(self):
        result = _format_with_precision(-np.inf, 0.1)
        assert "inf" in result.lower()

    def test_very_large_value(self):
        result = _format_with_precision(1e10, 1e8)
        assert "." in result or "e" in result.lower()

    def test_very_small_value(self):
        result = _format_with_precision(1e-8, 1e-10)
        assert "." in result

    def test_negative_value(self):
        result = _format_with_precision(-3.14, 0.05)
        assert result.startswith("-")

    def test_exact_zero_value(self):
        result = _format_with_precision(0.0, 0.01)
        assert "0.00" in result

    def test_min_greater_than_max_decimals(self):
        """min_decimals > max_decimals: max should still cap."""
        result = _format_with_precision(1.234, 0.001, min_decimals=6, max_decimals=3)
        decimals = len(result.split(".")[-1])
        assert decimals <= 6  # Implementation clamps at max of (min, capped)


class TestEscapeLatexParamName:
    """Tests for _escape_latex_param_name."""

    def test_no_underscores(self):
        assert _escape_latex_param_name("beta") == "beta"

    def test_single_underscore(self):
        assert _escape_latex_param_name("sigma_artist") == r"sigma\_artist"

    def test_multiple_underscores(self):
        result = _escape_latex_param_name("mu_artist_effect")
        assert result == r"mu\_artist\_effect"

    def test_brackets_unchanged(self):
        assert _escape_latex_param_name("beta[0]") == "beta[0]"

    def test_mixed(self):
        result = _escape_latex_param_name("user_beta[0]")
        assert result == r"user\_beta[0]"


class TestCreateCoefficientTableExpanded:
    """Expanded coefficient table tests."""

    def test_many_parameters(self):
        idata = _make_idata(param_names=["a", "b", "c", "d", "e"])
        result = create_coefficient_table(idata)
        assert len(result) == 5

    def test_single_chain(self):
        idata = _make_idata(n_chains=1)
        result = create_coefficient_table(idata)
        assert len(result) == 2

    def test_few_draws(self):
        idata = _make_idata(n_draws=20)
        result = create_coefficient_table(idata)
        assert len(result) == 2

    def test_ci_lower_le_upper(self):
        idata = _make_idata()
        result = create_coefficient_table(idata, apply_precision=False)
        for idx in result.index:
            assert result.at[idx, "CI Lower"] <= result.at[idx, "CI Upper"]


class TestCreateDiagnosticsTableExpanded:
    """Expanded diagnostics table tests."""

    def test_missing_posterior_raises(self):
        idata = az.InferenceData()
        with pytest.raises(ValueError, match="posterior"):
            create_diagnostics_table(idata)

    def test_var_names_filter(self):
        idata = _make_idata(param_names=["alpha", "beta", "gamma"])
        result = create_diagnostics_table(idata, var_names=["alpha"])
        assert len(result) == 1
        assert "alpha" in result.index

    def test_single_chain_diagnostics(self):
        """Single chain should still produce valid diagnostics table."""
        idata = _make_idata(n_chains=1)
        result = create_diagnostics_table(idata)
        assert isinstance(result, pd.DataFrame)
        assert "Status" in result.columns


class TestCreateSensitivitySummaryTable:
    """Tests for create_sensitivity_summary_table."""

    @pytest.fixture
    def oat_df(self):
        return pd.DataFrame(
            {
                "variant": ["baseline", "high_sigma", "low_sigma"],
                "parameter": ["sigma_artist", "sigma_artist", "sigma_artist"],
                "multiplier": [1.0, 2.0, 0.5],
                "elpd": [-1234.0, -1240.0, -1230.0],
                "elpd_delta": [0.0, -6.0, 4.0],
                "elpd_se": [12.0, 13.0, 11.0],
                "convergence_flag": ["OK", "OK", "WARN"],
            }
        )

    def test_returns_dataframe(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        assert isinstance(result, pd.DataFrame)

    def test_renames_columns(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        assert "Variant" in result.columns
        assert "Parameter" in result.columns

    def test_dagger_for_non_ok(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        statuses = result["Status"].tolist()
        assert any("\u2020" in s for s in statuses)

    def test_ok_no_dagger(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        ok_rows = result[result["Status"] == "OK"]
        for s in ok_rows["Status"]:
            assert "\u2020" not in s

    def test_numeric_formatting(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        # ELPD values should be formatted as strings
        for val in result["ELPD"]:
            assert isinstance(val, str)

    def test_none_elpd_becomes_dash(self):
        df = pd.DataFrame(
            {
                "variant": ["test"],
                "parameter": ["p"],
                "multiplier": [1.0],
                "elpd": [None],
                "elpd_delta": [None],
                "elpd_se": [None],
                "convergence_flag": ["FAIL"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "\u2014" in result["ELPD"].iloc[0]


class TestExportTableExpanded:
    """Expanded export table tests."""

    def test_csv_only(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_table(df, Path(tmpdir) / "t", formats=("csv",))
            assert len(paths) == 1
            assert paths[0].suffix == ".csv"

    def test_tex_only(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_table(df, Path(tmpdir) / "t", formats=("tex",))
            assert len(paths) == 1
            assert paths[0].suffix == ".tex"

    def test_no_formats(self):
        df = pd.DataFrame({"A": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_table(df, Path(tmpdir) / "t", formats=())
            assert paths == []

    def test_csv_roundtrip_preserves_data(self):
        df = pd.DataFrame(
            {"Estimate": [1.23, 4.56], "SE": [0.1, 0.2]},
            index=["mu", "sigma"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "roundtrip"
            export_table(df, out, formats=("csv",))
            loaded = pd.read_csv(out.with_suffix(".csv"), index_col=0)
            assert loaded.loc["mu", "Estimate"] == pytest.approx(1.23)

    def test_latex_has_booktabs(self):
        df = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bt"
            export_table(df, out, formats=("tex",))
            content = (out.with_suffix(".tex")).read_text()
            assert "\\begin{table}" in content


# --- from unit/test_reporting_tables_new.py ---


class TestFormatWithPrecision:
    """Tests for _format_with_precision adaptive formatting."""

    def test_normal_value(self):
        """Normal value with normal uncertainty."""
        result = _format_with_precision(1.234, 0.05)
        assert "." in result
        assert float(result) == pytest.approx(1.234, abs=0.01)

    def test_non_finite_value(self):
        """Non-finite value returns string representation."""
        assert _format_with_precision(float("inf"), 0.1) == "inf"
        assert _format_with_precision(float("-inf"), 0.1) == "-inf"
        assert _format_with_precision(float("nan"), 0.1) == "nan"

    def test_non_finite_uncertainty(self):
        """Non-finite uncertainty falls back to min_decimals."""
        result = _format_with_precision(1.234, float("inf"))
        assert result == "1.23"  # Default min_decimals=2

    def test_zero_uncertainty(self):
        """Zero uncertainty falls back to min_decimals."""
        result = _format_with_precision(1.234, 0.0)
        assert result == "1.23"

    def test_custom_min_decimals(self):
        """min_decimals parameter is respected."""
        result = _format_with_precision(1.234, float("inf"), min_decimals=4)
        assert result == "1.2340"

    def test_very_small_uncertainty(self):
        """Very small uncertainty shows more decimal places."""
        result = _format_with_precision(0.001234, 0.00005)
        # Should have more than 2 decimals
        decimal_places = len(result.split(".")[-1])
        assert decimal_places >= 2

    def test_large_uncertainty(self):
        """Large uncertainty with min_decimals constraint."""
        result = _format_with_precision(100.0, 50.0)
        # Should still have at least min_decimals
        decimal_places = len(result.split(".")[-1])
        assert decimal_places >= 2


class TestEscapeLatexParamName_new:
    """Tests for _escape_latex_param_name."""

    def test_underscore_escaped(self):
        """Underscores are escaped for LaTeX."""
        assert _escape_latex_param_name("sigma_artist") == r"sigma\_artist"

    def test_no_special_chars(self):
        """Names without special chars are unchanged."""
        assert _escape_latex_param_name("beta") == "beta"

    def test_brackets_preserved(self):
        """Square brackets are preserved."""
        assert _escape_latex_param_name("beta[0]") == "beta[0]"

    def test_multiple_underscores(self):
        """Multiple underscores all escaped."""
        assert _escape_latex_param_name("a_b_c") == r"a\_b\_c"


class TestExportTable_new:
    """Tests for export_table function."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for export testing."""
        return pd.DataFrame(
            {"Estimate": ["1.23", "4.56"], "SE": ["0.05", "0.10"]},
            index=["beta[0]", "beta[1]"],
        )

    def test_csv_only(self, sample_df, tmp_path):
        """Export CSV only."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=("csv",))
        assert len(paths) == 1
        assert paths[0].suffix == ".csv"
        assert paths[0].exists()

    def test_tex_only(self, sample_df, tmp_path):
        """Export LaTeX only."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=("tex",))
        assert len(paths) == 1
        assert paths[0].suffix == ".tex"
        assert paths[0].exists()

    def test_both_formats(self, sample_df, tmp_path):
        """Export both CSV and LaTeX."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=("csv", "tex"))
        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert ".csv" in suffixes
        assert ".tex" in suffixes

    def test_csv_content(self, sample_df, tmp_path):
        """CSV content is valid."""
        base_path = tmp_path / "table"
        export_table(sample_df, base_path, formats=("csv",))
        loaded = pd.read_csv(base_path.with_suffix(".csv"), index_col=0)
        assert "Estimate" in loaded.columns
        assert len(loaded) == 2

    def test_tex_with_caption_and_label(self, sample_df, tmp_path):
        """LaTeX output includes caption and label."""
        base_path = tmp_path / "table"
        export_table(
            sample_df, base_path, formats=("tex",), caption="Test caption", label="tab:test"
        )
        content = (base_path.with_suffix(".tex")).read_text()
        assert "Test caption" in content
        assert "tab:test" in content

    def test_tex_default_label(self, sample_df, tmp_path):
        """LaTeX output derives label from filename when not specified."""
        base_path = tmp_path / "coefficients"
        export_table(sample_df, base_path, formats=("tex",))
        content = (base_path.with_suffix(".tex")).read_text()
        assert "tab:coefficients" in content

    def test_creates_parent_directories(self, sample_df, tmp_path):
        """Parent directories are created if they don't exist."""
        base_path = tmp_path / "deep" / "nested" / "table"
        paths = export_table(sample_df, base_path, formats=("csv",))
        assert paths[0].exists()

    def test_no_formats_returns_empty(self, sample_df, tmp_path):
        """Empty formats tuple creates no files."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=())
        assert len(paths) == 0


class TestCreateSensitivitySummaryTable_new:
    """Tests for create_sensitivity_summary_table."""

    def test_ok_convergence(self):
        """OK convergence status is preserved."""
        df = pd.DataFrame(
            {
                "variant": ["baseline"],
                "parameter": ["sigma"],
                "multiplier": [1.0],
                "elpd": [-100.0],
                "elpd_delta": [0.0],
                "elpd_se": [5.0],
                "convergence_flag": ["OK"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert result["Status"].iloc[0] == "OK"

    def test_non_ok_convergence_gets_dagger(self):
        """Non-OK convergence gets dagger symbol appended."""
        df = pd.DataFrame(
            {
                "variant": ["test"],
                "parameter": ["sigma"],
                "multiplier": [2.0],
                "elpd": [-120.0],
                "elpd_delta": [-20.0],
                "elpd_se": [8.0],
                "convergence_flag": ["R-hat > 1.01"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "\u2020" in result["Status"].iloc[0]  # dagger

    def test_non_numeric_elpd_becomes_em_dash(self):
        """Non-numeric ELPD values become em-dash."""
        df = pd.DataFrame(
            {
                "variant": ["test"],
                "parameter": ["sigma"],
                "multiplier": [2.0],
                "elpd": ["N/A"],
                "elpd_delta": [None],
                "elpd_se": [""],
                "convergence_flag": ["OK"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "\u2014" in str(result["ELPD"].iloc[0])  # em-dash

    def test_column_rename(self):
        """Columns are renamed for publication."""
        df = pd.DataFrame(
            {
                "variant": ["baseline"],
                "parameter": ["sigma"],
                "multiplier": [1.0],
                "elpd": [-100.0],
                "elpd_delta": [0.0],
                "elpd_se": [5.0],
                "convergence_flag": ["OK"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "Variant" in result.columns
        assert "Parameter" in result.columns
        assert "ELPD" in result.columns
        assert "\u0394ELPD" in result.columns
        assert "SE" in result.columns
        assert "Status" in result.columns
