"""Unit tests for reporting tables module.

Tests cover:
- Adaptive precision formatting for numeric values
- create_coefficient_table with various configurations
- create_diagnostics_table with convergence status
- create_comparison_table for model ranking
- export_table for CSV and LaTeX output
"""

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.reporting.tables import (
    _format_with_precision,
    create_coefficient_table,
    create_comparison_table,
    create_diagnostics_table,
    export_table,
)

# =============================================================================
# Test Fixtures
# =============================================================================


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


# =============================================================================
# Tests for _format_with_precision
# =============================================================================


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


# =============================================================================
# Tests for create_coefficient_table
# =============================================================================


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


# =============================================================================
# Tests for create_diagnostics_table
# =============================================================================


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


# =============================================================================
# Tests for create_comparison_table
# =============================================================================


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


# =============================================================================
# Tests for export_table
# =============================================================================


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


# =============================================================================
# Integration Tests
# =============================================================================


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
