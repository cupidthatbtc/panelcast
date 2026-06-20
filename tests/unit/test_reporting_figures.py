"""Tests for publication-quality figure generation.

Tests verify:
- File creation in both PDF and PNG formats
- Context manager restores rcParams
- Correct path return values
- Figure sizing and auto-scaling
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib

# Use non-interactive backend for tests (avoids Tkinter issues)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.evaluation.calibration import ReliabilityData
from panelcast.reporting.figures import (
    COLORBLIND_COLORS,
    get_trace_plot_vars,
    save_forest_plot,
    save_posterior_plot,
    save_predictions_plot,
    save_reliability_plot,
    save_trace_plot,
    set_publication_style,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_idata():
    """Create minimal InferenceData for testing."""
    np.random.seed(42)
    posterior = xr.Dataset(
        {
            "mu": (["chain", "draw"], np.random.randn(2, 50)),
            "sigma": (["chain", "draw"], np.abs(np.random.randn(2, 50)) + 0.1),
        }
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def mock_predictions():
    """Create mock prediction data for testing."""
    np.random.seed(42)
    n = 20
    y_true = np.linspace(40, 80, n)
    y_pred_mean = y_true + np.random.randn(n) * 2
    y_pred_lower = y_pred_mean - 5
    y_pred_upper = y_pred_mean + 5
    return y_true, y_pred_mean, y_pred_lower, y_pred_upper


@pytest.fixture
def mock_reliability_data():
    """Create mock ReliabilityData for testing."""
    n_bins = 5
    return ReliabilityData(
        bin_edges=np.linspace(0, 1, n_bins + 1),
        predicted_probs=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        observed_freq=np.array([0.12, 0.28, 0.52, 0.68, 0.88]),
        counts=np.array([15, 20, 25, 20, 15]),
    )


@pytest.fixture
def mock_comparison_df():
    """Create mock DataFrame for forest plot testing."""
    return pd.DataFrame(
        {
            "param": ["mu", "mu", "sigma", "sigma"],
            "variant": ["default", "diffuse", "default", "diffuse"],
            "mean": [1.5, 1.3, 0.8, 0.9],
            "hdi_3%": [1.2, 0.8, 0.6, 0.5],
            "hdi_97%": [1.8, 1.8, 1.0, 1.3],
        }
    )


# =============================================================================
# TestPublicationStyle
# =============================================================================


class TestPublicationStyle:
    """Tests for set_publication_style context manager."""

    def test_style_context_manager(self):
        """rcParams should reset after context exits."""
        # Get a baseline value
        original_font_size = plt.rcParams["font.size"]

        with set_publication_style():
            # Font size should be 9 inside context
            assert plt.rcParams["font.size"] == 9

        # Should be restored after exit
        assert plt.rcParams["font.size"] == original_font_size

    def test_colorblind_palette_length(self):
        """Should have 7 colorblind-safe colors."""
        assert len(COLORBLIND_COLORS) == 7

    def test_font_settings(self):
        """Should set serif font family inside context."""
        with set_publication_style():
            # font.family returns a list in matplotlib
            font_family = plt.rcParams["font.family"]
            assert "serif" in font_family or font_family == "serif"

    def test_spines_removed(self):
        """Top and right spines should be removed."""
        with set_publication_style():
            assert plt.rcParams["axes.spines.top"] is False
            assert plt.rcParams["axes.spines.right"] is False

    def test_dpi_settings(self):
        """Should set correct DPI for screen and export."""
        with set_publication_style():
            assert plt.rcParams["figure.dpi"] == 100
            assert plt.rcParams["savefig.dpi"] == 300


# =============================================================================
# TestSaveTracePlot
# =============================================================================


class TestSaveTracePlot:
    """Tests for save_trace_plot function."""

    def test_creates_pdf_and_png(self, tmp_path, mock_idata):
        """Should create both PDF and PNG files."""
        pdf_path, png_path = save_trace_plot(mock_idata, ["mu"], tmp_path, "trace_test")
        assert pdf_path.exists()
        assert png_path.exists()
        assert pdf_path.suffix == ".pdf"
        assert png_path.suffix == ".png"

    def test_returns_paths(self, tmp_path, mock_idata):
        """Should return tuple of Path objects."""
        pdf_path, png_path = save_trace_plot(mock_idata, ["mu"], tmp_path, "trace_test")
        assert isinstance(pdf_path, Path)
        assert isinstance(png_path, Path)

    def test_auto_figsize(self, tmp_path, mock_idata):
        """Figure size should scale with variable count."""
        # Test with one variable - should create valid files
        pdf_path, _ = save_trace_plot(mock_idata, ["mu"], tmp_path, "trace_one")
        assert pdf_path.exists()

        # Test with two variables - should also work
        pdf_path2, _ = save_trace_plot(mock_idata, ["mu", "sigma"], tmp_path, "trace_two")
        assert pdf_path2.exists()

    def test_output_dir_created(self, tmp_path, mock_idata):
        """Should create output directory if missing."""
        new_dir = tmp_path / "subdir" / "figures"
        pdf_path, png_path = save_trace_plot(mock_idata, ["mu"], new_dir, "trace_test")
        assert new_dir.exists()
        assert pdf_path.exists()

    def test_custom_figsize(self, tmp_path, mock_idata):
        """Should accept custom figure size."""
        pdf_path, png_path = save_trace_plot(
            mock_idata, ["mu"], tmp_path, "trace_custom", figsize=(12, 4)
        )
        assert pdf_path.exists()


# =============================================================================
# TestSavePosteriorPlot
# =============================================================================


class TestSavePosteriorPlot:
    """Tests for save_posterior_plot function."""

    def test_creates_dual_format(self, tmp_path, mock_idata):
        """Should create both PDF and PNG files."""
        pdf_path, png_path = save_posterior_plot(mock_idata, ["mu"], tmp_path, "posterior_test")
        assert pdf_path.exists()
        assert png_path.exists()

    def test_custom_hdi_prob(self, tmp_path, mock_idata):
        """Should work with different HDI probability."""
        pdf_path, _ = save_posterior_plot(
            mock_idata, ["mu"], tmp_path, "posterior_90", hdi_prob=0.90
        )
        assert pdf_path.exists()

    def test_returns_correct_paths(self, tmp_path, mock_idata):
        """Paths should match expected filenames."""
        pdf_path, png_path = save_posterior_plot(mock_idata, ["mu"], tmp_path, "posterior_test")
        assert pdf_path.name == "posterior_test.pdf"
        assert png_path.name == "posterior_test.png"

    def test_multiple_variables(self, tmp_path, mock_idata):
        """Should handle multiple variables."""
        pdf_path, _ = save_posterior_plot(mock_idata, ["mu", "sigma"], tmp_path, "posterior_multi")
        assert pdf_path.exists()


# =============================================================================
# TestSavePredictionsPlot
# =============================================================================


class TestSavePredictionsPlot:
    """Tests for save_predictions_plot function."""

    def test_creates_files(self, tmp_path, mock_predictions):
        """Should create both formats."""
        y_true, y_pred_mean, y_pred_lower, y_pred_upper = mock_predictions
        pdf_path, png_path = save_predictions_plot(
            y_true, y_pred_mean, y_pred_lower, y_pred_upper, tmp_path, "pred_test"
        )
        assert pdf_path.exists()
        assert png_path.exists()

    def test_accepts_numpy_arrays(self, tmp_path):
        """Should work with np.ndarray inputs."""
        n = 10
        y_true = np.random.randn(n) * 10 + 50
        y_pred = y_true + np.random.randn(n) * 2
        pdf_path, _ = save_predictions_plot(
            y_true, y_pred, y_pred - 3, y_pred + 3, tmp_path, "pred_numpy"
        )
        assert pdf_path.exists()

    def test_custom_ci_label(self, tmp_path, mock_predictions):
        """Should accept custom CI label."""
        y_true, y_pred_mean, y_pred_lower, y_pred_upper = mock_predictions
        pdf_path, _ = save_predictions_plot(
            y_true,
            y_pred_mean,
            y_pred_lower,
            y_pred_upper,
            tmp_path,
            "pred_custom_label",
            ci_label="95% CI",
        )
        assert pdf_path.exists()

    def test_accepts_lists(self, tmp_path):
        """Should also accept list inputs (converted to arrays)."""
        y_true = [50, 60, 70, 80]
        y_pred = [52, 58, 72, 78]
        y_lower = [48, 54, 68, 74]
        y_upper = [56, 62, 76, 82]
        pdf_path, _ = save_predictions_plot(
            y_true, y_pred, y_lower, y_upper, tmp_path, "pred_lists"
        )
        assert pdf_path.exists()


# =============================================================================
# TestSaveReliabilityPlot
# =============================================================================


class TestSaveReliabilityPlot:
    """Tests for save_reliability_plot function."""

    def test_creates_files(self, tmp_path, mock_reliability_data):
        """Should create both formats."""
        pdf_path, png_path = save_reliability_plot(
            mock_reliability_data, tmp_path, "reliability_test"
        )
        assert pdf_path.exists()
        assert png_path.exists()

    def test_accepts_reliability_data(self, tmp_path, mock_reliability_data):
        """Should work with ReliabilityData dataclass."""
        pdf_path, _ = save_reliability_plot(
            mock_reliability_data, tmp_path, "reliability_dataclass"
        )
        assert pdf_path.exists()

    def test_custom_figsize(self, tmp_path, mock_reliability_data):
        """Should accept custom figure size."""
        pdf_path, _ = save_reliability_plot(
            mock_reliability_data, tmp_path, "reliability_custom", figsize=(8, 6)
        )
        assert pdf_path.exists()


# =============================================================================
# TestSaveForestPlot
# =============================================================================


class TestSaveForestPlot:
    """Tests for save_forest_plot function."""

    def test_creates_files(self, tmp_path, mock_comparison_df):
        """Should create both formats."""
        pdf_path, png_path = save_forest_plot(mock_comparison_df, tmp_path, "forest_test")
        assert pdf_path.exists()
        assert png_path.exists()

    def test_accepts_comparison_df(self, tmp_path, mock_comparison_df):
        """Should work with DataFrame from sensitivity analysis."""
        pdf_path, _ = save_forest_plot(mock_comparison_df, tmp_path, "forest_df")
        assert pdf_path.exists()

    def test_custom_column_names(self, tmp_path):
        """Should work with custom column names."""
        df = pd.DataFrame(
            {
                "parameter": ["mu", "sigma"],
                "model": ["v1", "v1"],
                "estimate": [1.0, 0.5],
                "lower": [0.8, 0.3],
                "upper": [1.2, 0.7],
            }
        )
        pdf_path, _ = save_forest_plot(
            df,
            tmp_path,
            "forest_custom",
            param_col="parameter",
            variant_col="model",
            estimate_col="estimate",
            lower_col="lower",
            upper_col="upper",
        )
        assert pdf_path.exists()

    def test_auto_figsize(self, tmp_path, mock_comparison_df):
        """Figure height should auto-scale with number of rows."""
        # Basic test that auto-sizing works
        pdf_path, _ = save_forest_plot(mock_comparison_df, tmp_path, "forest_auto")
        assert pdf_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestFigureMemoryCleanup:
    """Tests that figures are properly closed to avoid memory leaks."""

    def test_trace_plot_closes_figure(self, tmp_path, mock_idata):
        """Trace plot should close figure after saving."""
        n_figures_before = len(plt.get_fignums())
        save_trace_plot(mock_idata, ["mu"], tmp_path, "trace_cleanup")
        n_figures_after = len(plt.get_fignums())
        # Should not have added any open figures
        assert n_figures_after == n_figures_before

    def test_predictions_plot_closes_figure(self, tmp_path, mock_predictions):
        """Predictions plot should close figure after saving."""
        y_true, y_pred_mean, y_pred_lower, y_pred_upper = mock_predictions
        n_figures_before = len(plt.get_fignums())
        save_predictions_plot(
            y_true, y_pred_mean, y_pred_lower, y_pred_upper, tmp_path, "pred_cleanup"
        )
        n_figures_after = len(plt.get_fignums())
        assert n_figures_after == n_figures_before


# =============================================================================
# TestGetTracePlotVars - Dynamic sigma_ref detection
# =============================================================================


class TestGetTracePlotVarsSigmaRef:
    """Tests for dynamic sigma_ref detection in get_trace_plot_vars()."""

    def test_includes_sigma_ref_when_present(self):
        """sigma_ref should appear in var list when present in posterior."""
        posterior = {
            "user_mu_artist": np.random.randn(1, 10),
            "user_sigma_artist": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_sigma_rw": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_sigma_obs": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_rho": np.random.randn(1, 10),
            "user_sigma_ref": np.abs(np.random.randn(1, 10)) + 0.1,
        }
        idata = az.from_dict(posterior=posterior)

        var_names = get_trace_plot_vars(idata)

        assert "user_sigma_ref" in var_names
        # sigma_ref should appear BEFORE sigma_obs
        ref_idx = var_names.index("user_sigma_ref")
        obs_idx = var_names.index("user_sigma_obs")
        assert (
            ref_idx < obs_idx
        ), f"sigma_ref (idx={ref_idx}) should precede sigma_obs (idx={obs_idx})"

    def test_no_sigma_ref_when_absent(self):
        """sigma_ref should NOT appear when absent from posterior (homoscedastic)."""
        posterior = {
            "user_mu_artist": np.random.randn(1, 10),
            "user_sigma_artist": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_sigma_rw": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_sigma_obs": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_rho": np.random.randn(1, 10),
        }
        idata = az.from_dict(posterior=posterior)

        var_names = get_trace_plot_vars(idata)

        assert "user_sigma_ref" not in var_names
        assert "user_sigma_obs" in var_names

    def test_sigma_ref_and_n_exponent(self):
        """Both sigma_ref and n_exponent should appear when both present."""
        posterior = {
            "user_mu_artist": np.random.randn(1, 10),
            "user_sigma_artist": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_sigma_rw": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_sigma_obs": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_rho": np.random.randn(1, 10),
            "user_sigma_ref": np.abs(np.random.randn(1, 10)) + 0.1,
            "user_n_exponent": np.random.randn(1, 10) * 0.1,
        }
        idata = az.from_dict(posterior=posterior)

        var_names = get_trace_plot_vars(idata)

        assert "user_sigma_ref" in var_names
        assert "user_n_exponent" in var_names
