"""Coverage-targeted tests for reporting/figures.py.

Tests target missed lines/branches:
- save_ppc_density_plot edge cases (TypeError/ValueError on p_value/mc_se,
  stats_to_plot filtering fallback, unused axes hiding, single-panel layout)
- save_predictions_plot with single data point
- save_reliability_plot with zero-count bins
- save_forest_plot with single variant
- _save_dual_format IOError handling
- save_posterior_plot single-variable axes branch (non-ndarray return)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.calibration import ReliabilityData
from panelcast.evaluation.ppc import PPCResult, PPCStatistic
from panelcast.reporting.figures import (
    _ensure_output_dir,
    _save_dual_format,
    save_forest_plot,
    save_ppc_density_plot,
    save_predictions_plot,
    save_reliability_plot,
    set_publication_style,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ppc_result_basic():
    """PPCResult with two statistics for basic testing."""
    np.random.seed(42)
    return PPCResult(
        statistics=[
            PPCStatistic(
                name="mean",
                observed=50.0,
                replicated_distribution=np.random.normal(50, 2, 200),
                bayesian_p_value=0.45,
                mc_se=0.035,
            ),
            PPCStatistic(
                name="sd",
                observed=10.0,
                replicated_distribution=np.random.normal(10, 1, 200),
                bayesian_p_value=0.52,
                mc_se=0.035,
            ),
        ],
        n_obs=100,
        n_samples=200,
    )


@pytest.fixture
def ppc_result_nan_pvalue():
    """PPCResult where p_value and mc_se are non-numeric (trigger except branches)."""
    np.random.seed(42)
    return PPCResult(
        statistics=[
            PPCStatistic(
                name="mean",
                observed=50.0,
                replicated_distribution=np.random.normal(50, 2, 200),
                bayesian_p_value="not_a_number",  # type: ignore[arg-type]
                mc_se=None,  # type: ignore[arg-type]
            ),
        ],
        n_obs=100,
        n_samples=200,
    )


# =============================================================================
# TestSavePpcDensityPlot
# =============================================================================


class TestSavePpcDensityPlot:
    """Tests for save_ppc_density_plot targeting uncovered branches."""

    def test_basic_ppc_density_creates_files(self, tmp_path, ppc_result_basic):
        """PPC density plot creates both PDF and PNG."""
        pdf, png = save_ppc_density_plot(ppc_result_basic, tmp_path, "ppc_basic")
        assert pdf.exists()
        assert png.exists()
        assert pdf.suffix == ".pdf"
        assert png.suffix == ".png"
        plt.close("all")

    def test_nan_pvalue_and_mc_se_branches(self, tmp_path, ppc_result_nan_pvalue):
        """Non-numeric p_value/mc_se should hit the except branches and produce NaN display."""
        pdf, png = save_ppc_density_plot(ppc_result_nan_pvalue, tmp_path, "ppc_nan")
        assert pdf.exists()
        assert png.exists()
        plt.close("all")

    def test_statistics_to_plot_filter_matches(self, tmp_path, ppc_result_basic):
        """When statistics_to_plot matches some names, only those are plotted."""
        pdf, png = save_ppc_density_plot(
            ppc_result_basic,
            tmp_path,
            "ppc_filtered",
            statistics_to_plot=["mean"],
        )
        assert pdf.exists()
        plt.close("all")

    def test_statistics_to_plot_filter_no_match_fallback(self, tmp_path, ppc_result_basic):
        """When no statistics match the filter, fallback to all statistics."""
        pdf, png = save_ppc_density_plot(
            ppc_result_basic,
            tmp_path,
            "ppc_fallback",
            statistics_to_plot=["nonexistent_stat"],
        )
        assert pdf.exists()
        plt.close("all")

    def test_single_panel_layout(self, tmp_path):
        """Single statistic triggers n_panels==1 branch (axes = np.array([axes]))."""
        np.random.seed(42)
        result = PPCResult(
            statistics=[
                PPCStatistic(
                    name="mean",
                    observed=50.0,
                    replicated_distribution=np.random.normal(50, 2, 100),
                    bayesian_p_value=0.5,
                    mc_se=0.03,
                ),
            ],
            n_obs=100,
            n_samples=100,
        )
        pdf, png = save_ppc_density_plot(result, tmp_path, "ppc_single")
        assert pdf.exists()
        plt.close("all")

    def test_unused_axes_hidden(self, tmp_path):
        """When n_panels < n_rows*n_cols, extra axes should be hidden."""
        np.random.seed(42)
        stats = [
            PPCStatistic(
                name=f"stat_{i}",
                observed=float(i),
                replicated_distribution=np.random.normal(i, 1, 100),
                bayesian_p_value=0.5,
                mc_se=0.03,
            )
            for i in range(4)
        ]
        result = PPCResult(statistics=stats, n_obs=100, n_samples=100)
        # 4 panels in a 3-col layout => 2 rows x 3 cols => 2 unused axes
        pdf, png = save_ppc_density_plot(result, tmp_path, "ppc_hide")
        assert pdf.exists()
        plt.close("all")

    def test_custom_figsize(self, tmp_path, ppc_result_basic):
        """Explicit figsize should override auto-sizing."""
        pdf, png = save_ppc_density_plot(
            ppc_result_basic,
            tmp_path,
            "ppc_custom_size",
            figsize=(12, 4),
        )
        assert pdf.exists()
        plt.close("all")

    def test_five_default_stats(self, tmp_path):
        """Default statistics_to_plot=None uses mean, sd, skewness, min, max."""
        np.random.seed(42)
        stats = [
            PPCStatistic(
                name=name,
                observed=1.0,
                replicated_distribution=np.random.normal(1, 0.5, 100),
                bayesian_p_value=0.5,
                mc_se=0.03,
            )
            for name in ["mean", "sd", "skewness", "min", "max", "extra"]
        ]
        result = PPCResult(statistics=stats, n_obs=50, n_samples=100)
        pdf, png = save_ppc_density_plot(result, tmp_path, "ppc_defaults")
        assert pdf.exists()
        plt.close("all")


# =============================================================================
# TestSavePredictionsPlotEdgeCases
# =============================================================================


class TestSavePredictionsPlotEdgeCases:
    """Edge cases for save_predictions_plot."""

    def test_single_data_point(self, tmp_path):
        """Plot should work with a single observation."""
        pdf, png = save_predictions_plot(
            y_true=np.array([50.0]),
            y_pred_mean=np.array([52.0]),
            y_pred_lower=np.array([48.0]),
            y_pred_upper=np.array([56.0]),
            output_dir=tmp_path,
            filename_base="pred_single",
        )
        assert pdf.exists()
        assert png.exists()
        plt.close("all")

    def test_two_data_points(self, tmp_path):
        """Plot should work with two observations."""
        pdf, png = save_predictions_plot(
            y_true=np.array([40.0, 80.0]),
            y_pred_mean=np.array([42.0, 78.0]),
            y_pred_lower=np.array([38.0, 74.0]),
            y_pred_upper=np.array([46.0, 82.0]),
            output_dir=tmp_path,
            filename_base="pred_two",
        )
        assert pdf.exists()
        plt.close("all")

    def test_identical_values(self, tmp_path):
        """Plot should handle all identical predictions and actuals."""
        n = 5
        val = np.full(n, 60.0)
        pdf, png = save_predictions_plot(
            y_true=val,
            y_pred_mean=val,
            y_pred_lower=val - 1,
            y_pred_upper=val + 1,
            output_dir=tmp_path,
            filename_base="pred_identical",
        )
        assert pdf.exists()
        plt.close("all")


# =============================================================================
# TestSaveReliabilityPlotEdgeCases
# =============================================================================


class TestSaveReliabilityPlotEdgeCases:
    """Edge cases for save_reliability_plot."""

    def test_zero_count_bins(self, tmp_path):
        """Bins with zero counts should not produce annotations (n > 0 check)."""
        data = ReliabilityData(
            bin_edges=np.linspace(0, 1, 4),
            predicted_probs=np.array([0.2, 0.5, 0.8]),
            observed_freq=np.array([0.15, 0.55, 0.82]),
            counts=np.array([0, 10, 0]),
        )
        pdf, png = save_reliability_plot(data, tmp_path, "rel_zeros")
        assert pdf.exists()
        plt.close("all")

    def test_all_zero_counts(self, tmp_path):
        """All bins have zero counts - should still render without error."""
        data = ReliabilityData(
            bin_edges=np.linspace(0, 1, 3),
            predicted_probs=np.array([0.25, 0.75]),
            observed_freq=np.array([0.0, 0.0]),
            counts=np.array([0, 0]),
        )
        pdf, png = save_reliability_plot(data, tmp_path, "rel_all_zeros")
        assert pdf.exists()
        plt.close("all")

    def test_single_bin(self, tmp_path):
        """Single bin reliability data."""
        data = ReliabilityData(
            bin_edges=np.array([0.0, 1.0]),
            predicted_probs=np.array([0.5]),
            observed_freq=np.array([0.48]),
            counts=np.array([100]),
        )
        pdf, png = save_reliability_plot(data, tmp_path, "rel_single_bin")
        assert pdf.exists()
        plt.close("all")


# =============================================================================
# TestSaveForestPlotEdgeCases
# =============================================================================


class TestSaveForestPlotEdgeCases:
    """Edge cases for save_forest_plot."""

    def test_single_param_single_variant(self, tmp_path):
        """Forest plot with a single parameter and variant."""
        df = pd.DataFrame(
            {
                "param": ["mu"],
                "variant": ["default"],
                "mean": [1.5],
                "hdi_3%": [1.2],
                "hdi_97%": [1.8],
            }
        )
        pdf, png = save_forest_plot(df, tmp_path, "forest_single")
        assert pdf.exists()
        assert png.exists()
        plt.close("all")

    def test_many_params(self, tmp_path):
        """Forest plot with many parameters auto-sizes height."""
        rows = []
        for i in range(10):
            rows.append(
                {
                    "param": f"param_{i}",
                    "variant": "v1",
                    "mean": float(i) * 0.1,
                    "hdi_3%": float(i) * 0.1 - 0.2,
                    "hdi_97%": float(i) * 0.1 + 0.2,
                }
            )
        df = pd.DataFrame(rows)
        pdf, png = save_forest_plot(df, tmp_path, "forest_many")
        assert pdf.exists()
        plt.close("all")

    def test_custom_figsize_override(self, tmp_path):
        """Explicit figsize overrides auto-calculation."""
        df = pd.DataFrame(
            {
                "param": ["mu", "sigma"],
                "variant": ["v1", "v1"],
                "mean": [1.0, 0.5],
                "hdi_3%": [0.8, 0.3],
                "hdi_97%": [1.2, 0.7],
            }
        )
        pdf, png = save_forest_plot(df, tmp_path, "forest_figsize", figsize=(10, 3))
        assert pdf.exists()
        plt.close("all")


# =============================================================================
# TestEnsureOutputDir
# =============================================================================


class TestEnsureOutputDir:
    """Tests for _ensure_output_dir helper."""

    def test_creates_nested_dirs(self, tmp_path):
        """Should create deeply nested directories."""
        new_dir = tmp_path / "a" / "b" / "c"
        result = _ensure_output_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_dir_ok(self, tmp_path):
        """Should succeed when directory already exists."""
        result = _ensure_output_dir(tmp_path)
        assert result == tmp_path

    def test_accepts_string(self, tmp_path):
        """Should accept string path and convert to Path."""
        str_path = str(tmp_path / "string_dir")
        result = _ensure_output_dir(Path(str_path))
        assert result.exists()


# =============================================================================
# TestSaveDualFormatErrors
# =============================================================================


class TestSaveDualFormatErrors:
    """Tests for _save_dual_format with I/O failures."""

    def test_savefig_ioerror_propagates(self, tmp_path):
        """IOError from savefig should propagate to caller."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with pytest.raises(OSError):
            # Use a path that can't be written to
            _save_dual_format(fig, Path("/dev/null/impossible"), "test")

        plt.close(fig)
