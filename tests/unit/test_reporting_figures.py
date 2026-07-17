"""Tests for publication-quality figure generation.

Tests verify:
- File creation in both PDF and PNG formats
- Context manager restores rcParams
- Correct path return values
- Figure sizing and auto-scaling
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.evaluation.calibration import ReliabilityData
from panelcast.evaluation.ppc import PPCResult, PPCStatistic
from panelcast.reporting.figures import (
    COLORBLIND_COLORS,
    _ensure_output_dir,
    _fan_chart_quantiles,
    _save_dual_format,
    get_trace_plot_vars,
    save_artist_prediction_plot,
    save_forest_plot,
    save_posterior_plot,
    save_ppc_density_plot,
    save_predictions_plot,
    save_reliability_plot,
    save_trace_plot,
    select_artist_subsets,
    set_publication_style,
)

matplotlib.use("Agg")


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
        assert ref_idx < obs_idx, (
            f"sigma_ref (idx={ref_idx}) should precede sigma_obs (idx={obs_idx})"
        )

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


class TestSaveArtistPredictionPlot:
    """Tests for save_artist_prediction_plot.

    Regression coverage for the per-entity fan chart: ``actual_scores`` and the
    columns of ``pred_samples`` must share the same length. The publication
    pipeline appends one forecast point to the prediction fan, so the observed
    series is extended with a trailing NaN (rendered as a gap) — exercised here.
    """

    def test_creates_pdf_and_png(self, tmp_path):
        actual = np.array([70.0, 75.0, 72.0, 80.0])
        pred_samples = np.tile(actual, (5, 1))  # (n_samples, n_albums)
        pdf_path, png_path = save_artist_prediction_plot(
            artist="Test Artist",
            actual_scores=actual,
            pred_samples=pred_samples,
            album_labels=["A1", "A2", "A3", "A4"],
            output_dir=tmp_path,
            filename_base="artist_test",
        )
        assert pdf_path.exists()
        assert png_path.exists()

    def test_appended_forecast_point_with_nan_actual(self, tmp_path):
        """The pipeline shape: N observed albums + 1 forecast column.

        ``pred_samples`` is (n_samples, N+1) and ``actual_scores`` is the N
        observed values plus a trailing NaN for the forecast slot. Dimensions
        line up, so the figure renders instead of being silently swallowed.
        """
        actual = np.array([70.0, 75.0, 72.0])
        quantiles = np.array([60.0, 68.0, 74.0, 80.0, 88.0])  # q05..q95
        pred_for_fan = np.tile(actual, (5, 1))
        pred_for_fan = np.column_stack([pred_for_fan, quantiles[:, None]])  # (5, N+1)

        actual_for_fan = np.append(actual, np.nan)  # N+1, gap at forecast
        albums_for_fan = ["A1", "A2", "A3", "next"]

        assert pred_for_fan.shape[1] == len(actual_for_fan) == len(albums_for_fan)

        pdf_path, png_path = save_artist_prediction_plot(
            artist="Test Artist",
            actual_scores=actual_for_fan,
            pred_samples=pred_for_fan,
            album_labels=albums_for_fan,
            output_dir=tmp_path,
            filename_base="artist_forecast",
        )
        assert pdf_path.exists()
        assert png_path.exists()

    def test_forecast_quantiles_render_exactly(self):
        """Stored q05/q95 must appear exactly as the plotted band bounds.

        Re-percentiling the 5 stacked quantiles shrinks the outer band
        (np.percentile interpolates over 5 order statistics); passing them
        through as forecast_quantiles must recover the stored values.
        """
        actual = np.array([70.0, 75.0, 72.0])
        stored = np.array([60.0, 68.0, 74.0, 80.0, 88.0])  # q05, q25, q50, q75, q95
        pred_for_fan = np.tile(actual, (5, 1))
        pred_for_fan = np.column_stack([pred_for_fan, stored[:, None]])

        # Without passthrough the outer band shrinks — the bug being fixed.
        q05_bad, _, _, _, _, _, q95_bad = _fan_chart_quantiles(pred_for_fan)
        assert q05_bad[-1] > stored[0]
        assert q95_bad[-1] < stored[4]

        q05, q10, q25, q50, q75, q90, q95 = _fan_chart_quantiles(pred_for_fan, stored)
        assert q05[-1] == stored[0]
        assert q25[-1] == stored[1]
        assert q50[-1] == stored[2]
        assert q75[-1] == stored[3]
        assert q95[-1] == stored[4]
        # The interpolated 10/90 pair stays inside the exact bands.
        assert stored[0] < q10[-1] < stored[1]
        assert stored[3] < q90[-1] < stored[4]
        # Historical points are untouched by the passthrough.
        np.testing.assert_array_equal(q05[:-1], actual)
        np.testing.assert_array_equal(q95[:-1], actual)

    def test_forecast_quantiles_accepted_end_to_end(self, tmp_path):
        """save_artist_prediction_plot renders with forecast_quantiles supplied."""
        actual = np.array([70.0, 75.0, 72.0])
        stored = np.array([60.0, 68.0, 74.0, 80.0, 88.0])
        pred_for_fan = np.column_stack([np.tile(actual, (5, 1)), stored[:, None]])
        actual_for_fan = np.append(actual, np.nan)

        pdf_path, png_path = save_artist_prediction_plot(
            artist="Test Artist",
            actual_scores=actual_for_fan,
            pred_samples=pred_for_fan,
            album_labels=["A1", "A2", "A3", "next"],
            output_dir=tmp_path,
            filename_base="artist_forecast_quantiles",
            forecast_quantiles=stored,
        )
        assert pdf_path.exists()
        assert png_path.exists()

    def test_mismatched_dimensions_raise(self, tmp_path):
        """Guard the contract: N actual vs N+1 prediction columns is invalid.

        This is exactly the off-by-one the publication pipeline previously hit
        (and swallowed); the function must not silently accept it.
        """
        actual = np.array([70.0, 75.0, 72.0])  # N = 3
        pred_samples = np.tile(np.append(actual, 80.0), (5, 1))  # (5, N+1)
        with pytest.raises(ValueError):
            save_artist_prediction_plot(
                artist="Test Artist",
                actual_scores=actual,
                pred_samples=pred_samples,
                album_labels=None,
                output_dir=tmp_path,
                filename_base="artist_bad",
            )


class TestSelectArtistSubsets:
    """Subset selection for the per-entity fan charts (generic entity schema)."""

    @staticmethod
    def _known_df():
        # A: exact prediction, most prolific; B: worst residual; C: widest interval.
        return pd.DataFrame(
            {
                "entity": ["A", "B", "C"],
                "scenario": ["same", "same", "same"],
                "pred_mean": [80.0, 60.0, 70.0],
                "last_score": [80.0, 80.0, 75.0],
                "pred_q05": [75.0, 55.0, 60.0],
                "pred_q95": [85.0, 75.0, 90.0],
                "n_training_events": [10, 6, 8],
            }
        )

    def test_selects_by_each_criterion(self):
        subsets = select_artist_subsets(self._known_df(), min_albums=5, n_per_category=2)
        assert set(subsets) == {
            "best_predicted",
            "worst_predicted",
            "most_prolific",
            "high_uncertainty",
        }
        assert subsets["best_predicted"][0] == "A"  # zero residual
        assert subsets["worst_predicted"][0] == "B"  # largest residual
        assert subsets["most_prolific"][0] == "A"  # most training events
        assert subsets["high_uncertainty"][0] == "C"  # widest 90% interval

    def test_min_albums_filter_excludes_short_histories(self):
        df = self._known_df()
        df["n_training_events"] = [3, 2, 4]  # all below the threshold
        assert select_artist_subsets(df, min_albums=5) == {}

    def test_ignores_non_same_scenarios(self):
        df = self._known_df()
        df["scenario"] = "entity_mean"
        assert select_artist_subsets(df, min_albums=5) == {}


# --- from unit/test_reporting_figures_coverage.py ---


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


class TestSaveDualFormatErrors:
    """Tests for _save_dual_format with I/O failures."""

    def test_savefig_ioerror_propagates(self, tmp_path, monkeypatch):
        """IOError from savefig should propagate to caller."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        def _raise(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(fig, "savefig", _raise)

        with pytest.raises(OSError):
            _save_dual_format(fig, tmp_path / "test", "test")

        plt.close(fig)


# --- from unit/test_reporting_figures_expanded.py ---


@pytest.fixture
def idata_basic():
    """Minimal InferenceData with user-prefixed parameters."""
    np.random.seed(42)
    posterior = xr.Dataset(
        {
            "user_mu_artist": (["chain", "draw"], np.random.randn(2, 50)),
            "user_sigma_artist": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
            "user_sigma_rw": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
            "user_sigma_obs": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
            "user_rho": (["chain", "draw"], np.random.randn(2, 50) * 0.3),
        }
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def idata_with_sigma_ref(idata_basic):
    """InferenceData with sigma_ref parameter."""
    idata_basic.posterior["user_sigma_ref"] = xr.DataArray(
        np.abs(np.random.randn(2, 50)),
        dims=["chain", "draw"],
    )
    return idata_basic


@pytest.fixture
def idata_with_n_exponent(idata_basic):
    """InferenceData with n_exponent parameter."""
    idata_basic.posterior["user_n_exponent"] = xr.DataArray(
        np.random.rand(2, 50),
        dims=["chain", "draw"],
    )
    return idata_basic


class TestColorblindColors:
    """Tests for COLORBLIND_COLORS constant."""

    def test_is_list(self):
        assert isinstance(COLORBLIND_COLORS, list)

    def test_has_seven_colors(self):
        assert len(COLORBLIND_COLORS) == 7

    def test_all_hex(self):
        for color in COLORBLIND_COLORS:
            assert color.startswith("#")
            assert len(color) == 7

    def test_all_unique(self):
        assert len(set(COLORBLIND_COLORS)) == 7


class TestSetPublicationStyle:
    """Tests for set_publication_style context manager."""

    def test_restores_rcparams(self):
        original_fontsize = plt.rcParams["font.size"]
        with set_publication_style():
            assert plt.rcParams["font.size"] == 9
        assert plt.rcParams["font.size"] == original_fontsize

    def test_sets_serif_font(self):
        with set_publication_style():
            assert plt.rcParams["font.family"] == ["serif"]

    def test_sets_savefig_dpi(self):
        with set_publication_style():
            assert plt.rcParams["savefig.dpi"] == 300

    def test_removes_top_spine(self):
        with set_publication_style():
            assert plt.rcParams["axes.spines.top"] is False

    def test_removes_right_spine(self):
        with set_publication_style():
            assert plt.rcParams["axes.spines.right"] is False

    def test_pdf_fonttype_42(self):
        with set_publication_style():
            assert plt.rcParams["pdf.fonttype"] == 42

    def test_figure_creation_inside(self):
        with set_publication_style():
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            plt.close(fig)

    def test_nested_context(self):
        with set_publication_style():
            with set_publication_style():
                assert plt.rcParams["font.size"] == 9
            assert plt.rcParams["font.size"] == 9


class TestGetTracePlotVars:
    """Tests for get_trace_plot_vars."""

    def test_basic_vars(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert "user_sigma_obs" in vars
        assert "user_rho" in vars

    def test_includes_hyperpriors(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_", include_hyperpriors=True)
        assert "user_mu_artist" in vars
        assert "user_sigma_artist" in vars
        assert "user_sigma_rw" in vars

    def test_excludes_hyperpriors(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_", include_hyperpriors=False)
        assert "user_mu_artist" not in vars
        assert "user_sigma_artist" not in vars

    def test_includes_sigma_ref(self, idata_with_sigma_ref):
        vars = get_trace_plot_vars(idata_with_sigma_ref, prefix="user_")
        assert "user_sigma_ref" in vars
        # sigma_ref should appear before sigma_obs
        ref_idx = vars.index("user_sigma_ref")
        obs_idx = vars.index("user_sigma_obs")
        assert ref_idx < obs_idx

    def test_no_sigma_ref_when_absent(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert "user_sigma_ref" not in vars

    def test_includes_n_exponent(self, idata_with_n_exponent):
        vars = get_trace_plot_vars(idata_with_n_exponent, prefix="user_")
        assert "user_n_exponent" in vars

    def test_no_n_exponent_when_absent(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert "user_n_exponent" not in vars

    def test_critic_prefix(self):
        posterior = xr.Dataset(
            {
                "critic_mu_artist": (["chain", "draw"], np.random.randn(2, 50)),
                "critic_sigma_artist": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
                "critic_sigma_rw": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
                "critic_sigma_obs": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
                "critic_rho": (["chain", "draw"], np.random.randn(2, 50) * 0.3),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        vars = get_trace_plot_vars(idata, prefix="critic_")
        assert all(v.startswith("critic_") for v in vars)

    def test_returns_list(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert isinstance(vars, list)


class TestSaveSliceCoveragePlot:
    """Smoke coverage for the sliced-calibration figure (#181)."""

    def _by_slice(self):
        return {
            "expected_false_flags": 0.4,
            "slices": [
                {
                    "dimension": "group",
                    "label": "rock",
                    "n": 120,
                    "levels": {
                        "0.80": {
                            "nominal": 0.8, "empirical": 0.78,
                            "wilson_lo": 0.70, "wilson_hi": 0.85,
                            "mean_interval_width": 12.0, "flagged": False,
                        },
                    },
                    "pit_max_abs_dev": 0.03,
                    "flagged": False,
                },
                {
                    "dimension": "train_history",
                    "label": "1-2",
                    "n": 60,
                    "levels": {
                        "0.80": {
                            "nominal": 0.8, "empirical": 0.55,
                            "wilson_lo": 0.42, "wilson_hi": 0.67,
                            "mean_interval_width": 8.0, "flagged": True,
                        },
                    },
                    "pit_max_abs_dev": 0.2,
                    "flagged": True,
                },
            ],
        }

    def test_creates_files(self, tmp_path):
        from panelcast.reporting.figures import save_slice_coverage_plot

        pdf, png = save_slice_coverage_plot(self._by_slice(), tmp_path, "slice_cov")
        assert pdf.exists() and pdf.suffix == ".pdf"
        assert png.exists() and png.suffix == ".png"

    def test_empty_payload_raises(self, tmp_path):
        from panelcast.reporting.figures import save_slice_coverage_plot

        with pytest.raises(ValueError, match="no slices"):
            save_slice_coverage_plot({"slices": []}, tmp_path, "slice_cov")


class TestSaveRankScatterPlot:
    """Smoke coverage for the predicted-vs-realized rank scatter (#182)."""

    def test_creates_files(self, tmp_path):
        from panelcast.reporting.figures import save_rank_scatter_plot

        slate = pd.DataFrame(
            {
                "entity": ["a", "b", "c"],
                "predicted_rank": [1, 2, 3],
                "realized_rank": [2, 1, 3],
                "p_top10": [0.9, 0.8, 0.1],
            }
        )
        pdf, png = save_rank_scatter_plot(slate, tmp_path, "rank_scatter")
        assert pdf.exists() and png.exists()

    def test_missing_columns_raise(self, tmp_path):
        from panelcast.reporting.figures import save_rank_scatter_plot

        with pytest.raises(ValueError, match="lacks"):
            save_rank_scatter_plot(pd.DataFrame({"x": [1]}), tmp_path, "rank_scatter")


class TestRankScatterNoColorColumn:
    def test_plots_without_p_top_column(self, tmp_path):
        from panelcast.reporting.figures import save_rank_scatter_plot

        slate = pd.DataFrame({"predicted_rank": [1, 2], "realized_rank": [2, 1]})
        pdf, png = save_rank_scatter_plot(slate, tmp_path, "rank_scatter_plain")
        assert pdf.exists() and png.exists()
