"""Unit tests for visualization charts module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from panelcast.visualization.charts import (
    _hdi_label_from_columns,
    create_forest_plot,
    create_posterior_plot,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)


class TestCreateTracePlot:
    """Tests for create_trace_plot."""

    def test_returns_figure(self):
        samples = np.random.randn(4, 100)
        fig = create_trace_plot(samples, "mu")
        assert isinstance(fig, go.Figure)

    def test_title_contains_var_name(self):
        samples = np.random.randn(2, 50)
        fig = create_trace_plot(samples, "sigma")
        assert "sigma" in fig.layout.title.text

    def test_two_chains_produce_four_traces(self):
        samples = np.random.randn(2, 50)
        fig = create_trace_plot(samples, "mu")
        # 2 chains * 2 panels (trace + histogram) = 4 traces
        assert len(fig.data) == 4

    def test_four_chains_produce_eight_traces(self):
        samples = np.random.randn(4, 50)
        fig = create_trace_plot(samples, "mu")
        assert len(fig.data) == 8

    def test_single_chain(self):
        samples = np.random.randn(1, 100)
        fig = create_trace_plot(samples, "mu")
        assert len(fig.data) == 2  # 1 trace + 1 histogram

    def test_custom_template(self):
        samples = np.random.randn(2, 50)
        fig = create_trace_plot(samples, "mu", template="aoty_dark")
        # Dark theme has dark background
        assert fig.layout.template.layout.paper_bgcolor == "#1E1E1E"

    def test_default_template_is_light(self):
        samples = np.random.randn(2, 50)
        fig = create_trace_plot(samples, "mu")
        # Light theme has white background
        assert fig.layout.template.layout.paper_bgcolor == "white"

    def test_overlay_barmode(self):
        samples = np.random.randn(2, 50)
        fig = create_trace_plot(samples, "mu")
        assert fig.layout.barmode == "overlay"

    def test_accepts_list_input(self):
        samples = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        fig = create_trace_plot(np.array(samples), "mu")
        assert isinstance(fig, go.Figure)

    def test_trace_names_include_chain(self):
        samples = np.random.randn(3, 50)
        fig = create_trace_plot(samples, "mu")
        trace_names = [t.name for t in fig.data if hasattr(t, "name")]
        assert "Chain 0" in trace_names
        assert "Chain 1" in trace_names
        assert "Chain 2" in trace_names

    def test_histogram_showlegend_false(self):
        samples = np.random.randn(2, 50)
        fig = create_trace_plot(samples, "mu")
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        for h in histograms:
            assert h.showlegend is False


class TestCreatePosteriorPlot:
    """Tests for create_posterior_plot."""

    def test_returns_figure(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu")
        assert isinstance(fig, go.Figure)

    def test_title_contains_var_name(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "sigma")
        assert "sigma" in fig.layout.title.text

    def test_has_histogram_trace(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu")
        histograms = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histograms) == 1

    def test_2d_samples_flattened(self):
        samples = np.random.randn(4, 250)
        fig = create_posterior_plot(samples, "mu")
        assert isinstance(fig, go.Figure)

    def test_custom_hdi_prob(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu", hdi_prob=0.90)
        assert isinstance(fig, go.Figure)

    def test_showlegend_false(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu")
        assert fig.layout.showlegend is False

    def test_default_template(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu")
        assert fig.layout.template.layout.paper_bgcolor == "white"

    def test_dark_template(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu", template="aoty_dark")
        assert fig.layout.template.layout.paper_bgcolor == "#1E1E1E"

    def test_xaxis_label(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "beta")
        assert fig.layout.xaxis.title.text == "beta"

    def test_yaxis_label(self):
        samples = np.random.randn(1000)
        fig = create_posterior_plot(samples, "mu")
        assert fig.layout.yaxis.title.text == "Count"

    def test_hdi_bounds_match_arviz_for_skewed_samples(self):
        # For skewed posteriors ETI != HDI; the annotated bounds must be the
        # true HDI, consistent with the matplotlib publication figures.
        import arviz as az

        rng = np.random.default_rng(42)
        samples = rng.lognormal(mean=0.0, sigma=0.75, size=4000)
        expected = az.hdi(samples, hdi_prob=0.94)
        eti = np.percentile(samples, [3.0, 97.0])
        assert abs(expected[0] - eti[0]) > 0.01  # skew separates the two

        fig = create_posterior_plot(samples, "sigma", hdi_prob=0.94)
        # add_vline order: hdi lower, hdi upper, mean
        assert fig.layout.shapes[0].x0 == pytest.approx(expected[0])
        assert fig.layout.shapes[1].x0 == pytest.approx(expected[1])
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert any("HDI 94% lower" in t for t in annotation_texts)
        assert any("HDI 94% upper" in t for t in annotation_texts)


class TestCreatePredictionsPlot:
    """Tests for create_predictions_plot."""

    def test_returns_figure(self):
        y_true = np.array([70, 75, 80, 85])
        y_pred = np.array([72, 74, 78, 86])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 5, y_pred + 5)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self):
        y_true = np.array([70, 75, 80, 85])
        y_pred = np.array([72, 74, 78, 86])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 5, y_pred + 5)
        assert len(fig.data) == 2  # scatter + diagonal line

    def test_title(self):
        y_true = np.array([70, 75, 80])
        y_pred = np.array([72, 74, 78])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 3, y_pred + 3)
        assert "Predicted" in fig.layout.title.text

    def test_diagonal_reference_line(self):
        y_true = np.array([70, 75, 80])
        y_pred = np.array([72, 74, 78])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 3, y_pred + 3)
        diag = fig.data[1]
        assert diag.name == "Perfect Prediction"
        assert diag.line.dash == "dash"

    def test_custom_ci_label(self):
        y_true = np.array([70, 75])
        y_pred = np.array([72, 74])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 3, y_pred + 3, ci_label="90% CI")
        assert fig.data[0].name == "90% CI"

    def test_default_ci_label_is_generic(self):
        # No level information -> generic "CI", not a made-up percentage
        y_true = np.array([70, 75])
        y_pred = np.array([72, 74])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 3, y_pred + 3)
        assert fig.data[0].name == "CI"

    def test_accepts_list_input(self):
        fig = create_predictions_plot([70, 75], [72, 74], [67, 69], [77, 79])
        assert isinstance(fig, go.Figure)

    def test_axis_labels(self):
        y_true = np.array([70, 75])
        y_pred = np.array([72, 74])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 3, y_pred + 3)
        assert fig.layout.xaxis.title.text == "Predicted Score"
        assert fig.layout.yaxis.title.text == "Actual Score"

    def test_scatter_has_error_bars(self):
        y_true = np.array([70, 75])
        y_pred = np.array([72, 74])
        fig = create_predictions_plot(y_true, y_pred, y_pred - 3, y_pred + 3)
        scatter = fig.data[0]
        assert scatter.error_x.type == "data"
        assert scatter.error_x.symmetric is False


class TestCreateForestPlot:
    """Tests for create_forest_plot."""

    def test_returns_figure(self):
        df = pd.DataFrame(
            {
                "param": ["intercept", "slope"],
                "mean": [0.5, 1.2],
                "hdi_3%": [0.3, 0.9],
                "hdi_97%": [0.7, 1.5],
            }
        )
        fig = create_forest_plot(df)
        assert isinstance(fig, go.Figure)

    def test_title(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        fig = create_forest_plot(df)
        assert "Forest" in fig.layout.title.text or "Coefficient" in fig.layout.title.text

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "name": ["a", "b"],
                "estimate": [1.0, 2.0],
                "lower": [0.5, 1.5],
                "upper": [1.5, 2.5],
            }
        )
        fig = create_forest_plot(
            df,
            estimate_col="estimate",
            lower_col="lower",
            upper_col="upper",
            label_col="name",
        )
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        fig = create_forest_plot(df)
        scatters = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatters) >= 1

    def test_error_bars(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        fig = create_forest_plot(df)
        scatter = fig.data[0]
        assert scatter.error_x is not None
        assert scatter.error_x.symmetric is False

    def test_multiple_params(self):
        df = pd.DataFrame(
            {
                "param": ["a", "b", "c"],
                "mean": [1.0, 2.0, 3.0],
                "hdi_3%": [0.5, 1.5, 2.5],
                "hdi_97%": [1.5, 2.5, 3.5],
            }
        )
        fig = create_forest_plot(df)
        assert isinstance(fig, go.Figure)

    def test_dark_template(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        fig = create_forest_plot(df, template="aoty_dark")
        assert fig.layout.template.layout.paper_bgcolor == "#1E1E1E"

    def test_hover_label_matches_default_columns(self):
        # Default hdi_3%/hdi_97% columns are a 94% HDI, not 95%
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        fig = create_forest_plot(df)
        assert "94% HDI" in fig.data[0].hovertemplate
        assert "95% HDI" not in fig.data[0].hovertemplate

    def test_hover_label_derived_from_columns(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_2.5%": [0.5],
                "hdi_97.5%": [1.5],
            }
        )
        fig = create_forest_plot(df, lower_col="hdi_2.5%", upper_col="hdi_97.5%")
        assert "95% HDI" in fig.data[0].hovertemplate

    def test_hover_label_generic_when_level_unknown(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "lower": [0.5],
                "upper": [1.5],
            }
        )
        fig = create_forest_plot(df, lower_col="lower", upper_col="upper")
        assert "HDI: [" in fig.data[0].hovertemplate
        assert "% HDI" not in fig.data[0].hovertemplate

    def test_hdi_label_from_columns_helper(self):
        assert _hdi_label_from_columns("hdi_3%", "hdi_97%") == "94% HDI"
        assert _hdi_label_from_columns("hdi_2.5%", "hdi_97.5%") == "95% HDI"
        assert _hdi_label_from_columns("ci_lower", "ci_upper") == "HDI"
        assert _hdi_label_from_columns("hdi_97%", "hdi_3%") == "HDI"  # inverted


class TestCreateReliabilityPlot:
    """Tests for create_reliability_plot."""

    def test_returns_figure(self):
        pred = np.linspace(0.1, 0.9, 9)
        obs = pred + np.random.randn(9) * 0.01
        counts = np.ones(9) * 100
        fig = create_reliability_plot(pred, obs, counts)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self):
        pred = np.linspace(0.1, 0.9, 5)
        obs = pred
        counts = np.ones(5) * 50
        fig = create_reliability_plot(pred, obs, counts)
        assert len(fig.data) == 2  # scatter + diagonal

    def test_title(self):
        pred = np.array([0.2, 0.5, 0.8])
        obs = np.array([0.25, 0.45, 0.82])
        counts = np.array([10, 20, 30])
        fig = create_reliability_plot(pred, obs, counts)
        assert "Reliability" in fig.layout.title.text

    def test_diagonal_line(self):
        pred = np.array([0.2, 0.5])
        obs = np.array([0.25, 0.45])
        counts = np.array([10, 20])
        fig = create_reliability_plot(pred, obs, counts)
        diag = fig.data[1]
        assert diag.name == "Perfect Calibration"
        assert diag.line.dash == "dash"

    def test_axis_range(self):
        pred = np.array([0.2, 0.5])
        obs = np.array([0.25, 0.45])
        counts = np.array([10, 20])
        fig = create_reliability_plot(pred, obs, counts)
        assert fig.layout.xaxis.range[0] == -0.05
        assert fig.layout.xaxis.range[1] == 1.05

    def test_marker_sizes_scale_with_counts(self):
        pred = np.array([0.2, 0.5, 0.8])
        obs = np.array([0.25, 0.45, 0.82])
        counts = np.array([10, 100, 50])
        fig = create_reliability_plot(pred, obs, counts)
        sizes = fig.data[0].marker.size
        # Largest count should have largest marker
        assert sizes[1] > sizes[0]

    def test_zero_counts_no_division_error(self):
        pred = np.array([0.5])
        obs = np.array([0.5])
        counts = np.array([0])
        fig = create_reliability_plot(pred, obs, counts)
        assert isinstance(fig, go.Figure)

    def test_single_bin(self):
        pred = np.array([0.5])
        obs = np.array([0.4])
        counts = np.array([100])
        fig = create_reliability_plot(pred, obs, counts)
        assert isinstance(fig, go.Figure)

    def test_accepts_list_input(self):
        fig = create_reliability_plot([0.2, 0.5], [0.25, 0.45], [10, 20])
        assert isinstance(fig, go.Figure)

    def test_custom_template(self):
        pred = np.array([0.2, 0.5])
        obs = np.array([0.25, 0.45])
        counts = np.array([10, 20])
        fig = create_reliability_plot(pred, obs, counts, template="aoty_dark")
        assert fig.layout.template.layout.paper_bgcolor == "#1E1E1E"
