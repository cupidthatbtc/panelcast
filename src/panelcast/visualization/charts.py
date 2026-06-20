"""Interactive Plotly chart creation functions for model visualization.

This module provides functions to create interactive Plotly figures for
MCMC diagnostics, posterior distributions, predictions, and calibration.
These mirror the matplotlib-based figures in reporting/figures.py but
add interactivity (hover, zoom, pan).

All charts use the custom aoty_light/aoty_dark templates registered in
theme.py, with consistent colorblind-safe styling.

Usage:
    >>> from panelcast.visualization.charts import create_trace_plot
    >>> import numpy as np
    >>> samples = np.random.randn(4, 1000)  # 4 chains, 1000 draws
    >>> fig = create_trace_plot(samples, "mu")
    >>> fig.show()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from panelcast.visualization.theme import COLORBLIND_COLORS

__all__ = [
    "create_forest_plot",
    "create_posterior_plot",
    "create_predictions_plot",
    "create_reliability_plot",
    "create_trace_plot",
]


def create_trace_plot(
    samples: np.ndarray,
    var_name: str,
    template: str = "aoty_light",
) -> go.Figure:
    """Create interactive trace plot for MCMC diagnostics.

    Shows trace (left panel, 60%) and density histogram (right panel, 40%)
    for MCMC samples across multiple chains.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples with shape (chains, draws).
    var_name : str
        Variable name for plot title and axis labels.
    template : str, default "aoty_light"
        Plotly template name ("aoty_light" or "aoty_dark").

    Returns
    -------
    go.Figure
        Interactive Plotly figure.

    Examples
    --------
    >>> samples = np.random.randn(4, 1000)  # 4 chains, 1000 draws
    >>> fig = create_trace_plot(samples, "mu")
    >>> fig.show()
    """
    samples = np.asarray(samples)
    n_chains, n_draws = samples.shape

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=["Trace", "Density"],
    )

    # Trace plot (left panel)
    for chain in range(n_chains):
        color = COLORBLIND_COLORS[chain % len(COLORBLIND_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=np.arange(n_draws),
                y=samples[chain],
                mode="lines",
                name=f"Chain {chain}",
                line=dict(color=color, width=0.5),
                opacity=0.7,
                hovertemplate=(
                    f"<b>Chain {chain}</b><br>"
                    "Iteration: %{x}<br>"
                    f"{var_name}: %{{y:.4f}}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Density histogram (right panel)
    for chain in range(n_chains):
        color = COLORBLIND_COLORS[chain % len(COLORBLIND_COLORS)]
        fig.add_trace(
            go.Histogram(
                x=samples[chain],
                name=f"Chain {chain}",
                marker=dict(color=color),
                opacity=0.5,
                showlegend=False,
                hovertemplate=(
                    f"<b>Chain {chain}</b><br>"
                    f"{var_name}: %{{x:.4f}}<br>"
                    "Count: %{y}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=f"Trace Plot: {var_name}",
        template=template,
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text=var_name, row=1, col=2)
    fig.update_yaxes(title_text=var_name, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return fig


def create_posterior_plot(
    samples: np.ndarray,
    var_name: str,
    hdi_prob: float = 0.94,
    template: str = "aoty_light",
) -> go.Figure:
    """Create interactive posterior distribution plot with HDI annotation.

    Shows histogram of posterior samples with vertical lines marking
    the HDI bounds and mean.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (n_samples,) or flattened from chains.
    var_name : str
        Variable name for plot title and axis labels.
    hdi_prob : float, default 0.94
        Probability for highest density interval (e.g., 0.94 for 94% HDI).
    template : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.

    Examples
    --------
    >>> samples = np.random.randn(4000)
    >>> fig = create_posterior_plot(samples, "mu", hdi_prob=0.94)
    >>> fig.show()
    """
    samples = np.asarray(samples).ravel()

    # Calculate HDI bounds using percentiles
    lower_q = (1 - hdi_prob) / 2
    upper_q = 1 - lower_q
    hdi_lower = np.percentile(samples, lower_q * 100)
    hdi_upper = np.percentile(samples, upper_q * 100)
    mean_val = np.mean(samples)

    fig = go.Figure()

    # Histogram of posterior
    fig.add_trace(
        go.Histogram(
            x=samples,
            name="Posterior",
            marker=dict(color=COLORBLIND_COLORS[0]),
            opacity=0.7,
            hovertemplate=(f"{var_name}: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>"),
        )
    )

    # HDI lower bound line
    fig.add_vline(
        x=hdi_lower,
        line=dict(color=COLORBLIND_COLORS[1], width=2, dash="dash"),
        annotation_text=f"HDI {hdi_prob * 100:.0f}% lower: {hdi_lower:.3f}",
        annotation_position="top left",
    )

    # HDI upper bound line
    fig.add_vline(
        x=hdi_upper,
        line=dict(color=COLORBLIND_COLORS[1], width=2, dash="dash"),
        annotation_text=f"HDI {hdi_prob * 100:.0f}% upper: {hdi_upper:.3f}",
        annotation_position="top right",
    )

    # Mean line
    fig.add_vline(
        x=mean_val,
        line=dict(color=COLORBLIND_COLORS[2], width=2),
        annotation_text=f"Mean: {mean_val:.3f}",
        annotation_position="top",
    )

    fig.update_layout(
        title=f"Posterior Distribution: {var_name}",
        xaxis_title=var_name,
        yaxis_title="Count",
        template=template,
        showlegend=False,
    )

    return fig


def create_predictions_plot(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    ci_label: str = "94% CI",
    template: str = "aoty_light",
) -> go.Figure:
    """Create predicted vs actual scatter plot with uncertainty intervals.

    Shows predictions on x-axis, actuals on y-axis with horizontal error
    bars for credible intervals and a diagonal reference line.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_pred_mean : np.ndarray
        Predicted values (posterior mean), shape (n_obs,).
    y_pred_lower : np.ndarray
        Lower bound of credible interval, shape (n_obs,).
    y_pred_upper : np.ndarray
        Upper bound of credible interval, shape (n_obs,).
    ci_label : str, default "94% CI"
        Label for the credible interval in legend/hover.
    template : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.

    Examples
    --------
    >>> y_true = np.array([70, 75, 80, 85])
    >>> y_pred_mean = np.array([72, 74, 78, 86])
    >>> y_pred_lower = y_pred_mean - 5
    >>> y_pred_upper = y_pred_mean + 5
    >>> fig = create_predictions_plot(y_true, y_pred_mean, y_pred_lower, y_pred_upper)
    >>> fig.show()
    """
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_lower = np.asarray(y_pred_lower)
    y_pred_upper = np.asarray(y_pred_upper)

    # Calculate asymmetric error bars
    error_minus = y_pred_mean - y_pred_lower
    error_plus = y_pred_upper - y_pred_mean

    fig = go.Figure()

    # Prediction points with error bars
    fig.add_trace(
        go.Scatter(
            x=y_pred_mean,
            y=y_true,
            mode="markers",
            marker=dict(size=8, color=COLORBLIND_COLORS[0], opacity=0.6),
            error_x=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color=COLORBLIND_COLORS[5],  # Light blue
                thickness=1,
            ),
            name=ci_label,
            hovertemplate=(
                "Predicted: %{x:.1f}<br>"
                "Actual: %{y:.1f}<br>"
                f"<b>{ci_label}</b>: [%{{customdata[0]:.1f}}, %{{customdata[1]:.1f}}]"
                "<extra></extra>"
            ),
            customdata=np.column_stack([y_pred_lower, y_pred_upper]),
        )
    )

    # Diagonal reference line (perfect prediction)
    all_vals = np.concatenate([y_true, y_pred_mean, y_pred_lower, y_pred_upper])
    min_val = np.min(all_vals) - 2
    max_val = np.max(all_vals) + 2

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect Prediction",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Predicted vs Actual Scores",
        xaxis_title="Predicted Score",
        yaxis_title="Actual Score",
        template=template,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[min_val, max_val]),
        yaxis=dict(range=[min_val, max_val]),
    )

    return fig


def create_forest_plot(
    df: pd.DataFrame,
    estimate_col: str = "mean",
    lower_col: str = "hdi_3%",
    upper_col: str = "hdi_97%",
    label_col: str = "param",
    template: str = "aoty_light",
) -> go.Figure:
    """Create interactive forest plot for coefficient visualization.

    Shows horizontal error bars for coefficient estimates with HDI intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for parameter name, estimate, and HDI bounds.
    estimate_col : str, default "mean"
        Column name for point estimate.
    lower_col : str, default "hdi_3%"
        Column name for lower HDI bound.
    upper_col : str, default "hdi_97%"
        Column name for upper HDI bound.
    label_col : str, default "param"
        Column name for parameter labels.
    template : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'param': ['intercept', 'slope'],
    ...     'mean': [0.5, 1.2],
    ...     'hdi_3%': [0.3, 0.9],
    ...     'hdi_97%': [0.7, 1.5]
    ... })
    >>> fig = create_forest_plot(df)
    >>> fig.show()
    """
    # Calculate asymmetric error bars
    error_minus = df[estimate_col] - df[lower_col]
    error_plus = df[upper_col] - df[estimate_col]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[estimate_col],
            y=df[label_col],
            mode="markers",
            marker=dict(size=10, color=COLORBLIND_COLORS[0]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color=COLORBLIND_COLORS[0],
                thickness=1.5,
                width=5,
            ),
            name="Estimate",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Estimate: %{x:.3f}<br>"
                "95% HDI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            ),
            customdata=df[[lower_col, upper_col]].values,
        )
    )

    # Vertical line at zero
    fig.add_vline(
        x=0,
        line=dict(dash="dash", color="gray", width=1),
        opacity=0.7,
    )

    fig.update_layout(
        title="Coefficient Forest Plot",
        xaxis_title="Coefficient Value",
        yaxis_title="",
        template=template,
    )

    return fig


def create_reliability_plot(
    predicted_probs: np.ndarray,
    observed_freq: np.ndarray,
    counts: np.ndarray,
    template: str = "aoty_light",
) -> go.Figure:
    """Create interactive reliability diagram (calibration plot).

    Shows predicted probability vs observed frequency with marker sizes
    proportional to bin counts.

    Parameters
    ----------
    predicted_probs : np.ndarray
        Bin centers or predicted probabilities, shape (n_bins,).
    observed_freq : np.ndarray
        Observed frequencies in each bin, shape (n_bins,).
    counts : np.ndarray
        Number of observations in each bin, shape (n_bins,).
    template : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.

    Examples
    --------
    >>> predicted = np.linspace(0.1, 0.9, 9)
    >>> observed = predicted + np.random.randn(9) * 0.05
    >>> counts = np.ones(9) * 100
    >>> fig = create_reliability_plot(predicted, observed, counts)
    >>> fig.show()
    """
    predicted_probs = np.asarray(predicted_probs)
    observed_freq = np.asarray(observed_freq)
    counts = np.asarray(counts)

    # Normalize counts for marker sizing
    max_count = max(counts.max(), 1)  # Avoid division by zero
    size_scale = 30 * counts / max_count + 10  # Size range [10, 40]

    fig = go.Figure()

    # Scatter plot with size proportional to counts
    fig.add_trace(
        go.Scatter(
            x=predicted_probs,
            y=observed_freq,
            mode="markers+lines",
            marker=dict(
                size=size_scale,
                color=COLORBLIND_COLORS[0],
                opacity=0.7,
            ),
            line=dict(color=COLORBLIND_COLORS[0], width=1),
            name="Observed",
            hovertemplate=(
                "Predicted: %{x:.3f}<br>Observed: %{y:.3f}<br>Count: %{customdata}<extra></extra>"
            ),
            customdata=counts,
        )
    )

    # Diagonal reference line (perfect calibration)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect Calibration",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Reliability Diagram",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        template=template,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
    )

    return fig
