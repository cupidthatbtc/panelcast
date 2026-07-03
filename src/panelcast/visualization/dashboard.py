"""Dashboard assembly functions for multi-chart visualization.

This module provides functions to assemble multiple interactive charts
into dashboard views. It integrates with the chart functions from
charts.py and provides data structures for dashboard state.

Usage:
    >>> from panelcast.visualization.dashboard import DashboardData, create_dashboard_figures
    >>> data = DashboardData(predictions={...}, coefficients=df)
    >>> figures = create_dashboard_figures(data)
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from panelcast.paths import resolve_latest
from panelcast.visualization.charts import (
    create_forest_plot,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DashboardData",
    "create_artist_view",
    "create_coefficients_table",
    "create_dashboard_figures",
    "get_artist_list",
    "load_dashboard_data",
]


@dataclass
class DashboardData:
    """Data container for dashboard visualization.

    This dataclass holds all data needed to generate dashboard views.
    All fields are optional - views are only generated for available data.

    Attributes
    ----------
    idata : az.InferenceData | None
        ArviZ inference data object with posterior samples.
    predictions : dict | None
        Prediction results with keys:
        - y_true: array of actual values
        - y_pred_mean: array of predicted means
        - y_pred_lower: array of lower CI bounds
        - y_pred_upper: array of upper CI bounds
    coefficients : pd.DataFrame | None
        Coefficient summary table with columns:
        - param: parameter name
        - mean: posterior mean
        - hdi_3%: lower HDI bound
        - hdi_97%: upper HDI bound
    reliability : dict | None
        Calibration data with keys:
        - predicted_probs: array of bin centers
        - observed_freq: array of observed frequencies
        - counts: array of bin counts
    artist_data : pd.DataFrame | None
        Per-artist predictions for artist search view.
        Should have 'artist' column and prediction columns.
    """

    idata: Any | None = None  # az.InferenceData
    predictions: dict[str, np.ndarray] | None = None
    coefficients: pd.DataFrame | None = None
    reliability: dict[str, np.ndarray] | None = None
    artist_data: pd.DataFrame | None = field(default=None)


def create_dashboard_figures(
    data: DashboardData,
    theme: str = "aoty_light",
) -> dict[str, str]:
    """Generate all dashboard figures as HTML strings.

    Creates HTML representations of each available view based on
    the data provided. Only views with corresponding data are generated.

    Parameters
    ----------
    data : DashboardData
        Dashboard data container.
    theme : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    dict[str, str]
        Dictionary mapping view_id to Plotly HTML string.
        Possible keys: "trace", "predictions", "coefficients", "reliability"

    Notes
    -----
    Only the first figure includes Plotly.js to minimize total size.
    Subsequent figures use include_plotlyjs=False.

    Examples
    --------
    >>> data = DashboardData(predictions={...})
    >>> figures = create_dashboard_figures(data)
    >>> print(list(figures.keys()))
    ['predictions']
    """
    figures: dict[str, str] = {}
    first_figure = True

    # Trace plot (from idata posterior)
    if data.idata is not None:
        try:
            # Get first parameter from posterior
            posterior = data.idata.posterior
            if hasattr(posterior, "data_vars"):
                var_names = list(posterior.data_vars)
                if var_names:
                    var_name = var_names[0]
                    samples = posterior[var_name].values
                    # Handle multi-dimensional samples
                    if samples.ndim > 2:
                        samples = samples.reshape(samples.shape[0], -1)[:, 0:100]
                    elif samples.ndim == 1:
                        samples = samples.reshape(1, -1)

                    fig = create_trace_plot(samples, var_name, template=theme)
                    figures["trace"] = fig.to_html(
                        full_html=False,
                        include_plotlyjs=first_figure,
                    )
                    first_figure = False
        except Exception as e:
            logger.debug("Skipping trace plot due to unexpected idata format: %s", e)

    # Predictions scatter plot
    if data.predictions is not None:
        pred = data.predictions
        required = ["y_true", "y_pred_mean", "y_pred_lower", "y_pred_upper"]
        if all(k in pred for k in required):
            fig = create_predictions_plot(
                pred["y_true"],
                pred["y_pred_mean"],
                pred["y_pred_lower"],
                pred["y_pred_upper"],
                template=theme,
            )
            figures["predictions"] = fig.to_html(
                full_html=False,
                include_plotlyjs=first_figure,
            )
            first_figure = False

    # Coefficient forest plot
    if data.coefficients is not None:
        df = data.coefficients
        # Auto-detect column names
        estimate_col = _find_column(df, ["mean", "estimate", "coef"])
        lower_col = _find_column(df, ["hdi_3%", "hdi_2.5%", "lower", "ci_lower"])
        upper_col = _find_column(df, ["hdi_97%", "hdi_97.5%", "upper", "ci_upper"])
        label_col = _find_column(df, ["param", "parameter", "name", "index"])

        if all([estimate_col, lower_col, upper_col, label_col]):
            fig = create_forest_plot(
                df,
                estimate_col=estimate_col,
                lower_col=lower_col,
                upper_col=upper_col,
                label_col=label_col,
                template=theme,
            )
            figures["coefficients"] = fig.to_html(
                full_html=False,
                include_plotlyjs=first_figure,
            )
            first_figure = False

    # Reliability diagram
    if data.reliability is not None:
        rel = data.reliability
        required = ["predicted_probs", "observed_freq", "counts"]
        if all(k in rel for k in required):
            fig = create_reliability_plot(
                rel["predicted_probs"],
                rel["observed_freq"],
                rel["counts"],
                template=theme,
            )
            figures["reliability"] = fig.to_html(
                full_html=False,
                include_plotlyjs=first_figure,
            )
            first_figure = False

    return figures


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_artist_view(
    artist_name: str,
    artist_data: pd.DataFrame,
    theme: str = "aoty_light",
) -> str:
    """Generate artist-specific view showing prediction history.

    Creates a line chart showing album scores over time with
    prediction intervals for the specified artist.

    Parameters
    ----------
    artist_name : str
        Name of the artist to display.
    artist_data : pd.DataFrame
        DataFrame with artist predictions. Expected columns:
        - artist: artist name
        - date or year: time column
        - score or y_true: actual score
        - prediction or y_pred: predicted score (optional)
        - lower or y_pred_lower: lower bound (optional)
        - upper or y_pred_upper: upper bound (optional)
    theme : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    str
        Plotly HTML string for the artist view, or an informative
        message if the artist is not found.

    Examples
    --------
    >>> html = create_artist_view("Radiohead", artist_data)
    >>> print(html[:50])
    '<div id="...'
    """
    import plotly.graph_objects as go

    # Filter to artist
    artist_mask = artist_data["artist"].str.lower() == artist_name.lower()
    artist_df = artist_data[artist_mask].copy()

    if artist_df.empty:
        return f'<div class="not-found">Artist "{html.escape(artist_name)}" not found.</div>'

    # Auto-detect columns
    time_col = _find_column(artist_df, ["date", "year", "release_date", "time"])
    score_col = _find_column(artist_df, ["score", "y_true", "actual", "user_score"])
    pred_col = _find_column(artist_df, ["prediction", "y_pred", "y_pred_mean", "predicted"])
    lower_col = _find_column(artist_df, ["lower", "y_pred_lower", "ci_lower"])
    upper_col = _find_column(artist_df, ["upper", "y_pred_upper", "ci_upper"])

    # Sort by time if available
    if time_col:
        artist_df = artist_df.sort_values(time_col)
        x_data = artist_df[time_col]
    else:
        x_data = np.arange(len(artist_df))

    fig = go.Figure()

    # Add prediction interval if available
    if lower_col and upper_col and pred_col:
        fig.add_trace(
            go.Scatter(
                x=list(x_data) + list(x_data[::-1]),
                y=list(artist_df[upper_col]) + list(artist_df[lower_col][::-1]),
                fill="toself",
                fillcolor="rgba(0, 114, 178, 0.2)",
                line=dict(color="rgba(0, 114, 178, 0)"),
                name="94% CI",
                hoverinfo="skip",
            )
        )

    # Add actual scores
    if score_col:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=artist_df[score_col],
                mode="lines+markers",
                name="Actual Score",
                line=dict(color="#0072B2", width=2),
                marker=dict(size=8),
                hovertemplate="Actual: %{y:.1f}<extra></extra>",
            )
        )

    # Add predictions
    if pred_col:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=artist_df[pred_col],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#E69F00", dash="dash", width=2),
                marker=dict(size=6),
                hovertemplate="Predicted: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Album Scores: {artist_name}",
        xaxis_title=time_col.title() if time_col else "Album Index",
        yaxis_title="Score",
        template=theme,
    )

    return fig.to_html(full_html=False, include_plotlyjs=True)


def get_artist_list(artist_data: pd.DataFrame) -> list[str]:
    """Return sorted list of unique artist names for autocomplete.

    Parameters
    ----------
    artist_data : pd.DataFrame
        DataFrame with 'artist' column.

    Returns
    -------
    list[str]
        Sorted list of unique artist names.

    Examples
    --------
    >>> df = pd.DataFrame({"artist": ["Radiohead", "Beatles", "Radiohead"]})
    >>> get_artist_list(df)
    ['Beatles', 'Radiohead']
    """
    if "artist" not in artist_data.columns:
        return []
    artists = artist_data["artist"].dropna().unique().tolist()
    return sorted(artists)


def create_coefficients_table(coefficients: pd.DataFrame) -> str:
    """Generate sortable HTML table of coefficients.

    Creates a simple HTML table with data attributes for JavaScript
    sorting. Numbers are formatted to 3 decimal places.

    Parameters
    ----------
    coefficients : pd.DataFrame
        Coefficient summary with columns for parameter name, mean,
        and HDI bounds.

    Returns
    -------
    str
        HTML table string.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "param": ["intercept", "slope"],
    ...     "mean": [0.5, 1.2],
    ...     "hdi_3%": [0.3, 0.9],
    ...     "hdi_97%": [0.7, 1.5]
    ... })
    >>> html = create_coefficients_table(df)
    """
    # Auto-detect columns
    label_col = _find_column(coefficients, ["param", "parameter", "name", "index"])
    estimate_col = _find_column(coefficients, ["mean", "estimate", "coef"])
    lower_col = _find_column(coefficients, ["hdi_3%", "hdi_2.5%", "lower", "ci_lower"])
    upper_col = _find_column(coefficients, ["hdi_97%", "hdi_97.5%", "upper", "ci_upper"])

    if not all([label_col, estimate_col, lower_col, upper_col]):
        return "<p>Unable to generate table: missing required columns.</p>"

    html_parts = [
        '<table class="coefficient-table" id="coef-table">',
        "<thead>",
        "<tr>",
        '    <th data-sort="string">Parameter</th>',
        '    <th data-sort="number">Mean</th>',
        '    <th data-sort="number">HDI Low</th>',
        '    <th data-sort="number">HDI High</th>',
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    for _, row in coefficients.iterrows():
        html_parts.append("<tr>")
        html_parts.append(f"    <td>{html.escape(str(row[label_col]))}</td>")
        html_parts.append(
            f'    <td data-value="{row[estimate_col]:.6f}">{row[estimate_col]:.3f}</td>'
        )
        html_parts.append(f'    <td data-value="{row[lower_col]:.6f}">{row[lower_col]:.3f}</td>')
        html_parts.append(f'    <td data-value="{row[upper_col]:.6f}">{row[upper_col]:.3f}</td>')
        html_parts.append("</tr>")

    html_parts.extend(
        [
            "</tbody>",
            "</table>",
        ]
    )

    return "\n".join(html_parts)


def load_dashboard_data(run_dir: Path | None = None) -> DashboardData:  # noqa: C901  # tracked complexity debt
    """Load dashboard data from the most recent pipeline run or specified directory.

    Parameters
    ----------
    run_dir : Path | None, default None
        Path to a specific pipeline run directory (e.g., outputs/2026-01-19_143052).
        If None, looks for the most recent run in outputs/.

    Returns
    -------
    DashboardData
        Dataclass containing loaded data for dashboard views.
        Fields are None if corresponding data is not found.

    Notes
    -----
    Looks for:
    - InferenceData: models/*.nc or .json
    - Predictions: evaluation results
    - Coefficients: reports/tables/*.csv
    - Artist data: data/processed/*.parquet
    """

    # Start with empty data
    data = DashboardData()

    # Determine run directory: the latest pointer first, then the legacy
    # scan over timestamped dirs for outputs written by older checkouts.
    if run_dir is None:
        run_dir = resolve_latest()
        if run_dir is not None:
            logger.info("Using latest run: %s", run_dir)
    if run_dir is None:
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            run_dirs = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
                key=lambda d: d.name,
                reverse=True,
            )
            if run_dirs:
                run_dir = run_dirs[0]
                logger.info("Using most recent run: %s", run_dir)

    # Try to load inference data
    try:
        import arviz as az

        model_files = []
        if run_dir is not None:
            # Look for NetCDF files in run directory
            model_files = sorted(
                run_dir.glob("models/*.nc"), key=lambda f: f.stat().st_mtime, reverse=True
            )
            if not model_files:
                model_files = sorted(
                    run_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime, reverse=True
                )

        # Fallback: check models/ directory relative to project root
        if not model_files:
            # Find project root by looking for pyproject.toml or .git
            project_root = Path.cwd()
            for parent in [Path.cwd()] + list(Path.cwd().parents):
                if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                    project_root = parent
                    break
            models_dir = project_root / "models"
            if models_dir.exists():
                model_files = sorted(
                    models_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime, reverse=True
                )
                if model_files:
                    logger.info("Using fallback models directory: %s", models_dir)

        if model_files:
            data.idata = az.from_netcdf(model_files[0])
            logger.info("Loaded inference data from %s", model_files[0])
    except (FileNotFoundError, ValueError, OSError, ImportError, TypeError) as e:
        logger.warning("Could not load inference data: %s", e)

    # Evaluation artifacts live under the run dir in the run-scoped layout;
    # the flat path keeps pre-run-scoping outputs readable.
    eval_dir = Path("outputs/evaluation")
    if run_dir is not None and (run_dir / "evaluation").exists():
        eval_dir = run_dir / "evaluation"
    try:
        pred_path = eval_dir / "predictions.json"
        if pred_path.exists():
            import json

            with open(pred_path) as f:
                pred = json.load(f)

            required = ["y_true", "y_pred_mean", "y_pred_lower", "y_pred_upper"]
            if all(k in pred for k in required):
                data.predictions = {
                    "y_true": np.array(pred["y_true"]),
                    "y_pred_mean": np.array(pred["y_pred_mean"]),
                    "y_pred_lower": np.array(pred["y_pred_lower"]),
                    "y_pred_upper": np.array(pred["y_pred_upper"]),
                }
                logger.info("Loaded predictions from %s", pred_path)
    except Exception as e:
        logger.debug("Could not load predictions: %s", e)

    # Try to load coefficient tables
    try:
        # Check reports/tables/ at project root first, then inside run dir
        search_dirs = [Path("reports/tables")]
        if run_dir is not None:
            search_dirs.append(run_dir / "reports" / "tables")
            search_dirs.append(run_dir / "tables")

        for table_dir in search_dirs:
            if not table_dir.exists():
                continue
            table_files = list(table_dir.glob("*coefficient*.csv"))
            if not table_files:
                table_files = list(table_dir.glob("*summary*.csv"))
            if table_files:
                data.coefficients = pd.read_csv(table_files[0])
                logger.info("Loaded coefficients from %s", table_files[0])
                break
    except Exception as e:
        logger.debug("Could not load coefficients: %s", e)

    # Try to load calibration/reliability data
    try:
        cal_path = eval_dir / "calibration.json"
        if cal_path.exists():
            import json

            with open(cal_path) as f:
                cal_data = json.load(f)

            if all(k in cal_data for k in ["predicted_probs", "observed_freq", "counts"]):
                data.reliability = {
                    "predicted_probs": np.array(cal_data["predicted_probs"]),
                    "observed_freq": np.array(cal_data["observed_freq"]),
                    "counts": np.array(cal_data["counts"]),
                }
                logger.info("Loaded calibration data from %s", cal_path)
    except Exception as e:
        logger.debug("Could not load calibration data: %s", e)

    # Try to load artist data from processed directory
    try:
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            # Prefer user score data
            artist_files = list(processed_dir.glob("*user_score*.parquet"))
            if not artist_files:
                artist_files = list(processed_dir.glob("cleaned*.parquet"))
            if artist_files:
                df = pd.read_parquet(artist_files[0])
                # Ensure we have an artist column
                artist_col = None
                for col in ["artist", "Artist", "ARTIST"]:
                    if col in df.columns:
                        artist_col = col
                        break
                if artist_col:
                    if artist_col != "artist":
                        df = df.rename(columns={artist_col: "artist"})
                    data.artist_data = df
                    logger.info("Loaded artist data from %s", artist_files[0])
    except Exception as e:
        logger.debug("Could not load artist data: %s", e)

    return data
