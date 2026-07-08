"""Publication-quality figure generation for Bayesian model results.

This module provides functions for generating trace plots, posterior distributions,
prediction plots, and calibration diagrams suitable for journal publication.
All figures use colorblind-safe palettes and export to both PDF (vector) and
PNG (300dpi raster) formats.

Key features:
- Consistent publication styling via context manager
- Colorblind-safe color palette (Wong, 2011)
- Dual-format export (PDF + PNG)
- Automatic figure sizing based on content
- Proper memory management (figures closed after saving)

Usage:
    >>> from panelcast.reporting.figures import (
    ...     set_publication_style,
    ...     save_trace_plot,
    ...     save_predictions_plot,
    ... )
    >>> with set_publication_style():
    ...     pdf, png = save_trace_plot(idata, ["mu"], Path("figs"), "trace")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator

    from panelcast.evaluation.calibration import ReliabilityData

__all__ = [
    "COLORBLIND_COLORS",
    "get_trace_plot_vars",
    "save_artist_prediction_plot",
    "save_forest_plot",
    "save_ppc_density_plot",
    "save_posterior_plot",
    "save_predictions_plot",
    "save_rank_scatter_plot",
    "save_reliability_plot",
    "save_slice_coverage_plot",
    "save_trace_plot",
    "select_artist_subsets",
    "set_publication_style",
]

# Colorblind-safe palette from Wong (2011), Nature Methods
# https://www.nature.com/articles/nmeth.1618
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light blue
    "#D55E00",  # Red-orange
]


def get_trace_plot_vars(
    idata: az.InferenceData,
    prefix: str = "user_",
    include_hyperpriors: bool = True,
) -> list[str]:
    """Get variable names for trace plots, with dynamic sigma_ref and n_exponent detection.

    Dynamically builds the list of variables based on what exists in the posterior,
    handling both homoscedastic models (no n_exponent) and heteroscedastic models
    with learned exponent (has n_exponent). When sigma_ref reparameterization is
    active, sigma_ref is inserted before sigma_obs for logical ordering.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    prefix : str, default "user_"
        Parameter name prefix ("user_" or "critic_").
    include_hyperpriors : bool, default True
        If True, include population-level hyperpriors (mu_artist, sigma_artist, etc.).
        If False, only include observation-level parameters.

    Returns
    -------
    list[str]
        Variable names to include in trace plots.

    Example
    -------
    >>> var_names = get_trace_plot_vars(idata, prefix="user_")
    >>> pdf, png = save_trace_plot(idata, var_names, Path("figs"), "trace")
    """
    # Base variables that always exist
    base_vars = [
        f"{prefix}sigma_obs",
        f"{prefix}rho",
    ]

    if include_hyperpriors:
        hyperprior_vars = [
            f"{prefix}mu_artist",
            f"{prefix}sigma_artist",
            f"{prefix}sigma_rw",
        ]
        base_vars = hyperprior_vars + base_vars

    # Add sigma_ref if present (sigma-ref reparameterization mode)
    # Insert before sigma_obs for logical ordering (sampled before derived)
    if f"{prefix}sigma_ref" in idata.posterior:
        obs_idx = base_vars.index(f"{prefix}sigma_obs")
        base_vars.insert(obs_idx, f"{prefix}sigma_ref")

    # Add n_exponent only if it exists in posterior (learned mode)
    if f"{prefix}n_exponent" in idata.posterior:
        base_vars.append(f"{prefix}n_exponent")

    return base_vars


@contextmanager
def set_publication_style() -> Generator[None, None, None]:
    """Context manager for publication-quality figure styling.

    Uses plt.rc_context() to avoid global state pollution. All rcParams
    are restored after the context exits.

    Style settings:
    - Serif font family (Times New Roman with DejaVu Serif fallback)
    - Font sizes: 9pt body, 10pt titles, 8pt legends/ticks
    - DPI: 100 for screen, 300 for savefig
    - PDF/PS fonttype 42 for vector font embedding
    - Removed top/right spines for cleaner appearance
    - Colorblind-safe color cycle

    Examples
    --------
    >>> with set_publication_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3], [1, 4, 9])
    ...     fig.savefig("plot.pdf")
    ...     plt.close(fig)
    """
    style_params = {
        # Font settings
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Figure settings
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.figsize": (6.5, 4),  # Single column width
        # Export settings for vector compatibility
        "pdf.fonttype": 42,  # TrueType fonts for Illustrator
        "ps.fonttype": 42,
        # Line settings
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        # Clean appearance
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Colorblind-safe color cycle
        "axes.prop_cycle": plt.cycler("color", COLORBLIND_COLORS),
    }

    with plt.rc_context(style_params):
        yield


def _ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists, create if missing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_dual_format(
    fig: plt.Figure,
    output_dir: Path,
    filename_base: str,
) -> tuple[Path, Path]:
    """Save figure in both PDF and PNG formats.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_dir : Path
        Output directory.
    filename_base : str
        Base filename without extension.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).
    """
    output_dir = _ensure_output_dir(output_dir)

    pdf_path = output_dir / f"{filename_base}.pdf"
    png_path = output_dir / f"{filename_base}.png"

    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", format="png", dpi=300)

    return pdf_path, png_path


def save_trace_plot(
    idata: az.InferenceData,
    var_names: list[str],
    output_dir: Path,
    filename_base: str,
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save MCMC trace plot in PDF and PNG formats.

    Creates side-by-side trace plots (chain traces and posterior density)
    using ArviZ's plot_trace function.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    var_names : list[str]
        Variable names to include in the trace plot.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension (e.g., "trace_user_score").
    figsize : tuple[float, float], optional
        Figure size in inches (width, height). If None, auto-sizes
        based on number of variables: (10, 2 * n_vars).

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> pdf, png = save_trace_plot(
    ...     idata, ["mu", "sigma"], Path("figs"), "trace_main"
    ... )
    >>> print(f"Created: {pdf}")
    """
    with set_publication_style():
        # Auto-size figure based on number of variables
        if figsize is None:
            figsize = (10, 2 * len(var_names))

        # Create trace plot with ArviZ
        axes = az.plot_trace(
            idata,
            var_names=var_names,
            compact=True,
            divergences="bottom",
            figsize=figsize,
        )

        # Get figure from axes array
        fig = axes.ravel()[0].figure
        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up to avoid memory leaks
        plt.close(fig)

    return pdf_path, png_path


def save_posterior_plot(
    idata: az.InferenceData,
    var_names: list[str],
    output_dir: Path,
    filename_base: str,
    hdi_prob: float = 0.94,
    point_estimate: str = "mean",
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save posterior distribution plot in PDF and PNG formats.

    Creates posterior density plots with HDI and point estimate annotations
    using ArviZ's plot_posterior function.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    var_names : list[str]
        Variable names to include in the plot.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    hdi_prob : float, default 0.94
        Probability for highest density interval.
    point_estimate : str, default "mean"
        Point estimate to display ("mean", "median", or "mode").
    figsize : tuple[float, float], optional
        Figure size in inches. If None, uses ArviZ default.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> pdf, png = save_posterior_plot(
    ...     idata, ["mu", "sigma"], Path("figs"), "posterior_main",
    ...     hdi_prob=0.95
    ... )
    """
    with set_publication_style():
        # Create posterior plot with ArviZ
        plot_kwargs = {
            "hdi_prob": hdi_prob,
            "point_estimate": point_estimate,
        }
        if figsize is not None:
            plot_kwargs["figsize"] = figsize

        axes = az.plot_posterior(
            idata,
            var_names=var_names,
            **plot_kwargs,
        )

        # Handle single variable case (returns single Axes, not array)
        if isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = axes.figure

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_predictions_plot(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    output_dir: Path,
    filename_base: str,
    ci_label: str = "94% CI",
    figsize: tuple[float, float] = (6, 6),
) -> tuple[Path, Path]:
    """Generate and save predicted vs actual scatter plot with uncertainty bands.

    Creates a scatter plot comparing predictions to actual values, with
    uncertainty intervals shown as error bars or bands.

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
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    ci_label : str, default "94% CI"
        Label for the credible interval in legend.
    figsize : tuple[float, float], default (6, 6)
        Figure size in inches (square for equal aspect).

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> pdf, png = save_predictions_plot(
    ...     y_true=actual_scores,
    ...     y_pred_mean=predicted_mean,
    ...     y_pred_lower=ci_lower,
    ...     y_pred_upper=ci_upper,
    ...     output_dir=Path("figs"),
    ...     filename_base="predictions_test",
    ... )
    """
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_lower = np.asarray(y_pred_lower)
    y_pred_upper = np.asarray(y_pred_upper)

    with set_publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Plot error bars for uncertainty
        ax.errorbar(
            y_pred_mean,
            y_true,
            xerr=[y_pred_mean - y_pred_lower, y_pred_upper - y_pred_mean],
            fmt="o",
            alpha=0.5,
            markersize=4,
            color=COLORBLIND_COLORS[0],
            ecolor=COLORBLIND_COLORS[5],  # Light blue for error bars
            elinewidth=0.5,
            capsize=0,
            label=ci_label,
        )

        # Add diagonal reference line (perfect prediction)
        all_values = np.concatenate([y_true, y_pred_mean, y_pred_lower, y_pred_upper])
        lims = [np.min(all_values) - 2, np.max(all_values) + 2]
        ax.plot(
            lims,
            lims,
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Perfect prediction",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Labels
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Actual Score")
        ax.legend(loc="lower right")

        # Equal aspect ratio for square plot
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_reliability_plot(
    reliability_data: ReliabilityData,
    output_dir: Path,
    filename_base: str,
    figsize: tuple[float, float] = (6, 5),
) -> tuple[Path, Path]:
    """Generate and save reliability diagram (calibration plot).

    Creates a reliability diagram showing predicted probability vs observed
    frequency. A well-calibrated model shows points along the diagonal.

    Parameters
    ----------
    reliability_data : ReliabilityData
        Data from compute_reliability_data().
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    figsize : tuple[float, float], default (6, 5)
        Figure size in inches.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> from panelcast.evaluation.calibration import compute_reliability_data
    >>> rel_data = compute_reliability_data(y_true, y_samples)
    >>> pdf, png = save_reliability_plot(
    ...     rel_data, Path("figs"), "reliability_user"
    ... )
    """
    with set_publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Plot reliability points with bin counts as marker size
        # Normalize counts for marker sizing
        counts = reliability_data.counts
        size_scale = 100 * counts / max(counts.max(), 1)  # Avoid division by zero

        ax.scatter(
            reliability_data.predicted_probs,
            reliability_data.observed_freq,
            s=size_scale + 20,  # Minimum size of 20
            alpha=0.7,
            color=COLORBLIND_COLORS[0],
            edgecolors=COLORBLIND_COLORS[0],
            linewidth=1,
            label="Observed",
        )

        # Connect points with line for visual clarity
        ax.plot(
            reliability_data.predicted_probs,
            reliability_data.observed_freq,
            "-",
            alpha=0.5,
            color=COLORBLIND_COLORS[0],
            linewidth=1,
        )

        # Add diagonal reference line (perfect calibration)
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Perfect calibration",
        )

        # Add count annotations
        for i, (x, y, n) in enumerate(
            zip(
                reliability_data.predicted_probs,
                reliability_data.observed_freq,
                reliability_data.counts,
                strict=True,
            )
        ):
            if n > 0:
                ax.annotate(
                    f"n={n}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    alpha=0.7,
                )

        # Labels
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_rank_scatter_plot(
    slate: pd.DataFrame,
    output_dir: Path,
    filename_base: str = "rank_scatter",
    figsize: tuple[float, float] = (5.5, 5),
) -> tuple[Path, Path]:
    """Predicted vs realized rank for the held-out slate (#182).

    Input is the ranked_slate frame from the evaluate stage. Points on the
    diagonal are perfectly ordered; color encodes P(top-10) when present.
    """
    required = {"predicted_rank", "realized_rank"}
    if not required <= set(slate.columns):
        raise ValueError(f"slate frame lacks {sorted(required - set(slate.columns))}")

    with set_publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        color_col = next((c for c in slate.columns if c.startswith("p_top")), None)
        if color_col is not None:
            sc = ax.scatter(
                slate["predicted_rank"],
                slate["realized_rank"],
                c=slate[color_col],
                cmap="viridis",
                s=14,
                alpha=0.8,
            )
            fig.colorbar(sc, ax=ax, label=f"P({color_col.removeprefix('p_')})")
        else:
            ax.scatter(
                slate["predicted_rank"],
                slate["realized_rank"],
                s=14,
                alpha=0.8,
                color=COLORBLIND_COLORS[0],
            )
        lim = max(len(slate), 1)
        ax.plot([1, lim], [1, lim], "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Predicted rank")
        ax.set_ylabel("Realized rank")
        ax.set_title("Held-out slate: predicted vs realized rank", fontsize=10)
        fig.tight_layout()
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)
        plt.close(fig)

    return pdf_path, png_path


def save_slice_coverage_plot(
    by_slice: dict,
    output_dir: Path,
    filename_base: str = "slice_coverage",
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Small-multiples coverage audit: one panel per slice dimension.

    Each panel plots empirical coverage with Wilson CI whiskers per slice
    against dashed lines at the nominal levels; slices whose nominal level
    falls outside the CI are drawn in the warning color. Input is the
    ``calibration.by_slice`` block from metrics.json (#181).
    """
    slices = by_slice.get("slices") or []
    dimensions: dict[str, list[dict]] = {}
    for s in slices:
        dimensions.setdefault(s["dimension"], []).append(s)
    if not dimensions:
        raise ValueError("by_slice payload has no slices to plot")

    levels = sorted({lv for s in slices for lv in s["levels"]})
    n_panels = len(dimensions)
    if figsize is None:
        figsize = (max(6.0, 3.2 * n_panels), 4.0)

    with set_publication_style():
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (dimension, dim_slices) in zip(axes, sorted(dimensions.items()), strict=True):
            xs = np.arange(len(dim_slices))
            for li, level in enumerate(levels):
                per = [s["levels"].get(level) for s in dim_slices]
                keep = [i for i, p in enumerate(per) if p is not None]
                if not keep:
                    continue
                emp = np.array([per[i]["empirical"] for i in keep])
                lo = np.array([per[i]["wilson_lo"] for i in keep])
                hi = np.array([per[i]["wilson_hi"] for i in keep])
                flagged = np.array([bool(per[i]["flagged"]) for i in keep])
                x = xs[keep] + (li - (len(levels) - 1) / 2) * 0.15
                # errorbar takes one ecolor, so flagged/ok are drawn separately.
                for mask, color in (
                    (~flagged, COLORBLIND_COLORS[0]),
                    (flagged, COLORBLIND_COLORS[1]),
                ):
                    if not mask.any():
                        continue
                    ax.errorbar(
                        x[mask],
                        emp[mask],
                        yerr=[emp[mask] - lo[mask], hi[mask] - emp[mask]],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1,
                        capsize=2,
                    )
                    ax.scatter(x[mask], emp[mask], s=18, color=color, zorder=3)
                ax.axhline(
                    float(level), linestyle="--", linewidth=0.8, color="grey", alpha=0.6
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [f"{s['label']}\n(n={s['n']})" for s in dim_slices],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.set_title(dimension, fontsize=9)
        axes[0].set_ylabel("Empirical coverage")
        fig.suptitle(
            f"Coverage by slice (Wilson 95% CI; ~{by_slice.get('expected_false_flags', '?')} "
            "false flags expected under perfect calibration)",
            fontsize=9,
        )
        fig.tight_layout()
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)
        plt.close(fig)

    return pdf_path, png_path


def save_ppc_density_plot(
    ppc_result,
    output_dir: Path,
    filename_base: str = "ppc_density",
    statistics_to_plot: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save PPC density plots showing T(y_rep) vs T(y_obs).

    Creates a multi-panel subplot with one panel per test statistic. Each panel
    shows the replicated distribution as a histogram with a vertical dashed line
    at the observed statistic value, annotated with Bayesian p-value and MC SE.

    Parameters
    ----------
    ppc_result : PPCResult
        Result from compute_ppc_statistics().
    output_dir : Path
        Directory to save figures.
    filename_base : str, default "ppc_density"
        Base filename without extension.
    statistics_to_plot : list[str] | None, optional
        Names of statistics to include. If None, plots mean, sd, skewness, min, max.
    figsize : tuple[float, float] | None, optional
        Figure size in inches. If None, auto-sizes based on number of panels.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).
    """
    default_stats = ["mean", "sd", "skewness", "min", "max"]
    if statistics_to_plot is None:
        statistics_to_plot = default_stats

    # Filter to requested statistics
    stats_to_plot = [s for s in ppc_result.statistics if s.name in statistics_to_plot]
    if not stats_to_plot:
        stats_to_plot = ppc_result.statistics

    n_panels = len(stats_to_plot)
    if figsize is None:
        n_cols = min(n_panels, 3)
        n_rows = (n_panels + n_cols - 1) // n_cols
        figsize = (3.5 * n_cols, 3.0 * n_rows)

    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    with set_publication_style():
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_panels == 1:
            axes = np.array([axes])
        axes = axes.ravel()

        for i, stat in enumerate(stats_to_plot):
            ax = axes[i]
            try:
                p_value_display = float(stat.bayesian_p_value)
            except (TypeError, ValueError):
                p_value_display = float("nan")
            try:
                mc_se_display = float(stat.mc_se)
            except (TypeError, ValueError):
                mc_se_display = float("nan")
            ax.hist(
                stat.replicated_distribution,
                bins=30,
                density=True,
                alpha=0.7,
                color=COLORBLIND_COLORS[0],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axvline(
                stat.observed,
                color=COLORBLIND_COLORS[6],
                linestyle="--",
                linewidth=1.5,
                label=f"T(y_obs) = {stat.observed:.2f}",
            )
            ax.set_title(f"T = {stat.name}")
            ax.set_xlabel("Statistic value")
            ax.set_ylabel("Density")
            ax.annotate(
                f"p = {p_value_display:.3f}\n(MC SE: {mc_se_display:.3f})",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )
            ax.legend(fontsize=7, loc="upper left")

        # Hide unused axes
        for j in range(n_panels, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_forest_plot(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    filename_base: str,
    param_col: str = "param",
    variant_col: str = "variant",
    estimate_col: str = "mean",
    lower_col: str = "hdi_3%",
    upper_col: str = "hdi_97%",
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save forest plot for coefficient comparison.

    Creates a horizontal error bar plot showing coefficients across
    different model variants or sensitivity analyses.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns for parameter name, variant name,
        estimate, and HDI bounds.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    param_col : str, default "param"
        Column name for parameter identifier.
    variant_col : str, default "variant"
        Column name for model variant identifier.
    estimate_col : str, default "mean"
        Column name for point estimate.
    lower_col : str, default "hdi_3%"
        Column name for lower HDI bound.
    upper_col : str, default "hdi_97%"
        Column name for upper HDI bound.
    figsize : tuple[float, float], optional
        Figure size in inches. If None, auto-sizes based on
        number of parameters: (8, 0.5 * n_rows + 2).

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> from panelcast.pipelines.sensitivity import create_coefficient_comparison_df
    >>> comparison_df = create_coefficient_comparison_df(results)
    >>> pdf, png = save_forest_plot(
    ...     comparison_df, Path("figs"), "coefficient_comparison"
    ... )
    """
    with set_publication_style():
        # Get unique parameters and variants
        params = comparison_df[param_col].unique()
        variants = comparison_df[variant_col].unique()

        n_params = len(params)
        n_variants = len(variants)

        # Auto-size figure
        if figsize is None:
            height = max(0.4 * n_params * n_variants + 2, 4)
            figsize = (8, height)

        fig, ax = plt.subplots(figsize=figsize)

        # Create y-positions for each parameter-variant combination
        y_positions = []
        y_labels = []
        y_pos = 0

        for param in params:
            param_data = comparison_df[comparison_df[param_col] == param]

            for i, (_, row) in enumerate(param_data.iterrows()):
                y_positions.append(y_pos)
                y_labels.append(f"{row[variant_col]}")

                # Plot error bar
                color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]
                ax.errorbar(
                    row[estimate_col],
                    y_pos,
                    xerr=[
                        [row[estimate_col] - row[lower_col]],
                        [row[upper_col] - row[estimate_col]],
                    ],
                    fmt="o",
                    color=color,
                    markersize=6,
                    capsize=3,
                    capthick=1,
                    elinewidth=1.5,
                )
                y_pos += 1

            # Add parameter label
            ax.axhline(y=y_pos - 0.5, color="gray", linewidth=0.5, linestyle="-")
            ax.text(
                ax.get_xlim()[0] if ax.get_xlim()[0] != ax.get_xlim()[1] else -0.5,
                y_pos - (n_variants / 2) - 0.5,
                param,
                fontweight="bold",
                fontsize=9,
                ha="right",
                va="center",
            )
            y_pos += 0.5  # Gap between parameter groups

        # Add vertical reference line at zero
        ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.7)

        # Labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Coefficient Value")
        ax.invert_yaxis()  # Top to bottom ordering

        # Update x-axis limits after all data plotted
        ax.autoscale(axis="x")

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def select_artist_subsets(
    known_df: pd.DataFrame,
    min_albums: int = 5,
    n_per_category: int = 5,
) -> dict[str, list[str]]:
    """Select artist subsets for per-artist plots.

    Picks artists from four categories (best predicted, worst predicted,
    most prolific, highest uncertainty).  Artists appearing in multiple
    categories are deduplicated at plot time — a single plot is generated
    with all applicable category labels.

    Parameters
    ----------
    known_df : pd.DataFrame
        Known-artist predictions with columns ``entity``, ``scenario``,
        ``pred_mean``, ``pred_std``, ``pred_q05``, ``pred_q95``,
        ``last_score``, ``n_training_events``.
    min_albums : int, default 5
        Minimum training albums for inclusion (fan charts with < 5
        points are uninformative).
    n_per_category : int, default 5
        Number of artists per category.

    Returns
    -------
    dict[str, list[str]]
        Mapping from category name to list of artist names.
    """
    # Use the "same" scenario for per-artist metrics
    same = known_df[known_df["scenario"] == "same"].copy()
    same = same[same["n_training_events"] >= min_albums]
    if same.empty:
        return {}

    # 90% CI width as uncertainty metric
    same["ci_width"] = same["pred_q95"] - same["pred_q05"]

    # Per-artist residual (proxy for RMSE with single prediction)
    same["abs_resid"] = (same["pred_mean"] - same["last_score"]).abs()

    subsets: dict[str, list[str]] = {}
    subsets["best_predicted"] = same.nsmallest(n_per_category, "abs_resid")["entity"].tolist()
    subsets["worst_predicted"] = same.nlargest(n_per_category, "abs_resid")["entity"].tolist()
    subsets["most_prolific"] = same.nlargest(n_per_category, "n_training_events")["entity"].tolist()
    subsets["high_uncertainty"] = same.nlargest(n_per_category, "ci_width")["entity"].tolist()
    return subsets


def _fan_chart_quantiles(
    pred_samples: np.ndarray,
    forecast_quantiles: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Percentile bands (q05..q95) for the fan chart.

    When ``forecast_quantiles`` carries the stored (q05, q25, q50, q75, q95)
    for the final (forecast) point, those values are used directly —
    re-percentiling five stacked order statistics shrinks the outer band.
    The unstored 10/90 pair is interpolated between adjacent stored levels.
    """
    q05, q10, q25, q50, q75, q90, q95 = (
        np.percentile(pred_samples, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)
    )
    if forecast_quantiles is not None:
        f05, f25, f50, f75, f95 = (float(v) for v in np.asarray(forecast_quantiles))
        q05[-1], q25[-1], q50[-1], q75[-1], q95[-1] = f05, f25, f50, f75, f95
        q10[-1] = f05 + (f25 - f05) * (0.10 - 0.05) / (0.25 - 0.05)
        q90[-1] = f75 + (f95 - f75) * (0.90 - 0.75) / (0.95 - 0.75)
    return q05, q10, q25, q50, q75, q90, q95


def save_artist_prediction_plot(
    artist: str,
    actual_scores: np.ndarray,
    pred_samples: np.ndarray,
    album_labels: list[str] | None,
    output_dir: Path,
    filename_base: str,
    categories: list[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    forecast_quantiles: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Generate and save a per-artist prediction fan chart.

    Shows actual album scores overlaid on posterior predictive percentile
    bands (10/25/50/75/90).  Y-axis fixed to [0, 100] for cross-artist
    comparability.

    Parameters
    ----------
    artist : str
        Artist name (used in title).
    actual_scores : np.ndarray
        True scores per album, shape (n_albums,).
    pred_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_albums).
    album_labels : list[str] | None
        Album names for x-axis ticks. If None, uses 1-indexed integers.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    categories : list[str] | None
        Category labels to annotate (e.g., ["best_predicted", "most_prolific"]).
    figsize : tuple[float, float]
        Figure size in inches.
    forecast_quantiles : np.ndarray | None
        Precomputed (q05, q25, q50, q75, q95) for the final (forecast) point.
        When given, these render exactly as the band bounds instead of being
        re-percentiled from ``pred_samples``.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).
    """
    n_albums = len(actual_scores)
    x = np.arange(n_albums)

    q05, q10, q25, q50, q75, q90, q95 = _fan_chart_quantiles(pred_samples, forecast_quantiles)

    with set_publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Fan chart bands (light to dark)
        ax.fill_between(
            x,
            q05,
            q95,
            alpha=0.15,
            color=COLORBLIND_COLORS[0],
            label="90% CI",
        )
        ax.fill_between(x, q10, q90, alpha=0.25, color=COLORBLIND_COLORS[0])
        ax.fill_between(
            x,
            q25,
            q75,
            alpha=0.35,
            color=COLORBLIND_COLORS[0],
            label="50% CI",
        )
        ax.plot(x, q50, "-", color=COLORBLIND_COLORS[0], linewidth=1.5, label="Median")

        # Actual scores
        ax.plot(
            x,
            actual_scores,
            "o-",
            color=COLORBLIND_COLORS[6],
            markersize=6,
            linewidth=1,
            label="Actual",
        )

        # Labels
        title = artist
        if categories:
            title += f"  ({', '.join(categories)})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Album")
        ax.set_ylabel("User Score")
        ax.set_ylim(0, 100)

        if album_labels is not None and n_albums <= 20:
            ax.set_xticks(x)
            ax.set_xticklabels(album_labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(i + 1) for i in x])

        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()

        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)
        plt.close(fig)

    return pdf_path, png_path
