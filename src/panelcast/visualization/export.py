"""Static export pipeline for Plotly figures.

This module provides functions to export Plotly figures to static formats
(SVG, PNG, PDF) using Kaleido, as well as multi-chart HTML dashboards.

Kaleido v1+ requires Chrome to be available. The ensure_kaleido_chrome()
function checks availability and attempts to download Chrome if needed.

Usage:
    >>> from panelcast.visualization.export import export_figure
    >>> from panelcast.visualization.charts import create_trace_plot
    >>> import numpy as np
    >>> fig = create_trace_plot(np.random.randn(4, 100), "mu")
    >>> paths = export_figure(fig, Path("output/mu"), formats=("svg", "png"))
"""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go

__all__ = [
    "ensure_kaleido_chrome",
    "export_all_figures",
    "export_dashboard_html",
    "export_figure",
]

logger = logging.getLogger(__name__)


def ensure_kaleido_chrome() -> bool:
    """Check if Chrome is available for Kaleido and attempt to download if not.

    Kaleido v1+ requires Chrome to render static images. This function
    checks availability and downloads Chrome on demand if needed.

    Returns
    -------
    bool
        True if Chrome is available (or was successfully downloaded),
        False if Chrome cannot be obtained.

    Examples
    --------
    >>> if ensure_kaleido_chrome():
    ...     print("Ready for static export")
    ... else:
    ...     print("Static export unavailable")
    """
    try:
        import kaleido
    except ImportError:
        logger.warning("Kaleido not installed. Static export unavailable.")
        return False

    # Check if Chrome is already available
    try:
        # Kaleido v1+ provides get_chrome_sync for downloading Chrome
        if hasattr(kaleido, "get_chrome_sync"):
            # get_chrome_sync returns path to Chrome if available, downloads if not
            chrome_path = kaleido.get_chrome_sync()
            if chrome_path:
                logger.debug("Chrome available at: %s", chrome_path)
                return True
            else:
                logger.warning("Kaleido get_chrome_sync returned None.")
                return False
        else:
            # Older Kaleido versions have Chrome bundled
            logger.debug("Using bundled Chrome from older Kaleido version.")
            return True
    except Exception as e:
        logger.warning("Failed to ensure Chrome for Kaleido: %s", e)
        return False


def export_figure(
    fig: go.Figure,
    output_path: Path,
    formats: tuple[str, ...] = ("svg", "png"),
    width: int = 800,
    height: int = 600,
    scale: float = 2.0,
) -> list[Path]:
    """Export Plotly figure to static image formats.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to export.
    output_path : Path
        Base path without extension. Files will be created with appropriate
        extensions (e.g., output_path.svg, output_path.png).
    formats : tuple[str, ...], default ("svg", "png")
        Output formats. Supported: "svg", "png", "jpeg", "webp", "pdf".
    width : int, default 800
        Figure width in pixels.
    height : int, default 600
        Figure height in pixels.
    scale : float, default 2.0
        Scale factor for raster formats. 2.0 gives ~300dpi at 4" width.
        Vector formats (svg, pdf) ignore this parameter.

    Returns
    -------
    list[Path]
        List of paths to created files.

    Raises
    ------
    ValueError
        If an unsupported format is requested.

    Examples
    --------
    >>> fig = create_trace_plot(samples, "mu")
    >>> paths = export_figure(fig, Path("output/mu"), formats=("svg", "png"))
    >>> print([p.name for p in paths])
    ['mu.svg', 'mu.png']
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vector_formats = {"svg", "pdf"}
    raster_formats = {"png", "jpeg", "jpg", "webp"}
    supported_formats = vector_formats | raster_formats

    created_paths: list[Path] = []

    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in supported_formats:
            raise ValueError(
                f"Unsupported format: {fmt}. Supported: {', '.join(sorted(supported_formats))}"
            )

        file_path = output_path.with_suffix(f".{fmt_lower}")

        # Scale factor only applies to raster formats
        effective_scale = scale if fmt_lower in raster_formats else 1

        try:
            fig.write_image(
                file_path,
                width=width,
                height=height,
                scale=effective_scale,
            )
            logger.info("Exported: %s", file_path)
            created_paths.append(file_path)
        except Exception as e:
            logger.error("Failed to export %s: %s", file_path, e)
            raise

    return created_paths


def export_all_figures(
    output_dir: Path,
    figures: dict[str, go.Figure],
    formats: tuple[str, ...] = ("svg", "png"),
    width: int = 800,
    height: int = 600,
    scale: float = 2.0,
) -> dict[str, list[Path]]:
    """Batch export multiple Plotly figures to static formats.

    Parameters
    ----------
    output_dir : Path
        Directory for output files.
    figures : dict[str, go.Figure]
        Dictionary mapping filename base (without extension) to Figure.
    formats : tuple[str, ...], default ("svg", "png")
        Output formats for all figures.
    width : int, default 800
        Figure width in pixels.
    height : int, default 600
        Figure height in pixels.
    scale : float, default 2.0
        Scale factor for raster formats.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary mapping filename base to list of created paths.

    Examples
    --------
    >>> figures = {"trace_mu": fig1, "posterior_sigma": fig2}
    >>> results = export_all_figures(Path("output"), figures)
    >>> print(list(results.keys()))
    ['trace_mu', 'posterior_sigma']
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Path]] = {}

    for name, fig in figures.items():
        logger.info("Exporting figure: %s", name)
        output_path = output_dir / name
        paths = export_figure(
            fig,
            output_path,
            formats=formats,
            width=width,
            height=height,
            scale=scale,
        )
        results[name] = paths

    logger.info("Exported %d figures to %s", len(figures), output_dir)
    return results


def export_dashboard_html(
    figures: list[go.Figure],
    output_path: Path,
    title: str = "AOTY Model Dashboard",
    include_plotlyjs: bool | str = True,
) -> Path:
    """Export multiple figures to a single self-contained HTML file.

    Creates an HTML file containing all provided figures with proper
    styling and layout. Only the first figure includes Plotly.js to
    minimize file size.

    Parameters
    ----------
    figures : list[go.Figure]
        List of Plotly figures to include.
    output_path : Path
        Path for the output HTML file.
    title : str, default "AOTY Model Dashboard"
        HTML page title.
    include_plotlyjs : bool | str, default True
        How to include Plotly.js:
        - True: Embed full library (~3MB, offline-ready)
        - 'cdn': Use CDN link (smaller file, requires internet)
        - False: Don't include (assumes external Plotly.js)

    Returns
    -------
    Path
        Path to created HTML file.

    Examples
    --------
    >>> figures = [fig1, fig2, fig3]
    >>> path = export_dashboard_html(figures, Path("dashboard.html"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build HTML content
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"    <title>{title}</title>",
        "    <style>",
        "        body { font-family: serif; margin: 0; padding: 20px; background: #fff; }",
        "        .chart-container { margin-bottom: 30px; }",
        "        h1 { text-align: center; color: #333; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>{title}</h1>",
    ]

    # Add each figure
    for i, fig in enumerate(figures):
        # Only first figure includes Plotly.js
        if i == 0:
            plotlyjs = include_plotlyjs
        else:
            plotlyjs = False

        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=plotlyjs,
        )

        html_parts.append(f'    <div class="chart-container" id="chart-{i}">')
        html_parts.append(f"        {fig_html}")
        html_parts.append("    </div>")

    html_parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    html_content = "\n".join(html_parts)

    output_path.write_text(html_content, encoding="utf-8")
    logger.info("Created dashboard HTML: %s", output_path)

    return output_path
