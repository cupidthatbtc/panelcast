"""Custom Plotly themes for AOTY prediction visualization.

This module provides colorblind-safe Plotly templates that match the publication
styling from reporting/figures.py. Both light and dark themes are available.

The Wong (2011) colorblind-safe palette is used for all charts:
https://www.nature.com/articles/nmeth.1618

Usage:
    >>> from panelcast.visualization import COLORBLIND_COLORS
    >>> import plotly.io as pio
    >>> # Templates are registered on import
    >>> fig = go.Figure()
    >>> fig.update_layout(template="aoty_light")  # or "aoty_dark"
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "COLORBLIND_COLORS",
    "register_themes",
]

# Colorblind-safe palette from Wong (2011), Nature Methods
# Same palette as reporting/figures.py for consistency
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light blue
    "#D55E00",  # Red-orange
]


def register_themes() -> None:
    """Register custom Plotly templates for AOTY visualization.

    Creates and registers two templates:
    - "aoty_light": Light background with dark text (default)
    - "aoty_dark": Dark background with light text

    Both templates use the Wong (2011) colorblind-safe palette and
    serif fonts for publication consistency.

    This function is called automatically on module import.
    """
    # Light theme template
    pio.templates["aoty_light"] = go.layout.Template(
        layout=go.Layout(
            colorway=COLORBLIND_COLORS,
            font=dict(family="serif", size=12, color="#333333"),
            title=dict(font=dict(size=16, family="serif")),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                gridwidth=0.5,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                gridwidth=0.5,
            ),
            hovermode="x unified",
        )
    )

    # Dark theme template
    pio.templates["aoty_dark"] = go.layout.Template(
        layout=go.Layout(
            colorway=COLORBLIND_COLORS,
            font=dict(family="serif", size=12, color="#E0E0E0"),
            title=dict(font=dict(size=16, family="serif")),
            paper_bgcolor="#1E1E1E",
            plot_bgcolor="#2D2D2D",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                gridwidth=0.5,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                gridwidth=0.5,
            ),
            hovermode="x unified",
        )
    )

    # Set default template
    pio.templates.default = "aoty_light"


# Register themes on module import
register_themes()
