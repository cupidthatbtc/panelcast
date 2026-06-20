"""Interactive visualization module for AOTY prediction results.

This module provides Plotly-based interactive charts and custom themes
for exploring model results. It includes:

- Custom Plotly templates (light and dark themes)
- Chart creation functions for diagnostics and predictions
- Static export pipeline (SVG, PNG via Kaleido)
- Dashboard assembly for multi-chart views
- Colorblind-safe color palette

Usage:
    >>> from panelcast.visualization import COLORBLIND_COLORS, create_trace_plot
    >>> import numpy as np
    >>> samples = np.random.randn(4, 1000)  # 4 chains, 1000 draws
    >>> fig = create_trace_plot(samples, "mu")
    >>> fig.show()
"""

from __future__ import annotations

# Import chart creation functions
from panelcast.visualization.charts import (
    create_forest_plot,
    create_posterior_plot,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)

# Import dashboard/figure-assembly functions
from panelcast.visualization.dashboard import (
    DashboardData,
    create_artist_view,
    create_dashboard_figures,
    get_artist_list,
    load_dashboard_data,
)

# Import export functions
from panelcast.visualization.export import (
    ensure_kaleido_chrome,
    export_all_figures,
    export_dashboard_html,
    export_figure,
)

# Import themes (registered automatically on theme module import)
from panelcast.visualization.theme import COLORBLIND_COLORS, register_themes

__all__ = [
    "COLORBLIND_COLORS",
    "DashboardData",
    "create_artist_view",
    "create_dashboard_figures",
    "create_forest_plot",
    "create_posterior_plot",
    "create_predictions_plot",
    "create_reliability_plot",
    "create_trace_plot",
    "ensure_kaleido_chrome",
    "export_all_figures",
    "export_dashboard_html",
    "export_figure",
    "get_artist_list",
    "load_dashboard_data",
    "register_themes",
]
