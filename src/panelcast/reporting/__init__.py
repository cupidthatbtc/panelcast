"""Reporting module for publication-quality artifacts.

This module provides tools for generating publication-ready tables, figures,
and model documentation from Bayesian model results. All outputs are designed
for direct inclusion in academic manuscripts.

Key capabilities:
- Tables: Coefficient summaries, diagnostics, model comparisons (CSV + LaTeX)
- Figures: Trace plots, posterior distributions, prediction plots (PDF + PNG)
- Model cards: Structured documentation for reproducibility

Table functions:
- create_coefficient_table: Parameter estimates with credible intervals
- create_diagnostics_table: R-hat, ESS, and convergence status
- create_comparison_table: Model comparison by ELPD/LOO-CV
- export_table: Dual-format export (CSV + LaTeX)

Usage:
    >>> from panelcast.reporting import (
    ...     create_coefficient_table,
    ...     create_diagnostics_table,
    ...     export_table,
    ... )
    >>> coef_df = create_coefficient_table(idata, var_names=["beta", "sigma"])
    >>> export_table(coef_df, "reports/coefficients", caption="Model coefficients")
"""

from .curves import (
    CurvePeakSummary,
    PosteriorCurve,
    basis_matrix,
    extract_curve_draws,
    extract_posterior_curve,
    summarize_curve_peak,
)
from .tables import (
    create_coefficient_table,
    create_comparison_table,
    create_diagnostics_table,
    export_table,
)

__all__ = [
    "PosteriorCurve",
    "CurvePeakSummary",
    "basis_matrix",
    "extract_curve_draws",
    "extract_posterior_curve",
    "summarize_curve_peak",
    # Tables
    "create_coefficient_table",
    "create_diagnostics_table",
    "create_comparison_table",
    "export_table",
]
