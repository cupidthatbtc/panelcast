"""Evaluation module for Bayesian model assessment.

Provides calibration, metrics, and cross-validation utilities for
assessing NumPyro model quality.

Key capabilities:
- Calibration: Check if credible intervals have proper coverage
- Metrics: CRPS (probabilistic) and point prediction metrics
- Cross-validation: LOO-CV with Pareto-k diagnostics
- Prior predictive: Generate predictions from prior distributions

Usage:
    >>> from panelcast.evaluation import compute_crps, compute_coverage, compute_loo
    >>> crps_result = compute_crps(y_true, y_samples)
    >>> coverage = compute_coverage(y_true, y_samples, prob=0.95)
    >>> loo_result = compute_loo(idata_with_loglik)
"""

from .calibration import (
    CoverageResult,
    ReliabilityData,
    compute_coverage,
    compute_multi_coverage,
    compute_reliability_data,
)
from .cv import (
    LOOResult,
    add_log_likelihood_to_idata,
    compare_models,
    compute_log_likelihood,
    compute_loo,
    generate_prior_predictive,
)
from .metrics import (
    CRPSResult,
    PointMetrics,
    compute_crps,
    compute_point_metrics,
    posterior_mean,
)

__all__ = [
    # calibration
    "CoverageResult",
    "compute_coverage",
    "compute_multi_coverage",
    "ReliabilityData",
    "compute_reliability_data",
    # cv
    "compute_log_likelihood",
    "add_log_likelihood_to_idata",
    "LOOResult",
    "compute_loo",
    "compare_models",
    "generate_prior_predictive",
    # metrics
    "CRPSResult",
    "compute_crps",
    "PointMetrics",
    "compute_point_metrics",
    "posterior_mean",
]
