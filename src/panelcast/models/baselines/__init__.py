"""Simple non-Bayesian baselines and a shared scoring harness.

These give the hierarchical Bayesian model something to be incrementally better
than, scored through the same metrics/calibration/CRPS/PPC toolkit.
"""

from panelcast.models.baselines.core import (
    Baseline,
    BaselinePrediction,
    BaselineScore,
    EntityMeanBaseline,
    GBMBaseline,
    GlobalMeanBaseline,
    LastScoreBaseline,
    PanelData,
    RidgeBaseline,
    benchmark_baselines,
    build_default_baselines,
    score_prediction,
)

__all__ = [
    "Baseline",
    "BaselinePrediction",
    "BaselineScore",
    "PanelData",
    "GlobalMeanBaseline",
    "EntityMeanBaseline",
    "LastScoreBaseline",
    "RidgeBaseline",
    "GBMBaseline",
    "build_default_baselines",
    "score_prediction",
    "benchmark_baselines",
]
