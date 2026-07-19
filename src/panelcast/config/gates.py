"""Literal type aliases for the pipeline's string-valued model gates.

These name the closed sets of values that the ``PipelineConfig`` gates accept,
so a typo (``"identiy"``) is a type error at the boundary rather than a runtime
``ValueError`` deep in a fit. Runtime validation still lives in
``PipelineConfig.__post_init__`` / the model factories — these are the
type-checker's view of the same contract.
"""

from __future__ import annotations

from typing import Literal

TargetTransform = Literal["identity", "offset_logit"]
LatentProcess = Literal["rw", "ar1"]
# Shared by sigma_obs_prior_type and sigma_artist_prior_type: typing caches
# identical Literals, so a second alias for the same value set would be
# indistinguishable at runtime (and orphaned by the select-space guard).
SigmaObsPriorType = Literal["halfnormal", "lognormal"]
ArtistEffectParam = Literal["noncentered", "zerosum"]
InitStrategy = Literal["uniform", "median", "feasible"]
BetaPriorType = Literal["normal", "horseshoe"]
ArCenter = Literal["global", "none", "artist_running"]
DebutPrevScoreSource = Literal["train_mean", "dataset_stats"]
NExponentPrior = Literal["logit-normal", "beta"]
# Kept in sync with panelcast.models.bayes.likelihoods.REGISTRY by
# tests/unit/models/bayes/test_likelihood_registry.py (a Literal is a static
# type and can't be derived from the runtime registry).
LikelihoodFamily = Literal[
    "studentt", "normal", "skew_studentt", "beta", "skew_normal", "split_normal",
    "mixture", "beta_binomial", "beta_ceiling",
]
ChainMethod = Literal["sequential", "vectorized", "parallel", "auto"]

__all__ = [
    "TargetTransform",
    "LatentProcess",
    "SigmaObsPriorType",
    "ArtistEffectParam",
    "InitStrategy",
    "BetaPriorType",
    "ArCenter",
    "DebutPrevScoreSource",
    "NExponentPrior",
    "LikelihoodFamily",
    "ChainMethod",
]
