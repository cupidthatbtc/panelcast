"""Prior predictive simulation and value-aware justification text.

This module provides tools for:
1. Running prior predictive checks on training data structure
2. Generating domain-driven prior justification text from PriorConfig values

Prior predictive checks verify that the prior-implied predictions are
scientifically reasonable (e.g., most predicted scores fall within [0, 100]).

Usage:
    >>> from panelcast.evaluation.prior_predictive import (
    ...     run_prior_predictive,
    ...     generate_prior_justification_text,
    ... )
"""

import logging
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PriorPredictiveResult",
    "run_prior_predictive",
    "generate_prior_justification_text",
]

logger = logging.getLogger(__name__)


@dataclass
class PriorPredictiveResult:
    """Result of prior predictive simulation.

    Attributes
    ----------
    y_samples : np.ndarray
        Prior predictive samples ON THE SCORE SCALE, shape (n_samples, n_obs).
        When the model trains under a non-identity target transform the raw
        draws are back-transformed before any statistic is computed.
    summary : dict[str, float]
        Summary statistics of prior predictive (mean, sd, skewness, q2.5,
        q97.5, min, max) on the score scale.
    reasonable : bool
        Whether fraction of samples in bounds exceeds threshold.
    bounds : tuple[float, float]
        Score bounds used for reasonableness check.
    fraction_in_bounds : float
        Fraction of prior predictive samples within bounds.
    n_samples : int
        Number of prior samples drawn.
    n_obs_original : int
        Original number of observations before subsampling.
    max_obs : int
        Maximum observations used (subsampling limit).
    sampled_indices : np.ndarray | None
        Indices used for subsampling, None if no subsampling applied.
    seed : int
        Random seed used for reproducibility.
    checks : dict[str, dict]
        Score-scale plausibility checks. Each entry maps a statistic name to
        {"value", "low", "high", "passed"}; ranges scale with the bounds span
        so non-(0,100) domains get proportionate gates.
    checks_passed : bool
        True when every plausibility check passed (informational by default;
        callers may enforce it under strict mode).
    informational_flags : list[str]
        Human-readable descriptions of failed checks.
    """

    y_samples: np.ndarray
    summary: dict[str, float]
    reasonable: bool
    bounds: tuple[float, float]
    fraction_in_bounds: float
    n_samples: int = 0
    n_obs_original: int = 0
    max_obs: int = 2000
    sampled_indices: np.ndarray | None = None
    seed: int = 42
    checks: dict | None = None
    checks_passed: bool = True
    informational_flags: list | None = None


def run_prior_predictive(
    model,
    model_args: dict,
    priors=None,
    n_samples: int = 500,
    max_obs: int = 2000,
    seed: int = 42,
    score_bounds: tuple[float, float] = (0, 100),
    fraction_threshold: float = 0.90,
    transform=None,
) -> PriorPredictiveResult:
    """Run prior predictive simulation on training data structure.

    Uses numpyro.infer.Predictive with no posterior samples to generate
    predictions from the prior alone. Checks whether prior-implied predictions
    are scientifically reasonable.

    Parameters
    ----------
    model : callable
        NumPyro model function.
    model_args : dict
        Training model arguments (covariate structure). y is set to None.
    priors : PriorConfig | None, optional
        Prior configuration. If None, uses model_args["priors"].
    n_samples : int, default 500
        Number of prior predictive samples.
    max_obs : int, default 2000
        Maximum observations to use. If data has more, subsamples.
    seed : int, default 42
        Random seed for reproducibility.
    score_bounds : tuple[float, float], default (0, 100)
        Expected range for reasonable predictions ON THE SCORE SCALE.
    fraction_threshold : float, default 0.90
        Minimum fraction of samples within bounds for "reasonable".
    transform : TargetTransform | None, optional
        Target transform the model trains under. When given and non-identity,
        the raw draws (model scale, e.g. logit) are back-transformed to the
        score scale before any statistic or bound check is computed —
        otherwise the in-bounds fraction would be evaluated on the wrong
        scale entirely.

    Returns
    -------
    PriorPredictiveResult
        Prior predictive results with score-scale summary, the legacy
        in-bounds reasonableness check, and proportionate plausibility
        checks on mean / sd / skewness (informational; gate under strict
        at the call site).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if max_obs < 1:
        raise ValueError(f"max_obs must be >= 1, got {max_obs}")
    if len(score_bounds) != 2 or score_bounds[0] >= score_bounds[1]:
        raise ValueError(
            f"score_bounds must be (lower, upper) with lower < upper, got {score_bounds}"
        )
    if not (0.0 <= fraction_threshold <= 1.0):
        raise ValueError(f"fraction_threshold must be in [0, 1], got {fraction_threshold}")

    from jax import random
    from numpyro.infer import Predictive

    # Prepare model args: set y=None for prior predictive
    args = dict(model_args)
    args["y"] = None
    if priors is not None:
        args["priors"] = priors

    # Determine n_obs and subsample if needed
    n_obs_original = None
    array_keys = ["artist_idx", "album_seq", "prev_score", "X"]
    for key in array_keys:
        if key in args and hasattr(args[key], "__len__"):
            n_obs_original = len(args[key])
            break

    if n_obs_original is None:
        n_obs_original = 0

    sampled_indices = None
    if n_obs_original > max_obs:
        rng = np.random.default_rng(seed)
        sampled_indices = np.sort(rng.choice(n_obs_original, size=max_obs, replace=False))
        # Subsample all array-valued model args
        for key in array_keys:
            if key in args and hasattr(args[key], "shape"):
                val = args[key]
                if val.ndim == 1 and len(val) == n_obs_original:
                    args[key] = val[sampled_indices]
                elif val.ndim == 2 and val.shape[0] == n_obs_original:
                    args[key] = val[sampled_indices]
        # Also subsample n_reviews if present
        if "n_reviews" in args and hasattr(args["n_reviews"], "shape"):
            if len(args["n_reviews"]) == n_obs_original:
                args["n_reviews"] = args["n_reviews"][sampled_indices]
        logger.info(
            "Prior predictive subsampled observations: %d -> %d",
            n_obs_original,
            max_obs,
        )

    # Run prior predictive
    rng_key = random.key(seed)
    predictive = Predictive(model, num_samples=n_samples)
    preds = predictive(rng_key, **args)

    # Extract y samples
    y_key = next((k for k in preds if k.endswith("_y")), None)
    if y_key is None:
        raise ValueError("Unable to locate observed site in prior predictive output.")
    y_samples = np.asarray(preds[y_key])

    # Back-transform to the score scale: under offset_logit the raw draws
    # live on the logit scale, where the in-bounds fraction and the
    # plausibility ranges below would be meaningless.
    if transform is not None and getattr(transform, "name", "identity") != "identity":
        y_samples = np.asarray(transform.inverse(y_samples))

    # Compute summary statistics (score scale)
    flat = y_samples.ravel()
    mean = float(np.mean(flat))
    sd = float(np.std(flat))
    skewness = float(np.mean(((flat - mean) / sd) ** 3)) if sd > 0 else 0.0
    summary = {
        "mean": mean,
        "sd": sd,
        "skewness": skewness,
        "q2.5": float(np.percentile(flat, 2.5)),
        "q97.5": float(np.percentile(flat, 97.5)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }

    # Check reasonableness (legacy gate)
    lower, upper = score_bounds
    in_bounds = np.mean((flat >= lower) & (flat <= upper))
    fraction_in_bounds = float(in_bounds)
    reasonable = fraction_in_bounds >= fraction_threshold

    # Plausibility checks on the score scale. Ranges are proportionate to
    # the bounds span so non-(0,100) domains gate sensibly: for (0,100)
    # these reduce to mean in [60, 90], sd in [5, 20], skewness in [-3, 0].
    span = upper - lower
    checks = {
        "mean": {
            "value": mean,
            "low": lower + 0.60 * span,
            "high": lower + 0.90 * span,
        },
        "sd": {
            "value": sd,
            "low": 0.05 * span,
            "high": 0.20 * span,
        },
        "skewness": {
            "value": skewness,
            "low": -3.0,
            "high": 0.0,
        },
    }
    informational_flags = []
    for name, check in checks.items():
        check["passed"] = bool(check["low"] <= check["value"] <= check["high"])
        if not check["passed"]:
            informational_flags.append(
                f"prior predictive {name}={check['value']:.2f} outside "
                f"[{check['low']:.2f}, {check['high']:.2f}]"
            )
    checks_passed = not informational_flags
    for flag in informational_flags:
        logger.warning("Prior predictive plausibility flag: %s", flag)

    return PriorPredictiveResult(
        y_samples=y_samples,
        summary=summary,
        reasonable=reasonable,
        bounds=score_bounds,
        fraction_in_bounds=fraction_in_bounds,
        n_samples=n_samples,
        n_obs_original=n_obs_original,
        max_obs=max_obs,
        sampled_indices=sampled_indices,
        seed=seed,
        checks=checks,
        checks_passed=checks_passed,
        informational_flags=informational_flags,
    )


def generate_prior_justification_text(
    priors,
    prior_predictive_result: PriorPredictiveResult | None = None,
    sensitivity_summary=None,
) -> str:
    """Generate value-aware domain-driven prior justification text.

    Reads actual field values from priors and incorporates them into prose.
    When PriorConfig changes, generated text automatically updates.

    Parameters
    ----------
    priors : PriorConfig
        Prior configuration with actual parameter values.
    prior_predictive_result : PriorPredictiveResult | None, optional
        If provided, appends prior predictive check paragraph.
    sensitivity_summary : pd.DataFrame | None, optional
        If provided, appends sensitivity analysis paragraph.
        Expected to have columns: parameter, elpd_delta, eligible_for_ranking.

    Returns
    -------
    str
        Multi-paragraph prior justification text.
    """
    sections = []

    # Domain justification for each parameter
    sections.append(
        "Prior distributions are weakly informative, chosen to regularize "
        "inference while allowing the data to dominate:\n"
    )

    justifications = [
        f"- **mu_artist** ~ Normal({priors.mu_artist_loc}, {priors.mu_artist_scale}): "
        f"Centered at {priors.mu_artist_loc} because artist effects represent deviations "
        f"from feature-based predictions. Scale of {priors.mu_artist_scale} permits the "
        f"population center to shift by ~{priors.mu_artist_scale} SD on the standardized "
        f"score scale.",
        f"- **sigma_artist** ~ HalfNormal({priors.sigma_artist_scale}): "
        f"Scale of {priors.sigma_artist_scale} encourages moderate partial pooling. "
        f"Implies most artist effects within +/-{2 * priors.sigma_artist_scale:.1f}, "
        f"consistent with observed between-artist spread.",
        f"- **sigma_rw** ~ HalfNormal({priors.sigma_rw_scale}): "
        f"Scale of {priors.sigma_rw_scale} produces smooth career trajectories where "
        f"album-to-album quality changes are small relative to overall artist variation.",
        f"- **rho** ~ TruncatedNormal({priors.rho_loc}, {priors.rho_scale}, -0.99, 0.99): "
        f"Centered at {priors.rho_loc} with scale {priors.rho_scale}, allowing moderate "
        f"autoregressive momentum without strong prior commitment to direction.",
        f"- **beta** ~ Normal({priors.beta_loc}, {priors.beta_scale}): "
        f"Scale of {priors.beta_scale} is weakly informative for standardized features, "
        f"allowing data to determine effect sizes.",
        f"- **sigma_obs** ~ HalfNormal({priors.sigma_obs_scale}): "
        f"Scale of {priors.sigma_obs_scale} allows data to determine observation-level noise.",
    ]
    sections.append("\n".join(justifications))

    # Prior predictive check paragraph
    if prior_predictive_result is not None:
        ppr = prior_predictive_result
        pp_text = (
            f"\n\n**Prior Predictive Check**: Prior predictive simulation "
            f"(n_samples={ppr.n_samples}) shows {ppr.fraction_in_bounds:.1%} of "
            f"prior-implied predictions fall within [{ppr.bounds[0]}, {ppr.bounds[1]}]. "
            f"Summary: mean={ppr.summary['mean']:.1f}, sd={ppr.summary['sd']:.1f}, "
            f"range=[{ppr.summary['min']:.1f}, {ppr.summary['max']:.1f}]."
        )
        sections.append(pp_text)

    # Sensitivity analysis paragraph
    if sensitivity_summary is not None:
        import pandas as pd

        if isinstance(sensitivity_summary, pd.DataFrame) and not sensitivity_summary.empty:
            # Filter to convergence-passed, non-baseline variants
            eligible = sensitivity_summary[
                (sensitivity_summary["eligible_for_ranking"] == True)  # noqa: E712
                & (
                    sensitivity_summary.get(
                        "parameter",
                        sensitivity_summary.get("variant", ""),
                    )
                    != "baseline"
                )
            ]
            # Filter out baseline rows
            if "variant" in eligible.columns:
                eligible = eligible[eligible["variant"] != "default"]

            if not eligible.empty and "elpd_delta" in eligible.columns:
                eligible_with_delta = eligible.dropna(subset=["elpd_delta"])
                if not eligible_with_delta.empty:
                    max_idx = eligible_with_delta["elpd_delta"].abs().idxmax()
                    most_sensitive_param = eligible_with_delta.loc[max_idx, "parameter"]
                    max_delta = eligible_with_delta.loc[max_idx, "elpd_delta"]

                    sens_text = (
                        f"\n\n**Sensitivity to Prior Choice**: One-at-a-time perturbation "
                        f"analysis shows the most sensitive parameter is {most_sensitive_param} "
                        f"(max |ΔELPD| = {abs(max_delta):.1f}), indicating "
                        f"{'minimal' if abs(max_delta) < 5 else 'moderate'} sensitivity "
                        f"to prior specification among convergence-passed variants."
                    )
                    sections.append(sens_text)

    return "\n".join(sections)
