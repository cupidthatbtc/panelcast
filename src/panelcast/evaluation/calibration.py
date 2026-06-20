"""Calibration assessment for Bayesian predictions.

This module provides tools for assessing whether Bayesian credible intervals
are well-calibrated. A model is calibrated if a 95% credible interval contains
the true value approximately 95% of the time.

Key concepts:
- Coverage: The fraction of observations that fall within a credible interval.
  For a well-calibrated model, empirical coverage should match nominal coverage.
- Sharpness: The width of credible intervals. Narrower intervals are better
  (more informative) as long as coverage is maintained.
- Reliability curve: A plot of nominal quantile level (predicted probability)
  versus empirical quantile hit-rate across levels. A calibrated model follows
  the diagonal.

Usage:
    >>> from panelcast.evaluation.calibration import compute_coverage
    >>> result = compute_coverage(y_true, y_samples, prob=0.95)
    >>> print(f"Nominal: {result.nominal}, Empirical: {result.empirical}")
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CoverageResult",
    "IntervalScoreResult",
    "ReliabilityData",
    "WISResult",
    "compute_coverage",
    "compute_interval_score",
    "compute_multi_coverage",
    "compute_reliability_data",
    "compute_weighted_interval_score",
]


def _validate_probability(prob: float) -> None:
    """Validate a nominal probability for interval-based metrics."""
    if not (0.0 < prob < 1.0):
        raise ValueError(f"prob must satisfy 0 < prob < 1, got {prob}")


@dataclass
class CoverageResult:
    """Container for coverage assessment results.

    Attributes
    ----------
    nominal : float
        The nominal probability level (e.g., 0.95 for 95% CI).
    empirical : float
        The empirical coverage (fraction of observations within CI).
    n_obs : int
        Total number of observations.
    n_covered : int
        Number of observations within the credible interval.
    lower_bound : np.ndarray
        Lower bound of the credible interval for each observation.
    upper_bound : np.ndarray
        Upper bound of the credible interval for each observation.
    interval_width : float
        Mean width of credible intervals (sharpness metric).
        Narrower intervals are better if coverage is maintained.
    """

    nominal: float
    empirical: float
    n_obs: int
    n_covered: int
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    interval_width: float


@dataclass
class ReliabilityData:
    """Data for constructing reliability diagrams.

    A reliability diagram plots nominal quantile level (x-axis) against
    empirical hit-rate (y-axis). For a calibrated model, points should
    fall along the diagonal.

    Attributes
    ----------
    bin_edges : np.ndarray
        Visualization helper edges, shape (n_bins + 1,).
    predicted_probs : np.ndarray
        Nominal quantile levels, shape (n_bins,).
    observed_freq : np.ndarray
        Empirical frequency Pr(y_true <= q_p), shape (n_bins,).
    counts : np.ndarray
        Number of observations used per quantile level, shape (n_bins,).
    """

    bin_edges: np.ndarray
    predicted_probs: np.ndarray
    observed_freq: np.ndarray
    counts: np.ndarray


@dataclass
class IntervalScoreResult:
    """Container for interval score results.

    The interval score (IS) is a proper scoring rule for interval forecasts
    that rewards both calibration (coverage) and sharpness (narrow intervals).
    It decomposes additively into a sharpness component (interval width) and a
    calibration penalty (for observations outside the interval).

    Attributes
    ----------
    nominal : float
        The nominal probability level (e.g., 0.95 for 95% CI).
    mean_score : float
        Mean interval score across observations.
    score_values : np.ndarray
        Per-observation interval scores, shape (n_obs,).
    sharpness_component : float
        Mean interval width (sharpness term).
    calibration_penalty : float
        Mean penalty from uncovered observations.
    n_obs : int
        Total number of observations.
    """

    nominal: float
    mean_score: float
    score_values: np.ndarray
    sharpness_component: float
    calibration_penalty: float
    n_obs: int


@dataclass
class WISResult:
    """Container for weighted interval score results.

    The weighted interval score (WIS) is a proper scoring rule that
    approximates the continuous ranked probability score (CRPS) using
    a finite set of central prediction intervals plus the median.
    Follows Bracher et al. (2021).

    Attributes
    ----------
    wis : float
        Weighted interval score.
    median_component : float
        Contribution from median absolute deviation: 0.5 * mean|y - median_pred|.
    per_level : dict[float, float]
        Mapping from probability level to its weighted IS contribution.
    n_obs : int
        Total number of observations.
    """

    wis: float
    median_component: float
    per_level: dict[float, float]
    n_obs: int


def _hdi_per_observation(
    y_samples: np.ndarray,
    prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute highest density interval per observation via sliding window.

    Sorts samples once, then finds the narrowest window containing
    ``prob`` fraction of samples for each observation.

    Returns (lower, upper) arrays of shape (n_obs,).
    """
    n_samples, n_obs = y_samples.shape
    # Sort once (O(n_samples * log(n_samples) * n_obs))
    sorted_samples = np.sort(y_samples, axis=0)

    # Window size: number of samples in the HDI
    window = max(1, int(np.ceil(prob * n_samples)))
    if window >= n_samples:
        return sorted_samples[0], sorted_samples[-1]

    # For each observation, find the start index minimizing interval width
    # widths[i, j] = sorted_samples[i + window, j] - sorted_samples[i, j]
    widths = sorted_samples[window:] - sorted_samples[: n_samples - window]
    best_starts = np.argmin(widths, axis=0)  # shape (n_obs,)

    obs_indices = np.arange(n_obs)
    lower = sorted_samples[best_starts, obs_indices]
    upper = sorted_samples[best_starts + window, obs_indices]
    return lower, upper


def compute_coverage(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    prob: float = 0.95,
    interval_type: str = "equal_tailed",
) -> CoverageResult:
    """Compute empirical coverage of credible intervals.

    For a well-calibrated model, the empirical coverage should approximately
    equal the nominal probability. For example, a 95% credible interval
    should contain about 95% of observed values.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).
        Each column contains samples from the posterior predictive
        distribution for one observation.
    prob : float, default 0.95
        Nominal probability level for the credible interval.
        Common values: 0.50 (50% CI), 0.80 (80% CI), 0.95 (95% CI).
    interval_type : str, default "equal_tailed"
        Type of credible interval.  ``"equal_tailed"`` uses symmetric
        percentiles.  ``"hdi"`` uses the highest density interval per
        observation (tighter for skewed posteriors, but slower).

    Returns
    -------
    CoverageResult
        Container with nominal and empirical coverage, plus interval bounds.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n_obs = 100
    >>> y_true = np.random.normal(50, 10, n_obs)
    >>> # Well-calibrated samples: centered on y_true with known spread
    >>> y_samples = y_true + np.random.normal(0, 10, (1000, n_obs))
    >>> result = compute_coverage(y_true, y_samples, prob=0.95)
    >>> print(f"Coverage: {result.empirical:.2%}")  # Should be ~95%

    Notes
    -----
    The equal-tailed interval is computed via percentiles. For prob=0.95:
    - lower = 2.5th percentile
    - upper = 97.5th percentile

    The HDI is the narrowest interval containing ``prob`` mass.  It is
    computed per observation using a sliding-window scan on sorted samples.
    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_samples.ndim != 2:
        raise ValueError(f"y_samples must be 2D, got shape {y_samples.shape}")
    if y_samples.shape[1] != len(y_true):
        raise ValueError(
            f"y_samples has {y_samples.shape[1]} observations, but y_true has {len(y_true)}"
        )
    _validate_probability(prob)
    if interval_type not in ("equal_tailed", "hdi"):
        raise ValueError(f"interval_type must be 'equal_tailed' or 'hdi', got '{interval_type}'")
    if y_samples.shape[0] < 1:
        raise ValueError("y_samples must include at least one posterior sample.")

    # Compute credible interval bounds
    if interval_type == "hdi":
        lower, upper = _hdi_per_observation(y_samples, prob)
    else:
        alpha = 1 - prob
        lower = np.percentile(y_samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)

    # Check which observations fall within the interval
    covered = (y_true >= lower) & (y_true <= upper)
    n_covered = int(covered.sum())
    n_obs = len(y_true)

    # Compute sharpness (mean interval width)
    interval_width = float(np.mean(upper - lower))

    return CoverageResult(
        nominal=prob,
        empirical=n_covered / n_obs,
        n_obs=n_obs,
        n_covered=n_covered,
        lower_bound=lower,
        upper_bound=upper,
        interval_width=interval_width,
    )


def compute_multi_coverage(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    probs: tuple[float, ...] = (0.50, 0.80, 0.95),
) -> dict[float, CoverageResult]:
    """Compute coverage for multiple probability levels.

    This is useful for assessing calibration across different credible
    interval widths. A well-calibrated model should have approximately
    correct coverage at all levels.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).
    probs : tuple of float, default (0.50, 0.80, 0.95)
        Probability levels to compute coverage for.

    Returns
    -------
    dict[float, CoverageResult]
        Mapping from probability level to coverage result.

    Examples
    --------
    >>> results = compute_multi_coverage(y_true, y_samples)
    >>> for prob, result in results.items():
    ...     print(f"{prob*100:.0f}% CI: {result.empirical:.2%} coverage")
    """
    return {prob: compute_coverage(y_true, y_samples, prob) for prob in probs}


def compute_reliability_data(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityData:
    """Compute data for reliability diagrams.

    Uses quantile calibration for continuous outcomes:
    - Choose nominal probability levels p in (0, 1)
    - Compute predictive quantile q_p for each observation
    - Compute empirical hit-rate Pr(y_true <= q_p)

    For a calibrated model, empirical hit-rate should match p.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).
    n_bins : int, default 10
        Number of nominal quantile levels used for the curve.

    Returns
    -------
    ReliabilityData
        Data for constructing the reliability diagram.

    Examples
    --------
    >>> data = compute_reliability_data(y_true, y_samples, n_bins=10)
    >>> # For plotting:
    >>> # plt.plot(data.predicted_probs, data.observed_freq, 'o-')
    >>> # plt.plot([0, 1], [0, 1], 'k--')  # Diagonal for perfect calibration

    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_samples.ndim != 2:
        raise ValueError(f"y_samples must be 2D, got shape {y_samples.shape}")
    if y_samples.shape[1] != len(y_true):
        raise ValueError(
            f"y_samples has {y_samples.shape[1]} observations, but y_true has {len(y_true)}"
        )

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # Nominal levels avoid exact 0/1 tails where finite-sample quantiles are unstable.
    if n_bins == 1:
        predicted_probs = np.array([0.5], dtype=float)
    else:
        predicted_probs = np.linspace(0.05, 0.95, n_bins, dtype=float)

    # quantiles shape: (n_bins, n_obs)
    quantiles = np.quantile(y_samples, predicted_probs, axis=0)
    observed_freq = np.mean(y_true[None, :] <= quantiles, axis=1)
    counts = np.full(predicted_probs.shape, len(y_true), dtype=int)
    bin_edges = np.linspace(0.0, 1.0, len(predicted_probs) + 1)

    return ReliabilityData(
        bin_edges=bin_edges,
        predicted_probs=predicted_probs,
        observed_freq=observed_freq,
        counts=counts,
    )


def compute_interval_score(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    prob: float = 0.95,
) -> IntervalScoreResult:
    """Compute the interval score for probabilistic interval forecasts.

    The interval score (Gneiting & Raftery, 2007) is a proper scoring rule
    that penalises both wide intervals (poor sharpness) and uncovered
    observations (poor calibration). It decomposes additively:

        IS_alpha = (u - l) + (2/alpha) * max(l - y, 0) + (2/alpha) * max(y - u, 0)

    where alpha = 1 - prob.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).
        Each column contains samples from the posterior predictive
        distribution for one observation.
    prob : float, default 0.95
        Nominal probability level for the credible interval.

    Returns
    -------
    IntervalScoreResult
        Container with interval score and its decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n_obs = 100
    >>> y_true = np.random.normal(50, 10, n_obs)
    >>> y_samples = y_true + np.random.normal(0, 10, (1000, n_obs))
    >>> result = compute_interval_score(y_true, y_samples, prob=0.95)
    >>> print(f"IS: {result.mean_score:.2f}")
    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_samples.ndim != 2:
        raise ValueError(f"y_samples must be 2D, got shape {y_samples.shape}")
    if y_samples.shape[1] != len(y_true):
        raise ValueError(
            f"y_samples has {y_samples.shape[1]} observations, but y_true has {len(y_true)}"
        )
    _validate_probability(prob)
    if y_samples.shape[0] < 1:
        raise ValueError("y_samples must include at least one posterior sample.")

    alpha = 1 - prob
    lower = np.percentile(y_samples, 100 * alpha / 2, axis=0)
    upper = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)

    # Decomposition
    width = upper - lower
    penalty_lower = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_upper = (2 / alpha) * np.maximum(y_true - upper, 0)

    score_values = width + penalty_lower + penalty_upper
    sharpness_component = float(np.mean(width))
    calibration_penalty = float(np.mean(penalty_lower + penalty_upper))

    return IntervalScoreResult(
        nominal=prob,
        mean_score=float(np.mean(score_values)),
        score_values=score_values,
        sharpness_component=sharpness_component,
        calibration_penalty=calibration_penalty,
        n_obs=len(y_true),
    )


def compute_weighted_interval_score(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    probs: tuple[float, ...] = (0.50, 0.80, 0.95),
) -> WISResult:
    """Compute the weighted interval score (WIS).

    The WIS (Bracher et al., 2021) approximates the continuous ranked
    probability score (CRPS) using a finite set of central prediction
    intervals plus the predictive median:

        WIS = (1 / (K + 0.5)) * (0.5 * mean|y - median| +
              sum_k (alpha_k / 2) * mean_IS_alpha_k)

    where K = len(probs), alpha_k = 1 - prob_k.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).
        Each column contains samples from the posterior predictive
        distribution for one observation.
    probs : tuple of float, default (0.50, 0.80, 0.95)
        Probability levels for the central prediction intervals.

    Returns
    -------
    WISResult
        Container with WIS, median component, and per-level contributions.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n_obs = 100
    >>> y_true = np.random.normal(50, 10, n_obs)
    >>> y_samples = y_true + np.random.normal(0, 10, (1000, n_obs))
    >>> result = compute_weighted_interval_score(y_true, y_samples)
    >>> print(f"WIS: {result.wis:.2f}")
    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_samples.ndim != 2:
        raise ValueError(f"y_samples must be 2D, got shape {y_samples.shape}")
    if y_samples.shape[1] != len(y_true):
        raise ValueError(
            f"y_samples has {y_samples.shape[1]} observations, but y_true has {len(y_true)}"
        )
    if y_samples.shape[0] < 1:
        raise ValueError("y_samples must include at least one posterior sample.")
    if len(probs) == 0:
        raise ValueError("probs must include at least one probability level.")
    for prob in probs:
        _validate_probability(prob)

    n_obs = len(y_true)
    K = len(probs)

    # Median component: 0.5 * mean|y - median_pred|
    median_pred = np.median(y_samples, axis=0)
    median_component = 0.5 * float(np.mean(np.abs(y_true - median_pred)))

    # Per-level IS contributions weighted by alpha_k / 2
    per_level: dict[float, float] = {}
    weighted_sum = 0.0
    for prob in probs:
        is_result = compute_interval_score(y_true, y_samples, prob=prob)
        alpha_k = 1 - prob
        weighted_contribution = (alpha_k / 2) * is_result.mean_score
        per_level[prob] = weighted_contribution
        weighted_sum += weighted_contribution

    wis = (1 / (K + 0.5)) * (median_component + weighted_sum)

    return WISResult(
        wis=wis,
        median_component=median_component,
        per_level=per_level,
        n_obs=n_obs,
    )


def compute_pit_values(
    y_true: np.ndarray,
    y_samples: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Probability integral transform values and histogram.

    PIT_i = P(Y_rep <= y_obs_i) estimated from the posterior predictive
    sample. For a perfectly calibrated forecast the PIT values are
    uniform on [0, 1]; a U-shape means intervals are too narrow, a hump
    means too wide, and a J/L-shape reveals skew mismatch (the symmetric
    likelihood against the ceiling shows up here before it moves
    coverage-at-two-levels).

    Uses the randomized-rank convention for discrete predictive samples:
    PIT = (#draws < y + 0.5 * #draws == y) / n_draws.

    Args:
        y_true: Observed values, shape (n_obs,).
        y_samples: Posterior predictive draws, shape (n_draws, n_obs).
        n_bins: Histogram bin count over [0, 1].

    Returns:
        Dict with pit mean/std, histogram counts and bin edges, and the
        max absolute deviation of bin frequency from uniform.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_samples = np.asarray(y_samples, dtype=float)
    if y_samples.ndim != 2 or y_samples.shape[1] != y_true.shape[0]:
        raise ValueError(
            f"y_samples shape {y_samples.shape} incompatible with y_true {y_true.shape}"
        )

    n_draws = y_samples.shape[0]
    below = (y_samples < y_true[None, :]).sum(axis=0)
    equal = (y_samples == y_true[None, :]).sum(axis=0)
    pit = (below + 0.5 * equal) / n_draws

    counts, bin_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    freq = counts / counts.sum() if counts.sum() else counts.astype(float)
    uniform = 1.0 / n_bins

    return {
        "mean": float(np.mean(pit)),
        "std": float(np.std(pit)),
        "n_obs": int(len(pit)),
        "n_bins": int(n_bins),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "max_abs_dev_from_uniform": float(np.max(np.abs(freq - uniform))) if len(pit) else None,
    }
