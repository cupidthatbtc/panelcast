"""Probabilistic and point metrics for Bayesian model evaluation.

This module provides metrics for evaluating Bayesian predictions:
- CRPS (Continuous Ranked Probability Score): A proper scoring rule for
  probabilistic regression that generalizes MAE to full distributions.
- Point metrics (MAE, RMSE, R2, MedianAE): Traditional regression metrics
  computed on posterior mean predictions.

Key concepts:
- CRPS is preferred over point metrics for Bayesian evaluation because it
  rewards both accuracy (predicting near the true value) and calibration
  (appropriate uncertainty).
- Lower CRPS is better. CRPS has the same units as the target variable.
- Point metrics are still useful for comparison with non-Bayesian baselines.

Usage:
    >>> from panelcast.evaluation.metrics import compute_crps, compute_point_metrics
    >>> crps = compute_crps(y_true, y_samples)
    >>> print(f"Mean CRPS: {crps.mean_crps:.2f}")
    >>> point = compute_point_metrics(y_true, y_pred_mean)
    >>> print(f"MAE: {point.mae:.2f}, R2: {point.r2:.3f}")
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CRPSResult",
    "PointMetrics",
    "compute_crps",
    "compute_point_metrics",
    "posterior_mean",
]


@dataclass
class CRPSResult:
    """Container for CRPS (Continuous Ranked Probability Score) results.

    CRPS is a proper scoring rule for probabilistic predictions. It measures
    how well the predicted distribution matches the true value. Lower is better.

    Attributes
    ----------
    mean_crps : float
        Average CRPS across all observations. Lower values indicate better
        probabilistic predictions.
    crps_values : np.ndarray
        Per-observation CRPS values, shape (n_obs,). Useful for identifying
        observations where the model performs poorly.
    n_obs : int
        Number of observations.
    """

    mean_crps: float
    crps_values: np.ndarray
    n_obs: int

    def to_summary_dict(self) -> dict[str, float | int]:
        """Serialize summary fields for JSON output.

        The per-observation crps_values array is excluded because it is
        too large for JSON metrics output.  Use crps_values directly for
        per-observation diagnostics.
        """
        return {
            "mean_crps": float(self.mean_crps),
            "n_obs": int(self.n_obs),
        }


@dataclass
class PointMetrics:
    """Container for traditional point prediction metrics.

    These metrics evaluate the posterior mean as a point prediction.
    They don't account for prediction uncertainty, so CRPS is preferred
    for Bayesian models. However, point metrics are useful for:
    - Comparison with non-Bayesian baselines
    - Understanding overall prediction accuracy
    - Familiar interpretation

    Attributes
    ----------
    mae : float
        Mean Absolute Error. Average absolute difference between predictions
        and true values. Same units as target variable.
    rmse : float
        Root Mean Squared Error. Square root of average squared differences.
        More sensitive to outliers than MAE. Same units as target variable.
    r2 : float
        Coefficient of determination. Proportion of variance explained by
        the model. 1.0 is perfect, 0.0 is no better than predicting the mean.
        Can be negative for very poor models.
    median_ae : float
        Median Absolute Error. More robust to outliers than MAE.
        Same units as target variable.
    n_observations : int
        Count of observations used to compute the metrics.
    mean_bias : float
        Mean prediction bias (mean of y_pred - y_true). Positive values
        indicate overprediction on average, negative values indicate
        underprediction.
    """

    mae: float
    rmse: float
    r2: float
    median_ae: float
    n_observations: int
    mean_bias: float

    def to_summary_dict(self) -> dict[str, float | int]:
        """Serialize all fields for JSON output.

        This is the single serialization path for PointMetrics — callers
        should use this method instead of hand-building dicts to prevent
        field drift.
        """
        return {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "r2": float(self.r2),
            "median_ae": float(self.median_ae),
            "n_observations": int(self.n_observations),
            "mean_bias": float(self.mean_bias),
        }


def compute_crps(
    y_true: np.ndarray,
    y_samples: np.ndarray,
) -> CRPSResult:
    """Compute CRPS (Continuous Ranked Probability Score) for Bayesian predictions.

    CRPS is a proper scoring rule that generalizes MAE to probabilistic
    predictions. It measures how well the predicted distribution matches
    the observed value. Lower CRPS indicates better probabilistic predictions.

    For a deterministic prediction (point estimate), CRPS equals MAE.
    For probabilistic predictions, CRPS also rewards appropriate uncertainty:
    - Overconfident predictions (too narrow intervals) are penalized
    - Underconfident predictions (too wide intervals) are penalized

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).
        Each column contains samples from the posterior predictive
        distribution for one observation.

    Returns
    -------
    CRPSResult
        Container with mean CRPS and per-observation values.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n_obs = 100
    >>> y_true = np.random.normal(50, 10, n_obs)
    >>> # Well-calibrated samples
    >>> y_samples = y_true + np.random.normal(0, 10, (1000, n_obs))
    >>> result = compute_crps(y_true, y_samples)
    >>> print(f"Mean CRPS: {result.mean_crps:.2f}")

    Notes
    -----
    This is a vectorized FAIR (unbiased) ensemble CRPS estimator
    (Ferro et al. 2008), computed per observation as

        CRPS = mean_j |Y_j - y| - (1 / (2 n (n-1))) * sum_{j,k} |Y_j - Y_k|

    where the spread term is the unbiased Gini mean difference of the n
    ensemble members, evaluated via sorted order statistics. The n(n-1)
    denominator makes the estimator unbiased for the CRPS of the underlying
    predictive distribution regardless of ensemble size; the classical 1/n^2
    form (e.g. properscoring's crps_ensemble) is biased upward by O(1/n).
    The two agree as n grows.

    CRPS is measured in the same units as the target variable, making it
    directly interpretable. A CRPS of 5 points means the model's probabilistic
    predictions are, on average, 5 points away from the true values
    (accounting for both location and spread).
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

    n_obs = len(y_true)

    # Vectorized CRPS via sorted order statistics (Gini mean difference form):
    #   CRPS = E|Y - y| - 0.5 * E|Y - Y'|
    # This avoids the per-observation Python loop and is ~10-50x faster.
    n_samples = y_samples.shape[0]
    if n_samples == 1:
        # Degenerate case: single sample CRPS = MAE
        crps_values = np.abs(y_true - y_samples[0])
    else:
        sorted_ens = np.sort(y_samples, axis=0)  # (n_samples, n_obs)
        # E|Y - y| per observation
        term1 = np.abs(sorted_ens - y_true[None, :]).mean(axis=0)
        # E|Y - Y'| via Gini mean difference on sorted samples (ascending convention)
        weights = (2 * np.arange(1, n_samples + 1) - n_samples - 1) / (n_samples * (n_samples - 1))
        term2 = (weights[:, None] * sorted_ens).sum(axis=0)
        crps_values = term1 - term2

    return CRPSResult(
        mean_crps=float(crps_values.mean()),
        crps_values=crps_values,
        n_obs=n_obs,
    )


def compute_point_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
) -> PointMetrics:
    """Compute traditional point prediction metrics.

    These metrics evaluate a single point prediction (typically the
    posterior mean) against true values. They don't account for
    prediction uncertainty, but are useful for:
    - Comparison with non-Bayesian methods
    - Understanding overall prediction accuracy
    - Familiar interpretation for non-Bayesian readers

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_pred_mean : np.ndarray
        Predicted values (typically posterior mean), shape (n_obs,).

    Returns
    -------
    PointMetrics
        Container with MAE, RMSE, R2, and MedianAE.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> y_true = np.array([50, 60, 70, 80, 90])
    >>> y_pred = np.array([52, 58, 72, 78, 88])
    >>> metrics = compute_point_metrics(y_true, y_pred)
    >>> print(f"MAE: {metrics.mae:.2f}, R2: {metrics.r2:.3f}")

    Notes
    -----
    R2 (coefficient of determination) is computed as:
        R2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares. R2 can be negative if the model is worse than
    predicting the mean.
    """
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if y_pred_mean.ndim != 1:
        raise ValueError(f"y_pred_mean must be 1D, got shape {y_pred_mean.shape}")
    if len(y_true) != len(y_pred_mean):
        raise ValueError(
            f"y_true has {len(y_true)} observations, but y_pred_mean has {len(y_pred_mean)}"
        )

    if np.isnan(y_true).any() or np.isnan(y_pred_mean).any():
        raise ValueError(
            "y_true or y_pred_mean contains NaN values. "
            "Filter invalid observations before computing metrics."
        )

    # Compute errors
    errors = y_true - y_pred_mean
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    # MAE
    mae = float(abs_errors.mean())

    # RMSE
    rmse = float(np.sqrt(squared_errors.mean()))

    # R2
    ss_res = squared_errors.sum()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        # All true values are identical - R2 is undefined
        # Return 1.0 if predictions are perfect, 0.0 otherwise
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = float(1 - ss_res / ss_tot)

    # Median AE
    median_ae = float(np.median(abs_errors))

    # Count of observations
    n_observations = len(y_true)

    # Mean bias (positive = overprediction, negative = underprediction)
    mean_bias = float((y_pred_mean - y_true).mean())

    return PointMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        median_ae=median_ae,
        n_observations=n_observations,
        mean_bias=mean_bias,
    )


def posterior_mean(y_samples: np.ndarray) -> np.ndarray:
    """Compute posterior mean from posterior predictive samples.

    This is a helper function that extracts a point prediction from
    full posterior samples by taking the mean across the sample dimension.

    Parameters
    ----------
    y_samples : np.ndarray
        Posterior predictive samples, shape (n_samples, n_obs).

    Returns
    -------
    np.ndarray
        Mean prediction for each observation, shape (n_obs,).

    Examples
    --------
    >>> import numpy as np
    >>> y_samples = np.random.normal(0, 1, (1000, 50))  # 1000 samples, 50 obs
    >>> y_mean = posterior_mean(y_samples)
    >>> print(f"Shape: {y_mean.shape}")  # (50,)
    """
    y_samples = np.asarray(y_samples)

    if y_samples.ndim != 2:
        raise ValueError(f"y_samples must be 2D, got shape {y_samples.shape}")

    return y_samples.mean(axis=0)
