"""Posterior predictive checks for Bayesian model validation.

Computes test statistics on replicated vs observed data and produces
Bayesian p-values with Monte Carlo uncertainty context. These are
informational diagnostics — not pipeline gates.

Key concepts:
- T(y_obs): A test statistic computed on observed data
- T(y_rep): The same statistic computed on replicated data from the posterior
- Bayesian p-value: P(T(y_rep) >= T(y_obs)), should be near 0.5 for well-specified models
- MC SE: Monte Carlo standard error on p-value, sqrt(p*(1-p)/n_samples)
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.stats

__all__ = [
    "PPCStatistic",
    "PPCResult",
    "compute_ppc_statistics",
    "DEFAULT_PPC_STATISTICS",
]

logger = logging.getLogger(__name__)


def _safe_skewness(x: np.ndarray) -> float:
    """Compute skewness with safe handling of constant vectors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = scipy.stats.skew(x, nan_policy="omit")
    result = float(result)
    if not np.isfinite(result):
        logger.warning("Skewness returned non-finite value for constant-like vector, using 0.0")
        return 0.0
    return result


DEFAULT_PPC_STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "sd": np.std,
    "skewness": lambda x: _safe_skewness(x),
    "min": np.min,
    "max": np.max,
    "q10": lambda x: float(np.percentile(x, 10)),
    "q50": lambda x: float(np.percentile(x, 50)),
    "q90": lambda x: float(np.percentile(x, 90)),
}


@dataclass
class PPCStatistic:
    """Result for a single PPC test statistic.

    Attributes
    ----------
    name : str
        Name of the test statistic.
    observed : float
        T(y_obs) - statistic value on observed data.
    replicated_distribution : np.ndarray
        T(y_rep) values across posterior replications, shape (n_samples,).
    bayesian_p_value : float
        P(T(y_rep) >= T(y_obs)).
    mc_se : float
        Monte Carlo standard error: sqrt(p*(1-p)/n_samples).
    """

    name: str
    observed: float
    replicated_distribution: np.ndarray
    bayesian_p_value: float
    mc_se: float


@dataclass
class PPCResult:
    """Container for posterior predictive check results.

    Attributes
    ----------
    statistics : list[PPCStatistic]
        List of PPC results for each test statistic.
    n_obs : int
        Number of observations in y_obs.
    n_samples : int
        Number of posterior replications.
    """

    statistics: list[PPCStatistic] = field(default_factory=list)
    n_obs: int = 0
    n_samples: int = 0

    @property
    def summary(self) -> dict[str, dict[str, float]]:
        """JSON-serializable summary of PPC results."""
        return {
            stat.name: {
                "observed": stat.observed,
                "p_value": stat.bayesian_p_value,
                "mc_se": stat.mc_se,
            }
            for stat in self.statistics
        }

    def check_extreme(self, lower: float = 0.01, upper: float = 0.99) -> list[str]:
        """Return names of statistics with extreme Bayesian p-values.

        Parameters
        ----------
        lower : float, default 0.01
            Lower threshold for extreme p-values.
        upper : float, default 0.99
            Upper threshold for extreme p-values.

        Returns
        -------
        list[str]
            Names of statistics with p-value < lower or p-value > upper.
        """
        return [
            stat.name
            for stat in self.statistics
            if stat.bayesian_p_value < lower or stat.bayesian_p_value > upper
        ]


def compute_ppc_statistics(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    statistics: dict[str, Callable[[np.ndarray], float]] | None = None,
) -> PPCResult:
    """Compute posterior predictive check statistics.

    For each test statistic T, computes T(y_obs) and T(y_rep[i]) for each
    posterior replicated dataset, then calculates the Bayesian p-value
    P(T(y_rep) >= T(y_obs)).

    Parameters
    ----------
    y_obs : np.ndarray
        Observed data, shape (n_obs,).
    y_rep : np.ndarray
        Replicated data from posterior predictive, shape (n_samples, n_obs).
    statistics : dict[str, callable] | None, optional
        Dictionary mapping statistic names to callables. Each callable takes
        a 1D array and returns a float. If None, uses DEFAULT_PPC_STATISTICS.

    Returns
    -------
    PPCResult
        Container with per-statistic results and overall metadata.
    """
    y_obs = np.asarray(y_obs)
    y_rep = np.asarray(y_rep)

    if y_obs.ndim != 1:
        raise ValueError(f"y_obs must be 1D, got shape {y_obs.shape}")
    if y_rep.ndim != 2:
        raise ValueError(f"y_rep must be 2D, got shape {y_rep.shape}")
    if y_rep.shape[1] != len(y_obs):
        raise ValueError(f"y_rep has {y_rep.shape[1]} observations, but y_obs has {len(y_obs)}")

    if statistics is None:
        statistics = DEFAULT_PPC_STATISTICS

    n_samples = y_rep.shape[0]
    n_obs = len(y_obs)
    if n_samples < 1:
        raise ValueError("y_rep must include at least one replicated sample.")

    ppc_statistics = []
    for name, stat_fn in statistics.items():
        # Compute T(y_obs)
        t_obs = float(stat_fn(y_obs))

        # Compute T(y_rep[i]) for each replicated dataset
        t_rep = np.array([float(stat_fn(y_rep[i])) for i in range(n_samples)])

        # Bayesian p-value
        p_value = float(np.mean(t_rep >= t_obs))

        # MC standard error
        mc_se = float(np.sqrt(p_value * (1 - p_value) / n_samples))

        ppc_statistics.append(
            PPCStatistic(
                name=name,
                observed=t_obs,
                replicated_distribution=t_rep,
                bayesian_p_value=p_value,
                mc_se=mc_se,
            )
        )

    return PPCResult(
        statistics=ppc_statistics,
        n_obs=n_obs,
        n_samples=n_samples,
    )
