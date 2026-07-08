"""Cross-validation and predictive check functions for NumPyro models.

This module provides LOO-CV (PSIS-LOO) and predictive check functions for
Bayesian model evaluation. Key features:
- Log-likelihood computation for NumPyro models with prefixed site names
- LOO-CV with Pareto-k diagnostics for influential observation detection
- Model comparison via ELPD (expected log pointwise predictive density)
- Prior predictive sampling for model checking

The log-likelihood computation handles NumPyro's prefixed site names
(e.g., "user_y", "critic_y") which require manual computation since
ArviZ's automatic conversion doesn't always handle custom names correctly.
"""

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from jax import random
from numpyro.infer import MCMC, Predictive, log_likelihood

__all__ = [
    "compute_log_likelihood",
    "add_log_likelihood_to_idata",
    "LOOResult",
    "compute_loo",
    "compare_models",
    "generate_prior_predictive",
]


def compute_log_likelihood(
    model,
    mcmc: MCMC,
    model_args: dict,
    obs_name: str = "user_y",
) -> xr.DataArray:
    """Compute pointwise log-likelihood for LOO-CV.

    NumPyro models with prefixed site names (e.g., "user_y") need
    manual log-likelihood computation since az.from_numpyro doesn't
    always handle prefixed names correctly.

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    mcmc : MCMC
        Fitted MCMC object with posterior samples.
    model_args : dict
        Arguments passed to model during fitting. Must include observed
        data 'y' for log-likelihood computation.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model. Use "user_y"
        for user_score_model, "critic_y" for critic_score_model.

    Returns
    -------
    xr.DataArray
        Log-likelihood with dims (chain, draw, obs). Shape is
        (n_chains, n_samples, n_obs) ready for ArviZ consumption.

    Example
    -------
    >>> from panelcast.models.bayes import user_score_model, fit_model
    >>> result = fit_model(user_score_model, model_args)
    >>> log_lik = compute_log_likelihood(
    ...     user_score_model, result.mcmc, model_args, obs_name="user_y"
    ... )
    >>> print(f"Log-lik shape: {log_lik.shape}")  # (4, 1000, n_obs)

    Notes
    -----
    CRITICAL: batch_ndims=1 is used because mcmc.get_samples() returns
    flattened samples with shape (num_chains * num_samples, ...).
    The function reshapes the output to (chain, draw, obs) for ArviZ.
    """
    # Get posterior samples (flattened across chains)
    posterior_samples = mcmc.get_samples()

    # Compute log-likelihood using NumPyro's utility
    # Note: batch_ndims=1 because samples are flattened
    log_lik_dict = log_likelihood(
        model,
        posterior_samples,
        batch_ndims=1,
        **model_args,
    )

    # Extract the relevant site
    if obs_name not in log_lik_dict:
        available_sites = list(log_lik_dict.keys())
        raise KeyError(
            f"Observation site '{obs_name}' not found in log-likelihood dict. "
            f"Available sites: {available_sites}"
        )

    log_lik = log_lik_dict[obs_name]

    # Reshape from (n_samples_total, n_obs) to (chain, draw, n_obs)
    n_chains = mcmc.num_chains
    n_samples = mcmc.num_samples
    n_obs = log_lik.shape[-1]

    # Samples are ordered by chain, then draw
    log_lik_reshaped = np.array(log_lik).reshape(n_chains, n_samples, n_obs)

    # Convert to xarray for ArviZ compatibility
    return xr.DataArray(
        log_lik_reshaped,
        dims=["chain", "draw", "obs"],
        coords={
            "chain": range(n_chains),
            "draw": range(n_samples),
            "obs": range(n_obs),
        },
    )


def add_log_likelihood_to_idata(
    idata: az.InferenceData,
    log_lik_da: xr.DataArray,
    var_name: str = "y",
) -> az.InferenceData:
    """Add log-likelihood group to InferenceData.

    Adds a log_likelihood group to an existing InferenceData object.
    If log_likelihood group already exists, the variable is added to it.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object from model fitting.
    log_lik_da : xr.DataArray
        Log-likelihood array with dims (chain, draw, obs).
    var_name : str, default "y"
        Name for the log-likelihood variable in the group.

    Returns
    -------
    az.InferenceData
        Updated InferenceData with log_likelihood group.

    Example
    -------
    >>> log_lik = compute_log_likelihood(model, mcmc, model_args)
    >>> idata = add_log_likelihood_to_idata(result.idata, log_lik, var_name="y")
    >>> print("log_likelihood" in idata.groups())
    True
    """
    if "log_likelihood" in idata.groups():
        # Add variable to existing group
        idata.log_likelihood[var_name] = log_lik_da
    else:
        # Create new log_likelihood group
        log_lik_ds = xr.Dataset({var_name: log_lik_da})
        idata.add_groups(log_likelihood=log_lik_ds)

    return idata


@dataclass
class LOOResult:
    """Result container for LOO-CV computation.

    Attributes
    ----------
    loo : az.ELPDData
        ArviZ LOO result object with all internal data.
    elpd_loo : float
        Expected log pointwise predictive density (higher is better).
    se_elpd : float
        Standard error of ELPD estimate.
    p_loo : float
        Effective number of parameters (measure of model complexity).
    n_high_pareto_k : int
        Count of observations with Pareto k > 0.7 (problematic).
    high_pareto_k_indices : np.ndarray
        Indices of observations with k > 0.7.
    warning : str | None
        ArviZ warning message if any (e.g., high k values).
    """

    loo: az.ELPDData
    elpd_loo: float
    se_elpd: float
    p_loo: float
    n_high_pareto_k: int
    high_pareto_k_indices: np.ndarray
    warning: str | None


def compute_loo(
    idata: az.InferenceData,
    var_name: str | None = None,
    pointwise: bool = True,
) -> LOOResult:
    """Compute LOO-CV (PSIS-LOO) with Pareto-k diagnostics.

    Computes Leave-One-Out Cross-Validation using Pareto Smoothed
    Importance Sampling (PSIS-LOO). Identifies influential observations
    via Pareto-k diagnostics.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with log_likelihood group. Use add_log_likelihood_to_idata
        to add it if not present.
    var_name : str | None, optional
        Name of the log-likelihood variable. If None, uses the first
        variable in the log_likelihood group.
    pointwise : bool, default True
        If True, compute pointwise LOO values (needed for diagnostics).
        When False, az.loo returns no Pareto-k values, so the diagnostic
        fields come back empty (n_high_pareto_k=0, no indices).

    Returns
    -------
    LOOResult
        Container with ELPD, standard error, effective parameters,
        and Pareto-k diagnostic information.

    Raises
    ------
    ValueError
        If log_likelihood group is not present in idata.

    Example
    -------
    >>> log_lik = compute_log_likelihood(model, mcmc, model_args)
    >>> idata = add_log_likelihood_to_idata(result.idata, log_lik)
    >>> loo_result = compute_loo(idata)
    >>> print(f"ELPD: {loo_result.elpd_loo:.1f} +/- {loo_result.se_elpd:.1f}")
    >>> print(f"High Pareto-k observations: {loo_result.n_high_pareto_k}")

    Notes
    -----
    Pareto-k diagnostic interpretation:
    - k < 0.5: Good, LOO estimate is reliable
    - 0.5 < k < 0.7: Okay, some uncertainty in estimate
    - k > 0.7: Problematic, observation is highly influential
    - k > 1.0: Very problematic, moment matching recommended

    High Pareto-k values indicate observations that have undue influence
    on the posterior and may warrant investigation or model refinement.
    """
    if "log_likelihood" not in idata.groups():
        raise ValueError(
            "InferenceData does not have log_likelihood group. "
            "Use add_log_likelihood_to_idata() to add it."
        )

    # Compute LOO
    loo_data = az.loo(idata, var_name=var_name, pointwise=pointwise)

    # Pareto-k values only exist on the pointwise result; without them the
    # diagnostics are empty rather than an AttributeError.
    pareto_k = getattr(loo_data, "pareto_k", None) if pointwise else None
    if pareto_k is not None:
        high_pareto_k_indices = np.where(np.asarray(pareto_k) > 0.7)[0]
    else:
        high_pareto_k_indices = np.array([], dtype=int)
    n_high_pareto_k = len(high_pareto_k_indices)

    # Extract warning if present
    warning = None
    if hasattr(loo_data, "warning") and loo_data.warning:
        warning = str(loo_data.warning)

    return LOOResult(
        loo=loo_data,
        elpd_loo=float(loo_data.elpd_loo),
        se_elpd=float(loo_data.se),
        p_loo=float(loo_data.p_loo),
        n_high_pareto_k=n_high_pareto_k,
        high_pareto_k_indices=high_pareto_k_indices,
        warning=warning,
    )


def compare_models(
    model_dict: dict[str, az.InferenceData],
    ic: str = "loo",
) -> "pd.DataFrame":
    """Compare multiple models using information criteria.

    Compares fitted models using Leave-One-Out CV or WAIC,
    returning a ranked DataFrame with ELPD differences and weights.

    Parameters
    ----------
    model_dict : dict[str, az.InferenceData]
        Dictionary mapping model names to InferenceData objects.
        Each InferenceData must have a log_likelihood group.
    ic : str, default "loo"
        Information criterion to use. Either "loo" (LOO-CV) or "waic".

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - rank: Model ranking (0 = best)
        - elpd_{ic}: Expected log pointwise predictive density
        - p_{ic}: Effective number of parameters
        - d_{ic}: Difference from best model
        - weight: Model weights (from stacking)
        - se: Standard error of ELPD
        - dse: Standard error of ELPD difference
        - warning: Any warnings about Pareto-k values
        - scale: Scale of the ELPD values

    Example
    -------
    >>> comparison = compare_models({
    ...     "baseline": idata_baseline,
    ...     "full_model": idata_full,
    ... })
    >>> print(comparison)
    #            rank  elpd_loo  ...  warning  scale
    # full_model    0   -1234.5  ...     None    log
    # baseline      1   -1289.3  ...     None    log

    Notes
    -----
    Model comparison via ELPD is preferred over point estimates of
    fit (like R2) because it accounts for model complexity and
    predictive uncertainty. Stacking weights provide optimal
    combination weights for model averaging.
    """

    comparison = az.compare(model_dict, ic=ic)
    return comparison


def generate_prior_predictive(
    model,
    model_args: dict,
    num_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Generate prior predictive samples.

    Samples from the model's prior distribution (before conditioning
    on observed data) to check if priors produce sensible predictions.

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Should have y=None to generate
        predictions rather than conditioning on observations.
    num_samples : int, default 1000
        Number of prior predictive samples to generate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary of prior predictive samples. Keys are site names
        from the model (e.g., "user_y", "user_beta", etc.).

    Example
    -------
    >>> prior_args = {**model_args, "y": None}
    >>> prior_pred = generate_prior_predictive(model, prior_args)
    >>> print(f"Prior predictive shape: {prior_pred['user_y'].shape}")
    >>> print(f"Prior mean score: {prior_pred['user_y'].mean():.1f}")

    Notes
    -----
    Prior predictive checks are useful for:
    1. Verifying priors are not too diffuse (predictions span reasonable range)
    2. Checking priors are not too tight (allow for observed data range)
    3. Detecting prior-likelihood conflicts before fitting

    Good practice is to compare prior predictive distribution to the
    domain of valid scores (e.g., 0-100 for AOTY scores).
    """
    # Ensure y is None for prior predictive (don't condition on observations)
    pred_args = {**model_args, "y": None}

    # Create Predictive for prior (no posterior samples)
    predictive = Predictive(model, num_samples=num_samples)

    # Generate prior predictive samples
    rng_key = random.key(seed)
    prior_samples = predictive(rng_key, **pred_args)

    # Convert JAX arrays to numpy for easier manipulation
    return {k: np.array(v) for k, v in prior_samples.items()}
