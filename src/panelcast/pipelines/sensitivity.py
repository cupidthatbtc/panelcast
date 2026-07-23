"""Sensitivity analysis pipeline for robustness assessment.

This module provides orchestration for sensitivity analyses required for publication:
- SENS-01: Prior sensitivity (diffuse, default, informative priors)
- SENS-02: Threshold sensitivity (min-ratings 5, 10, 25)
- SENS-03: Feature ablation (remove feature groups to measure importance)
- SENS-04: Coefficient stability across analyses

Key outputs:
- SensitivityResult dataclass for each analysis variant
- Aggregation functions for comparison DataFrames
- Coefficient extraction for forest plot visualization

Usage:
    >>> from panelcast.pipelines.sensitivity import run_prior_sensitivity, PRIOR_CONFIGS
    >>> results = run_prior_sensitivity(model, model_args, mcmc_config)
    >>> comparison = aggregate_sensitivity_results(results, metric="elpd")
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import arviz as az
import numpy as np
import pandas as pd
import structlog

from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.evaluation.cv import (
    LOOResult,
    add_log_likelihood_to_idata,
    compute_log_likelihood,
    compute_loo,
)
from panelcast.evaluation.metrics import CRPSResult
from panelcast.models.bayes.diagnostics import ConvergenceDiagnostics, check_convergence
from panelcast.models.bayes.fit import MCMCConfig, fit_model, resolve_progress_bar
from panelcast.models.bayes.priors import PriorConfig, get_default_priors
from panelcast.paths import ArtifactPaths

__all__ = [
    "SensitivityResult",
    "PRIOR_CONFIGS",
    "OAT_MULTIPLIERS",
    "OAT_PARAMETERS",
    "run_prior_sensitivity",
    "run_threshold_sensitivity",
    "run_feature_ablation",
    "aggregate_sensitivity_results",
    "create_coefficient_comparison_df",
    "extract_coefficient_summary",
    "generate_oat_configs",
    "create_oat_summary_table",
]

logger = structlog.get_logger()


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results.

    Each sensitivity analysis variant produces one SensitivityResult containing
    the configuration used, diagnostics, evaluation metrics, and coefficient
    estimates for comparison.

    Attributes
    ----------
    name : str
        Descriptive name for this variant (e.g., "diffuse_priors", "threshold_10", "no_genre").
    config : dict
        Configuration used for this variant. Contains priors, threshold, or feature mask
        depending on the analysis type.
    idata : az.InferenceData | None
        Fitted model's InferenceData. May be None if memory conservation is needed
        (e.g., after extracting coefficients and metrics).
    convergence : ConvergenceDiagnostics | None
        Convergence diagnostics (R-hat, ESS, divergences). None if not computed.
    loo : LOOResult | None
        LOO-CV results with ELPD and Pareto-k diagnostics. None if not computed.
    crps : CRPSResult | None
        CRPS (probabilistic prediction quality) result. None if not computed.
    coefficients : pd.DataFrame
        Posterior summary for key parameters. Contains columns:
        mean, sd, hdi_3%, hdi_97% (or similar HDI bounds).

    Example
    -------
    >>> result = SensitivityResult(
    ...     name="diffuse_priors",
    ...     config={"priors": {"mu_artist_scale": 5.0}},
    ...     idata=idata,
    ...     convergence=check_convergence(idata),
    ...     loo=None,
    ...     crps=None,
    ...     coefficients=extract_coefficient_summary(idata, ["user_beta", "user_rho"]),
    ... )
    """

    name: str
    config: dict
    idata: az.InferenceData | None = None
    convergence: ConvergenceDiagnostics | None = None
    loo: LOOResult | None = None
    crps: CRPSResult | None = None
    coefficients: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


# Prior configurations for sensitivity analysis (SENS-01)
# These span from highly regularized (informative) to minimally regularized (diffuse)
PRIOR_CONFIGS: dict[str, PriorConfig] = {
    "default": get_default_priors(),
    "diffuse": PriorConfig(
        mu_artist_scale=5.0,  # Much wider (default: 1.0)
        sigma_artist_scale=2.0,  # Allow more variance (default: 0.5)
        sigma_rw_scale=0.5,  # More flexible time-varying (default: 0.1)
        sigma_rw_lognormal_loc=-2.0,  # Wider LogNormal (median ~0.14)
        sigma_rw_lognormal_sigma=1.0,  # Wider spread
        rho_scale=0.5,  # Wider AR(1) prior (default: 0.3)
        beta_scale=5.0,  # Weaker regularization (default: 1.0)
        sigma_obs_scale=2.0,  # Wider observation noise (default: 1.0)
    ),
    "informative": PriorConfig(
        mu_artist_scale=0.5,  # Tighter (default: 1.0)
        sigma_artist_scale=0.25,  # Encourage pooling (default: 0.5)
        sigma_rw_scale=0.05,  # Smoother careers (default: 0.1)
        sigma_rw_lognormal_loc=-3.5,  # Tighter LogNormal (median ~0.03)
        sigma_rw_lognormal_sigma=0.4,  # Tighter spread
        rho_scale=0.2,  # Tighter AR(1) (default: 0.3)
        beta_scale=0.5,  # Stronger regularization (default: 1.0)
        sigma_obs_scale=0.5,  # Tighter observation noise (default: 1.0)
    ),
}


def extract_coefficient_summary(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Extract posterior summary for specified parameters.

    Uses ArviZ summary to compute mean, standard deviation, and HDI bounds
    for specified parameters. Useful for comparing coefficient estimates
    across sensitivity analyses.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with posterior samples.
    var_names : list[str] | None, optional
        Parameter names to summarize. If None, summarizes all posterior variables.
        Can include prefixed names (e.g., "user_beta", "user_rho").
    prefix : str, optional
        Optional prefix to add to all var_names. Useful when you want to
        specify generic names and add model-specific prefixes.

    Returns
    -------
    pd.DataFrame
        Summary with columns: mean, sd, hdi_3%, hdi_97% (default HDI bounds).
        Index is parameter names (potentially multi-indexed for arrays).

    Example
    -------
    >>> summary = extract_coefficient_summary(idata, ["user_beta", "user_rho"])
    >>> print(summary[["mean", "hdi_3%", "hdi_97%"]])
    """
    if prefix and var_names:
        var_names = [f"{prefix}{name}" for name in var_names]

    # Use ArviZ summary with stats and diagnostics
    # hdi_prob=0.94 gives 3% and 97% bounds
    try:
        summary = az.summary(
            idata,
            var_names=var_names,
            kind="stats",
            hdi_prob=0.94,
        )
    except KeyError as e:
        # Handle case where var_names don't exist
        logger.warning("Some variables not found in posterior", error=str(e))
        if var_names is None:
            raise
        # Try each variable individually
        valid_vars = []
        for var in var_names:
            try:
                az.summary(idata, var_names=[var], kind="stats")
                valid_vars.append(var)
            except KeyError:
                pass
        if not valid_vars:
            return pd.DataFrame()
        summary = az.summary(idata, var_names=valid_vars, kind="stats", hdi_prob=0.94)

    return summary


def run_prior_sensitivity(
    model: Callable,
    model_args: dict,
    mcmc_config: MCMCConfig | None = None,
    configs: dict[str, PriorConfig] | None = None,
    compute_loo_cv: bool = True,
    obs_name: str = "user_y",
    coefficient_vars: list[str] | None = None,
    progress_bar: bool = True,
) -> dict[str, SensitivityResult]:
    """Run prior sensitivity analysis (SENS-01).

    Fits the model with multiple prior configurations to assess sensitivity
    of inference to prior choices. For publication, results should show
    that conclusions are robust across reasonable prior specifications.

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Must include all required arrays
        (artist_idx, album_seq, prev_score, X, y, n_artists, max_seq).
    mcmc_config : MCMCConfig | None, optional
        MCMC configuration. If None, uses default MCMCConfig().
    configs : dict[str, PriorConfig] | None, optional
        Dictionary mapping config names to PriorConfig objects.
        If None, uses PRIOR_CONFIGS (default, diffuse, informative).
    compute_loo_cv : bool, default True
        Whether to compute LOO-CV for each variant. Set to False for
        faster execution when only comparing coefficients.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model. Use "user_y"
        for user_score_model, "critic_y" for critic_score_model.
    coefficient_vars : list[str] | None, optional
        Parameter names to extract for coefficient comparison.
        If None, extracts all posterior variables.

    Returns
    -------
    dict[str, SensitivityResult]
        Mapping from config name to SensitivityResult.

    Example
    -------
    >>> from panelcast.models.bayes import user_score_model
    >>> results = run_prior_sensitivity(user_score_model, model_args)
    >>> for name, result in results.items():
    ...     print(f"{name}: ELPD={result.loo.elpd_loo:.1f}")

    Notes
    -----
    Each model fit is logged with progress information. For large datasets,
    consider using reduced MCMC iterations for initial sensitivity checks.
    """
    if configs is None:
        configs = PRIOR_CONFIGS

    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    results = {}

    for name, prior_config in configs.items():
        logger.info("Prior sensitivity: fitting configuration", config_name=name)

        # Add priors to model args
        args_with_priors = {**model_args, "priors": prior_config}

        # Fit model
        fit_result = fit_model(
            model, args_with_priors, config=mcmc_config, progress_bar=progress_bar
        )

        # Check convergence (allow divergences for sensitivity analysis)
        convergence = check_convergence(fit_result.idata, allow_divergences=True)

        # Extract coefficient summary
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        # Optionally compute LOO-CV
        loo_result = None
        if compute_loo_cv:
            try:
                # The trace must run under the SAME priors the variant was
                # fitted with — bare model_args would score densities under
                # get_default_priors().
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, args_with_priors, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("LOO computation failed", config_name=name, error=str(e))

        # Store result
        results[name] = SensitivityResult(
            name=name,
            config={"priors": prior_config.__dict__},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,  # CRPS requires posterior predictive, compute separately if needed
            coefficients=coefficients,
        )

        # Log summary
        status = "PASSED" if convergence.passed else "FAILED"
        logger.info(
            "Prior sensitivity result",
            config_name=name,
            convergence=status,
            rhat_max=f"{convergence.rhat_max:.4f}",
            divergences=convergence.divergences,
            elpd=f"{loo_result.elpd_loo:.1f}" if loo_result else None,
        )

    return results


def run_threshold_sensitivity(
    model: Callable,
    data_loader: Callable[[int], tuple[pd.DataFrame, dict]],
    thresholds: tuple[int, ...] = (5, 10, 25),
    mcmc_config: MCMCConfig | None = None,
    compute_loo_cv: bool = True,
    obs_name: str = "user_y",
    coefficient_vars: list[str] | None = None,
    progress_bar: bool = True,
) -> dict[int, SensitivityResult]:
    """Run threshold sensitivity analysis (SENS-02).

    Tests model inference across different minimum ratings thresholds.
    Each threshold produces a different subset of the data (e.g., threshold=10
    requires albums to have at least 10 user ratings).

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    data_loader : Callable[[int], tuple[pd.DataFrame, dict]]
        Function that takes a threshold integer and returns:
        - DataFrame with the filtered data
        - dict of model_args ready for fitting (artist_idx, X, y, etc.)
        This abstraction allows flexibility in data loading.
    thresholds : tuple of int, default (5, 10, 25)
        Minimum ratings thresholds to test.
    mcmc_config : MCMCConfig | None, optional
        MCMC configuration. If None, uses default MCMCConfig().
    compute_loo_cv : bool, default True
        Whether to compute LOO-CV for each variant.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model.
    coefficient_vars : list[str] | None, optional
        Parameter names to extract for coefficient comparison.

    Returns
    -------
    dict[int, SensitivityResult]
        Mapping from threshold to SensitivityResult.

    Example
    -------
    >>> def load_threshold_data(threshold):
    ...     df = pd.read_parquet(f"data/processed/user_score_minratings_{threshold}.parquet")
    ...     model_args = prepare_model_args(df)  # User-defined function
    ...     return df, model_args
    >>> results = run_threshold_sensitivity(user_score_model, load_threshold_data)

    Notes
    -----
    Threshold sensitivity demonstrates that conclusions hold across different
    data quality filters. Higher thresholds have fewer albums but more reliable
    scores; lower thresholds have more albums but potentially noisier scores.
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    results = {}

    for threshold in thresholds:
        logger.info("Threshold sensitivity: fitting", threshold=threshold)

        # Load data for this threshold
        df, model_args = data_loader(threshold)
        n_obs = len(df) if hasattr(df, "__len__") else model_args.get("y", []).shape[0]
        logger.info("Loaded observations", n_obs=n_obs, threshold=threshold)

        # Fit model
        fit_result = fit_model(model, model_args, config=mcmc_config, progress_bar=progress_bar)

        # Check convergence
        convergence = check_convergence(fit_result.idata, allow_divergences=True)

        # Extract coefficient summary
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        # Optionally compute LOO-CV
        loo_result = None
        if compute_loo_cv:
            try:
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, model_args, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("LOO computation failed", threshold=threshold, error=str(e))

        # Store result
        results[threshold] = SensitivityResult(
            name=f"threshold_{threshold}",
            config={"threshold": threshold, "n_obs": n_obs},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,
            coefficients=coefficients,
        )

        # Log summary
        status = "PASSED" if convergence.passed else "FAILED"
        logger.info(
            "Threshold sensitivity result",
            threshold=threshold,
            convergence=status,
            rhat_max=f"{convergence.rhat_max:.4f}",
            divergences=convergence.divergences,
            elpd=f"{loo_result.elpd_loo:.1f}" if loo_result else None,
        )

    return results


def run_feature_ablation(
    model: Callable,
    model_args: dict,
    feature_groups: dict[str, list[int]],
    mcmc_config: MCMCConfig | None = None,
    compute_loo_cv: bool = True,
    obs_name: str = "user_y",
    coefficient_vars: list[str] | None = None,
    baseline: SensitivityResult | None = None,
    progress_bar: bool = True,
    priors: PriorConfig | None = None,
) -> dict[str, SensitivityResult]:
    """Run feature ablation study (SENS-03).

    Measures the importance of each feature group by zeroing out those
    features and measuring the impact on model performance. Includes
    a "full" baseline with all features.

    Parameters
    ----------
    model : Callable
        NumPyro model function (e.g., user_score_model).
    model_args : dict
        Arguments to pass to the model. Must include "X" (feature matrix).
    feature_groups : dict[str, list[int]]
        Mapping from group name to column indices in X to ablate.
        E.g., {"genre": [0,1,2,3,4], "temporal": [5,6,7], "album_type": [8,9,10,11]}
    mcmc_config : MCMCConfig | None, optional
        MCMC configuration. If None, uses default MCMCConfig().
    compute_loo_cv : bool, default True
        Whether to compute LOO-CV for each variant.
    obs_name : str, default "user_y"
        Name of the observed variable site in the model.
    coefficient_vars : list[str] | None, optional
        Parameter names to extract for coefficient comparison.
    baseline : SensitivityResult | None, optional
        Pre-fitted full-model baseline. When provided, the redundant
        baseline refit is skipped and this result is used as "full"
        (the suite passes the prior-sensitivity "default" fit here).
    priors : PriorConfig | None, optional
        Prior configuration threaded into every fit AND its log-likelihood
        trace. Pass the same located config the baseline was fitted with;
        otherwise ablated variants fall back to get_default_priors() and
        ELPD deltas conflate feature removal with prior differences.

    Returns
    -------
    dict[str, SensitivityResult]
        Mapping from ablation name to SensitivityResult.
        Includes "full" (baseline) and "no_{group}" for each ablated group.

    Example
    -------
    >>> feature_groups = {
    ...     "genre": [0, 1, 2, 3, 4],  # Genre PCA columns
    ...     "temporal": [5, 6, 7],      # Temporal features
    ...     "album_type": [8, 9, 10],   # Album type one-hot
    ... }
    >>> results = run_feature_ablation(user_score_model, model_args, feature_groups)
    >>> for name, result in results.items():
    ...     print(f"{name}: ELPD={result.loo.elpd_loo:.1f}")

    Notes
    -----
    Feature ablation reveals which feature groups contribute most to
    predictive performance. Larger ELPD drops indicate more important features.

    Features are ablated by setting their values to zero, which assumes
    features are standardized (zero = mean). For non-standardized features,
    consider using the feature mean instead.
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    if priors is not None:
        model_args = {**model_args, "priors": priors}

    # Get original feature matrix
    X_original = model_args["X"]

    results = {}

    if baseline is not None:
        # Reuse the caller's pre-fitted full model instead of refitting it.
        logger.info("Feature ablation: reusing pre-fitted full-model baseline")
        results["full"] = baseline
        convergence = baseline.convergence
        loo_result = baseline.loo
    else:
        # Fit the full model as baseline
        logger.info("Feature ablation: fitting full model (baseline)")
        fit_result = fit_model(model, model_args, config=mcmc_config, progress_bar=progress_bar)
        convergence = check_convergence(fit_result.idata, allow_divergences=True)
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        loo_result = None
        if compute_loo_cv:
            try:
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, model_args, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("LOO computation failed for full model", error=str(e))

        results["full"] = SensitivityResult(
            name="full",
            config={"ablated_features": None, "n_features": X_original.shape[1]},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,
            coefficients=coefficients,
        )

    if convergence is not None:
        status = "PASSED" if convergence.passed else "FAILED"
        logger.info(
            "Feature ablation result",
            variant="full",
            convergence=status,
            rhat_max=f"{convergence.rhat_max:.4f}",
            divergences=convergence.divergences,
            elpd=f"{loo_result.elpd_loo:.1f}" if loo_result else None,
        )

    # Now ablate each feature group
    for group_name, col_indices in feature_groups.items():
        ablation_name = f"no_{group_name}"
        logger.info("Feature ablation: fitting", variant=ablation_name, columns=col_indices)

        # Create modified feature matrix with ablated columns set to zero
        X_ablated = np.array(X_original, copy=True)
        X_ablated[:, col_indices] = 0.0

        # Create modified model args
        ablated_args = {**model_args, "X": X_ablated}

        # Fit model
        fit_result = fit_model(model, ablated_args, config=mcmc_config, progress_bar=progress_bar)
        convergence = check_convergence(fit_result.idata, allow_divergences=True)
        coefficients = extract_coefficient_summary(fit_result.idata, var_names=coefficient_vars)

        # Optionally compute LOO-CV
        loo_result = None
        if compute_loo_cv:
            try:
                log_lik = compute_log_likelihood(
                    model, fit_result.mcmc, ablated_args, obs_name=obs_name
                )
                idata_with_ll = add_log_likelihood_to_idata(fit_result.idata, log_lik)
                loo_result = compute_loo(idata_with_ll)
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("LOO computation failed", variant=ablation_name, error=str(e))

        results[ablation_name] = SensitivityResult(
            name=ablation_name,
            config={"ablated_features": group_name, "ablated_columns": col_indices},
            idata=fit_result.idata,
            convergence=convergence,
            loo=loo_result,
            crps=None,
            coefficients=coefficients,
        )

        status = "PASSED" if convergence.passed else "FAILED"
        logger.info(
            "Feature ablation result",
            variant=ablation_name,
            convergence=status,
            rhat_max=f"{convergence.rhat_max:.4f}",
            divergences=convergence.divergences,
            elpd=f"{loo_result.elpd_loo:.1f}" if loo_result else None,
        )

    return results


def aggregate_sensitivity_results(
    results: dict[str, SensitivityResult],
    metric: str = "elpd",
) -> pd.DataFrame:
    """Aggregate sensitivity results into a comparison DataFrame.

    Creates a summary table comparing all sensitivity variants on the
    specified metric.

    Parameters
    ----------
    results : dict[str, SensitivityResult]
        Dictionary mapping variant names to SensitivityResult objects.
    metric : str, default "elpd"
        Metric to aggregate. Options:
        - "elpd": ELPD from LOO-CV (higher is better)
        - "crps": Mean CRPS (lower is better)
        - "convergence": Convergence diagnostics summary
        - "coefficients": Coefficient estimates (mean, se)

    Returns
    -------
    pd.DataFrame
        Comparison table with rows for each variant and columns for:
        - name: Variant name
        - metric value(s)
        - convergence_passed: Whether convergence passed
        - divergences: Number of divergent transitions
        - For coefficients: additional columns for each parameter

    Example
    -------
    >>> results = run_prior_sensitivity(model, model_args)
    >>> comparison = aggregate_sensitivity_results(results, metric="elpd")
    >>> print(comparison)
    #                   elpd  elpd_se  convergence_passed  divergences
    # default        -1234.5     45.2               True            0
    # diffuse        -1256.3     48.1               True            3
    # informative    -1240.1     44.8               True            0
    """
    rows = []

    for name, result in results.items():
        row = {"name": name}

        # Add convergence info
        if result.convergence is not None:
            row["convergence_passed"] = result.convergence.passed
            row["divergences"] = result.convergence.divergences
            row["rhat_max"] = result.convergence.rhat_max
            row["ess_bulk_min"] = result.convergence.ess_bulk_min
        else:
            row["convergence_passed"] = None
            row["divergences"] = None
            row["rhat_max"] = None
            row["ess_bulk_min"] = None

        if metric == "elpd":
            if result.loo is not None:
                row["elpd"] = result.loo.elpd_loo
                row["elpd_se"] = result.loo.se_elpd
                row["p_loo"] = result.loo.p_loo
                row["n_high_pareto_k"] = result.loo.n_high_pareto_k
            else:
                row["elpd"] = None
                row["elpd_se"] = None
                row["p_loo"] = None
                row["n_high_pareto_k"] = None

        elif metric == "crps":
            if result.crps is not None:
                row["mean_crps"] = result.crps.mean_crps
                row["n_obs"] = result.crps.n_obs
            else:
                row["mean_crps"] = None
                row["n_obs"] = None

        elif metric == "convergence":
            # Already added convergence info above
            pass

        elif metric == "coefficients":
            # Add coefficient estimates from the summary
            if not result.coefficients.empty:
                for param in result.coefficients.index:
                    row[f"{param}_mean"] = result.coefficients.loc[param, "mean"]
                    if "sd" in result.coefficients.columns:
                        row[f"{param}_sd"] = result.coefficients.loc[param, "sd"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Handle empty results
    if df.empty:
        return df

    df = df.set_index("name")

    # Sort by ELPD (descending) if available
    if metric == "elpd" and "elpd" in df.columns:
        df = df.sort_values("elpd", ascending=False)

    return df


def create_coefficient_comparison_df(
    results: dict[str, SensitivityResult],
    params: list[str],
) -> pd.DataFrame:
    """Create coefficient comparison DataFrame for forest plots.

    Extracts specified parameter estimates from each sensitivity variant
    into a format suitable for forest plot visualization.

    Parameters
    ----------
    results : dict[str, SensitivityResult]
        Dictionary mapping variant names to SensitivityResult objects.
    params : list[str]
        Parameter names to compare (e.g., ["user_beta[0]", "user_rho"]).
        Must match the index in each result's coefficients DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - variant: Name of the sensitivity variant
        - param: Parameter name
        - mean: Posterior mean
        - lower: Lower HDI bound (e.g., 3%)
        - upper: Upper HDI bound (e.g., 97%)

        This format is suitable for plotting with:
        >>> for param in params:
        ...     subset = df[df["param"] == param]
        ...     plt.errorbar(subset["mean"], subset["variant"],
        ...                  xerr=[subset["mean"]-subset["lower"],
        ...                        subset["upper"]-subset["mean"]])

    Example
    -------
    >>> results = run_prior_sensitivity(model, model_args, coefficient_vars=["user_rho"])
    >>> forest_df = create_coefficient_comparison_df(results, ["user_rho"])
    >>> print(forest_df)
    #        variant        param   mean  lower  upper
    # 0      default     user_rho  0.150  0.085  0.215
    # 1      diffuse     user_rho  0.142  0.078  0.206
    # 2  informative     user_rho  0.155  0.090  0.220
    """
    rows = []

    for variant_name, result in results.items():
        if result.coefficients.empty:
            continue

        for param in params:
            if param not in result.coefficients.index:
                logger.warning(
                    "Parameter not found in coefficients",
                    param=param,
                    variant=variant_name,
                )
                continue

            coef_row = result.coefficients.loc[param]

            # Extract HDI bounds - try common column names
            lower = None
            upper = None

            # ArviZ summary uses hdi_X% format
            for col in coef_row.index:
                if "hdi" in col.lower() and "%" in col:
                    pct = float(col.replace("hdi_", "").replace("%", ""))
                    if pct < 50:
                        lower = coef_row[col]
                    else:
                        upper = coef_row[col]

            rows.append(
                {
                    "variant": variant_name,
                    "param": param,
                    "mean": coef_row["mean"],
                    "lower": lower,
                    "upper": upper,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# One-At-a-Time (OAT) Prior Sensitivity Analysis
# ============================================================================

OAT_MULTIPLIERS = (0.5, 2.0, 5.0)
OAT_PARAMETERS = (
    "mu_artist_scale",
    "sigma_artist_scale",
    "sigma_rw_scale",
    "rho_scale",
    "beta_scale",
    "sigma_obs_scale",
    "n_exponent_scale",
)


def generate_oat_configs(
    base: PriorConfig | None = None,
    parameters: tuple[str, ...] = OAT_PARAMETERS,
    multipliers: tuple[float, ...] = OAT_MULTIPLIERS,
) -> dict[str, PriorConfig]:
    """Generate one-at-a-time perturbation configs for sensitivity analysis.

    For each parameter x multiplier combination, creates a PriorConfig that
    differs from the base in exactly one field (the target parameter is
    multiplied by the given multiplier).

    Parameters
    ----------
    base : PriorConfig | None, optional
        Base configuration. If None, uses get_default_priors().
    parameters : tuple[str, ...], default OAT_PARAMETERS
        Prior parameter names to perturb.
    multipliers : tuple[float, ...], default OAT_MULTIPLIERS
        Multiplier values to apply.

    Returns
    -------
    dict[str, PriorConfig]
        Dictionary with "default" baseline plus "{param}_x{mult}" variants.
    """
    import dataclasses

    if base is None:
        base = get_default_priors()

    configs = {"default": base}
    base_dict = dataclasses.asdict(base)

    for param in parameters:
        if param not in base_dict:
            logger.warning("OAT parameter not in PriorConfig", param=param)
            continue
        for mult in multipliers:
            # Format multiplier: x0.5, x2, x5
            mult_str = f"x{mult:g}"
            name = f"{param}_{mult_str}"
            modified = dict(base_dict)
            modified[param] = base_dict[param] * mult
            configs[name] = PriorConfig(**modified)

    return configs


def create_oat_summary_table(
    results: dict[str, SensitivityResult],
    base_name: str = "default",
) -> pd.DataFrame:
    """Create convergence-aware OAT summary table from sensitivity results.

    Parses parameter/multiplier from variant names, computes ELPD deltas
    from baseline, and flags failed-convergence variants.

    Parameters
    ----------
    results : dict[str, SensitivityResult]
        Dictionary mapping variant names to SensitivityResult objects.
        Should include a baseline entry (default name: "default").
    base_name : str, default "default"
        Name of the baseline variant for ELPD delta computation.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: parameter, multiplier, elpd, elpd_delta,
        elpd_se, convergence_passed, convergence_flag, eligible_for_ranking.
        Eligible variants sorted by |elpd_delta| descending, ineligible appended.
    """
    rows = []

    base_result = results.get(base_name)
    base_elpd = None
    if base_result and base_result.loo:
        try:
            base_elpd_value = float(base_result.loo.elpd_loo)
            if np.isfinite(base_elpd_value):
                base_elpd = base_elpd_value
        except (TypeError, ValueError):
            base_elpd = None
    base_convergence_passed = (
        base_result.convergence.passed if (base_result and base_result.convergence) else None
    )
    base_eligible = (base_convergence_passed is True) and (base_elpd is not None)

    for name, result in results.items():
        # Parse parameter and multiplier from name
        if name == base_name:
            parameter = "baseline"
            multiplier = 1.0
        else:
            # Format: {param}_x{mult}
            parts = name.rsplit("_x", 1)
            if len(parts) == 2:
                parameter = parts[0]
                try:
                    multiplier = float(parts[1])
                except ValueError:
                    parameter = name
                    multiplier = None
            else:
                parameter = name
                multiplier = None

        # Extract metrics
        elpd = None
        elpd_se = None
        if result.loo:
            try:
                elpd_candidate = float(result.loo.elpd_loo)
                if np.isfinite(elpd_candidate):
                    elpd = elpd_candidate
            except (TypeError, ValueError):
                elpd = None
            try:
                elpd_se_candidate = float(result.loo.se_elpd)
                if np.isfinite(elpd_se_candidate):
                    elpd_se = elpd_se_candidate
            except (TypeError, ValueError):
                elpd_se = None
        convergence_passed = result.convergence.passed if result.convergence else None
        if convergence_passed is True:
            convergence_flag = "OK"
        elif convergence_passed is False:
            convergence_flag = "FAILED"
        else:
            convergence_flag = "MISSING"
        eligible = (convergence_passed is True) and (elpd is not None) and base_eligible

        # Compute delta from baseline
        if eligible and elpd is not None and name != base_name and base_elpd is not None:
            elpd_delta = elpd - base_elpd
        elif name == base_name and eligible:
            elpd_delta = 0.0
        else:
            elpd_delta = None

        rows.append(
            {
                "variant": name,
                "parameter": parameter,
                "multiplier": multiplier,
                "elpd": elpd,
                "elpd_delta": elpd_delta,
                "elpd_se": elpd_se,
                "convergence_passed": convergence_passed,
                "convergence_flag": convergence_flag,
                "eligible_for_ranking": eligible,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sort: eligible variants by |elpd_delta| descending, ineligible at bottom
    eligible_mask = df["eligible_for_ranking"]
    eligible_df = df[eligible_mask].copy()
    ineligible_df = df[~eligible_mask].copy()

    if not eligible_df.empty and "elpd_delta" in eligible_df.columns:
        eligible_df = (
            eligible_df.assign(_abs_delta=eligible_df["elpd_delta"].abs())
            .sort_values("_abs_delta", ascending=False)
            .drop(columns="_abs_delta")
        )

    return pd.concat([eligible_df, ineligible_df], ignore_index=True)


def run_split_seed_sensitivity(
    source_df: pd.DataFrame,
    posterior_samples: dict,
    summary: dict,
    seeds: tuple[int, ...] = (42, 43),
    test_size: float = 0.15,
    val_size: float = 0.15,
    interval: float = 0.95,
    seed_offset: int = 11,
) -> dict[str, dict]:
    """Cold-start coverage sensitivity to the artist-disjoint split seed.

    The disjoint split seed has been pinned (42) for every published number;
    this axis asks how much the cold-start coverage figure moves when the
    held-out artist set changes. It is deliberately CHEAP: no refit and no
    feature rebuild — the population-level cold-start predictive (mean
    feature vector, median observation count) is scored against each seed's
    test artists, so the axis isolates test-set composition, which dominates
    the seed effect. Per-artist feature variation is not included.

    Parameters
    ----------
    source_df : pd.DataFrame
        The primary processed dataset (pre-split).
    posterior_samples : dict
        Flattened posterior samples from the fitted model.
    summary : dict
        Training summary (supplies domain columns, transform, centering).
    seeds : tuple[int, ...], default (42, 43)
        Disjoint split seeds to compare (42 = the pinned publication seed).
    test_size, val_size : float
        GroupShuffleSplit proportions (must match create_splits defaults).
    interval : float, default 0.95
        Central interval level for the coverage figure.
    seed_offset : int, default 11
        Offset for the predictive rng (decoupled from the split seed).

    Returns
    -------
    dict[str, dict]
        Per-seed coverage rows plus a "spread" entry (max - min coverage).
    """
    import jax.numpy as jnp

    from panelcast.data.split import entity_disjoint_split
    from panelcast.models.bayes.predict import predict_new_entity
    from panelcast.pipelines.training_summary import ar_center_on_model_scale

    ds = summary.get("dataset", {})
    entity_col = ds.get("entity_col", "Artist")
    target_col = ds.get("target_col", "User_Score")
    prefix = ds.get("model_prefix", "user")
    bounds = tuple(ds.get("target_bounds", (0.0, 100.0)))
    target_transform = summary.get("target_transform", "identity")
    logit_offset = float(summary.get("logit_offset", 0.5))
    n_features = int(np.asarray(posterior_samples[f"{prefix}_beta"]).shape[-1])
    n_reviews_median = float(summary.get("n_reviews_stats", {}).get("median", 100.0))
    global_mean = float(summary.get("global_mean_score", (bounds[0] + bounds[1]) / 2))
    ar_center = float(ar_center_on_model_scale(summary))
    # Match the cold-start evaluation's predictive distribution exactly
    # (evaluate._run_new_artist_predictive); scoring coverage under a different
    # observation model than metrics.json silently shifts the published
    # seed-sensitivity number for any non-default fit.
    priors_obj = PriorConfig(**summary["priors"])
    learn_n_exponent = bool(summary.get("learn_n_exponent", False))
    fixed_n_exponent = float(summary.get("n_exponent", 0.0) or 0.0)
    obs_kwargs: dict = {}
    if not learn_n_exponent and fixed_n_exponent != 0.0:
        obs_kwargs["fixed_n_exponent"] = fixed_n_exponent

    prev = global_mean
    if target_transform != "identity":
        from panelcast.models.bayes.transforms import get_transform

        prev = float(
            get_transform(target_transform, target_bounds=bounds, offset=logit_offset).forward(prev)
        )

    lo_q = 100.0 * (1.0 - interval) / 2.0
    hi_q = 100.0 - lo_q

    results: dict[str, dict] = {}
    for seed in seeds:
        _, _, test_df = entity_disjoint_split(
            source_df,
            entity_col=entity_col,
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
        )
        y_true = test_df[target_col].to_numpy(dtype=float)
        n = len(y_true)
        pred = predict_new_entity(
            posterior_samples,
            X_new=jnp.zeros((n, n_features), dtype=jnp.float32),
            prev_score=jnp.full((n,), prev, dtype=jnp.float32),
            n_reviews_new=jnp.full((n,), n_reviews_median, dtype=jnp.float32),
            prefix=f"{prefix}_",
            seed=seed_offset + seed,
            target_bounds=bounds,
            likelihood_df=float(summary.get("likelihood_df", 4.0)),
            likelihood_family=priors_obj.likelihood_family,
            skew_tailweight=priors_obj.skew_tailweight,
            discretize_observation=priors_obj.discretize_observation,
            target_transform=target_transform,
            logit_offset=logit_offset,
            ar_center=ar_center,
            **obs_kwargs,
        )
        y_samples = np.asarray(pred["y"]).reshape(-1, n)
        lo = np.percentile(y_samples, lo_q, axis=0)
        hi = np.percentile(y_samples, hi_q, axis=0)
        coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
        results[f"seed_{seed}"] = {
            "seed": seed,
            "n_test": n,
            "n_test_artists": int(test_df[entity_col].nunique()),
            "interval": interval,
            "coverage": coverage,
            "y_true_mean": float(np.mean(y_true)),
        }
        logger.info(
            "split_seed_sensitivity",
            seed=seed,
            coverage=round(coverage, 4),
            n_test=n,
        )

    coverages = [row["coverage"] for row in results.values()]
    results["spread"] = {
        "coverage_max_minus_min": float(max(coverages) - min(coverages)),
        "seeds": list(seeds),
    }
    return results


def _feature_groups_from_names(
    feature_cols: list[str],
    summary: dict,
) -> dict[str, list[int]]:
    """Map descriptor ablation groups onto feature-matrix column indices.

    Groups are derived from the canonical feature names each block emits:
    genre PCA components, entity-history priors (incl. the debut flag), and
    the temporal block columns. Empty groups are dropped.
    """
    ds = summary.get("dataset", {})
    prefix = ds.get("model_prefix", "user")
    secondary = ds.get("secondary_prefix") or "critic"
    temporal_names = {
        "album_sequence",
        "career_years",
        "release_gap_days",
        "release_year",
        "date_risk_ordinal",
        "date_missing",
    }

    def indices(predicate) -> list[int]:
        return [i for i, col in enumerate(feature_cols) if predicate(col)]

    groups = {
        "genre": indices(lambda c: c.startswith("genre_pc")),
        "artist_history": indices(
            lambda c: c.startswith(f"{prefix}_prior")
            or c.startswith(f"{prefix}_trajectory")
            or c.startswith(f"{secondary}_prior")
            or c.startswith(f"{secondary}_trajectory")
            or c == "is_debut"
        ),
        "temporal": indices(lambda c: c in temporal_names),
    }
    return {name: cols for name, cols in groups.items() if cols}


def run_sensitivity_suite(ctx) -> dict:
    """Run the opt-in sensitivity stage: priors, ablations, split seed.

    Axes:
    - "priors": refits under PRIOR_CONFIGS (default/diffuse/informative),
      each located for AR centering like the production fit.
    - "ablation": feature-group ablations, REUSING the prior-sensitivity
      "default" fit as the full-model baseline (no redundant refit).
    - "split_seed": cold-start disjoint coverage across split seeds 42/43
      (no refit; see run_split_seed_sensitivity).

    Threshold sensitivity (run_threshold_sensitivity) is not part of the
    default suite: it needs per-threshold split/feature artifacts that the
    pipeline only builds for the primary threshold.

    Writes reports/sensitivity/sensitivity_results.json and
    reports/sensitivity/oat_summary.csv (when prior axes ran).
    """
    import json

    from panelcast.models.bayes.model import make_score_model
    from panelcast.models.bayes.predict import extract_posterior_samples
    from panelcast.pipelines.train_bayes import (
        MODEL_ARGS_METADATA_KEYS,
        _apply_max_albums_cap,
        load_training_data,
        locate_level_prior,
    )
    from panelcast.pipelines.training_summary import load_training_summary

    descriptor = ctx.descriptor
    prefix = descriptor.model_prefix
    paths = ArtifactPaths.from_ctx(ctx)
    axes = tuple(getattr(ctx, "sensitivity_axes", ("priors", "ablation", "split_seed")))
    summary = load_training_summary(paths.models / "training_summary.json").to_json_dict()

    # Replicate the training-stage data prep (same gates as the fitted model).
    # The pooling gate reuses the EFFECTIVE value the train stage resolved and
    # recorded, not the configured tri-state.
    entity_group_pooling = bool(summary.get("entity_group_pooling", False))
    model_args, feature_cols, _train_df, _imputation = load_training_data(
        features_path=paths.features / "train_features.parquet",
        splits_path=resolve_split_dir(paths.splits, SplitType.WITHIN_ENTITY_TEMPORAL)
        / "train.parquet",
        min_albums_filter=getattr(ctx, "min_albums_filter", 2),
        descriptor=descriptor,
        debut_prev_score_source=summary.get("debut_prev_score_source", "train_mean"),
        target_transform=summary.get("target_transform", "identity"),
        logit_offset=float(summary.get("logit_offset", 0.5)),
        ar_center=(summary.get("priors") or {}).get("ar_center", "global"),
        entity_group_pooling=entity_group_pooling,
        # Replay the fitted model's recorded imputation — never a re-fit
        # median; sensitivity must see exactly the X the model saw.
        impute_missing=bool((summary.get("feature_scaler") or {}).get("imputation")),
        imputation_record=(summary.get("feature_scaler") or {}).get("imputation"),
    )
    artist_album_counts = model_args.pop("artist_album_counts")
    ar_center_value = float(model_args.pop("ar_center_value", 0.0))
    # Strip every bookkeeping key via the shared list so new metadata added to
    # prepare_model_data can't leak into mcmc.run(**model_args) here.
    for key in MODEL_ARGS_METADATA_KEYS:
        model_args.pop(key, None)
    model_args = _apply_max_albums_cap(
        model_args, getattr(ctx, "max_albums", 50), artist_album_counts
    )
    X = np.asarray(model_args["X"])
    std = X.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    model_args["X"] = ((X - X.mean(axis=0)) / std_safe).astype(np.float32)
    model_args["n_exponent"] = getattr(ctx, "n_exponent", 0.0)
    model_args["learn_n_exponent"] = getattr(ctx, "learn_n_exponent", False)
    model_args["n_ref"] = summary.get("n_ref")
    model_args["likelihood_df"] = float(summary.get("likelihood_df", 4.0))
    model_args["target_bounds"] = tuple(descriptor.target_bounds)

    mcmc_config = MCMCConfig(
        num_warmup=getattr(ctx, "num_warmup", 500),
        num_samples=getattr(ctx, "num_samples", 500),
        num_chains=getattr(ctx, "num_chains", 2),
        seed=getattr(ctx, "seed", 42),
        target_accept_prob=getattr(ctx, "target_accept", 0.9),
        max_tree_depth=getattr(ctx, "max_tree_depth", 10),
        # Sensitivity refits are diagnostic-scale; 'auto' resolves in train only.
        chain_method=(
            "sequential"
            if getattr(ctx, "chain_method", "sequential") == "auto"
            else getattr(ctx, "chain_method", "sequential")
        ),
    )
    model = make_score_model(prefix)
    obs_name = f"{prefix}_y"
    progress_bar = resolve_progress_bar(getattr(ctx, "progress_bar", None))

    payload: dict = {"axes": list(axes), "prefix": prefix}
    prior_results: dict[str, SensitivityResult] = {}

    def _locate(config: PriorConfig) -> PriorConfig:
        return locate_level_prior(
            replace(config, entity_group_pooling=entity_group_pooling),
            ar_center_value=ar_center_value,
            target_transform=summary.get("target_transform", "identity"),
            logit_offset=float(summary.get("logit_offset", 0.5)),
            target_bounds=tuple(descriptor.target_bounds),
        )

    if "priors" in axes:
        located = {name: _locate(config) for name, config in PRIOR_CONFIGS.items()}
        prior_results = run_prior_sensitivity(
            model,
            model_args,
            mcmc_config=mcmc_config,
            configs=located,
            obs_name=obs_name,
            progress_bar=progress_bar,
        )
        payload["priors"] = (
            aggregate_sensitivity_results(prior_results).reset_index().to_dict(orient="records")
        )

    if "ablation" in axes:
        feature_groups = _feature_groups_from_names(feature_cols, summary)
        ablation_results = run_feature_ablation(
            model,
            model_args,
            feature_groups,
            mcmc_config=mcmc_config,
            obs_name=obs_name,
            baseline=prior_results.get("default"),
            progress_bar=progress_bar,
            # Same located config the "full" baseline was fitted under.
            priors=_locate(PRIOR_CONFIGS["default"]),
        )
        payload["ablation"] = (
            aggregate_sensitivity_results(ablation_results).reset_index().to_dict(orient="records")
        )

    if "split_seed" in axes:
        import arviz as az_local

        manifest = json.loads((paths.models / "manifest.json").read_text(encoding="utf-8"))
        model_name = manifest["current"][f"{prefix}_score"]
        model_path = paths.models / model_name
        if model_path.suffix != ".nc":
            model_path = model_path.with_suffix(".nc")
        idata = az_local.from_netcdf(model_path)
        posterior_samples = extract_posterior_samples(idata)
        source_path = paths.processed / (
            descriptor.processed_name(getattr(ctx, "min_ratings", None)) + ".parquet"
        )
        source_df = pd.read_parquet(source_path)
        payload["split_seed"] = run_split_seed_sensitivity(
            source_df,
            posterior_samples,
            summary,
            seeds=tuple(getattr(ctx, "sensitivity_split_seeds", (42, 43))),
        )

    out_dir = paths.reports / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitivity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    if prior_results:
        oat_table = create_oat_summary_table(prior_results)
        oat_table.to_csv(out_dir / "oat_summary.csv", index=False)

    logger.info("sensitivity_suite_complete", path=str(out_path), axes=list(axes))
    return {"sensitivity_results": str(out_path), "axes": list(axes)}
