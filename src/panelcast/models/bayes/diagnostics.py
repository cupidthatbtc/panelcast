"""Convergence diagnostics for MCMC samples using ArviZ.

This module provides publication-standard convergence assessment for NumPyro models.
The recommended thresholds follow current best practices:
- R-hat < 1.01 (Vehtari et al. 2021, rank-normalized split-R-hat)
- ESS-bulk >= 400 total across chains (Vehtari et al. 2021)
- Zero divergent transitions (or minimal with documented justification)

ESS-tail is reported for inspection but is deliberately NOT part of the
pass/fail gate (see docs/DATA_LINEAGE.md section 3.4.5).

Usage:
    >>> from panelcast.models.bayes import fit_model, user_score_model
    >>> result = fit_model(user_score_model, model_args)
    >>> diags = check_convergence(result.idata)
    >>> if diags.passed:
    ...     print("Convergence OK")
    ... else:
    ...     print(f"Failing params: {diags.failing_params}")

References:
    - Vehtari et al. (2021) "Rank-normalization, folding, and localization"
    - ArviZ documentation: https://python.arviz.org/en/stable/
"""

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

__all__ = [
    "CagedChain",
    "CagedChainDiagnostics",
    "ConvergenceDiagnostics",
    "check_convergence",
    "detect_caged_chains",
    "get_divergence_info",
]


@dataclass(frozen=True)
class CagedChain:
    """One chain meeting both caged-chain criteria."""

    chain_id: int | str
    mean_num_steps: float
    mean_sigma: float
    consensus_sigma: float


@dataclass(frozen=True)
class CagedChainDiagnostics:
    """Post-fit caged-chain classification and its exact criterion."""

    scale_parameter: str
    max_num_steps: int
    tree_depth_fraction: float
    boundary_sigma: float
    consensus_ratio: float
    chains: list[CagedChain]

    @property
    def chain_ids(self) -> list[int | str]:
        return [chain.chain_id for chain in self.chains]

    def to_dict(self) -> dict:
        return {
            "count": len(self.chains),
            "chain_ids": self.chain_ids,
            "scale_parameter": self.scale_parameter,
            "criterion": {
                "mean_num_steps_gte": self.tree_depth_fraction * self.max_num_steps,
                "max_num_steps": self.max_num_steps,
                "posterior_mean_sigma_lte": self.boundary_sigma,
                "other_chain_median_ratio_gte": self.consensus_ratio,
            },
            "chains": [
                {
                    "chain_id": chain.chain_id,
                    "mean_num_steps": chain.mean_num_steps,
                    "posterior_mean_sigma": chain.mean_sigma,
                    "other_chain_median_sigma": chain.consensus_sigma,
                }
                for chain in self.chains
            ],
        }


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """Container for convergence diagnostic results.

    All diagnostic values are extracted from a single ArviZ summary call
    for efficiency. The pass/fail determination uses configurable thresholds.

    Attributes:
        rhat_max: Maximum R-hat across all parameters. Should be < 1.01.
        ess_bulk_min: Minimum ESS-bulk (total across chains). Should be >= ess_threshold.
        ess_tail_min: Minimum ESS-tail (total across chains). Informational only —
            not part of the pass/fail gate.
        divergences: Total divergent transitions across all chains.
        passed: True if the gate criteria are met: R-hat, bulk-ESS (total floor),
            and divergences. Tail ESS is reported but never gates.
        failing_params: List of parameter names failing R-hat or ESS thresholds.
        summary_df: Full ArviZ diagnostic summary DataFrame for inspection.

    Example:
        >>> diags = check_convergence(idata)
        >>> print(f"R-hat max: {diags.rhat_max:.4f}")
        >>> print(f"ESS bulk min: {diags.ess_bulk_min}")
        >>> print(f"Passed: {diags.passed}")
    """

    rhat_max: float
    ess_bulk_min: int
    ess_tail_min: int
    divergences: int
    passed: bool
    failing_params: list[str]
    summary_df: pd.DataFrame
    rhat_threshold: float
    ess_threshold: int

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ConvergenceDiagnostics({status}: "
            f"rhat_max={self.rhat_max:.4f}, "
            f"ess_bulk_min={self.ess_bulk_min}, "
            f"divergences={self.divergences})"
        )


def detect_caged_chains(
    idata: az.InferenceData,
    *,
    scale_parameter: str,
    max_tree_depth: int,
    tree_depth_fraction: float = 0.95,
    boundary_sigma: float = 0.005,
    consensus_ratio: float = 5.0,
) -> CagedChainDiagnostics:
    """Find chains that are both tree-depth saturated and at a scale boundary."""
    max_num_steps = 2**max_tree_depth - 1
    result = CagedChainDiagnostics(
        scale_parameter=scale_parameter,
        max_num_steps=max_num_steps,
        tree_depth_fraction=tree_depth_fraction,
        boundary_sigma=boundary_sigma,
        consensus_ratio=consensus_ratio,
        chains=[],
    )
    if (
        "posterior" not in idata.groups()
        or "sample_stats" not in idata.groups()
        or scale_parameter not in idata.posterior
        or "num_steps" not in idata.sample_stats
    ):
        return result

    steps = idata.sample_stats["num_steps"]
    sigma = idata.posterior[scale_parameter]
    step_means = steps.mean(dim=[dim for dim in steps.dims if dim != "chain"])
    sigma_means = sigma.mean(dim=[dim for dim in sigma.dims if dim != "chain"])
    chain_ids = list(step_means.coords["chain"].values)
    if len(chain_ids) < 2:
        return result

    caged: list[CagedChain] = []
    for chain_id in chain_ids:
        mean_steps = float(step_means.sel(chain=chain_id).values)
        mean_sigma = float(sigma_means.sel(chain=chain_id).values)
        other_sigmas = [
            float(sigma_means.sel(chain=other).values) for other in chain_ids if other != chain_id
        ]
        consensus_sigma = float(np.median(other_sigmas))
        saturated = mean_steps >= tree_depth_fraction * max_num_steps
        boundary = mean_sigma <= boundary_sigma
        separated = consensus_sigma >= consensus_ratio * mean_sigma
        if saturated and boundary and separated:
            serial_id = chain_id.item() if isinstance(chain_id, np.generic) else chain_id
            caged.append(
                CagedChain(
                    chain_id=serial_id,
                    mean_num_steps=mean_steps,
                    mean_sigma=mean_sigma,
                    consensus_sigma=consensus_sigma,
                )
            )
    return CagedChainDiagnostics(
        scale_parameter=scale_parameter,
        max_num_steps=max_num_steps,
        tree_depth_fraction=tree_depth_fraction,
        boundary_sigma=boundary_sigma,
        consensus_ratio=consensus_ratio,
        chains=caged,
    )


def check_convergence(
    idata: az.InferenceData,
    rhat_threshold: float = 1.01,
    ess_threshold: int = 400,
    allow_divergences: bool = False,
) -> ConvergenceDiagnostics:
    """Check MCMC convergence using ArviZ diagnostics.

    Extracts R-hat, ESS-bulk, ESS-tail from ArviZ summary and divergences
    from sample_stats. Returns a ConvergenceDiagnostics container with
    pass/fail status based on publication-standard thresholds.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with posterior samples. Must have at least 2 chains
        for R-hat computation and sample_stats with 'diverging' field.
    rhat_threshold : float, default 1.01
        Maximum acceptable R-hat value. Parameters with R-hat >= threshold
        are flagged as failing. Default follows publication standards.
    ess_threshold : int, default 400
        Minimum acceptable total ESS-bulk (summed across chains), matching
        ArviZ's reporting convention. Default 400 total per Vehtari et al.
        (2021) and the publication-readiness gate.
    allow_divergences : bool, default False
        If True, divergences do not cause overall failure (useful for
        sensitivity analysis where some divergences may be acceptable).
        Divergence count is still reported regardless.

    Returns
    -------
    ConvergenceDiagnostics
        Container with all diagnostic values, pass/fail status, and
        list of failing parameter names.

    Raises
    ------
    ValueError
        If idata doesn't have required groups (posterior, sample_stats).

    Example
    -------
    >>> from panelcast.models.bayes import fit_model, user_score_model
    >>> result = fit_model(user_score_model, model_args)
    >>> diags = check_convergence(result.idata, rhat_threshold=1.01)
    >>> if not diags.passed:
    ...     print(f"Convergence issues: {diags.failing_params}")
    ...     print(f"Divergences: {diags.divergences}")

    Notes
    -----
    The ESS threshold is a floor on total ESS (summed across chains), the
    same quantity az.summary() reports and the same floor the publication
    gate applies. Tail ESS is extracted for reporting but never gates.

    R-hat uses ArviZ's rank-normalized split-R-hat implementation, which
    is more robust than the original Gelman-Rubin statistic.
    """
    # Validate required groups exist
    if "posterior" not in idata.groups():
        raise ValueError("InferenceData must have 'posterior' group")
    if "sample_stats" not in idata.groups():
        raise ValueError("InferenceData must have 'sample_stats' group for divergence extraction")

    # Get diagnostic summary (R-hat, ESS-bulk, ESS-tail, MCSE)
    # kind="diagnostics" is more efficient than "all"
    summary = az.summary(idata, kind="diagnostics")

    # Extract R-hat (max across all parameters)
    rhat_max = float(summary["r_hat"].max())

    # Extract ESS (min across all parameters); ArviZ reports total ESS.
    # All-NaN columns would make int() raise; report 0 (the gate fails anyway).
    ess_bulk_min = int(summary["ess_bulk"].min()) if summary["ess_bulk"].notna().any() else 0
    ess_tail_min = int(summary["ess_tail"].min()) if summary["ess_tail"].notna().any() else 0

    # NaN diagnostics must fail, not pass: pandas skips NaN in max/min and NaN
    # compares False in the threshold filters, so a numerically blown-up
    # parameter would otherwise sail through the gate. Sites whose draws are
    # exactly constant (deterministics like beta_ceiling's effective_ceiling)
    # legitimately yield NaN and stay exempt.
    posterior = idata.posterior

    def _constant_site(label: str) -> bool:
        name = label.split("[")[0]
        if name not in posterior:
            return False
        values = np.asarray(posterior[name].values)
        return bool(np.all(values == values[:1, :1]))

    nan_labels = summary.index[summary["r_hat"].isna() | summary["ess_bulk"].isna()]
    nan_failing = [label for label in nan_labels if not _constant_site(label)]

    # Identify failing parameters
    failing_rhat = summary[summary["r_hat"] >= rhat_threshold].index.tolist()
    failing_ess = summary[summary["ess_bulk"] < ess_threshold].index.tolist()
    failing_params = list(set(failing_rhat + failing_ess + nan_failing))

    # Extract divergences from sample_stats
    if "diverging" in idata.sample_stats:
        diverging = idata.sample_stats["diverging"]
        divergences = int(diverging.sum().values)
    else:
        # No diverging field - assume zero
        divergences = 0

    # Determine pass/fail
    rhat_ok = rhat_max < rhat_threshold
    ess_ok = ess_bulk_min >= ess_threshold
    divergences_ok = (divergences == 0) or allow_divergences

    passed = rhat_ok and ess_ok and divergences_ok and not nan_failing

    return ConvergenceDiagnostics(
        rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min,
        ess_tail_min=ess_tail_min,
        divergences=divergences,
        passed=passed,
        failing_params=failing_params,
        summary_df=summary,
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
    )


def get_divergence_info(idata: az.InferenceData) -> dict:
    """Extract detailed divergence information from InferenceData.

    Provides total count and per-chain breakdown of divergent transitions.
    Useful for diagnosing which chains have problems.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with sample_stats group containing 'diverging' field.

    Returns
    -------
    dict
        Dictionary with keys:
        - total: Total divergent transitions across all chains
        - per_chain: List of divergence counts per chain
        - rate: Divergence rate as fraction of total samples
        - locations: Dict mapping chain index to draw indices where divergences occurred

    Example
    -------
    >>> info = get_divergence_info(result.idata)
    >>> print(f"Total divergences: {info['total']}")
    >>> print(f"Per chain: {info['per_chain']}")
    >>> if info['total'] > 0:
    ...     print(f"Divergence rate: {info['rate']:.4%}")

    Notes
    -----
    High divergence rates (> 0.5%) typically indicate model misspecification
    or numerical issues. Per-chain breakdown helps identify if the problem
    is isolated to specific chains.
    """
    if "sample_stats" not in idata.groups():
        return {
            "total": 0,
            "per_chain": [],
            "rate": 0.0,
            "locations": {},
        }

    if "diverging" not in idata.sample_stats:
        return {
            "total": 0,
            "per_chain": [],
            "rate": 0.0,
            "locations": {},
        }

    diverging = idata.sample_stats["diverging"]

    # Total across all chains and draws
    total = int(diverging.sum().values)

    # Per-chain counts
    per_chain = [int(diverging.sel(chain=c).sum().values) for c in diverging.coords["chain"].values]

    # Total samples for rate calculation (use .sizes for dict-like access on DataArray)
    num_chains = diverging.sizes["chain"]
    num_draws = diverging.sizes["draw"]
    total_samples = num_chains * num_draws
    rate = total / total_samples if total_samples > 0 else 0.0

    # Locations where divergences occurred (chain -> list of draw indices)
    locations = {}
    for c in diverging.coords["chain"].values:
        chain_div = diverging.sel(chain=c)
        div_draws = np.where(chain_div.values)[0].tolist()
        if div_draws:
            locations[int(c)] = div_draws

    return {
        "total": total,
        "per_chain": per_chain,
        "rate": rate,
        "locations": locations,
    }


def compute_residual_autocorrelation(
    residuals: np.ndarray,
    entity_idx: np.ndarray,
) -> dict:
    """Within-entity lag-1 autocorrelation of residuals.

    Tests AR(1) adequacy: if the model's AR term captures album-to-album
    dependency, posterior-mean residuals should be approximately white
    *within* each entity. Substantial positive lag-1 ACF means sequential
    structure is left on the table.

    Residuals must be in observation order (consecutive rows of the same
    entity are consecutive events). Pairs are formed within entities only —
    never across an entity boundary.

    Args:
        residuals: Posterior-mean residuals, shape (n_obs,).
        entity_idx: Integer entity index per observation, shape (n_obs,).

    Returns:
        Dict with lag1_acf (None when fewer than 3 pairs exist), n_pairs,
        and n_entities_multi (entities contributing at least one pair).
    """
    residuals = np.asarray(residuals, dtype=float)
    entity_idx = np.asarray(entity_idx)
    if residuals.shape != entity_idx.shape:
        raise ValueError(
            f"residuals shape {residuals.shape} != entity_idx shape {entity_idx.shape}"
        )

    # Consecutive observations belong to a pair only when the entity matches.
    if len(residuals) < 2:
        return {"lag1_acf": None, "n_pairs": 0, "n_entities_multi": 0}
    same_entity = entity_idx[1:] == entity_idx[:-1]
    r_prev = residuals[:-1][same_entity]
    r_next = residuals[1:][same_entity]
    n_pairs = int(same_entity.sum())
    n_entities_multi = int(np.unique(entity_idx[1:][same_entity]).size)

    if n_pairs < 3 or np.std(r_prev) == 0 or np.std(r_next) == 0:
        return {"lag1_acf": None, "n_pairs": n_pairs, "n_entities_multi": n_entities_multi}

    acf = float(np.corrcoef(r_prev, r_next)[0, 1])
    return {"lag1_acf": acf, "n_pairs": n_pairs, "n_entities_multi": n_entities_multi}
