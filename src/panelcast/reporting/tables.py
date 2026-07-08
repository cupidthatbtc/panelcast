"""Publication-quality summary tables for Bayesian model results.

This module provides functions to create coefficient, diagnostics, and model
comparison tables from ArviZ InferenceData objects. Tables use adaptive
precision (matching decimal places to uncertainty magnitude) and export to
both LaTeX and CSV formats for manuscript inclusion.

Key functions:
- create_coefficient_table: Parameter estimates with credible intervals
- create_diagnostics_table: R-hat, ESS, and convergence status
- create_comparison_table: Model comparison by ELPD/LOO-CV
- export_table: Dual-format export (CSV + LaTeX)

Usage:
    >>> from panelcast.reporting.tables import (
    ...     create_coefficient_table,
    ...     create_diagnostics_table,
    ...     export_table,
    ... )
    >>> coef_df = create_coefficient_table(idata, var_names=["beta", "sigma"])
    >>> export_table(coef_df, "reports/coefficients", caption="Model coefficients")
"""

from pathlib import Path
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
from uncertainties import ufloat

__all__ = [
    "create_coefficient_table",
    "create_diagnostics_table",
    "create_comparison_table",
    "create_baseline_benchmark_table",
    "create_sensitivity_summary_table",
    "export_table",
]


def _format_with_precision(
    value: float,
    uncertainty: float,
    min_decimals: int = 2,
    max_decimals: int = 6,
) -> str:
    """Format a value with adaptive precision based on uncertainty magnitude.

    Uses the Particle Data Group (PDG) convention: round uncertainty to 2
    significant figures, then match the value's decimal places to the
    uncertainty.

    Parameters
    ----------
    value : float
        The value to format.
    uncertainty : float
        The uncertainty (standard error) associated with the value.
    min_decimals : int, default 2
        Minimum decimal places to show even for large uncertainties.
    max_decimals : int, default 6
        Maximum decimal places to prevent excessive precision.

    Returns
    -------
    str
        Formatted value string with appropriate decimal places.

    Examples
    --------
    >>> _format_with_precision(1.234, 0.05)
    '1.23'
    >>> _format_with_precision(0.001234, 0.00005)
    '0.00123'
    >>> _format_with_precision(100.5, 10.0)
    '100.5'
    """
    # Handle edge cases
    if not np.isfinite(value):
        return str(value)
    if not np.isfinite(uncertainty) or uncertainty == 0:
        return f"{value:.{min_decimals}f}"

    try:
        # Create uncertain number and get its formatted string
        x = ufloat(value, uncertainty)
        # Format with 2 sig figs on uncertainty, short format
        formatted = f"{x:.2uS}"

        # Parse the formatted value part
        if "+/-" in formatted:
            val_str = formatted.split("+/-")[0].strip()
        elif "(" in formatted:
            val_str = formatted.split("(")[0].strip()
        else:
            val_str = formatted.strip()

        # Count decimals in the formatted string
        if "." in val_str:
            decimals = len(val_str.split(".")[-1])
        else:
            decimals = 0

        # Apply bounds
        decimals = max(min_decimals, min(decimals, max_decimals))
        return f"{value:.{decimals}f}"

    except (ValueError, OverflowError):
        # Fallback for problematic values
        return f"{value:.{min_decimals}f}"


def _escape_latex_param_name(name: str) -> str:
    """Escape parameter names for LaTeX compatibility.

    Parameters
    ----------
    name : str
        Parameter name that may contain special LaTeX characters.

    Returns
    -------
    str
        LaTeX-safe parameter name.

    Examples
    --------
    >>> _escape_latex_param_name("beta[0]")
    'beta[0]'  # brackets are OK in text
    >>> _escape_latex_param_name("sigma_artist")
    'sigma\\\\_artist'
    """
    # Underscores need escaping in LaTeX
    return name.replace("_", r"\_")


def create_coefficient_table(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.94,
    apply_precision: bool = True,
) -> pd.DataFrame:
    """Create publication-ready coefficient table from InferenceData.

    Extracts parameter estimates (mean), standard errors (sd), and highest
    density interval bounds from ArviZ summary. Optionally applies adaptive
    precision formatting.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with posterior samples.
    var_names : list[str] | None, optional
        List of variable names to include. If None, includes all variables.
    hdi_prob : float, default 0.94
        Probability mass for the highest density interval. Default 0.94
        follows ArviZ convention and has better coverage properties than 0.95.
    apply_precision : bool, default True
        If True, apply adaptive precision formatting to numeric values.
        If False, return raw numeric DataFrame for further processing.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Estimate, SE, CI Lower, CI Upper.
        Index contains parameter names.

    Raises
    ------
    ValueError
        If idata doesn't have posterior group or var_names are not found.

    Examples
    --------
    >>> coef_df = create_coefficient_table(idata, var_names=["user_beta"])
    >>> print(coef_df)
                  Estimate    SE  CI Lower  CI Upper
    user_beta[0]      0.12  0.03      0.06      0.18
    user_beta[1]     -0.05  0.02     -0.09     -0.01

    Notes
    -----
    The table uses highest density intervals (HDI), not equal-tailed intervals.
    For skewed posteriors, HDI provides the narrowest interval containing the
    specified probability mass.
    """
    if "posterior" not in idata.groups():
        raise ValueError("InferenceData must have 'posterior' group")

    # Handle empty var_names
    if var_names is not None and len(var_names) == 0:
        return pd.DataFrame(columns=["Estimate", "SE", "CI Lower", "CI Upper"])

    # Get ArviZ summary with stats only (no diagnostics)
    summary = pd.DataFrame(
        az.summary(
            idata,
            var_names=var_names,
            kind="stats",
            hdi_prob=hdi_prob,
            round_to="none",  # We handle precision ourselves
        )
    )

    # Find HDI column names from summary - ArviZ uses varying decimal precision
    # e.g., "hdi_3%" for 94% HDI but "hdi_5.5%" for 89% HDI
    hdi_cols = [c for c in summary.columns if c.startswith("hdi_")]
    if len(hdi_cols) != 2:
        raise ValueError(f"Expected 2 HDI columns, found: {hdi_cols}")

    # Sort to get lower and upper bounds
    hdi_cols_sorted = sorted(hdi_cols, key=lambda c: float(c.replace("hdi_", "").replace("%", "")))
    lower_col, upper_col = hdi_cols_sorted

    # Rename columns for publication
    column_mapping = {
        "mean": "Estimate",
        "sd": "SE",
        lower_col: "CI Lower",
        upper_col: "CI Upper",
    }

    # Select and rename columns
    result = summary[["mean", "sd", lower_col, upper_col]].copy()
    result = result.rename(columns=column_mapping)

    if apply_precision:
        # Convert to object dtype first to avoid FutureWarning
        result = result.astype(object)

        # Apply adaptive precision based on SE
        for idx in result.index:
            est = float(result.at[idx, "Estimate"])
            se = float(result.at[idx, "SE"])

            # Format estimate and SE
            est_formatted = _format_with_precision(est, se)
            se_formatted = _format_with_precision(se, se)

            # Determine decimal places from estimate formatting
            if "." in est_formatted:
                n_decimals = len(est_formatted.split(".")[-1])
            else:
                n_decimals = 2

            # Apply same precision to CI bounds
            ci_lower = float(result.at[idx, "CI Lower"])
            ci_upper = float(result.at[idx, "CI Upper"])

            result.at[idx, "Estimate"] = est_formatted
            result.at[idx, "SE"] = se_formatted
            result.at[idx, "CI Lower"] = f"{ci_lower:.{n_decimals}f}"
            result.at[idx, "CI Upper"] = f"{ci_upper:.{n_decimals}f}"

    return result


def create_diagnostics_table(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    rhat_threshold: float = 1.01,
    ess_threshold: int = 400,
) -> pd.DataFrame:
    """Create diagnostics table with R-hat, ESS, and convergence status.

    Extracts MCMC diagnostics from ArviZ summary and adds a pass/fail
    convergence status column based on configurable thresholds.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object with posterior samples.
    var_names : list[str] | None, optional
        List of variable names to include. If None, includes all variables.
    rhat_threshold : float, default 1.01
        Maximum acceptable R-hat value for convergence.
    ess_threshold : int, default 400
        Minimum acceptable total ESS-bulk (summed across chains), matching
        the check_convergence gate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: R-hat, ESS Bulk, ESS Tail, MCSE Mean, Status.
        Index contains parameter names.

    Raises
    ------
    ValueError
        If idata doesn't have posterior group.

    Examples
    --------
    >>> diag_df = create_diagnostics_table(idata)
    >>> print(diag_df)
                  R-hat  ESS Bulk  ESS Tail  MCSE Mean  Status
    user_beta[0]  1.001      2456      2234      0.001    Pass
    user_beta[1]  1.002      2123      1987      0.001    Pass

    Notes
    -----
    The ESS threshold is a floor on total ESS-bulk (summed across chains),
    the quantity ArviZ reports. R-hat values are formatted to 4 decimal
    places; ESS values are shown as integers.
    """
    if "posterior" not in idata.groups():
        raise ValueError("InferenceData must have 'posterior' group")

    # Handle empty var_names
    if var_names is not None and len(var_names) == 0:
        return pd.DataFrame(columns=["R-hat", "ESS Bulk", "ESS Tail", "MCSE Mean", "Status"])

    # Get ArviZ summary with diagnostics only
    summary = pd.DataFrame(az.summary(idata, var_names=var_names, kind="diagnostics"))

    # Build result DataFrame
    result = pd.DataFrame(index=summary.index)

    # R-hat: 4 decimal places
    result["R-hat"] = summary["r_hat"].apply(lambda x: f"{x:.4f}")

    # ESS: integers
    result["ESS Bulk"] = summary["ess_bulk"].astype(int)
    result["ESS Tail"] = summary["ess_tail"].astype(int)

    # MCSE Mean: adaptive precision
    if "mcse_mean" in summary.columns:
        result["MCSE Mean"] = summary["mcse_mean"].apply(
            lambda x: f"{x:.4f}" if np.isfinite(x) else str(x)
        )
    else:
        result["MCSE Mean"] = "N/A"

    # Convergence status
    def get_status(row_name: str) -> str:
        rhat = float(summary.at[row_name, "r_hat"])
        ess_bulk = float(summary.at[row_name, "ess_bulk"])

        rhat_ok = rhat <= rhat_threshold
        ess_ok = ess_bulk >= ess_threshold

        if rhat_ok and ess_ok:
            return "Pass"
        elif not rhat_ok and not ess_ok:
            return "Fail (R-hat, ESS)"
        elif not rhat_ok:
            return "Fail (R-hat)"
        else:
            return "Fail (ESS)"

    result["Status"] = [get_status(idx) for idx in result.index]

    return result


def create_comparison_table(
    model_dict: dict[str, az.InferenceData],
    ic: Literal["loo", "waic"] = "loo",
) -> pd.DataFrame:
    """Create model comparison table ranked by information criterion.

    Wraps ArviZ compare function and reformats output for publication.
    Models are ranked by ELPD (expected log pointwise predictive density)
    in descending order (best model first).

    Parameters
    ----------
    model_dict : dict[str, az.InferenceData]
        Dictionary mapping model names to InferenceData objects.
        Each InferenceData must have log_likelihood group.
    ic : {"loo", "waic"}, default "loo"
        Information criterion to use. LOO (Pareto-smoothed importance
        sampling leave-one-out) is preferred over WAIC.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ELPD, SE, p_eff, Delta, Weight.
        Index contains model names, sorted by ELPD descending.

    Raises
    ------
    ValueError
        If model_dict is empty or models lack log_likelihood group.

    Examples
    --------
    >>> models = {"baseline": idata1, "hierarchical": idata2}
    >>> comp_df = create_comparison_table(models)
    >>> print(comp_df)
                   ELPD    SE  p_eff   Delta  Weight
    hierarchical  -1234  12.3   45.2    0.00    0.85
    baseline      -1256  13.1   23.4  -22.00    0.15

    Notes
    -----
    The Delta column shows the difference from the best model (ELPD_i - ELPD_best).
    Weights are computed using stacking or pseudo-BMA depending on ArviZ version.
    Single-model comparisons return a table with that model and Delta=0, Weight=1.
    """
    if not model_dict:
        raise ValueError("model_dict cannot be empty")

    # Handle single model case
    if len(model_dict) == 1:
        name = next(iter(model_dict.keys()))
        idata = model_dict[name]

        # Compute LOO/WAIC for single model
        if ic == "loo":
            loo_result = az.loo(idata)
            elpd = loo_result.elpd_loo
            se = loo_result.se
            p_eff = loo_result.p_loo
        else:
            waic_result = az.waic(idata)
            elpd = waic_result.elpd_waic
            se = waic_result.se
            p_eff = waic_result.p_waic

        result = pd.DataFrame(
            {
                "ELPD": [f"{elpd:.1f}"],
                "SE": [f"{se:.1f}"],
                "p_eff": [f"{p_eff:.1f}"],
                "Delta": ["0.0"],
                "Weight": ["1.00"],
            },
            index=[name],
        )
        return result

    # Multiple models: use az.compare
    comparison = pd.DataFrame(az.compare(model_dict, ic=ic))

    # Build result DataFrame with renamed columns
    result = pd.DataFrame(index=comparison.index)

    # ELPD column name depends on IC
    elpd_col = f"elpd_{ic}"
    p_col = f"p_{ic}"

    result["ELPD"] = comparison[elpd_col].apply(lambda x: f"{x:.1f}")
    result["SE"] = comparison["se"].apply(lambda x: f"{x:.1f}")
    result["p_eff"] = comparison[p_col].apply(lambda x: f"{x:.1f}")

    # Delta: difference from best model
    best_elpd = comparison[elpd_col].iloc[0]  # Already sorted by ArviZ
    deltas = comparison[elpd_col] - best_elpd
    result["Delta"] = deltas.apply(lambda x: f"{x:.1f}")

    # Weights
    result["Weight"] = comparison["weight"].apply(lambda x: f"{x:.2f}")

    return result


def create_baseline_benchmark_table(
    rows: list[dict],
    levels: tuple[float, ...] = (0.80, 0.95),
) -> pd.DataFrame:
    """Format a baseline/model benchmark table for publication.

    Turns the flat rows produced by
    ``panelcast.models.baselines.BaselineScore.to_row`` (one per model x split)
    into a publication table: Model x {MAE, RMSE, R2, CRPS, coverage at each
    level, 95% interval width, PPC skewness p-value, runtime}. Empty input
    yields an empty, correctly-columned frame rather than raising, so a partial
    run still emits a (clearly empty) table instead of a TBD placeholder.

    Parameters
    ----------
    rows : list[dict]
        Flattened benchmark rows. Each must carry ``model``, ``split``, and the
        metric keys (``mae``, ``rmse``, ``r2``, ``crps``, ``cov80``/``cov95``,
        ``width95``, ``ppc_skew_p``, ``runtime_s``).
    levels : tuple of float, default (0.80, 0.95)
        Coverage levels present in the rows (drives the coverage columns).

    Returns
    -------
    pd.DataFrame
        Publication-formatted benchmark table.
    """
    cov_keys = [f"cov{int(round(level * 100))}" for level in levels]
    pub_cols = {
        "model": "Model",
        "split": "Split",
        "n_obs": "N",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
        "crps": "CRPS",
        **{
            key: f"{int(round(level * 100))}% Cov"
            for key, level in zip(cov_keys, levels, strict=True)
        },
        "width95": "95% Width",
        "ppc_skew_p": "PPC skew p",
        "runtime_s": "Runtime (s)",
    }

    if not rows:
        return pd.DataFrame(columns=list(pub_cols.values()))

    df = pd.DataFrame(rows)

    def _fmt(value, places: int) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return "—"  # em-dash
        return f"{float(numeric):.{places}f}"

    out = pd.DataFrame()
    out["Model"] = df["model"].astype(str)
    out["Split"] = df["split"].astype(str)
    if "n_obs" in df.columns:
        out["N"] = df["n_obs"].apply(lambda v: _fmt(v, 0))
    for key, places in (("mae", 2), ("rmse", 2), ("r2", 3), ("crps", 2)):
        if key in df.columns:
            out[pub_cols[key]] = df[key].apply(lambda v, p=places: _fmt(v, p))
    for key in cov_keys:
        if key in df.columns:
            out[pub_cols[key]] = df[key].apply(lambda v: _fmt(v, 3))
    for key, places in (("width95", 2), ("ppc_skew_p", 3), ("runtime_s", 2)):
        if key in df.columns:
            out[pub_cols[key]] = df[key].apply(lambda v, p=places: _fmt(v, p))

    return out


def create_sensitivity_summary_table(oat_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Format OAT sensitivity summary for publication.

    Parameters
    ----------
    oat_summary_df : pd.DataFrame
        Raw OAT summary from create_oat_summary_table().

    Returns
    -------
    pd.DataFrame
        Publication-formatted table with renamed columns and dagger symbols
        for failed-convergence rows.
    """
    df = oat_summary_df.copy()

    # Format convergence flag with dagger
    df["Status"] = df["convergence_flag"].apply(
        lambda x: x if x == "OK" else f"{x}\u2020"  # dagger symbol
    )

    # Format ELPD values with adaptive precision
    for col in ["elpd", "elpd_delta", "elpd_se"]:
        if col in df.columns:

            def _fmt_numeric(value):
                numeric = pd.to_numeric(value, errors="coerce")
                if pd.isna(numeric):
                    return "\u2014"
                return f"{float(numeric):.1f}"

            df[col] = df[col].apply(
                _fmt_numeric  # em-dash for None/non-numeric
            )

    # Select and rename columns for publication
    pub_cols = {
        "variant": "Variant",
        "parameter": "Parameter",
        "multiplier": "Multiplier",
        "elpd": "ELPD",
        "elpd_delta": "\u0394ELPD",
        "elpd_se": "SE",
        "Status": "Status",
    }

    result = df[[c for c in pub_cols if c in df.columns]].rename(columns=pub_cols)
    return result


def export_table(
    df: pd.DataFrame,
    output_path: str | Path,
    formats: tuple[str, ...] = ("csv", "tex"),
    caption: str | None = None,
    label: str | None = None,
) -> list[Path]:
    """Export table to CSV and/or LaTeX formats.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export. Index is preserved in output.
    output_path : str | Path
        Base path for output files (without extension).
        E.g., "reports/tables/coefficients" creates coefficients.csv and coefficients.tex.
    formats : tuple[str, ...], default ("csv", "tex")
        Output formats to generate. Options: "csv", "tex".
    caption : str | None, optional
        Caption for LaTeX table. If None, no caption is added.
    label : str | None, optional
        Label for LaTeX table (for cross-referencing). If None, derived from filename.

    Returns
    -------
    list[Path]
        List of paths to created files.

    Examples
    --------
    >>> paths = export_table(coef_df, "reports/coefficients", caption="Model coefficients")
    >>> print(paths)
    [Path('reports/coefficients.csv'), Path('reports/coefficients.tex')]

    Notes
    -----
    LaTeX output uses booktabs package for professional formatting.
    Special characters in parameter names (underscores, brackets) are
    escaped automatically via pandas escape=True option.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_files = []

    if "csv" in formats:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=True)
        created_files.append(csv_path)

    if "tex" in formats:
        tex_path = output_path.with_suffix(".tex")

        # Derive label from filename if not provided
        if label is None:
            label = f"tab:{output_path.stem}"

        # Build LaTeX string with booktabs
        latex_str = df.to_latex(
            index=True,
            escape=True,  # Escape special LaTeX characters
            caption=caption,
            label=label,
            position="htbp",
        )

        # Write to file
        tex_path.write_text(latex_str)
        created_files.append(tex_path)

    return created_files
