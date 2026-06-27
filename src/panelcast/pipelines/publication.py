"""Publication artifact generation pipeline.

Generates publication-ready tables, figures, and model cards from
evaluation results.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.evaluation.calibration import ReliabilityData
from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.reporting.figures import (
    get_trace_plot_vars,
    save_artist_prediction_plot,
    save_posterior_plot,
    save_ppc_density_plot,
    save_predictions_plot,
    save_reliability_plot,
    save_trace_plot,
    select_artist_subsets,
)
from panelcast.reporting.model_card import (
    create_default_model_card_data,
    update_model_card_with_results,
    write_model_card,
)
from panelcast.reporting.tables import (
    create_coefficient_table,
    create_diagnostics_table,
    export_table,
)

if TYPE_CHECKING:
    import arviz as az

    from panelcast.pipelines.stages import StageContext

log = structlog.get_logger()

SECONDARY_SPLIT = str(SplitType.ENTITY_DISJOINT.value)


@dataclass(frozen=True)
class _CoverageLike:
    empirical: float
    interval_width: float | None = None


@dataclass(frozen=True)
class _PointMetricsLike:
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class _LooLike:
    elpd_loo: float
    se_elpd: float


@dataclass(frozen=True)
class _ConvergenceLike:
    passed: bool
    rhat_max: float | None
    ess_bulk_min: float | None
    ess_tail_min: float | None
    divergences: int
    failing_params: list[str]


def _safe_float(value: Any) -> float | None:
    """Best-effort conversion to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Best-effort conversion to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_primary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Resolve primary split metrics from legacy or split-aware payloads."""
    if "splits" in metrics and "primary_split" in metrics:
        split_name = metrics.get("primary_split")
        splits = metrics.get("splits", {})
        if isinstance(splits, dict) and isinstance(split_name, str):
            split_metrics = splits.get(split_name)
            if isinstance(split_metrics, dict):
                return split_metrics
    return metrics


def _parse_coverage_results(primary_metrics: dict[str, Any]) -> dict[float, _CoverageLike] | None:
    """Parse coverage results from current or legacy calibration payloads."""
    calibration = primary_metrics.get("calibration", {})
    if not isinstance(calibration, dict):
        return None

    parsed: dict[float, _CoverageLike] = {}

    # Current schema: calibration.coverages.{prob}.empirical
    coverages = calibration.get("coverages", {})
    if isinstance(coverages, dict):
        for prob_key, entry in coverages.items():
            if not isinstance(entry, dict):
                continue
            nominal = _safe_float(entry.get("nominal", prob_key))
            empirical = _safe_float(entry.get("empirical"))
            if nominal is None or empirical is None or not (0.0 < nominal < 1.0):
                continue
            width = _safe_float(entry.get("interval_width"))
            parsed[nominal] = _CoverageLike(empirical=empirical, interval_width=width)

    # Legacy schema: coverage_90, coverage_50, ...
    for key, value in calibration.items():
        if not isinstance(key, str) or not key.startswith("coverage_"):
            continue
        if key == "coverages":
            continue
        suffix = key.split("_", 1)[1]
        nominal = _safe_float(suffix)
        empirical = _safe_float(value)
        if nominal is None or empirical is None:
            continue
        prob = nominal / 100.0
        if 0.0 < prob < 1.0 and prob not in parsed:
            parsed[prob] = _CoverageLike(empirical=empirical)

    return parsed or None


def _parse_point_metrics(primary_metrics: dict[str, Any]) -> _PointMetricsLike | None:
    """Parse point metrics for model card summary."""
    candidate_sources: list[dict[str, Any]] = []
    point_metrics = primary_metrics.get("point_metrics", {})
    if isinstance(point_metrics, dict):
        candidate_sources.append(point_metrics)
    candidate_sources.append(primary_metrics)

    for source in candidate_sources:
        mae = _safe_float(source.get("mae"))
        rmse = _safe_float(source.get("rmse"))
        r2 = _safe_float(source.get("r2"))
        if mae is not None and rmse is not None and r2 is not None:
            return _PointMetricsLike(mae=mae, rmse=rmse, r2=r2)

    return None


def _parse_loo_result(primary_metrics: dict[str, Any]) -> _LooLike | None:
    """Parse LOO metrics from evaluation payload."""
    info = primary_metrics.get("info_criteria", {})
    if isinstance(info, dict):
        loo = info.get("loo", {})
        if isinstance(loo, dict):
            elpd = _safe_float(loo.get("elpd"))
            se = _safe_float(loo.get("se"))
            if elpd is not None and se is not None:
                return _LooLike(elpd_loo=elpd, se_elpd=se)
    return None


def _parse_convergence(
    diagnostics: dict[str, Any],
    metrics: dict[str, Any],
) -> _ConvergenceLike | None:
    """Parse convergence diagnostics from diagnostics.json or metrics fallback."""
    payload: dict[str, Any] = {}
    if isinstance(diagnostics, dict) and diagnostics:
        payload = diagnostics
    elif isinstance(metrics, dict):
        maybe = metrics.get("diagnostics")
        if isinstance(maybe, dict):
            payload = maybe

    if not payload:
        return None

    rhat_max = _safe_float(payload.get("rhat_max"))
    ess_bulk_min = _safe_float(payload.get("ess_bulk_min"))
    try:
        divergences_int = int(payload.get("divergences", 0))
    except (TypeError, ValueError):
        divergences_int = 0

    ess_tail = _safe_float(payload.get("ess_tail_min"))
    if ess_tail is None:
        ess_tail = ess_bulk_min

    # Keep partial diagnostics when available (e.g., single-chain runs with no R-hat).
    if (
        rhat_max is None
        and ess_bulk_min is None
        and ess_tail is None
        and divergences_int == 0
        and "passed" not in payload
    ):
        return None

    return _ConvergenceLike(
        passed=bool(payload.get("passed", False)),
        rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min,
        ess_tail_min=ess_tail,
        divergences=divergences_int,
        failing_params=list(payload.get("failing_params", [])),
    )


def _build_publication_readiness(
    *,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    training_summary: dict[str, Any],
    artifact_errors: list[dict[str, str]],
    require_secondary_split: bool,
) -> dict[str, Any]:
    """Build publication-readiness checks from generated artifacts."""
    checks: list[dict[str, Any]] = []

    def add_check(name: str, severity: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "name": name,
                "severity": severity,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    mcmc_cfg = training_summary.get("mcmc_config", {})
    if not isinstance(mcmc_cfg, dict):
        mcmc_cfg = {}
    num_chains = _safe_int(mcmc_cfg.get("num_chains"))
    if num_chains is None:
        add_check(
            "mcmc_num_chains_present",
            "critical",
            False,
            "training_summary.mcmc_config.num_chains missing or invalid",
        )
    else:
        add_check(
            "mcmc_min_2_chains",
            "critical",
            num_chains >= 2,
            f"num_chains={num_chains} (required >=2 for R-hat)",
        )
        add_check(
            "mcmc_recommended_4_chains",
            "recommended",
            num_chains >= 4,
            f"num_chains={num_chains} (recommended >=4 for publication-grade diagnostics)",
        )

    diag_passed = diagnostics.get("passed")
    add_check(
        "convergence_passed",
        "critical",
        diag_passed is True,
        f"diagnostics.passed={diag_passed}",
    )

    rhat_max = _safe_float(diagnostics.get("rhat_max"))
    add_check(
        "rhat_available",
        "critical",
        rhat_max is not None,
        f"rhat_max={rhat_max}",
    )
    if rhat_max is not None:
        rhat_threshold = _safe_float(diagnostics.get("rhat_threshold")) or 1.01
        add_check(
            "rhat_within_threshold",
            "critical",
            rhat_max < rhat_threshold,
            f"rhat_max={rhat_max:.4f}, threshold={rhat_threshold:.4f}",
        )

    ess_bulk_min = _safe_float(diagnostics.get("ess_bulk_min"))
    add_check(
        "ess_available",
        "critical",
        ess_bulk_min is not None,
        f"ess_bulk_min={ess_bulk_min}",
    )
    if ess_bulk_min is not None and num_chains is not None:
        ess_threshold_per_chain = _safe_int(diagnostics.get("ess_threshold")) or 400
        ess_required_total = ess_threshold_per_chain * num_chains
        add_check(
            "ess_within_threshold",
            "critical",
            ess_bulk_min >= ess_required_total,
            f"ess_bulk_min={ess_bulk_min:.0f}, required_total={ess_required_total}",
        )

    primary_metrics = _resolve_primary_metrics(metrics) if isinstance(metrics, dict) else {}
    primary_calibration = primary_metrics.get("calibration", {})
    primary_within = False
    if isinstance(primary_calibration, dict):
        primary_within = bool(primary_calibration.get("within_tolerance", False))
    add_check(
        "primary_calibration_within_tolerance",
        "critical",
        primary_within,
        f"within_tolerance={primary_within}",
    )

    splits_payload = metrics.get("splits", {}) if isinstance(metrics, dict) else {}
    secondary_payload = {}
    if isinstance(splits_payload, dict):
        secondary_payload = splits_payload.get(SECONDARY_SPLIT, {})
    has_secondary = isinstance(secondary_payload, dict) and bool(secondary_payload)

    if require_secondary_split:
        add_check(
            "secondary_split_evaluated",
            "critical",
            has_secondary,
            f"secondary_split_present={has_secondary}",
        )
        secondary_within = False
        if has_secondary:
            sec_cal = secondary_payload.get("calibration", {})
            if isinstance(sec_cal, dict):
                secondary_within = bool(sec_cal.get("within_tolerance", False))
        add_check(
            "secondary_calibration_within_tolerance",
            "critical",
            secondary_within if has_secondary else False,
            f"within_tolerance={secondary_within if has_secondary else 'n/a'}",
        )
    else:
        add_check(
            "secondary_split_evaluated",
            "recommended",
            has_secondary,
            "secondary split disabled for this run; enabled evaluation is recommended",
        )

    n_artifact_errors = len(artifact_errors)
    add_check(
        "publication_artifact_errors",
        "critical",
        n_artifact_errors == 0,
        f"n_errors={n_artifact_errors}",
    )

    critical_failed = [c["name"] for c in checks if c["severity"] == "critical" and not c["passed"]]
    recommended_failed = [
        c["name"] for c in checks if c["severity"] != "critical" and not c["passed"]
    ]

    return {
        "ready": len(critical_failed) == 0,
        "critical_failed": critical_failed,
        "recommended_failed": recommended_failed,
        "checks": checks,
    }


def _render_publication_readiness_markdown(payload: dict[str, Any]) -> str:
    """Render publication-readiness payload as Markdown."""

    def md_cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    ready = bool(payload.get("ready", False))
    status = "PASS" if ready else "FAIL"

    lines = [
        "# Publication Readiness",
        "",
        f"- **Status:** {status}",
        f"- **Critical failures:** {len(payload.get('critical_failed', []))}",
        f"- **Recommended failures:** {len(payload.get('recommended_failed', []))}",
        "",
        "## Checks",
        "",
        "| Check | Severity | Passed | Detail |",
        "|---|---|---|---|",
    ]

    for check in payload.get("checks", []):
        passed = "yes" if bool(check.get("passed")) else "no"
        name = md_cell(check.get("name", ""))
        severity = md_cell(check.get("severity", ""))
        detail = md_cell(check.get("detail", ""))
        lines.append(f"| {name} | {severity} | {passed} | {detail} |")

    return "\n".join(lines) + "\n"


def _get_coefficient_var_names(
    idata: az.InferenceData,
    prefix: str = "user_",
) -> list[str]:
    """Build var_names for coefficient table based on InferenceData contents.

    Returns a dynamic list including sigma_ref and n_exponent when present
    in the posterior, ensuring publication tables adapt to the model
    configuration without hardcoded lists.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    prefix : str, default "user_"
        Parameter name prefix ("user_" or "critic_").

    Returns
    -------
    list[str]
        Variable names for coefficient table.
    """
    var_names = [
        f"{prefix}beta",
        f"{prefix}mu_artist",
        f"{prefix}sigma_artist",
    ]
    # Include sigma_ref if present (sampled parameter with full diagnostics)
    if f"{prefix}sigma_ref" in idata.posterior:
        var_names.append(f"{prefix}sigma_ref")
    # Always include sigma_obs (sampled or deterministic)
    var_names.append(f"{prefix}sigma_obs")
    # Include n_exponent if learned
    if f"{prefix}n_exponent" in idata.posterior:
        var_names.append(f"{prefix}n_exponent")
    return var_names


def _resolve_model_key(model_dir: Path) -> tuple[str, str]:
    """Resolve the posterior-site prefix and manifest model key.

    The training summary's dataset block records which descriptor the model
    was fit under; legacy summaries (or a missing summary) fall back to the
    AOTY ``user`` prefix, preserving the original behavior.

    Returns:
        ``(site_prefix, model_key)`` — e.g. ``("user_", "user_score")`` for
        AOTY or ``("perf_", "perf_score")`` for the aero example.
    """
    prefix = "user"
    summary_path = model_dir / "training_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                dataset_block = json.load(f).get("dataset") or {}
            prefix = dataset_block.get("model_prefix") or "user"
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could_not_resolve_model_prefix", error=str(e))
    return f"{prefix}_", f"{prefix}_score"


def generate_publication_artifacts(ctx: StageContext) -> dict:
    """Generate publication-ready artifacts.

    Creates tables, figures, and model documentation from the fitted
    model and evaluation results.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with paths to generated artifacts.
    """
    log.info("publication_pipeline_start")

    # Set up output directories
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_dir = Path("models")
    manifest = load_manifest(model_dir)

    site_prefix, model_key = _resolve_model_key(model_dir)
    if manifest is None or model_key not in manifest.current:
        raise ValueError(f"No trained {model_key} model found")

    model_filename = manifest.current[model_key]
    model_path = model_dir / model_filename

    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    # Load evaluation results
    eval_dir = Path("outputs/evaluation")
    metrics_error: str | None = None
    try:
        with open(eval_dir / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        metrics_error = str(e)
        log.warning("could_not_load_metrics", error=metrics_error)
        metrics = {}

    diagnostics_error: str | None = None
    try:
        with open(eval_dir / "diagnostics.json", "r", encoding="utf-8") as f:
            diagnostics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        diagnostics_error = str(e)
        log.warning("could_not_load_diagnostics", error=diagnostics_error)
        diagnostics = {}

    training_summary_error: str | None = None
    try:
        with open(model_dir / "training_summary.json", "r", encoding="utf-8") as f:
            training_summary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        training_summary_error = str(e)
        log.warning("could_not_load_training_summary", error=training_summary_error)
        training_summary = {}

    artifacts = {
        "tables": [],
        "figures": [],
        "docs": [],
        "errors": [],
    }

    if metrics_error is not None:
        artifacts["errors"].append({"artifact": "metrics_input", "error": metrics_error})
    if diagnostics_error is not None:
        artifacts["errors"].append({"artifact": "diagnostics_input", "error": diagnostics_error})
    if training_summary_error is not None:
        artifacts["errors"].append(
            {"artifact": "training_summary_input", "error": training_summary_error}
        )
    primary_metrics = _resolve_primary_metrics(metrics) if isinstance(metrics, dict) else {}

    # =========================================================================
    # Generate Tables
    # =========================================================================
    # Note: Each artifact uses broad exception handling intentionally.
    # This is best-effort generation: log failures but continue to generate
    # remaining artifacts. Failures in one artifact should not block others.
    # =========================================================================

    log.info("generating_tables")

    # Coefficient table
    try:
        coef_df = create_coefficient_table(
            idata,
            var_names=_get_coefficient_var_names(idata, prefix=site_prefix),
        )
        coef_path = tables_dir / "coefficients"
        export_table(coef_df, str(coef_path), caption="Model coefficient estimates")
        artifacts["tables"].append(str(coef_path) + ".csv")
        artifacts["tables"].append(str(coef_path) + ".tex")
        log.info("coefficient_table_saved", path=str(coef_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("coefficient_table_failed")
        artifacts["errors"].append({"artifact": "coefficients_table", "error": str(e)})

    # Diagnostics table
    try:
        diag_df = create_diagnostics_table(idata)
        diag_path = tables_dir / "diagnostics"
        export_table(diag_df, str(diag_path), caption="Convergence diagnostics")
        artifacts["tables"].append(str(diag_path) + ".csv")
        artifacts["tables"].append(str(diag_path) + ".tex")
        log.info("diagnostics_table_saved", path=str(diag_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("diagnostics_table_failed")
        artifacts["errors"].append({"artifact": "diagnostics_table", "error": str(e)})

    # Metrics summary table
    try:
        point_metrics = _parse_point_metrics(primary_metrics)

        calibration_rows = []
        sharpness_rows = []
        coverage_results = _parse_coverage_results(primary_metrics) or {}
        for prob, entry in sorted(coverage_results.items()):
            nominal_pct = prob * 100.0
            calibration_rows.append(
                {
                    "Metric": f"Coverage ({nominal_pct:.0f}%)",
                    "Value": entry.empirical,
                }
            )
            if entry.interval_width is not None:
                sharpness_rows.append(
                    {
                        "Metric": f"Sharpness ({nominal_pct:.0f}% CI width)",
                        "Value": entry.interval_width,
                    }
                )

        # WIS from calibration block
        calibration_block = primary_metrics.get("calibration", {})
        wis_value = (
            _safe_float(calibration_block.get("wis"))
            if isinstance(calibration_block, dict)
            else None
        )
        wis_rows = []
        if wis_value is not None:
            wis_rows.append({"Metric": "WIS", "Value": wis_value})

        metric_rows: list[dict[str, Any]] = []
        if point_metrics is not None:
            metric_rows.extend(
                [
                    {"Metric": "RMSE", "Value": point_metrics.rmse},
                    {"Metric": "MAE", "Value": point_metrics.mae},
                    {"Metric": "R-squared", "Value": point_metrics.r2},
                ]
            )
        else:
            log.warning(
                "metrics_table_point_metrics_unavailable",
                message="Skipping RMSE/MAE/R-squared rows due to missing metrics payload.",
            )

        metrics_df = pd.DataFrame(
            [
                *metric_rows,
                *calibration_rows,
                *sharpness_rows,
                *wis_rows,
            ]
        )
        metrics_path = tables_dir / "metrics_summary"
        export_table(metrics_df, str(metrics_path), caption="Model performance metrics")
        artifacts["tables"].append(str(metrics_path) + ".csv")
        artifacts["tables"].append(str(metrics_path) + ".tex")
        log.info("metrics_table_saved", path=str(metrics_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("metrics_table_failed")
        artifacts["errors"].append({"artifact": "metrics_summary_table", "error": str(e)})

    # Prediction summary table
    try:
        known_csv = Path("outputs/predictions/next_event_known_entities.csv")
        if known_csv.exists():
            known_df = pd.read_csv(known_csv)
            # Create scenario comparison table
            scenario_stats = (
                known_df.groupby("scenario")
                .agg(
                    mean_pred=("pred_mean", "mean"),
                    std_pred=("pred_mean", "std"),
                    n_artists=("entity", "nunique"),
                )
                .round(2)
            )

            pred_table_path = tables_dir / "prediction_scenarios"
            export_table(
                scenario_stats,
                str(pred_table_path),
                caption="Next-album prediction scenarios (known artists)",
            )
            artifacts["tables"].append(str(pred_table_path) + ".csv")
            artifacts["tables"].append(str(pred_table_path) + ".tex")
            log.info("prediction_table_saved", path=str(pred_table_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("prediction_table_failed")
        artifacts["errors"].append({"artifact": "prediction_scenarios_table", "error": str(e)})

    # =========================================================================
    # Generate Figures
    # =========================================================================

    log.info("generating_figures")

    # Trace plots
    try:
        pdf_path, png_path = save_trace_plot(
            idata,
            var_names=get_trace_plot_vars(idata, prefix=site_prefix),
            output_dir=figures_dir,
            filename_base="trace_plot",
        )
        artifacts["figures"].append(str(pdf_path))
        artifacts["figures"].append(str(png_path))
        log.info("trace_plot_saved", pdf=str(pdf_path), png=str(png_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("trace_plot_failed")
        artifacts["errors"].append({"artifact": "trace_plot", "error": str(e)})

    # Posterior plots
    try:
        pdf_path, png_path = save_posterior_plot(
            idata,
            var_names=get_trace_plot_vars(idata, prefix=site_prefix),
            output_dir=figures_dir,
            filename_base="posterior_plot",
        )
        artifacts["figures"].append(str(pdf_path))
        artifacts["figures"].append(str(png_path))
        log.info("posterior_plot_saved", pdf=str(pdf_path), png=str(png_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("posterior_plot_failed")
        artifacts["errors"].append({"artifact": "posterior_plot", "error": str(e)})

    # PPC density plot
    try:
        ppc_data = primary_metrics.get("ppc")
        if isinstance(ppc_data, dict) and "summary" in ppc_data:
            from panelcast.evaluation.ppc import PPCResult, PPCStatistic

            ppc_stats = []
            for stat_name, stat_info in ppc_data["summary"].items():
                if isinstance(stat_info, dict):
                    observed = _safe_float(stat_info.get("observed"))
                    p_value = _safe_float(stat_info.get("p_value"))
                    mc_se = _safe_float(stat_info.get("mc_se"))
                    if observed is None or p_value is None:
                        log.warning(
                            "ppc_stat_missing_fields",
                            statistic=stat_name,
                            observed_present=observed is not None,
                            p_value_present=p_value is not None,
                        )
                        continue
                    if mc_se is None:
                        n_samples_for_se = ppc_data.get("n_samples", 0)
                        try:
                            n_samples_for_se = int(n_samples_for_se)
                        except (TypeError, ValueError):
                            n_samples_for_se = 0
                        if n_samples_for_se > 0:
                            mc_se = float(np.sqrt(p_value * (1 - p_value) / n_samples_for_se))
                        else:
                            mc_se = 0.0

                    ppc_stats.append(
                        PPCStatistic(
                            name=stat_name,
                            observed=observed,
                            replicated_distribution=np.array([]),
                            bayesian_p_value=p_value,
                            mc_se=mc_se,
                        )
                    )

            if ppc_stats:
                n_obs_ppc = ppc_data.get("n_obs", 0)
                n_samples_ppc = ppc_data.get("n_samples", 0)
                try:
                    n_obs_ppc = int(n_obs_ppc)
                except (TypeError, ValueError):
                    n_obs_ppc = 0
                try:
                    n_samples_ppc = int(n_samples_ppc)
                except (TypeError, ValueError):
                    n_samples_ppc = 0
                ppc_result_obj = PPCResult(
                    statistics=ppc_stats,
                    n_obs=n_obs_ppc,
                    n_samples=n_samples_ppc,
                )
                # Only generate plot if we have replicated distributions
                has_distributions = any(len(s.replicated_distribution) > 0 for s in ppc_stats)
                if has_distributions:
                    pdf_path, png_path = save_ppc_density_plot(
                        ppc_result_obj,
                        output_dir=figures_dir,
                        filename_base="ppc_density",
                    )
                    artifacts["figures"].append(str(pdf_path))
                    artifacts["figures"].append(str(png_path))
                    log.info("ppc_density_plot_saved", pdf=str(pdf_path), png=str(png_path))
                else:
                    log.info(
                        "ppc_density_plot_skipped",
                        reason="no replicated distributions in artifact",
                    )
    except Exception as e:
        log.exception("ppc_density_plot_failed")
        artifacts["errors"].append({"artifact": "ppc_density_plot", "error": str(e)})

    # =========================================================================
    # Predictions scatter plot
    # =========================================================================
    try:
        primary_split_name = metrics.get(
            "primary_split", str(SplitType.WITHIN_ENTITY_TEMPORAL.value)
        )
        pred_candidates = [
            resolve_split_dir(eval_dir, primary_split_name) / "predictions.json",
            eval_dir / "predictions.json",
        ]
        pred_path = next((p for p in pred_candidates if p.exists()), None)
        if pred_path is not None:
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
            interval_level = pred_data.get("interval_level", 0.90)
            pdf_path, png_path = save_predictions_plot(
                y_true=np.array(pred_data["y_true"]),
                y_pred_mean=np.array(pred_data["y_pred_mean"]),
                y_pred_lower=np.array(pred_data["y_pred_lower"]),
                y_pred_upper=np.array(pred_data["y_pred_upper"]),
                output_dir=figures_dir,
                filename_base="predictions_primary",
                ci_label=f"{interval_level * 100:.0f}% CI",
            )
            artifacts["figures"].append(str(pdf_path))
            artifacts["figures"].append(str(png_path))
            log.info("predictions_plot_saved", pdf=str(pdf_path), png=str(png_path))
        else:
            log.warning("predictions_plot_skipped", reason="predictions.json not found")
    except Exception as e:
        log.exception("predictions_plot_failed")
        artifacts["errors"].append({"artifact": "predictions_plot", "error": str(e)})

    # =========================================================================
    # Reliability diagram (calibration plot)
    # =========================================================================
    try:
        cal_candidates = [
            resolve_split_dir(eval_dir, primary_split_name) / "calibration.json",
            eval_dir / "calibration.json",
        ]
        cal_path = next((p for p in cal_candidates if p.exists()), None)
        if cal_path is not None:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
            probs = np.array(cal_data["predicted_probs"])
            # Use stored bin_edges if available, otherwise reconstruct from probs
            if "bin_edges" in cal_data:
                bin_edges = np.array(cal_data["bin_edges"])
            else:
                bin_edges = np.linspace(0.0, 1.0, len(probs) + 1)
            reliability = ReliabilityData(
                bin_edges=bin_edges,
                predicted_probs=probs,
                observed_freq=np.array(cal_data["observed_freq"]),
                counts=np.array(cal_data["counts"]),
            )
            pdf_path, png_path = save_reliability_plot(
                reliability, figures_dir, "reliability_primary"
            )
            artifacts["figures"].append(str(pdf_path))
            artifacts["figures"].append(str(png_path))
            log.info("reliability_plot_saved", pdf=str(pdf_path), png=str(png_path))
        else:
            log.warning("reliability_plot_skipped", reason="calibration.json not found")
    except Exception as e:
        log.exception("reliability_plot_failed")
        artifacts["errors"].append({"artifact": "reliability_plot", "error": str(e)})

    # =========================================================================
    # Per-artist fan charts
    # =========================================================================
    try:
        known_csv = Path("outputs/predictions/next_event_known_entities.csv")
        pred_path_for_fans = next(
            (
                p
                for p in [
                    resolve_split_dir(eval_dir, primary_split_name) / "predictions.json",
                    eval_dir / "predictions.json",
                ]
                if p.exists()
            ),
            None,
        )
        if known_csv.exists() and pred_path_for_fans is not None:
            known_df_fans = pd.read_csv(known_csv)
            subsets = select_artist_subsets(known_df_fans)
            # Build set of unique artists across all categories
            all_selected: dict[str, list[str]] = {}
            for category, artists in subsets.items():
                for a in artists:
                    all_selected.setdefault(a, []).append(category)

            # Load training data for actual scores
            train_path = Path("data/splits") / primary_split_name / "train.parquet"
            if train_path.exists():
                from panelcast.config.descriptor import DatasetDescriptor

                fan_descriptor = getattr(ctx, "descriptor", None) or DatasetDescriptor()
                train_for_fans = pd.read_parquet(train_path)
                sort_cols_fans = [fan_descriptor.entity_col]
                if fan_descriptor.parsed_date_col in train_for_fans.columns:
                    sort_cols_fans.append(fan_descriptor.parsed_date_col)
                train_for_fans = train_for_fans.sort_values(sort_cols_fans)

                for artist, categories in all_selected.items():
                    artist_train = train_for_fans[
                        train_for_fans[fan_descriptor.entity_col] == artist
                    ]
                    if len(artist_train) < 2:
                        continue
                    actual = artist_train[fan_descriptor.target_col].values
                    has_album_col = fan_descriptor.event_col in artist_train.columns
                    albums = (
                        artist_train[fan_descriptor.event_col].tolist() if has_album_col else None
                    )
                    # Use known predictions for fan chart samples (quantiles)
                    same_preds = known_df_fans[
                        (known_df_fans["entity"] == artist) & (known_df_fans["scenario"] == "same")
                    ]
                    if same_preds.empty:
                        continue
                    # Build pseudo-samples from quantiles for the last point
                    q_cols = ["pred_q05", "pred_q25", "pred_q50", "pred_q75", "pred_q95"]
                    if not all(c in same_preds.columns for c in q_cols):
                        continue
                    pred_quantiles = same_preds[q_cols].values[0]
                    # Build a fan: the observed trajectory (no predictive spread)
                    # plus one appended forecast point carrying the quantile fan.
                    pred_for_fan = np.tile(actual, (5, 1))
                    pred_for_fan = np.column_stack([pred_for_fan, pred_quantiles[:, None]])
                    # The appended column adds a forecast time point, so extend the
                    # observed series and labels to match pred_for_fan's width: the
                    # forecast slot has no observed value (NaN -> rendered as a gap).
                    actual_for_fan = np.append(np.asarray(actual, dtype=float), np.nan)
                    albums_for_fan = (list(albums) + ["next"]) if albums is not None else None

                    safe_name = artist.replace("/", "_").replace(" ", "_")[:50]
                    try:
                        pdf_path, png_path = save_artist_prediction_plot(
                            artist=artist,
                            actual_scores=actual_for_fan,
                            pred_samples=pred_for_fan,
                            album_labels=albums_for_fan,
                            output_dir=figures_dir,
                            filename_base=f"artist_{safe_name}",
                            categories=categories,
                        )
                        artifacts["figures"].append(str(pdf_path))
                        artifacts["figures"].append(str(png_path))
                    except Exception as e:
                        log.warning("artist_fan_chart_failed", artist=artist, error=str(e))

                log.info("artist_fan_charts_complete", n_artists=len(all_selected))
            else:
                log.warning("artist_fan_charts_skipped", reason="train.parquet not found")
        else:
            log.warning("artist_fan_charts_skipped", reason="required artifacts not found")
    except Exception as e:
        log.exception("artist_fan_charts_failed")
        artifacts["errors"].append({"artifact": "artist_fan_charts", "error": str(e)})

    # =========================================================================
    # Generate Model Card
    # =========================================================================

    log.info("generating_model_card")

    try:
        # Create model card data (prose templated from the dataset descriptor)
        model_card_data = create_default_model_card_data(getattr(ctx, "descriptor", None))

        # Populate run-specific dataset and hyperparameter metadata when available.
        n_obs = training_summary.get("n_observations")
        if isinstance(n_obs, int) and n_obs > 0:
            model_card_data.dataset_size = n_obs
        n_features = training_summary.get("n_features")
        if isinstance(n_features, int):
            model_card_data.hyperparameters["n_features"] = n_features
        n_artists = training_summary.get("n_artists")
        if isinstance(n_artists, int):
            model_card_data.hyperparameters["n_artists"] = n_artists
        max_albums = training_summary.get("max_albums")
        if isinstance(max_albums, int):
            model_card_data.hyperparameters["max_albums"] = max_albums
        mcmc_cfg = training_summary.get("mcmc_config", {})
        if isinstance(mcmc_cfg, dict):
            for key in (
                "num_chains",
                "num_warmup",
                "num_samples",
                "chain_method",
                "target_accept_prob",
                "max_tree_depth",
            ):
                if key in mcmc_cfg:
                    model_card_data.hyperparameters[key] = mcmc_cfg[key]

        primary_metrics = _resolve_primary_metrics(metrics)
        convergence = _parse_convergence(diagnostics, metrics)
        coverage_results = _parse_coverage_results(primary_metrics)
        point_metrics = _parse_point_metrics(primary_metrics)
        loo_result = _parse_loo_result(primary_metrics)

        # Extract PPC summary for model card
        ppc_summary = None
        ppc_data = primary_metrics.get("ppc")
        if isinstance(ppc_data, dict):
            ppc_summary = ppc_data.get("summary")

        # Generate prior justification text
        prior_justification = None
        try:
            from panelcast.evaluation.prior_predictive import (
                PriorPredictiveResult,
                generate_prior_justification_text,
            )
            from panelcast.models.bayes.priors import PriorConfig

            priors_dict = training_summary.get("priors")
            if isinstance(priors_dict, dict):
                priors = PriorConfig(**priors_dict)

                # Load prior predictive result if available
                pp_result = None
                pp_path = eval_dir / "prior_predictive.json"
                if pp_path.exists():
                    try:
                        with open(pp_path, "r", encoding="utf-8") as f:
                            pp_data = json.load(f)
                        pp_result = PriorPredictiveResult(
                            y_samples=np.array([]),
                            summary=pp_data.get("summary", {}),
                            reasonable=pp_data.get("reasonable", False),
                            bounds=tuple(pp_data.get("bounds", (0, 100))),
                            fraction_in_bounds=pp_data.get("fraction_in_bounds", 0.0),
                            n_samples=pp_data.get("n_samples", 0),
                            n_obs_original=pp_data.get("n_obs_original", 0),
                            max_obs=pp_data.get("max_obs", 2000),
                            seed=pp_data.get("seed", 42),
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        log.warning("prior_predictive_load_failed", error=str(e))

                # Load OAT sensitivity summary if available
                sensitivity_summary = None
                oat_path = Path("outputs/sensitivity/oat_summary.csv")
                if oat_path.exists():
                    try:
                        sensitivity_summary = pd.read_csv(oat_path)
                    except Exception as e:
                        log.warning("oat_summary_load_failed", error=str(e))

                prior_justification = generate_prior_justification_text(
                    priors,
                    prior_predictive_result=pp_result,
                    sensitivity_summary=sensitivity_summary,
                )
        except Exception as e:
            log.warning("prior_justification_generation_failed", error=str(e))

        model_card_data = update_model_card_with_results(
            model_card_data,
            idata=idata,
            convergence=convergence,
            coverage_results=coverage_results,
            loo_result=loo_result,
            point_metrics=point_metrics,
            ppc_summary=ppc_summary,
            prior_justification=prior_justification,
        )

        # Write model card
        model_card_path = reports_dir / "MODEL_CARD.md"
        write_model_card(model_card_data, model_card_path)
        artifacts["docs"].append(str(model_card_path))

        # Also copy to project root
        root_card_path = Path("MODEL_CARD.md")
        shutil.copy(model_card_path, root_card_path)
        artifacts["docs"].append(str(root_card_path))

        log.info("model_card_saved", path=str(model_card_path))
    # Individual artifact failures are collected and reported at pipeline end.
    except Exception as e:
        log.exception("model_card_failed")
        artifacts["errors"].append({"artifact": "model_card", "error": str(e)})

    readiness_payload = _build_publication_readiness(
        metrics=metrics if isinstance(metrics, dict) else {},
        diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
        training_summary=training_summary if isinstance(training_summary, dict) else {},
        artifact_errors=artifacts["errors"],
        require_secondary_split=bool(getattr(ctx, "evaluate_secondary_split", True)),
    )
    readiness_json_path = reports_dir / "publication_readiness.json"
    with open(readiness_json_path, "w", encoding="utf-8") as f:
        json.dump(readiness_payload, f, indent=2)
    readiness_md_path = reports_dir / "PUBLICATION_READINESS.md"
    readiness_md_path.write_text(
        _render_publication_readiness_markdown(readiness_payload),
        encoding="utf-8",
    )
    artifacts["docs"].append(str(readiness_json_path))
    artifacts["docs"].append(str(readiness_md_path))
    artifacts["readiness"] = str(readiness_json_path)

    status_payload = {
        "n_tables": len(artifacts["tables"]),
        "n_figures": len(artifacts["figures"]),
        "n_docs": len(artifacts["docs"]),
        "n_errors": len(artifacts["errors"]),
        "errors": artifacts["errors"],
        "publication_ready": readiness_payload["ready"],
        "critical_readiness_failures": readiness_payload["critical_failed"],
        "recommended_readiness_failures": readiness_payload["recommended_failed"],
    }
    status_path = reports_dir / "artifact_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)
    artifacts["status"] = str(status_path)

    if artifacts["errors"]:
        log.warning("publication_artifact_failures", n_errors=len(artifacts["errors"]))
        if ctx.strict:
            raise ValueError(
                "Publication artifact generation failed for one or more required artifacts. "
                f"See {status_path} for details."
            )
    if not readiness_payload["ready"]:
        log.warning(
            "publication_readiness_failed",
            critical_failed=readiness_payload["critical_failed"],
        )
        if ctx.strict:
            raise ValueError(
                f"Publication readiness checks failed. See {readiness_json_path} for details."
            )

    # =========================================================================
    # Copy artifacts to run directory if available
    # =========================================================================

    if ctx.run_dir and ctx.run_dir.exists():
        run_reports_dir = ctx.run_dir / "reports"
        run_reports_dir.mkdir(parents=True, exist_ok=True)

        # Copy figures
        for fig_path in artifacts["figures"]:
            if Path(fig_path).exists():
                dest = run_reports_dir / "figures" / Path(fig_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(fig_path, dest)

        # Copy tables
        for table_path in artifacts["tables"]:
            if Path(table_path).exists():
                dest = run_reports_dir / "tables" / Path(table_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(table_path, dest)

        if status_path.exists():
            shutil.copy(status_path, run_reports_dir / status_path.name)
        if readiness_json_path.exists():
            shutil.copy(readiness_json_path, run_reports_dir / readiness_json_path.name)
        if readiness_md_path.exists():
            shutil.copy(readiness_md_path, run_reports_dir / readiness_md_path.name)

        log.info("artifacts_copied_to_run_dir", run_dir=str(ctx.run_dir))

    log.info(
        "publication_pipeline_complete",
        n_tables=len(artifacts["tables"]),
        n_figures=len(artifacts["figures"]),
        n_docs=len(artifacts["docs"]),
        publication_ready=readiness_payload["ready"],
    )

    return artifacts
