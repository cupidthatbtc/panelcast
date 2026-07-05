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
from panelcast.paths import ArtifactPaths
from panelcast.reporting.figures import (
    get_trace_plot_vars,
    save_artist_prediction_plot,
    save_posterior_plot,
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
    """Parse held-out ELPD from the evaluation payload (legacy `loo` fallback)."""
    info = primary_metrics.get("info_criteria", {})
    if isinstance(info, dict):
        for key in ("heldout_elpd", "loo"):
            payload = info.get(key, {})
            if isinstance(payload, dict):
                elpd = _safe_float(payload.get("elpd"))
                se = _safe_float(payload.get("se"))
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
    if ess_bulk_min is not None:
        # ess_threshold is a TOTAL bulk-ESS floor (the evaluate stage checks
        # ess_bulk_min >= ess_threshold directly); it is not per-chain, so don't
        # multiply by num_chains.
        ess_threshold = _safe_int(diagnostics.get("ess_threshold")) or 400
        add_check(
            "ess_within_threshold",
            "critical",
            ess_bulk_min >= ess_threshold,
            f"ess_bulk_min={ess_bulk_min:.0f}, threshold={ess_threshold}",
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
            with open(summary_path, encoding="utf-8") as f:
                dataset_block = json.load(f).get("dataset") or {}
            prefix = dataset_block.get("model_prefix") or "user"
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could_not_resolve_model_prefix", error=str(e))
    return f"{prefix}_", f"{prefix}_score"


@dataclass
class _PublicationInputs:
    """Loaded-once inputs shared by every publication artifact helper."""

    ctx: StageContext
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path
    eval_dir: Path
    idata: az.InferenceData
    site_prefix: str
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    training_summary: dict[str, Any]
    primary_metrics: dict[str, Any]
    primary_split_name: str
    point_metrics: _PointMetricsLike | None
    known_csv: Path
    input_errors: list[dict[str, str]]


def _load_publication_inputs(ctx: StageContext) -> _PublicationInputs:
    """Set up the output dirs and load the model + evaluation artifacts once.

    Hoists primary_split_name and the deduplicated primary_metrics / point_metrics
    so the per-artifact helpers don't recompute them. Raises only when no trained
    model exists (the pipeline's one hard failure); a missing evaluation JSON is
    recorded as a soft input error instead.
    """
    log.info("publication_pipeline_start")

    # Roots come from ctx.paths but are rebuilt through the module-local Path
    # so test patches keep applying.
    paths = ArtifactPaths.from_ctx(ctx)
    reports_dir = Path(paths.reports)
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(paths.models)
    manifest = load_manifest(model_dir)
    site_prefix, model_key = _resolve_model_key(model_dir)
    if manifest is None or model_key not in manifest.current:
        raise ValueError(f"No trained {model_key} model found")
    model_path = model_dir / manifest.current[model_key]
    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    eval_dir = Path(paths.evaluation)
    input_errors: list[dict[str, str]] = []

    def _load_json(path: Path, log_event: str, artifact: str) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.warning(log_event, error=str(e))
            input_errors.append({"artifact": artifact, "error": str(e)})
            return {}

    metrics = _load_json(eval_dir / "metrics.json", "could_not_load_metrics", "metrics_input")
    diagnostics = _load_json(
        eval_dir / "diagnostics.json", "could_not_load_diagnostics", "diagnostics_input"
    )
    training_summary = _load_json(
        model_dir / "training_summary.json",
        "could_not_load_training_summary",
        "training_summary_input",
    )

    primary_metrics = _resolve_primary_metrics(metrics) if isinstance(metrics, dict) else {}
    primary_split_name = (
        metrics.get("primary_split", str(SplitType.WITHIN_ENTITY_TEMPORAL.value))
        if isinstance(metrics, dict)
        else str(SplitType.WITHIN_ENTITY_TEMPORAL.value)
    )
    return _PublicationInputs(
        ctx=ctx,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        eval_dir=eval_dir,
        idata=idata,
        site_prefix=site_prefix,
        metrics=metrics,
        diagnostics=diagnostics,
        training_summary=training_summary,
        primary_metrics=primary_metrics,
        primary_split_name=primary_split_name,
        point_metrics=_parse_point_metrics(primary_metrics),
        known_csv=Path(paths.predictions) / "next_event_known_entities.csv",
        input_errors=input_errors,
    )


def _build_coefficient_table(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        coef_df = create_coefficient_table(
            inp.idata,
            var_names=_get_coefficient_var_names(inp.idata, prefix=inp.site_prefix),
        )
        coef_path = inp.tables_dir / "coefficients"
        export_table(coef_df, str(coef_path), caption="Model coefficient estimates")
        artifacts["tables"].append(str(coef_path) + ".csv")
        artifacts["tables"].append(str(coef_path) + ".tex")
        log.info("coefficient_table_saved", path=str(coef_path))
    except Exception as e:
        log.exception("coefficient_table_failed")
        artifacts["errors"].append({"artifact": "coefficients_table", "error": str(e)})


def _build_diagnostics_table(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        diag_df = create_diagnostics_table(inp.idata)
        diag_path = inp.tables_dir / "diagnostics"
        export_table(diag_df, str(diag_path), caption="Convergence diagnostics")
        artifacts["tables"].append(str(diag_path) + ".csv")
        artifacts["tables"].append(str(diag_path) + ".tex")
        log.info("diagnostics_table_saved", path=str(diag_path))
    except Exception as e:
        log.exception("diagnostics_table_failed")
        artifacts["errors"].append({"artifact": "diagnostics_table", "error": str(e)})


def _build_metrics_summary_table(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        point_metrics = inp.point_metrics

        calibration_rows = []
        sharpness_rows = []
        coverage_results = _parse_coverage_results(inp.primary_metrics) or {}
        for prob, entry in sorted(coverage_results.items()):
            nominal_pct = prob * 100.0
            calibration_rows.append(
                {"Metric": f"Coverage ({nominal_pct:.0f}%)", "Value": entry.empirical}
            )
            if entry.interval_width is not None:
                sharpness_rows.append(
                    {
                        "Metric": f"Sharpness ({nominal_pct:.0f}% CI width)",
                        "Value": entry.interval_width,
                    }
                )

        calibration_block = inp.primary_metrics.get("calibration", {})
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

        metrics_df = pd.DataFrame([*metric_rows, *calibration_rows, *sharpness_rows, *wis_rows])
        metrics_path = inp.tables_dir / "metrics_summary"
        export_table(metrics_df, str(metrics_path), caption="Model performance metrics")
        artifacts["tables"].append(str(metrics_path) + ".csv")
        artifacts["tables"].append(str(metrics_path) + ".tex")
        log.info("metrics_table_saved", path=str(metrics_path))
    except Exception as e:
        log.exception("metrics_table_failed")
        artifacts["errors"].append({"artifact": "metrics_summary_table", "error": str(e)})


def _build_prediction_scenarios_table(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        if inp.known_csv.exists():
            known_df = pd.read_csv(inp.known_csv)
            scenario_stats = (
                known_df.groupby("scenario")
                .agg(
                    mean_pred=("pred_mean", "mean"),
                    std_pred=("pred_mean", "std"),
                    n_artists=("entity", "nunique"),
                )
                .round(2)
            )
            pred_table_path = inp.tables_dir / "prediction_scenarios"
            export_table(
                scenario_stats,
                str(pred_table_path),
                caption="Next-album prediction scenarios (known artists)",
            )
            artifacts["tables"].append(str(pred_table_path) + ".csv")
            artifacts["tables"].append(str(pred_table_path) + ".tex")
            log.info("prediction_table_saved", path=str(pred_table_path))
    except Exception as e:
        log.exception("prediction_table_failed")
        artifacts["errors"].append({"artifact": "prediction_scenarios_table", "error": str(e)})


def _save_trace_plot(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        pdf_path, png_path = save_trace_plot(
            inp.idata,
            var_names=get_trace_plot_vars(inp.idata, prefix=inp.site_prefix),
            output_dir=inp.figures_dir,
            filename_base="trace_plot",
        )
        artifacts["figures"].append(str(pdf_path))
        artifacts["figures"].append(str(png_path))
        log.info("trace_plot_saved", pdf=str(pdf_path), png=str(png_path))
    except Exception as e:
        log.exception("trace_plot_failed")
        artifacts["errors"].append({"artifact": "trace_plot", "error": str(e)})


def _save_posterior_plot(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        pdf_path, png_path = save_posterior_plot(
            inp.idata,
            var_names=get_trace_plot_vars(inp.idata, prefix=inp.site_prefix),
            output_dir=inp.figures_dir,
            filename_base="posterior_plot",
        )
        artifacts["figures"].append(str(pdf_path))
        artifacts["figures"].append(str(png_path))
        log.info("posterior_plot_saved", pdf=str(pdf_path), png=str(png_path))
    except Exception as e:
        log.exception("posterior_plot_failed")
        artifacts["errors"].append({"artifact": "posterior_plot", "error": str(e)})


def _save_predictions_plot(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        pred_candidates = [
            resolve_split_dir(inp.eval_dir, inp.primary_split_name) / "predictions.json",
            inp.eval_dir / "predictions.json",
        ]
        pred_path = next((p for p in pred_candidates if p.exists()), None)
        if pred_path is not None:
            with open(pred_path, encoding="utf-8") as f:
                pred_data = json.load(f)
            interval_level = pred_data.get("interval_level", 0.90)
            pdf_path, png_path = save_predictions_plot(
                y_true=np.array(pred_data["y_true"]),
                y_pred_mean=np.array(pred_data["y_pred_mean"]),
                y_pred_lower=np.array(pred_data["y_pred_lower"]),
                y_pred_upper=np.array(pred_data["y_pred_upper"]),
                output_dir=inp.figures_dir,
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


def _save_reliability_plot(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        cal_candidates = [
            resolve_split_dir(inp.eval_dir, inp.primary_split_name) / "calibration.json",
            inp.eval_dir / "calibration.json",
        ]
        cal_path = next((p for p in cal_candidates if p.exists()), None)
        if cal_path is not None:
            with open(cal_path, encoding="utf-8") as f:
                cal_data = json.load(f)
            probs = np.array(cal_data["predicted_probs"])
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
                reliability, inp.figures_dir, "reliability_primary"
            )
            artifacts["figures"].append(str(pdf_path))
            artifacts["figures"].append(str(png_path))
            log.info("reliability_plot_saved", pdf=str(pdf_path), png=str(png_path))
        else:
            log.warning("reliability_plot_skipped", reason="calibration.json not found")
    except Exception as e:
        log.exception("reliability_plot_failed")
        artifacts["errors"].append({"artifact": "reliability_plot", "error": str(e)})


def _save_artist_fan_charts(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    try:
        pred_path_for_fans = next(
            (
                p
                for p in [
                    resolve_split_dir(inp.eval_dir, inp.primary_split_name) / "predictions.json",
                    inp.eval_dir / "predictions.json",
                ]
                if p.exists()
            ),
            None,
        )
        if not (inp.known_csv.exists() and pred_path_for_fans is not None):
            log.warning("artist_fan_charts_skipped", reason="required artifacts not found")
            return

        known_df_fans = pd.read_csv(inp.known_csv)
        subsets = select_artist_subsets(known_df_fans)
        all_selected: dict[str, list[str]] = {}
        for category, artists in subsets.items():
            for a in artists:
                all_selected.setdefault(a, []).append(category)

        train_path = Path("data/splits") / inp.primary_split_name / "train.parquet"
        if not train_path.exists():
            log.warning("artist_fan_charts_skipped", reason="train.parquet not found")
            return

        from panelcast.config.descriptor import DatasetDescriptor

        fan_descriptor = getattr(inp.ctx, "descriptor", None) or DatasetDescriptor()
        train_for_fans = pd.read_parquet(train_path)
        sort_cols_fans = [fan_descriptor.entity_col]
        if fan_descriptor.parsed_date_col in train_for_fans.columns:
            sort_cols_fans.append(fan_descriptor.parsed_date_col)
        train_for_fans = train_for_fans.sort_values(sort_cols_fans)

        def _fan_chart_for_artist(artist: str, categories: list[str]) -> None:
            artist_train = train_for_fans[train_for_fans[fan_descriptor.entity_col] == artist]
            if len(artist_train) < 2:
                return
            actual = artist_train[fan_descriptor.target_col].values
            has_album_col = fan_descriptor.event_col in artist_train.columns
            albums = artist_train[fan_descriptor.event_col].tolist() if has_album_col else None
            # Use known predictions for fan chart samples (quantiles)
            same_preds = known_df_fans[
                (known_df_fans["entity"] == artist) & (known_df_fans["scenario"] == "same")
            ]
            if same_preds.empty:
                return
            q_cols = ["pred_q05", "pred_q25", "pred_q50", "pred_q75", "pred_q95"]
            if not all(c in same_preds.columns for c in q_cols):
                return
            pred_quantiles = same_preds[q_cols].values[0]
            # Build a fan: the observed trajectory (no predictive spread) plus one
            # appended forecast point carrying the quantile fan. The appended slot
            # has no observed value (NaN -> rendered as a gap).
            pred_for_fan = np.tile(actual, (5, 1))
            pred_for_fan = np.column_stack([pred_for_fan, pred_quantiles[:, None]])
            actual_for_fan = np.append(np.asarray(actual, dtype=float), np.nan)
            albums_for_fan = (list(albums) + ["next"]) if albums is not None else None

            safe_name = artist.replace("/", "_").replace(" ", "_")[:50]
            try:
                pdf_path, png_path = save_artist_prediction_plot(
                    artist=artist,
                    actual_scores=actual_for_fan,
                    pred_samples=pred_for_fan,
                    album_labels=albums_for_fan,
                    output_dir=inp.figures_dir,
                    filename_base=f"artist_{safe_name}",
                    categories=categories,
                )
                artifacts["figures"].append(str(pdf_path))
                artifacts["figures"].append(str(png_path))
            except Exception as e:
                # str(e) alone can be empty (e.g. MemoryError); record the type
                # and traceback so the failure is diagnosable.
                log.warning(
                    "artist_fan_chart_failed",
                    artist=artist,
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                # Record per-artist failures so the readiness gate sees them; a
                # run where every chart failed must not report ready.
                artifacts["errors"].append(
                    {"artifact": f"artist_fan_chart:{artist}", "error": str(e)}
                )

        for artist, categories in all_selected.items():
            _fan_chart_for_artist(artist, categories)
        log.info("artist_fan_charts_complete", n_artists=len(all_selected))
    except Exception as e:
        log.exception("artist_fan_charts_failed")
        artifacts["errors"].append({"artifact": "artist_fan_charts", "error": str(e)})


def _build_prior_justification(inp: _PublicationInputs) -> str | None:
    """Best-effort prior-justification prose for the model card.

    Returns None (logged) on any failure so the model card still writes.
    """
    try:
        from panelcast.evaluation.prior_predictive import (
            PriorPredictiveResult,
            generate_prior_justification_text,
        )
        from panelcast.models.bayes.priors import PriorConfig

        priors_dict = inp.training_summary.get("priors")
        if not isinstance(priors_dict, dict):
            return None
        priors = PriorConfig(**priors_dict)

        pp_result = None
        pp_path = inp.eval_dir / "prior_predictive.json"
        if pp_path.exists():
            try:
                with open(pp_path, encoding="utf-8") as f:
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

        sensitivity_summary = None
        oat_path = Path("outputs/sensitivity/oat_summary.csv")
        if oat_path.exists():
            try:
                sensitivity_summary = pd.read_csv(oat_path)
            except Exception as e:
                log.warning(
                    "oat_summary_load_failed",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )

        return generate_prior_justification_text(
            priors,
            prior_predictive_result=pp_result,
            sensitivity_summary=sensitivity_summary,
        )
    except Exception as e:
        log.warning("prior_justification_generation_failed", error=str(e))
        return None


def _generate_model_card(inp: _PublicationInputs, artifacts: dict[str, Any]) -> None:
    log.info("generating_model_card")
    try:
        model_card_data = create_default_model_card_data(getattr(inp.ctx, "descriptor", None))

        training_summary = inp.training_summary
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

        convergence = _parse_convergence(inp.diagnostics, inp.metrics)
        coverage_results = _parse_coverage_results(inp.primary_metrics)
        loo_result = _parse_loo_result(inp.primary_metrics)

        ppc_summary = None
        ppc_data = inp.primary_metrics.get("ppc")
        if isinstance(ppc_data, dict):
            ppc_summary = ppc_data.get("summary")

        prior_justification = _build_prior_justification(inp)

        model_card_data = update_model_card_with_results(
            model_card_data,
            idata=inp.idata,
            convergence=convergence,
            coverage_results=coverage_results,
            loo_result=loo_result,
            point_metrics=inp.point_metrics,
            ppc_summary=ppc_summary,
            prior_justification=prior_justification,
        )

        model_card_path = inp.reports_dir / "MODEL_CARD.md"
        write_model_card(model_card_data, model_card_path)
        artifacts["docs"].append(str(model_card_path))

        # The run's card stays run-scoped (#81). It must NOT be copied to the
        # repo-root MODEL_CARD.md — that is the curated, hand-maintained card, and
        # overwriting it on every run silently clobbers it (a #118-class leak the
        # artifact guard misses because the file lives outside the guarded dirs).
        log.info("model_card_saved", path=str(model_card_path))
    except Exception as e:
        log.exception("model_card_failed")
        artifacts["errors"].append({"artifact": "model_card", "error": str(e)})


def _copy_artifacts_to_run_dir(
    ctx: StageContext,
    artifacts: dict[str, Any],
    status_path: Path,
    readiness_json_path: Path,
    readiness_md_path: Path,
) -> None:
    run_reports_dir = ctx.run_dir / "reports"
    run_reports_dir.mkdir(parents=True, exist_ok=True)

    for fig_path in artifacts["figures"]:
        if Path(fig_path).exists():
            dest = run_reports_dir / "figures" / Path(fig_path).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(fig_path, dest)

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


def generate_publication_artifacts(ctx: StageContext) -> dict:
    """Generate publication-ready artifacts.

    Creates tables, figures, and model documentation from the fitted
    model and evaluation results.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with paths to generated artifacts.
    """
    inp = _load_publication_inputs(ctx)

    artifacts: dict[str, Any] = {
        "tables": [],
        "figures": [],
        "docs": [],
        "errors": list(inp.input_errors),
    }

    # Each artifact is best-effort: a helper logs and records its own failure in
    # artifacts["errors"] but never blocks the others.
    _build_coefficient_table(inp, artifacts)
    _build_diagnostics_table(inp, artifacts)
    _build_metrics_summary_table(inp, artifacts)
    _build_prediction_scenarios_table(inp, artifacts)
    _save_trace_plot(inp, artifacts)
    _save_posterior_plot(inp, artifacts)
    _save_predictions_plot(inp, artifacts)
    _save_reliability_plot(inp, artifacts)
    _save_artist_fan_charts(inp, artifacts)
    _generate_model_card(inp, artifacts)

    readiness_payload = _build_publication_readiness(
        metrics=inp.metrics if isinstance(inp.metrics, dict) else {},
        diagnostics=inp.diagnostics if isinstance(inp.diagnostics, dict) else {},
        training_summary=inp.training_summary if isinstance(inp.training_summary, dict) else {},
        artifact_errors=artifacts["errors"],
        require_secondary_split=bool(getattr(ctx, "evaluate_secondary_split", True)),
    )
    readiness_json_path = inp.reports_dir / "publication_readiness.json"
    with open(readiness_json_path, "w", encoding="utf-8") as f:
        json.dump(readiness_payload, f, indent=2)
    readiness_md_path = inp.reports_dir / "PUBLICATION_READINESS.md"
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
    status_path = inp.reports_dir / "artifact_status.json"
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

    # Legacy convenience for the flat layout only: with run-scoped paths the
    # artifacts already live under run_dir/reports, so copying would be a
    # same-file error.
    if (
        ctx.run_dir
        and ctx.run_dir.exists()
        and Path(inp.reports_dir).resolve() != (ctx.run_dir / "reports").resolve()
    ):
        _copy_artifacts_to_run_dir(
            ctx, artifacts, status_path, readiness_json_path, readiness_md_path
        )

    log.info(
        "publication_pipeline_complete",
        n_tables=len(artifacts["tables"]),
        n_figures=len(artifacts["figures"]),
        n_docs=len(artifacts["docs"]),
        publication_ready=readiness_payload["ready"],
    )

    return artifacts
