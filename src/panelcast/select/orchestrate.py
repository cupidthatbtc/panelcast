"""Top-level `panelcast select` orchestration (#103, A6+A7).

Ties the pieces into one flow: a pre-run plan with predicted cost (for informed
consent), then prior-predictive screen → staged sweep → paired scoring →
pre-registered rules → multi-seed confirmation → one ranked report that IS the
`.audit/` entry for the domain. `select` RECOMMENDS; the default flip stays a
manual PR.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from panelcast.config.descriptor import DatasetDescriptor, load_descriptor
from panelcast.select.rules import DecisionRules, promotable, screenable
from panelcast.select.runner import STAGE2_MAX_WINNERS, SweepConfig, ofat_arms
from panelcast.select.space import KNOBS, enumerate_space
from panelcast.select.tiers import EffortTier

log = structlog.get_logger()


@dataclass
class SelectPlan:
    """The pre-run consent summary: what will run and what it should cost."""

    dataset: str
    effort: str
    n_stage1_arms: int
    min_fits: int
    max_fits_planned: int
    predicted_gpu_hours: float | None
    predicted_peak_gb: float | None
    cost_source: str
    space: dict[str, tuple]
    pruned: dict[str, list]
    notes: list[str] = field(default_factory=list)


def resolve_dims(paths_hint: dict | None) -> dict[str, int] | None:
    """Model dimensions for the cost projection from the prepared feature matrix.

    Domain-neutral: the entity count and sequence length aren't in the feature
    matrix, so the caller passes ``n_artists`` (derived from the descriptor's
    entity column) when it has it; otherwise a coarse estimate is fine — the
    projection is an order-of-magnitude consent number, not a guarantee.
    """
    if not paths_hint:
        return None
    try:
        import pandas as pd

        features = pd.read_parquet(paths_hint["features"])
        n_obs = int(len(features))
        n_features = int(sum(1 for c in features.columns if c != "original_row_id"))
        n_artists = int(paths_hint.get("n_artists") or max(1, n_obs // 5))
        return {
            "n_observations": n_obs,
            "n_features": max(n_features, 1),
            "n_artists": n_artists,
            "max_seq": int(paths_hint.get("max_seq") or 30),
        }
    except (OSError, ValueError):
        return None


def build_plan(
    descriptor: DatasetDescriptor,
    tier: EffortTier,
    cfg: SweepConfig,
    dataset_label: str,
    n_confirmation_seeds: int = 3,
    dims: dict[str, int] | None = None,
    calibration_store_path: Path | None = None,
) -> SelectPlan:
    """Enumerate the space, size the staged plan, and predict its GPU cost."""
    space = enumerate_space(descriptor)
    full_space = {knob.name: knob.values for knob in KNOBS}
    pruned = {
        name: [v for v in full_space[name] if v not in space[name]]
        for name in space
        if len(space[name]) < len(full_space[name])
    }
    n_stage1 = 1 + len(ofat_arms(descriptor))

    # Priority-ordered stage sizing: stage 1 is exact; stage 2 is bounded by
    # the runner's winner cap, stage 3 and confirmation are fixed by the tier.
    notes: list[str] = []
    stage2_upper = 0
    if tier.include_stage2:
        stage2_upper = 1 + math.comb(STAGE2_MAX_WINNERS, 2)
        notes.append(
            f"stage 2 composes at most the top {STAGE2_MAX_WINNERS} stage-1 winners "
            f"(≤{stage2_upper} composed/pairwise arms)"
        )
    # run_confirmation applies the tier's publication_confirm overrides to
    # EVERY confirmation fit, so all 2×seeds fits are priced at that scale —
    # there is no separate publication pass.
    confirm_fits = (2 * n_confirmation_seeds) if tier.confirm else 0
    min_fits = n_stage1 + confirm_fits
    stage3 = tier.stage3_fits if 3 in tier.stages else 0
    max_fits_planned = n_stage1 + stage2_upper + stage3 + confirm_fits
    if cfg.max_fits is not None:
        max_fits_planned = min(max_fits_planned, cfg.max_fits)
        # A cap below the baseline truncates the sweep mid-stage-1; the floor
        # can't exceed the ceiling.
        min_fits = min(min_fits, max_fits_planned)

    predicted_hours: float | None = None
    predicted_peak: float | None = None
    cost_source = "no prepared data — run splits+features first for a GPU cost estimate"
    if dims is not None:
        n_confirm_priced = min(confirm_fits, max_fits_planned)
        predicted_hours, predicted_peak, cost_source = _predict_cost(
            dims, tier, max_fits_planned - n_confirm_priced, n_confirm_priced,
            calibration_store_path,
        )
    if cfg.budget_hours is not None:
        notes.append(f"budget cap: {cfg.budget_hours:g} GPU-h (stages truncate in priority order)")
    return SelectPlan(
        dataset=dataset_label,
        effort=tier.name,
        n_stage1_arms=n_stage1,
        min_fits=min_fits,
        max_fits_planned=max_fits_planned,
        predicted_gpu_hours=predicted_hours,
        predicted_peak_gb=predicted_peak,
        cost_source=cost_source,
        space=space,
        pruned=pruned,
        notes=notes,
    )


def _predict_cost(
    dims: dict[str, int],
    tier: EffortTier,
    n_screen_fits: int,
    n_confirm_fits: int,
    store_path: Path | None,
) -> tuple[float, float, str]:
    """Screening fits priced at tier scale; confirmation fits at the tier's
    publication_confirm scale when set (run_confirmation applies it to all)."""
    from panelcast.gpu_memory.calibration_store import estimate_with_calibration
    from panelcast.gpu_memory.runtime_predictor import predict_fit_seconds

    diag = predict_fit_seconds(
        tier.num_chains, tier.num_samples, tier.num_warmup, dims["n_observations"],
        transform="offset_logit", store_path=store_path,
    )
    total_seconds = diag.seconds * n_screen_fits
    if n_confirm_fits:
        if tier.publication_confirm:
            pub = predict_fit_seconds(
                tier.publication_confirm.get("num_chains", tier.num_chains),
                tier.publication_confirm.get("num_samples", 5000),
                tier.publication_confirm.get("num_warmup", 5000),
                dims["n_observations"], transform="offset_logit", store_path=store_path,
            )
            total_seconds += pub.seconds * n_confirm_fits
        else:
            total_seconds += diag.seconds * n_confirm_fits
    estimate, mem_source = estimate_with_calibration(
        store_path,
        n_observations=dims["n_observations"],
        n_features=dims["n_features"],
        n_artists=dims["n_artists"],
        max_seq=dims["max_seq"],
        num_chains=tier.num_chains,
        num_samples=tier.num_samples,
        num_warmup=tier.num_warmup,
    )
    return total_seconds / 3600.0, estimate.total_gb, f"{diag.source}; {mem_source}"


def render_plan(plan: SelectPlan) -> str:
    """The consent printout — what runs, what it costs, what was pruned."""
    lines = [
        f"panelcast select — {plan.dataset}, effort={plan.effort}",
        "",
        f"  stage-1 arms (one-factor-at-a-time): {plan.n_stage1_arms}",
        f"  planned fits: {plan.min_fits}–{plan.max_fits_planned}",
    ]
    if plan.predicted_gpu_hours is not None:
        lines.append(
            f"  predicted cost: ≈{plan.predicted_gpu_hours:.1f} GPU-h, "
            f"peak ≈{plan.predicted_peak_gb:.1f} GB"
        )
    lines.append(f"  cost basis: {plan.cost_source}")
    for note in plan.notes:
        lines.append(f"  note: {note}")
    lines.append("")
    lines.append("  candidate space (structural pruning shown; nothing dropped for past results):")
    for name, values in plan.space.items():
        pruned = plan.pruned.get(name)
        suffix = f"   [pruned: {', '.join(map(str, pruned))}]" if pruned else ""
        lines.append(f"    {name}: {', '.join(str(v) for v in values)}{suffix}")
    return "\n".join(lines) + "\n"


def run_select(
    dataset: str | None,
    tier: EffortTier,
    rules: DecisionRules,
    cfg: SweepConfig,
    train_df=None,
    feature_cols: list[str] | None = None,
    available_columns: frozenset[str] | None = None,
    launch=None,
    audit_root: Path = Path(".audit"),
) -> dict[str, Any]:
    """Run the full protocol and write the report; returns a result summary.

    ``train_df`` / ``feature_cols`` drive the prior screen and data diagnostics;
    when absent the sweep still runs (the runner rebuilds features per arm) but
    the prior screen is skipped with a note.
    """
    from panelcast.select.runner import run_sweep
    from panelcast.select.scoring import rank_arms, render_report, score_arm

    descriptor = load_descriptor(dataset)
    report_dir = audit_root / f"select_{descriptor.name}"
    report_dir.mkdir(parents=True, exist_ok=True)

    prior_block = ""
    if train_df is not None and feature_cols is not None:
        from panelcast.select.prior_screen import render_prior_block, screen_transforms

        screens = screen_transforms(train_df, feature_cols, descriptor)
        prior_block, prior_payload = render_prior_block(screens)
        (report_dir / "prior_screen.json").write_text(
            json.dumps(prior_payload, indent=2), encoding="utf-8"
        )

    ledger = run_sweep(
        cfg, descriptor, train_df=train_df, available_columns=available_columns, launch=launch,
        scorer=_snapshot_scorer,
    )

    reference_nc: Path | None = None
    for record in ledger.records.values():
        if not record.knobs and record.run_dir:
            reference_nc = Path(record.run_dir) / "evaluation" / "log_likelihood.nc"
    scores = []
    for record in ledger.records.values():
        if record.status != "completed" or not record.run_dir:
            continue
        scores.append(
            score_arm(
                Path(record.run_dir),
                arm=record.arm_id,
                knobs=record.knobs,
                reference_nc=reference_nc,
            )
        )
    ranked = rank_arms(scores)
    verdicts = promotable(scores, rules)

    report_md, report_json = render_report(
        ranked,
        reference_label="shipped defaults (reference arm)",
        title=f"panelcast select — {descriptor.name}",
    )
    report_md += "\n" + _render_verdicts(verdicts, rules)

    # Arms that consumed budget (or never ran) but produced no score: the
    # ranking must not read as coverage of the whole space when it isn't.
    not_evaluated = sorted(
        (r for r in ledger.records.values() if r.status != "completed"),
        key=lambda r: (r.status, r.arm_id),
    )
    if not_evaluated:
        report_md += "\n" + _render_not_evaluated(not_evaluated)
    report_json["not_evaluated"] = [
        {
            "arm": r.arm_id,
            "knobs": r.knobs,
            "stage": r.stage,
            "status": r.status,
            "error": (r.error or "")[-300:] or None,
        }
        for r in not_evaluated
    ]

    # The confirmation candidate is the highest-z arm that clears the promotion
    # bar at SCREENING scale (z + coverage) — convergence is NOT required here,
    # because reduced-sample fits rarely converge and nothing would ever reach
    # confirmation. Convergence is enforced at the publication-scale confirmation
    # fits below; a single-seed z is one draw from the selection lottery.
    winner = max(
        (s for s in scores if screenable(s, rules)),
        key=lambda s: s.elpd_z,
        default=None,
    )
    confirmed: bool | None = None
    if tier.confirm and winner is not None:
        from panelcast.select.confirmation import render_confirmation, run_confirmation

        result = run_confirmation(
            dict(winner.knobs),
            cfg,
            seeds=rules.confirmation_seeds,
            promote_z=rules.promote_z,
            sampler_overrides=tier.publication_confirm,
            launch=launch,
        )
        report_md += "\n" + render_confirmation(result)
        confirmed = result.confirmed
        report_json["confirmation"] = result.to_dict()

    (report_dir / "report.md").write_text(
        (prior_block + "\n" if prior_block else "") + report_md, encoding="utf-8"
    )
    (report_dir / "report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    # Recommend only what survives confirmation (when the tier confirms).
    recommend = winner is not None and (confirmed is None or confirmed)
    return {
        "report_dir": str(report_dir),
        "n_arms_scored": len(scores),
        "n_arms_not_evaluated": len(not_evaluated),
        "not_evaluated": dict(Counter(r.status for r in not_evaluated)),
        "promotable": [v.arm for v in verdicts if v.promote],
        "winner_arm": winner.arm if (winner and recommend) else None,
        "confirmed": confirmed,
        "ledger": str(cfg.sweep_dir / "ledger.json"),
    }


def _snapshot_scorer(run_dir: Path, reference_run: Path | None) -> dict:
    """Light per-arm score the runner attaches to the ledger (z gate for stage 2)."""
    from panelcast.select.scoring import score_arm

    ref_nc = (reference_run / "evaluation" / "log_likelihood.nc") if reference_run else None
    score = score_arm(run_dir, arm="_probe", reference_nc=ref_nc)
    return {"z": score.elpd_z, "elpd_diff": score.elpd_diff}


def _render_not_evaluated(records) -> str:
    lines = [
        "## Not evaluated",
        "",
        "These arms produced no score; the ranking above covers completed arms only.",
        "",
        "| arm | stage | status | knobs | error (tail) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        err_lines = (r.error or "").strip().splitlines()
        tail = err_lines[-1][-200:].replace("|", "\\|") if err_lines else "-"
        knobs = json.dumps(r.knobs, sort_keys=True, default=str).replace("|", "\\|")
        lines.append(f"| {r.arm_id} | {r.stage} | {r.status} | {knobs} | {tail} |")
    return "\n".join(lines) + "\n"


def _render_verdicts(verdicts, rules: DecisionRules) -> str:
    lines = ["## Promotion verdicts (pre-registered rules)", ""]
    lines.append(
        f"Bar: paired-ELPD z ≥ {rules.promote_z:g}, coverage within "
        f"±{rules.coverage_tolerance:g}, convergence "
        f"{'required' if rules.require_convergence else 'not required'}. "
        "`select` recommends; a default flip is a manual PR."
    )
    lines.append("")
    for v in verdicts:
        if v.promote:
            lines.append(f"- **{v.arm}**: PROMOTABLE — clears every pre-registered bar.")
        else:
            lines.append(f"- {v.arm}: held — {'; '.join(v.reasons)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "SelectPlan",
    "build_plan",
    "render_plan",
    "resolve_dims",
    "run_select",
]
