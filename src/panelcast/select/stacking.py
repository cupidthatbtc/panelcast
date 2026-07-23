"""Predictive stacking over the select arm ledger (#154).

The sweep already persists the evidence stacking needs: per-arm pointwise
held-out log-likelihood snapshots on identical test rows. Stacking (Yao et
al. 2018) turns that ledger from a selection tool into an ensemble — a convex
mixture of arms can strictly beat the best single arm even when no individual
challenger does. The mixture is a forecast product, not a posterior.

Weights are fit by maximizing the stacked log score on the split the elpd
snapshots cover (the primary within-entity split), so the honest headline is
always evaluated elsewhere: the secondary entity-disjoint split's predictive
snapshots when present, never only the split the weights were fit on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from panelcast.evaluation.calibration import (
    compute_coverage,
    compute_weighted_interval_score,
)
from panelcast.evaluation.metrics import compute_crps, compute_point_metrics
from panelcast.select.scoring import _baseline_rows, _baseline_section, pointwise_elpd

log = structlog.get_logger()

WEIGHT_SPLIT = "primary"
HONEST_SPLIT = "secondary"


def _validated_elpd(elpd_matrix: np.ndarray) -> np.ndarray:
    elpd = np.asarray(elpd_matrix, dtype=float)
    if elpd.ndim != 2 or elpd.shape[0] < 1 or elpd.shape[1] < 1:
        raise ValueError(f"elpd_matrix must be (n_arms, n_obs), got shape {elpd.shape}")
    if not np.isfinite(elpd).all():
        raise ValueError("elpd_matrix contains non-finite values")
    return elpd


def stacking_weights(elpd_matrix: np.ndarray) -> np.ndarray:
    """Stacking weights maximizing the stacked log score (Yao et al. 2018).

    ``elpd_matrix`` holds per-observation held-out elpds, one row per arm, on
    identical test rows. Maximizes ``sum_i log(sum_k w_k exp(elpd_ik))`` over
    the simplex via a softmax parameterization (L-BFGS-B from the uniform
    start, analytic gradient), so the weights are a deterministic function of
    the matrix.
    """
    elpd = _validated_elpd(elpd_matrix)
    n_arms = elpd.shape[0]
    if n_arms == 1:
        return np.ones(1)

    def neg_score_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w = softmax(theta)
        log_w = np.log(np.clip(w, 1e-300, None))
        log_mix = logsumexp(elpd + log_w[:, None], axis=0)
        grad_w = np.exp(elpd - log_mix[None, :]).sum(axis=1)
        grad_theta = w * (grad_w - float(w @ grad_w))
        return -float(log_mix.sum()), -grad_theta

    result = minimize(neg_score_and_grad, np.zeros(n_arms), jac=True, method="L-BFGS-B")
    if not result.success:
        # The weights are the whole product: a silent optimizer failure would
        # ship a plausible-looking mixture, so make it loud (still returned —
        # the last iterate is the best available point on the simplex).
        log.warning("stacking_optimizer_not_converged", message=str(result.message))
    return softmax(result.x)


def pseudo_bma_plus_weights(
    elpd_matrix: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> np.ndarray:
    """Pseudo-BMA+ weights: Bayesian bootstrap over observations (Yao et al. 2018).

    The cheap, more stable companion to stacking: each bootstrap draws
    Dirichlet(1) row weights, reweights each arm's total elpd, and softmaxes;
    the reported weight is the bootstrap mean. Deterministic given the seed.
    """
    elpd = _validated_elpd(elpd_matrix)
    n_arms, n_obs = elpd.shape
    if n_arms == 1:
        return np.ones(1)
    rng = np.random.default_rng(seed)
    alpha = rng.dirichlet(np.ones(n_obs), size=n_boot)
    z = (alpha @ elpd.T) * n_obs
    return softmax(z, axis=1).mean(axis=0)


def allocate_mixture_draws(weights: np.ndarray, n_draws: int) -> np.ndarray:
    """Largest-remainder apportionment of ``n_draws`` mixture slots by weight.

    Deterministic (no RNG), so the stacked predictive is as reproducible as
    the weights themselves. Weights are normalized defensively so an
    unnormalized vector cannot over-allocate.
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    target = weights * n_draws
    counts = np.floor(target).astype(int)
    remainder = int(n_draws - counts.sum())
    if remainder > 0:
        order = np.argsort(-(target - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def mixture_predictive(draws_by_arm: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Stack score-scale predictive draws across arms proportional to weight.

    Each arm contributes evenly-thinned draws from its own snapshot (spread
    through the chain, not the head), sized by ``allocate_mixture_draws`` at
    the smallest snapshot's draw count.
    """
    if len(draws_by_arm) != len(weights):
        raise ValueError("one draws matrix per weight is required")
    n = min(int(np.asarray(d).shape[0]) for d in draws_by_arm)
    counts = allocate_mixture_draws(np.asarray(weights), n)
    parts = []
    for draws, k in zip(draws_by_arm, counts):
        if k == 0:
            continue
        draws = np.asarray(draws)
        idx = np.linspace(0, draws.shape[0] - 1, num=int(k)).astype(int)
        parts.append(draws[idx])
    return np.concatenate(parts, axis=0)


def score_predictive(y_true: np.ndarray, draws: np.ndarray) -> dict[str, float]:
    """CRPS / point / coverage / WIS for one predictive matrix — the same
    estimator set the per-run evaluation reports, no new metric code."""
    point = compute_point_metrics(y_true, draws.mean(axis=0))
    return {
        "crps": float(compute_crps(y_true, draws).mean_crps),
        "mae": float(point.mae),
        "rmse": float(point.rmse),
        "r2": float(point.r2),
        "cov80": float(compute_coverage(y_true, draws, prob=0.80).empirical),
        "cov95": float(compute_coverage(y_true, draws, prob=0.95).empirical),
        "wis": float(compute_weighted_interval_score(y_true, draws).wis),
    }


@dataclass
class StackArm:
    """One ledger arm admitted to the stack, with its fitted weights."""

    arm_id: str
    knobs: dict[str, Any]
    run_dir: Path
    elpd: np.ndarray | None = None
    total_elpd: float | None = None
    stacking_weight: float | None = None
    pseudo_bma_weight: float | None = None
    predictive: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "reference" if not self.knobs else self.arm_id


def load_predictive_snapshot(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{split: (draws, y_true)}`` from ``evaluation/predictive.npz``; {} when absent."""
    path = Path(run_dir) / "evaluation" / "predictive.npz"
    if not path.exists():
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(path) as npz:
        for key in npz.files:
            if not key.endswith("_draws"):
                continue
            split = key[: -len("_draws")]
            truth_key = f"{split}_y_true"
            if truth_key in npz.files:
                out[split] = (np.asarray(npz[key]), np.asarray(npz[truth_key]))
    return out


def _final_rung(sweep_dir: Path, entries: list[dict]) -> int:
    """Only final-rung fits carry report-grade evidence (#164) — mirror
    run_select. The sweep config is authoritative; a ledger written without
    one falls back to the highest rung any record reached."""
    config_path = sweep_dir / "sweep_config.json"
    if config_path.exists():
        try:
            rungs = json.loads(config_path.read_text(encoding="utf-8")).get("rungs")
        except (OSError, ValueError, AttributeError):
            rungs = None
        if rungs:
            return len(rungs) - 1
    return max((int(e.get("rung") or 0) for e in entries), default=0)


def load_stack_arms(sweep_dir: Path) -> tuple[list[StackArm], list[tuple[str, str]]]:
    """Ledger arms with usable elpd snapshots, plus (arm_id, reason) exclusions.

    The same "no substitute estimator" discipline as scoring.py: an arm
    without a pointwise snapshot is excluded, never scored another way.
    Screening-rung records are excluded — a ladder sweep persists one record
    per (arm, rung), and mixing screening-scale fits into the stack would
    duplicate every promoted arm at low fidelity. Arms whose obs dimension
    disagrees with the first admitted arm evaluated different test rows and
    cannot be stacked with it.
    """
    sweep_dir = Path(sweep_dir)
    ledger_path = sweep_dir / "ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"no ledger.json in {sweep_dir}")
    entries = json.loads(ledger_path.read_text(encoding="utf-8")).get("arms", [])
    final_rung = _final_rung(sweep_dir, entries)
    arms: list[StackArm] = []
    excluded: list[tuple[str, str]] = []
    n_obs: int | None = None
    for entry in entries:
        aid = str(entry.get("arm_id"))
        status = entry.get("status")
        run_dir = entry.get("run_dir")
        if status != "completed" or not run_dir:
            excluded.append((aid, f"status {status}"))
            continue
        rung = int(entry.get("rung") or 0)
        if rung != final_rung:
            excluded.append((aid, f"screening rung {rung} (final is {final_rung})"))
            continue
        nc_path = Path(run_dir) / "evaluation" / "log_likelihood.nc"
        if not nc_path.exists():
            excluded.append((aid, "no pointwise log_likelihood snapshot"))
            continue
        try:
            elpd = pointwise_elpd(nc_path)
        except (OSError, ValueError, KeyError) as exc:
            excluded.append((aid, f"unreadable snapshot: {exc}"))
            continue
        if n_obs is None:
            n_obs = int(elpd.shape[0])
        elif int(elpd.shape[0]) != n_obs:
            excluded.append((aid, f"obs dimension {elpd.shape[0]} differs from {n_obs}"))
            continue
        arm = StackArm(
            arm_id=aid,
            knobs=dict(entry.get("knobs") or {}),
            run_dir=Path(run_dir),
            elpd=elpd,
            total_elpd=float(elpd.sum()),
        )
        arm.predictive = load_predictive_snapshot(arm.run_dir)
        arms.append(arm)
    return arms, excluded


def _thin_draws(draws: np.ndarray, n: int) -> np.ndarray:
    """Evenly thin to ``n`` draws (spread through the chain, not the head)."""
    draws = np.asarray(draws)
    if draws.shape[0] <= n:
        return draws
    idx = np.linspace(0, draws.shape[0] - 1, num=n).astype(int)
    return draws[idx]


def _split_evaluation(
    arms: list[StackArm], weights: np.ndarray, split: str
) -> tuple[dict[str, Any] | None, str | None]:
    """(rows payload, note) for one split's mixture-vs-singles scoreboard.

    rows=None means the split cannot be scored (note says why). rows with a
    note is a scored-with-caveat: arms lacking this split's snapshot are
    dropped and the mixture renormalized over the rest, with the dropped
    weight mass disclosed rather than an all-or-nothing refusal.
    """
    weights = np.asarray(weights, dtype=float)
    have = np.asarray([split in a.predictive for a in arms])
    covered = float(weights[have].sum()) if have.any() else 0.0
    if covered <= 0.0:
        return None, (
            f"no {split} predictive snapshots in this sweep — evaluate persists "
            "evaluation/predictive.npz per arm on newer sweeps"
        )
    contributing = [(a, w) for a, w, h in zip(arms, weights, have) if h]
    y_ref = contributing[0][0].predictive[split][1]
    for arm, _ in contributing[1:]:
        y_arm = arm.predictive[split][1]
        if y_arm.shape != y_ref.shape or not np.allclose(y_arm, y_ref):
            return None, f"arm {arm.label} evaluated different {split} test rows"
    dropped = float(weights.sum() - covered)
    note = None
    if dropped > 1e-9:
        missing = [a.label for a, h in zip(arms, have) if not h]
        note = (
            f"{dropped:.1%} of stacking weight belongs to arms without a {split} "
            f"snapshot ({', '.join(missing)}); mixture renormalized over the rest"
        )
    subset_w = np.asarray([w for _, w in contributing]) / covered
    mixture = mixture_predictive([a.predictive[split][0] for a, _ in contributing], subset_w)
    n_mix = int(mixture.shape[0])
    rows: dict[str, Any] = {"stacked mixture": score_predictive(y_ref, mixture)}
    champion = max(arms, key=lambda a: a.total_elpd if a.total_elpd is not None else -np.inf)
    reference = next((a for a in arms if not a.knobs), None)
    if reference is champion:
        reference = None  # one row is enough when the champion IS the reference
    for label, arm in (("champion", champion), ("reference", reference)):
        if arm is None or split not in arm.predictive:
            continue
        draws, y_arm = arm.predictive[split]
        if y_arm.shape == y_ref.shape and np.allclose(y_arm, y_ref):
            # Thin to the mixture's draw count: CRPS/coverage carry a mild
            # finite-sample dependence on draws, so keep the comparison fair.
            rows[f"{label} ({arm.label})"] = score_predictive(y_ref, _thin_draws(draws, n_mix))
    return rows, note


_METRIC_COLS = ("crps", "mae", "rmse", "r2", "cov80", "cov95", "wis")


def _metric_table(rows: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| model | " + " | ".join(_METRIC_COLS) + " |",
        "| --- | " + " | ".join("---" for _ in _METRIC_COLS) + " |",
    ]
    for label, metrics in rows.items():
        cells = " | ".join(f"{metrics[c]:.4g}" for c in _METRIC_COLS)
        lines.append(f"| {label} | {cells} |")
    return lines


def render_stack_report(
    arms: list[StackArm],
    excluded: list[tuple[str, str]],
    split_rows: dict[str, dict[str, Any] | None],
    split_notes: dict[str, str],
    baseline_block: Any = None,
    title: str = "panelcast stack",
) -> tuple[str, dict]:
    """Markdown report plus its JSON payload (the render_report pattern)."""
    lines = [
        f"# {title}",
        "",
        "Stacking weights maximize the stacked log score on the primary split's",
        "per-point held-out elpd snapshots (Yao et al. 2018); pseudo-BMA+ is the",
        "Bayesian-bootstrap companion. The mixture is a forecast product, not a",
        "posterior.",
        "",
        "## Weights",
        "",
        "| arm | knobs | total_elpd | stacking_w | pseudo_bma_w |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in sorted(arms, key=lambda a: -(a.stacking_weight or 0.0)):
        knobs = json.dumps(arm.knobs, sort_keys=True, default=str).replace("|", "\\|")
        lines.append(
            f"| {arm.label} | {knobs} | {arm.total_elpd:.1f} | "
            f"{arm.stacking_weight:.3f} | {arm.pseudo_bma_weight:.3f} |"
        )
    for split in (WEIGHT_SPLIT, HONEST_SPLIT):
        caveat = (
            " — weights were fit on this split; in-sample for the mixture"
            if split == WEIGHT_SPLIT
            else " — honest headline (weights never saw this split)"
        )
        lines += ["", f"## Mixture vs singles: {split} split{caveat}", ""]
        rows = split_rows.get(split)
        if rows:
            lines += _metric_table(rows)
            if split in split_notes:
                lines += ["", f"_Caveat: {split_notes[split]}._"]
        else:
            lines.append(f"_Not scored: {split_notes.get(split, 'unavailable')}._")
    lines += ["", _stack_verdict(split_rows)]
    if excluded:
        lines += ["", "## Excluded arms", ""]
        lines += [f"- {aid}: {reason}" for aid, reason in excluded]
    if baseline_block is not None:
        lines += _baseline_section(_baseline_rows(baseline_block))
    payload = {
        "title": title,
        "arms": [
            {
                "arm": a.arm_id,
                "knobs": a.knobs,
                "run_dir": str(a.run_dir),
                "total_elpd": a.total_elpd,
                "stacking_weight": a.stacking_weight,
                "pseudo_bma_weight": a.pseudo_bma_weight,
            }
            for a in arms
        ],
        "excluded": [{"arm": aid, "reason": reason} for aid, reason in excluded],
        "splits": {
            split: (
                {**rows, **({"note": split_notes[split]} if split in split_notes else {})}
                if rows
                else {"note": split_notes.get(split)}
            )
            for split, rows in split_rows.items()
        },
        "verdict": _stack_verdict(split_rows),
    }
    return "\n".join(lines) + "\n", payload


def _stack_verdict(split_rows: dict[str, dict[str, Any] | None]) -> str:
    """The headline sentence; the honest split decides, never the weight split."""
    honest = split_rows.get(HONEST_SPLIT)
    if not honest:
        return (
            "**Verdict:** no honest headline — the weights were fit on the "
            f"{WEIGHT_SPLIT} split and no {HONEST_SPLIT}-split predictive snapshots "
            "exist. Treat the in-split table as optimistic; re-run the sweep with "
            "predictive snapshots or confirm via multi-seed refits before acting."
        )
    mix = honest.get("stacked mixture", {})
    single = next((v for k, v in honest.items() if k.startswith("champion")), None)
    if not mix or single is None:
        return "**Verdict:** honest split scored, but no champion snapshot to compare against."
    better = mix["crps"] < single["crps"]
    return (
        f"**Verdict:** on the {HONEST_SPLIT} (entity-disjoint) split the stacked "
        f"mixture {'beats' if better else 'does not beat'} the champion single arm "
        f"on CRPS ({mix['crps']:.4g} vs {single['crps']:.4g}); "
        f"WIS {mix['wis']:.4g} vs {single['wis']:.4g}, "
        f"MAE {mix['mae']:.4g} vs {single['mae']:.4g}."
    )


def run_stack(
    sweep_dir: Path,
    baselines_path: Path | None = None,
    seed: int = 0,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit weights over the ledger, score the mixture per split, write the report.

    Returns a summary with the report paths and the headline verdict.
    """
    sweep_dir = Path(sweep_dir)
    arms, excluded = load_stack_arms(sweep_dir)
    if len(arms) < 2:
        raise ValueError(
            f"stacking needs at least two arms with elpd snapshots; found {len(arms)} "
            f"in {sweep_dir} ({len(excluded)} excluded)"
        )
    matrix = np.vstack([a.elpd for a in arms if a.elpd is not None])
    w_stack = stacking_weights(matrix)
    w_bma = pseudo_bma_plus_weights(matrix, seed=seed)
    for arm, ws, wb in zip(arms, w_stack, w_bma):
        arm.stacking_weight = float(ws)
        arm.pseudo_bma_weight = float(wb)

    split_rows: dict[str, dict[str, Any] | None] = {}
    split_notes: dict[str, str] = {}
    for split in (WEIGHT_SPLIT, HONEST_SPLIT):
        rows, note = _split_evaluation(arms, w_stack, split)
        split_rows[split] = rows
        if note:
            split_notes[split] = note

    baseline_block = None
    if baselines_path is not None:
        baseline_block = json.loads(Path(baselines_path).read_text(encoding="utf-8"))

    report_md, payload = render_stack_report(
        arms, excluded, split_rows, split_notes, baseline_block,
        title=f"panelcast stack — {sweep_dir.name}",
    )
    out_dir = Path(out_dir) if out_dir is not None else sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stacking.md").write_text(report_md, encoding="utf-8")
    (out_dir / "stacking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "report_md": str(out_dir / "stacking.md"),
        "report_json": str(out_dir / "stacking.json"),
        "n_arms_stacked": len(arms),
        "n_excluded": len(excluded),
        "verdict": payload["verdict"],
    }


__all__ = [
    "StackArm",
    "allocate_mixture_draws",
    "load_predictive_snapshot",
    "load_stack_arms",
    "mixture_predictive",
    "pseudo_bma_plus_weights",
    "render_stack_report",
    "run_stack",
    "score_predictive",
    "stacking_weights",
]
