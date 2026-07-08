"""Per-arm scoring for `panelcast select`: paired held-out ELPD plus the gates.

The estimator is the #63 discipline promoted out of
``scripts/bakeoff_transform_latent.py``: per-observation elpd from a persisted
pointwise ``log_likelihood.nc``, paired per-point against a reference arm on
identical test rows. An arm without a snapshot scores ``None`` — no other
estimator (PSIS-LOO on test, WAIC) is ever substituted; the report renders
"-" instead of a stale number.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp


def pointwise_elpd(nc_path: Path) -> np.ndarray:
    """Per-observation elpd_i from a persisted pointwise log-likelihood idata."""
    import arviz as az

    idata = az.from_netcdf(str(nc_path))
    arr = np.asarray(idata.log_likelihood["y"].values)  # (chain, draw, obs)
    n_samples = arr.shape[0] * arr.shape[1]
    return logsumexp(arr.reshape(n_samples, arr.shape[2]), axis=0) - np.log(n_samples)


@dataclass(frozen=True)
class PairedElpd:
    """Paired per-point held-out elpd difference of one arm vs a reference."""

    diff: float
    dse: float
    z: float | None
    n: int
    note: str | None = None


def paired_elpd(cell_nc: Path, reference_nc: Path) -> PairedElpd:
    """Pair d_i = elpd_i(cell) - elpd_i(reference) on identical test rows.

    diff = sum(d_i), signed cell-minus-reference (positive = beats the
    reference); dse = sqrt(n * var(d_i, ddof=1)) — the paired SE, smaller (and
    the honest significance number) than each side's marginal SE (#63).

    Degenerate pairs get an explicit verdict instead of NaN/None ambiguity:
    identical snapshots (the reference against itself) are z=0 with diff 0;
    a single observation or zero variance with a nonzero diff leaves z None
    with a note saying why.
    """
    elpd_cell = pointwise_elpd(cell_nc)
    elpd_ref = pointwise_elpd(reference_nc)
    if elpd_cell.shape != elpd_ref.shape:
        raise ValueError(
            f"obs dimensions differ: {elpd_cell.shape[0]} ({cell_nc}) vs "
            f"{elpd_ref.shape[0]} ({reference_nc}); paired elpd needs identical test rows"
        )
    d = elpd_cell - elpd_ref
    diff = float(np.sum(d))
    n = int(d.size)
    if n < 2:
        return PairedElpd(
            diff=diff, dse=float("nan"), z=None, n=n,
            note="paired elpd degenerate: single observation, dse undefined",
        )
    dse = float(np.sqrt(n * np.var(d, ddof=1)))
    if dse == 0.0:
        if diff == 0.0:
            return PairedElpd(
                diff=0.0, dse=0.0, z=0.0, n=n,
                note="paired diff identically zero (arm matches the reference)",
            )
        return PairedElpd(
            diff=diff, dse=0.0, z=None, n=n,
            note="paired elpd degenerate: zero variance with nonzero diff",
        )
    return PairedElpd(diff=diff, dse=dse, z=diff / dse, n=n)


@dataclass
class ArmScore:
    """One arm's full scorecard; None marks evidence that does not exist."""

    arm: str
    knobs: dict[str, Any] = field(default_factory=dict)
    elpd_diff: float | None = None
    elpd_dse: float | None = None
    elpd_z: float | None = None
    pit_dev: float | None = None
    cov80: float | None = None
    cov80_delta: float | None = None
    cov95: float | None = None
    cov95_delta: float | None = None
    ppc_pinned: int | None = None
    ppc_pinned_names: tuple[str, ...] = ()
    converged: bool | None = None
    rhat_max: float | None = None
    ess_bulk_min: float | None = None
    divergences: int | None = None
    wall_clock_seconds: float | None = None
    expected_gb: float | None = None
    actual_peak_gb: float | None = None
    resource_ratio: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(obj):
    """Recursively drop non-finite floats (and tuples) so json.dumps stays valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _load_json(path: Path | None, notes: list[str], label: str) -> dict:
    if path is None or not Path(path).exists():
        notes.append(f"{label} missing")
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        notes.append(f"{label} unreadable: {exc}")
        return {}
    if not isinstance(payload, dict):
        notes.append(f"{label} is not a JSON object")
        return {}
    return payload


def _apply_metrics(score: ArmScore, metrics: dict) -> None:
    calibration = metrics.get("calibration") or {}
    coverages = calibration.get("coverages") or {}
    for level, cov_attr, delta_attr in (
        ("0.80", "cov80", "cov80_delta"),
        ("0.95", "cov95", "cov95_delta"),
    ):
        entry = coverages.get(level) or {}
        empirical = entry.get("empirical")
        nominal = entry.get("nominal")
        setattr(score, cov_attr, empirical)
        if empirical is not None and nominal is not None:
            setattr(score, delta_attr, empirical - nominal)
    score.pit_dev = (calibration.get("pit") or {}).get("max_abs_dev_from_uniform")
    pinned = (metrics.get("ppc") or {}).get("extreme_statistics")
    if pinned is not None:
        score.ppc_pinned = len(pinned)
        score.ppc_pinned_names = tuple(pinned)
    absent = [
        label
        for label, value in (
            ("coverage", score.cov95),
            ("pit", score.pit_dev),
            ("ppc", score.ppc_pinned),
        )
        if value is None
    ]
    if metrics and absent:
        score.notes.append("metrics.json missing: " + ", ".join(absent))


def _apply_manifest(score: ArmScore, manifest: dict) -> None:
    durations = manifest.get("stage_durations") or {}
    usage = (manifest.get("resources") or {}).get("train") or {}
    # Fit telemetry first (#88), then the train stage clock, then the whole run.
    wall = usage.get("wall_clock_seconds")
    if wall is None:
        wall = durations.get("train")
    if wall is None and manifest.get("duration_seconds"):
        wall = manifest["duration_seconds"]
    score.wall_clock_seconds = wall
    score.expected_gb = usage.get("expected_gb")
    score.actual_peak_gb = usage.get("actual_peak_gb")
    score.resource_ratio = usage.get("ratio")


def score_arm(
    run_dir: Path | None = None,
    *,
    arm: str,
    knobs: Mapping[str, Any] | None = None,
    reference_nc: Path | None = None,
    metrics_path: Path | None = None,
    diagnostics_path: Path | None = None,
    log_likelihood_path: Path | None = None,
    manifest_path: Path | None = None,
) -> ArmScore:
    """Score one run against the reference snapshot; None-fill whatever is absent.

    Paths default to the run-scoped layout under ``run_dir``
    (``evaluation/{metrics,diagnostics}.json``, ``evaluation/log_likelihood.nc``,
    ``manifest.json``); explicit paths override. A missing pointwise snapshot
    (either side) leaves the elpd fields None — it never falls back to
    metrics.json's PSIS-LOO numbers.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        metrics_path = metrics_path or run_dir / "evaluation" / "metrics.json"
        diagnostics_path = diagnostics_path or run_dir / "evaluation" / "diagnostics.json"
        log_likelihood_path = log_likelihood_path or run_dir / "evaluation" / "log_likelihood.nc"
        manifest_path = manifest_path or run_dir / "manifest.json"

    score = ArmScore(arm=arm, knobs=dict(knobs or {}))
    _apply_metrics(score, _load_json(metrics_path, score.notes, "metrics.json"))

    diagnostics = _load_json(diagnostics_path, score.notes, "diagnostics.json")
    score.converged = diagnostics.get("passed")
    score.rhat_max = diagnostics.get("rhat_max")
    score.ess_bulk_min = diagnostics.get("ess_bulk_min")
    score.divergences = diagnostics.get("divergences")

    _apply_manifest(score, _load_json(manifest_path, score.notes, "manifest.json"))

    if log_likelihood_path is None or not Path(log_likelihood_path).exists():
        score.notes.append("no pointwise log_likelihood.nc — elpd unscored (no substitute)")
    elif reference_nc is None or not Path(reference_nc).exists():
        score.notes.append("no reference log_likelihood.nc — elpd unscored (no substitute)")
    else:
        pair = paired_elpd(Path(log_likelihood_path), Path(reference_nc))
        score.elpd_diff, score.elpd_dse, score.elpd_z = pair.diff, pair.dse, pair.z
        if pair.note:
            score.notes.append(pair.note)
    return score


def rank_arms(scores: Iterable[ArmScore]) -> list[ArmScore]:
    """Rank by paired-elpd z (desc); unscored arms sink below every scored arm.

    Convergence failures keep their rank — the report flags them rather than
    dropping them (a gate failure is a caveat, not missing evidence).
    """
    return sorted(
        scores,
        key=lambda s: (s.elpd_z is None, -(s.elpd_z if s.elpd_z is not None else 0.0)),
    )


def _fmt(v, spec: str = ".3g") -> str:
    """Format one metric for the markdown table ('-' for missing)."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    if isinstance(v, int):
        return str(v)
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


_COLUMNS = [
    ("arm", "arm", "{}"),
    ("elpd_diff", "elpd_diff", "+.1f"),
    ("elpd_dse", "dse", ".1f"),
    ("elpd_z", "z", "+.2f"),
    ("cov80_delta", "d_cov80", "+.3f"),
    ("cov95_delta", "d_cov95", "+.3f"),
    ("pit_dev", "pit_dev", ".3f"),
    ("ppc_pinned", "ppc_pin", None),
    ("ppc_pinned_names", "pinned", "{}"),
    ("converged", "conv", None),
    ("wall_clock_seconds", "wall_s", ".0f"),
    ("actual_peak_gb", "peak_gb", ".2f"),
]


def _row_cells(score: ArmScore) -> list[str]:
    cells = []
    for attr, _, spec in _COLUMNS:
        v = getattr(score, attr)
        if attr == "ppc_pinned_names":
            cells.append(",".join(v) if v else "-")
        elif spec == "{}":
            cells.append(str(v))
        else:
            cells.append(_fmt(v, spec or ".3g"))
    return cells


def _verdict(ranked: list[ArmScore], reference_label: str) -> str:
    scored = [s for s in ranked if s.elpd_z is not None]
    if not scored:
        head = (
            "No arm carries a pointwise log-likelihood snapshot, so nothing is "
            f"scored against {reference_label}."
        )
    else:
        top = scored[0]
        stats = f"{top.elpd_diff:+.1f} +/- {top.elpd_dse:.1f} held-out ELPD (z {top.elpd_z:+.2f})"
        if top.elpd_diff > 0:
            head = f"{top.arm} leads: {stats} vs {reference_label}."
        else:
            head = f"No arm beats {reference_label}; the closest is {top.arm} at {stats}."
    caveats = [
        f"{s.arm} failed the convergence gate (rhat {_fmt(s.rhat_max, '.3f')}, "
        f"ess {_fmt(s.ess_bulk_min, '.0f')}, div {_fmt(s.divergences)}) — treat its "
        "score as diagnostic-scale."
        for s in ranked
        if s.converged is False
    ]
    # z None means either "never paired" (no snapshot: diff is None too) or
    # "paired but degenerate" — the caveats must not conflate the two.
    unscored = [s for s in ranked if s.elpd_z is None]
    missing = [s.arm for s in unscored if s.elpd_diff is None]
    if missing:
        caveats.append(
            "No pointwise log-likelihood snapshot for " + ", ".join(missing) + "; "
            "unranked rather than scored with a different estimator."
        )
    degenerate = [s.arm for s in unscored if s.elpd_diff is not None]
    if degenerate:
        caveats.append(
            "Paired elpd degenerate (z undefined) for " + ", ".join(degenerate) + "."
        )
    return " ".join(["**Verdict:**", head, *caveats])


def _baseline_rows(block: Mapping[str, Any] | list | None) -> list[Mapping[str, Any]]:
    rows = block.get("rows", []) if isinstance(block, Mapping) else (block or [])
    return [r for r in rows if isinstance(r, Mapping)]


def _baseline_conclusion(rows: list[Mapping[str, Any]]) -> str | None:
    def _mae(row) -> float | None:
        v = row.get("mae")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None

    bayes = next((r for r in rows if "bayes" in str(r.get("model", "")).lower()), None)
    if bayes is None or _mae(bayes) is None:
        return None
    split = bayes.get("split")
    gbm = next(
        (
            r
            for r in rows
            if "gbm" in str(r.get("model", "")).lower()
            and (split is None or r.get("split") == split)
            and _mae(r) is not None
        ),
        None,
    )
    if gbm is None:
        return None
    verb = "beats" if _mae(bayes) < _mae(gbm) else "does not beat"
    return (
        f"The structured model {verb} the GBM floor on MAE "
        f"({_mae(bayes):.2f} vs {_mae(gbm):.2f})."
    )


def _baseline_section(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = ["", "## Baseline floor", ""]
    if not rows:
        return lines + ["_No baseline rows provided._"]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines += [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    conclusion = _baseline_conclusion(rows)
    if conclusion:
        lines += ["", conclusion]
    return lines


def render_report(
    scores: Iterable[ArmScore],
    reference_label: str,
    baseline_block: Mapping[str, Any] | list | None = None,
    title: str = "Arm scores",
) -> tuple[str, dict]:
    """Ranked markdown scoreboard plus its JSON-able payload.

    ``baseline_block`` is the serialized rows of a ``run_baseline_comparison``
    ComparisonResult (a plain dict with "rows", or the row list itself) —
    rendered as a "Baseline floor" section so the report can say whether the
    structured model clears the GBM at all.
    """
    ranked = rank_arms(scores)
    lines = [
        f"# {title}",
        "",
        f"elpd_diff is arm minus reference ({reference_label}); positive = beats it. dse is",
        "the *paired* difference SE from per-point elpd diffs on identical test data (#63).",
        'Arms without a pointwise log-likelihood snapshot show "-" — no other estimator is',
        "substituted.",
        "",
        "| " + " | ".join(label for _, label, _ in _COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _COLUMNS) + " |",
    ]
    for score in ranked:
        lines.append("| " + " | ".join(_row_cells(score)) + " |")
    verdict = _verdict(ranked, reference_label)
    lines += ["", verdict]
    if baseline_block is not None:
        lines += _baseline_section(_baseline_rows(baseline_block))
    payload = {
        "title": title,
        "reference": reference_label,
        "arms": [s.to_dict() for s in ranked],
        "verdict": verdict,
        "baseline_floor": (
            _json_safe([dict(r) for r in _baseline_rows(baseline_block)])
            if baseline_block is not None
            else None
        ),
    }
    return "\n".join(lines) + "\n", payload
