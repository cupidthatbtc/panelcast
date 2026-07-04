"""Pre-registered promotion rules for `panelcast select` (#102, A4).

Thresholds live in ``configs/select.yaml`` — committed BEFORE a sweep runs,
never chosen after seeing the results. This encodes the guardrail from the
invalid-LOO episode: selection over many candidates on one dataset overfits
the selection, so promotion demands pre-declared evidence plus confirmation
on untouched settings. `select` RECOMMENDS; default flips remain manual PRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from panelcast.select.scoring import ArmScore

DEFAULT_RULES_PATH = Path("configs") / "select.yaml"


@dataclass(frozen=True)
class DecisionRules:
    """Promotion thresholds, pre-registered in YAML."""

    promote_z: float = 2.0
    coverage_tolerance: float = 0.03
    require_convergence: bool = True
    confirmation_seeds: tuple[int, ...] = (42, 43, 44)

    @classmethod
    def load(cls, path: Path | None = None) -> DecisionRules:
        """Rules from the YAML ``rules:`` block; shipped defaults when absent."""
        path = path or DEFAULT_RULES_PATH
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return cls()
        block = payload.get("rules") or {}
        known = {
            "promote_z": float,
            "coverage_tolerance": float,
            "require_convergence": bool,
            "confirmation_seeds": lambda v: tuple(int(s) for s in v),
        }
        kwargs: dict[str, Any] = {}
        for key, cast in known.items():
            if key in block:
                kwargs[key] = cast(block[key])
        return cls(**kwargs)


@dataclass
class CandidateVerdict:
    """Whether one scored arm clears the pre-registered bar."""

    arm: str
    promote: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_candidate(score: ArmScore, rules: DecisionRules) -> CandidateVerdict:
    """Apply the pre-registered rules to one arm's scorecard.

    Absent evidence fails the bar — an arm without a paired-ELPD snapshot or
    calibration numbers cannot be promoted, only re-run.
    """
    reasons: list[str] = []

    if score.elpd_z is None:
        reasons.append("no paired-ELPD evidence (missing snapshot on one side)")
    elif score.elpd_z < rules.promote_z:
        reasons.append(
            f"paired-ELPD z {score.elpd_z:+.2f} below the pre-registered "
            f"threshold {rules.promote_z:+.2f}"
        )

    for label, delta in (("80%", score.cov80_delta), ("95%", score.cov95_delta)):
        if delta is None:
            reasons.append(f"no {label} coverage evidence")
        elif abs(delta) > rules.coverage_tolerance:
            reasons.append(
                f"{label} coverage off nominal by {delta:+.3f} "
                f"(tolerance ±{rules.coverage_tolerance:.3f})"
            )

    if rules.require_convergence:
        if score.converged is None:
            reasons.append("no convergence verdict")
        elif not score.converged:
            reasons.append(
                f"convergence gate failed (rhat_max={score.rhat_max}, "
                f"ess_bulk_min={score.ess_bulk_min}, divergences={score.divergences})"
            )

    return CandidateVerdict(arm=score.arm, promote=not reasons, reasons=reasons)


def promotable(scores: list[ArmScore], rules: DecisionRules) -> list[CandidateVerdict]:
    """Verdicts for every scored arm, promotable first."""
    verdicts = [evaluate_candidate(s, rules) for s in scores]
    return sorted(verdicts, key=lambda v: (not v.promote, v.arm))


__all__ = [
    "DEFAULT_RULES_PATH",
    "CandidateVerdict",
    "DecisionRules",
    "evaluate_candidate",
    "promotable",
]
