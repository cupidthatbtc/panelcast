"""Claim grading and the verdict table (#272).

The grade ladder, strongest first:

- ``match`` — the declared quantitative assertion holds at the claimed
  posterior probability.
- ``qualitative`` — the effect points the right way: direction claims grade
  identically; threshold claims reduce to the sign of the effect; interval
  claims accept a band widened by its own width on each side.
- ``shape_only`` — the quantity is structurally well-defined (e.g. the
  fitted age curve really has a peak), regardless of where it lands.

A claim that fails its target rung but passes a lower one is a DIVERGENCE,
not an error: divergences under a fixed model are findings. FAIL means no
rung passed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panelcast.replicate.extractors import ArtifactBundle, ExtractedQuantity, extract
from panelcast.replicate.spec import GRADES, ClaimsFile, ClaimSpec, ExpectSpec


@dataclass(frozen=True)
class ClaimVerdict:
    name: str
    quantity: str
    expected: str
    observed: str
    achieved: str  # match | qualitative | shape_only | none
    target: str
    verdict: str  # PASS | DIVERGENCE | FAIL
    detail: str


def _prob(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0


def _match_passes(expect: ExpectSpec, draws: np.ndarray) -> bool:
    if expect.direction is not None:
        signed = draws > 0 if expect.direction == "increasing" else draws < 0
        return _prob(signed) >= expect.prob
    if expect.within is not None:
        low, high = expect.within
        return _prob((draws >= low) & (draws <= high)) >= expect.prob
    if expect.greater_than is not None:
        return _prob(draws > expect.greater_than) >= expect.prob
    assert expect.less_than is not None  # ExpectSpec guarantees one assertion
    return _prob(draws < expect.less_than) >= expect.prob


def _qualitative_passes(expect: ExpectSpec, draws: np.ndarray) -> bool:
    if expect.direction is not None:
        return _match_passes(expect, draws)
    if expect.within is not None:
        low, high = expect.within
        width = high - low
        return _prob((draws >= low - width) & (draws <= high + width)) >= expect.prob
    if expect.greater_than is not None:
        return _prob(draws > min(0.0, expect.greater_than)) >= expect.prob
    assert expect.less_than is not None  # ExpectSpec guarantees one assertion
    return _prob(draws < max(0.0, expect.less_than)) >= expect.prob


def _describe_expect(expect: ExpectSpec) -> str:
    if expect.direction is not None:
        suffix = f" from {expect.from_}" if expect.from_ else ""
        return f"{expect.direction}{suffix} (P>={expect.prob:g})"
    if expect.within is not None:
        return f"in [{expect.within[0]:g}, {expect.within[1]:g}] (P>={expect.prob:g})"
    if expect.greater_than is not None:
        return f"> {expect.greater_than:g} (P>={expect.prob:g})"
    return f"< {expect.less_than:g} (P>={expect.prob:g})"


def _describe_draws(draws: np.ndarray) -> str:
    if not draws.size:
        return "no draws"
    lo, mid, hi = np.percentile(draws, [5, 50, 95])
    return f"{mid:.3g} [{lo:.3g}, {hi:.3g}]"


def grade_claim(quantity: ExtractedQuantity, claim: ClaimSpec) -> ClaimVerdict:
    achieved = "none"
    if quantity.shape_ok:
        achieved = "shape_only"
        if _qualitative_passes(claim.expect, quantity.draws):
            achieved = "qualitative"
            if _match_passes(claim.expect, quantity.draws):
                achieved = "match"
    if achieved in GRADES and GRADES.index(achieved) <= GRADES.index(claim.grade):
        verdict = "PASS"
    elif achieved != "none":
        verdict = "DIVERGENCE"
    else:
        verdict = "FAIL"
    return ClaimVerdict(
        name=claim.name,
        quantity=claim.quantity,
        expected=_describe_expect(claim.expect),
        observed=_describe_draws(quantity.draws),
        achieved=achieved,
        target=claim.grade,
        verdict=verdict,
        detail=quantity.detail,
    )


def evaluate_claims(bundle: ArtifactBundle, claims: ClaimsFile) -> list[ClaimVerdict]:
    """Extract and grade every claim; extractor errors become FAIL verdicts."""
    verdicts = []
    for claim in claims.claims:
        try:
            quantity = extract(bundle, claim)
        except ValueError as exc:
            verdicts.append(
                ClaimVerdict(
                    name=claim.name,
                    quantity=claim.quantity,
                    expected=_describe_expect(claim.expect),
                    observed="—",
                    achieved="none",
                    target=claim.grade,
                    verdict="FAIL",
                    detail=str(exc),
                )
            )
            continue
        verdicts.append(grade_claim(quantity, claim))
    return verdicts


def exit_code_for(verdicts: list[ClaimVerdict]) -> int:
    """0 = every claim met its target; 1 = divergences only; 2 = hard fail."""
    if any(v.verdict == "FAIL" for v in verdicts):
        return 2
    if any(v.verdict == "DIVERGENCE" for v in verdicts):
        return 1
    return 0
