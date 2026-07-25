"""Machine-checkable replication claims (#272).

A domain declares its paper's quantitative claims in a claims.yaml; the
evaluator grades each claim against posterior draws from a fitted run and
produces the verdict table the replication READMEs assemble by hand.
"""

from panelcast.replicate.evaluate import ClaimVerdict, evaluate_claims, exit_code_for
from panelcast.replicate.spec import ClaimsFile, ClaimSpec, load_claims

__all__ = [
    "ClaimSpec",
    "ClaimsFile",
    "ClaimVerdict",
    "evaluate_claims",
    "exit_code_for",
    "load_claims",
]
