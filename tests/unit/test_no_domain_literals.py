"""Straggler lint: no new AOTY column literals outside sanctioned files.

The portability contract is that domain names ("Artist", "User_Score", ...)
enter the pipeline only through the DatasetDescriptor. Files in the whitelist
hold sanctioned occurrences: descriptor defaults (default-equals-AOTY),
parameter defaults that mirror them, legacy-summary fallbacks, the AOTY
feature pack and presentation tooling. Any occurrence in a file outside the
whitelist — or a whitelisted file going clean (shrink the list!) — fails.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "panelcast"

LITERAL_PATTERN = re.compile(r"[\"']User_Score[\"']|[\"']Artist[\"']")

# Files with sanctioned occurrences (posix paths relative to src/panelcast).
WHITELIST = {
    "config/descriptor.py",  # default-equals-AOTY field defaults
    "data/cleaning.py",  # legacy constants + AOTY wrapper defaults
    "data/validation.py",  # required-column constants + AOTY column specs
    "data/lineage.py",  # audit-record column defaults
    "data/split.py",  # split function parameter defaults
    "data/ingest.py",  # dimension-extraction defaults
    "data/manifests.py",  # manifest stats parameter default
    "pipelines/build_features.py",  # leakage-mask parameter default
    "pipelines/create_splits.py",  # SplitConfig defaults
    "pipelines/evaluate.py",  # legacy-summary dataset-block fallbacks
    "pipelines/predict_next.py",  # legacy-summary dataset-block fallbacks
    "pipelines/sensitivity.py",  # legacy-summary dataset-block fallbacks
    "pipelines/training_summary.py",  # typed summary dataset-block defaults
    "visualization/dashboard.py",  # dashboard data-loader candidate-column lookup
    "features/artist.py",  # AOTY-pinned subclass
    "features/history.py",  # AOTY default score specs
    "features/temporal.py",  # AOTY column-name parameter defaults
}


def _occurrences() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = [
            lineno
            for lineno, line in enumerate(text.splitlines(), start=1)
            if LITERAL_PATTERN.search(line)
        ]
        if lines:
            found[rel] = lines
    return found


def test_no_domain_literals_outside_whitelist():
    found = _occurrences()
    offenders = {rel: lines for rel, lines in found.items() if rel not in WHITELIST}
    assert not offenders, (
        "AOTY column literals found outside the sanctioned whitelist. Route "
        "these through the DatasetDescriptor (or, for a genuine AOTY default, "
        f"add the file to the whitelist with a justification): {offenders}"
    )


def test_whitelist_has_no_stale_entries():
    found = _occurrences()
    stale = sorted(WHITELIST - set(found))
    assert not stale, f"Whitelist entries with no remaining literals (remove them): {stale}"
