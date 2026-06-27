"""Straggler lint: no new AOTY column literals outside sanctioned files.

The portability contract is that domain names ("Artist", "User_Score", ...)
enter the pipeline only through the DatasetDescriptor. Files in the whitelist
hold sanctioned occurrences: descriptor defaults (default-equals-AOTY),
parameter defaults that mirror them, legacy-summary fallbacks, the AOTY
feature pack and presentation tooling. Any occurrence in a file outside the
whitelist — or a whitelisted file going clean (shrink the list!) — fails.

A second, lower-tier scan guards the *output-artifact* schema. The prediction
artifacts now use only the generic entity/event names; the legacy AOTY-named
copies (``artist_mean`` scenario, ``n_training_albums`` column, ``next_album_*``
filenames) have been dropped. The LEGACY_ARTIFACT_ALLOWLIST is therefore empty
and any reappearance of those literals — in the producer, a consumer, or new
code — fails the scan.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "panelcast"

LITERAL_PATTERN = re.compile(r"[\"']User_Score[\"']|[\"']Artist[\"']")

# Second tier: legacy prediction-artifact schema literals. Matches the exact
# quoted scenario/column tokens (not Python identifiers like
# ``predict_new_entity`` or variables like ``artist_mean_features``) and the
# legacy CSV filename stems, which only ever appear in artifact paths.
LEGACY_ARTIFACT_PATTERN = re.compile(
    r"[\"'](?:artist_mean|n_training_albums)[\"']"
    r"|next_album_known_artists"
    r"|next_album_new_artist"
)

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


# The dual-write window is closed: no file may carry the legacy artifact schema.
LEGACY_ARTIFACT_ALLOWLIST: set[str] = set()


def _occurrences(pattern: re.Pattern[str] = LITERAL_PATTERN) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = [
            lineno
            for lineno, line in enumerate(text.splitlines(), start=1)
            if pattern.search(line)
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


def test_no_legacy_artifact_literals_outside_allowlist():
    found = _occurrences(LEGACY_ARTIFACT_PATTERN)
    offenders = {rel: lines for rel, lines in found.items() if rel not in LEGACY_ARTIFACT_ALLOWLIST}
    assert not offenders, (
        "Legacy prediction-artifact schema literals (artist_mean / "
        "n_training_albums / next_album_*) found outside the dual-write "
        "allowlist. New code should write the generic entity/event schema "
        f"(next_event_*, entity, n_training_events): {offenders}"
    )


def test_legacy_artifact_allowlist_has_no_stale_entries():
    found = _occurrences(LEGACY_ARTIFACT_PATTERN)
    stale = sorted(LEGACY_ARTIFACT_ALLOWLIST - set(found))
    assert not stale, (
        "Legacy-artifact allowlist entries with no remaining legacy literals. "
        "The dual-write copies are gone here — drop these from the allowlist: "
        f"{stale}"
    )
