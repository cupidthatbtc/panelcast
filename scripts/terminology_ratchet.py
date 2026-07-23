"""Domain-terminology ratchet for generic modules (#303).

Counts album/artist-derived identifiers per source file and compares against
the committed baseline. Generic modules must not grow new domain terminology;
renames shrink the baseline and are banked with --update. Sanctioned domain
surfaces (the AOTY feature pack, descriptor defaults, AOTY data plumbing) are
exempt from the growth check — they are the place domain names belong.

Usage:
    python scripts/terminology_ratchet.py --check    # CI gate
    python scripts/terminology_ratchet.py --update   # rewrite the baseline
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "panelcast"
BASELINE_PATH = REPO_ROOT / "terminology_baseline.json"

# Any identifier-ish token containing the AOTY domain nouns, case-insensitive
# (albums, artist_idx, n_artists, AlbumTypeBlock, mu_artist, ...).
_TERM_RE = re.compile(r"\b\w*(?:album|artist)\w*\b", re.IGNORECASE)

# Sanctioned domain surfaces: intentionally AOTY-specific code whose counts may
# move freely. Everything else is generic and may only shrink.
SANCTIONED = {
    "config/descriptor.py",  # default-equals-AOTY field defaults
    "data/cleaning.py",  # AOTY raw-column plumbing
    "data/validation.py",  # AOTY column specs
    "features/album_type.py",  # AOTY feature pack
    "features/artist.py",
    "features/genre.py",
    "features/collaboration.py",
    "features/history.py",  # AOTY default score specs
    "features/packs/aoty.py",
}


def collect_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        n = len(_TERM_RE.findall(path.read_text(encoding="utf-8")))
        if n:
            counts[rel] = n
    return counts


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        raise SystemExit(f"{BASELINE_PATH.name} missing — run with --update to create it")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["files"]


def check(current: dict[str, int]) -> int:
    baseline = load_baseline()
    regressions: list[str] = []
    improvements: list[str] = []
    for rel in sorted(set(current) | set(baseline)):
        if rel in SANCTIONED:
            continue
        now, then = current.get(rel, 0), baseline.get(rel, 0)
        if now > then:
            regressions.append(f"  {rel}: {then} -> {now}")
        elif now < then:
            improvements.append(f"  {rel}: {then} -> {now}")
    if regressions:
        print(
            "Terminology ratchet REGRESSIONS — generic modules must not grow "
            "album/artist terminology (route domain names through the "
            "DatasetDescriptor, or add a genuinely AOTY-specific file to "
            "SANCTIONED with a justification):"
        )
        print("\n".join(regressions))
    if improvements:
        print(
            "Terminology shrank — bank it so the ratchet holds "
            "(python scripts/terminology_ratchet.py --update):"
        )
        print("\n".join(improvements))
    if regressions or improvements:
        return 1
    generic_total = sum(n for rel, n in current.items() if rel not in SANCTIONED)
    print(f"terminology ratchet OK: {generic_total} generic-module occurrences, no drift")
    return 0


def update(current: dict[str, int]) -> int:
    generic_total = sum(n for rel, n in current.items() if rel not in SANCTIONED)
    payload = {
        "description": (
            "Per-file counts of album/artist-derived identifiers (#303). "
            "scripts/terminology_ratchet.py --check fails CI when a generic "
            "module's count drifts; sanctioned AOTY surfaces are exempt."
        ),
        "generic_total": generic_total,
        "files": current,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"baseline updated: {generic_total} generic occurrences across {len(current)} files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    args = parser.parse_args()
    counts = collect_counts()
    return update(counts) if args.update else check(counts)


if __name__ == "__main__":
    raise SystemExit(main())
