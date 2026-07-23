"""Whole-package typing ratchet (#302).

Runs mypy over src/panelcast with every globally-disabled error code
re-enabled, aggregates errors per file and code, and compares against the
committed baseline. Any growth fails; any shrink must be banked by rerunning
with --update so the baseline stays exact (the repo's whitelist convention:
improvements are recorded, not silently absorbed).

Usage:
    python scripts/typing_ratchet.py --check    # CI gate
    python scripts/typing_ratchet.py --update   # rewrite the baseline
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "typing_baseline.json"

# Kept in sync with [tool.mypy] disable_error_code in pyproject.toml; the
# ratchet measures exactly what the global config forgives.
RATCHETED_CODES = (
    "arg-type",
    "assignment",
    "attr-defined",
)

_ERROR_RE = re.compile(r"^(?P<path>[^:]+):\d+: error: .*\[(?P<code>[a-z-]+)\]$")
_SUMMARY_RE = re.compile(r"^Found (?P<n>\d+) errors? in \d+ files?", re.MULTILINE)


def run_mypy() -> str:
    cmd = [sys.executable, "-m", "mypy", "src/panelcast"]
    for code in RATCHETED_CODES:
        cmd += ["--enable-error-code", code]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):  # 1 = errors found; anything else is mypy crashing
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"mypy failed to run (exit {result.returncode})")
    return result.stdout


def collect_counts(output: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    parsed = 0
    for line in output.splitlines():
        match = _ERROR_RE.match(line.strip())
        if not match:
            continue
        parsed += 1
        path = match.group("path").replace("\\", "/")
        if not path.startswith("src/panelcast/"):
            raise SystemExit(f"mypy error outside src/panelcast (follow-imports leak?): {line}")
        counts[path][match.group("code")] += 1
    # Insurance against a mypy output-format change silently zeroing the
    # counts: the parsed total must match mypy's own summary line.
    summary = _SUMMARY_RE.search(output)
    reported = int(summary.group("n")) if summary else 0
    if reported != parsed:
        raise SystemExit(
            f"parsed {parsed} error lines but mypy reported {reported} — "
            "has the mypy output format changed?"
        )
    return {path: dict(codes) for path, codes in sorted(counts.items())}


def load_baseline() -> dict[str, dict[str, int]]:
    if not BASELINE_PATH.exists():
        raise SystemExit(f"{BASELINE_PATH.name} missing — run with --update to create it")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["files"]


def check(current: dict[str, dict[str, int]]) -> int:
    baseline = load_baseline()
    regressions: list[str] = []
    improvements: list[str] = []
    for path in sorted(set(current) | set(baseline)):
        codes = sorted(set(current.get(path, {})) | set(baseline.get(path, {})))
        for code in codes:
            now = current.get(path, {}).get(code, 0)
            then = baseline.get(path, {}).get(code, 0)
            if now > then:
                regressions.append(f"  {path} [{code}]: {then} -> {now}")
            elif now < then:
                improvements.append(f"  {path} [{code}]: {then} -> {now}")
    if regressions:
        print("Typing ratchet REGRESSIONS (fix them or type the new code):")
        print("\n".join(regressions))
    if improvements:
        print(
            "Typing improved — bank it so the ratchet holds "
            "(python scripts/typing_ratchet.py --update):"
        )
        print("\n".join(improvements))
    if regressions or improvements:
        return 1
    total = sum(sum(codes.values()) for codes in current.values())
    print(f"typing ratchet OK: {total} known errors, no drift")
    return 0


def update(current: dict[str, dict[str, int]]) -> int:
    total = sum(sum(codes.values()) for codes in current.values())
    payload = {
        "description": (
            "Per-file, per-code mypy error counts with the globally-disabled "
            "codes re-enabled (#302). scripts/typing_ratchet.py --check fails "
            "CI on any drift; --update rewrites this file."
        ),
        "total_errors": total,
        "files": current,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"baseline updated: {total} errors across {len(current)} files")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    args = parser.parse_args()
    counts = collect_counts(run_mypy())
    return update(counts) if args.update else check(counts)


if __name__ == "__main__":
    raise SystemExit(main())
