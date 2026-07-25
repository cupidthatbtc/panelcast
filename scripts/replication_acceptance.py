"""Three-domain replication acceptance suite (#273).

Runs `panelcast replicate` for every domain in the acceptance manifest and
diffs each verdict table against its committed expected results — the
regression net that keeps model changes (period effects, spline curves,
per-group variances) from silently breaking domain portability. GPU-scale:
run nightly or by hand, never in PR CI.

Usage:
    python scripts/replication_acceptance.py --config configs/replication_acceptance.yaml
    python scripts/replication_acceptance.py --config ... --only baseball
    python scripts/replication_acceptance.py --config ... --record   # (re)write expected

The manifest declares, per domain: the dataset descriptor, the claims.yaml,
and the committed expected-verdicts JSON. The diff compares each claim's
(achieved grade, verdict) — the graded conclusions — never the observed
posterior numbers, which legitimately wiggle run to run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    domains = (data or {}).get("domains") or []
    if not domains:
        raise SystemExit(f"{path}: manifest declares no domains.")
    required = {"name", "dataset", "claims", "expected"}
    for domain in domains:
        missing = required - set(domain)
        if missing:
            raise SystemExit(
                f"{path}: domain entry {domain.get('name', '?')} lacks {sorted(missing)}."
            )
    return domains


def check_domain_paths(domain: dict) -> None:
    """Fail fast, per selected domain, so --only works with partial checkouts."""
    for key in ("dataset", "claims"):
        target = (REPO_ROOT / domain[key]).resolve()
        if not target.exists():
            raise SystemExit(
                f"{domain['name']}: {key} path {target} does not exist — "
                "check out the sibling replication repo (or add its claims.yaml)."
            )


def run_replicate(domain: dict) -> list[dict]:
    """Run the chain + grading for one domain; return its verdict list."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = Path(handle.name)
    command = [
        sys.executable,
        "-c",
        "from panelcast.cli import main; main()",
        "replicate",
        "--dataset",
        str(domain["dataset"]),
        "--claims",
        str(domain["claims"]),
        "--json",
        str(out_path),
    ]
    # Exit 1 (divergences) still writes verdicts — divergences at the
    # recorded grade are the expected state for the adversarial domains.
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode == 2 or not out_path.exists():
        raise SystemExit(
            f"{domain['name']}: replicate hard-failed (exit {result.returncode})."
        )
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def graded_conclusions(verdicts: list[dict]) -> dict[str, tuple[str, str]]:
    return {v["name"]: (v["achieved"], v["verdict"]) for v in verdicts}


def diff_verdicts(actual: list[dict], expected: list[dict]) -> list[str]:
    """Human lines for every graded-conclusion mismatch; empty means clean."""
    got = graded_conclusions(actual)
    want = graded_conclusions(expected)
    lines = []
    for name in sorted(set(got) | set(want)):
        if name not in got:
            lines.append(f"missing claim '{name}' (expected {want[name]})")
        elif name not in want:
            lines.append(f"unexpected claim '{name}' (got {got[name]})")
        elif got[name] != want[name]:
            lines.append(f"claim '{name}': expected {want[name]}, got {got[name]}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", help="run a single manifest domain by name")
    parser.add_argument(
        "--record",
        action="store_true",
        help="write each domain's actual verdicts as its new expected file",
    )
    args = parser.parse_args()

    domains = load_manifest(args.config)
    if args.only:
        domains = [d for d in domains if d["name"] == args.only]
        if not domains:
            raise SystemExit(f"no manifest domain named '{args.only}'.")

    failures = 0
    for domain in domains:
        print(f"=== {domain['name']} ===")
        check_domain_paths(domain)
        actual = run_replicate(domain)
        expected_path = REPO_ROOT / domain["expected"]
        if args.record:
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(json.dumps(actual, indent=2), encoding="utf-8")
            print(f"recorded {len(actual)} verdicts -> {expected_path}")
            continue
        if not expected_path.exists():
            print(f"FAIL: no expected results at {expected_path} (run --record once).")
            failures += 1
            continue
        with open(expected_path, encoding="utf-8") as f:
            expected = json.load(f)
        mismatches = diff_verdicts(actual, expected)
        if mismatches:
            failures += 1
            for line in mismatches:
                print(f"FAIL: {line}")
        else:
            print(f"OK: {len(actual)} graded conclusions match.")

    if failures:
        print(f"\n{failures} domain(s) diverged from their recorded verdicts.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
