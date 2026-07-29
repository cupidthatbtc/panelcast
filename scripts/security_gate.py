"""Aggregate fail-closed gate for the dependency-security workflow (#372).

Every scanner in that workflow runs to completion and records its own outcome,
so one finding never hides the next and the evidence artifact is always
complete. This is the single step that decides whether the job passes, and it
fails on three separate things: a scanner that errored, a finding that is not an
explicit acceptance, and evidence that is missing or does not say what it is.

    python scripts/security_gate.py --step "OSV audit=success" \\
        --audit evidence/osv-findings.json \\
        --pip-audit evidence/pip-audit-lock.json \\
        --sbom evidence/panelcast-environment.cdx.json:pixi-environment \\
        --present evidence/pypi-pins.txt

An absent or unreadable file is a failure, never a pass: "no findings" from a
scanner that never ran is the one report worth banning outright.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path, problems: list[str]) -> dict | list | None:
    if not path.exists():
        problems.append(f"{path}: missing; the step that writes it did not get that far")
        return None
    if path.stat().st_size == 0:
        problems.append(f"{path}: empty")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"{path}: not valid JSON ({error})")
        return None


def check_steps(steps: list[str], problems: list[str]) -> None:
    for spec in steps:
        label, _, outcome = spec.partition("=")
        if outcome != "success":
            problems.append(f"scanner step {label!r} finished as {outcome or 'unknown'!r}")


def check_audit(path: Path, problems: list[str]) -> None:
    document = _load(path, problems)
    if not isinstance(document, dict):
        return
    if not document.get("osv_queried"):
        problems.append(f"{path}: the audit did not query OSV, so it is not a vulnerability scan")
    for key in ("minimum_violations", "ledger_errors", "expired_acceptances", "unaccepted_gated"):
        entries = document.get(key) or []
        if entries:
            problems.append(f"{path}: {len(entries)} {key.replace('_', ' ')}")
    if (document.get("gate") or {}).get("failed"):
        problems.append(f"{path}: the audit reports its own gate as failed")


def check_pip_audit(path: Path, problems: list[str]) -> None:
    document = _load(path, problems)
    if document is None:
        return
    # pip-audit emits {"dependencies": [...]} in 2.x and a bare list in 1.x.
    dependencies = document.get("dependencies") if isinstance(document, dict) else document
    if not isinstance(dependencies, list) or not dependencies:
        problems.append(f"{path}: no dependencies audited")
        return
    for dependency in dependencies:
        for vulnerability in dependency.get("vulns") or []:
            problems.append(
                f"{path}: {dependency.get('name')} {dependency.get('version')} "
                f"{vulnerability.get('id')}"
            )
        if dependency.get("skip_reason"):
            problems.append(
                f"{path}: {dependency.get('name')} was skipped ({dependency['skip_reason']})"
            )


def check_sbom(spec: str, problems: list[str]) -> None:
    raw_path, _, scope = spec.rpartition(":")
    path = Path(raw_path)
    document = _load(path, problems)
    if not isinstance(document, dict):
        return
    if document.get("bomFormat") != "CycloneDX":
        problems.append(f"{path}: not a CycloneDX document")
    if not document.get("components"):
        problems.append(f"{path}: no components")
    properties = {p["name"]: p["value"] for p in document.get("metadata", {}).get("properties", [])}
    if properties.get("panelcast:scope") != scope:
        problems.append(
            f"{path}: declares scope {properties.get('panelcast:scope')!r}, expected {scope!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", action="append", default=[], metavar="LABEL=OUTCOME")
    parser.add_argument("--audit", type=Path, action="append", default=[])
    parser.add_argument("--pip-audit", type=Path, action="append", default=[])
    parser.add_argument("--sbom", action="append", default=[], metavar="PATH:SCOPE")
    parser.add_argument("--present", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    problems: list[str] = []
    check_steps(args.step, problems)
    for path in args.audit:
        check_audit(path, problems)
    for path in args.pip_audit:
        check_pip_audit(path, problems)
    for spec in args.sbom:
        check_sbom(spec, problems)
    for path in args.present:
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"{path}: missing or empty")

    if problems:
        print(f"Dependency-security gate failed ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    checked = len(args.audit) + len(args.pip_audit) + len(args.sbom) + len(args.present)
    print(
        f"Dependency-security gate passed: {len(args.step)} scanner(s), {checked} evidence file(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
