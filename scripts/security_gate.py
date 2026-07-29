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
from datetime import date
from pathlib import Path

import dependency_audit
from packaging.utils import canonicalize_name


def _load(path: Path, problems: list[str]) -> dict | list | None:
    try:
        if not path.exists():
            problems.append(f"{path}: missing; the step that writes it did not get that far")
            return None
        if not path.is_file():
            problems.append(f"{path}: not a regular file")
            return None
        if path.stat().st_size == 0:
            problems.append(f"{path}: empty")
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, (dict, list)):
            problems.append(f"{path}: expected a JSON object or array")
            return None
        return document
    except json.JSONDecodeError as error:
        problems.append(f"{path}: not valid JSON ({error})")
    except OSError as error:
        problems.append(f"{path}: unreadable ({error})")
    return None


def check_steps(steps: list[str], problems: list[str]) -> None:
    for spec in steps:
        label, _, outcome = spec.partition("=")
        if outcome != "success":
            problems.append(f"scanner step {label!r} finished as {outcome or 'unknown'!r}")


def check_audit(path: Path, problems: list[str]) -> None:
    document = _load(path, problems)
    if document is None:
        return
    if not isinstance(document, dict):
        problems.append(f"{path}: expected a JSON object")
        return
    if not document.get("osv_queried"):
        problems.append(f"{path}: the audit did not query OSV, so it is not a vulnerability scan")
    for key in ("minimum_violations", "ledger_errors", "expired_acceptances", "unaccepted_gated"):
        entries = document.get(key) or []
        if entries:
            problems.append(f"{path}: {len(entries)} {key.replace('_', ' ')}")
    if (document.get("gate") or {}).get("failed"):
        problems.append(f"{path}: the audit reports its own gate as failed")


def _scoped_path(spec: str, problems: list[str]) -> tuple[Path, str] | None:
    raw_path, separator, scope = spec.rpartition(":")
    if not separator or not raw_path or not scope:
        problems.append(f"{spec}: expected PATH:SCOPE")
        return None
    return Path(raw_path), scope


def check_pip_audit(
    spec: str,
    problems: list[str],
    accepted: list[dependency_audit.Acceptance],
) -> None:
    parsed = _scoped_path(spec, problems)
    if parsed is None:
        return
    path, scope = parsed
    if scope not in dependency_audit.ACCEPTANCE_SCOPES:
        problems.append(f"{spec}: unknown audit scope")
        return
    document = _load(path, problems)
    if document is None:
        return
    # pip-audit emits {"dependencies": [...]} in 2.x and a bare list in 1.x.
    dependencies = document.get("dependencies") if isinstance(document, dict) else document
    if not isinstance(dependencies, list) or not dependencies:
        problems.append(f"{path}: no dependencies audited")
        return
    known = {acceptance.key for acceptance in accepted if acceptance.scope == scope}
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            problems.append(f"{path}: dependency entry is not a JSON object")
            continue
        name = canonicalize_name(str(dependency.get("name") or ""))
        version = str(dependency.get("version") or "")
        for vulnerability in dependency.get("vulns") or []:
            if not isinstance(vulnerability, dict):
                problems.append(f"{path}: vulnerability entry for {name} is malformed")
                continue
            vuln_id = str(vulnerability.get("id") or "")
            if (scope, name, version, vuln_id) not in known:
                problems.append(f"{path}: {name} {version} {vuln_id}")
        if dependency.get("skip_reason"):
            problems.append(f"{path}: {name} was skipped ({dependency['skip_reason']})")


def check_sbom(spec: str, problems: list[str]) -> None:
    parsed = _scoped_path(spec, problems)
    if parsed is None:
        return
    path, scope = parsed
    document = _load(path, problems)
    if document is None:
        return
    if not isinstance(document, dict):
        problems.append(f"{path}: expected a JSON object")
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
    if scope == "wheel-runtime":
        component_refs = {
            component.get("bom-ref") for component in document.get("components") or []
        }
        dependencies = document.get("dependencies")
        dependency_refs = {
            entry.get("ref") for entry in dependencies or [] if isinstance(entry, dict)
        }
        targets = {
            target
            for entry in dependencies or []
            if isinstance(entry, dict)
            for target in entry.get("dependsOn") or []
        }
        if not isinstance(dependencies, list) or dependency_refs != component_refs:
            problems.append(f"{path}: wheel-runtime dependency graph is missing components")
        if not targets <= component_refs:
            problems.append(f"{path}: wheel-runtime dependency graph names unknown components")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=dependency_audit.DEFAULT_BASELINE)
    parser.add_argument("--step", action="append", default=[], metavar="LABEL=OUTCOME")
    parser.add_argument("--audit", type=Path, action="append", default=[])
    parser.add_argument("--pip-audit", action="append", default=[], metavar="PATH:SCOPE")
    parser.add_argument("--sbom", action="append", default=[], metavar="PATH:SCOPE")
    parser.add_argument("--present", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    problems: list[str] = []
    accepted, ledger_errors, expired = dependency_audit.load_ledger(args.baseline, date.today())
    problems.extend(ledger_errors)
    problems.extend(expired)
    check_steps(args.step, problems)
    for path in args.audit:
        check_audit(path, problems)
    for path in args.pip_audit:
        check_pip_audit(path, problems, accepted)
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
