"""Vulnerability audit for the cross-platform pixi lock (#372).

    python scripts/dependency_audit.py --offline   # version floors only, no network
    python scripts/dependency_audit.py --check     # CI gate (OSV)
    python scripts/dependency_audit.py --update    # rewrite security_baseline.json

The environment has two halves, and they do not get the same treatment because
the advisory data behind them is not equally trustworthy.

**PyPI-aware.** Every `pypi:` entry, plus every conda package pixi maps to a
`pkg:pypi/...` purl, is queried against OSV's PyPI ecosystem. Those records
describe exactly the distribution the lock pins, so anything new here fails the
audit. The purl mapping is what makes this conda-aware for the Python half of
the conda environment — most of this project's stack installs from conda-forge,
not from PyPI.

**Native conda.** Conda packages with no PyPI mapping are C libraries. OSV has
no conda ecosystem, so they are matched by name across every ecosystem, which
surfaces the Debian/Ubuntu/Alpine advisories tracking the same upstream source.
Two things make that a report rather than a gate: a distro's fixed-version
range says nothing about what conda-forge built or backported, and a bare name
also collides with unrelated packages in other ecosystems (an npm package named
`seaborn`, a Ruby `zlib`). Use `--strict-conda` to gate on it anyway.

Findings already known against the pinned builds live in
`security_baseline.json`, the same ratchet shape the typing and terminology
gates use. New ids fail; ids that disappear are reported so the baseline can be
tightened. Listing an id says it has been triaged, not that no fixed release
exists — moving the pinned scientific stack is an environment refresh that has
to clear the whole suite, which is a different change from a security floor.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pixi_lock
from packaging.version import InvalidVersion, Version
from pixi_lock import Lock, LockedPackage

OSV_QUERYBATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/{vuln_id}"
BATCH_SIZE = 500
ATTEMPTS = 3

DEFAULT_BASELINE = pixi_lock.REPO_ROOT / "security_baseline.json"

# Floors that must never regress, independent of anything OSV says today. Each is
# the first release carrying the fix for every advisory open against the version
# this repository previously locked.
MINIMUM_VERSIONS: dict[str, str] = {
    # GHSA-94p4-4cq8-9g67 closes the last of the 3.1.47-3.1.55 chain of command
    # injection, config injection, and environment-variable exfiltration.
    "gitpython": "3.1.55",
    # GHSA-hx9q-6w63-j58v / CVE-2025-67221: unbounded recursion on nested JSON.
    "orjson": "3.11.6",
}


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    vuln_id: str
    summary: str = ""


@dataclass
class Report:
    scanned: bool = False  # False means only the floors were checked
    minimums: list[str] = field(default_factory=list)
    pypi: list[Finding] = field(default_factory=list)
    native: list[Finding] = field(default_factory=list)
    new_pypi: list[Finding] = field(default_factory=list)
    new_native: list[Finding] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)

    def failed(self, strict_conda: bool) -> bool:
        return bool(self.minimums or self.new_pypi or (strict_conda and self.new_native))


def check_minimums(lock: Lock) -> list[str]:
    """Report every locked platform whose version sits below a declared floor."""
    violations: list[str] = []
    for (environment, platform), packages in sorted(lock.platform_packages().items()):
        seen: dict[str, set[str]] = defaultdict(set)
        for package in packages:
            key = package.pypi_name or package.name
            if key in MINIMUM_VERSIONS:
                seen[key].add(package.version)
        for name, floor in sorted(MINIMUM_VERSIONS.items()):
            for version in sorted(seen.get(name, set())):
                try:
                    below = Version(version) < Version(floor)
                except InvalidVersion:
                    below = True  # an unreadable version cannot be shown to be safe
                if below:
                    violations.append(
                        f"{environment}/{platform}: {name} {version} is below the required {floor}"
                    )
    return violations


def pypi_requirements(lock: Lock) -> list[str]:
    """Exact pins for every PyPI distribution in the lock, for pip-audit."""
    return sorted({f"{p.name}=={p.version}" for p in lock.packages if p.kind == "pypi"})


def load_baseline(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {name: list(ids) for name, ids in (document.get("packages") or {}).items()}


def write_baseline(path: Path, findings: list[Finding]) -> None:
    grouped: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        grouped[finding.package].add(finding.vuln_id)
    packages = {name: sorted(ids) for name, ids in sorted(grouped.items())}
    document = {
        "description": (
            "Advisories OSV reports against the versions pixi.lock pins (#372). "
            "scripts/dependency_audit.py --check fails CI on any id not listed here; "
            "--update rewrites the file. Listing an id means it has been seen and "
            "triaged, not that no fixed release exists: most of these sit in the "
            "pinned scientific stack, and moving that stack is a separate, "
            "test-gated environment refresh rather than a security floor."
        ),
        "total_advisories": sum(len(ids) for ids in packages.values()),
        "packages": packages,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    last: Exception | None = None
    for _ in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
    raise RuntimeError(f"OSV request to {url} failed after {ATTEMPTS} attempts: {last}")


def _summaries(vuln_ids: set[str]) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for vuln_id in sorted(vuln_ids):
        url = OSV_VULN.format(vuln_id=vuln_id)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                summaries[vuln_id] = json.load(response).get("summary", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            summaries[vuln_id] = ""
    return summaries


def _query(package: LockedPackage) -> dict:
    if package.pypi_name:
        return {
            "package": {"name": package.pypi_name, "ecosystem": "PyPI"},
            "version": package.version,
        }
    # No ecosystem: OSV then matches the name across all of them, which is the
    # only handle there is on a conda-forge build of an upstream C library.
    return {"package": {"name": package.name}, "version": package.version}


def scan(lock: Lock, baseline: dict[str, list[str]]) -> Report:
    """Query OSV for every distinct (name, version) the lock pins."""
    distinct: dict[tuple[str, str, bool], LockedPackage] = {}
    for package in lock.packages:
        key = (package.pypi_name or package.name, package.version, bool(package.pypi_name))
        distinct.setdefault(key, package)
    ordered = [distinct[key] for key in sorted(distinct)]

    results: list[dict] = []
    queries = [_query(package) for package in ordered]
    for start in range(0, len(queries), BATCH_SIZE):
        chunk = queries[start : start + BATCH_SIZE]
        results.extend(_post(OSV_QUERYBATCH, {"queries": chunk}).get("results", []))

    hits: list[tuple[LockedPackage, str]] = [
        (package, vuln["id"])
        for package, result in zip(ordered, results, strict=True)
        for vuln in result.get("vulns") or []
    ]
    summaries = _summaries({vuln_id for package, vuln_id in hits if package.pypi_name})

    report = Report(scanned=True, minimums=check_minimums(lock))
    for package, vuln_id in hits:
        name = package.pypi_name or package.name
        finding = Finding(name, package.version, vuln_id, summaries.get(vuln_id, ""))
        known = vuln_id in baseline.get(name, [])
        if package.pypi_name:
            report.pypi.append(finding)
            if not known:
                report.new_pypi.append(finding)
        else:
            report.native.append(finding)
            if not known:
                report.new_native.append(finding)

    still_present = {(f.package, f.vuln_id) for f in report.pypi + report.native}
    report.resolved = sorted(
        f"{name} {vuln_id}"
        for name, ids in baseline.items()
        for vuln_id in ids
        if (name, vuln_id) not in still_present
    )
    return report


def _render(report: Report, strict_conda: bool) -> str:
    lines: list[str] = []
    if report.minimums:
        lines.append(f"Minimum-version violations ({len(report.minimums)}):")
        lines += [f"  {violation}" for violation in report.minimums]
    else:
        lines.append(f"Version floors: all {len(MINIMUM_VERSIONS)} clear on every locked platform")

    if not report.scanned:
        lines.append("OSV was not queried (--offline)")
        return "\n".join(lines)

    lines.append(
        f"PyPI ecosystem: {len(report.pypi)} advisory match(es), {len(report.new_pypi)} new"
    )
    for finding in sorted(report.new_pypi, key=lambda f: (f.package, f.vuln_id)):
        lines.append(
            f"  NEW {finding.package} {finding.version} {finding.vuln_id} - {finding.summary[:110]}"
        )

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for finding in report.native:
        counts[(finding.package, finding.version)] += 1
    gate = f"gated, {len(report.new_native)} new" if strict_conda else "report only"
    lines.append(
        f"Native conda, name-matched ({gate}): {len(report.native)} match(es) across "
        f"{len(counts)} package(s)"
    )
    for (name, version), count in sorted(counts.items()):
        lines.append(f"  {name} {version}: {count}")

    if report.resolved:
        lines.append(
            f"Baseline can be tightened, {len(report.resolved)} entry/entries no longer match:"
        )
        lines += [f"  {entry}" for entry in report.resolved]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=pixi_lock.DEFAULT_LOCK)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="check the declared minimum versions only, without contacting OSV",
    )
    parser.add_argument("--check", action="store_true", help="gate on the baseline (the default)")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline from this scan")
    parser.add_argument(
        "--strict-conda",
        action="store_true",
        help="also gate on name-matched advisories for native conda packages",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        help="write the lock's PyPI pins to this path (for pip-audit) and exit",
    )
    parser.add_argument("--json", type=Path, help="write the findings as JSON to this path")
    args = parser.parse_args(argv)

    if args.update and (args.offline or args.check):
        parser.error("--update needs a full scan, so it cannot be combined with --offline/--check")

    lock = pixi_lock.parse(args.lock)

    if args.requirements:
        args.requirements.write_text("\n".join(pypi_requirements(lock)) + "\n", encoding="utf-8")
        print(f"wrote {args.requirements}")
        return 0

    if args.offline:
        report = Report(minimums=check_minimums(lock))
    else:
        report = scan(lock, load_baseline(args.baseline))

    print(_render(report, args.strict_conda))

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "minimum_violations": report.minimums,
                    "pypi": [vars(f) for f in report.pypi],
                    "native": [vars(f) for f in report.native],
                    "new": [vars(f) for f in report.new_pypi + report.new_native],
                    "resolved": report.resolved,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.update:
        if report.minimums:
            print("refusing to update the baseline while a version floor is violated")
            return 1
        # Only baseline what the run actually gates on, or --strict-conda could
        # never be satisfied.
        write_baseline(
            args.baseline, report.pypi + report.native if args.strict_conda else report.pypi
        )
        print(f"wrote {args.baseline}")
        return 0

    return 1 if report.failed(args.strict_conda) else 0


if __name__ == "__main__":
    sys.exit(main())
