"""Vulnerability audit for the cross-platform pixi lock (#372).

    python scripts/dependency_audit.py              # CI gate: floors + OSV + ledger
    python scripts/dependency_audit.py --offline    # version floors only, no network
    python scripts/dependency_audit.py --scaffold   # stub new ledger entries to triage

The environment has two halves, and the audit is honest about the fact that only
one of them can be gated.

**PyPI-identified (gated).** Every `pypi:` entry, plus every conda package that
carries -- or shares a conda name with an entry that carries -- a
`pkg:pypi/...` purl, is queried against OSV's PyPI ecosystem. Those records
describe exactly the distribution the lock pins, so any match that is not an
explicit, unexpired acceptance fails the audit. This is what makes the audit
reach conda: almost the whole scientific stack installs from conda-forge.

**Name-matched, cross-ecosystem (reported, never gated).** Conda packages with
no PyPI identity are C libraries. OSV has no conda ecosystem, so the only handle
on them is a bare name lookup across every ecosystem at once, and the result is
not evidence about this environment: the version ranges belong to Debian,
Ubuntu, Alpine, or SUSE builds rather than to what conda-forge compiled, and
bare names collide outright (`seaborn` matches a malicious-npm-package advisory,
`yaml` an npm CVE, `cpython` a RUSTSEC advisory). Reporting it is useful;
gating on it would be a false claim of coverage. `--strict-conda` opts in
locally for anyone who wants to read the tier as a gate.

`security_baseline.json` is the ledger of accepted findings. It is not a
ratchet of bare ids: each acceptance names the package and the exact locked
version, the advisory, why it does or does not reach this codebase, the
remediation, an owner, when it was reviewed, and when the acceptance expires.
Anything missing, generic, or past its expiry fails the audit, and `--scaffold`
writes entries that are *invalid until a human fills them in* -- the tool can
never grant an acceptance on its own.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pixi_lock
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from pixi_lock import Lock, LockedPackage

OSV_QUERY = "https://api.osv.dev/v1/query"
OSV_QUERYBATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/{vuln_id}"
BATCH_SIZE = 500
ATTEMPTS = 3

DEFAULT_BASELINE = pixi_lock.REPO_ROOT / "security_baseline.json"
LEDGER_SCHEMA = 3
MAX_ACCEPTANCE_DAYS = 90
ACCEPTANCE_SCOPES = {"lock", "wheel-runtime"}

ADVISORY_ID = re.compile(r"^[A-Z][A-Z0-9]*-[\w.-]+$")
DECISIONS = {
    # No fixed release exists yet; we run the vulnerable version knowingly.
    "accepted-no-fix",
    # A fix exists but the advisory cannot reach this codebase; upgrade on the
    # next environment refresh rather than as a security change.
    "accepted-not-reachable",
    # A fix exists and is wanted, but the upgrade is blocked on something named
    # in `remediation`; the expiry is the deadline for that work.
    "deferred-upgrade-blocked",
}
# Stock phrases that say a human looked without saying what they concluded.
EMPTY_RATIONALE = re.compile(
    r"^(?:(?:n/?a|none|tbd|todo|triaged|reviewed|accepted|known|not applicable|"
    r"no fix|wont ?fix|by design)[\s.,;:—-]*)+$",
    re.IGNORECASE,
)
MIN_RATIONALE = 60
MIN_REMEDIATION = 25


@dataclass(frozen=True)
class Floor:
    """A version that must never regress, whatever OSV says on any given day."""

    version: str
    advisories: tuple[str, ...]
    reason: str


# Each floor is the first release carrying the fix for every advisory OSV
# reported against the version this repository previously locked.
MINIMUM_VERSIONS: dict[str, Floor] = {
    "gitpython": Floor(
        "3.1.55",
        ("GHSA-94p4-4cq8-9g67",),
        "closes the 3.1.47-3.1.55 command injection, config injection, and "
        "environment-variable exfiltration chain; panelcast reads repository "
        "state through GitPython in utils/git_state.py",
    ),
    "orjson": Floor(
        "3.11.6",
        ("GHSA-hx9q-6w63-j58v", "CVE-2025-67221"),
        "unbounded recursion on nested JSON; reached transitively through "
        "kaleido when plotly exports static figures",
    ),
    "click": Floor(
        "8.3.3",
        ("PYSEC-2026-2132",),
        "local privilege issue in click's completion handling; click is the CLI layer under typer",
    ),
    "pillow": Floor(
        "12.3.0",
        ("GHSA-45hq-cxwh-f6vc", "GHSA-6r8x-57c9-28j4", "GHSA-9hw9-ch79-4vh6"),
        "the 12.1.x heap out-of-bounds write and decompression-bomb wave; "
        "Pillow arrives through matplotlib's image writers",
    ),
    "pip": Floor(
        "26.1.2",
        ("GHSA-wf93-45jw-7689", "GHSA-58qw-9mgm-455v", "GHSA-jp4c-xjxw-mgf9"),
        "path traversal via entry-point names and the concatenated tar/ZIP "
        "confusion; pip installs into the pixi environment",
    ),
    "pyarrow": Floor(
        "23.0.1",
        ("GHSA-rgxp-2hwp-jwgg",),
        "use-after-free reading an Arrow IPC file with pre-buffering; "
        "panelcast only uses pyarrow.parquet, so the fix is taken in the "
        "tested environment while the wheel keeps its wider >=15 range",
    ),
    "pygments": Floor(
        "2.20.0",
        ("GHSA-5239-wwwm-4pmq",),
        "ReDoS in a lexer regex; pygments renders tracebacks under rich",
    ),
    "pytest": Floor(
        "9.0.3",
        ("GHSA-6w46-j5rx-g56g",),
        "world-writable tmpdir handling in the test runner",
    ),
    "setuptools": Floor(
        "83.0.0",
        ("GHSA-h35f-9h28-mq5c",),
        "MANIFEST.in exclusion bypass when building an sdist, which is what "
        "the release workflow does",
    ),
    "tornado": Floor(
        "6.5.7",
        ("GHSA-3x9g-8vmp-wqvf", "GHSA-fqwm-6jpj-5wxc", "GHSA-mgf9-4vpg-hj56"),
        "cookie injection, cross-origin Authorization forwarding, and "
        "unbounded gzip inflation; tornado arrives through matplotlib's webagg "
        "backend and is never served by panelcast",
    ),
}


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    vuln_id: str
    summary: str = ""


@dataclass(frozen=True)
class Acceptance:
    scope: str
    package: str
    version: str
    vuln_id: str
    decision: str
    expires: date

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.scope, canonicalize_name(self.package), self.version, self.vuln_id)


@dataclass
class Report:
    scanned: bool = False  # False means only the floors were checked
    minimums: list[str] = field(default_factory=list)
    ledger_errors: list[str] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    pypi: list[Finding] = field(default_factory=list)
    named: list[Finding] = field(default_factory=list)
    new_pypi: list[Finding] = field(default_factory=list)
    new_named: list[Finding] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    def failed(self, strict_conda: bool = False) -> bool:
        return bool(
            self.minimums
            or self.ledger_errors
            or self.expired
            or self.stale
            or self.new_pypi
            or (strict_conda and self.new_named)
        )


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
                    below = Version(version) < Version(floor.version)
                except InvalidVersion:
                    below = True  # an unreadable version cannot be shown to be safe
                if below:
                    violations.append(
                        f"{environment}/{platform}: {name} {version} is below the "
                        f"required {floor.version} ({', '.join(floor.advisories)})"
                    )
    return violations


def pypi_requirements(lock: Lock) -> list[str]:
    """Exact pins for every PyPI distribution in the lock, for pip-audit."""
    return sorted({f"{p.name}=={p.version}" for p in lock.packages if p.kind == "pypi"})


def _iso(value: Any, field_name: str, where: str, errors: list[str]) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        errors.append(f"{where}: {field_name} is not an ISO date (got {value!r})")
        return None


def _check_text(entry: dict, name: str, minimum: int, where: str, errors: list[str]) -> None:
    text = str(entry.get(name) or "").strip()
    if EMPTY_RATIONALE.match(text):
        errors.append(f"{where}: {name} is a stock phrase ({text!r}), not a rationale")
    elif len(text) < minimum:
        errors.append(
            f"{where}: {name} must be a concrete sentence of at least {minimum} characters"
        )


def _shipped_ledger_errors(path: Path, document: dict[str, Any]) -> list[str]:
    try:
        if path.resolve() != DEFAULT_BASELINE.resolve():
            return []
    except OSError:
        return []

    errors: list[str] = []
    policy = document.get("policy") or {}
    if policy.get("max_acceptance_days") != MAX_ACCEPTANCE_DAYS:
        errors.append(f"{path.name}: policy.max_acceptance_days must be {MAX_ACCEPTANCE_DAYS}")
    if set(policy.get("acceptance_scopes") or []) != ACCEPTANCE_SCOPES:
        errors.append(f"{path.name}: policy.acceptance_scopes must be {sorted(ACCEPTANCE_SCOPES)}")
    try:
        lock_digest = pixi_lock.parse(pixi_lock.DEFAULT_LOCK).digest
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: cannot verify its lock triage digest ({exc})")
    else:
        recorded_digest = (document.get("last_triage") or {}).get("pixi_lock_sha256")
        if recorded_digest != lock_digest:
            errors.append(f"{path.name}: last_triage.pixi_lock_sha256 does not match pixi.lock")
    return errors


def load_ledger(path: Path, today: date) -> tuple[list[Acceptance], list[str], list[str]]:
    """Parse and validate the acceptance ledger. Returns (accepted, errors, expired)."""
    if not path.exists():
        return [], [f"{path.name} is missing; an audit with no ledger cannot be trusted"], []

    errors: list[str] = []
    expired: list[str] = []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [], [f"{path.name} is not valid JSON: {error}"], []

    if document.get("schema") != LEDGER_SCHEMA:
        errors.append(
            f"{path.name}: schema must be {LEDGER_SCHEMA}, got {document.get('schema')!r}"
        )
    errors.extend(_shipped_ledger_errors(path, document))
    entries = document.get("acceptances")
    if not isinstance(entries, list):
        return [], errors + [f"{path.name}: 'acceptances' must be a list"], []

    accepted: list[Acceptance] = []
    for index, entry in enumerate(entries):
        where = f"{path.name}[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}: not an object")
            continue

        # Scoped to this entry: a half-written acceptance must not suppress
        # anything, however well-formed its dates happen to be.
        faults: list[str] = []
        scope = str(entry.get("scope") or "").strip()
        package = str(entry.get("package") or "").strip()
        version = str(entry.get("version") or "").strip()
        vuln_id = str(entry.get("advisory") or "").strip()
        decision = str(entry.get("decision") or "").strip()
        owner = str(entry.get("owner") or "").strip()
        where = (
            f"{path.name}[{index}] {scope or '?'} "
            f"{package or '?'} {version or '?'} {vuln_id or '?'}"
        )

        if scope not in ACCEPTANCE_SCOPES:
            faults.append(f"{where}: scope must be one of {sorted(ACCEPTANCE_SCOPES)}")
        if not package or not version:
            faults.append(f"{where}: package and version are both required")
        if not ADVISORY_ID.match(vuln_id):
            faults.append(f"{where}: advisory must be an OSV id")
        if decision not in DECISIONS:
            faults.append(f"{where}: decision must be one of {sorted(DECISIONS)}")
        if not owner:
            faults.append(f"{where}: owner is required")
        _check_text(entry, "applicability", MIN_RATIONALE, where, faults)
        _check_text(entry, "remediation", MIN_REMEDIATION, where, faults)
        if decision == "accepted-no-fix" and entry.get("fixed_in"):
            faults.append(
                f"{where}: decision says no fix exists but fixed_in is {entry['fixed_in']!r}"
            )
        if decision != "accepted-no-fix" and "fixed_in" not in entry:
            faults.append(f"{where}: fixed_in is required (use null only with accepted-no-fix)")

        reviewed = _iso(entry.get("reviewed"), "reviewed", where, faults)
        expires = _iso(entry.get("expires"), "expires", where, faults)
        if reviewed and expires:
            if expires <= reviewed:
                faults.append(f"{where}: expires must be after reviewed")
            elif expires - reviewed > timedelta(days=MAX_ACCEPTANCE_DAYS):
                faults.append(
                    f"{where}: an acceptance may not run longer than {MAX_ACCEPTANCE_DAYS} days"
                )

        errors += faults
        if faults or expires is None:
            continue
        if expires < today:
            expired.append(f"{where}: acceptance expired on {expires}, re-triage it")
        else:
            accepted.append(Acceptance(scope, package, version, vuln_id, decision, expires))

    return accepted, errors, expired


def write_scaffold(path: Path, findings: list[Finding], today: date) -> list[str]:
    """Merge unknown findings into the ledger as entries a human must complete."""
    document: dict[str, Any] = {"schema": LEDGER_SCHEMA, "acceptances": []}
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        document.setdefault("acceptances", [])
    existing = {
        (
            e.get("scope"),
            canonicalize_name(str(e.get("package") or "")),
            e.get("version"),
            e.get("advisory"),
        )
        for e in document["acceptances"]
    }

    added: list[str] = []
    for finding in sorted(findings, key=lambda f: (f.package, f.version, f.vuln_id)):
        package = canonicalize_name(finding.package)
        key = ("lock", package, finding.version, finding.vuln_id)
        if key in existing:
            continue
        document["acceptances"].append(
            {
                "scope": "lock",
                "package": package,
                "version": finding.version,
                "advisory": finding.vuln_id,
                "summary": finding.summary,
                "fixed_in": "FILL IN: the first fixed release, or null if none exists",
                "applicability": "FILL IN: which code path reaches this, or why none does",
                "decision": "FILL IN: one of " + ", ".join(sorted(DECISIONS)),
                "remediation": "FILL IN: the upgrade or mitigation and what gates it",
                "owner": "FILL IN",
                "reviewed": today.isoformat(),
                "expires": (today + timedelta(days=MAX_ACCEPTANCE_DAYS)).isoformat(),
            }
        )
        added.append(f"{package} {finding.version} {finding.vuln_id}")

    document["schema"] = LEDGER_SCHEMA
    document.setdefault(
        "description",
        "Accepted advisories against the versions pixi.lock pins (#372). Every entry "
        "carries its own applicability, remediation, owner, review date, and expiry; "
        "scripts/dependency_audit.py rejects generic or expired entries and cannot "
        "grant an acceptance on its own.",
    )
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return added


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


def _remaining_pages(query: dict, token: str) -> list[str]:
    """Drain a truncated querybatch result so a long list cannot hide an id."""
    ids: list[str] = []
    while token:
        page = _post(OSV_QUERY, dict(query, page_token=token))
        ids += [vuln["id"] for vuln in page.get("vulns") or []]
        token = page.get("next_page_token", "")
    return ids


def scan(lock: Lock) -> tuple[list[Finding], list[Finding]]:
    """Query OSV for every distinct (name, version) the lock pins."""
    distinct: dict[tuple[str, str, bool], LockedPackage] = {}
    for package in lock.packages:
        key = (package.pypi_name or package.name, package.version, bool(package.pypi_name))
        distinct.setdefault(key, package)
    ordered = [distinct[key] for key in sorted(distinct)]

    queries = [_query(package) for package in ordered]
    results: list[dict] = []
    for start in range(0, len(queries), BATCH_SIZE):
        chunk = queries[start : start + BATCH_SIZE]
        results.extend(_post(OSV_QUERYBATCH, {"queries": chunk}).get("results", []))

    hits: list[tuple[LockedPackage, str]] = []
    for package, query, result in zip(ordered, queries, results, strict=True):
        ids = [vuln["id"] for vuln in result.get("vulns") or []]
        ids += _remaining_pages(query, result.get("next_page_token", ""))
        hits += [(package, vuln_id) for vuln_id in ids]

    summaries = _summaries({vuln_id for package, vuln_id in hits if package.pypi_name})
    pypi: list[Finding] = []
    named: list[Finding] = []
    for package, vuln_id in hits:
        name = package.pypi_name or package.name
        finding = Finding(name, package.version, vuln_id, summaries.get(vuln_id, ""))
        (pypi if package.pypi_name else named).append(finding)
    return pypi, named


def adjudicate(
    lock: Lock,
    pypi: list[Finding],
    named: list[Finding],
    accepted: list[Acceptance],
    errors: list[str],
    expired: list[str],
) -> Report:
    report = Report(
        scanned=True,
        minimums=check_minimums(lock),
        ledger_errors=errors,
        expired=expired,
        pypi=pypi,
        named=named,
    )
    # An acceptance is bound to the exact locked version: a version bump has to
    # be re-adjudicated rather than inheriting the old decision.
    known = {acceptance.key[1:] for acceptance in accepted if acceptance.scope == "lock"}
    report.new_pypi = [
        f for f in pypi if (canonicalize_name(f.package), f.version, f.vuln_id) not in known
    ]
    report.new_named = [
        f for f in named if (canonicalize_name(f.package), f.version, f.vuln_id) not in known
    ]
    matched = {
        (canonicalize_name(f.package), f.version, f.vuln_id) for f in pypi + named
    }
    report.stale = sorted(
        f"{package} {version} {vuln_id}" for package, version, vuln_id in known - matched
    )
    return report


def _render(report: Report, strict_conda: bool) -> str:
    lines: list[str] = []
    if report.minimums:
        lines.append(f"Minimum-version violations ({len(report.minimums)}):")
        lines += [f"  {violation}" for violation in report.minimums]
    else:
        lines.append(f"Version floors: all {len(MINIMUM_VERSIONS)} clear on every locked platform")

    for label, problems in (("Ledger errors", report.ledger_errors), ("Expired", report.expired)):
        if problems:
            lines.append(f"{label} ({len(problems)}):")
            lines += [f"  {problem}" for problem in problems]

    if not report.scanned:
        lines.append("OSV was not queried (--offline): this run is not a vulnerability audit")
        return "\n".join(lines)

    lines.append(
        f"PyPI-identified (gated): {len(report.pypi)} advisory match(es), "
        f"{len(report.new_pypi)} without a current acceptance"
    )
    for finding in sorted(report.new_pypi, key=lambda f: (f.package, f.vuln_id)):
        lines.append(
            f"  NEW {finding.package} {finding.version} {finding.vuln_id} - {finding.summary[:110]}"
        )

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for finding in report.named:
        counts[(finding.package, finding.version)] += 1
    gate = "gated by --strict-conda" if strict_conda else "reported, not gated"
    lines.append(
        f"Name-matched across ecosystems ({gate}): {len(report.named)} match(es) across "
        f"{len(counts)} package(s). These are distro and other-ecosystem records keyed on a "
        f"bare name; they are not evidence about the conda-forge build."
    )
    for (name, version), count in sorted(counts.items(), key=lambda item: -item[1])[:10]:
        lines.append(f"  {name} {version}: {count}")
    if len(counts) > 10:
        lines.append(f"  ... {len(counts) - 10} more package(s)")

    if report.stale:
        lines.append(
            f"Acceptances that no longer match anything ({len(report.stale)}), remove them:"
        )
        lines += [f"  {entry}" for entry in report.stale]
    return "\n".join(lines)


def _evidence(report: Report, lock: Lock, strict_conda: bool, today: date) -> dict:
    return {
        "scanned_on": today.isoformat(),
        "osv_queried": report.scanned,
        "pixi_lock_sha256": lock.digest,
        "packages_in_lock": len(lock.packages),
        "gate": {
            "failed": report.failed(strict_conda),
            "strict_conda": strict_conda,
            "gated_tiers": ["pypi"] + (["name-matched"] if strict_conda else []),
            "reported_only_tiers": [] if strict_conda else ["name-matched"],
        },
        "minimum_violations": report.minimums,
        "ledger_errors": report.ledger_errors,
        "expired_acceptances": report.expired,
        "pypi": [vars(f) for f in report.pypi],
        "name_matched": [vars(f) for f in report.named],
        # Split so a consumer cannot mistake the reported tier for the gated
        # one and fail a build on a Debian advisory about somebody else's build.
        "unaccepted_gated": [
            vars(f) for f in report.new_pypi + (report.new_named if strict_conda else [])
        ],
        "unaccepted_reported_only": [] if strict_conda else [vars(f) for f in report.new_named],
        "stale_acceptances": report.stale,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=pixi_lock.DEFAULT_LOCK)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="check the declared minimum versions only, without contacting OSV",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="add unaccepted findings to the ledger as entries a human must complete",
    )
    parser.add_argument(
        "--strict-conda",
        action="store_true",
        help="also gate on the name-matched cross-ecosystem tier",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        help="write the lock's PyPI pins to this path (for pip-audit) and exit",
    )
    parser.add_argument("--json", type=Path, help="write the findings as JSON to this path")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)

    if args.scaffold and args.offline:
        parser.error("--scaffold needs a full scan, so it cannot be combined with --offline")

    lock = pixi_lock.parse(args.lock)

    if args.requirements:
        args.requirements.write_text("\n".join(pypi_requirements(lock)) + "\n", encoding="utf-8")
        print(f"wrote {args.requirements}")
        return 0

    accepted, errors, expired = load_ledger(args.baseline, args.today)

    if args.offline:
        report = Report(minimums=check_minimums(lock), ledger_errors=errors, expired=expired)
    else:
        pypi, named = scan(lock)
        report = adjudicate(lock, pypi, named, accepted, errors, expired)

    print(_render(report, args.strict_conda))

    if args.json:
        args.json.write_text(
            json.dumps(_evidence(report, lock, args.strict_conda, args.today), indent=2) + "\n",
            encoding="utf-8",
        )

    if args.scaffold:
        if report.minimums:
            print("refusing to touch the ledger while a version floor is violated")
            return 1
        unaccepted = report.new_pypi + (report.new_named if args.strict_conda else [])
        added = write_scaffold(args.baseline, unaccepted, args.today)
        print(f"wrote {args.baseline} ({len(added)} entry/entries to triage)")
        # Still a failure: a scaffolded entry is not an acceptance.
        return 1 if added or report.failed(args.strict_conda) else 0

    return 1 if report.failed(args.strict_conda) else 0


if __name__ == "__main__":
    sys.exit(main())
