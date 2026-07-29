"""Dependency-security guards for the cross-platform lock (#372).

Three things have to hold, and each has an adversarial half here, because a
guard that has never been shown to fail is not a guard.

The floors must hold without a network: a vulnerable version must not be able to
re-enter `pixi.lock` on any platform, through either the PyPI or the conda half
of the environment. The acceptance ledger must refuse to launder a finding: an
entry with no rationale, no owner, no expiry, or a stale version is not an
acceptance and neither is a scaffolded stub. And the workflows have to keep
running every scanner and stay fail-closed on the aggregate, so a release cannot
ship without its SBOMs.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dependency_audit  # noqa: E402
import generate_sbom  # noqa: E402
import pixi_lock  # noqa: E402
import security_gate  # noqa: E402

PLATFORMS = {"linux-64", "osx-arm64", "win-64"}
TODAY = date(2026, 7, 29)

# What the lock shipped before the #372 sweep, and what must never come back.
PREVIOUSLY_LOCKED = {
    "click": "8.3.1",
    "gitpython": "3.1.46",
    "orjson": "3.11.5",
    "pillow": "12.1.0",
    "pip": "25.3",
    "pyarrow": "22.0.0",
    "pygments": "2.19.2",
    "pytest": "9.0.2",
    "setuptools": "80.9.0",
    "tornado": "6.5.4",
}

PYPI_WHEEL = "https://files.pythonhosted.org/packages/ab/{name}-{version}-py3-none-any.whl"
CONDA_ARTIFACT = "https://conda.anaconda.org/conda-forge/{subdir}/{name}-{version}-pyh0_0.conda"


@pytest.fixture(scope="module")
def lock() -> pixi_lock.Lock:
    return pixi_lock.parse(REPO / "pixi.lock")


def _pypi_entry(name: str, version: str) -> dict[str, Any]:
    return {
        "pypi": PYPI_WHEEL.format(name=name, version=version),
        "name": name,
        "version": version,
        "sha256": "0" * 64,
    }


def _conda_entry(
    name: str, version: str, pypi_name: str | None, subdir: str = "noarch"
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "conda": CONDA_ARTIFACT.format(name=name, version=version, subdir=subdir),
        "sha256": "1" * 64,
        "purls": [],
    }
    if pypi_name:
        entry["purls"] = [f"pkg:pypi/{pypi_name}?source=hash-mapping"]
    return entry


def _write_lock(path: Path, entries: dict[str, list[dict[str, Any]]]) -> Path:
    packages = [entry for platform_entries in entries.values() for entry in platform_entries]
    document = {
        "version": 6,
        "environments": {
            "default": {
                "packages": {
                    platform: [
                        {kind: entry[kind]}
                        for entry in platform_entries
                        for kind in ("conda", "pypi")
                        if kind in entry
                    ]
                    for platform, platform_entries in entries.items()
                }
            }
        },
        "packages": packages,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _acceptance(**overrides: Any) -> dict[str, Any]:
    entry = {
        "scope": "lock",
        "package": "tornado",
        "version": "6.5.7",
        "advisory": "GHSA-0000-0000-0000",
        "fixed_in": None,
        "applicability": (
            "tornado arrives only through matplotlib's webagg backend; panelcast never "
            "starts an HTTP server, so no request ever reaches the affected parser."
        ),
        "decision": "accepted-no-fix",
        "remediation": "Upgrade as soon as conda-forge ships a patched build.",
        "owner": "cupidthatbtc",
        "reviewed": TODAY.isoformat(),
        "expires": (TODAY + timedelta(days=60)).isoformat(),
    }
    entry.update(overrides)
    return entry


def _ledger(tmp_path: Path, *entries: dict[str, Any]) -> Path:
    path = tmp_path / "security_baseline.json"
    path.write_text(json.dumps({"schema": 3, "acceptances": list(entries)}), encoding="utf-8")
    return path


# --- version floors ---------------------------------------------------------


def test_package_names_use_pep_503_normalization() -> None:
    assert pixi_lock._normalise("Zope.Interface") == "zope-interface"
    assert pixi_lock._normalise("a__b..c--d") == "a-b-c-d"


def test_every_floor_names_the_advisories_and_the_path_that_reaches_it() -> None:
    for name, floor in dependency_audit.MINIMUM_VERSIONS.items():
        assert floor.advisories, f"{name} has no advisory behind its floor"
        assert len(floor.reason) > 40, f"{name} has no stated reason"
        Version(floor.version)


def test_locked_environment_clears_every_floor(lock: pixi_lock.Lock) -> None:
    assert dependency_audit.check_minimums(lock) == []


@pytest.mark.parametrize("name", sorted(dependency_audit.MINIMUM_VERSIONS))
def test_every_platform_actually_pins_the_guarded_package(lock: pixi_lock.Lock, name: str) -> None:
    floor = Version(dependency_audit.MINIMUM_VERSIONS[name].version)
    found: dict[str, set[str]] = {}
    for (_environment, platform), packages in lock.platform_packages().items():
        versions = {p.version for p in packages if (p.pypi_name or p.name) == name}
        if versions:
            found[platform] = versions

    assert PLATFORMS <= set(found), f"{name} is missing from {PLATFORMS - set(found)}"
    for platform, versions in found.items():
        for version in versions:
            assert Version(version) >= floor, f"{platform} pins {name} {version}"


@pytest.mark.parametrize("name", sorted(dependency_audit.MINIMUM_VERSIONS))
def test_manifest_constraint_cannot_resolve_below_the_floor(name: str) -> None:
    with open(REPO / "pixi.toml", "rb") as handle:
        manifest = tomllib.load(handle)

    # A re-solve is only as safe as the manifest: the lock is downstream of it.
    declared = {**manifest["dependencies"], **manifest["pypi-dependencies"]}
    assert name in declared, f"{name} holds a security floor but is not pinned in pixi.toml"
    constraint = SpecifierSet(declared[name])

    assert Version(dependency_audit.MINIMUM_VERSIONS[name].version) in constraint
    assert Version(PREVIOUSLY_LOCKED[name]) not in constraint


def test_vulnerable_pypi_version_is_caught_on_the_one_platform_that_has_it(
    tmp_path: Path,
) -> None:
    path = _write_lock(
        tmp_path / "pixi.lock",
        {
            "linux-64": [_pypi_entry("gitpython", "3.1.57"), _pypi_entry("orjson", "3.11.9")],
            "win-64": [_pypi_entry("gitpython", "3.1.46"), _pypi_entry("orjson", "3.11.9")],
        },
    )

    violations = dependency_audit.check_minimums(pixi_lock.parse(path))

    assert len(violations) == 1
    assert "default/win-64: gitpython 3.1.46 is below the required 3.1.55" in violations[0]
    assert "GHSA-94p4-4cq8-9g67" in violations[0]


def test_vulnerable_orjson_is_caught(tmp_path: Path) -> None:
    path = _write_lock(tmp_path / "pixi.lock", {"linux-64": [_pypi_entry("orjson", "3.11.5")]})

    violations = dependency_audit.check_minimums(pixi_lock.parse(path))

    assert any("orjson 3.11.5 is below the required 3.11.6" in v for v in violations)


def test_switching_a_guarded_package_to_conda_does_not_dodge_the_floor(tmp_path: Path) -> None:
    # The conda build of a Python package carries a pkg:pypi purl; the guard has
    # to follow that mapping or the floor is one `pixi add` away from useless.
    path = _write_lock(
        tmp_path / "pixi.lock",
        {"linux-64": [_conda_entry("gitpython", "3.1.46", "gitpython")]},
    )

    violations = dependency_audit.check_minimums(pixi_lock.parse(path))

    assert any("gitpython 3.1.46 is below the required 3.1.55" in v for v in violations)


def test_a_purl_on_one_platform_covers_the_same_package_on_the_others(tmp_path: Path) -> None:
    # conda-forge's pyarrow declares pkg:pypi/pyarrow on osx-arm64 and nothing on
    # linux-64 and win-64; without propagation the floor and the PyPI-ecosystem
    # query both quietly skip two thirds of the platforms.
    path = _write_lock(
        tmp_path / "pixi.lock",
        {
            "osx-arm64": [_conda_entry("pyarrow", "22.0.0", "pyarrow", subdir="osx-arm64")],
            "linux-64": [_conda_entry("pyarrow", "22.0.0", None, subdir="linux-64")],
        },
    )

    parsed = pixi_lock.parse(path)

    assert all(package.pypi_name == "pyarrow" for package in parsed.packages)
    assert len(dependency_audit.check_minimums(parsed)) == 2


def test_unparseable_version_is_treated_as_unsafe(tmp_path: Path) -> None:
    path = _write_lock(
        tmp_path / "pixi.lock", {"linux-64": [_pypi_entry("orjson", "not-a-version")]}
    )

    assert dependency_audit.check_minimums(pixi_lock.parse(path))


def test_requirements_export_pins_every_pypi_wheel(lock: pixi_lock.Lock) -> None:
    pins = dependency_audit.pypi_requirements(lock)

    assert "gitpython==3.1.57" in pins
    assert all(re.fullmatch(r"[\w.-]+==[\w.+!-]+", pin) for pin in pins)
    assert len(pins) == len({(p.name, p.version) for p in lock.packages if p.kind == "pypi"})


# --- the acceptance ledger --------------------------------------------------


def test_shipped_ledger_is_valid_and_carries_no_unexplained_acceptance() -> None:
    accepted, errors, expired = dependency_audit.load_ledger(REPO / "security_baseline.json", TODAY)

    assert errors == []
    assert expired == []
    # Every advisory the July 2026 sweep found had a fixed release, so the
    # honest ledger is the empty one.
    assert accepted == []


def test_a_complete_acceptance_suppresses_exactly_its_own_finding(tmp_path: Path) -> None:
    path = _ledger(tmp_path, _acceptance())
    accepted, errors, expired = dependency_audit.load_ledger(path, TODAY)
    assert (errors, expired) == ([], [])

    covered = dependency_audit.Finding("tornado", "6.5.7", "GHSA-0000-0000-0000")
    other = dependency_audit.Finding("tornado", "6.5.7", "GHSA-1111-1111-1111")
    report = dependency_audit.adjudicate(
        pixi_lock.Lock((), {}, "digest"), [covered, other], [], accepted, [], []
    )

    assert [f.vuln_id for f in report.new_pypi] == ["GHSA-1111-1111-1111"]
    assert report.failed()


def test_wheel_acceptance_cannot_suppress_a_lock_finding(tmp_path: Path) -> None:
    path = _ledger(tmp_path, _acceptance(scope="wheel-runtime"))
    accepted, errors, expired = dependency_audit.load_ledger(path, TODAY)
    assert (errors, expired) == ([], [])
    finding = dependency_audit.Finding("tornado", "6.5.7", "GHSA-0000-0000-0000")

    report = dependency_audit.adjudicate(
        pixi_lock.Lock((), {}, "digest"), [finding], [], accepted, [], []
    )

    assert report.new_pypi == [finding]


def test_stale_acceptance_fails_the_gate() -> None:
    assert dependency_audit.Report(stale=["pillow 12.3.0 GHSA-x"]).failed()


def test_an_acceptance_does_not_carry_over_to_a_new_version(tmp_path: Path) -> None:
    accepted, _, _ = dependency_audit.load_ledger(tmp_path / "x.json", TODAY)
    path = _ledger(tmp_path, _acceptance())
    accepted, _, _ = dependency_audit.load_ledger(path, TODAY)

    bumped = dependency_audit.Finding("tornado", "6.5.8", "GHSA-0000-0000-0000")
    report = dependency_audit.adjudicate(
        pixi_lock.Lock((), {}, "digest"), [bumped], [], accepted, [], []
    )

    assert report.new_pypi == [bumped]


@pytest.mark.parametrize(
    "overrides",
    [
        {"applicability": "triaged"},
        {"applicability": "n/a"},
        {"applicability": "not applicable"},
        {"applicability": "Reviewed. " * 8},
        {"remediation": "none"},
        {"scope": "unknown"},
        {"owner": ""},
        {"decision": "accepted"},
        {"reviewed": "not-a-date"},
        {"expires": "2026-13-01"},
        {"fixed_in": "6.5.8"},  # contradicts accepted-no-fix
    ],
    ids=lambda o: "-".join(o),
)
def test_a_ledger_entry_without_a_real_rationale_is_rejected(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    path = _ledger(tmp_path, _acceptance(**overrides))

    accepted, errors, _ = dependency_audit.load_ledger(path, TODAY)

    assert errors, f"{overrides} should not validate"
    assert accepted == []


def test_an_expired_acceptance_fails_instead_of_lapsing_into_silence(tmp_path: Path) -> None:
    stale = _acceptance(
        reviewed=(TODAY - timedelta(days=120)).isoformat(),
        expires=(TODAY - timedelta(days=30)).isoformat(),
    )
    path = _ledger(tmp_path, stale)

    accepted, errors, expired = dependency_audit.load_ledger(path, TODAY)

    assert accepted == [] and errors == []
    assert len(expired) == 1 and "expired" in expired[0]
    assert dependency_audit.Report(expired=expired).failed()


def test_an_acceptance_cannot_be_written_to_outlast_the_review_window(tmp_path: Path) -> None:
    path = _ledger(tmp_path, _acceptance(expires=(TODAY + timedelta(days=400)).isoformat()))

    _, errors, _ = dependency_audit.load_ledger(path, TODAY)

    assert any("may not run longer than" in error for error in errors)


def test_a_missing_ledger_is_an_error_not_an_empty_one(tmp_path: Path) -> None:
    _, errors, _ = dependency_audit.load_ledger(tmp_path / "absent.json", TODAY)

    assert errors and "missing" in errors[0]


def test_scaffolding_writes_entries_that_still_fail_the_gate(tmp_path: Path) -> None:
    path = tmp_path / "security_baseline.json"
    finding = dependency_audit.Finding("pillow", "12.3.0", "GHSA-2222-2222-2222", "summary")

    added = dependency_audit.write_scaffold(path, [finding], TODAY)

    assert added == ["pillow 12.3.0 GHSA-2222-2222-2222"]
    assert json.loads(path.read_text())["acceptances"][0]["scope"] == "lock"
    accepted, errors, _ = dependency_audit.load_ledger(path, TODAY)
    assert accepted == [], "a scaffolded stub must never count as an acceptance"
    assert errors
    # Re-scaffolding is idempotent, so a human's edits are not overwritten.
    assert dependency_audit.write_scaffold(path, [finding], TODAY) == []


def test_scaffold_deduplicates_canonical_package_names(tmp_path: Path) -> None:
    path = _ledger(tmp_path, _acceptance(package="Pillow"))
    finding = dependency_audit.Finding(
        "pillow", "6.5.7", "GHSA-0000-0000-0000", "summary"
    )

    assert dependency_audit.write_scaffold(path, [finding], TODAY) == []


def test_shipped_ledger_policy_and_lock_digest_cannot_drift(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "security_baseline.json"
    document = json.loads((REPO / "security_baseline.json").read_text(encoding="utf-8"))
    document["policy"]["max_acceptance_days"] = 999
    document["last_triage"]["pixi_lock_sha256"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(dependency_audit, "DEFAULT_BASELINE", path)

    _, errors, _ = dependency_audit.load_ledger(path, TODAY)

    assert any("max_acceptance_days" in error for error in errors)
    assert any("pixi_lock_sha256" in error for error in errors)


def test_offline_gate_passes_and_says_it_did_not_scan(capsys: pytest.CaptureFixture[str]) -> None:
    assert dependency_audit.main(["--offline"]) == 0
    # "0 findings" from a run that never asked is the one report worth banning.
    assert "OSV was not queried" in capsys.readouterr().out


def test_scaffold_cannot_run_from_an_offline_scan() -> None:
    # An offline run has no findings, so --scaffold would look like a clean sweep.
    with pytest.raises(SystemExit):
        dependency_audit.main(["--scaffold", "--offline"])


def test_name_matched_findings_are_reported_and_never_gate_by_default() -> None:
    report = dependency_audit.Report(
        scanned=True,
        named=[dependency_audit.Finding("zlib", "1.3.2", "DEBIAN-CVE-2016-9840")],
        new_named=[dependency_audit.Finding("zlib", "1.3.2", "DEBIAN-CVE-2016-9840")],
    )

    assert not report.failed(strict_conda=False)
    assert report.failed(strict_conda=True)


def test_the_report_labels_which_tier_is_gated(capsys: pytest.CaptureFixture[str]) -> None:
    report = dependency_audit.Report(
        scanned=True, named=[dependency_audit.Finding("openssl", "3.6.0", "DEBIAN-CVE-2016-1")]
    )

    print(dependency_audit._render(report, strict_conda=False))
    rendered = capsys.readouterr().out

    assert "PyPI-identified (gated)" in rendered
    assert "Name-matched across ecosystems (reported, not gated)" in rendered
    assert "not evidence about the conda-forge build" in rendered


# --- SBOMs ------------------------------------------------------------------


def test_environment_sbom_describes_every_locked_artifact(lock: pixi_lock.Lock) -> None:
    sbom = generate_sbom.build_environment(lock, "9.9.9")
    components = sbom["components"]
    properties = {p["name"]: p["value"] for p in sbom["metadata"]["properties"]}

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == generate_sbom.SPEC_VERSION
    assert len(components) == len(lock.packages)
    assert len({c["bom-ref"] for c in components}) == len(components)
    assert sbom["metadata"]["component"]["version"] == "9.9.9"
    assert properties["panelcast:scope"] == "pixi-environment"
    assert "not the dependency closure of the published wheel" in properties["panelcast:scope-note"]
    assert properties["panelcast:pixi-lock-sha256"] == lock.digest
    expected = uuid.uuid5(generate_sbom.SERIAL_NAMESPACE, lock.digest)
    assert sbom["serialNumber"] == f"urn:uuid:{expected}"


def test_environment_sbom_purls_identify_both_halves(lock: pixi_lock.Lock) -> None:
    purls = {c["purl"] for c in generate_sbom.build_environment(lock, "9.9.9")["components"]}

    assert "pkg:pypi/gitpython@3.1.57" in purls
    conda = [p for p in purls if p.startswith("pkg:conda/")]
    assert conda
    # Without subdir and build the conda purls collapse across platforms.
    assert all("channel=" in p and "subdir=" in p and "build=" in p for p in conda)


def test_environment_sbom_is_a_pure_function_of_the_lock(lock: pixi_lock.Lock) -> None:
    first = json.dumps(generate_sbom.build_environment(lock, "1.2.3"))
    second = json.dumps(generate_sbom.build_environment(lock, "1.2.3"))

    assert first == second


def test_wheel_sbom_is_a_different_document_that_says_what_it_is() -> None:
    installed = {
        "distributions": {"numpy": "2.4.1", "pip": "26.1.2", "panelcast": "9.9.9"},
        "requires": {"numpy": [], "pip": [], "panelcast": ["numpy"]},
        "python_version": "3.12.8",
        "implementation": "CPython",
        "platform_tag": "linux-x86_64",
        "executable": "/tmp/venv/bin/python",
    }

    sbom = generate_sbom.build_wheel_runtime(installed, "9.9.9")
    properties = {p["name"]: p["value"] for p in sbom["metadata"]["properties"]}
    origin = {
        c["name"]: next(p["value"] for p in c["properties"] if p["name"] == "panelcast:origin")
        for c in sbom["components"]
    }

    assert properties["panelcast:scope"] == "wheel-runtime"
    assert properties["panelcast:python-version"] == "3.12.8"
    assert "not a lock" in properties["panelcast:scope-note"]
    assert origin == {
        "numpy": "wheel-closure",
        "pip": "venv-bootstrap",
        "panelcast": "wheel-closure",
    }
    graph = {entry["ref"]: entry["dependsOn"] for entry in sbom["dependencies"]}
    assert graph["pkg:pypi/panelcast@9.9.9"] == ["pkg:pypi/numpy@2.4.1"]
    assert graph["pkg:pypi/numpy@2.4.1"] == []
    # Same closure, same serial number: two builds of a tag stay comparable.
    assert (
        sbom["serialNumber"]
        == generate_sbom.build_wheel_runtime(installed, "9.9.9")["serialNumber"]
    )


def test_wheel_sbom_reads_a_real_interpreter_and_refuses_the_wrong_one() -> None:
    installed = generate_sbom.installed_distributions(Path(sys.executable))

    assert installed["distributions"], "no distributions found in the running interpreter"
    assert installed["python_version"].startswith("3.")
    # An SBOM labelled wheel-runtime must not be able to describe some other
    # environment's packages as if they were the wheel's dependencies.
    with pytest.raises(generate_sbom.NotTheWheelEnvironment):
        generate_sbom.build_wheel_runtime(installed, "0.0.0-never-installed")


def test_sbom_refuses_to_write_the_wrong_version(tmp_path: Path) -> None:
    output = tmp_path / "sbom.json"

    assert generate_sbom.main(["--expect-version", "0.0.0-not-the-tag", "--output", str(output)])
    assert not output.exists()


def test_wheel_sbom_cli_fails_rather_than_describing_the_wrong_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "wheel.cdx.json"
    monkeypatch.setattr(generate_sbom, "_project_version", lambda: "0.0.0-never-installed")

    code = generate_sbom.main(
        ["--scope", "wheel", "--python", sys.executable, "--output", str(output)]
    )

    assert code == 1
    assert not output.exists()


# --- the aggregate gate -----------------------------------------------------


def _evidence(tmp_path: Path) -> dict[str, Path]:
    audit = tmp_path / "osv.json"
    audit.write_text(
        json.dumps(
            {
                "osv_queried": True,
                "gate": {"failed": False},
                "minimum_violations": [],
                "ledger_errors": [],
                "expired_acceptances": [],
                "unaccepted_gated": [],
                "unaccepted_reported_only": [{"package": "openssl"}],
            }
        )
    )
    pip_audit = tmp_path / "pip-audit.json"
    pip_audit.write_text(json.dumps({"dependencies": [{"name": "numpy", "version": "2.4.1"}]}))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "components": [{"name": "numpy", "bom-ref": "pkg:pypi/numpy@2.4.1"}],
                "dependencies": [{"ref": "pkg:pypi/numpy@2.4.1", "dependsOn": []}],
                "metadata": {"properties": [{"name": "panelcast:scope", "value": "wheel-runtime"}]},
            }
        )
    )
    return {"audit": audit, "pip_audit": pip_audit, "sbom": sbom}


def _gate(paths: dict[str, Path], *extra: str) -> int:
    return security_gate.main(
        [
            "--step",
            "OSV audit=success",
            "--audit",
            str(paths["audit"]),
            "--pip-audit",
            f"{paths['pip_audit']}:wheel-runtime",
            "--sbom",
            f"{paths['sbom']}:wheel-runtime",
            *extra,
        ]
    )


def test_gate_passes_only_when_every_scanner_and_every_file_is_there(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)

    # The reported-only tier is present in the evidence and must not fail the
    # build: gating on it would be the false coverage claim this split avoids.
    assert _gate(paths) == 0
    assert _gate(paths, "--step", "pip-audit=failure") == 1
    assert _gate(paths, "--present", str(tmp_path / "never-written.txt")) == 1


def test_gate_fails_on_an_unaccepted_finding_in_the_gated_tier(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    document = json.loads(paths["audit"].read_text())
    document["unaccepted_gated"] = [{"package": "pillow", "vuln_id": "GHSA-x"}]
    paths["audit"].write_text(json.dumps(document))

    assert _gate(paths) == 1


def test_gate_fails_on_an_audit_that_never_queried_osv(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    paths["audit"].write_text(json.dumps({"osv_queried": False, "gate": {"failed": False}}))

    assert _gate(paths) == 1


def test_gate_fails_on_a_pip_audit_finding_or_a_skipped_package(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    paths["pip_audit"].write_text(
        json.dumps(
            {
                "dependencies": [
                    {"name": "pillow", "version": "12.3.0", "vulns": [{"id": "GHSA-x"}]}
                ]
            }
        )
    )
    assert _gate(paths) == 1

    paths["pip_audit"].write_text(
        json.dumps({"dependencies": [{"name": "pillow", "skip_reason": "not on PyPI"}]})
    )
    assert _gate(paths) == 1


def test_gate_accepts_only_a_matching_wheel_runtime_exception(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    paths["pip_audit"].write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "Pillow",
                        "version": "12.3.0",
                        "vulns": [{"id": "GHSA-0000-0000-0000"}],
                    }
                ]
            }
        )
    )
    wheel_ledger = _ledger(
        tmp_path,
        _acceptance(scope="wheel-runtime", package="pillow", version="12.3.0"),
    )
    assert _gate(paths, "--baseline", str(wheel_ledger)) == 0

    lock_ledger = _ledger(tmp_path, _acceptance(scope="lock", package="pillow", version="12.3.0"))
    assert _gate(paths, "--baseline", str(lock_ledger)) == 1
    assert (
        security_gate.main(
            [
                "--baseline",
                str(lock_ledger),
                "--pip-audit",
                f"{paths['pip_audit']}:lock",
            ]
        )
        == 0
    )


def test_gate_rejects_an_sbom_spec_without_a_scope_separator(tmp_path: Path) -> None:
    problems: list[str] = []

    security_gate.check_sbom(str(tmp_path / "sbom.json"), problems)

    assert len(problems) == 1
    assert "expected PATH:SCOPE" in problems[0]

    windows_problems: list[str] = []
    security_gate.check_sbom(r"C:\evidence\sbom.json", windows_problems)
    assert len(windows_problems) == 1
    assert "expected PATH:SCOPE" in windows_problems[0]


def test_gate_fails_when_an_sbom_is_the_wrong_scope_or_empty(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    document = json.loads(paths["sbom"].read_text())

    document["metadata"]["properties"] = [{"name": "panelcast:scope", "value": "pixi-environment"}]
    paths["sbom"].write_text(json.dumps(document))
    assert _gate(paths) == 1

    document["metadata"]["properties"] = [{"name": "panelcast:scope", "value": "wheel-runtime"}]
    document["components"] = []
    paths["sbom"].write_text(json.dumps(document))
    assert _gate(paths) == 1

    document["components"] = [{"name": "numpy", "bom-ref": "pkg:pypi/numpy@2.4.1"}]
    document.pop("dependencies")
    paths["sbom"].write_text(json.dumps(document))
    assert _gate(paths) == 1


def test_gate_fails_on_valid_json_of_the_wrong_type(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    for key in ("audit", "pip_audit", "sbom"):
        original = paths[key].read_text(encoding="utf-8")
        paths[key].write_text("null", encoding="utf-8")
        assert _gate(paths) == 1
        paths[key].write_text(original, encoding="utf-8")


def test_gate_fails_on_missing_evidence_rather_than_reporting_nothing(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    paths["audit"].unlink()

    assert _gate(paths) == 1


# --- workflows --------------------------------------------------------------


def _workflow_text(name: str) -> str:
    return (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _workflow(name: str) -> dict[str, Any]:
    return yaml.safe_load(_workflow_text(name))


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps") or []


def _run(step: dict[str, Any]) -> str:
    return str(step.get("run", ""))


def test_security_workflow_scans_on_its_own_schedule_without_credentials() -> None:
    workflow = _workflow("security.yml")
    triggers = workflow.get("on", workflow.get(True))
    job = workflow["jobs"]["audit"]
    text = yaml.safe_dump(job, sort_keys=True)

    assert "schedule" in triggers, "advisories appear without anyone touching the lock"
    assert "secrets." not in text
    assert job["permissions"] == {"contents": "read"}
    assert "pixi run audit" in text
    assert "pip_audit" in text
    assert all(
        re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", str(step["uses"]))
        for step in _steps(job)
        if "uses" in step
    )


def test_every_scanner_runs_even_when_an_earlier_one_finds_something() -> None:
    job = _workflow("security.yml")["jobs"]["audit"]
    scanners = [step for step in _steps(job) if step.get("id")]

    assert len(scanners) >= 4
    for step in scanners:
        assert step.get("continue-on-error") is True, f"{step['name']} can hide the steps after it"


def test_exactly_one_aggregate_gate_decides_the_job() -> None:
    job = _workflow("security.yml")["jobs"]["audit"]
    gates = [step for step in _steps(job) if "security_gate.py" in _run(step)]
    upload = next(step for step in _steps(job) if "upload-artifact" in str(step.get("uses", "")))

    assert len(gates) == 1
    gate = gates[0]
    assert gate.get("if") == "always()", "the gate must run even after a scanner errors"
    assert gate.get("continue-on-error") is not True
    # It has to be told about every scanner, or a failure slips past it.
    assert _run(gate).count("--step") == 5
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload.get("if") == "always()"


def test_scanner_versions_are_pinned() -> None:
    workflow = _workflow("security.yml")
    environment = workflow["env"]
    setup_pixi = next(
        step
        for step in _steps(workflow["jobs"]["audit"])
        if "setup-pixi" in str(step.get("uses", ""))
    )

    for name in ("PIP_AUDIT_VERSION", "BUILD_VERSION", "PYYAML_VERSION", "PACKAGING_VERSION"):
        assert re.fullmatch(r"\d+\.\d+(\.\d+)?", str(environment[name])), name
    assert "pip-audit==${PIP_AUDIT_VERSION}" in _workflow_text("security.yml")
    assert re.fullmatch(r"v\d+\.\d+\.\d+", str(setup_pixi["with"]["pixi-version"]))
    # The release SBOM is evidence too, so its toolchain is pinned the same way.
    release_env = _workflow("release.yml")["env"]
    for name in (
        "PYYAML_VERSION",
        "PACKAGING_VERSION",
        "PIP_VERSION",
        "PIP_AUDIT_VERSION",
        "BUILD_VERSION",
        "TWINE_VERSION",
        "PYTEST_VERSION",
    ):
        assert re.fullmatch(r"\d+\.\d+(\.\d+)?", str(release_env[name])), name


def test_the_wheel_closure_is_audited_and_not_just_the_lock() -> None:
    job = _workflow("security.yml")["jobs"]["audit"]
    step = next(step for step in _steps(job) if step.get("id") == "wheel_closure")

    assert "python -m build" in _run(step)
    assert "--scope wheel" in _run(step)
    assert "pip_audit -r evidence/wheel-pins.txt" in _run(step)


def test_release_builds_both_sboms_and_keeps_them_out_of_pypi() -> None:
    jobs = _workflow("release.yml")["jobs"]

    generate = next(step for step in _steps(jobs["build"]) if "generate_sbom.py" in _run(step))
    assert "--scope environment" in _run(generate)
    assert "--scope wheel" in _run(generate)
    assert "--expect-version" in _run(generate), "an SBOM for another version is not this release"
    assert "dist" not in _run(generate), "anything under dist/ is uploaded to PyPI"

    upload = next(
        step for step in _steps(jobs["build"]) if (step.get("with") or {}).get("path") == "sbom/"
    )
    assert "upload-artifact" in str(upload["uses"])
    assert upload["with"]["if-no-files-found"] == "error"


def test_publication_waits_on_the_sboms_being_verifiably_attached() -> None:
    jobs = _workflow("release.yml")["jobs"]
    attach = next(
        step for step in _steps(jobs["release-sbom"]) if "gh release upload" in _run(step)
    )
    text = _run(attach)

    assert "release-sbom" in jobs["publish"]["needs"]
    assert jobs["release-sbom"]["permissions"] == {"contents": "write"}
    # The release is created rather than waited for, and the upload is read back.
    assert "gh release create" in text
    assert "--verify-tag" in text
    assert "gh release download" in text
    assert "cmp -s" in text
    assert "security_gate.py" in text
    assert "::warning" not in yaml.safe_dump(jobs["release-sbom"]), (
        "a warning is not a gate: the SBOM has to block the release"
    )

    finalizer = jobs["publish-release"]
    assert set(finalizer["needs"]) == {"publish", "release-sbom"}
    assert finalizer["permissions"] == {"contents": "write"}
    finalizer_run = "\n".join(_run(step) for step in _steps(finalizer))
    assert "gh release edit" in finalizer_run
    assert "--draft=false" in finalizer_run
    assert "--draft=false" not in text, "the release must stay draft until PyPI succeeds"


def test_tag_publication_reruns_advisory_scans_and_metadata_guards() -> None:
    build = _workflow("release.yml")["jobs"]["build"]
    steps = _steps(build)
    text = "\n".join(_run(step) for step in steps)
    metadata_step = next(
        step
        for step in steps
        if step.get("name") == "Verify release metadata at the tagged commit"
    )

    assert (
        "pytest --confcutdir=tests/unit tests/unit/test_release_metadata.py"
        in _run(metadata_step)
    )
    assert "dependency_audit.py" in text
    assert "pip-audit-lock.json:lock" in text
    assert "pip-audit-wheel.json:wheel-runtime" in text
    assert "--audit security/osv-findings.json" in text
    assert "steps.release_osv.outcome" in text
    assert "steps.release_pip_lock.outcome" in text
    assert "steps.release_pip_wheel.outcome" in text


def test_pr_ci_proves_release_metadata_without_project_dependencies() -> None:
    job = _workflow("ci.yml")["jobs"]["release-metadata"]
    text = "\n".join(_run(step) for step in _steps(job))

    assert "pytest==9.1.1" in text
    assert 'find_spec("jax") is None' in text
    assert (
        "pytest --confcutdir=tests/unit tests/unit/test_release_metadata.py"
        in text
    )


def test_every_standalone_security_gate_installs_its_imports_first() -> None:
    release_job = _workflow("release.yml")["jobs"]["release-sbom"]
    release_runs = [_run(step) for step in _steps(release_job)]
    release_install = next(i for i, text in enumerate(release_runs) if "pip install" in text)
    release_gate = next(i for i, text in enumerate(release_runs) if "security_gate.py" in text)
    assert release_install < release_gate
    assert "pyyaml==${PYYAML_VERSION}" in release_runs[release_install]
    assert "packaging==${PACKAGING_VERSION}" in release_runs[release_install]

    security_job = _workflow("security.yml")["jobs"]["audit"]
    security_runs = [_run(step) for step in _steps(security_job)]
    scanner_install = next(i for i, text in enumerate(security_runs) if "pip-audit" in text)
    aggregate_gate = next(i for i, text in enumerate(security_runs) if "security_gate.py" in text)
    assert scanner_install < aggregate_gate
    assert "pyyaml==${PYYAML_VERSION}" in security_runs[scanner_install]
    assert "packaging==${PACKAGING_VERSION}" in security_runs[scanner_install]


def test_cross_platform_wheel_and_lock_ci_are_still_in_place() -> None:
    matrix = _workflow("wheels.yml")["jobs"]["wheel-install"]["strategy"]["matrix"]
    ci = _workflow("ci.yml")["jobs"]

    assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest", "windows-latest"}
    assert {"3.11", "3.12", "3.13"} <= set(matrix["python-version"])
    assert "pixi run pytest" in yaml.safe_dump(ci["test-fast"])


def test_the_audit_task_is_wired_into_pixi() -> None:
    with open(REPO / "pixi.toml", "rb") as handle:
        tasks = tomllib.load(handle)["tasks"]

    assert tasks["audit"] == "python scripts/dependency_audit.py"
    assert tasks["sbom"] == "python scripts/generate_sbom.py"


@pytest.mark.parametrize("script", ["dependency_audit.py", "generate_sbom.py", "security_gate.py"])
def test_each_tool_runs_from_a_clean_interpreter(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
