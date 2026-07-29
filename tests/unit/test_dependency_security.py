"""Dependency-security guards for the cross-platform lock (#372).

The floors in `scripts/dependency_audit.py` are the part of the audit that has
to hold without a network: a vulnerable GitPython or orjson must not be able to
re-enter `pixi.lock` on any platform, through either the PyPI or the conda half
of the environment. The synthetic locks below are the adversarial half — a guard
that has never been shown to fail is not a guard.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
import uuid
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

PLATFORMS = {"linux-64", "osx-arm64", "win-64"}
OSV_ID = re.compile(r"^[A-Z][A-Z0-9]*-[\w.-]+$")

# What the lock actually shipped before #372, and what must never come back.
PREVIOUSLY_LOCKED = {"gitpython": "3.1.46", "orjson": "3.11.5"}

PYPI_WHEEL = "https://files.pythonhosted.org/packages/ab/{name}-{version}-py3-none-any.whl"
CONDA_ARTIFACT = "https://conda.anaconda.org/conda-forge/noarch/{name}-{version}-pyh0_0.conda"


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


def _conda_entry(name: str, version: str, pypi_name: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "conda": CONDA_ARTIFACT.format(name=name, version=version),
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


def test_floors_match_the_advisories_that_set_them() -> None:
    assert dependency_audit.MINIMUM_VERSIONS == {"gitpython": "3.1.55", "orjson": "3.11.6"}


def test_locked_environment_clears_every_floor(lock: pixi_lock.Lock) -> None:
    assert dependency_audit.check_minimums(lock) == []


@pytest.mark.parametrize("name", sorted(dependency_audit.MINIMUM_VERSIONS))
def test_every_platform_actually_pins_the_guarded_package(lock: pixi_lock.Lock, name: str) -> None:
    floor = Version(dependency_audit.MINIMUM_VERSIONS[name])
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
    constraint = SpecifierSet(manifest["pypi-dependencies"][name])

    assert Version(dependency_audit.MINIMUM_VERSIONS[name]) in constraint
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


def test_baseline_is_a_readable_ledger_of_accepted_findings() -> None:
    document = json.loads((REPO / "security_baseline.json").read_text(encoding="utf-8"))
    ids = [i for entries in document["packages"].values() for i in entries]

    assert document["total_advisories"] == len(ids)
    assert all(OSV_ID.fullmatch(i) for i in ids)
    assert "triaged" in document["description"]
    # A floor package must never be parked in the baseline instead of upgraded.
    assert not set(document["packages"]) & set(dependency_audit.MINIMUM_VERSIONS)


def test_new_advisory_ids_fail_the_gate_and_baselined_ones_do_not() -> None:
    finding = dependency_audit.Finding("pillow", "12.1.0", "GHSA-fake-0000-0000")
    unknown = dependency_audit.Report(new_pypi=[finding])
    known = dependency_audit.Report(pypi=[finding])

    assert unknown.failed(strict_conda=False)
    assert not known.failed(strict_conda=False)
    assert dependency_audit.Report(minimums=["anything"]).failed(strict_conda=False)


def test_offline_gate_passes_and_says_it_did_not_scan(capsys: pytest.CaptureFixture[str]) -> None:
    assert dependency_audit.main(["--offline"]) == 0
    # "0 findings" from a run that never asked is the one report worth banning.
    assert "OSV was not queried" in capsys.readouterr().out


def test_update_cannot_wipe_the_baseline_from_an_offline_run() -> None:
    # An offline run has no findings to write, so --update would empty the file.
    with pytest.raises(SystemExit):
        dependency_audit.main(["--update", "--offline"])


def test_native_findings_gate_only_under_strict_conda() -> None:
    report = dependency_audit.Report(
        new_native=[dependency_audit.Finding("zlib", "1.3.1", "DEBIAN-CVE-2016-9840")]
    )

    assert not report.failed(strict_conda=False)
    assert report.failed(strict_conda=True)


def test_sbom_describes_every_locked_artifact(lock: pixi_lock.Lock) -> None:
    sbom = generate_sbom.build(lock, "9.9.9")
    components = sbom["components"]

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == generate_sbom.SPEC_VERSION
    assert len(components) == len(lock.packages)
    assert len({c["bom-ref"] for c in components}) == len(components)
    assert sbom["metadata"]["component"]["version"] == "9.9.9"
    assert sbom["metadata"]["properties"] == [
        {"name": "panelcast:pixi-lock-sha256", "value": lock.digest}
    ]
    expected = uuid.uuid5(generate_sbom.SERIAL_NAMESPACE, lock.digest)
    assert sbom["serialNumber"] == f"urn:uuid:{expected}"


def test_sbom_purls_identify_both_halves_of_the_environment(lock: pixi_lock.Lock) -> None:
    purls = {c["purl"] for c in generate_sbom.build(lock, "9.9.9")["components"]}

    assert "pkg:pypi/gitpython@3.1.57" in purls
    conda = [p for p in purls if p.startswith("pkg:conda/")]
    assert conda
    # Without subdir and build the conda purls collapse across platforms.
    assert all("channel=" in p and "subdir=" in p and "build=" in p for p in conda)


def test_sbom_is_a_pure_function_of_the_lock(lock: pixi_lock.Lock) -> None:
    first = json.dumps(generate_sbom.build(lock, "1.2.3"), sort_keys=False)
    second = json.dumps(generate_sbom.build(lock, "1.2.3"), sort_keys=False)

    assert first == second


def _workflow(name: str) -> dict[str, Any]:
    path = REPO / ".github" / "workflows" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps") or []


def step_uses(step: dict[str, Any]) -> str:
    return str(step.get("uses", ""))


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


def test_release_retains_an_sbom_without_pushing_it_to_pypi() -> None:
    jobs = _workflow("release.yml")["jobs"]

    generate = next(
        step for step in _steps(jobs["build"]) if "generate_sbom.py" in step.get("run", "")
    )
    assert "sbom/panelcast-" in generate["run"]
    assert "dist" not in generate["run"], "anything under dist/ is uploaded to PyPI"

    upload = next(
        step for step in _steps(jobs["build"]) if (step.get("with") or {}).get("path") == "sbom/"
    )
    assert "upload-artifact" in step_uses(upload)
    assert upload["with"]["if-no-files-found"] == "error"
    assert int(upload["with"]["retention-days"]) >= 30

    assert jobs["sbom"]["permissions"] == {"contents": "write"}
    assert "gh release upload" in yaml.safe_dump(jobs["sbom"], sort_keys=True)
