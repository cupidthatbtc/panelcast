"""Emit CycloneDX 1.6 SBOMs for the two things a release actually ships (#372).

    python scripts/generate_sbom.py --scope environment --output env.cdx.json
    python scripts/generate_sbom.py --scope wheel --python .venv/bin/python --output whl.cdx.json

They describe different software and are not substitutes for one another:

**environment** is `pixi.lock`: every conda and PyPI artifact, on all three
locked platforms, including the test, lint, and plotting toolchain. It is the
environment the results were produced in. It is *not* what `pip install
panelcast` gives anyone. It is a pure function of the lock -- no timestamp, and
a serial number derived from the lock digest -- so the same lock always produces
byte-identical output and two releases can be diffed directly.

**wheel** is the runtime closure of the built wheel: the distributions
`importlib.metadata` reports in an interpreter where only that wheel was
installed. That is a point-in-time pip resolution for one platform and one
Python version, not a lock, and it is recorded as such in the document. The
venv's own bootstrap (pip, setuptools, wheel) is marked so it is not mistaken
for a dependency of panelcast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
import uuid
from collections import defaultdict
from pathlib import Path

import pixi_lock
from pixi_lock import Lock

SPEC_VERSION = "1.6"
SERIAL_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # RFC 4122 URL namespace
BOOTSTRAP = {"pip", "setuptools", "wheel", "pkg-resources"}

ENVIRONMENT_NOTE = (
    "Every artifact pixi.lock pins, across all locked platforms, including the "
    "development and plotting toolchain. This is not the dependency closure of "
    "the published wheel; see the wheel-runtime SBOM for that."
)
WHEEL_NOTE = (
    "The distributions installed alongside the panelcast wheel in a single "
    "interpreter. A pip resolution for one platform and one Python version at "
    "build time, not a lock: another install may resolve different versions."
)

# Read in the target interpreter. `packaging` is a panelcast runtime dependency,
# so requirement markers are evaluated under the environment the SBOM describes.
INSTALLED_QUERY = """
import json, platform, sys, sysconfig
from importlib.metadata import distributions
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
seen = {}
requires = {}
for dist in distributions():
    name = canonicalize_name((dist.metadata["Name"] or "").strip())
    if not name:
        continue
    seen[name] = dist.version
    dependencies = []
    for raw in dist.requires or []:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate():
            dependencies.append(canonicalize_name(requirement.name))
    requires[name] = sorted(set(dependencies))
print(json.dumps({
    "distributions": seen,
    "requires": requires,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "platform_tag": sysconfig.get_platform(),
    "executable": sys.executable,
}))
"""


def _project_version() -> str:
    with open(pixi_lock.REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _document(version: str, scope: str, note: str, serial: str, properties: list[dict]) -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": f"pkg:pypi/panelcast@{version}",
                "name": "panelcast",
                "version": version,
                "purl": f"pkg:pypi/panelcast@{version}",
            },
            "tools": {
                "components": [
                    {"type": "application", "name": "panelcast generate_sbom", "version": version}
                ]
            },
            "properties": [
                {"name": "panelcast:scope", "value": scope},
                {"name": "panelcast:scope-note", "value": note},
                *properties,
            ],
        },
        "components": [],
    }


def build_environment(lock: Lock, version: str) -> dict:
    platforms: dict[str, set[str]] = defaultdict(set)
    for (environment, platform), packages in lock.platform_packages().items():
        for package in packages:
            platforms[package.url].add(f"{environment}/{platform}")

    components = []
    for package in sorted(lock.packages, key=lambda p: (p.kind, p.name, p.version, p.url)):
        component = {
            "type": "library",
            # The artifact URL, not the purl: several platforms ship the same
            # name@version as separate wheels, and bom-ref has to stay unique.
            "bom-ref": package.url,
            "name": package.name,
            "version": package.version,
            "purl": package.purl,
            "externalReferences": [{"type": "distribution", "url": package.url}],
            "properties": [
                {"name": "panelcast:source", "value": package.kind},
                {"name": "panelcast:platforms", "value": " ".join(sorted(platforms[package.url]))},
            ],
        }
        if package.sha256:
            component["hashes"] = [{"alg": "SHA-256", "content": package.sha256}]
        components.append(component)

    document = _document(
        version,
        "pixi-environment",
        ENVIRONMENT_NOTE,
        str(uuid.uuid5(SERIAL_NAMESPACE, lock.digest)),
        [{"name": "panelcast:pixi-lock-sha256", "value": lock.digest}],
    )
    document["components"] = components
    return document


class NotTheWheelEnvironment(RuntimeError):
    """The interpreter given does not hold the wheel this SBOM claims to describe."""


def build_wheel_runtime(installed: dict, version: str) -> dict:
    distributions: dict[str, str] = installed["distributions"]
    # Without this an SBOM labelled "wheel-runtime" could quietly describe any
    # interpreter at all -- a development environment, say, whose extra packages
    # would read as dependencies of the published wheel.
    if distributions.get("panelcast") != version:
        raise NotTheWheelEnvironment(
            f"{installed['executable']} has panelcast "
            f"{distributions.get('panelcast', '(not installed)')}, expected {version}"
        )

    components = []
    refs = {name: f"pkg:pypi/{name}@{package_version}" for name, package_version in distributions.items()}
    for name, package_version in sorted(distributions.items()):
        components.append(
            {
                "type": "library",
                "bom-ref": refs[name],
                "name": name,
                "version": package_version,
                "purl": refs[name],
                "properties": [
                    {
                        "name": "panelcast:origin",
                        "value": "venv-bootstrap" if name in BOOTSTRAP else "wheel-closure",
                    }
                ],
            }
        )

    dependency_graph = []
    requires = installed.get("requires") or {}
    for name in sorted(distributions):
        dependency_graph.append(
            {
                "ref": refs[name],
                "dependsOn": sorted(
                    refs[dependency]
                    for dependency in requires.get(name, [])
                    if dependency in refs
                ),
            }
        )

    # Deterministic in the resolution, not in time: the same resolved closure on
    # the same interpreter always yields the same serial number.
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                installed["python_version"],
                installed["platform_tag"],
                sorted(distributions.items()),
                sorted((name, sorted(dependencies)) for name, dependencies in requires.items()),
            ]
        ).encode("utf-8")
    ).hexdigest()

    document = _document(
        version,
        "wheel-runtime",
        WHEEL_NOTE,
        str(uuid.uuid5(SERIAL_NAMESPACE, fingerprint)),
        [
            {"name": "panelcast:python-version", "value": installed["python_version"]},
            {"name": "panelcast:python-implementation", "value": installed["implementation"]},
            {"name": "panelcast:platform-tag", "value": installed["platform_tag"]},
            {"name": "panelcast:resolution-fingerprint", "value": fingerprint},
        ],
    )
    document["components"] = components
    document["dependencies"] = dependency_graph
    return document


def installed_distributions(python: Path) -> dict:
    result = subprocess.run(
        [str(python), "-c", INSTALLED_QUERY],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("environment", "wheel"), default="environment")
    parser.add_argument("--lock", type=Path, default=pixi_lock.DEFAULT_LOCK)
    parser.add_argument("--python", type=Path, help="interpreter to read for --scope wheel")
    parser.add_argument("--output", type=Path, help="write here instead of stdout")
    parser.add_argument(
        "--expect-version",
        help="fail unless the project version is exactly this (the release tag)",
    )
    args = parser.parse_args(argv)

    version = _project_version()
    if args.expect_version and args.expect_version != version:
        print(
            f"refusing to write an SBOM: pyproject says {version}, expected {args.expect_version}",
            file=sys.stderr,
        )
        return 1

    if args.scope == "environment":
        document = build_environment(pixi_lock.parse(args.lock), version)
    else:
        if not args.python:
            parser.error("--scope wheel needs --python pointing at the interpreter to describe")
        try:
            document = build_wheel_runtime(installed_distributions(args.python), version)
        except NotTheWheelEnvironment as error:
            print(f"refusing to write a wheel-runtime SBOM: {error}", file=sys.stderr)
            return 1

    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output} ({len(document['components'])} components, {args.scope})")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
