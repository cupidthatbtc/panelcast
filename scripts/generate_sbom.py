"""Emit a CycloneDX 1.6 SBOM for the environment pixi.lock pins (#372).

    python scripts/generate_sbom.py --output panelcast-sbom.cdx.json

One component per distinct locked artifact, carrying its purl, its SHA-256, and
the platforms it is installed on. The document is a pure function of the lock:
no timestamp, and a serial number derived from the lock digest, so the same lock
always produces byte-identical output and two releases can be diffed directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import uuid
from collections import defaultdict
from pathlib import Path

import pixi_lock
from pixi_lock import Lock

SPEC_VERSION = "1.6"
SERIAL_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # RFC 4122 URL namespace


def _project_version() -> str:
    with open(pixi_lock.REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def build(lock: Lock, version: str) -> dict:
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

    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid5(SERIAL_NAMESPACE, lock.digest)}",
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
            "properties": [{"name": "panelcast:pixi-lock-sha256", "value": lock.digest}],
        },
        "components": components,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=pixi_lock.DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    lock = pixi_lock.parse(args.lock)
    document = json.dumps(build(lock, _project_version()), indent=2, sort_keys=False) + "\n"

    if args.output:
        args.output.write_text(document, encoding="utf-8")
        print(f"wrote {args.output} ({len(lock.packages)} components)")
    else:
        sys.stdout.write(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
