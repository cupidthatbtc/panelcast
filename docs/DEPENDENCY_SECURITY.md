# Dependency security

`pixi.lock` is the authoritative environment, so it is also the thing that gets
audited. Two tools read it, both driven from the lock alone:

| Command | What it does |
| --- | --- |
| `pixi run audit` | Queries OSV for every locked package and gates on the result |
| `pixi run audit --offline` | Checks the declared version floors only, no network |
| `pixi run audit --update` | Rewrites `security_baseline.json` from the current scan |
| `pixi run audit --requirements pins.txt` | Exports the lock's PyPI pins for `pip-audit` |
| `pixi run sbom --output panelcast-sbom.cdx.json` | Writes a CycloneDX 1.6 SBOM |

`.github/workflows/security.yml` runs all of it on every pull request, on pushes
to `main`, and weekly on a schedule — advisories are published long after a lock
is written, so the gate cannot only fire when someone edits a dependency.

## The two halves, and why they are gated differently

**PyPI-aware.** Every `pypi:` entry in the lock, plus every conda package pixi
maps to a `pkg:pypi/...` purl, is queried against OSV's PyPI ecosystem. Those
records describe exactly the distribution the lock pins. Anything new here fails
the audit. The purl mapping is what makes this conda-aware where it counts:
almost the whole scientific stack installs from conda-forge, not from PyPI, and
this is what reaches it. `pip-audit` runs in CI as a second, independent opinion
over the `pypi:` half.

**Native conda.** Conda packages with no PyPI mapping are C libraries — zlib,
openssl, libpng. OSV has no conda ecosystem, so they are matched by name across
every ecosystem, which surfaces the Debian, Ubuntu, Alpine, and SUSE advisories
tracking the same upstream source. This tier is reported, not gated, and the
reasons are worth stating plainly:

- A distro's affected/fixed version range describes *that distro's* package. It
  says nothing about what conda-forge built, patched, or backported, so a match
  is a prompt to investigate rather than a finding.
- A bare name collides across ecosystems. The scan matches an npm package called
  `seaborn` and a Ruby `zlib` gem that have nothing to do with this environment.
- The volume is high for the same reason: openssl and libpng carry hundreds of
  historical distro advisories between them.

`--strict-conda` gates on this tier for anyone who wants the stricter reading;
`--strict-conda --update` baselines it first.

## Floors and the baseline

Two separate mechanisms, deliberately:

- `MINIMUM_VERSIONS` in `scripts/dependency_audit.py` are hard floors. They are
  enforced offline, on every locked platform, and through either half of the
  environment — switching a package from PyPI to conda-forge does not dodge them
  (`tests/unit/test_dependency_security.py` proves it). `pixi.toml` carries the
  matching constraint so a re-solve cannot reintroduce a vulnerable version.
- `security_baseline.json` is a ratchet, the same shape as the typing and
  terminology baselines. It lists advisory ids that are already known against the
  currently pinned builds. New ids fail; ids that disappear are reported so the
  file can be tightened with `--update`.

Listing an id in the baseline means it has been seen and triaged, **not** that no
fixed release exists. Most of the current entries sit in the pinned scientific
stack (Pillow, Tornado, pyarrow, and friends), where moving a version is an
environment refresh that has to clear the full test suite — a separate change
from a security floor, and one that should not be smuggled into an unrelated PR.

## SBOM

`scripts/generate_sbom.py` emits CycloneDX 1.6 JSON: one component per locked
artifact, with its purl, its SHA-256, the distribution URL, and the platforms it
installs on. The document is a pure function of the lock — no timestamp, and a
serial number derived from the lock digest — so the same lock always produces
byte-identical output and two releases can be diffed directly.

The release workflow builds it from the tagged lock, keeps it as a 90-day run
artifact, and attaches it to the GitHub Release. The tag is pushed before the
release is written, so the attach step waits a bounded five minutes for the
release to appear; if it never does, the run artifact is the copy of record.
