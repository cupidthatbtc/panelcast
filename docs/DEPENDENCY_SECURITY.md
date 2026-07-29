# Dependency security

`pixi.lock` is the authoritative environment, so it is the thing that gets
audited — but it is not the only thing that ships. A `pip install panelcast`
resolves its own closure from `pyproject.toml`, and that closure is audited
separately. Both are described by their own SBOM.

| Command | What it does |
| --- | --- |
| `pixi run audit` | Queries OSV for every locked package and gates on the result |
| `pixi run audit --offline` | Checks the declared version floors only, no network |
| `pixi run audit --scaffold` | Stubs ledger entries for findings a human still has to triage |
| `pixi run audit --requirements pins.txt` | Exports the lock's PyPI pins for `pip-audit` |
| `pixi run sbom --scope environment --output env.cdx.json` | CycloneDX 1.6 SBOM of the lock |
| `pixi run sbom --scope wheel --python .venv/bin/python --output whl.cdx.json` | CycloneDX 1.6 SBOM of an installed wheel's closure |

`.github/workflows/security.yml` runs all of it on every pull request, on pushes
to `main`, and weekly on a schedule — advisories are published long after a lock
is written, so the gate cannot only fire when someone edits a dependency.

## What each scanner can actually see

**PyPI-identified — gated.** Every `pypi:` entry in the lock, plus every conda
package carrying a `pkg:pypi/...` purl, is queried against OSV's PyPI ecosystem.
Those records describe exactly the distribution the lock pins, so a match that is
not an explicit, unexpired acceptance fails the audit. This is what reaches the
conda half: nearly the whole scientific stack installs from conda-forge, and
pixi records the PyPI identity of each Python package it resolves there.

Two limits are worth stating rather than assuming away:

- pixi does not write the `purls` field consistently. conda-forge's `pyarrow`
  declares `pkg:pypi/pyarrow` on osx-arm64 and an empty `purls` on linux-64 and
  win-64, so the same distribution was PyPI-audited on one platform and silently
  demoted on the other two. `scripts/pixi_lock.py` now propagates a mapping
  across every entry sharing a conda name, and a test pins that behaviour.
- an advisory's affected range is stated for the *PyPI* release. conda-forge
  sometimes patches a build without changing the version, so this tier can
  report a finding that a particular conda build has already fixed. It errs
  toward reporting, which is the right direction for a gate.
- the interpreter is not in this tier at all. There is no `pkg:pypi/python`, so
  CPython itself only ever appears in the name-matched tier below, and it is not
  gated. What covers it is conda-forge tracking upstream releases and the lock
  pinning a current one (3.14.2 today).

**pip-audit over the locked pins — gated.** A second, independent PyPI-aware
opinion over the same wheels, from a different advisory pipeline. It reads the
exported `name==version` pins with `--no-deps --strict`, so it audits what the
lock pins rather than re-resolving.

**pip-audit over the wheel's runtime closure — gated.** The lock is not what a
PyPI user installs. CI builds the wheel, installs it into a clean virtualenv,
and audits *that* closure. Without this step nothing in the repository ever
scanned the dependency set the published package actually resolves.

**Name-matched across ecosystems — reported, never gated.** Conda packages with
no PyPI identity are C libraries: openssl, zlib, libpng. OSV has no conda
ecosystem, so the only available handle is a bare-name lookup across every
ecosystem at once. The current lock produces ~1250 such matches across ~49
packages, and they are not evidence about this environment:

- the affected/fixed ranges belong to Debian, Ubuntu, Alpine, or SUSE builds and
  say nothing about what conda-forge compiled, patched, or backported;
- bare names collide outright — `seaborn` matches a malicious-npm-package
  advisory, `yaml` an npm CVE, `cpython` a RUSTSEC advisory;
- the volume is dominated by historical records: openssl (313) and hdf5 (281)
  are close to half of it between them.

Gating on that tier would be a claim of coverage the data cannot support, so the
audit reports it, labels it, and does not fail on it. `--strict-conda` gates on
it locally for anyone who wants to read it as a gate, and the report is written
into the CI evidence artifact either way.

Be clear about what that leaves: **the native tier is not adjudicated, and no
tool here can adjudicate it.** What the project does instead is keep conda-forge
current — the July 2026 re-solve moved libpng, libjpeg-turbo, freetype, krb5,
libcurl, libglib, openldap, lcms2, and zlib to their newest conda-forge builds,
and a full re-solve leaves openssl, hdf5, and libssh2 where they are, which is
the evidence that those are already current. That is currency, not analysis, and
the weekly workflow audits the lock rather than re-solving it: moving native
libraries is a deliberate lock refresh someone has to run and review.

## Floors and the acceptance ledger

Two separate mechanisms, deliberately:

- `MINIMUM_VERSIONS` in `scripts/dependency_audit.py` are hard floors, each
  carrying the advisories that set it and a sentence on how the package is
  reached from this codebase. They are enforced offline, on every locked
  platform, and through either half of the environment — switching a package
  from PyPI to conda-forge does not dodge them
  (`tests/unit/test_dependency_security.py` proves it). `pixi.toml` carries the
  matching constraint so a re-solve cannot reintroduce a vulnerable version.
- `security_baseline.json` is the ledger of *accepted* findings. It is not a
  list of ids. Every entry names its scope (`lock` or `wheel-runtime`), package,
  exact version, advisory, applicability, remediation, owner, review date, and
  an expiry no more than 90 days out. The audit rejects entries that are
  incomplete, generic, or expired — an acceptance is a decision with a
  deadline, not a parking space. Scope is part of the key, so an exception for
  the installed wheel cannot suppress a finding in the lock audit or vice versa.

`--scaffold` writes lock-scoped entries in the shape a human has to fill in, and
the run still fails: the tool cannot grant an acceptance. An acceptance is bound
to the exact version it was written against, so a version bump has to be
re-adjudicated rather than inheriting the old decision. Wheel-runtime findings
also fail every pull request until the dependency is upgraded or a complete,
expiring `wheel-runtime` acceptance is added manually.

The July 2026 sweep found no acceptances to write. All 67 advisories that the
previous baseline listed had a fixed release available, so all 67 were
remediated by upgrade — click, Pillow, pip, pyarrow, Pygments, pytest,
setuptools, and Tornado moved, the floors hold them, and the ledger is empty.

## SBOMs

`scripts/generate_sbom.py` emits CycloneDX 1.6 JSON in two scopes, and they are
not interchangeable:

- `--scope environment` is `pixi.lock`: one component per locked artifact, with
  its purl, SHA-256, distribution URL, and the platforms it installs on, across
  all three locked platforms and including the test and plotting toolchain. It
  is a pure function of the lock — no timestamp, serial number derived from the
  lock digest — so the same lock always produces byte-identical output and two
  releases can be diffed directly. It is *not* the dependency set of the
  published wheel.
- `--scope wheel` is the runtime closure of the built wheel: the distributions
  `importlib.metadata` reports in an interpreter where only that wheel was
  installed, with the virtualenv's own bootstrap marked as such. Its CycloneDX
  dependency graph is derived from the active `Requires-Dist` metadata under
  that interpreter's markers. This is a pip resolution for one platform and one
  Python version at build time, not a lock, and the document says so.

Each document declares its scope in `metadata.properties` under
`panelcast:scope`, and `scripts/security_gate.py` refuses one that does not.

The release workflow builds both from the tagged lock and the smoke-tested
wheel, attaches them to a draft GitHub Release for the tag, and then downloads
them back and compares them byte for byte before PyPI publication is allowed to
start. The GitHub Release remains a draft through that publication and is made
public only after PyPI succeeds. If any earlier step fails, the permanent assets
remain on the draft and no public release is announced. The run artifact is a
90-day convenience copy, not the record.

## When a scanner fails

Every scanner and SBOM step in `security.yml` runs to completion and records its
own outcome; none of them stops the others. A single aggregate step,
`scripts/security_gate.py`, then fails the job if any scanner errored, any
finding lacks a current acceptance, or any evidence file is missing, empty, or
mislabelled. "No findings" from a scanner that never ran is treated as a
failure, not a pass.
