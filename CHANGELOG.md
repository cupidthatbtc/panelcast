# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-19

First release under the **panelcast** name. The project was previously developed
as an Album of the Year (AOTY) score predictor; this release presents it as the
general, domain-agnostic tool it had already become, with the AOTY model kept as
the flagship example domain.

### Added

- **YAML descriptor system.** Every dataset-specific name (columns, target
  bounds, date formats, posterior-site prefixes, feature blocks) flows through a
  single `DatasetDescriptor`, with a default-equals-AOTY contract. New domains
  run with zero source changes.
- **Worked aerospace example** (`configs/datasets/aero.yaml`) plus an
  end-to-end domain-portability test that proves `--dataset aoty_full` is
  byte-identical to the built-in defaults.
- **Optional per-entity overdispersion** with a lognormal variance prior, behind
  a gate, alongside the A/B (bake-off) harness and decision docs that evaluated
  it.
- **Porting guide** (`docs/PORTING.md`) and extensibility guide documenting how
  to retarget the pipeline.
- Community and packaging files: `CONTRIBUTING.md`, `CITATION.cff`, this
  changelog, issue/PR templates, and generalized project metadata.

### Changed

- **Renamed** the package `aoty_pred` → `panelcast`, the CLI `aoty-pipeline` →
  `panelcast`, and the distribution `aoty-pred` → `panelcast`. The AOTY domain
  references (columns, descriptors, feature packs) are unchanged — AOTY is now
  the flagship example, not the tool's identity.
- Generalized the README into a tool front page with a domains table, and
  generalized the package description and metadata.

### Notes

- The 4×5000 publication-configuration run has been executed on a ~5k-album AOTY
  **subset** (R-hat 1.00, bulk ESS 3,134, 0 divergences); the `MODEL_CARD.md`
  numbers come from that subset. The full-corpus run (`configs/publication.yaml`
  over all ~62k albums) remains the open item.
