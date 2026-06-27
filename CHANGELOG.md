# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-06-27

### Removed

- **BREAKING: legacy `next_album_*` prediction artifacts.** The predict stage
  writes only the generic `next_event_known_entities.csv` /
  `next_event_new_entity.csv` (columns `entity` / `n_training_events`, scenario
  `entity_mean`); the dual-written AOTY-named copies are gone. Consumers that
  read `next_album_known_artists.csv` / `next_album_new_artist.csv` must switch
  to the generic names and the `entity` / `n_training_events` columns.

### Changed

- **BREAKING: AOTY-internal symbols renamed to entity/event terms.**
  `models.bayes.predict_new_artist` → `predict_new_entity` (and its result type
  `NewArtistPrediction` → `NewEntityPrediction`); the predict pipeline stage
  `predict_next_albums` → `predict_next_events`, with the internal helpers
  `_predict_known_artists` / `_predict_new_artists` →
  `_predict_known_entities` / `_predict_new_entities`; and the single-entity
  convenience wrapper `predict_artist_next` → `predict_entity_next`. No
  compatibility aliases — import the new names. Model-parameter site names
  (`mu_artist`, `artist_idx`, …) are intentionally unchanged.
- Generalized the remaining AOTY-flavored docs to entity/event language
  (`PIPELINE_PLAN.md`, `GLOSSARY.md`, `EVALUATION_PROTOCOL.md`).
- The prediction explorer script reads its column names and model prefix from the
  trained descriptor and is renamed `scripts/predict_artist.py` →
  `scripts/predict_entity.py`.

### Notes

- Full-corpus (~62k-album) publication-scale validation is **not** included; it
  needs more GPU than is available locally (>24 GB even with
  `--exclude-rw-raw-from-collection`) and is tracked separately (#15).

## [0.2.0] — 2026-06-25

A backward-compatible feature and fix release. `studentt` stays the default
likelihood and a parity test pins the original families bit-identical, so
existing runs are unchanged; everything new is opt-in.

### Added

- **Likelihood registry.** Observation families are defined once in a
  `REGISTRY` (`models/bayes/likelihoods.py`) and resolved by name across the
  model and the cold-start prediction path; adding a family is a single entry.
- **Opt-in likelihood families** via `--likelihood-family`: `skew_studentt`,
  `beta`, `skew_normal`, `split_normal`, `beta_binomial`, and a two-component
  `mixture`.
- **`--discretize-observation`** — integer-aware dequantization for honest PPC
  on integer-valued scores (replaces the diverging interval-CDF; #4).
- **New CLI commands** — `diagnose` (model health), `compare --baselines`
  (benchmark vs. the five non-Bayesian baselines on the real splits), and
  `demo` (tiny synthetic end-to-end).
- **`--preset`** — named configuration bundles (e.g. diagnostic, publication)
  on `run` and the `stage` subcommands.
- **Generic prediction artifacts** — `next_event_known_entities.csv` /
  `next_event_new_entity.csv`.
- **Baseline benchmark** — five non-Bayesian baselines scored through the same
  metrics / calibration / CRPS / PPC toolkit as the model.
- **Tiered CI** — lint and type-check → fast tests with coverage → a PR smoke
  check, with the slow/e2e tiers on nightly.
- **Coverage gate at 95%** (the fast suite sits at ~98%).

### Changed

- `--min-ratings` now defaults from the descriptor's `primary_min_obs`.
- The `stage` subcommands accept `--dataset` / `--config` / `--preset`.
- Split names are entity-prefixed (`within_entity_temporal`, `entity_disjoint`)
  with backward-compatible aliases for legacy artifacts.
- The legacy `next_album_*` prediction files are now dual-written alongside the
  generic artifacts (deprecated; removal in 0.3.0).
- Documentation synced; the Graphviz pipeline-diagram generator removed.

### Fixed

- Cold-start prediction honors the trained likelihood family.
- Dequantization replaces the diverging interval-censored CDF discretization (#4).
- Preflight reads the merged config; `resume` keeps `min_ratings`.
- `last_score` baseline ordering; dependency upper-bound caps; CLI override
  precedence over YAML; assorted test hardening.

### Notes

- The bounded-skew PPC limitation is **confirmed structural** across the five
  families evaluated on real data — none moves the `skewness`/`max` pins (#3
  downgraded, open). See [`docs/LIKELIHOOD_CANDIDATES.md`](docs/LIKELIHOOD_CANDIDATES.md).
- The legacy `next_album_*` artifacts and the split aliases are deprecated;
  removal is tracked for 0.3.0 (#14).

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
