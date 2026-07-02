# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-07-02

A modeling-decision release: the `offset_logit` target transform is promoted to
the default on corrected held-out evidence, the estimator that evidence rests
on is replaced by a direct held-out lppd, and two gated modeling experiments
land with measured verdicts. Every verdict in the decision journal was
re-audited against the corrected estimator (#78) — all survive.

### Changed

- **`offset_logit` is the default target transform** (#43). The original
  PSIS-LOO evidence was invalidated by the estimator fix below; the corrected
  paired advantage over identity is +22.2 ± 4.5 held-out ELPD (z ≈ +4.9),
  stable across seeds 42/43/44, and the promotion was re-baselined with a
  fresh publication-scale fit that passes the convergence gate (R-hat 1.00,
  bulk ESS 2,333, 0 divergences). Published numbers move: R² 0.417 → 0.429,
  CRPS 4.19 → 4.13, PPC pins reduced from seven to four, and **cold-start
  flips from the model's weakest split to its best** — MAE 7.01 / R² 0.095
  leads every baseline with at-nominal coverage and properly widened
  intervals. `--target-transform identity` restores the old behavior; the
  direct-constructor `PriorConfig` default stays `identity` for parity. Model
  card, README, and BASELINES.md regenerated from the new baseline
  (`.audit/baseline_metrics.json`).
- **Held-out ELPD replaces PSIS-LOO/WAIC in the evaluate stage** (#63). The
  old path misused PSIS-LOO on held-out data (uniform Pareto-k > 1.14,
  invalid numbers like "LOO +60.1"). `info_criteria` now reports
  `heldout_elpd` — a direct test-set lppd with an analytic SE — and the
  bake-off reassembly recomputes pairwise ELPD from persisted per-point
  log-likelihood snapshots.

### Added

- **Genre pooling gate** (`entity_group_pooling`, default off; #41).
  Hierarchical partial pooling of new-entity intercepts toward genre-group
  means. Measured verdict (paired 2×2000 screening): improves every
  cold-start metric at once — MAE −0.14, R² +0.034, CRPS −0.11 at unchanged
  nominal coverage — capturing essentially the entire genre-block headroom
  the covariate diagnostic predicted. Default-on promotion deferred to a
  publication-scale bake-off (#75).
- **`beta_ceiling` likelihood family** (#42). A scaled Beta on the occupied
  score range, probing whether the bounded-skew PPC pins are a support
  artifact. Verdict: they are not — the upper-tail pins do not move and the
  lower tail re-pins (six pinned stats vs the default's four); the ceiling
  does fix plain `beta`'s mixing pathology. Available, not adopted.
- **Conformal GBM baseline** (#50): split-conformal calibration around the
  GBM point forecast, sharpening the honest value proposition in
  BASELINES.md.
- **Multi-seed decision check** (#40): the transform ledger is stable across
  seeds 42/43/44.
- **Point-accuracy gap ablation** (`.audit/point_accuracy_gap/`): the
  entity effects already extract the user-history signal in full (GBM on
  history-only ≈ the model); the remaining GBM edge is covariate conversion
  and nonlinearity — scoped for 0.6.0 (#76).

### Fixed

- **Test-safety machinery** (#68): `pytest-timeout` as a real dev dependency,
  `--strict-markers` with a registered `timeout` marker, the likelihood-parity
  golden fails loudly when missing, CI/nightly job timeouts.
- n_exponent re-run at publication scale confirms ≈ 0 (#44).
- Two flip-collision test fixes: the gate-recording coverage test and the
  discretization command-string test both assumed the identity default and
  broke when #43 landed alongside them.
- Stale pre-correction numbers scrubbed from the decision journal and model
  card (the invalid "ELPD (LOO-CV): −28264.9" headline; the ar1 −5.2 ± 2.6 →
  −2.2 ± 0.93), backed by the verdict evidence-chain audit on #78.
- Docs corrections: the lineage doc names the current `filter_for_target_model`
  API; `REVIEW_RESPONSE.md` carries a superseded-history banner; every
  output-affecting gate is recorded in the manifest command string.

### Notes

- Full-corpus validation remains out of scope and GPU-bound (#15). The 0.6.0
  candidate list lives in #75; the model-selection protocol sketch in #78.

## [0.4.0] — 2026-06-28

A backward-compatible modeling and code-quality release. Both new model-v2
options are **default-off and parity-locked** — a unit test pins the posterior
draws bit-identical with the gates off — so existing runs and the published
model-card numbers are unchanged; everything new is opt-in.

### Added

- **Errors-in-variables AR(1) gate** (`errors_in_variables`, default off; #30).
  The album-to-album term regresses on the *observed* previous score as if it
  were noise-free, attenuating `rho` toward zero for sparse-review entities. The
  gate de-noises the regressor with a fixed, data-derived measurement-error
  latent (`prev_latent = prev_score + (global_std / √prev_n_reviews)·z`, debuts
  pinned to zero) rather than a second latent AR, so it adds no new funnel
  geometry. Synthetic-recovery tests confirm `rho` de-attenuates with the gate on.
- **Long-horizon random-walk variance gate** (`propagate_rw_horizon`, default
  off; #30). Prediction past the longest training trajectory reused the final
  latent step and dropped the accumulated random-walk variance, so
  deep-extrapolation intervals were too narrow. With the gate on, the
  evaluate/predict stages remove the sequence clamp and the re-sampled trajectory
  carries the full innovations; training and within-horizon draws are unchanged.
- **Baseline comparison** (`docs/BASELINES.md`), reframing the model as a
  calibrated-uncertainty engine rather than a point forecaster — competitive with
  ridge, behind gradient boosting on point accuracy, and better calibrated than
  GBM (#39).
- A **C901 cyclomatic-complexity gate** (max 22) and a `@claude` GitHub Action
  workflow.

### Changed

- **Code-quality remediation** (#14): decomposed the publication-artifact and
  evaluate spines into focused helpers, de-nested the known-entity prediction
  path, split run-command validation and preflight, modernized annotations and
  stdlib idioms, and made the best-effort handlers record exception type and
  traceback. Behavior-preserving.
- Tightened the portability claim across the docs — the harness ports to a new
  domain with no source changes, but predictive accuracy off the AOTY example is
  untested by construction (#48).
- Surfaced LOO (elpd + SE) and pairwise ELPD in the transform × latent bake-off,
  and corrected the `offset_logit` and AR(1) verdicts in the decision journal
  (#34, #35).

### Fixed

- `resume` now restores `chain_method` — it was missing from `RESUME_CONFIG_KEYS`,
  so a resumed run silently reverted to the default (#36).
- The RW-horizon clamp count excludes below-`min_albums` entities that were
  inflating it (#37).

### Notes

- The v1-vs-v2 subset bake-off finds **both gates immaterial on the ~5k-album
  subset** — LOO moves within its SE and every point/calibration metric is
  unchanged, consistent with the `n_exponent ≈ 0` result. They ship default-off;
  their value is only measurable at full-corpus scale and longer horizons (#49,
  [`.audit/model_v2_bakeoff/comparison.md`](.audit/model_v2_bakeoff/comparison.md)).
- Full-corpus (~62k-album) publication-scale validation remains out of scope and
  GPU-bound (#15).

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
