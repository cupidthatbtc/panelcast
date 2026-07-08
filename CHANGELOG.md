# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] — 2026-07-08

Inference performance and robustness (#138). Fits and sweeps now plan their
cost, schedule inside their budget, and survive interruption — plus the
full-repo audit hardening that ran alongside.

### Added

- **Adaptive per-arm select timeouts** (#148): `--arm-timeout auto` sizes each
  arm's kill threshold from its own predicted runtime, so slow-but-legitimate
  arms (offset_logit-retaining) can finish and become champions instead of
  being structurally excluded by the fixed 1800s ceiling.
- **Budget lookahead and cost-ordered scheduling** (#166): an arm whose
  predicted cost exceeds the remaining `--budget-hours` is recorded as
  `skipped_budget` (retryable under a bigger budget) instead of truncating the
  stage, and predicted cost breaks ties within each diagnostic-priority group.
- **Predicted train runtime and ETA** (#161): `panelcast run` logs the
  predicted fit duration, ETA, and prediction source before sampling starts,
  records `predicted_seconds` next to the actual in the manifest, and the
  pipeline progress bar advances by predicted per-stage seconds with a
  time-remaining column.
- **Resumable multi-seed confirmation** (#165): `confirmation.json` is now a
  protocol-echoing ledger; a re-entry reuses any prior seed whose snapshots
  survive (re-pairing is cheap), refits only what is missing, and archives the
  ledger on any protocol change.
- **MCMC checkpoint and resume** (#177): `--checkpoint-every N` samples in
  blocks through `post_warmup_state`, persisting draws + sampler state after
  each block; an interrupted fit resumes from the last block. Blocked draws
  are bit-identical to the single-shot chain (parity-tested); mismatched
  checkpoints refuse loudly.
- **Memory-informed auto chain_method** (#176): `--chain-method auto` runs
  NumPyro's vectorized chains when the memory estimator says all chains fit in
  free VRAM, sequential otherwise (always sequential on CPU). Opt-in only;
  runtime telemetry is keyed by chain method. The measured vectorized memory
  ladder and speedup bake-off remain on #138 before docs recommend it.
- **Warm-started sweep arms** (#178): `panelcast select --warmup-transfer`
  reuses the reference fit's adapted mass matrix for screening arms at reduced
  warmup (exact latent-signature match only; confirmation always runs cold).
  The subset z-stability bake-off remains on #138 before docs recommend it.
- **Persistent JAX compilation cache** (#141): sweep arms, preflight
  mini-runs, and confirmation fits reuse compiled XLA programs across
  processes instead of recompiling in every subprocess.
- **`panelcast runs history`** (#153): longitudinal table of headline metrics
  across runs, grouped into feature-stamp epochs, with drift flags and
  `--tag` for findable release fits.

### Fixed

- Audit-cycle hardening across the whole pipeline (#140–#152): fair baseline
  conditioning (information sets equalized with the model's evaluation
  protocol, ridge features standardized), the training prior-predictive gate
  restored, publication diagnostics reported truthfully with exact fan-chart
  quantiles, ESS gate reconciled to the documented total floor, stage-wise
  runs and the run lifecycle repaired, calibration-store appends hardened and
  gated vector sites modeled in the memory estimate, split/cleaning
  data-integrity guards (undated events excluded from temporal holdout,
  duplicate-key warnings, persisted-form hashing), degenerate paired-elpd
  verdicts made explicit, and dashboard trace/interval rendering fixed.

## [0.7.2] — 2026-07-05

A patch release: the report stage no longer writes into your working tree.

### Fixed

- **`report` overwrote the repo-root `MODEL_CARD.md`** (#135): the stage copied
  its run-scoped model card over the tracked `MODEL_CARD.md` on every run,
  silently clobbering the curated card — a #118-class working-tree leak the
  artifact guard missed because the file lives outside the guarded dirs. The
  card now stays run-scoped under `outputs/<run>/reports/`.

## [0.7.1] — 2026-07-05

A patch release. The genre-pooling gate (`entity_group_pooling`, default-on
since 0.6.0) trained correctly but the predict stage never handed its group
indices to the model, so a standard run trained for hours and then failed at
predict — producing no next-event predictions or figures. This fixes that, plus
the GPU fit-time predictor that had been badly over-estimating.

### Fixed

- **Prediction under genre pooling** (#132): `predict` now resolves
  `group_idx_by_artist` / `n_groups` from the training summary (as `evaluate`
  already did), so a model trained with `entity_group_pooling` no longer raises
  `entity_group_pooling=True requires group_idx_by_artist and n_groups` at
  predict time. Every run on the shipped 0.6.0/0.7.0 defaults was affected.
- **GPU runtime predictor** (#130): the per-machine fit-time estimate is now
  affine and model-aware — a shared startup intercept plus a per-transform rate
  — instead of a naive median rate that a pool of tiny probe fits could inflate
  (it returned ~29 h for a ~1 h fit). Also closes a test-isolation leak that
  wrote fake-peak probe records into the real calibration store.
- **Publication-readiness ESS gate**: the check compared bulk-ESS against
  `ess_threshold * num_chains`, treating the threshold as per-chain, so a healthy
  fit (e.g. 623 bulk-ESS, 4 chains) failed against a phantom 1600 floor. It now
  uses the total floor the evaluate stage applies (`ess_bulk_min >= ess_threshold`).
- **`export-figures` coefficient plot**: the forest plot read a hardcoded `mean`
  column and crashed (`KeyError: 'mean'`) on the report's `Estimate` / `CI Lower`
  / `CI Upper` table; column resolution is now format-agnostic.

## [0.7.0] — 2026-07-05

A model-selection release. `panelcast select` turns the manual, per-domain hunt
for a transform / likelihood / gate configuration into a pre-registered,
budget-aware protocol that any descriptor can run, and the GPU memory it plans
against is recalibrated to measured hardware. Run against the AOTY flagship, the
protocol independently reproduces the 0.6.0 defaults — no candidate beats the
shipped configuration. All selection evidence is committed under `.audit/`.

### Added

- **`panelcast select`** (#78; #98–#103): a portable, staged model-selection
  protocol. The candidate space is enumerated from the code's own registries, so
  a frozen option is genuinely re-tried rather than pre-pruned; a
  prior-predictive screen flags mis-scaled priors before any fit (#99). A
  budget-aware, resumable sweep runner stages each arm through
  splits/features/train/evaluate and skips completed work on resume (#100).
  Paired held-out ELPD (the #63 estimator) with coverage and convergence gates
  ranks arms into one ledger and a `report.md` (#101). Pre-registered decision
  rules — paired-ELPD z ≥ 2, coverage within ±0.03, convergence required — with
  multi-seed confirmation decide promotion; a default flip stays a manual PR
  (#102). The CLI exposes effort tiers (quick/standard/thorough), a pre-run cost
  printout, `--dry-run`, and a new-domain playbook (#103), and the recorded
  selection history is re-verified against the post-#63 estimator (#98).
- **Per-machine GPU self-calibration** (#112): a calibration store and runtime
  predictor learn each machine's memory/time profile, so budget planning
  reflects real hardware instead of a fixed guess.
- **AOTY reproduction evidence** (`.audit/select_aoty/`): `select` swept the
  enumerated space against the flagship and promoted nothing — no arm beats the
  shipped defaults (the closest, disabling `offset_logit`, is −14.9 ± 4.6
  held-out ELPD, z −3.27). The 0.6.0 `gbm_offset` / genre-pooling /
  `offset_logit` decisions hold.

### Changed

- **Honest GPU memory expectations** (#106, #113): the preflight ladder extends
  to the post-0.5.0 model dimensions and the memory estimator is recalibrated
  against a measured ladder, with cold-start calibration pinned by test.
- **Sweep mechanics hardened** (#117, #119): a per-arm wall-clock timeout
  (`--arm-timeout`, default 1800 s) kills a stalled arm instead of hanging the
  sweep; arms screen at 1000 draws and only a candidate that clears the bar is
  confirmed at 5000; a timed-out arm is terminal, not retried.

### Fixed

- Test isolation: every `ArtifactPaths` root is guarded and the tests that wrote
  to the real `data/` dirs are isolated, so a test run can no longer pollute the
  working tree (#118, #120).
- The run seed is restored on resume, so a resumed fit reproduces the original
  posterior (#121).
- The preflight cache is replaced atomically and stale log handlers are closed,
  fixing a Windows file-handle leak (#122).
- Script correctness (#123, #124): the trajectory plot applies the target
  transform (raw scores no longer leak into the AR term; draws are clipped to
  bounds), the preflight experiment extrapolates calibration at post-warmup
  samples, and `predict_entity --robust` no longer crashes when followed by an
  entity name.
- Git provenance survives a non-editable install nested in an unrelated repo,
  and the non-strict `ConvergenceError` path no longer crashes on an unbound
  local (#123, #125).

## [0.6.0] — 2026-07-03

A point-accuracy and run-hygiene release: the model takes the lead on the
baselines table it used to trail (a stacked-GBM covariate promoted on
publication-scale evidence, together with genre pooling), pipeline runs stop
being able to corrupt each other, and the CLI and docs get their
quality-of-life pass. All promotion evidence is committed under `.audit/`.

### Changed

- **`gbm_offset` and genre pooling are default-on** (#85, #86). Both cleared
  the pre-registered protocol — multi-seed screening (42/43/44) plus one
  combined publication confirmation fit: R-hat 1.00, bulk ESS 2,612, 0
  divergences, paired held-out ELPD **+224.2 ± 12.6 (z ≈ +17.9)** over the
  0.5.0 baseline. Published numbers move: within-entity MAE 5.66 → **5.30**,
  R² 0.429 → **0.501**, CRPS 4.13 → 3.87 with *narrower* intervals at nominal
  coverage — the model now leads ridge and the GBM on every point metric
  while keeping calibrated, modeled intervals; cold-start improves to MAE
  6.91 / R² 0.113 and the deep-catalog R² decline disappears (11+ bin 0.33 →
  0.53). `entity_group_pooling` is tri-state: the `null` default resolves to
  on exactly where the descriptor names a usable `entity_group_col`, so
  group-less domains run unchanged; explicit `true`/`false` always wins.
  Feature golden hashes were deliberately regenerated (the block joined the
  default roster). Model card, BASELINES.md, and
  `.audit/baseline_metrics.json` regenerated from the new baseline.
- **Mutable products are run-scoped** (#81; breaking). `models/`,
  `evaluation/`, `predictions/`, and `reports/` write under
  `outputs/<run_id>/`; `data/*` stays flat as the stamped cross-run cache.
  `outputs/latest.json` is the authoritative pointer (atomic, success-only)
  and `compare`/`diagnose`/`dashboard`/`demo` resolve through it with a
  legacy-flat fallback. User scripts move to `outputs/latest/...`.

### Added

- **`gbm_offset` feature block** (#86): a gradient-boosted prediction of the
  target over the other blocks' outputs enters X as one more covariate —
  out-of-fold for train rows so no row sees its own label, full-train model
  for held-out rows, nothing persisted. Cold-start CLI predictions for
  hypothetical entities degrade to the train-mean offset by construction.
- **Staleness stamps** (#82): data stages stamp their product roots with the
  producing run and input hash; `train`/`evaluate`/`predict`/`sensitivity`
  and `compare --baselines` fail fast (exit 7) on artifacts regenerated by
  another run mid-flight instead of silently consuming them.
- **Resource recording** (#78-B1): every fit records expected-vs-actual GPU
  memory and wall clock into the training summary, run manifest, and log
  (first publication datapoint: estimate 8.31 GB vs 7.39 GB peak, ratio
  0.89); the orchestrator records per-stage durations.
- **Run-isolation e2e suite**: disjoint runs leave each other byte-identical,
  `latest.json` tracks the newest success, resume writes into the original
  run dir, skip-existing still reuses the flat data cache.
- **CLI batch** (#83): `compare --metrics PATH`; bad `--config` paths become
  one-line `typer.BadParameter` errors; "no trained model" messages name the
  real directory and the remediating command; `--no-progress` with automatic
  non-TTY detection; shell completion; `stage <name> --dry-run`;
  `panelcast runs list`; `demo --dataset` alias.
- **Negative verdicts committed** (`.audit/point_accuracy_gap/`): dropping
  the `*_prior_*` history columns breaks cold-start calibration (the
  redundancy is conditional on a fitted entity effect — no knob ships), and
  the posterior-median point estimator trades a −0.61 bias for 1.3% MAE
  (the mean stays).

### Fixed

- Sensitivity refits reuse the training summary's effective pooling gate —
  previously a pooled fit's refits silently ran unpooled.
- The report stage no longer self-copies artifacts (`SameFileError`) when
  reports already live in the run directory.
- Docs de-slopped (#84): six process-residue files deleted, decision records
  moved to `docs/decisions/`, lineage flattened to `docs/DATA_LINEAGE.md`
  with a guard test that greps its symbol checklist against `src/` (it
  caught two already-stranded references on arrival), stale pre-#63 LOO
  numbers scrubbed.

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
