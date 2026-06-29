# Decisions To Lock

Purpose
- Make publication-critical defaults explicit.
- Keep defaults synchronized with executable code (`src/panelcast/cli.py`, `src/panelcast/pipelines/*`).
- The publication run itself is fully specified by `configs/publication.yaml`
  (`panelcast run --config configs/publication.yaml --strict`).

Dataset and scope (defaults)
- All domain names flow through the dataset descriptor
  (`src/panelcast/config/descriptor.py`); every default equals the AOTY
  literal it replaced, so `--dataset aoty_full` is byte-identical to no flag
  (verified by `tests/e2e/test_domain_portability.py`).
- Raw dataset path: `${AOTY_DATASET_PATH}`
- Primary target: next album `User Score` (0-100)
- Default split input threshold: `min_ratings = 10`
- Data stage emits user-score datasets at thresholds: `5, 10, 25`
- Canonical column mapping: descriptor `raw_column_map`
  (AOTY default mirrored as `RAW_TO_CANONICAL` in `src/panelcast/data/cleaning.py`)
- Package name stays `panelcast` until after publication (deferred rename:
  artifact paths, the editable install and the paper's repository link all
  pin it; renaming mid-stream risks silent breakage for zero scientific gain).

Splits and leakage controls (defaults)
- Primary split: within-artist temporal holdout (`within_entity_temporal`)
- Secondary split: artist-disjoint cold-start split (`entity_disjoint`)
- Split seed: `42` (split-seed sensitivity axis available via the opt-in
  sensitivity stage, seeds 42/43)
- Temporal split params: `test_albums=1`, `val_albums=0` (default; the CLI
  exposes `--val-albums`), `min_train_albums` CLI default `2`
- Artist-disjoint split params: `test_size=0.15`, `val_size=0.15`
- Validation and test transforms mask held-out score labels before history-based feature transforms
- Primary eval `prev_score` is train-history-only (never uses held-out test labels)
- Artist-disjoint eval `prev_score` is fixed to training global mean for all rows
- Debut `prev_score` fill: `train_mean` (train-split-only mean; the legacy
  pre-split `dataset_stats` source remains available as a rollback gate)

Feature scope (defaults)
- Block composition comes from the descriptor's `feature_blocks` list
  (AOTY default: temporal, album_type, artist_history,
  genre PCA `n_components=10`/`min_genre_count=20`, collaboration).
- Music-specific blocks register through the `aoty` feature pack
  (`src/panelcast/features/packs/aoty.py`); `feature_packs: []` disables them.
- Descriptor PCA is not part of the active default feature pipeline.

Bayesian modeling (adopted gates — validated on 2x500 cheap runs, 2026-06-10)
- Sampler: NumPyro NUTS
- `target_transform = identity` — offset_logit HELD as a deliberate values
  choice, not a failure. The old "failed PPC/PIT/coverage, would not mix (R-hat
  1.27–1.37)" note was a cheap-2×500 artifact, now refuted: the diagnostic
  4×1000 transform×latent bake-off (`docs/LIKELIHOOD_CANDIDATES.md`,
  `.audit/transform_latent_bakeoff/`) shows offset_logit *mixes* (R-hat 1.01,
  0 div, ~10× slower) and in fact *wins every held-out predictive metric* — LOO
  +60.1 (paired dse 4.6, z ≈ 13), R² 0.428 vs 0.417, RMSE 8.19 vs 8.27, CRPS
  4.13 vs 4.19, PIT 0.049 vs 0.056 (at one Pareto-k > 0.7). It is held because it
  does not move the bounded-skew skewness/max/q90 PPC pins — it relieves q50 only
  to newly pin q10 — so the project trades a predictive gain for matching the
  generative distribution. A values choice, not a defect.
- `ar_center = global` — ADOPTED: corr(rho, mu_artist) -0.997 -> +0.016,
  debut AR terms exactly zero, prior predictive flipped to passing.
- `latent_process = rw` — ar1 registered behind a LOO-clear-win gate that has
  not triggered. On the default transform the paired LOO difference is −5.2 ± 2.6
  (z ≈ −2; `.audit/transform_latent_bakeoff/`) — a small, marginally-resolved
  *decrement*, not the clear win the gate requires — while bulk ESS drops 787 →
  635 and one more PPC statistic pins, at identical point accuracy and
  calibration. Held; rw stays default.
- `artist_effect_param = noncentered`, `sigma_artist_prior_type = halfnormal`
  — won the 4-variant mixing bake-off
  (`outputs/experiments/sigma_artist_mixing.json`).
- mu_artist prior auto-locates at the model-scale train mean when AR
  centering is active (`locate_level_prior`).
- Heteroscedastic mode defaults: `n_exponent = 0.0`, `learn_n_exponent =
  false`, learned-prior default `logit-normal`, n_exponent prior mode ~0.10.
- CLI sampling defaults: `num_chains=4`, `num_warmup=1000`,
  `num_samples=1000`, `target_accept=0.90`, `max_tree_depth=10`,
  `max_albums=50`, `min_albums_filter=2`.
- Publication sampling config: `4x5000`, warmup `3000`,
  `target_accept=0.95` (`configs/publication.yaml`).

GPU memory (measured 2026-06-10; outputs/experiments/preflight_validation.json)
- Warmup draws are never stored; peak memory is linear in post-warmup
  samples per chain; sequential chains accumulate collected draws.
- The 4x5000 publication run does NOT fit on a 24 GB GPU with rw_raw
  collection on (~60-90 GiB true peak). It REQUIRES
  `exclude_rw_raw_from_collection: true` (in publication.yaml): chain-aware
  calibrated projection 1.4 GB vs 22.5 GB available (94% headroom, PASS).
  Posterior draws for all other sites are bit-identical (parity-tested).
- Calibration runs at the production chain count with the production
  exclusion gate; its cache key includes the NumPyro version and a
  model-structure signature.

Diagnostics and evaluation (defaults)
- Convergence thresholds:
  - `R-hat <= 1.01`
  - `ESS bulk >= 400` (per chain threshold)
  - no divergences unless `--allow-divergences`
- Calibration checks:
  - intervals: `0.80, 0.95`
  - tolerance: `±0.03`
- Prediction interval exported: `0.95`
- Secondary split evaluation enabled by default
- LOO/WAIC: pointwise, with excluded latents marginalized per posterior draw
  (the same predictive semantics as the rest of the test-set evaluation);
  ELPDs carry the transform Jacobian so they are score-scale comparable.
- Stratified history diagnostics (per training-album-count bin) exported
  with the primary-split metrics.

Strict mode behavior
- `--strict` enables fail-fast behavior for:
  - convergence failures
  - calibration tolerance failures
  - strict raw-schema validation in data stage
  - prior-predictive plausibility check failures
  - missing publication artifacts in report stage
  - prediction/evaluation horizon extrapolation beyond trained sequence support

Reproducibility and artifacts
- Environment lockfile (`pixi.lock`) is required by default (`--allow-unlocked-env` to bypass)
- Run manifests are written under `outputs/<run_id>/manifest.json`
  (including the dataset descriptor hash; resume hard-errors on descriptor drift)
- Run-level dataset hash artifact: `outputs/<run_id>/dataset_hash.txt`
- Peak GPU memory telemetry recorded in `models/training_summary.json`
- Evaluation artifacts: `outputs/evaluation/*`
- Reporting artifacts: `reports/tables/*`, `reports/figures/*`, `reports/MODEL_CARD.md`
- Root copy of model card: `MODEL_CARD.md`
- Coverage gate: `fail_under = 80` (`pyproject.toml [tool.coverage.report]`;
  presentation/dev tooling omitted with rationale in `[tool.coverage.run]`)
- The default model path is held **bit-identical** to the pre-refactor code
  (parity-tested): every new transform / likelihood / latent-process option is
  opt-in, so the published numbers and golden fixtures stay valid without
  re-running. The trade-off (the model-spec review's reproducibility-capture
  point) is that adopting a better default is not a one-line flip — it
  re-baselines every published metric and regenerates the golden fixtures, so
  any default change is gated as its own decision rather than bundled with the
  experiment that motivates it.
