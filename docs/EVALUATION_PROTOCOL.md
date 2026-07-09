# Evaluation Protocol

Protocol terms are generic (entities and their sequential events); in the AOTY
example domain an entity is an artist and an event is an album.

Metrics
- R2 on held-out test sets (primary and secondary splits)
- RMSE and MAE
- Calibration curves and coverage of credible intervals (80% and 95%)
- CRPS (proper scoring rule for probabilistic regression)
- Ranking metrics on the held-out slate (`ranking` block in `metrics.json`,
  `ranked_slate.csv` per split): Spearman/Kendall rank correlation, expected vs
  realized rank, and P(top-K) for K in {5, 10, 25}, where each P(top-K) carries
  its own audited reliability curve (binned predicted-vs-realized frequency).
  Read as descriptive on a single slate (top-K is high-variance) until the
  rolling-origin backtest supplies multiple slates.
- Sliced calibration (`calibration.by_slice` in `metrics.json`): empirical
  coverage per subgroup — genre group, review-count decile, target tercile,
  training-history bin — each with Wilson 95% CIs and a min-n floor (default 20
  rows) below which a slice is skipped as noise. Informational, not gating: it
  surfaces offsetting miscalibration the global coverage gate can hide, and the
  payload states the expected false-flag count so lone flags are not over-read.
- Per-row error decomposition (`panelcast diagnose --errors`): entity / group /
  review-count-decile rollups and a worst-25 table from the identified
  predictions payload — read-only, no refit.

Cross-validation
- Primary evaluation: within-entity temporal holdout (last event per entity)
- Secondary evaluation: entity-disjoint split (no entity overlap)
- Secondary split uses cold-start predictive path (population-level entity effect)
- Secondary split sets `prev_score` to training global mean for every row (no held-out label usage)
- Primary split fails fast if unknown entities appear in test data (no silent row dropping)

Rolling-origin backtest (`panelcast backtest --origins K`)
- Origin k holds out each entity's (last-k)-th event as test and drops the k
  later events entirely; origin 0 is exactly the standard primary split.
- Each origin runs the full splits->features->train->evaluate chain as its own
  run directory with fresh data stamps, so every leakage control holds
  unchanged; the origin's split content hash lands in the backtest ledger.
- Metrics are reported as mean ± SE across origins (plus min/max). Deeper
  origins shrink the eligible entity set — per-origin n_test and n_entities
  are reported and cross-origin variation includes that population shift.
- The JSON ledger under `outputs/backtest/<id>/` makes an interrupted backtest
  resume at the next unfinished origin; rerun the same command to resume.

Conformal calibration wrapper (`conformal_calibration: true`, default off)
- Calibrated on the validation split with train-only history (needs
  `val_albums >= 1`); test rows never inform the calibration.
- Two layers reported in `metrics.json` under `calibration.conformal`, next
  to the raw Bayesian numbers: split-conformal (CQR) interval widening with
  its finite-sample adjustment per nominal level, and quantile recalibration
  (PIT remapping) of the whole predictive CDF.
- The guarantee is marginal coverage >= the nominal level **under
  exchangeability**. Within-entity temporal validation/test eras drift, so
  read the guarantee as approximate; weighted conformal is a possible
  follow-up.
- With the flag on, `predict_next` emits `conformal_q05`/`conformal_q95`
  (recalibrated quantiles) alongside the posterior quantile columns; with it
  off, outputs are byte-identical to before.

Multi-step rollout evaluation (`eval_horizon: H`, default 0 = off)
- Every flagship number above is ONE-step-ahead: each test event conditions on
  the observed previous score (teacher forcing). With `eval_horizon: H` the
  evaluate stage additionally scores genuine h-step forecasts for h = 1..H by
  ancestral rollout: per posterior draw the sampled score feeds back as the AR
  lag and a fresh latent innovation compounds per step, so uncertainty grows
  through both channels.
- Step-h covariates are the REALIZED held-out values — the curves isolate the
  model's dynamic honesty, not covariate forecasting. Production callers must
  supply their own future covariates (or hold at last-known).
- CRPS/coverage/RMSE per horizon land in `horizon_rollout.json`, a separate,
  clearly-labeled artifact. Horizon-decay numbers are NOT comparable to and
  never mix into the flagship one-step metrics; h=1 reconciles with the
  one-step evaluation in distribution (the sanity anchor).

History cap (`max_albums`)
- `--max-albums` (default 50 for AOTY) caps the length of the time-varying
  trajectory per entity. It is a max-EVENTS cap, not a row filter: an entity's
  events beyond the most recent `max_albums` are NOT dropped — they collapse
  onto sequence position 1 (the initial entity effect), so every row still
  contributes to the likelihood. The cap bounds the random-walk trajectory
  length (and peak GPU memory); distant positions carry little signal about the
  current state because cumulative random-walk variance grows over steps.
- Domains with longer histories than AOTY should raise `--max-albums`
  accordingly. The cap is computed on training data only (no leakage).

Diagnostics
- R-hat <= 1.01 for all key parameters (`rhat_threshold`, default 1.01)
- ESS >= 400 per chain (`ess_threshold`, default 400)
- No divergent transitions (`allow_divergences` is false by default)
- Coverage must be within configured tolerance (`coverage_tolerance`, default 0.03)
- Posterior predictive checks are informational, not gating: with AOTY's known
  left-skewed target against the symmetric likelihood, several PPC p-values are
  expected to pin at 0.000/1.000 (see `MODEL_CARD.md`)

Model comparison
- LOO and WAIC computed for the primary split when pointwise log-likelihood is available
- Secondary split reports info-criteria as unavailable when using the new-entity predictive path

Sensitivity analyses (opt-in `sensitivity` stage, not part of a default run)
- Min-ratings thresholds (5, 10, 25)
- Prior variants (default, diffuse, informative; `PRIOR_CONFIGS` in `src/panelcast/pipelines/sensitivity.py`)
- Feature ablations (remove feature groups to measure importance)
- Split-seed axis (seeds 42/43)

Reproducibility of MCMC draws is a two-tier claim. Every run manifest records
an environment **fingerprint** — a canonical hash over python/jax/jaxlib/
numpyro versions, the accelerator platform and device kind, and the machine
architecture. Draws reproduce **bit-exactly** only within a matching
fingerprint; across fingerprints (e.g. a GPU fit re-run on CPU, or a jaxlib
upgrade) expect **statistical** reproduction — same posterior up to sampling
noise, different bits. The fingerprint deliberately excludes the pixi.lock
hash (churn in non-numerical dependencies does not change the exactness
domain) and the OS release.
