# Config Specification

`panelcast run --config file.yaml` loads one or more YAML files whose
**top-level keys mirror `PipelineConfig` field names** — a flat namespace, no
nesting. `configs/base.yaml` is the canonical example: every key it sets equals
the effective CLI default, so it doubles as the reference for names, types, and
defaults. Presets (`--preset quick|dev|diagnostic|publication`) are ordinary
config files layered first.

Precedence: built-in defaults < preset < `--config` files (later files win) <
options given explicitly on the command line.

**Unknown keys are a hard error** with the offending path and nearest known
keys, so a typo'd or nested key cannot silently fall back to the default. The
`--allow-unknown-config-keys` escape downgrades the error to a warning for
migrating old configs; ignored keys are then preserved in the run manifest
under `unknown_config_keys` as provenance only — their intended effect is not
applied, and they are never written to `resolved_config.yaml`. The
same strictness applies to dataset descriptors: unknown descriptor fields are
fatal.

Domain/dataset settings (column names, score bounds, feature blocks, file
paths) do **not** live here — they belong to a dataset descriptor under
`configs/datasets/`, selected with `--dataset` (or the `dataset` key below).

## Supported keys

The authoritative list is the mapping table in
`src/panelcast/config/pipeline_yaml.py`; defaults live in `configs/base.yaml`
and `panelcast run --help`.

Run control
- `seed`: int — RNG seed for the whole pipeline (default 42)
- `stages`: list of stage names, or a comma string (`data`, `splits`,
  `features`, `train`, `evaluate`, `predict`, `report`, `sensitivity`)
- `skip_existing`: bool — reuse stage outputs already on disk
- `dry_run`: bool
- `strict`: bool — strict raw-schema validation during ingest
- `enforce_lockfile`: bool — refuse to run if the environment doesn't match
  `pixi.lock`
- `verbose`: bool
- `dataset`: string — dataset descriptor name (same as `--dataset`)

MCMC
- `num_chains`, `num_samples`, `num_warmup`: int (default 4 / 1000 / 1000)
- `target_accept`: float (default 0.90)
- `max_tree_depth`: int (default 10)
- `chain_method`: `sequential` | `vectorized` | `parallel`
- `init_strategy`: `uniform` (default) | `median` | `feasible`
- `caged_chain_retries`: integer from 0 to 10 (default 0/off; booleans and fractional values are rejected). When enabled, a fit is reseeded only if at least one chain meets both caged-chain criteria and the remaining chains pass the configured R-hat/ESS/divergence gate. Retry seeds are `seed + 1`, `seed + 2`, and so on; the first result with no caged chains is retained.
- `caged_chain_tree_depth_fraction`: finite float in (0, 1] (default 0.95)
- `caged_chain_boundary_sigma`: positive finite float (default 0.005)
- `caged_chain_consensus_ratio`: finite float > 1 (default 5.0). A chain is caged when its mean `num_steps` reaches the configured fraction of `2^max_tree_depth - 1`, its posterior-mean `{model_prefix}_sigma_artist` is at or below the boundary, and the median posterior-mean sigma of the other chains is at least this multiple larger.

Convergence gates
- `rhat_threshold`: float (default 1.01)
- `ess_threshold`: float (default 400)
- `allow_divergences`: bool (default false)

Data filtering
- `min_ratings`: int (default 10)
- `min_albums_filter`: int (default 2)
- `max_albums`: int (default 50)

Feature ablation (block composition comes from the dataset descriptor; these
flags disable ablation groups)
- `enable_genre`, `enable_artist`, `enable_temporal`: bool (default true)

Observation noise
- `n_exponent`: float in [0, 1] (default 0.0 — homoscedastic)
- `learn_n_exponent`: bool (default false)
- `n_exponent_prior`: `logit-normal` | `beta`
- `n_exponent_alpha`, `n_exponent_beta`: float (legacy beta-prior parameters)
- `sigma_obs_prior_type`: `halfnormal` | `lognormal`
- `sigma_artist_prior_type`: `halfnormal` (default) | `lognormal` (no mass at the sigma=0 boundary)
- `sigma_rw_lognormal_loc`, `sigma_rw_lognormal_sigma`: float (default -2.8 / 0.6; LogNormal params for the sigma_rw prior, used when the prior type is lognormal — right-size per domain scale)
- `sigma_artist_lognormal_loc`, `sigma_artist_lognormal_sigma`: float (default -0.9 / 0.6; LogNormal params for the sigma_artist prior, used when `sigma_artist_prior_type` is lognormal)
- `rho_loc`, `rho_scale`: float (default 0.0 / 0.3; Normal params for the AR(1) coefficient prior. `rho_scale` must be > 0; set it small — e.g. ~0.02 — to pin rho near zero and effectively disable the AR persistence channel per domain)
- `heteroscedastic_entity_obs`: bool (AOTY default true since 0.13.0, #238; per-entity multiplicative noise, promoted on the three-seed subset confirmation under #237's coverage non-inferiority rule — see [`decisions/entity_overdispersion.md`](decisions/entity_overdispersion.md). Domains that rejected it — IMDb, econ — pin false)
- `tau_entity_scale`: float (default 0.25)

Likelihood and model gates
- `artist_effect_param`: `noncentered` (default) | `zerosum` (removes the mu_artist <-> effects location ridge)
- `likelihood_family`: one of the registry families (default `studentt`)
- `likelihood_df`: float (default 4.0; >= 100 degrades Student-t to Normal)
- `discretize_observation`: bool (default false)
- `target_transform`: `offset_logit` (default since 0.5.0, #43) | `identity`
- `logit_offset`: float (offset_logit transform tuning; YAML-only)
- `ar_center`: `global` | `none` | `artist_running`
- `latent_process`: `rw` | `ar1`
- `debut_prev_score_source`: `train_mean` | `dataset_stats`
- `errors_in_variables`: bool (default false; YAML-only gate)
- `propagate_rw_horizon`: bool (default false; YAML-only gate)
- `exclude_rw_raw_from_collection`: bool (GPU-memory reduction)

Splits
- `val_albums`: int (default 0)
- `min_train_albums`: int (default 2)

Evaluation
- `calibration_intervals`: list of floats in (0, 1) (default `[0.80, 0.95]`)
- `coverage_tolerance`: float (default 0.03)
- `prediction_interval`: float (default 0.95)
- `evaluate_secondary_split`: bool (default true)

Prediction batching (YAML-only)
- `predictive_batch_size`: int
- `predict_artist_batch_size`: int

"YAML-only" keys have no CLI flag; everything else also exists as a
`panelcast run` option, and the explicit CLI spelling wins over any YAML value.

## Pre-fit check: `panelcast preflight`

`panelcast preflight` is a statistical sanity check that runs **after the
features stage and before the first fit**. It reads the prepared splits and
feature matrices, resolves the exact `PriorConfig` and `X / artist_idx / y` the
fit would use (same `--dataset` / `--config` arguments as `run`), and never
touches the GPU or MCMC. Warn-only by default (exit 0); under `--strict` a
statistical FAIL exits 1 while a setup error (features not built yet) exits 2,
so a strict `1` unambiguously means the statistics are bad. `--json` emits a
machine-readable payload for CI.

Not to be confused with `panelcast run --preflight`, which estimates **GPU
memory** — a different concern. `preflight` audits the *statistics* of the fit:

- **Check A — prior/data scale.** Resolved `sigma_rw` / `sigma_artist` prior
  medians vs the data moments they govern, on the model-training scale: the
  within-entity per-step SD of the target, and the cross-entity SD of
  entity-mean targets. Gaps beyond ~0.75 / ~1.5 orders of magnitude WARN / FAIL.
  The `sigma_rw` moment is an upper bound on the latent scale, so it is treated
  asymmetrically: a prior *below* the moment skips the WARN band and only FAILs
  once it is more than ~1.5 orders below, while a prior above WARNs/FAILs at the
  usual thresholds. On a flag it prints a ready-to-paste, data-derived YAML
  block. This catches AOTY-scale sigma priors carried onto a differently-scaled
  domain, which push the data far into the prior tail.
- **Check B — collinearity given entity intercepts.** Within-entity demeans and
  standardizes the covariate matrix (appending cohort dummies when group pooling
  is active), then reports the *residual* condition number — after machine-exact
  structural nulls (benign one-hot / sequence-count redundancies the `beta`
  ridge prior absorbs) are stripped. A large residual condition number names the
  participating feature set, e.g. an `age_c + album_sequence + release_year +
  cohort` age-period-cohort identity that per-entity intercepts otherwise hide.
- **Check C — Beta-Binomial trial scale.** When `likelihood_family:
  beta_binomial` consumes a true aggregation count, a target span other than one
  expands each count by the score span. Preflight FAILs with guidance to rescale
  genuine proportions and their `target_bounds` to `[0, 1]`; the run path also
  emits a warning because preflight is optional.
