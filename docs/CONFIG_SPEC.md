# Config Specification

`panelcast run --config file.yaml` loads one or more YAML files whose
**top-level keys mirror `PipelineConfig` field names** — a flat namespace, no
nesting. `configs/base.yaml` is the canonical example: every key it sets equals
the effective CLI default, so it doubles as the reference for names, types, and
defaults. Presets (`--preset quick|dev|diagnostic|publication`) are ordinary
config files layered first.

Precedence: built-in defaults < preset < `--config` files (later files win) <
options given explicitly on the command line.

**Unknown keys are warned about and ignored** (one structured warning per key),
so a typo'd or nested key silently falls back to the default — check the log
for unknown-key warnings when a setting doesn't seem to take.

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
- `heteroscedastic_entity_obs`: bool (default false)
- `tau_entity_scale`: float (default 0.25)

Likelihood and model gates
- `likelihood_family`: one of the registry families (default `studentt`)
- `likelihood_df`: float (default 4.0; >= 100 degrades Student-t to Normal)
- `discretize_observation`: bool (default false)
- `target_transform`: `identity` | `offset_logit`
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
