# Porting the pipeline to a new domain

The pipeline is domain-agnostic: every dataset-specific name — columns,
target bounds, date formats, posterior-site prefixes, feature blocks — comes
from a `DatasetDescriptor` (`src/panelcast/config/descriptor.py`). Retargeting
from AOTY (artists releasing albums scored 0–100) to a completely different
domain is **one YAML file and zero source changes**.

This guide walks the bundled aerospace example end to end:
airframes fly sequential test flights, each scored 0–10 by flight-test
telemetry. The full descriptor lives at `configs/datasets/aero.yaml`;
the e2e proof at `tests/e2e/test_domain_portability.py`.

## The mental model

| Concept | AOTY (default) | Aero example |
|---|---|---|
| Entity (the thing with a career) | `Artist` | `Airframe` |
| Event (one observation in the career) | `Album` | `Flight_ID` |
| Target (bounded score) | `User_Score` in [0, 100] | `Perf_Score` in [0, 10] |
| Observation count (heteroscedastic noise) | `User_Ratings` | `Sensor_Samples` |
| Chronology | `Release_Date` (`"April 10, 2018"`) | `Flight_Date` (ISO `2021-03-04`) |
| Posterior site prefix | `user_` | `perf_` |

Everything in the model is expressed against these roles: the hierarchical
entity effect, the per-event random walk, the AR(1) on the previous score,
the n^exponent noise scaling. None of it cares whether the entity is a
musician or an airframe.

## Step 1 — Describe your raw data

Create `configs/datasets/{name}.yaml`. Only override what differs from the
AOTY defaults (the descriptor's "default-equals-AOTY" contract means omitted
keys are safe):

```yaml
name: aero
entity_col: Airframe
event_col: Flight_ID
target_col: Perf_Score
target_bounds: [0.0, 10.0]
model_prefix: perf
n_obs_col: Sensor_Samples
date_format: "%Y-%m-%d"
raw_column_map:
  "Flight Date": Flight_Date
  "Perf Score": Perf_Score
  # raw header -> canonical (underscore) name
secondary_target_col: null   # no critic-score analogue
secondary_prefix: null
secondary_n_obs_col: null
processed_name_template: "perf_minobs_{min_ratings}"
feature_packs: []            # drop the music-specific blocks
feature_blocks:
  - name: temporal
  - name: entity_history
  - name: core_numeric       # the domain's own numeric covariates
    params:
      columns: [Thrust_Margin, Payload_Fraction]
```

Key decisions:

- **`target_bounds`** drive the model's soft-clipping and (under the
  `offset_logit` transform) the exact logit offset. Get these right.
- **`model_prefix`** names every posterior site (`perf_beta`, `perf_rho`, …)
  and the model key in `models/manifest.json` (`perf_score`). Pick once;
  changing it later orphans fitted models.
- **`secondary_*: null`** disables the dual-model path entirely — no critic
  dataset, no secondary filters.
- **`feature_packs: []`** unregisters the music-domain blocks
  (genre, album type, collaboration). The generic blocks — `temporal`
  (sequence, career length, release gaps, date-missing flag) and
  `entity_history` (leave-one-out prior mean/std/count, trajectory,
  debut flag) — work for any domain.
- **`core_numeric`** passes any numeric columns you mapped in
  `raw_column_map` straight through as model features (train-fitted
  median/mean/zero imputation). This is pure-YAML feature selection: extra
  covariates need no Python at all.

## Step 2 — Point the pipeline at your data

```bash
# descriptor resolves bare names to configs/datasets/{name}.yaml
panelcast run --dataset aero --num-chains 2 --num-samples 500
```

The raw CSV path comes from `raw_path_env` (environment variable) or
`raw_path_default`. Processed datasets, splits, features, the fitted model
and predictions all flow through the same stages with descriptor-derived
names: `data/processed/perf_minobs_5.parquet`, posterior sites `perf_*`,
predictions whose `artist` column holds your entity names (airframes).

Pass `--min-ratings` (or a `--config` YAML with `min_ratings:`) matching the
descriptor's `primary_min_obs` so the splits stage reads the primary
processed dataset.

Run each domain from its own working directory: the `data/`, `models/` and
`outputs/` trees are per-cwd, so two domains never collide as long as you
keep one directory per domain. (The descriptor hash recorded in each run
manifest hard-errors a resume if the descriptor changed under it.) Note that
bare `--dataset` names resolve to `configs/datasets/{name}.yaml` *relative to
the working directory* — from a separate domain directory, pass the
descriptor's absolute path instead.

## Step 3 — Verify

- `training_summary.json` gains a `dataset` block recording the descriptor
  (name, columns, bounds, prefix, content hash). Downstream stages read the
  domain from there, so evaluate/predict always match the fitted model.
- The posterior-prefix guard fails loudly if `models/` holds a model fit
  under a different descriptor.
- The e2e tests in `tests/e2e/test_domain_portability.py` show the full
  assertion set: stage outputs exist under descriptor-derived names, no
  music-domain feature columns appear, predictions stay inside the target
  bounds.

## What you do NOT need to do

- No edits under `src/` — a lint test
  (`tests/unit/test_no_domain_literals.py`) fails the build if AOTY literals
  creep back into shared code paths.
- No renaming of pipeline-internal artifact keys (`n_reviews`,
  `global_mean_score`, `prev_score`, …). These are *role* names, not domain
  names; only their sources are descriptor-driven.

**Known cosmetic limitation.** A few AOTY-derived names are still hard-coded for
every domain: the predictions output column is literally `artist` (it holds your
entity names regardless of domain), and the split directories are always
`within_artist_temporal` and `artist_disjoint`. These are labels only — the
values and behavior are correct for any domain; making them descriptor-driven is
future work.

## Custom feature blocks

If your domain has analogues of genres/collaborations, write a feature pack:
a module registering blocks into the feature registry, listed under
`feature_packs:` in the descriptor. See `src/panelcast/features/packs/aoty.py`
for the music pack and `docs/EXTENSIBILITY.md` for the registry API.
