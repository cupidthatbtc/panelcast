# Porting the pipeline to a new domain

The pipeline is domain-agnostic: every dataset-specific name — columns,
target bounds, date formats, posterior-site prefixes, feature blocks — comes
from a `DatasetDescriptor` (`src/panelcast/config/descriptor.py`). Retargeting
from AOTY (artists releasing albums scored 0–100) to a completely different
domain is **one YAML file and zero source changes**.

> A second bundled retarget — real data, unit-interval bounds, and the
> `beta_binomial` likelihood — lives at `examples/elections/` (US Senate
> statewide returns, CC0; see its `ATTRIBUTION.md`). The walkthrough below
> uses the synthetic aerospace example; the elections descriptor follows the
> identical pattern.
>
> Descriptors also own the domain's model facts (#268): `likelihood_family`,
> `target_transform`, and `max_events` make `panelcast run --dataset X.yaml`
> the complete, correct fit with no extra flags (explicit CLI/config values
> still win). A true proportion recorded on a non-unit span (e.g. percent)
> can declare `rescale_target_to_unit: true` and the pipeline rescales it to
> [0, 1] internally, so Beta-Binomial trial counts are never span-inflated.

This guide walks the bundled aerospace example end to end:
airframes fly sequential test flights, each scored 0–10 by flight-test
telemetry. The full descriptor lives at `configs/datasets/aero.yaml`;
the e2e proof at `tests/e2e/test_domain_portability.py`.

**What "portable" means here:** the *apparatus* ports — a new domain runs the
full pipeline (data → … → report) with zero source changes, and AOTY stays
byte-identical. It does **not** mean the model predicts well on a new domain: the
e2e test asserts the pipeline runs and the output structure, not predictive
accuracy, which is untested off AOTY by construction.

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
  `offset_logit` transform) the exact logit offset. Get these right. Report
  figures derive non-default-domain labels and fan-chart limits from the
  descriptor/data; set `invert_target_axis: true` for magnitude-like targets.
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

`--min-ratings` defaults to the descriptor's `primary_min_obs`, so the splits
stage reads the primary processed dataset with no extra flag; pass
`--min-ratings` (or a `--config` YAML with `min_ratings:`) only to override it.

Run each domain from its own working directory: the `data/`, `models/` and
`outputs/` trees are per-cwd, so two domains never collide as long as you
keep one directory per domain. (The descriptor hash recorded in each run
manifest hard-errors a resume if the descriptor changed under it.) Note that
bare `--dataset` names resolve to `configs/datasets/{name}.yaml` *relative to
the working directory* — from a separate domain directory, pass the
descriptor's absolute path instead.

**Before the first fit, run `panelcast preflight`.** After the data/splits/
features stages exist (`panelcast run --dataset <name> --stages
data,splits,features`), run `panelcast preflight --dataset <name>` (plus the
same `--config` you will fit with). It reads the prepared data and flags three
common porting mistakes without touching the GPU: AOTY-scale `sigma_rw` /
`sigma_artist` priors that don't match your target's scale; covariate
collinearity given the per-entity intercepts (e.g. an age-period-cohort
identity from time-like covariates); and a Beta-Binomial target span that
silently multiplies genuine aggregation counts. It is warn-only; add `--strict`
to make a FAIL exit nonzero in CI. See `docs/CONFIG_SPEC.md` for the checks.

### First-fit acceptance checklist

Use this order for a new domain; it separates data/descriptor mistakes from
sampling failures before either can send you into a model-spec diagnostic loop.

1. **Record claims and provenance first.** Extract the source paper's target
   numbers, hash or otherwise identify the exact input bytes, and state what a
   successful replication must recover before fitting anything.
2. **Audit descriptor roles.** Confirm entity, event, and year columns have the
   intended distinctness; catalog IDs remain strings; target bounds match the
   stored scale; and the observation-count column has the meaning claimed by
   `n_obs_is_aggregation_count`.
3. **Protect true proportions.** A `beta_binomial` target with genuine trial
   counts must use a unit span (`target_bounds: [0.0, 1.0]`). A wider span
   expands the effective trial count and can make intervals absurdly tight;
   `panelcast preflight --strict` rejects that configuration.
4. **Audit feature identities.** Entity intercepts, cohort pooling, generic
   temporal/history blocks, and domain age/experience covariates can form an
   exact age-period-cohort identity. Strip redundant temporal/history blocks;
   represent era through the cohort group when that matches the design.
5. **Size priors from the data.** Compute the cross-entity SD and pooled
   within-entity step SD of the target. Set the corresponding lognormal prior
   locations near the log moments, with widths around 0.8–1.0, rather than
   inheriting scales from a differently-sized domain.
6. **Run preflight on the exact fit config.** Run the data, splits, and features
   stages, then `panelcast preflight --dataset <name> --config <fit.yaml>
   --strict`. Resolve every FAIL before starting MCMC.
7. **Start with the robust fit recipe.** Use uniform initialization, warmup of at
   least 2,000, and target acceptance of at least 0.90. If the source model has
   no persistence term, pin `rho` tightly near zero rather than leaving an
   unidentified AR channel.
8. **Verify resolved knobs.** Read the run's `resolved_config.yaml`; do not infer
   from the input YAML that a key reached the model.
9. **Autopsy chains before changing the model.** Inspect per-chain posteriors. A
   group of agreeing chains plus one boundary-scale, tree-depth-saturated chain
   is a caged chain, not evidence for a new specification. Retry or exclude it
   under a declared protocol, then accept only a diagnostics-passing run that
   agrees with the healthy-chain consensus across seeds.

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
`within_entity_temporal` and `entity_disjoint`. These are labels only — the
values and behavior are correct for any domain; making them descriptor-driven is
future work.

## Covariate curves

Descriptors can opt into reproducible cubic B-spline columns without writing a
feature pack:

```yaml
basis_curves:
  age_curve:
    col: age
    type: spline
    df: 5
    center: 27
```

The capability is off when `basis_curves` is omitted or empty. `df` requests the
maximum number of emitted columns (`age_curve__basis_00` onward) and must be at
least 4. The complete partition-of-unity basis is reference-coded by dropping
one recorded column so model mean-centering remains identifiable. Repeated
boundary or interior quantiles deterministically reduce the emitted dimension
until the centered training design is full rank; training fails clearly when no
identifiable cubic basis is possible. Only `type: spline` is currently supported.
Knots are fitted from each training split's centered source values and reused
unchanged for validation and test, including out-of-range values. The feature
manifest records the per-split knot state. Training then binds each basis name
and model index to the exact fitted feature mean and standard deviation in both
`training_summary.json` and the NetCDF model attributes. Pass that durable
training-summary curve state to `panelcast.reporting.extract_posterior_curve`;
it applies the model standardizer before `summarize_curve_peak` reports posterior
peak/vertex intervals.

## Custom feature blocks

If your domain has analogues of genres/collaborations, write a feature pack:
a module registering blocks into the feature registry, listed under
`feature_packs:` in the descriptor. See `src/panelcast/features/packs/aoty.py`
for the music pack and `docs/EXTENSIBILITY.md` for the registry API.
