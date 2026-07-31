# CLI Reference

Complete command-line interface documentation for `panelcast`.

## Installation

The CLI is installed as `panelcast` when the package is installed:

```bash
pip install -e .
# or with pixi
pixi install
```

## Quick Reference

```bash
panelcast --help              # Show all commands
panelcast run --help          # Full pipeline options
panelcast stage --help        # Individual stage commands
```

---

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version and exit |
| `--help` | | Show help message and exit |

```bash
panelcast --version
panelcast --help
```

---

## Commands

### `run` — Execute Full Pipeline

Runs all pipeline stages in dependency order: data → splits → features → train → evaluate → predict → report.

```bash
panelcast run [OPTIONS]
```

#### General Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--seed` | | `42` | Random seed for reproducibility |
| `--verbose` | `-v` | `false` | Enable DEBUG logging |
| `--dry-run` | | `false` | Show execution plan without running |
| `--strict` | | `false` | Fail on convergence/calibration/reporting guardrail violations |
| `--skip-existing` | | `false` | Skip stages with unchanged inputs |
| `--stages` | `-s` | all | Comma-separated stages (e.g., `data,splits,train`) |
| `--resume` | | | Resume failed run by run-id (e.g., `2026-01-19_143052`) |
| `--tag` | | | Free-form label recorded in the run manifest (shown by `runs history`) |

#### Preflight Memory Check Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preflight` | `false` | Quick memory check (~1s) with fixed estimates |
| `--preflight-full` | `false` | Mini-MCMC to measure actual peak memory (~30-60s) |
| `--preflight-only` | `false` | Run memory check and exit (0=pass, 1=fail, 2=warning) |
| `--force-run` | `false` | Override preflight failure and continue anyway |
| `--recalibrate` | `false` | Force fresh calibration even if cached calibration exists |

**Note:** `--preflight-full` takes precedence over `--preflight`. The `--preflight-only` flag controls whether to exit after the check:

| Flags | Behavior |
|-------|----------|
| `--preflight-only` | Quick check (~1s) and exit |
| `--preflight-only --preflight-full` | Full measured check (~30-60s) and exit |
| `--preflight` | Quick check, then run pipeline |
| `--preflight-full` | Full check, then run pipeline |

#### MCMC Configuration

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--num-chains` | `4` | ≥1 | Number of parallel MCMC chains |
| `--num-samples` | `1000` | ≥100 | Post-warmup samples per chain |
| `--num-warmup` | `1000` | ≥50 | Warmup iterations per chain |
| `--target-accept` | `0.90` | 0.5–0.999 | Target acceptance probability |
| `--max-events` | descriptor `max_events`, else 50 | ≥1 | Maximum events per entity |

#### Convergence Thresholds

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--rhat-threshold` | `1.01` | 1.0–1.1 | Maximum acceptable R-hat |
| `--ess-threshold` | `400` | ≥100 | Minimum ESS per chain |
| `--allow-divergences` | `false` | | Don't fail on divergences |

#### Data Filtering

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--min-ratings` | descriptor default | ≥1 | Minimum observations per event; defaults to the dataset descriptor's `primary_min_obs` (10 for AOTY) |
| `--min-events` | `2` | ≥1 | Minimum events per entity for dynamic effects |

#### Feature Ablation

These features are **enabled by default**. Use these flags to disable them:

| Option | Description |
|--------|-------------|
| `--no-genre` | Disable genre features (enabled by default) |
| `--no-artist` | Disable artist reputation features (enabled by default) |
| `--no-temporal` | Disable temporal features (enabled by default) |

#### Heteroscedastic Noise Configuration

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--n-exponent` | `0.0` | 0.0–1.0 | Scaling exponent (0=homoscedastic, 0.5=sqrt) |
| `--learn-n-exponent` | `false` | | Learn exponent from data |
| `--n-exponent-prior` | `logit-normal` | | Prior type: `logit-normal` (recommended) or `beta` (legacy) |
| `--n-exponent-alpha` | `2.0` | ≥0.01 | Beta prior alpha parameter (only with `--n-exponent-prior beta`) |
| `--n-exponent-beta` | `4.0` | ≥0.01 | Beta prior beta parameter (only with `--n-exponent-prior beta`) |

#### Likelihood

| Option | Default | Description |
|--------|---------|-------------|
| `--likelihood-df` | `4.0` | Student-t degrees of freedom (`≥100` behaves as Normal) |
| `--likelihood-family` | descriptor `likelihood_family`, else `studentt` | Observation likelihood: `studentt`, `normal`, `skew_studentt`, `skew_normal`, `split_normal`, `beta`, `mixture`, `beta_binomial`, or `beta_ceiling`. See [`LIKELIHOOD_CANDIDATES.md`](LIKELIHOOD_CANDIDATES.md) |
| `--discretize-observation` | `false` | Interval-censor the observation to integers (honest PPC for integer scores). Location-scale families only (`studentt`, `normal`, `skew_normal`, `split_normal`, `mixture`); rejected for `skew_studentt`, `beta`, `beta_binomial` (`beta_binomial` is already discrete) |

#### Domain & Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` / `-c` | | YAML config file(s) with `PipelineConfig` keys; repeatable, later files win. Explicit CLI options always win over YAML. Unknown keys are a hard error with nearest-key suggestions. |
| `--allow-unknown-config-keys` | `false` | Migration escape: load configs despite unknown keys — they are ignored (not applied), preserved in the run manifest under `unknown_config_keys`, and never written to `resolved_config.yaml` |
| `--dataset` | built-in AOTY | Dataset descriptor: bare name (resolves to `configs/datasets/{name}.yaml`) or YAML path. Unknown descriptor fields are a hard error |
| `--debut-prev-score-source` | `train_mean` | Debut `prev_score` fill: `train_mean` or `dataset_stats` (legacy; mild leakage) |
| `--target-transform` | descriptor `target_transform`, else `offset_logit` | Score-scale transform: `offset_logit` (default since 0.5.0; the model runs on the Smithson-Verkuilen logit scale, bounds hold by construction) or `identity` (soft-clip; the former default, still selectable) |
| `--ar-center` | `global` | AR(1) centering: `global`, `none` (legacy), or `artist_running` (sensitivity only) |
| `--latent-process` | `rw` | Artist-effect process: `rw` (random walk) or `ar1` (stationary). Experimental |
| `--exclude-rw-raw-from-collection` | `false` | Don't store `rw_raw` draws on device (~96% peak-GPU cut); required for the 4-chain publication run on 24 GB GPUs |

#### MCMC (advanced)

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--chain-method` | `sequential` | | `sequential`, `vectorized`, `parallel` (multi-GPU), or `auto` (memory-informed: vectorized when all chains fit the JAX pool budget, else sequential) |
| `--checkpoint-every` | single-shot | ≥1 | Checkpoint the fit every N post-warmup draws so an interrupted run resumes from the last block (default: a single-shot fit). Each block commits its draws and the state it ended in immutably, with the cursor naming both by SHA-256 last, so a resume continues the same chain or refuses. A resume must match the fit exactly — model, data, config, collected fields, warm start, and the numpyro/JAX/backend environment — and a checkpoint from an older panelcast layout is refused; delete the directory to start over |
| `--max-tree-depth` | `10` | 5–15 | Maximum NUTS tree depth |
| `--likelihood-df` | `4.0` | ≥1.0 | Student-t degrees of freedom; ≥100 ≈ Normal |

#### Splits & Calibration

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--val-events` | `0` | ≥0 | Events per entity held out for validation (0 = none) |
| `--min-train-events` | `2` | ≥1 | Minimum training events per entity |
| `--origin-offset` | `0` | ≥0 | Rolling-origin offset k: drop each entity's last k events as future and hold out the (last-k)-th; `0` = the standard split. `panelcast backtest` sweeps this |
| `--secondary-split` / `--no-secondary-split` | on | | Artist-disjoint secondary evaluation split |
| `--calibration-intervals` | `0.80,0.95` | | Comma-separated interval levels for calibration checks |
| `--coverage-tolerance` | `0.03` | ≥0.0 | Allowed absolute coverage error |
| `--prediction-interval` | `0.95` | 0.01–0.99 | Interval level for saved prediction bands |

#### Config Presets

`--preset` is sugar for `--config configs/<preset>.yaml`, layered **first** so
any explicit `--config` files and CLI options still win. All 47 advanced flags
remain available; presets just set defaults.

| Preset | Shape | Use |
|--------|-------|-----|
| `quick` | 1 chain × 200, relaxed gates | Wiring/smoke checks (not for reported inference) |
| `dev` | 2 chains × 500 | Fast iteration |
| `diagnostic` | 4 chains × 1000 | Meaningful convergence + PPC diagnostics |
| `publication` | 4 chains × 5000, warmup 3000 | The final publication run |

```bash
panelcast run --preset quick
panelcast run --preset diagnostic --num-samples 2000   # CLI flag overrides the preset
```

For the full list, run `panelcast run --help`.

#### Examples

```bash
# Default run
panelcast run

# High-accuracy run
panelcast run --num-chains 8 --num-samples 2000 --target-accept 0.95

# Fast exploratory run
panelcast run --num-chains 1 --num-samples 500 --num-warmup 500

# Feature ablation study
panelcast run --no-genre --no-temporal

# Relaxed convergence for testing
panelcast run --rhat-threshold 1.05 --allow-divergences

# Resume a failed run
panelcast run --resume 2026-01-19_143052

# Check memory before running
panelcast run --preflight

# Check memory only (CI/scripting)
panelcast run --preflight-only

# Force run despite preflight failure
panelcast run --preflight --force-run

# Full preflight with mini-MCMC measurement
panelcast run --preflight-full

# Run specific stages only
panelcast run --stages data,splits,features
```

---

### `stage` — Run Individual Stages

Run pipeline stages independently.

```bash
panelcast stage <STAGE> [OPTIONS]
```

#### Available Stages

| Stage | Description |
|-------|-------------|
| `data` | Load raw data, apply cleaning, create processed datasets |
| `splits` | Create train/validation/test splits |
| `features` | Build feature matrices from split data |
| `train` | Fit Bayesian models using NumPyro MCMC |
| `evaluate` | Compute diagnostics, calibration metrics, LOO-CV |
| `predict` | Generate next-event predictions for known and new entities; writes `next_event_known_entities.csv` / `next_event_new_entity.csv` |
| `report` | Generate publication artifacts (figures, tables, model cards) |
| `sensitivity` | Optional prior-variant / feature-ablation analysis (opt-in; run by name, not part of a default run) |

#### Common Stage Options

All stages support:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--seed` | | `42` | Random seed |
| `--verbose` | `-v` | `false` | Enable DEBUG logging |
| `--help` | | | Show stage-specific help |

#### Train Stage Additional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strict` | `false` | Fail on convergence warnings |
| `--rhat-threshold` | `1.01` | Maximum acceptable R-hat |
| `--ess-threshold` | `400` | Minimum ESS per chain |
| `--allow-divergences` | `false` | Don't fail on divergences |

#### Examples

```bash
# Run data preparation only
panelcast stage data --verbose

# Run training with relaxed thresholds
panelcast stage train --rhat-threshold 1.05 --allow-divergences

# Generate reports
panelcast stage report
```

---

### `export-figures` — Static Figure Export

Export visualization figures to static formats.

```bash
panelcast export-figures [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `reports/interactive` | Output directory |
| `--formats` | `-f` | `svg,png` | Comma-separated formats (svg,png,pdf) |
| `--width` | `-w` | `800` | Figure width in pixels |
| `--height` | | `600` | Figure height in pixels |
| `--scale` | `-s` | `2.0` | Scale factor (2.0 = ~300dpi) |
| `--run` | `-r` | | Path to pipeline run directory |
| `--help` | | | Show help |

#### Examples

```bash
# Default export
panelcast export-figures

# All formats, high resolution
panelcast export-figures --formats svg,png,pdf --scale 3.0

# Custom dimensions
panelcast export-figures --width 1200 --height 800

# Specific run
panelcast export-figures --run outputs/2026-01-19_143052
```

---

### `demo` — End-to-End Demonstration

Run the whole pipeline on the bundled synthetic aerospace example
(`examples/aerospace/`) at tiny scale — a one-command way to see every stage
execute with no external data. Finishes with a generated model card under
`outputs/<run_id>/reports/`.

```bash
panelcast demo [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--descriptor` | `examples/aerospace/descriptor.yaml` | Descriptor YAML for the demo dataset |
| `--num-chains` | `1` | MCMC chains |
| `--num-samples` | `300` | Post-warmup samples per chain |
| `--num-warmup` | `300` | Warmup iterations per chain |
| `--seed` | `42` | Random seed |

---

### `compare` — Baseline Benchmark

Fit the baseline predictors (global-mean, entity-mean, last-score, ridge, GBM)
on the existing splits and emit a populated comparison table (CSV + Markdown +
JSON) scored through the same metrics/calibration/CRPS toolkit as the model.
Requires the splits and features stages to have run.

```bash
panelcast compare --baselines [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--baselines` | | `false` | Run the baseline benchmark (required to do anything) |
| `--dataset` | | built-in AOTY | Dataset descriptor (bare name or YAML path) |
| `--output` | `-o` | latest run's `reports/baselines` | Output directory (default: the latest run's `outputs/<run_id>/reports/baselines`; flat `reports/baselines` when no run exists) |
| `--num-samples` | | `1000` | Predictive samples per baseline for interval scoring |
| `--seed` | | `0` | Random seed for predictive sampling |
| `--bayes` / `--no-bayes` | | on | Append the current model's metrics from the `--metrics` file |
| `--metrics` | | latest run's `evaluation/metrics.json` | Evaluation metrics.json supplying the model's row (with `--bayes`) |

```bash
panelcast compare --baselines
panelcast compare --baselines --dataset examples/aerospace/descriptor.yaml
```

---

### `diagnose` — Convergence + PPC Report

Re-present an existing run's convergence gate and posterior-predictive-check
p-values (PPC statistics pinned near 0/1 are flagged as the signature of
likelihood misspecification). No model refit.

```bash
panelcast diagnose [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--eval-dir` | | latest run's `evaluation/` | Directory with `diagnostics.json` / `metrics.json` (default: the latest run's `outputs/<run_id>/evaluation/`) |
| `--output` | `-o` | `reports/diagnostics` | Output directory for the report (run-scoped when `--errors` is set) |
| `--errors` | | `false` | Also decompose per-row errors from the identified predictions payload: full per-row CSV, entity / group / review-count-decile rollup CSVs, and a worst-25 Markdown table per split. Read-only, so it works on any past run whose payload is identified (pre-0.10.0 payloads get a clear re-run message) |

```bash
panelcast diagnose
panelcast diagnose --eval-dir outputs/2026-06-23_192630/evaluation
panelcast diagnose --errors
```

---

### `report` — Self-Contained HTML Run Dashboard

Compose a completed run's manifest, metrics, diagnostics, readiness verdicts,
figures, and coefficient table into a single portable `reports/index.html` that
renders offline (Plotly.js inlined, PNGs base64). Read-only over existing
artifacts, so it works on any past run; the report stage of a normal run also
writes this file best-effort, and this command regenerates it on demand.

```bash
panelcast report [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--run` | | latest run | Run directory to report on |
| `--output` | `-o` | `<run>/reports/index.html` | Output HTML path |
| `--interactive` / `--no-interactive` | | on | Interactive Plotly figures (self-contained, ~3–4 MB) vs embedded PNGs from `reports/figures` (smaller, quick shares) |

```bash
panelcast report
panelcast report --run outputs/2026-07-08_120000_000000_abcd
panelcast report --no-interactive -o run_summary.html
```

---

### `select` — Model-Selection Sweep

Run the portable model-selection protocol: enumerate the candidate space
(transform / likelihood / gates) from the code's own registries, print a
pre-run plan with predicted cost, then drive the staged sweep, score every
arm against pre-registered rules, and write one ranked report under
`outputs/select/<sweep-id>/`. `select` recommends; a default flip stays a
manual PR.

```bash
panelcast select [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | built-in AOTY | Dataset descriptor (bare name or YAML path) |
| `--effort` | `standard` | Effort tier: `quick` (screen), `standard`, `thorough` (+random +publication) |
| `--max-fits` | tier-defined | Hard cap on diagnostic fits (overrides the tier) |
| `--budget-hours` | | GPU-hour budget; stages truncate in priority order |
| `--arm-timeout` | `auto` | Per-arm wall-clock timeout in seconds, or `auto` to size each arm's timeout from its predicted runtime (max of an 1800s floor and 3x the transform-aware prediction); a fit exceeding it is killed and marked failed |
| `--warmup-transfer` | `false` | The reference arm exports its adapted warmup state (mass matrix); screening arms whose model signature matches exactly re-import it and run at reduced warmup. Confirmation fits always run cold |
| `--sweep-id` | `sweep` | Sweep directory name under `outputs/select/` (enables resume) |
| `--config` | `configs/select.yaml` | YAML with the rules and effort tiers |
| `--dry-run` | `false` | Print the enumerated space, staged plan, and predicted cost only |

```bash
panelcast select --dry-run
panelcast select --dataset examples/aerospace/descriptor.yaml --effort quick
```

---

### `stack` — Predictive Stacking Over a Sweep

Fit predictive-stacking weights (Yao et al. 2018) plus pseudo-BMA+ weights
over a completed sweep's arm ledger from the persisted per-point held-out
log-likelihood snapshots, then score the weighted arm mixture against the
champion single arm and the reference on CRPS, point metrics, coverage, and
WIS. The mixture is a forecast product, not a posterior. Weights are fit on
the primary split's snapshots, so the headline is only ever the secondary
(entity-disjoint) split's predictive snapshots — never the split the weights
were fit on. Arms without snapshots are excluded, not scored another way.
Writes `stacking.md` + `stacking.json` next to the ledger.

```bash
panelcast stack SWEEP_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `SWEEP_DIR` | required | Sweep directory (`outputs/select/<sweep-id>`) containing `ledger.json` |
| `--baselines` | | `baseline_comparison.json` rows for a baseline-floor section in the report |
| `--seed` | `0` | Bayesian-bootstrap seed for the pseudo-BMA+ weights (stacking weights are deterministic) |
| `--out-dir` | sweep dir | Report destination |

```bash
panelcast stack outputs/select/sweep
```

Predictive snapshots (`evaluation/predictive.npz`, ~500 thinned float32
draws per split) are written per arm by the evaluate stage when
`PANELCAST_SAVE_PREDICTIVE=1`; sweeps set it automatically. Older sweeps
without snapshots still get the weights table, but no mixture scoreboard.

---

### `backtest` — Rolling-Origin Backtest

Run the full leakage-safe stage chain (splits → features → train → evaluate)
once per rolling origin and report every headline metric as mean ± SE across
origins (plus min/max), so a headline number is a distribution rather than a
point. Origin *k* holds out each entity's (last-*k*)-th event and drops
everything after it; origin 0 is exactly the standard primary split. Each origin
runs as its own run directory with fresh data stamps, so every leakage control
holds unchanged and each origin's split content hash is recorded. Deeper origins
shrink the eligible entity set, so the aggregate table reports `n_test` and
`n_entities` per origin. A JSON ledger under `outputs/backtest/<id>/` makes a
killed backtest resumable — rerun the same command and completed origins are
skipped. Exits `1` while incomplete origins remain.

```bash
panelcast backtest [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--origins` | `3` | Number of rolling origins K (≥1); origin k holds out each entity's (last-k)-th event |
| `--backtest-id` | `default` | Ledger/report directory name under the output root; reuse the same id to resume |
| `--dataset` | built-in AOTY | Dataset descriptor (bare name or YAML path) |
| `--num-chains` | pipeline default | Chains per origin fit |
| `--num-samples` | pipeline default | Draws per origin fit |
| `--num-warmup` | pipeline default | Warmup per origin fit |
| `--origin-timeout` | none | Per-origin wall-clock timeout in seconds; an origin that exceeds it is killed and marked `timeout` |
| `--output-root` | `outputs/backtest` | Root directory for backtest ledgers/reports |

Writes `backtest_metrics.json`, `backtest_report.md`, and the resumable
`ledger.json` under `outputs/backtest/<id>/`.

```bash
panelcast backtest --origins 3
panelcast backtest --origins 5 --num-chains 2 --num-samples 500
panelcast backtest --backtest-id nightly   # rerun the same id to resume
```

### `replicate` — Machine-Checkable Replication Claims

Grade a domain's `claims.yaml` — the paper's quantitative claims declared as
assertions over posterior quantities — against a fitted run, and print the
verdict table the replication write-ups assemble by hand. Every claim grades
against posterior draws, never point estimates, on the ladder
`match > qualitative > shape_only`. A claim that fails its target rung but
passes a lower one is reported as a **divergence**, not an error: divergences
under a fixed model are findings (the baseball era translation, the chess
peak shift).

```bash
panelcast replicate <pack-dir>                                   # run a domain pack end-to-end
panelcast replicate --all <collection-dir>                       # every pack, one scoreboard
panelcast replicate --claims claims.yaml --models <run>/models   # grade an existing fit
panelcast replicate --claims claims.yaml --dataset domain.yaml   # run the chain, then grade
```

| Option | Default | Description |
|--------|---------|-------------|
| `PACK_DIR` | — | A domain pack folder (contains `pack.yaml`): build the panel if needed (gated on the manifest's `expected_panel`), run the chain with the pack's `fit.yaml`/`run:` overrides, keep pipeline artifacts under the pack's `data/` and `outputs/`, grade its claims, write results to `notes/` |
| `--all` | — | Collection mode: run every immediate non-underscore-prefixed subfolder holding a `pack.yaml`, print one scoreboard; exit code is the worst pack's. Underscore-prefixed folders are reserved for checked-in templates. Rejects `--json` (per-run; each pack writes its own `notes/replicate_verdicts.json`) |
| `--claims` | — | claims.yaml declaring the paper's claims (required with `--models`/`--dataset`) |
| `--models` | — | Models directory of an existing fit (`training_summary.json` + `.nc`) |
| `--dataset` | — | Dataset descriptor: run data → train first, then grade the fresh fit |
| `--allow-unlocked-env` | off | Permit `--dataset` execution without a `pixi.lock`; pack mode already uses its declared lock policy |
| `--json` | — | Also write the verdicts as JSON |

A **domain pack** is the drop-a-folder-and-run unit: `pack.yaml` (manifest —
citation, data provenance, `expected_panel` sanity gate, `run:` overrides),
`descriptor.yaml`, an optional `build.py` (raw deposit → tidy panel, the one
irreducibly per-paper step), optional `fit.yaml` and `claims.yaml`, and
gitignored `data/` and `outputs/` directories. Pack runs use the pack as their
working directory, so intermediates never leak into the caller or another pack.
A relative `raw_path_env` override is therefore pack-relative; use an absolute
path for a deposit stored outside the pack. Pack mode accepts an installed-wheel
environment by default because external packs do not ship panelcast's `pixi.lock`;
set `enforce_lockfile: true` in `fit.yaml` or `run:` when the pack does carry
that lock. Precedence is pack defaults < `fit.yaml` < manifest `run:`.
`panelcast pack new <name>` scaffolds a valid skeleton.
Note the two override vocabularies: the manifest's `run:` block uses
**pipeline config field names** with config-native values (e.g.
`min_ratings: 1`), while `fit.yaml` is a normal pipeline YAML using the
documented **config keys** — prefer `fit.yaml` for anything beyond a scalar
or two.

Named extractors: `group_mean_trend` (slope of the fitted group offsets over
label-ordered groups — right for zero-padded cohort labels like
`"1900s"…"2000s"`; don't use it where labels don't sort into their intended
order), `covariate_coefficient(feature)` (raw-scale coefficient),
`covariate_vertex(linear, quadratic)` (raw-scale peak of a quadratic pair),
`covariate_vertex_difference(linear, quadratic, delta_linear,
delta_quadratic)` (base peak minus the peak after adding interactions — positive
means the interacted group peaks earlier), `entity_contrast` (mean initial-effect
gap between declared entity sets), `entity_ranking(top_k)` (per-draw top-K
membership of a declared set), and
`decline_between_ages(linear, quadratic, a, b)` (covariate-curve change between
two raw values). Vertex differences are ratio quantities: interpret them only
when the reported `P(both curvature<0)` is near 1.

Exit codes: `0` every claim met its target grade; `1` divergences only;
`2` a claim failed every rung (or the chained run failed).

```yaml
claims:
  - name: cohort_improvement
    quantity: group_mean_trend
    expect: {direction: increasing, from: "1900s"}
  - name: peak_age
    quantity: covariate_vertex(age_c, age_sq)
    expect: {in: [30, 40]}
    grade: qualitative
  - name: distance_penalty
    quantity: covariate_coefficient(mean_distance)
    expect: {less_than: 0}
  - name: women_peak_earlier
    quantity: covariate_vertex_difference(age_c, age_sq, age_female, age_sq_female)
    expect: {greater_than: 0}
  - name: elite_premium
    quantity: entity_contrast
    entities: {group_a: [Kasparov, Carlsen], group_b: rest}
    expect: {greater_than: 0, prob: 0.95}
```

---

### `doctor` — Environment & Reproducibility Preflight

Read-only environment and reproducibility check in one screen: lockfile, package
versions + exactness fingerprint, accelerator, compile cache, git state, dataset
resolution, data-root stamps, calibration-store status, and free disk — each
line `PASS` / `WARN` / `FAIL` with a fix hint on anything that isn't a pass.
Exits `1` if any check FAILs, so it drops into CI as a gate.

```bash
panelcast doctor [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | current default | Dataset descriptor to check (bare name or YAML path) |
| `--json` | `false` | Machine-readable output for CI (one object per check); pipeable |

```bash
panelcast doctor
panelcast doctor --dataset examples/aerospace/descriptor.yaml
panelcast doctor --json | jq '.[] | select(.status == "FAIL")'
```

---

### `runs list` — List Pipeline Runs

List run directories under `outputs/` with their manifest summary: creation
time, success status, completed-stage count, and a `*` marking the run the
latest pointer refers to.

```bash
panelcast runs list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `outputs` | Directory holding the pipeline run directories |

---

### `runs history` — Cross-Run Metrics History

Show one row per successful run that completed evaluate: version, `--tag`
label, sampler settings (`chains x samples`), headline metrics (MAE, RMSE,
R², CRPS, coverage@80/95, WIS, elpd-per-obs), and wall-clock. Rows are
grouped by the feature stamp the metrics were computed against — a stamp
change renders as an explicit epoch break, and drift is only ever flagged
within an epoch. Within an epoch a metric cell is flagged `*` when a
coverage level moves more than the coverage tolerance (default 0.03) vs the
epoch's best-MAE run, or when MAE / elpd-per-obs regress more than 2% vs
the epoch best. Corrupt manifests and dry runs are skipped; runs from older
versions render `?` for the missing version.

```bash
panelcast runs history [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `outputs` | Directory holding the pipeline run directories |
| `--json` | `false` | Emit machine-readable JSON (groups with `feature_stamp` + `runs`) |

---

### `runs verify` — Check a Run Against Its Manifest

Re-hash a recorded run's entire provenance chain and exit `1` on anything it
cannot prove, in order: every recorded output, every recorded raw-input hash,
the shared data-root stamps, and the `pixi.lock` hash. Data-root stamps protect
the shared roots *during* a run; `runs verify` protects the whole run directory
*after* it, indefinitely.

Each output gets one line, with the reason after the status:

| Status | Meaning |
|--------|---------|
| `OK` | Present and hashing to the recorded digest |
| `MODIFIED` | Present and hashing to something else |
| `MISSING` | Recorded but not there, or unreadable |
| `UNBOUND` | Recorded at a path outside the run directory and the artifact roots |
| `UNVERIFIABLE` | Nothing to check it against — a path with no hash, a hash with no path, or an empty hash |

A manifest carrying no output hashes at all (pre-0.9.0, or a modern one whose
hash map was emptied) is reported as such and every recorded output comes back
`UNVERIFIABLE`: shape cannot tell the two apart, so neither is excused. A run
that recorded no outputs in the first place has nothing to check and passes.

**Run it from the project root.** A flat-layout run records its data, model and
report artifacts as paths relative to the project root, so `runs verify`
resolves them — and the roots it checks them against — against the working
directory. From an unrelated directory those outputs read as `MISSING`, which
is loud; from **another checkout of the same project** the path and its root
land in that tree together, containment passes, and a reproducible pipeline
reports `OK` against a workspace that is not the one you asked about. A green
result is only evidence about the intended run when the command runs where the
run did. Run-scoped artifacts are unaffected *as long as the output base keeps
its name*: they re-root onto `--output-base` whether or not a copy also sits
under the working directory, so `--output-base` alone decides which artifact is
verified. Moving `outputs/` elsewhere keeps working; renaming it does not, and
a run whose base was renamed reports `UNBOUND` on intact artifacts.

Output verification is the same primitive the incremental `--skip-existing`
path uses (`pipelines/output_integrity.py`), so the per-key rules and the
containment roots are one implementation. The two are scoped differently — a
stage checks only the keys carrying its own `<stage>:` prefix, `runs verify`
checks the whole manifest — which divides *which* keys each is responsible for
rather than what either accepts as proof. Two things differ in that second
sense, for different reasons:

- **Re-rooting.** `runs verify` resolves active and quarantined runs alike and
  cannot know which it has until it looks, so it maps a run-owned recorded
  path onto the run's current directory and a run under `outputs/failed/<id>/`
  still verifies. The skip path only ever follows the active `latest` pointer,
  so it has nothing to map and never looks under `failed/`.
  Run-owned is decided by the output base's *name*, so that a relocated
  workspace still verifies — which also means a manifest from another checkout
  of the same project re-roots here, and for a reproducible pipeline the
  hashes will agree. A green result says the bytes match, not that this
  manifest was written about this workspace.
- **The declared-path binding.** This one *is* about the missing stage
  objects. The skip path knows which paths a stage said it would write, so it
  refuses a manifest that redirects a static output at another file *inside
  the same run directory* — where containment has nothing to say. `runs
  verify` cannot: the manifest does not record which outputs were declared.
  Such a redirect is reported `OK` here (#439).

```bash
panelcast runs verify [RUN_ID] [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `RUN_ID` | `latest` | Run id under `outputs/` or `outputs/failed/` (or `latest`) |
| `--output-base` | `outputs` | Run directory root |

```bash
panelcast runs verify                               # verify the latest run
panelcast runs verify 2026-07-08_120000_000000_abcd
```

---

### `runs show` — Full Provenance of One Run

Render one run's complete provenance in a single screen: command, seed, dataset
(with descriptor hash), git commit/branch/dirty state, environment (python/jax/
numpyro, accelerator, exactness fingerprint, lockfile hash), stage durations,
recorded/hashed output counts, and the headline metrics + coverage when the run
completed evaluate.

```bash
panelcast runs show [RUN_ID] [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `RUN_ID` | `latest` | Run id under `outputs/` or `outputs/failed/` (or `latest`) |
| `--output-base` | `outputs` | Run directory root |

---

### `runs diff` — Compare Two Runs

Compare two runs with defaults-aware semantics: an output-affecting config delta
(only flags that actually differ, resolved against the defaults), generic numeric
metric deltas (B − A) flattened from `metrics.json`, and a run-facts table (seed,
fingerprint, jax/numpyro, accelerator, lockfile, version). Not-like-for-like
diffs are called out first — a differing dataset descriptor hash or git commit
invalidates the metric comparison.

```bash
panelcast runs diff RUN_A RUN_B [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `RUN_A` | (required) | Left run id (or `latest`) |
| `RUN_B` | (required) | Right run id (or `latest`) |
| `--output-base` | `outputs` | Run directory root |

---

### `runs reproduce` — Re-execute a Recorded Run

Re-execute a recorded run from its run directory alone, then compare. The config
is rebuilt from the run's `resolved_config.yaml` (falling back to the manifest
flags for pre-0.9.0 runs — weaker provenance). Two guards run before any compute:
the dataset descriptor must still hash-match the recorded one, and the recorded
raw inputs must be unchanged on disk, or it aborts (exit `1`). The environment
fingerprint frames the expectation up front — bit-exact outputs within a matching
fingerprint, statistical reproduction otherwise — and the post-run comparison
follows suit (exact output-hash match vs headline-metric deltas). A reproduction
always runs fresh: never resumes, never skips.

```bash
panelcast runs reproduce RUN_ID [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `RUN_ID` | (required) | Run id under `outputs/` (or `latest`) |
| `--output-base` | `outputs` | Run directory root for both the old and new runs |

---

### `runs why` — Explain a Failed Run

Explain a failed run from its `failure.json`: the stage that failed, the
exception type and message, an actionable hint, the exact resume command, the
traceback tail, and the last ~10 structured log events. With no argument it picks
the most recent run under `outputs/failed/`. Pre-0.9.0 failures with no
`failure.json` fall back to the manifest's error field plus a resume command.

```bash
panelcast runs why [RUN_ID] [OPTIONS]
```

| Argument / Option | Default | Description |
|-------------------|---------|-------------|
| `RUN_ID` | most recent failed | Failed run id (default: the newest run under `outputs/failed/`) |
| `--output-base` | `outputs` | Run directory root |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AOTY_DATASET_PATH` | Path to raw CSV dataset (required) |
| `WSL_DISTRO_NAME` | Auto-detected for WSL2 GPU memory adjustments |
| `WSL_INTEROP` | Auto-detected for WSL2 GPU memory adjustments |

### Setting AOTY_DATASET_PATH

```bash
# Unix/Linux/macOS
export AOTY_DATASET_PATH="/path/to/dataset.csv"

# Windows PowerShell
$env:AOTY_DATASET_PATH = "C:\path\to\dataset.csv"

# Windows CMD
set AOTY_DATASET_PATH=C:\path\to\dataset.csv
```

Or create a `.env` file (see `.env.example`):

```bash
AOTY_DATASET_PATH="path/to/your/dataset.csv"
```

---

## Exit Codes

| Code | Name | Description |
|------|------|-------------|
| `0` | Success | Pipeline completed successfully |
| `1` | General Error | Unspecified error / preflight FAIL |
| `2` | Convergence Error | MCMC convergence failure (R-hat, ESS, divergences) for normal pipeline runs |
| `3` | Data Validation Error | Input data validation failure |
| `4` | Stage Error | General stage execution error |
| `5` | Environment Error | Environment verification failure (e.g., missing `pixi.lock`) |
| `6` | GPU Memory Error | GPU memory check failure |
| `7` | Stale Artifact Error | Shared data artifacts (`data/processed`, `data/splits`, `data/features`) were regenerated by another run mid-flight |

**Note:** When using `--preflight-only`, exit code 2 indicates CANNOT_CHECK status (unable to check GPU memory). See [Preflight-Specific Exit Codes](#preflight-specific-exit-codes) below for the full mapping.

### Preflight-Specific Exit Codes

When using `--preflight-only`:

| Code | Status | Meaning |
|------|--------|---------|
| `0` | PASS | Sufficient GPU memory available |
| `1` | FAIL | Insufficient GPU memory |
| `2` | CANNOT_CHECK | Unable to check (no GPU, NVML unavailable, missing data) |

---

## Configuration Files

`panelcast` runtime behavior is driven by CLI flags, environment variables (for
example `AOTY_DATASET_PATH`), and optional YAML configs:

- `--preset {quick,dev,diagnostic,publication}` is sugar for `--config
  configs/<preset>.yaml`, layered first (any `--config` files and CLI options
  still win).
- `--config` / `-c` loads one or more YAML files of `PipelineConfig` keys
  (repeatable; later files win). Explicit CLI options always override YAML.
- `--dataset` loads a dataset descriptor (a bare name resolves to
  `configs/datasets/{name}.yaml`, or pass a YAML path) to retarget the pipeline
  to another domain — see `docs/PORTING.md`.

---

## Typical Workflows

### First-Time Setup

```bash
# 1. Set dataset path
export AOTY_DATASET_PATH="/path/to/aoty_data.csv"

# 2. Check GPU memory
panelcast run --preflight-only

# 3. Run full pipeline
panelcast run
```

### Development Iteration

```bash
# Quick test run
panelcast run --num-chains 1 --num-samples 500 --allow-divergences

# Run specific stages
panelcast stage train --verbose
panelcast stage evaluate
panelcast stage report
```

### Publication Run

```bash
# Full quality run
panelcast run --num-chains 8 --num-samples 2000 --target-accept 0.95 --strict

# Export figures
panelcast export-figures --formats svg,png,pdf --scale 3.0
```

### Preflight Checks (CI/Scripting)

```bash
# Quick estimate check (~1s, formula-based)
panelcast run --preflight-only
echo "Exit code: $?"
```

```bash
# Accurate measured check (~30-60s, mini-MCMC)
panelcast run --preflight-full --preflight-only
echo "Exit code: $?"
```

**Flag behavior:** `--preflight-full` determines the check type (full measured check vs quick formula-based). `--preflight-only` controls whether to exit after the check or continue with the pipeline.

---

## See Also

- `docs/PIPELINE_RUNBOOK.md` — End-to-end pipeline instructions
- `docs/DEV_SETUP.md` — Environment and test setup
- `docs/CONFIG_SPEC.md` — Configuration file specification
