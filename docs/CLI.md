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
| `--max-albums` | `50` | ≥1 | Maximum albums per artist |

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
| `--min-albums` | `2` | ≥1 | Minimum albums per artist for dynamic effects |

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
| `--likelihood-family` | `studentt` | Observation likelihood: `studentt`, `normal`, `skew_studentt`, `skew_normal`, `split_normal`, `beta`, `mixture`, `beta_binomial`, or `beta_ceiling`. See [`LIKELIHOOD_CANDIDATES.md`](LIKELIHOOD_CANDIDATES.md) |
| `--discretize-observation` | `false` | Interval-censor the observation to integers (honest PPC for integer scores). Location-scale families only (`studentt`, `normal`, `skew_normal`, `split_normal`, `mixture`); rejected for `skew_studentt`, `beta`, `beta_binomial` (`beta_binomial` is already discrete) |

#### Domain & Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` / `-c` | | YAML config file(s) with `PipelineConfig` keys; repeatable, later files win. Explicit CLI options always win over YAML. |
| `--dataset` | built-in AOTY | Dataset descriptor: bare name (resolves to `configs/datasets/{name}.yaml`) or YAML path |
| `--debut-prev-score-source` | `train_mean` | Debut `prev_score` fill: `train_mean` or `dataset_stats` (legacy; mild leakage) |
| `--target-transform` | `offset_logit` | Score-scale transform: `offset_logit` (default since 0.5.0; the model runs on the Smithson-Verkuilen logit scale, bounds hold by construction) or `identity` (soft-clip; the former default, still selectable) |
| `--ar-center` | `global` | AR(1) centering: `global`, `none` (legacy), or `artist_running` (sensitivity only) |
| `--latent-process` | `rw` | Artist-effect process: `rw` (random walk) or `ar1` (stationary). Experimental |
| `--exclude-rw-raw-from-collection` | `false` | Don't store `rw_raw` draws on device (~96% peak-GPU cut); required for the 4-chain publication run on 24 GB GPUs |

#### MCMC (advanced)

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--chain-method` | `sequential` | | `sequential`, `vectorized`, or `parallel` (multi-GPU) |
| `--max-tree-depth` | `10` | 5–15 | Maximum NUTS tree depth |
| `--likelihood-df` | `4.0` | ≥1.0 | Student-t degrees of freedom; ≥100 ≈ Normal |

#### Splits & Calibration

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--val-albums` | `0` | ≥0 | Albums per artist held out for validation (0 = none) |
| `--min-train-albums` | `2` | ≥1 | Minimum training albums per artist |
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
| `--output` | `-o` | `reports/diagnostics` | Output directory for the report |

```bash
panelcast diagnose
panelcast diagnose --eval-dir outputs/2026-06-23_192630/evaluation
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
| `--arm-timeout` | `1800` | Per-arm wall-clock timeout in seconds, or `auto` to size each arm's timeout from its predicted runtime (max of an 1800s floor and 3x the transform-aware prediction); a fit exceeding it is killed and marked failed |
| `--sweep-id` | `sweep` | Sweep directory name under `outputs/select/` (enables resume) |
| `--config` | `configs/select.yaml` | YAML with the rules and effort tiers |
| `--dry-run` | `false` | Print the enumerated space, staged plan, and predicted cost only |

```bash
panelcast select --dry-run
panelcast select --dataset examples/aerospace/descriptor.yaml --effort quick
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
