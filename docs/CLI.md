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
| `--min-ratings` | `10` | ≥1 | Minimum user ratings per album |
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
| `report` | Generate publication artifacts (figures, tables, model cards) |

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

`panelcast` runtime behavior is currently driven by CLI flags and environment
variables (for example `AOTY_DATASET_PATH`). YAML files under `configs/` are
project reference artifacts and are not loaded by the CLI command interface.

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
