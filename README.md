# panelcast

[![CI](https://github.com/cupidthatbtc/panelcast/actions/workflows/ci.yml/badge.svg)](https://github.com/cupidthatbtc/panelcast/actions/workflows/ci.yml)
[![Nightly](https://github.com/cupidthatbtc/panelcast/actions/workflows/nightly.yml/badge.svg)](https://github.com/cupidthatbtc/panelcast/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/cupidthatbtc/panelcast/branch/main/graph/badge.svg)](https://codecov.io/gh/cupidthatbtc/panelcast)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![pixi](https://img.shields.io/badge/pixi-package%20manager-brightgreen)](https://pixi.sh)
![Status: experimental](https://img.shields.io/badge/status-experimental-orange)

> **⚠️ Experimental — real-data validated on a subset; full corpus pending.**
>
> The reproducibility, diagnostics, and domain-portability scaffolding is the
> finished part. The headline *statistical* result is now **partially
> established on real data**: on a representative ~800-artist / ~5k-album AOTY
> subset (skewness −2.08), the model **passes the convergence gate** at the
> publication configuration (R-hat 1.00, bulk ESS 3,504, 0 divergences), and
> the baseline benchmark runs on the same real splits. Still open: the
> posterior-predictive p-values stay pinned at the extremes by a
> symmetric-likelihood / left-skewed-target mismatch (a bounded-Beta candidate
> was tested on real data and **rejected**), and this is a subset, not the full
> ~62k-album corpus. See [`MODEL_CARD.md`](MODEL_CARD.md) and
> [`docs/LIKELIHOOD_CANDIDATES.md`](docs/LIKELIHOOD_CANDIDATES.md). Treat the
> subset numbers as real but not final.

**Hierarchical Bayesian prediction for bounded scores of events nested in entities over time — configured by one YAML descriptor.**

Lots of forecasting problems share a shape: *entities* accumulate a history of
*events*, each event carries a *bounded score* and a noisy *observation count*,
and you want to predict the next score. Musicians release albums rated 0–100.
Airframes fly test flights scored 0–10. Candidates contest elections with a
vote share in [0, 1]. panelcast models that shape once — partial pooling across
entities, a time-varying entity effect, album-to-album (event-to-event)
dependence, and review-count-scaled noise — and lets you point it at a new
domain with a single descriptor file and **zero source changes**.

The emphasis is the infrastructure *around* the model as much as the model
itself: leakage controls, data lineage, preflight gates, and
convergence/calibration diagnostics as first-class, gating checks.

## Domains

Every dataset-specific name (columns, target bounds, date formats, posterior
prefixes, feature blocks) flows through a single `DatasetDescriptor`. Each field
**defaults to its AOTY value**, so a new domain only states what differs.

| Domain | Entity → Event | Bounded score | Status |
|---|---|---|---|
| **Album of the Year** (flagship) | Artist → Album | `User_Score` ∈ [0, 100] | Built-in defaults + the `aoty` feature pack (genre, album-type, collaboration). Run with no `--dataset` flag. |
| **Aerospace** (worked example) | Airframe → Test flight | `Perf_Score` ∈ [0, 10] | Bundled descriptor `configs/datasets/aero.yaml` + end-to-end portability test. One YAML, no music-specific code. |
| **US elections** | Candidate/seat → Contest | Vote share ∈ [0, 1] | Sibling project (`elections_pred`) that retargets this pipeline — lives in its own repo, not bundled here. |

The contract that `--dataset aoty_full` is byte-identical to running with no flag
at all is enforced by `tests/e2e/test_domain_portability.py`. See
[`docs/PORTING.md`](docs/PORTING.md) for the full walkthrough.

## Model structure

Hierarchical partial pooling across entities; a time-varying entity effect via a
Gaussian random walk; AR(1) event-to-event dependence; heteroscedastic
observation noise scaled by observation count; non-centered parameterization
(`LocScaleReparam`) plus a sigma-ref reparameterization to break the
multiplicative funnel; Student-t likelihood with a soft-clip to the target
bounds. Optional per-entity overdispersion with a lognormal variance prior is
available behind a gate. Built on [NumPyro](https://num.pyro.ai/) / JAX.

## Install

**Prerequisites:** Python ≥ 3.11. [pixi](https://pixi.sh) is the supported
environment manager (it pins the full conda + PyPI stack, including JAX).

```bash
# Install pixi if needed
curl -fsSL https://pixi.sh/install.sh | bash

git clone https://github.com/cupidthatbtc/panelcast.git
cd panelcast
pixi install                 # resolves the locked environment
pip install -e .             # install the panelcast package + CLI into it

panelcast --help
```

> **pixi is the supported, reproducible path.** `pixi.lock` pins the exact,
> tested versions of the whole stack (notably the tightly-coupled jax/numpyro
> pair). A standalone `pip install -e .` outside pixi is best-effort: the
> dependency bounds in `pyproject.toml` cap known-breaking majors, but they do
> not guarantee an identical environment. For anything reproducibility-sensitive,
> use pixi.

## 60-second quickstart (aerospace example)

Retarget the whole pipeline to a non-music domain with no code changes, using
the bundled synthetic aerospace dataset (committed under `examples/aerospace/`:
8 airframes flying ~39 sequential test flights scored 0–10):

```bash
# Run the entire pipeline end-to-end on the example, at tiny scale
panelcast demo
```

`demo` reads `examples/aerospace/descriptor.yaml` — one file that remaps the
columns, switches the score bounds to [0, 10], drops the music-specific feature
packs, and adds the domain's own numeric covariates — and runs data → splits →
features → train → evaluate → predict → report, finishing with a generated
model card under `reports/`. The model code is untouched.

The committed CSV is regenerated from the shared synthetic generator with
`python scripts/generate_aero_example.py`. To benchmark the model against simple
baselines on the splits it just produced:

```bash
panelcast compare --baselines --dataset examples/aerospace/descriptor.yaml
```

To run the flagship AOTY domain instead, point at your data and omit `--dataset`:

```bash
export AOTY_DATASET_PATH="/path/to/aoty_data.csv"
panelcast run --preflight-only      # GPU-memory / schema / calibration gate
panelcast run                       # full pipeline
panelcast stage train --verbose     # or run a single stage
```

See [`docs/CLI.md`](docs/CLI.md) for the complete command reference.

## Features

- Leak-safe data pipeline and evaluation (within-entity temporal split + an
  entity-disjoint secondary check)
- Explicit data contract and lineage from raw CSV to final artifacts
- Preflight gates (GPU memory, schema validation, calibration) before expensive runs
- Convergence + PPC + coverage diagnostics as first-class, gating checks
- Sensitivity matrix over priors, splits, and feature ablations
- Publication-ready artifacts: tables, figures, model card, citations
- Domain portability proven by an end-to-end test, not just asserted

## Documentation

- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — step-by-step startup guide (start here)
- [`docs/PORTING.md`](docs/PORTING.md) — retarget to a new domain (the aerospace walkthrough)
- [`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md) — adding features safely
- [`docs/CLI.md`](docs/CLI.md) — complete CLI reference
- [`docs/LEAKAGE_CONTROLS.md`](docs/LEAKAGE_CONTROLS.md) — guardrails and leakage prevention
- [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) — metrics, diagnostics, and thresholds
- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) — directory and file layout
- [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — raw schema and cleaned artifacts
- [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, results, and limitations

A note on results: the figures in `MODEL_CARD.md` are from a validation-scale
run that **fails the convergence gate**, and should be treated as indicative
until the full 4×5000 publication run is executed:

```bash
panelcast run --preset publication        # 4 chains × 5000, bounded Beta likelihood
panelcast diagnose                        # convergence + PPC of that run
panelcast compare --baselines             # the model vs. simple baselines
```

That run is the remaining compute-bound step (it needs the full AOTY dataset and
a GPU). What *is* resolved: leak-safe splits with role-based names, an honest
baseline comparison (`panelcast compare`), and a bounded Beta likelihood that
fixes the symmetric-likelihood / left-skewed-target mismatch by construction
(see [`docs/LIKELIHOOD_CANDIDATES.md`](docs/LIKELIHOOD_CANDIDATES.md)). The code,
the diagnostics, and the honest naming of what is and isn't resolved are the
point.

## License

MIT License. See [LICENSE](LICENSE) for details.
