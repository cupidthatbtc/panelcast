# Project Structure

Top-level
- `src/panelcast`: core package
- `configs`: dataset descriptors and pipeline config YAML
- `data`: raw/interim/processed artifacts and splits (created at runtime)
- `outputs`: run-specific manifests and evaluation artifacts (created at runtime)
- `reports`: figures, tables, and model cards (created by the report stage)
- `docs`: project documentation
- `citations`: bibliography and citation notes
- `tests`: unit/integration/e2e tests
- `scripts`: helper entrypoints (e.g. `verify_gpu.py`)

Package layout (`src/panelcast/`)
- `config`: dataset descriptor schema, YAML loader, and pipeline-config overrides
- `data`: ingestion, cleaning, validation, alignment, splitting, lineage/audit, manifests
- `evaluation`: metrics, cross-validation, calibration, posterior/prior predictive checks
- `features`: one-file-per-block feature modules, registry, and pipeline helper
- `features/packs/aoty`: the music-domain feature pack (genre, album type, collaboration)
- `gpu_memory`: GPU memory estimation, measurement, and platform/NVML queries
- `io`: path resolution and read/write helpers
- `models/bayes`: model definition, priors, transforms, fit, predict, diagnostics, model I/O
- `pipelines`: end-to-end stages and orchestration (data → splits → features → train → evaluate → predict → report; opt-in sensitivity)
- `preflight`: pre-run memory checks (quick estimate and mini-MCMC measurement) with caching
- `reporting`: tables, figures, and model-card generation
- `utils`: logging, hashing, random seeds, git state, environment checks
- `visualization`: charts, multi-panel dashboard figure, static figure export, theme
