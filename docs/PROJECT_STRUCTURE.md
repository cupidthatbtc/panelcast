# Project Structure

Top-level
- src/panelcast: core package
- configs: configuration files
- data: raw/interim/processed artifacts
- outputs: run-specific manifests and runtime artifacts
- reports: figures and tables
- docs: project documentation
- citations: bibliography and citation notes
- tests: unit/integration/e2e tests
- scripts: helper entrypoints

Package layout
- config: config schema and loader
- io: read/write helpers
- data: ingestion, cleaning, validation, splitting, lineage
- features: one-file-per-block feature modules, registry, and pipeline helper
- models/bayes: model definitions, priors, fit, predict, diagnostics
- evaluation: metrics, CV, calibration
- reporting: tables, figures, model cards
- pipelines: end-to-end workflows
- pipelines/build_features.py: feature matrix cache builder
- utils: logging, hashing, random seeds
