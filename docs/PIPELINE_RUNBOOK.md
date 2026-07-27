# Pipeline Runbook

Environment setup (recommended)
- Install `pixi`
- Run `pixi install` from repo root

End-to-end commands
- Full pipeline (strict reproducibility + strict diagnostics)
  ```bash
  panelcast run --strict
  ```
- Full pipeline with tuned MCMC settings
  ```bash
  panelcast run --strict --num-chains 4 --num-samples 2000 --num-warmup 2000 --target-accept 0.95
  ```
- Stage-wise execution
  ```bash
  panelcast stage data
  panelcast stage splits
  panelcast stage features
  panelcast stage train --strict
  panelcast stage evaluate
  panelcast stage predict
  panelcast stage report
  ```
  Each invocation creates its own run directory for what it writes. A stage
  whose upstream product (models, evaluation, predictions) is not produced in
  the same invocation reads it from the most recent successful run that has
  it; the log records which run supplied each product.
- Publication helper script
  ```powershell
  .\scripts\run_publication.ps1
  ```

Reproducibility notes
- `run` and `replicate --dataset` require `pixi.lock` by default; use
  `--allow-unlocked-env` only for an intentional installed-wheel run.
- External domain packs accept their installed environment by default, record
  the missing lock hash, and can require a shipped lock through `fit.yaml` or
  manifest `run:`.

Expected outputs
- `data/processed/*`
- `data/features/*`
- `outputs/<run_id>/*` (run manifest, models, evaluation, predictions, reports)
- `outputs/<run_id>/evaluation/*`
- `outputs/<run_id>/reports/tables/*`
- `outputs/<run_id>/reports/figures/*`
- `outputs/latest.json` (pointer to the latest successful run)
