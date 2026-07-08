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
- Publication helper script
  ```powershell
  .\scripts\run_publication.ps1
  ```

Reproducibility notes
- `pixi.lock` is required by default.
- Use `--allow-unlocked-env` only for local exploratory runs.

Expected outputs
- `data/processed/*`
- `data/features/*`
- `outputs/<run_id>/*` (run manifest, models, evaluation, predictions, reports)
- `outputs/<run_id>/evaluation/*`
- `outputs/<run_id>/reports/tables/*`
- `outputs/<run_id>/reports/figures/*`
- `outputs/latest.json` (pointer to the latest successful run)
