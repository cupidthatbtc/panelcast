# Evaluation Protocol

Metrics
- R2 on held-out test sets (primary and secondary splits)
- RMSE and MAE
- Calibration curves and coverage of credible intervals (80% and 95%)
- CRPS (proper scoring rule for probabilistic regression)

Cross-validation
- Primary evaluation: within-artist temporal holdout (last album per artist)
- Secondary evaluation: artist-group split (no artist overlap)
- Secondary split uses cold-start predictive path (population-level artist effect)
- Secondary split sets `prev_score` to training global mean for every row (no held-out label usage)
- Primary split fails fast if unknown artists appear in test data (no silent row dropping)

Diagnostics
- R-hat <= 1.01 for all key parameters
- ESS above threshold
- No divergent transitions after tuning
- Coverage must be within configured tolerance (`coverage_tolerance`)

Model comparison
- LOO and WAIC computed for the primary split when pointwise log-likelihood is available
- Secondary split reports info-criteria as unavailable when using new-artist predictive path
