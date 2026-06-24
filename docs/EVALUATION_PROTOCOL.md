# Evaluation Protocol

Protocol terms use AOTY's entity/event names ("artist"/"album"); they apply to
any domain's entities and events.

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

History cap (`max_albums`)
- `--max-albums` (default 50 for AOTY) caps the length of the time-varying
  trajectory per entity. It is a max-EVENTS cap, not a row filter: an entity's
  events beyond the most recent `max_albums` are NOT dropped — they collapse
  onto sequence position 1 (the initial entity effect), so every row still
  contributes to the likelihood. The cap bounds the random-walk trajectory
  length (and peak GPU memory); distant positions carry little signal about the
  current state because cumulative random-walk variance grows over steps.
- Domains with longer histories than AOTY should raise `--max-albums`
  accordingly. The cap is computed on training data only (no leakage).

Diagnostics
- R-hat <= 1.01 for all key parameters (`rhat_threshold`, default 1.01)
- ESS >= 400 per chain (`ess_threshold`, default 400)
- No divergent transitions (`allow_divergences` is false by default)
- Coverage must be within configured tolerance (`coverage_tolerance`, default 0.03)
- Posterior predictive checks are informational, not gating: with AOTY's known
  left-skewed target against the symmetric likelihood, several PPC p-values are
  expected to pin at 0.000/1.000 (see `MODEL_CARD.md`)

Model comparison
- LOO and WAIC computed for the primary split when pointwise log-likelihood is available
- Secondary split reports info-criteria as unavailable when using new-artist predictive path
