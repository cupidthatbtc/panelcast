# Evaluation Protocol

Protocol terms are generic (entities and their sequential events); in the AOTY
example domain an entity is an artist and an event is an album.

Metrics
- R2 on held-out test sets (primary and secondary splits)
- RMSE and MAE
- Calibration curves and coverage of credible intervals (80% and 95%)
- CRPS (proper scoring rule for probabilistic regression)

Cross-validation
- Primary evaluation: within-entity temporal holdout (last event per entity)
- Secondary evaluation: entity-disjoint split (no entity overlap)
- Secondary split uses cold-start predictive path (population-level entity effect)
- Secondary split sets `prev_score` to training global mean for every row (no held-out label usage)
- Primary split fails fast if unknown entities appear in test data (no silent row dropping)

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
- Secondary split reports info-criteria as unavailable when using the new-entity predictive path

Sensitivity analyses (opt-in `sensitivity` stage, not part of a default run)
- Min-ratings thresholds (5, 10, 25)
- Prior variants (default, diffuse, informative; `PRIOR_CONFIGS` in `src/panelcast/pipelines/sensitivity.py`)
- Feature ablations (remove feature groups to measure importance)
- Split-seed axis (seeds 42/43)
