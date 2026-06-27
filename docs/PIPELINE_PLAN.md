# Pipeline Plan (Publication-Focused)

This plan defines the full rebuild for Bayesian entity-level prediction. It is
written against the AOTY example domain, where an entity is an artist and an
event is an album; the structural steps apply to any domain's entities/events.

Phase 0 - Governance and Guardrails
- Define the prediction target: next event score for an entity (AOTY: next album user score for an artist).
- Lock leakage rules (see docs/LEAKAGE_CONTROLS.md).
- Define split policy:
  - Primary: within-entity temporal holdout (last event per entity).
  - Secondary: entity-disjoint split (no entity overlap).
- Define the minimal publication metrics and diagnostics.

Phase 1 - Data Ingestion
- Input: all_albums_full.csv.
- Validate schema (expected columns and types).
- Rename raw headers to canonical names (see RAW_TO_CANONICAL).
- Normalize string fields (artist, album, genre, descriptors).
- Deduplicate records deterministically.
- Record data hash and row counts in lineage logs.

Phase 2 - Filtering and Cleaning
- Apply min ratings thresholds for dataset generation (5/10/25); default pipeline split input uses 10.
- Drop rows with missing user score or critical numeric fields.
- Repair runtime or track counts when possible; log repairs.
- Generate explicit exclusion reasons per row for audit.

Phase 3 - Feature Engineering
- Core numeric features: ratings, critic score, review counts, track stats.
- Temporal features: release year, time since debut, album index per artist.
- Category encoders: genre multi-membership.
- Artist features: leave-one-out reputation and history-only aggregates.
- Album type dummies (Album, EP, Mixtape, Compilation).
- Collaboration features: artist-count and collaboration type indicators.

Phase 3b - Feature Block Assembly
- Assemble feature blocks per config (features.blocks).
- Persist combined feature matrix to data/features.
- Save manifest with block metadata and feature names.

Phase 4 - Leakage-Safe Splits
- Primary: within-entity temporal holdout (last event per entity with >=2 events).
- Secondary: entity-disjoint split (no entity overlap, fixed seed).
- Store split manifests in data/splits for both strategies.
- Use train-only statistics for imputation and scaling.
- For time-based checks: train on early years, test on later years.

Phase 5 - Imputation and Missing Data
- Numeric imputation hierarchy: artist -> genre -> decade -> global (train only).
- Track imputation source for every imputed value.
- Record imputation rates; if high, trigger sensitivity runs.

Phase 6 - Modeling (Bayesian)
- Baseline hierarchical model: global + genre + artist random effects.
- Dynamic slope model: artist-specific time slopes.
- Priors: weakly informative; validate with prior predictive checks.
- Sampling: default tune/draws, increase for publication run.

Phase 7 - Diagnostics and Model Comparison
- Convergence: R-hat <= 1.01, ESS thresholds.
- Posterior predictive checks (PPC) on held-out data.
- WAIC/LOO with robust error handling.
- Flag NaN or divergent samples and record failure modes.

Phase 8 - Prediction Output
- For each entity, predict the next event score with credible intervals.
- Output predictions with uncertainty and metadata.
- Store in `outputs/predictions/` and summarize in `reports/`.

Phase 9 - Sensitivity Analyses
- Vary min ratings threshold.
- Vary priors and model families.
- Compare with/without artist effects and genres.
- Document stability outcomes.

Phase 10 - Publication Artifacts
- Tables: summary stats, model comparisons, main effects.
- Figures: calibration, PPC, uncertainty by artist history.
- Model card and reproducibility report.

Phase 11 - Reproducibility
- Store exact configs, seeds, dataset hashes, and package versions.
- Provide a single publication pipeline entry point.

Outputs (expected)
- data/processed/cleaned_all.parquet
- data/processed/user_score_minratings_10.parquet
- data/splits/*/manifest.json
- data/features/manifest.json
- outputs/<run_id>/manifest.json
- outputs/evaluation/metrics.json
- outputs/predictions/next_event_known_entities.csv
- reports/tables/*.csv
- reports/figures/*.png
- reports/MODEL_CARD.md

Legacy reference
- docs/lineage/DATA_LINEAGE_DETAILED.md is copied from the old project for step parity.
