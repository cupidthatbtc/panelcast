# Artifacts

Data roots are a deterministic cross-run cache and stay flat; mutable products
(models, evaluation, predictions, reports) are scoped under a timestamped run
directory `outputs/<run_id>/`, with `outputs/latest.json` pointing at the most
recent successful run (failed runs move to `outputs/failed/<run_id>/`).

Data (shared cross-run cache)
- data/raw/all_albums_full.csv (raw input)
- data/processed/cleaned_all.parquet
- data/processed/user_score_minratings_*.parquet
- data/features/within_entity_temporal/*.parquet
- data/features/entity_disjoint/*.parquet
- data/features/manifest.json
- data/splits/within_entity_temporal/manifest.json
- data/splits/entity_disjoint/manifest.json
- data/splits/within_entity_temporal/split_*.json (immutable archive)
- data/splits/entity_disjoint/split_*.json (immutable archive)
- data/audit/* (cleaning/exclusion provenance)

Runs
- outputs/<run_id>/manifest.json
- outputs/<run_id>/dataset_hash.txt
- outputs/<run_id>/pipeline.log.json
- outputs/latest.json (pointer to the latest successful run)

Models
- outputs/<run_id>/models/user_score_*.nc (ArviZ InferenceData, NetCDF)
- outputs/<run_id>/models/manifest.json (current models + history)
- outputs/<run_id>/models/training_summary.json

Evaluation
- outputs/<run_id>/evaluation/metrics.json
- outputs/<run_id>/evaluation/diagnostics.json
- outputs/<run_id>/evaluation/within_entity_temporal/predictions.json
- outputs/<run_id>/evaluation/within_entity_temporal/calibration.json
- outputs/<run_id>/evaluation/entity_disjoint/predictions.json
- outputs/<run_id>/evaluation/entity_disjoint/calibration.json

Predictions
- outputs/<run_id>/predictions/next_event_known_entities.csv
- outputs/<run_id>/predictions/next_event_new_entity.csv
- outputs/<run_id>/predictions/prediction_summary.json

Reports
- outputs/<run_id>/reports/artifact_status.json
- outputs/<run_id>/reports/tables/*.csv
- outputs/<run_id>/reports/figures/*.png
- outputs/<run_id>/reports/MODEL_CARD.md (run-generated; the repo-root
  MODEL_CARD.md is the curated card)
