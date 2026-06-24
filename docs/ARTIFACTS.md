# Artifacts

Data
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

Reports
- reports/artifact_status.json
- reports/tables/*.csv
- reports/figures/*.png
- MODEL_CARD.md

Evaluation
- outputs/evaluation/metrics.json
- outputs/evaluation/diagnostics.json
- outputs/evaluation/within_entity_temporal/predictions.json
- outputs/evaluation/within_entity_temporal/calibration.json
- outputs/evaluation/entity_disjoint/predictions.json
- outputs/evaluation/entity_disjoint/calibration.json
