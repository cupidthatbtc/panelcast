# Leakage Controls

Rules use AOTY's "artist" entity name; they apply to any domain's entities.
Enforcement lives in `pipelines/create_splits.py` and `data/split.py` (split
construction and the artist/group-disjointness checks) and in the feature
pipeline (`features/pipeline.py`, train-only fitting and label masking).

Mandatory rules
- No test data used for imputation, scaling, or feature selection.
- Group splits must prevent artist overlap across train/val/test.
- Within-artist temporal holdouts must use only prior albums for features.
- Leave-one-out artist features must exclude the target album.
- Validation/test rows must mask held-out score labels during feature transforms.
- Artist-disjoint cold-start evaluation must not derive `prev_score` from held-out labels.
- Category vocabularies must be fit on train only.
- CV folds must be group-aware and nested when tuning.

Artifacts to store
- data/splits/within_entity_temporal/manifest.json
- data/splits/entity_disjoint/manifest.json
- data/splits/within_entity_temporal/split_*.json (versioned archive)
- data/splits/entity_disjoint/split_*.json (versioned archive)
- data/audit/summary_<run_id>.json and data/audit/exclusions_<run_id>.jsonl
- outputs/<run_id>/manifest.json
- outputs/<run_id>/dataset_hash.txt
