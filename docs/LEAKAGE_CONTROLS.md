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
- `gbm_offset` uses `entity_aware_temporal_v1` by default for panel data. A
  prospective row is trained only from observations before its canonical cutoff;
  a cold-start row excludes its entity entirely. The deployment GBM is refit on
  all training observations admissible before held-out prediction. Feature
  manifests retain each OOF row's protocol, cutoff, entity overlap, and hashes of
  fit/held rows. Missing dates sort after dated observations and are scored
  together without using another missing-date row as history. Direct non-panel
  construction without entity/date columns is the
  explicit legacy KFold migration path.

Artifacts to store
- data/splits/within_entity_temporal/manifest.json
- data/splits/entity_disjoint/manifest.json
- data/splits/within_entity_temporal/split_*.json (versioned archive)
- data/splits/entity_disjoint/split_*.json (versioned archive)
- data/audit/summary_<run_id>.json and data/audit/exclusions_<run_id>.jsonl
- outputs/<run_id>/manifest.json
- outputs/<run_id>/dataset_hash.txt
