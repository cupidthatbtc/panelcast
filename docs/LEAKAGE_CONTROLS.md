# Leakage Controls

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
- data/splits/within_artist_temporal/manifest.json
- data/splits/artist_disjoint/manifest.json
- data/splits/within_artist_temporal/split_*.json (versioned archive)
- data/splits/artist_disjoint/split_*.json (versioned archive)
- data/audit/summary.csv
- outputs/<run_id>/manifest.json
- outputs/<run_id>/dataset_hash.txt
