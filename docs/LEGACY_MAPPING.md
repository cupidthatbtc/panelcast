# Legacy Mapping

Use docs/lineage/DATA_LINEAGE_DETAILED.md as the authoritative step map.

Suggested mapping to new modules
- Step 1-3 (loading, dedup, filtering): src/panelcast/data/ingest.py + cleaning.py
- Step 4-4b (feature engineering, reputation): src/panelcast/features/*
- Step 5 (split): src/panelcast/data/split.py
- Step 6 (imputation): src/panelcast/data/cleaning.py
- Step 7-8 (encoding and PCA): src/panelcast/features/genre.py + pca.py
- Step 9 (CV): src/panelcast/evaluation/cv.py
- Step 10-13 (modeling and outputs): src/panelcast/models/bayes + reporting