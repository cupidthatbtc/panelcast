# Data Contract

This is the data contract for the **bundled AOTY example domain**. The raw →
canonical column mapping is descriptor-driven: AOTY uses the built-in defaults
(below), and other domains (e.g. `aero`) define their own `raw_column_map`,
bounds, and names in their descriptor YAML under `configs/datasets/`. The
processed-dataset names here (`user_score_minratings_*`) come from the
descriptor's `processed_name_template`; the split-directory names
(`within_entity_temporal`, `entity_disjoint`) are fixed, role-based, and
domain-agnostic (see `docs/PORTING.md`). Pre-rename artifacts written with the
old AOTY-flavored split names (`within_artist_temporal`, `artist_disjoint`)
still load via a backward-compatible alias. See `docs/EXTENSIBILITY.md` for
retargeting.

Raw CSV (actual columns from `all_albums_full.csv`)
- Artist
- Album
- Year
- Release Date
- Genres
- Critic Score
- User Score
- Avg Track Score
- User Ratings
- Critic Reviews
- Tracks
- Runtime (min)
- Avg Track Runtime (min)
- Label
- Descriptors
- Album URL
- All Artists
- Album Type

Required columns (baseline user-score pipeline)
- Artist, Album, Year, Release Date, Genres
- User Score, User Ratings
- Tracks, Runtime (min), Avg Track Runtime (min)
- Album Type, All Artists

Optional columns
- Critic Score, Critic Reviews
- Avg Track Score
- Descriptors
- Label
- Album URL

Schema enforcement
- Strict raw-schema validation runs when pipeline strict mode is enabled (`panelcast run --strict`).
- Optional columns are accepted as missing and will be added as nullable fields in cleaning.

Canonical names (internal)
- Artist -> Artist
- Album -> Album
- Year -> Year
- Release Date -> Release_Date
- Genres -> Genres
- Critic Score -> Critic_Score
- User Score -> User_Score
- Avg Track Score -> Avg_Track_Score
- User Ratings -> User_Ratings
- Critic Reviews -> Critic_Reviews
- Tracks -> Num_Tracks
- Runtime (min) -> Runtime_Min
- Avg Track Runtime (min) -> Avg_Runtime
- Label -> Label
- Descriptors -> Descriptors
- Album URL -> Album_URL
- All Artists -> All_Artists
- Album Type -> Album_Type

Mapping implementation
- See `src/panelcast/data/cleaning.py` for `RAW_TO_CANONICAL`.

Source reference
- See `docs/lineage/DATA_LINEAGE_DETAILED.md` for full lineage and derived columns.
- See `docs/RAW_SCHEMA_SNAPSHOT.md` for a snapshot of the raw CSV headers and sample dtypes.

Cleaning rules (baseline)
- Min ratings thresholds produced by data stage: 5, 10, 25.
- Default pipeline split input uses `min_ratings=10` (configurable with `--min-ratings`).
- Drop rows with missing User Score.
- Drop rows with missing critical numeric fields after repair attempts.
- Record exclusion reasons per row.

Outputs (minimum)
- `data/processed/cleaned_all.parquet` (+ `.csv`)
- `data/processed/user_score_minratings_5.parquet` (+ `.csv`)
- `data/processed/user_score_minratings_10.parquet` (+ `.csv`)
- `data/processed/user_score_minratings_25.parquet` (+ `.csv`)
- `data/processed/critic_score.parquet` (+ `.csv`)
- `data/audit/summary_<run_id>.json` and `data/audit/exclusions_<run_id>.jsonl`
- `data/splits/within_entity_temporal/manifest.json`
- `data/splits/entity_disjoint/manifest.json`
- `outputs/<run_id>/dataset_hash.txt`

Target variable
- User Score (continuous), prediction for next album per artist.
