# COMPREHENSIVE DATA LINEAGE DOCUMENTATION

## Album Score Prediction Pipeline - Every Operation Documented

**Input File:** `data/raw/all_albums_full.csv`
**Output Directory:** `outputs/{run_id}/` (timestamped, e.g., `outputs/2026-01-19_143052/`)
**Primary Code:** 90+ modules under `src/panelcast/` (see [Module Index](#module-index) below)
**Framework:** NumPyro (JAX backend) hierarchical Bayesian models
**CLI Entry Point:** `panelcast run` via `src/panelcast/cli.py`
**Tracked Example:** Kendrick Lamar - "To Pimp a Butterfly" (2015)

> **Split naming:** the split strategies are now role-based —
> `within_entity_temporal` and `entity_disjoint`. Pre-rename artifacts written
> with the AOTY-flavored `within_artist_temporal` / `artist_disjoint` literals
> still load via the aliases in `panelcast.data.split_types`.

---

# PART 1: PIPELINE OVERVIEW

## 1.1 High-Level 6-Stage Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ALBUM SCORE PREDICTION PIPELINE                     │
│                     6-Stage Modular Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ INPUT: data/raw/all_albums_full.csv                                │ │
│  │ ~130,000 rows x 18 columns                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ STAGE 1: DATA                                           │           │
│  │ Module: pipelines/prepare_dataset.py                    │           │
│  │ • Load raw CSV with SHA-256 hash                        │           │
│  │ • Rename columns (RAW_TO_CANONICAL mapping)             │           │
│  │ • Parse dates, extract collaboration features           │           │
│  │ • Filter for user score + critic score subsets           │           │
│  │ • Save to data/processed/*.parquet                      │           │
│  │ Input:  data/raw/all_albums_full.csv                    │           │
│  │ Output: data/processed/cleaned_all.parquet              │           │
│  │         data/processed/user_score_minratings_{5,10,25}  │           │
│  │         data/processed/critic_score.parquet             │           │
│  └─────────────────────────────────┬───────────────────────┘           │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ STAGE 2: SPLITS                                         │           │
│  │ Module: pipelines/create_splits.py                      │           │
│  │ • Within-artist temporal split (primary evaluation)     │           │
│  │ • Artist-disjoint split (cold-start robustness)         │           │
│  │ • Validation: temporal ordering + no artist overlap     │           │
│  │ • SHA-256 manifests with per-row assignment reasoning   │           │
│  │ Input:  data/processed/user_score_minratings_10.parquet │           │
│  │ Output: data/splits/within_entity_temporal/{train,      │           │
│  │           validation,test}.parquet                      │           │
│  │         data/splits/entity_disjoint/{train,             │           │
│  │           validation,test}.parquet                      │           │
│  └─────────────────────────────────┬───────────────────────┘           │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ STAGE 3: FEATURES                                       │           │
│  │ Module: pipelines/build_features.py                     │           │
│  │ • Fit feature blocks on training data ONLY              │           │
│  │ • Transform all splits (train/val/test)                 │           │
│  │ • Feature blocks: TemporalBlock, AlbumTypeBlock,        │           │
│  │   ArtistHistoryBlock, GenreBlock, CollaborationBlock    │           │
│  │ • n_reviews preserved from User_Ratings                 │           │
│  │ Input:  data/splits/within_entity_temporal/*.parquet    │           │
│  │ Output: data/features/{train,validation,test}           │           │
│  │           _features.parquet                             │           │
│  └─────────────────────────────────┬───────────────────────┘           │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ STAGE 4: TRAIN                                          │           │
│  │ Module: pipelines/train_bayes.py                        │           │
│  │ • Prepare model data (artist indices, album sequences)  │           │
│  │ • Apply max_albums cap (most recent per artist)         │           │
│  │ • Fit user_score_model via NUTS MCMC                    │           │
│  │ • Check convergence (R-hat, ESS, divergences)           │           │
│  │ • Save NetCDF + model manifest                          │           │
│  │ Input:  data/features/train_features.parquet            │           │
│  │         data/splits/.../train.parquet                   │           │
│  │ Output: models/user_score_model/                        │           │
│  │         models/training_summary.json                    │           │
│  └─────────────────────────────────┬───────────────────────┘           │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ STAGE 5: EVALUATE                                       │           │
│  │ Module: pipelines/evaluate.py                           │           │
│  │ • Model diagnostics and calibration metrics             │           │
│  │ • Posterior predictive checks                           │           │
│  │ Input:  models/user_score_model/                        │           │
│  │         data/features/test_features.parquet             │           │
│  │ Output: outputs/evaluation/metrics.json                 │           │
│  │         outputs/evaluation/diagnostics.json             │           │
│  └─────────────────────────────────┬───────────────────────┘           │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ STAGE 6: REPORT                                         │           │
│  │ Module: pipelines/publication.py                        │           │
│  │ • Generate publication-quality figures                   │           │
│  │ • Generate LaTeX/CSV tables                             │           │
│  │ • Trace plots, forest plots, posterior summaries        │           │
│  │ Input:  outputs/evaluation/*.json                       │           │
│  │ Output: reports/figures/                                │           │
│  │         reports/tables/                                 │           │
│  └─────────────────────────────────────────────────────────┘           │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ FINAL: outputs/{run_id}/manifest.json                              │ │
│  │        outputs/latest -> symlink to most recent successful run     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.2 Stage Dependency Graph

Stages execute in topological order via Kahn's algorithm (`stages.py:_topological_sort`):

```
data ──► splits ──► features ──► train ──► evaluate ──► report
```

Each stage is defined as a `PipelineStage` dataclass with:
- `name`: unique identifier (e.g., `"data"`, `"train"`)
- `description`: human-readable purpose
- `run_fn`: callable that receives a `StageContext`
- `input_paths`: files read (used for hash-based skip detection)
- `output_paths`: files created
- `depends_on`: list of stage names that must complete first

Stages are constructed by factory functions in `pipelines/stages.py`:
- `make_stage_data()` - loads raw CSV, outputs processed parquet files
- `make_stage_splits(min_ratings)` - creates train/val/test splits
- `make_stage_features()` - builds feature matrices
- `make_stage_train()` - fits Bayesian models
- `make_stage_evaluate()` - runs evaluation diagnostics
- `make_stage_report()` - generates publication artifacts

## 1.3 Orchestrator Lifecycle

The `PipelineOrchestrator` class (`pipelines/orchestrator.py`) manages the full pipeline:

1. **Environment verification** - checks `pixi.lock` for reproducibility
2. **Run directory creation** - `outputs/{timestamp}/` with manifest
3. **Logging setup** - structured JSON logs via `structlog`
4. **Seed initialization** - `set_seeds(config.seed)` for reproducibility
5. **Stage execution** - topologically sorted, with Rich progress display
6. **Skip detection** - hash-based comparison against `outputs/latest/manifest.json`
7. **Failure handling** - moves run to `outputs/failed/{run_id}/`
8. **Success finalization** - updates manifest, creates `outputs/latest` symlink

## 1.4 Module Index

```
src/panelcast/
├── cli.py                          # Typer CLI entry point (panelcast)
├── config/
│   ├── schema.py                   # Pydantic/dataclass config schemas
│   └── loader.py                   # Config file loading
├── data/
│   ├── ingest.py                   # Raw CSV loading with SHA-256 hash
│   ├── cleaning.py                 # Column renaming, date parsing, filtering
│   ├── validation.py               # Pandera schema validation
│   ├── split.py                    # Split algorithms (temporal, disjoint)
│   ├── lineage.py                  # AuditLogger for exclusion tracking
│   └── manifests.py                # SplitManifest dataclass
├── features/
│   ├── base.py                     # BaseFeatureBlock with fit/transform
│   ├── pipeline.py                 # FeaturePipeline orchestrator
│   ├── registry.py                 # FeatureRegistry with named builders
│   ├── album_type.py               # AlbumTypeBlock
│   ├── artist.py                   # ArtistHistoryBlock, ArtistReputationBlock
│   ├── collaboration.py            # CollaborationBlock
│   ├── core.py                     # CoreNumericBlock
│   ├── genre.py                    # GenreBlock, GenrePCABlock
│   ├── temporal.py                 # TemporalBlock
│   ├── pca.py                      # Shared PCA utilities
│   └── errors.py                   # NotFittedError
├── models/
│   └── bayes/
│       ├── model.py                # NumPyro model definitions (make_score_model)
│       ├── priors.py               # PriorConfig dataclass
│       ├── fit.py                  # MCMCConfig, fit_model, FitResult
│       ├── diagnostics.py          # check_convergence
│       ├── io.py                   # save_model (NetCDF + manifest)
│       └── predict.py              # Posterior predictive sampling
├── pipelines/
│   ├── orchestrator.py             # PipelineOrchestrator, PipelineConfig
│   ├── stages.py                   # PipelineStage definitions, topological sort
│   ├── prepare_dataset.py          # Stage 1: data preparation
│   ├── create_splits.py            # Stage 2: split creation
│   ├── build_features.py           # Stage 3: feature engineering
│   ├── train_bayes.py              # Stage 4: MCMC model fitting
│   ├── evaluate.py                 # Stage 5: evaluation pipeline
│   ├── publication.py              # Stage 6: publication artifacts
│   ├── sensitivity.py              # Sensitivity analysis framework
│   ├── manifest.py                 # RunManifest, GitStateModel
│   ├── errors.py                   # PipelineError, ConvergenceError
│   └── predict_next.py             # Next-album prediction
├── evaluation/
│   ├── metrics.py                  # Regression metrics (RMSE, MAE, R2, etc.)
│   ├── calibration.py              # Calibration assessment
│   └── cv.py                       # Cross-validation utilities
├── reporting/
│   ├── figures.py                  # Publication figure generation
│   ├── tables.py                   # LaTeX/CSV table generation
│   └── model_card.py               # Model card generation
├── utils/
│   ├── hashing.py                  # sha256_file, hash_dataframe
│   ├── git_state.py                # Git SHA/dirty state capture
│   ├── environment.py              # pixi.lock verification
│   ├── logging.py                  # structlog configuration
│   └── random.py                   # set_seeds for reproducibility
├── io/
│   ├── paths.py                    # Path conventions
│   ├── readers.py                  # read_csv with encoding handling
│   └── writers.py                  # Output writing utilities
├── preflight/
│   ├── check.py                    # GPU memory estimation
│   ├── full_check.py               # Full calibration preflight
│   ├── calibrate.py                # Mini-MCMC calibration
│   ├── mini_run.py                 # Mini-run execution
│   ├── cache.py                    # Calibration cache
│   └── output.py                   # Preflight result rendering
├── gpu_memory/
│   ├── estimate.py                 # Memory estimation formulas
│   ├── measure.py                  # Runtime memory measurement
│   ├── platform.py                 # Platform detection
│   └── query.py                    # GPU memory queries
└── visualization/
    ├── charts.py                   # Plotly chart builders
    ├── dashboard.py                # Figure assembly + run-data loading
    ├── export.py                   # Static figure export
    └── theme.py                    # Visualization theming
```

---

# PART 3: STAGE-BY-STAGE DETAIL

## 3.1 Stage 1: Data Preparation

**Module:** `src/panelcast/pipelines/prepare_dataset.py`
**Entry function:** `prepare_datasets(config: PrepareConfig) -> PrepareResult`
**Triggered by:** `_run_data_stage(ctx)` in `stages.py`

### 3.1.1 Input Loading

The raw CSV is loaded via `data/ingest.py:load_raw_albums()`:

```
Input: data/raw/all_albums_full.csv
├── SHA-256 hash computed before loading (utils/hashing.py:sha256_file)
├── Encoding: utf-8-sig (BOM-aware)
├── Reader: io/readers.py:read_csv
├── original_row_id column added (preserves raw CSV row numbers)
└── Optional schema validation via data/validation.py:validate_raw_dataframe
```

**LoadMetadata** captured:
- `file_path`: absolute path to CSV
- `file_hash`: SHA-256 of raw file
- `load_timestamp`: ISO-8601 timestamp
- `row_count`: total rows loaded
- `column_count`: total columns

### 3.1.2 Cleaning Pipeline

The `data/cleaning.py:clean_albums()` function applies transformations in order:

**Step 1: Column Renaming** (`rename_columns`)

Uses the `RAW_TO_CANONICAL` mapping defined in `cleaning.py`:

| Raw Column Name | Canonical Name |
|----------------|----------------|
| `Release Date` | `Release_Date` |
| `Critic Score` | `Critic_Score` |
| `User Score` | `User_Score` |
| `Avg Track Score` | `Avg_Track_Score` |
| `User Ratings` | `User_Ratings` |
| `Critic Reviews` | `Critic_Reviews` |
| `Tracks` | `Num_Tracks` |
| `Runtime (min)` | `Runtime_Min` |
| `Avg Track Runtime (min)` | `Avg_Runtime` |
| `Album URL` | `Album_URL` |
| `All Artists` | `All_Artists` |
| `Album Type` | `Album_Type` |

Columns not in this mapping pass through unchanged: `Artist`, `Album`, `Year`, `Genres`.

**Step 2: Date Parsing** (`parse_release_dates`)

Three-tier risk classification:

| Tier | Condition | Risk Level | Imputation | Column Value |
|------|-----------|------------|------------|-------------|
| 1 | Valid `Release_Date` parsed | `low` | `none` | Parsed datetime |
| 2 | Has `Year`, no parseable `Release_Date` | `medium` | `jan1` | `{Year}-01-01` |
| 3 | Missing both `Release_Date` and `Year` | `high` | `artist_inferred` | Null |

Creates columns: `Release_Date_Parsed`, `date_risk`, `date_imputation_type`, `flag_future_year`, `flag_sparse_era`.

**Step 3: Collaboration Features** (`extract_collaboration_features`)

Parses `All_Artists` column (pipe-delimited `" | "` separator):

| Column | Logic |
|--------|-------|
| `num_artists` | Count of `" \| "` splits in `All_Artists` |
| `is_collaboration` | `num_artists > 1` |
| `collab_type` | `solo` (1), `duo` (2), `small_group` (3-4), `ensemble` (5+) |

**Step 4: Primary Genre** (`extract_primary_genre`)

`primary_genre` = first genre in comma-separated `Genres` list.

**Step 5: Unknown Artist Flag** (`flag_unknown_artist`)

`is_unknown_artist` = `True` if `Artist == "[unknown artist]"`.

**Step 6: Drop Descriptors**

`Descriptors` column dropped per research finding: 4.2% coverage with severe selection bias.

### 3.1.3 Score Model Filtering

After cleaning, separate datasets are created for each model type.

**User Score Filtering** (`filter_for_user_score_model`):

Applied at three thresholds (5, 10, 25) via `PrepareConfig.min_ratings_thresholds`:

1. `User_Score` must not be NaN
2. `User_Score` must be in range [0, 100]
3. `User_Ratings >= threshold`

Each filter step is logged via `apply_exclusion_filter()` which records:
- Rows excluded (count and percentage)
- Reason string (e.g., `"missing_user_score"`, `"below_min_ratings_10"`)
- Optional value column for audit detail

**Critic Score Filtering** (`filter_for_critic_score_model`):

1. `Critic_Score` must not be NaN
2. `Critic_Score` must be in range [0, 100]
3. `Critic_Reviews >= 1` (default `min_critic_reviews`)

### 3.1.4 Outputs

All datasets saved in both Parquet (snappy compression) and CSV formats:

| Output File | Description |
|------------|-------------|
| `data/processed/cleaned_all.parquet` | Full cleaned dataset (before score filtering) |
| `data/processed/user_score_minratings_5.parquet` | User score, min 5 ratings |
| `data/processed/user_score_minratings_10.parquet` | User score, min 10 ratings (default) |
| `data/processed/user_score_minratings_25.parquet` | User score, min 25 ratings |
| `data/processed/critic_score.parquet` | Critic score dataset |

Audit log saved to `data/audit/` via `data/lineage.py:AuditLogger`.

---

## 3.2 Stage 2: Split Creation

**Module:** `src/panelcast/pipelines/create_splits.py`
**Entry function:** `create_splits(config: SplitConfig) -> SplitResult`
**Triggered by:** `_run_splits_stage(ctx)` in `stages.py`

### 3.2.1 Configuration

`SplitConfig` dataclass:

| Field | Default | Description |
|-------|---------|-------------|
| `min_ratings` | `10` | Determines source parquet file |
| `output_dir` | `data/splits` | Output directory |
| `version` | `"v1"` | Manifest version tag |
| `random_state` | `42` | Seed (from CLI `--seed`) |
| `test_albums` | `1` | Albums per artist held for test |
| `val_albums` | `1` | Albums per artist held for validation |
| `min_train_albums` | `1` | Minimum training albums to include artist |
| `disjoint_test_size` | `0.15` | Artist-disjoint test fraction |
| `disjoint_val_size` | `0.15` | Artist-disjoint validation fraction |
| `source_path` | Computed | `data/processed/user_score_minratings_{min_ratings}.parquet` |

### 3.2.2 Within-Artist Temporal Split

**Algorithm** (`data/split.py:within_entity_temporal_split`):

For each artist with sufficient albums (>= `min_train_albums + val_albums + test_albums`):
1. Sort albums chronologically
2. Last `test_albums` → test set
3. Previous `val_albums` → validation set
4. Remaining → training set

Artists with fewer albums are excluded entirely. This prevents data leakage by ensuring the model never trains on albums released after the test/validation albums.

**Validation** (`data/split.py:validate_temporal_split`):
- Verifies temporal ordering: max train date <= min validation date <= min test date per artist

### 3.2.3 Artist-Disjoint Split

**Algorithm** (`data/split.py:entity_disjoint_split`):

Artists (not albums) are split into train/val/test groups. No artist appears in more than one split. Uses `random_state` for reproducibility.

- `test_size=0.15` → 15% of artists
- `val_size=0.15` → 15% of artists
- Remaining 70% → training set

**Validation** (`data/split.py:assert_no_artist_overlap`):
- Asserts zero intersection between artist sets across all split pairs.

### 3.2.4 Manifests and Outputs

Each split strategy produces:

| Output | Description |
|--------|-------------|
| `data/splits/{strategy}/train.parquet` | Training split |
| `data/splits/{strategy}/validation.parquet` | Validation split |
| `data/splits/{strategy}/test.parquet` | Test split |
| `data/splits/{strategy}/manifest.json` | Split manifest |

**SplitManifest** (`data/manifests.py`) contains:
- `version`: manifest version
- `created_at`: ISO-8601 timestamp
- `split_type`: `"within_entity_temporal"` or `"entity_disjoint"`
- `parameters`: split configuration parameters
- `source_dataset`: path, SHA-256 hash, row count, unique artists
- `splits`: per-split `SplitStats` (row count, unique artists, SHA-256)
- `assignments`: per-row split assignment with reasoning
- `content_hash`: combined SHA-256 of all split hashes

Pipeline summary saved to `data/splits/pipeline_summary.json`.

---

## 3.3 Stage 3: Feature Engineering

**Module:** `src/panelcast/pipelines/build_features.py`
**Entry function:** `build_features(ctx: StageContext) -> dict`
**Triggered by:** `_run_features_stage(ctx)` in `stages.py`

### 3.3.1 Fit/Transform Architecture

The feature pipeline uses a scikit-learn-inspired fit/transform pattern to prevent data leakage:

```
┌─────────────────────────────────────────────────────┐
│ FeaturePipeline (features/pipeline.py)              │
│                                                     │
│  1. pipeline.fit(train_df, ctx)                     │
│     └── Each block learns statistics from train     │
│                                                     │
│  2. pipeline.transform(train_df, ctx) → train_feat  │
│  3. pipeline.transform(val_df, ctx)   → val_feat    │
│  4. pipeline.transform(test_df, ctx)  → test_feat   │
│     └── Blocks apply learned transforms only        │
└─────────────────────────────────────────────────────┘
```

**Key invariant:** `fit()` is called exactly once on training data. `transform()` can be called on any split.

**BaseFeatureBlock** interface (`features/base.py`):
- `name`: identifier string
- `requires`: list of dependency block names
- `required_columns`: DataFrame columns needed
- `fit(df, ctx) -> self`: learn from training data
- `transform(df, ctx) -> FeatureOutput`: apply learned transform
- `fit_transform(df, ctx) -> FeatureOutput`: convenience method (training only)
- `is_fitted` property: raises `NotFittedError` if transform called before fit

### 3.3.2 Feature Block Registry

The `features/registry.py:build_default_registry()` registers these blocks:

| Registry Name | Block Class | Module |
|--------------|-------------|--------|
| `core_numeric` | `CoreNumericBlock` | `features/core.py` |
| `temporal` | `TemporalBlock` | `features/temporal.py` |
| `artist_reputation` | `ArtistReputationBlock` | `features/artist.py` |
| `artist_history` | `ArtistHistoryBlock` | `features/artist.py` |
| `genre` | `GenreBlock` | `features/genre.py` |
| `genre_pca` | `GenrePCABlock` | `features/genre.py` |
| `album_type` | `AlbumTypeBlock` | `features/album_type.py` |
| `collaboration` | `CollaborationBlock` | `features/collaboration.py` |

### 3.3.3 Active Feature Blocks

The `build_features.py:get_feature_blocks()` function selects blocks based on CLI ablation flags:

| Block | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `TemporalBlock` | `--no-temporal` | Enabled | Release year, career age, time features |
| `AlbumTypeBlock` | Always | Always | Album/EP/Mixtape/Compilation encoding |
| `ArtistHistoryBlock` | `--no-artist` | Enabled | Artist album count, score history |
| `GenreBlock` | `--no-genre` | Enabled | Genre PCA components (n=10, min_count=20) |
| `CollaborationBlock` | Always | Always | Solo/duo/group/ensemble features |

Block execution order respects `requires` dependencies.

### 3.3.4 n_reviews Preservation

The feature pipeline preserves `User_Ratings` as `n_reviews` for heteroscedastic noise modeling:

```python
# From build_features.py
train_n_reviews = train_df["User_Ratings"].rename("n_reviews")
# ... after feature transformation ...
train_features["n_reviews"] = train_n_reviews  # with index alignment validation
```

This column passes through to the training stage where it controls per-observation noise scaling.

### 3.3.5 Outputs

| Output File | Description |
|------------|-------------|
| `data/features/train_features.parquet` | Training feature matrix (with `n_reviews`) |
| `data/features/validation_features.parquet` | Validation feature matrix |
| `data/features/test_features.parquet` | Test feature matrix |
| `data/features/manifest.json` | Feature manifest (blocks, shapes, ablation flags) |

Feature manifest records:
- `seed`: random state used
- `blocks`: list of block names (e.g., `["temporal", "album_type", "artist_history", "genre", "collaboration"]`)
- `feature_ablation`: which feature groups were enabled/disabled
- `feature_names`: ordered list of output column names
- `n_reviews_included`: `true`
- Per-split: path, row count, column count, n_reviews min/max/median

---

## 3.4 Stage 4: Model Training

**Module:** `src/panelcast/pipelines/train_bayes.py`
**Entry function:** `train_models(ctx: StageContext) -> dict`
**Triggered by:** `_run_train_stage(ctx)` in `stages.py`

### 3.4.1 Data Preparation

**Loading** (`train_bayes.py:load_training_data`):

1. Load `data/features/train_features.parquet` and `data/splits/within_entity_temporal/train.parquet`
2. Validate DataFrame alignment (row count and index match)
3. Join features with original split data (left join)
4. Fill NaN feature values with 0 for numeric stability

**Model data preparation** (`train_bayes.py:prepare_model_data`):

| Array | Shape | Source |
|-------|-------|--------|
| `artist_idx` | `(n_obs,)` | Unique artist string → integer index mapping |
| `album_seq` | `(n_obs,)` | Within-artist cumulative count (1-indexed) |
| `prev_score` | `(n_obs,)` | `groupby("Artist")["User_Score"].shift(1)`, NaN filled with global mean |
| `X` | `(n_obs, n_features)` | Feature matrix (float32) |
| `y` | `(n_obs,)` | `User_Score` target (float32) |
| `n_reviews` | `(n_obs,)` | `n_reviews` column (int32), from `User_Ratings` |
| `n_artists` | scalar | Count of unique artists |

**min_albums_filter:** Artists with fewer than `min_albums_filter` (default 2) albums have `album_seq` clamped to 1 (static effect only, no random walk).

**max_albums cap** (`_apply_max_albums_cap`):
- For artists with more than `max_albums` (default 50) albums, shifts album_seq so the most recent albums get positions 1 to max_albums
- Older albums share position 1
- `max_seq = album_seq.max()` after capping

**n_reviews validation:**
- Invalid values (NaN or <= 0) are detected before int32 cast
- Rows with invalid n_reviews are dropped (with warning if < 50%)
- Raises `ValueError` if > 50% invalid

### 3.4.2 MCMC Configuration

`MCMCConfig` dataclass (`models/bayes/fit.py`):

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `num_warmup` | `--num-warmup` | 1000 | Warmup iterations per chain |
| `num_samples` | `--num-samples` | 1000 | Post-warmup samples per chain |
| `num_chains` | `--num-chains` | 4 | Parallel chains |
| `chain_method` | `--chain-method` | `"sequential"` | `sequential`/`vectorized`/`parallel` |
| `seed` | `--seed` | 0 (pipeline passes 42) | Random seed |
| `max_tree_depth` | `--max-tree-depth` | 12 | Maximum NUTS tree depth |
| `target_accept_prob` | `--target-accept` | 0.90 | NUTS adaptation target |

### 3.4.3 Model Fitting

`fit_model()` in `models/bayes/fit.py`:

1. Log GPU info via `get_gpu_info()` (nvidia-smi query or JAX device fallback)
2. Create NUTS kernel with `max_tree_depth` and `target_accept_prob`
3. Create MCMC runner with `num_warmup`, `num_samples`, `num_chains`, `chain_method`
4. Generate JAX PRNGKey from `config.seed`
5. Run MCMC with `extra_fields=("diverging", "num_steps")`
6. Count divergences from `mcmc.get_extra_fields()["diverging"]`
7. Build ArviZ `InferenceData` with:
   - `posterior`: xarray Dataset of samples (chain, draw, *var_shape)
   - `sample_stats`: diverging flags and num_steps
   - `observed_data`: target scores `y`
   - `constant_data`: `X`, `artist_idx`, `album_seq`, `prev_score`, `n_reviews`, `n_ref`
8. Optionally exclude large tensors (e.g., `user_rw_innovations`) from InferenceData to prevent OOM
9. Return `FitResult` with MCMC object, InferenceData, divergences, runtime, GPU info

### 3.4.4 Heteroscedastic Noise Configuration

Three modes based on CLI flags:

| Mode | Condition | n_exponent | n_ref |
|------|-----------|------------|-------|
| **Homoscedastic** | `n_exponent=0.0, learn_n_exponent=False` | 0.0 (fixed) | `None` |
| **Fixed heteroscedastic** | `n_exponent>0, learn_n_exponent=False` | CLI value | `median(n_reviews)` |
| **Learned heteroscedastic** | `learn_n_exponent=True` | Sampled from prior | `median(n_reviews)` |

When `n_ref` is provided, the model uses sigma-ref reparameterization (see Part 5).

### 3.4.5 Convergence Checking

`check_convergence()` in `models/bayes/diagnostics.py`:

| Diagnostic | Threshold | Source |
|-----------|-----------|--------|
| R-hat | `< rhat_threshold` (default 1.01) | ArviZ rank-normalized split-R-hat |
| ESS-bulk | `>= ess_threshold * num_chains` (default 1600) | ArviZ ESS-bulk |
| ESS-tail | Reported but not in pass/fail | ArviZ ESS-tail |
| Divergences | 0 unless `allow_divergences=True` | `sample_stats["diverging"]` |

Returns `ConvergenceDiagnostics` dataclass with `passed`, `failing_params`, and full summary DataFrame.

**Strict mode behavior:**
- If `strict=True` and divergences > 0 (without `allow_divergences`): raises `ConvergenceError`
- If `strict=True` and R-hat/ESS fail: raises `ConvergenceError`
- If `strict=False`: logs warnings, continues execution

### 3.4.6 Model Persistence

`save_model()` in `models/bayes/io.py`:

1. Generate timestamped filename: `user_score_{YYYYMMDD_HHMMSS}.nc`
2. Save InferenceData to NetCDF via `idata.to_netcdf(filepath)`
3. Create `ModelManifest` with: version, timestamp, model_type, filename, MCMC config, priors, data hash, git commit, GPU info, runtime, divergences
4. Update `models/manifest.json` (`ModelsManifest`) with current pointer and history

**Output files:**
- `models/user_score_{timestamp}.nc` - NetCDF with posterior samples
- `models/manifest.json` - Model manifest with history
- `models/training_summary.json` - Full training summary with diagnostics

---

## 3.5 Stage 5: Model Evaluation

**Module:** `src/panelcast/pipelines/evaluate.py`
**Entry function:** `evaluate_models(ctx: StageContext)`
**Triggered by:** `_run_evaluate_stage(ctx)` in `stages.py`

### 3.5.1 Evaluation Pipeline

The evaluation stage loads trained models and test data, then computes:

1. **Model diagnostics** - convergence summary from stored InferenceData
2. **Predictive performance** - posterior predictive checks on test set
3. **Calibration assessment** - via `evaluation/calibration.py`
4. **Regression metrics** - via `evaluation/metrics.py` (RMSE, MAE, R-squared, etc.)

### 3.5.2 Metrics

The `evaluation/metrics.py` module provides standard regression metrics:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (coefficient of determination)
- Mean prediction interval width
- Coverage probability (fraction of observations within prediction intervals)

### 3.5.3 Calibration

The `evaluation/calibration.py` module assesses prediction interval calibration:
- Compares nominal coverage (e.g., 90% interval) vs. observed coverage
- Generates reliability data for calibration plots

### 3.5.4 Outputs

| Output File | Description |
|------------|-------------|
| `outputs/evaluation/metrics.json` | Regression performance metrics |
| `outputs/evaluation/diagnostics.json` | Convergence diagnostic summary |

---

## 3.6 Stage 6: Publication Artifacts

**Module:** `src/panelcast/pipelines/publication.py`
**Entry function:** `generate_publication_artifacts(ctx: StageContext)`
**Triggered by:** `_run_report_stage(ctx)` in `stages.py`

### 3.6.1 Artifact Generation

The report stage produces publication-quality outputs:

1. **Figures** (`reporting/figures.py`):
   - Posterior trace plots
   - Forest plots (coefficient summaries with credible intervals)
   - Posterior predictive check plots
   - Calibration plots
   - Residual diagnostics

2. **Tables** (`reporting/tables.py`):
   - Model coefficient summary (posterior mean, std, HDI)
   - Convergence diagnostics table (R-hat, ESS per parameter)
   - Performance metrics comparison table
   - LaTeX and CSV formats

3. **Model Card** (`reporting/model_card.py`):
   - Standardized documentation following model cards best practice
   - Includes sigma-ref detection for dynamic variable reporting

### 3.6.2 Sigma-Ref Detection

The publication pipeline dynamically detects whether the model used sigma-ref reparameterization by checking for `user_sigma_ref` in the posterior trace variables. This determines which noise parameters appear in trace plots and tables.

### 3.6.3 Outputs

| Output Directory | Contents |
|-----------------|----------|
| `reports/figures/` | PNG/SVG publication figures |
| `reports/tables/` | LaTeX/CSV tables |

---

# PART 4: STATISTICAL OPERATIONS REFERENCE

## 4.1 Model Structure

The hierarchical Bayesian model predicts album scores using a combination of:

1. **Time-varying artist effects** (random walk career trajectories)
2. **AR(1) term** (album-to-album score dependency)
3. **Fixed effects** (covariates from feature engineering)
4. **Observation noise** (optionally heteroscedastic)

### Mathematical Specification

For observation `i` by artist `j` at album sequence position `t`:

```
y_ij ~ StudentT(df=4, mu_ij, sigma_ij)   # family configurable via --likelihood-family

mu_ij = artist_effect_jt + X_ij @ beta + rho * (prev_score_ij - ar_center)
```

**Time-varying artist effects (random walk):**

```
artist_effect_j1 = init_artist_effect_j                    (initial)
artist_effect_jt = artist_effect_j(t-1) + innovation_jt    (t > 1)
innovation_jt ~ Normal(0, sigma_rw)
```

Equivalently:
```
rw_innovations_j ~ Normal(0, sigma_rw) [shape: (n_artists, max_seq-1)]
rw_trajectory_j  = cumsum(rw_innovations_j, axis=time)
artist_effects   = vstack([init_effect, init_effect + rw_trajectory])
```

**Partial pooling of initial artist effects:**

```
init_artist_effect_j ~ Normal(mu_artist, sigma_artist)     (for j = 1..n_artists)
```

**Hyperpriors:**

```
mu_artist    ~ Normal(mu_artist_loc, mu_artist_scale)      = Normal(0, 1)
sigma_artist ~ HalfNormal(sigma_artist_scale)              = HalfNormal(0.5)
sigma_rw     ~ HalfNormal(sigma_rw_scale)                  = HalfNormal(0.1)
rho          ~ TruncatedNormal(rho_loc, rho_scale, -0.99, 0.99) = TruncatedNormal(0, 0.3)
beta         ~ Normal(beta_loc, beta_scale)^n_features     = Normal(0, 1)^n
```

**Observation noise:**

```
# Homoscedastic mode (n_exponent = 0):
sigma_obs    ~ HalfNormal(sigma_obs_scale)                 = HalfNormal(1.0)
sigma_ij     = sigma_obs                                   (constant for all obs)

# Heteroscedastic mode (n_exponent > 0 or learned):
sigma_ij     = sigma_obs / n_reviews_i^n_exponent          (per-observation)

# Sigma-ref reparameterization (when n_ref provided):
sigma_ref    ~ HalfNormal(sigma_ref_scale)                 = HalfNormal(1.0)
sigma_obs    = sigma_ref * n_ref^n_exponent                (deterministic)
sigma_ij     = sigma_obs / n_reviews_i^n_exponent
             = sigma_ref * n_ref^n_exponent / n_reviews_i^n_exponent
```

## 4.2 Heteroscedastic Noise Scaling

The `compute_sigma_scaled()` function in `models/bayes/model.py` implements:

```
sigma_scaled_i = sigma_obs / n_reviews_i^exponent
```

Using log-space arithmetic to avoid overflow:

```
log(sigma_scaled) = log(sigma_obs) - exponent * log(n_reviews)
sigma_scaled      = exp(log(sigma_scaled))
```

**Special cases:**
- `n_reviews = 1`: multiplied by `single_review_multiplier` (default 2.0) when `exponent > 0`
- `exponent = 0`: returns `sigma_obs` unchanged (homoscedastic)
- All values floored at `min_sigma = 0.01` for numerical stability

## 4.3 Non-Centered Parameterization

The `init_artist_effect` parameter uses `LocScaleReparam(centered=0)` from NumPyro:

```python
# In model.py:make_score_model()
reparam_config = {
    f"{prefix}init_artist_effect": LocScaleReparam(centered=0),
}
reparameterized_model = reparam(_score_model_centered, config=reparam_config)
```

This transforms the centered parameterization:
```
init_artist_effect_j ~ Normal(mu_artist, sigma_artist)
```

Into a non-centered form:
```
z_j             ~ Normal(0, 1)
init_artist_effect_j = mu_artist + sigma_artist * z_j
```

This avoids the "funnel geometry" that causes NUTS sampling difficulties when `sigma_artist` is small (strong pooling regime).

## 4.4 Learned n_exponent Prior Options

When `learn_n_exponent=True`, the exponent is sampled from a prior:

**Logit-Normal (default, recommended):**
```
n_exponent ~ TransformedDistribution(
    Normal(n_exponent_loc, n_exponent_scale),  # = Normal(0, 1)
    [SigmoidTransform()]
)
```
Maps unbounded Normal through sigmoid to [0, 1]. Avoids funnel geometry.

**Beta (legacy):**
```
n_exponent ~ Beta(n_exponent_alpha, n_exponent_beta)       # = Beta(2, 4)
```
Beta(2, 4) has mode at 0.25, mean at 0.33. May cause divergences due to challenging posterior geometry.

## 4.5 Sigma-Ref Reparameterization

When heteroscedastic mode is active and `n_ref` is provided:

```
Instead of: sigma_obs ~ HalfNormal(...)                    (free parameter)
            sigma_scaled = sigma_obs / n^exponent           (derived)

Use:        sigma_ref ~ HalfNormal(sigma_ref_scale)        (free parameter)
            sigma_obs = sigma_ref * n_ref^exponent          (deterministic)
            sigma_scaled = sigma_obs / n^exponent           (derived)
```

Where `n_ref = median(n_reviews)` from training data.

**Why:** When both `sigma_obs` and `n_exponent` are sampled, they form a multiplicative funnel. Sampling `sigma_ref` at the reference point breaks this funnel because `sigma_ref` represents noise at the "typical" review count, which is more identifiable.

**Implementation in model.py:**
```python
if use_sigma_ref:
    sigma_ref = numpyro.sample(f"{prefix}sigma_ref", dist.HalfNormal(priors.sigma_ref_scale))
    sigma_obs = numpyro.deterministic(
        f"{prefix}sigma_obs", sigma_ref * jnp.power(n_ref, n_exp)
    )
else:
    sigma_obs = numpyro.sample(f"{prefix}sigma_obs", dist.HalfNormal(priors.sigma_obs_scale))
```

---

# PART 5: BAYESIAN MODEL IMPLEMENTATION DETAIL

## 5.1 Model Variants

Two model variants are created by the factory function `make_score_model()`:

| Variable | Factory Call | Prefix | Usage |
|----------|-------------|--------|-------|
| `user_score_model` | `make_score_model("user")` | `user_` | Primary model for user score prediction |
| `critic_score_model` | `make_score_model("critic")` | `critic_` | Critic score prediction |
| `album_score_model` | Alias for `user_score_model` | `user_` | Backward compatibility |

All sample site names use the prefix (e.g., `user_beta`, `user_rho`, `critic_sigma_obs`).

## 5.2 Sample Site Inventory

Complete list of NumPyro sample sites for the `user_` prefix model:

| Site Name | Distribution | Shape | Description |
|-----------|-------------|-------|-------------|
| `user_mu_artist` | `Normal(0, 1)` | scalar | Population mean of artist effects |
| `user_sigma_artist` | `HalfNormal(0.5)` | scalar | Between-artist std (pooling strength) |
| `user_sigma_rw` | `HalfNormal(0.1)` | scalar | Random walk innovation scale |
| `user_rho` | `TruncatedNormal(0, 0.3, -0.99, 0.99)` | scalar | AR(1) coefficient |
| `user_init_artist_effect` | `Normal(mu_artist, sigma_artist)` | `(n_artists,)` | Initial artist effects (non-centered) |
| `user_rw_innovations` | `Normal(0, sigma_rw)` | `(n_artists, max_seq-1)` | Random walk innovations |
| `user_beta` | `Normal(0, 1)` | `(n_features,)` | Fixed effect coefficients |
| `user_sigma_obs` | `HalfNormal(1.0)` | scalar | Base observation noise (sampled or deterministic) |
| `user_sigma_ref` | `HalfNormal(1.0)` | scalar | Noise at reference review count (sigma-ref mode only) |
| `user_n_exponent` | `LogitNormal(0, 1)` or `Beta(2, 4)` | scalar | Heteroscedastic exponent (learned mode only) |
| `user_y` | `Normal(mu, sigma_scaled)` | `(n_obs,)` | Observed/predicted scores |

**Notes on `user_rw_innovations`:**
- Shape `(n_artists, max_seq-1)` can be very large
- Excluded from InferenceData by default (`exclude_from_idata=("user_rw_innovations",)`) to prevent OOM
- Only used during model execution; trajectory is computed via `cumsum`

## 5.3 Prior Configuration

`PriorConfig` dataclass (`models/bayes/priors.py`), all fields frozen:

| Field | Default | Role |
|-------|---------|------|
| `mu_artist_loc` | 0.0 | Center of artist effect population mean |
| `mu_artist_scale` | 1.0 | Uncertainty in population center |
| `sigma_artist_scale` | 0.5 | Scale for between-artist variance prior |
| `sigma_rw_scale` | 0.1 | Scale for random walk innovation prior |
| `rho_loc` | 0.0 | Center of AR(1) coefficient prior |
| `rho_scale` | 0.3 | Uncertainty in AR(1) coefficient |
| `beta_loc` | 0.0 | Center of fixed effect priors |
| `beta_scale` | 1.0 | Scale of fixed effect priors |
| `sigma_obs_scale` | 1.0 | Scale for observation noise prior |
| `sigma_ref_scale` | 1.0 | Scale for sigma-ref noise prior |
| `n_exponent_alpha` | 2.0 | Beta prior alpha (legacy) |
| `n_exponent_beta` | 4.0 | Beta prior beta (legacy) |
| `n_exponent_loc` | 0.0 | Logit-normal location (maps to ~0.5 via sigmoid) |
| `n_exponent_scale` | 1.0 | Logit-normal scale |

## 5.4 MCMC Fitting Infrastructure

`fit_model()` workflow (`models/bayes/fit.py`):

```
1. get_gpu_info() → log GPU/CPU status
2. NUTS(model, max_tree_depth, target_accept_prob)
3. MCMC(kernel, num_warmup, num_samples, num_chains, chain_method)
4. mcmc.run(rng_key, extra_fields=("diverging", "num_steps"), **model_args)
5. Count divergences: mcmc.get_extra_fields()["diverging"].sum()
6. Get samples: mcmc.get_samples(group_by_chain=True) → (chains, draws, *shape)
7. Filter excluded sites from samples dict
8. Build xarray posterior Dataset
9. Build xarray sample_stats Dataset (diverging, num_steps)
10. Create az.InferenceData with posterior + sample_stats
11. Add observed_data group (y) and constant_data group (X, artist_idx, etc.)
12. Return FitResult(mcmc, idata, divergences, runtime, gpu_info)
```

**InferenceData groups:**
- `posterior`: all parameter samples (chain × draw × shape)
- `sample_stats`: diverging flags, num_steps per draw
- `observed_data`: target scores `y`
- `constant_data`: `X`, `artist_idx`, `album_seq`, `prev_score`, `n_reviews`, `n_ref`, `n_ref_method`

## 5.5 Model Save Format

Models are persisted as NetCDF files via `models/bayes/io.py:save_model()`:

```
models/
├── user_score_{YYYYMMDD_HHMMSS}.nc    # ArviZ InferenceData (NetCDF)
├── manifest.json                       # ModelsManifest (current + history)
└── training_summary.json               # Full training summary
```

**ModelManifest** fields:
- `version`: `"1.0"`
- `created_at`: ISO-8601 UTC timestamp
- `model_type`: `"user_score"` or `"critic_score"`
- `filename`: NetCDF filename
- `mcmc_config`: MCMC parameters as dict
- `priors`: PriorConfig as dict
- `data_hash`: SHA-256 of training DataFrame
- `git_commit`: 40-char git SHA
- `gpu_info`: GPU description string
- `runtime_seconds`: wall-clock fitting time
- `divergences`: total divergent transitions

**ModelsManifest** tracks:
- `current`: mapping of model_type → current filename
- `history`: list of all ModelManifests (most recent first)

---

# PART 6: COLUMN PROVENANCE REFERENCE

## 6.1 Raw CSV Columns

The input file `data/raw/all_albums_full.csv` contains these columns:

| Raw Column | Type | Description |
|-----------|------|-------------|
| `Artist` | string | Primary artist name |
| `Album` | string | Album title |
| `Year` | int/float | Release year |
| `Release Date` | string | Full date (e.g., "April 10, 2018") |
| `Genres` | string | Comma-separated genre list |
| `Descriptors` | string | Comma-separated descriptors (dropped in cleaning) |
| `Critic Score` | float | Critic aggregate score (0-100) |
| `User Score` | float | User aggregate score (0-100) |
| `Avg Track Score` | float | Average per-track score |
| `User Ratings` | int | Number of user ratings |
| `Critic Reviews` | int | Number of critic reviews |
| `Tracks` | int | Number of tracks |
| `Runtime (min)` | float | Total album runtime in minutes |
| `Avg Track Runtime (min)` | float | Average track runtime |
| `Album URL` | string | AOTY album page URL |
| `All Artists` | string | All artists (pipe-delimited) |
| `Album Type` | string | Album/EP/Mixtape/Compilation |

## 6.2 Column Renaming

After `cleaning.py:rename_columns()`, the `RAW_TO_CANONICAL` mapping transforms:

```
Release Date         → Release_Date
Critic Score         → Critic_Score
User Score           → User_Score
Avg Track Score      → Avg_Track_Score
User Ratings         → User_Ratings
Critic Reviews       → Critic_Reviews
Tracks               → Num_Tracks
Runtime (min)        → Runtime_Min
Avg Track Runtime    → Avg_Runtime
Album URL            → Album_URL
All Artists          → All_Artists
Album Type           → Album_Type
```

Unchanged: `Artist`, `Album`, `Year`, `Genres`.

## 6.3 Columns Added During Cleaning

| Column | Source | Stage |
|--------|--------|-------|
| `original_row_id` | `df.index` | `ingest.py:load_raw_albums` |
| `Release_Date_Parsed` | Parsed from `Release_Date` | `cleaning.py:parse_release_dates` |
| `date_risk` | `low`/`medium`/`high` | `cleaning.py:parse_release_dates` |
| `date_imputation_type` | `none`/`jan1`/`artist_inferred` | `cleaning.py:parse_release_dates` |
| `flag_future_year` | `Year > 2025` | `cleaning.py:parse_release_dates` |
| `flag_sparse_era` | `Year < 1950` | `cleaning.py:parse_release_dates` |
| `num_artists` | Count from `All_Artists` pipe split | `cleaning.py:extract_collaboration_features` |
| `is_collaboration` | `num_artists > 1` | `cleaning.py:extract_collaboration_features` |
| `collab_type` | `solo`/`duo`/`small_group`/`ensemble` | `cleaning.py:extract_collaboration_features` |
| `primary_genre` | First genre in `Genres` | `cleaning.py:extract_primary_genre` |
| `is_unknown_artist` | `Artist == "[unknown artist]"` | `cleaning.py:flag_unknown_artist` |

## 6.4 Columns Dropped

| Column | Reason | Stage |
|--------|--------|-------|
| `Descriptors` | 4.2% coverage, severe selection bias | `cleaning.py:clean_albums` |

## 6.5 Feature Pipeline Output Columns

The feature pipeline adds engineered columns. The exact set depends on active feature blocks:

| Block | Output Columns | Description |
|-------|---------------|-------------|
| `TemporalBlock` | Release year, career age, etc. | Time-based features |
| `AlbumTypeBlock` | Album type encoding | Album/EP/Mixtape/Compilation |
| `ArtistHistoryBlock` | Artist album count, score stats | Artist track record |
| `GenreBlock` | PCA components (10 dims) | Genre space reduction |
| `CollaborationBlock` | Collaboration encoding | Solo/duo/group features |
| (preserved) | `n_reviews` | From `User_Ratings`, for heteroscedastic noise |

## 6.6 Model Input Arrays

Final arrays passed to `user_score_model`:

| Array | Source Column(s) | Transform |
|-------|-----------------|-----------|
| `artist_idx` | `Artist` | Unique string → integer index |
| `album_seq` | Within-artist index | `groupby("Artist").cumcount() + 1`, capped at `max_albums` |
| `prev_score` | `User_Score` | `groupby("Artist").shift(1)`, NaN → global mean |
| `X` | Feature columns | float32 feature matrix |
| `y` | `User_Score` | float32 target |
| `n_reviews` | `n_reviews` (from `User_Ratings`) | int32, after NaN filtering |
| `n_artists` | `Artist` | Count of unique artists |
| `max_seq` | `album_seq` | `album_seq.max()` after capping |
| `n_ref` | `n_reviews` | `median(n_reviews)` (float, for sigma-ref mode) |

---

# PART 7: CONFIGURATION REFERENCE

## 7.1 CLI Entry Point

The CLI is built with Typer (`src/panelcast/cli.py`):

```
panelcast                    # Show help
panelcast run [OPTIONS]      # Full pipeline execution
panelcast stage <name>       # Individual stage execution
panelcast export-figures     # Static figure export
```

## 7.2 PipelineConfig

`PipelineConfig` dataclass in `pipelines/orchestrator.py`:

### Execution Control

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `seed` | `--seed` | `42` | Random seed for reproducibility |
| `skip_existing` | `--skip-existing` | `False` | Skip stages with unchanged inputs |
| `stages` | `--stages` | `None` (all) | Comma-separated stage names |
| `dry_run` | `--dry-run` | `False` | Show plan without executing |
| `strict` | `--strict` | `False` | Fail on missing pixi.lock and convergence errors |
| `verbose` | `--verbose` / `-v` | `False` | Enable DEBUG logging |
| `resume` | `--resume` | `None` | Resume failed run by ID |

### MCMC Configuration

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `num_chains` | `--num-chains` | `4` | Parallel MCMC chains |
| `num_samples` | `--num-samples` | `1000` | Post-warmup samples per chain |
| `num_warmup` | `--num-warmup` | `1000` | Warmup iterations per chain |
| `target_accept` | `--target-accept` | `0.90` | NUTS adaptation target |
| `max_tree_depth` | `--max-tree-depth` | `12` | Maximum NUTS tree depth (5-15) |
| `chain_method` | `--chain-method` | `"sequential"` | Chain parallelization method |

### Convergence Thresholds

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `rhat_threshold` | `--rhat-threshold` | `1.01` | Maximum acceptable R-hat |
| `ess_threshold` | `--ess-threshold` | `400` | Minimum ESS per chain |
| `allow_divergences` | `--allow-divergences` | `False` | Don't fail on divergences |

### Data Filtering

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `max_albums` | `--max-albums` | `50` | Max albums per artist |
| `min_ratings` | `--min-ratings` | `10` | Minimum user ratings per album |
| `min_albums_filter` | `--min-albums` | `2` | Min albums for dynamic effects |

### Feature Ablation

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `enable_genre` | `/--no-genre` | `True` | Enable genre features |
| `enable_artist` | `/--no-artist` | `True` | Enable artist features |
| `enable_temporal` | `/--no-temporal` | `True` | Enable temporal features |

### Heteroscedastic Noise

| Field | CLI Flag | Default | Description |
|-------|----------|---------|-------------|
| `n_exponent` | `--n-exponent` | `0.0` | Fixed noise scaling exponent |
| `learn_n_exponent` | `--learn-n-exponent` | `False` | Sample exponent from prior |
| `n_exponent_alpha` | `--n-exponent-alpha` | `2.0` | Beta prior alpha (legacy) |
| `n_exponent_beta` | `--n-exponent-beta` | `4.0` | Beta prior beta (legacy) |
| `n_exponent_prior` | `--n-exponent-prior` | `"logit-normal"` | Prior type for learned exponent |

### GPU Preflight

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--preflight` | `False` | Quick GPU memory check (~1s) |
| `--preflight-only` | `False` | Check memory and exit (exit codes: 0/1/2) |
| `--preflight-full` | `False` | Full calibration mini-MCMC check |
| `--force-run` | `False` | Override preflight failure |
| `--recalibrate` | `False` | Force fresh calibration |

## 7.3 StageContext

`StageContext` dataclass (`pipelines/stages.py`) mirrors `PipelineConfig` fields and is passed to every stage's `run_fn`. Created by `PipelineOrchestrator._create_stage_context()`.

## 7.4 SplitConfig

`SplitConfig` dataclass (`pipelines/create_splits.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `min_ratings` | `10` | Determines source parquet file |
| `output_dir` | `data/splits` | Output directory |
| `version` | `"v1"` | Manifest version |
| `random_state` | `42` | Seed from CLI |
| `test_albums` | `1` | Albums per artist for test |
| `val_albums` | `1` | Albums per artist for validation |
| `min_train_albums` | `1` | Minimum training albums to include artist |
| `disjoint_test_size` | `0.15` | Artist-disjoint test fraction |
| `disjoint_val_size` | `0.15` | Artist-disjoint validation fraction |

## 7.5 CleaningConfig

`CleaningConfig` dataclass (`data/cleaning.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `min_year` | `1950` | Minimum year (sparse era flag) |
| `max_year` | `2025` | Maximum year (future year flag) |
| `score_min` | `0.0` | Minimum valid score |
| `score_max` | `100.0` | Maximum valid score |
| `drop_descriptors` | `True` | Drop Descriptors column |

## 7.6 PrepareConfig

`PrepareConfig` dataclass (`pipelines/prepare_dataset.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `raw_path` | `"data/raw/all_albums_full.csv"` | Raw data path |
| `output_dir` | `"data/processed"` | Output directory |
| `audit_dir` | `"data/audit"` | Audit log directory |
| `min_ratings_thresholds` | `[5, 10, 25]` | User rating thresholds |
| `min_critic_reviews` | `1` | Minimum critic reviews |
| `cleaning` | `CleaningConfig()` | Cleaning configuration |

## 7.7 MCMCConfig

`MCMCConfig` dataclass (`models/bayes/fit.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `num_warmup` | `1000` | Warmup iterations per chain |
| `num_samples` | `1000` | Post-warmup samples per chain |
| `num_chains` | `4` | Number of chains |
| `chain_method` | `"sequential"` | Chain parallelization |
| `seed` | `0` | Random seed |
| `max_tree_depth` | `12` | NUTS tree depth |
| `target_accept_prob` | `0.90` | NUTS adaptation target |

## 7.8 PriorConfig

`PriorConfig` frozen dataclass (`models/bayes/priors.py`):

See [Section 5.3](#53-prior-configuration) for full field listing with defaults and roles.

---

# PART 8: DATA FLOW VISUALIZATION

## 8.1 Feature Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ FEATURE PIPELINE (features/pipeline.py:FeaturePipeline)         │
│                                                                 │
│ Input: data/splits/within_entity_temporal/{split}.parquet       │
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │ TemporalBlock │  │ AlbumTypeBlock│  │ ArtistHistory │       │
│  │ (conditional) │  │ (always)      │  │ (conditional) │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
│          │                  │                  │               │
│  ┌───────┴──────────────────┴──────────────────┴───────┐       │
│  │                    Concatenate                       │       │
│  └──────────────────────┬──────────────────────────────┘       │
│                          │                                      │
│  ┌───────────────┐      │      ┌───────────────┐               │
│  │  GenreBlock   │──────┼──────│ Collaboration │               │
│  │ (conditional) │      │      │ Block (always)│               │
│  └───────┬───────┘      │      └───────┬───────┘               │
│          │              │              │                       │
│  ┌───────┴──────────────┴──────────────┴───────┐               │
│  │           pd.concat(axis=1)                 │               │
│  └──────────────────────┬──────────────────────┘               │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────┐                  │
│  │ + n_reviews (User_Ratings, preserved)   │                  │
│  └──────────────────────┬──────────────────┘                  │
│                          │                                      │
│ Output: data/features/{split}_features.parquet                 │
└─────────────────────────────────────────────────────────────────┘
```

## 8.2 Split Strategy Diagram

```
data/processed/user_score_minratings_10.parquet
                    │
    ┌───────────────┴───────────────┐
    │                               │
    ▼                               ▼
┌──────────────────┐    ┌──────────────────┐
│ Within-Artist    │    │ Artist-Disjoint  │
│ Temporal Split   │    │ Split            │
│                  │    │                  │
│ Per artist:      │    │ Artists randomly │
│ chronological    │    │ assigned to      │
│ train/val/test   │    │ train/val/test   │
│ split            │    │ (no overlap)     │
└──────┬───────────┘    └──────┬───────────┘
       │                       │
       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ within_entity_   │    │ entity_disjoint/ │
│ temporal/        │    │ ├── train.pq     │
│ ├── train.pq     │    │ ├── val.pq       │
│ ├── val.pq       │    │ ├── test.pq      │
│ ├── test.pq      │    │ └── manifest.json│
│ └── manifest.json│    └──────────────────┘
└──────────────────┘
```

## 8.3 Output Directory Structure

```
outputs/
├── {run_id}/                       # Timestamped run directory
│   ├── manifest.json               # RunManifest (Pydantic model)
│   ├── pipeline.log.json           # Structured JSON logs
│   └── ... (stage outputs)
├── latest -> {most_recent_run_id}  # Symlink to latest successful run
└── failed/
    └── {failed_run_id}/            # Failed runs moved here
        └── manifest.json           # With error field set

models/
├── user_score_{timestamp}.nc       # ArviZ InferenceData (NetCDF)
├── manifest.json                   # ModelsManifest (current + history)
└── training_summary.json           # Full training summary

data/
├── raw/
│   └── all_albums_full.csv         # Input (never modified)
├── processed/
│   ├── cleaned_all.parquet
│   ├── user_score_minratings_{5,10,25}.parquet
│   ├── critic_score.parquet
│   └── *.csv (CSV duplicates)
├── splits/
│   ├── within_entity_temporal/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── manifest.json
│   ├── entity_disjoint/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── manifest.json
│   └── pipeline_summary.json
├── features/
│   ├── train_features.parquet
│   ├── validation_features.parquet
│   ├── test_features.parquet
│   └── manifest.json
└── audit/
    └── (exclusion audit logs)

reports/
├── figures/                        # Publication figures
└── tables/                         # LaTeX/CSV tables
```

---

# PART 9: REPRODUCIBILITY AND MANIFESTS

## 9.1 Run Manifest

Every pipeline run creates a `RunManifest` (Pydantic model, `pipelines/manifest.py`) that captures:

### RunManifest Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Timestamp ID: `"YYYY-MM-DD_HHMMSS"` |
| `created_at` | `str` | ISO-8601 timestamp |
| `command` | `str` | Reconstructed CLI invocation (non-default flags only) |
| `flags` | `dict` | All parsed flag values (seed, num_chains, etc.) |
| `seed` | `int` | Random seed used |
| `git` | `GitStateModel` | Commit hash, branch, dirty flag, untracked count |
| `environment` | `EnvironmentInfo` | Python/JAX/NumPyro versions, pixi.lock hash |
| `input_hashes` | `dict[str, str]` | Input file path → SHA-256 hash |
| `stage_hashes` | `dict[str, str]` | Stage name → combined input hash at execution |
| `stages_completed` | `list[str]` | Successfully completed stage names |
| `stages_skipped` | `list[str]` | Skipped stage names (unchanged inputs) |
| `outputs` | `dict[str, str]` | Artifact name → output path |
| `success` | `bool` | Whether run completed successfully |
| `error` | `str \| None` | Error message if failed |
| `duration_seconds` | `float` | Total wall-clock duration |

### Environment Tracking

`EnvironmentInfo` (Pydantic model) captures:
- `python_version`: Python version string
- `jax_version`: JAX version
- `numpyro_version`: NumPyro version
- `arviz_version`: ArviZ version
- `platform`: OS description
- `pixi_lock_hash`: SHA-256 of `pixi.lock` file (exact environment pin)

### Git State Tracking

`GitStateModel` (Pydantic model) captures:
- `commit`: 40-character git SHA
- `branch`: current branch name
- `dirty`: whether working tree has modifications
- `untracked_count`: number of untracked files

## 9.2 Hash-Based Skip Detection

The orchestrator implements incremental runs via hash comparison:

```
For each stage in topological order:
    1. Check if stage was already completed (resume case)
    2. If skip_existing=True:
       a. Load outputs/latest/manifest.json (previous manifest)
       b. Compute current input hash: SHA-256 of all input_paths
       c. Compare with previous manifest's stage_hashes[stage.name]
       d. If hashes match AND all output_paths exist → skip stage
    3. Execute stage and record hash in current manifest
```

**Hash computation** (`stages.py:PipelineStage.compute_input_hash`):
1. For each `input_path` (sorted), compute `sha256_file(path)`
2. Sort all hashes
3. Combine: `SHA256(sorted_hashes_joined)`
4. Return combined hash (or `""` if no input files exist)

**SHA-256 file hashing** (`utils/hashing.py:sha256_file`):
- Reads file in 8192-byte blocks
- Returns hex digest string

**DataFrame hashing** (`utils/hashing.py:hash_dataframe`):
- Deterministic hashing of pandas DataFrame contents
- Used for training data hash in model manifest

## 9.3 Resume Logic

The `--resume {run_id}` flag allows resuming a failed run:

1. Locate run directory: `outputs/{run_id}/` or `outputs/failed/{run_id}/`
2. If in `failed/`, move back to `outputs/`
3. Load existing `manifest.json`
4. Restore MCMC config from manifest flags (prevents config drift):
   - Keys restored: `target_accept`, `max_tree_depth`, `num_chains`, `num_samples`, `num_warmup`
   - Missing keys trigger warning with current default used
5. Re-validate config after restoration
6. Skip already-completed stages (from `manifest.stages_completed`)
7. Continue from first incomplete stage

## 9.4 Environment Verification

Before running, the orchestrator verifies the environment:

1. Check for `pixi.lock` file
2. In `strict` mode: fail if `pixi.lock` missing
3. In non-strict mode: warn but continue
4. Log pixi.lock hash for manifest

This ensures the exact package versions can be reproduced via `pixi install`.

## 9.5 Random Seed Management

`utils/random.py:set_seeds(seed)` sets random seeds for:
- Python's `random` module
- NumPy's random generator
- JAX's PRNGKey (passed to MCMC via `MCMCConfig.seed`)

The seed propagates through the pipeline:
```
CLI --seed 42
  → PipelineConfig.seed
    → PipelineOrchestrator.run() → set_seeds(42)
    → StageContext.seed
      → SplitConfig.random_state (for split reproducibility)
      → FeatureContext.random_state (for feature pipeline)
      → MCMCConfig.seed (for MCMC sampling)
```

## 9.6 Split Manifests

Each split strategy saves its own manifest (`data/manifests.py:SplitManifest`):

| Field | Description |
|-------|-------------|
| `version` | Manifest version tag |
| `created_at` | ISO-8601 UTC timestamp |
| `split_type` | `"within_entity_temporal"` or `"entity_disjoint"` |
| `parameters` | Split-specific parameters (test_albums, random_state, etc.) |
| `source_dataset` | Path, SHA-256, row count, unique artists |
| `splits` | Per-split `SplitStats`: row count, unique artists, SHA-256 |
| `assignments` | Per-row split assignment with reasoning |
| `content_hash` | Combined SHA-256 of all split hashes |

---

# PART 11: SENSITIVITY TESTING

## 11.1 Framework Overview

**Module:** `src/panelcast/pipelines/sensitivity.py`

The sensitivity analysis framework (`SensitivityResult` container) provides three structured analyses:

| Analysis | Code ID | Purpose | Variants |
|----------|---------|---------|----------|
| Prior sensitivity | SENS-01 | Test robustness to prior choices | default, diffuse, informative |
| Threshold sensitivity | SENS-02 | Test robustness to data quality filters | min_ratings 5, 10, 25 |
| Feature ablation | SENS-03 | Measure feature group importance | full, no_{group} |

Each analysis returns a `dict[str, SensitivityResult]` mapping variant names to results.

## 11.2 SensitivityResult Container

```python
@dataclass
class SensitivityResult:
    name: str                                      # Variant name
    config: dict                                   # Configuration used
    idata: az.InferenceData | None                 # Fitted model
    convergence: ConvergenceDiagnostics | None      # R-hat, ESS, divergences
    loo: LOOResult | None                          # LOO-CV (ELPD, Pareto-k)
    crps: CRPSResult | None                        # CRPS (probabilistic quality)
    coefficients: pd.DataFrame                     # Posterior summary
```

## 11.3 Prior Sensitivity (SENS-01)

`run_prior_sensitivity()` fits the model with three prior configurations:

| Config | mu_artist_scale | sigma_artist_scale | sigma_rw_scale | rho_scale | beta_scale | sigma_obs_scale |
|--------|----------------|-------------------|---------------|-----------|-----------|----------------|
| **default** | 1.0 | 0.5 | 0.1 | 0.3 | 1.0 | 1.0 |
| **diffuse** | 5.0 | 2.0 | 0.5 | 0.5 | 5.0 | 2.0 |
| **informative** | 0.5 | 0.25 | 0.05 | 0.2 | 0.5 | 0.5 |

The `PRIOR_CONFIGS` dict in `sensitivity.py` defines these configurations.

For publication, results should demonstrate that conclusions (coefficient signs, relative magnitudes, predictive performance) are robust across these prior specifications.

## 11.4 Threshold Sensitivity (SENS-02)

`run_threshold_sensitivity()` tests the model with different minimum ratings thresholds:

- **threshold=5**: Most albums, potentially noisier scores
- **threshold=10**: Default, balanced quality/quantity
- **threshold=25**: Fewer albums, most reliable scores

A `data_loader` callable is provided to load the appropriate dataset for each threshold.

## 11.5 Feature Ablation (SENS-03)

`run_feature_ablation()` measures feature group importance by zeroing out feature columns:

1. Fit **full** model (baseline) with all features
2. For each feature group, zero out its columns in `X` and refit
3. Compare ELPD/performance metrics

Feature groups are specified as `dict[str, list[int]]` mapping group names to column indices.

## 11.6 Aggregation and Comparison

**`aggregate_sensitivity_results()`** creates comparison tables:
- By ELPD: `metric="elpd"` → elpd, elpd_se, p_loo, n_high_pareto_k
- By CRPS: `metric="crps"` → mean_crps, n_obs
- By convergence: `metric="convergence"` → rhat_max, ess_bulk_min, divergences
- By coefficients: `metric="coefficients"` → per-parameter mean and sd

**`create_coefficient_comparison_df()`** creates forest plot data:
- Columns: variant, param, mean, lower (HDI), upper (HDI)
- Suitable for matplotlib/plotly forest plot visualization

## 11.7 LOO Cross-Validation

LOO-CV is computed via `evaluation/cv.py`:
- `compute_log_likelihood()`: compute pointwise log-likelihood
- `add_log_likelihood_to_idata()`: add to InferenceData
- `compute_loo()`: ArviZ LOO computation with Pareto-k diagnostics

The `LOOResult` contains:
- `elpd_loo`: Expected log pointwise predictive density
- `se_elpd`: Standard error of ELPD
- `p_loo`: Effective number of parameters
- `n_high_pareto_k`: Count of problematic Pareto-k values (k > 0.7)

---

# PART 2: KENDRICK LAMAR TRACKED EXAMPLE

## 2.1 Why This Album

**Kendrick Lamar - "To Pimp a Butterfly" (2015)** serves as our running example because:
- Widely recognized album with many ratings (high confidence)
- Clear genre classification (Hip Hop)
- Artist has sufficient discography for temporal modeling
- Released in a well-represented year

## 2.2 Stage 1: Data Preparation

**Raw CSV entry** (illustrative values):

| Column | Value |
|--------|-------|
| `Artist` | `Kendrick Lamar` |
| `Album` | `To Pimp a Butterfly` |
| `Year` | `2015` |
| `Release Date` | `March 15, 2015` |
| `Genres` | `Hip Hop, Conscious Hip Hop, Jazz Rap, West Coast Hip Hop, Funk` |
| `User Score` | `92` |
| `User Ratings` | `7834` |
| `Critic Score` | `96` |
| `Critic Reviews` | `46` |
| `Tracks` | `16` |
| `Runtime (min)` | `79` |
| `Album Type` | `Album` |
| `All Artists` | `Kendrick Lamar` |

**After column renaming:**
- `Release Date` → `Release_Date`
- `User Score` → `User_Score` (92)
- `User Ratings` → `User_Ratings` (7834)
- `Critic Score` → `Critic_Score` (96)

**After date parsing:**
- `Release_Date_Parsed` = `2015-03-15`
- `date_risk` = `low` (valid full date)
- `date_imputation_type` = `none`

**After collaboration extraction:**
- `num_artists` = `1` (solo artist)
- `is_collaboration` = `False`
- `collab_type` = `solo`

**After genre extraction:**
- `primary_genre` = `Hip Hop`

**Filtering:** Passes all filters (User_Score=92, User_Ratings=7834 >> min 10).

## 2.3 Stage 2: Split Assignment

With the within-artist temporal split, Kendrick Lamar's albums are sorted chronologically:

| Album | Year | Split Assignment |
|-------|------|-----------------|
| Section.80 | 2011 | train |
| good kid, m.A.A.d city | 2012 | train |
| To Pimp a Butterfly | 2015 | train (or val/test depending on discography size) |
| DAMN. | 2017 | validation (second-to-last) |
| Mr. Morale & The Big Steppers | 2022 | test (last album) |

The exact assignment depends on the artist's album count and `test_albums=1`, `val_albums=1` parameters. With 5+ albums, the last goes to test, second-to-last to validation, rest to train.

## 2.4 Stage 3: Feature Engineering

Assuming "To Pimp a Butterfly" lands in the training set:

| Feature Block | Example Output |
|--------------|----------------|
| `TemporalBlock` | Release year (2015), career age (~4 years since 2011 debut) |
| `AlbumTypeBlock` | Album type = "Album" (one-hot encoded) |
| `ArtistHistoryBlock` | Prior albums count, running mean score |
| `GenreBlock` | PCA components from genre indicators (Hip Hop genre loading) |
| `CollaborationBlock` | Solo indicator, num_artists=1 |

The feature pipeline outputs `n_reviews = 7834` from `User_Ratings`.

## 2.5 Stage 4: Model Training

**Model data for this observation:**

| Array | Value |
|-------|-------|
| `artist_idx` | Integer index for "Kendrick Lamar" (e.g., 1042) |
| `album_seq` | `3` (third album chronologically in training set) |
| `prev_score` | `User_Score` of good kid, m.A.A.d city (e.g., 90) |
| `X` | Feature vector (genre PCA, temporal, album type, etc.) |
| `y` | `92` (User_Score target) |
| `n_reviews` | `7834` |

**Mean prediction structure:**
```
mu = artist_effect[3, kendrick_idx]     # Time-varying artist quality at album 3
   + X @ beta                           # Feature effects (genre, temporal, etc.)
   + rho * 90                           # AR(1) term from previous album score
```

**Noise scaling (heteroscedastic mode):**
```
sigma_scaled = sigma_obs / 7834^n_exponent
```

With 7834 reviews, this album gets significantly lower observation noise than albums with few ratings.

## 2.6 Stage 5-6: Evaluation and Reporting

If this album is in the test set, the model predicts:
- Point estimate: posterior mean of `mu`
- Prediction interval: from posterior samples of `mu + Normal(0, sigma_scaled)`
- Coverage: whether the 90% prediction interval contains the observed score of 92

---

# PART 10: KENDRICK LAMAR COMPLETE PATH TRACE

## 10.1 Numeric Trace Through Pipeline

This section traces plausible numeric values for "To Pimp a Butterfly" through each transformation.

### Raw Input

```
original_row_id: 45231 (hypothetical)
Artist: "Kendrick Lamar"
Album: "To Pimp a Butterfly"
Year: 2015
User Score: 92
User Ratings: 7834
Critic Score: 96
Critic Reviews: 46
Genres: "Hip Hop, Conscious Hip Hop, Jazz Rap, West Coast Hip Hop, Funk"
Album Type: "Album"
All Artists: "Kendrick Lamar"
```

### After Cleaning (Stage 1)

```
# Column renaming
User_Score: 92
User_Ratings: 7834
Critic_Score: 96
Critic_Reviews: 46
Album_Type: "Album"

# Date parsing
Release_Date_Parsed: 2015-03-15 00:00:00
date_risk: "low"
date_imputation_type: "none"
flag_future_year: False
flag_sparse_era: False

# Collaboration
num_artists: 1
is_collaboration: False
collab_type: "solo"

# Genre
primary_genre: "Hip Hop"

# Unknown artist flag
is_unknown_artist: False

# Descriptors: DROPPED
```

### After Filtering

```
# User score filter: User_Score=92 (not NaN) → PASS
# Score range: 0 <= 92 <= 100 → PASS
# Min ratings: 7834 >= 10 → PASS
# Result: INCLUDED in user_score_minratings_10.parquet
```

### After Splitting (Stage 2)

```
# Kendrick Lamar's discography (chronological):
# 1. Section.80 (2011) → train
# 2. good kid, m.A.A.d city (2012) → train
# 3. To Pimp a Butterfly (2015) → train
# 4. DAMN. (2017) → validation
# 5. Mr. Morale & The Big Steppers (2022) → test

# Assignment for TPAB: train split
# Reason: within_entity_temporal, position 3 of 5 (not last 2)
```

### After Feature Engineering (Stage 3)

```
# TemporalBlock outputs:
release_year_scaled: ~0.15  (2015, standardized)
career_age: ~4 years since debut

# AlbumTypeBlock outputs:
album_type_Album: 1
album_type_EP: 0
album_type_Mixtape: 0

# ArtistHistoryBlock outputs:
artist_album_count: 3  (3rd album in training set)
artist_mean_score: ~89  (mean of prior albums)

# GenreBlock outputs (PCA):
genre_pc_0: ~0.8   (Hip Hop loading)
genre_pc_1: ~-0.2  (genre dimension 2)
... (10 PCA components total)

# CollaborationBlock:
is_solo: 1
num_artists: 1

# Preserved:
n_reviews: 7834
```

### Model Input Arrays (Stage 4)

```
# This album's position in the model arrays:
artist_idx[i]: 1042  (Kendrick's integer index)
album_seq[i]: 3      (3rd album, within max_albums=50 cap)
prev_score[i]: 90.0  (good kid, m.A.A.d city User_Score)
X[i, :]: [0.15, 4.0, 1, 0, 0, 3, 89, 0.8, -0.2, ..., 1, 1]
y[i]: 92.0
n_reviews[i]: 7834
```

### Model Computation

```
# Hierarchical artist effect at album 3:
init_artist_effect[1042] ~ Normal(mu_artist, sigma_artist)
                         ~ Normal(0.0, 0.5)  [non-centered via LocScaleReparam]

# Random walk:
artist_effect[3, 1042] = init_artist_effect[1042]
                       + cumsum(rw_innovations[1042, 0:2])

# Fixed effects:
X[i] @ beta = sum of feature contributions

# AR(1) term:
rho * 90.0 = ~0.15 * 90.0 = ~13.5  (if rho ≈ 0.15)

# Mean prediction:
mu = artist_effect[3, 1042] + X @ beta + rho * 90.0

# Observation noise (heteroscedastic with n_exponent ≈ 0.35):
sigma_obs ≈ 8.0  (posterior mean, hypothetical)
sigma_scaled = 8.0 / 7834^0.35 ≈ 8.0 / 22.1 ≈ 0.36

# Very tight noise for this well-reviewed album
# Compare: album with 10 reviews → sigma_scaled = 8.0 / 10^0.35 ≈ 3.57

# Likelihood:
y[i] = 92 ~ Normal(mu[i], 0.36)
```

---

# VERIFICATION CHECKLIST

## File Path Verification

Every file path referenced in this document should exist in the codebase:

- [ ] `src/panelcast/cli.py`
- [ ] `src/panelcast/config/schema.py`
- [ ] `src/panelcast/data/ingest.py`
- [ ] `src/panelcast/data/cleaning.py`
- [ ] `src/panelcast/data/validation.py`
- [ ] `src/panelcast/data/split.py`
- [ ] `src/panelcast/data/lineage.py`
- [ ] `src/panelcast/data/manifests.py`
- [ ] `src/panelcast/features/base.py`
- [ ] `src/panelcast/features/pipeline.py`
- [ ] `src/panelcast/features/registry.py`
- [ ] `src/panelcast/features/album_type.py`
- [ ] `src/panelcast/features/artist.py`
- [ ] `src/panelcast/features/collaboration.py`
- [ ] `src/panelcast/features/core.py`
- [ ] `src/panelcast/features/genre.py`
- [ ] `src/panelcast/features/temporal.py`
- [ ] `src/panelcast/features/pca.py`
- [ ] `src/panelcast/features/errors.py`
- [ ] `src/panelcast/models/bayes/model.py`
- [ ] `src/panelcast/models/bayes/priors.py`
- [ ] `src/panelcast/models/bayes/fit.py`
- [ ] `src/panelcast/models/bayes/diagnostics.py`
- [ ] `src/panelcast/models/bayes/io.py`
- [ ] `src/panelcast/models/bayes/predict.py`
- [ ] `src/panelcast/pipelines/orchestrator.py`
- [ ] `src/panelcast/pipelines/stages.py`
- [ ] `src/panelcast/pipelines/prepare_dataset.py`
- [ ] `src/panelcast/pipelines/create_splits.py`
- [ ] `src/panelcast/pipelines/build_features.py`
- [ ] `src/panelcast/pipelines/train_bayes.py`
- [ ] `src/panelcast/pipelines/evaluate.py`
- [ ] `src/panelcast/pipelines/publication.py`
- [ ] `src/panelcast/pipelines/sensitivity.py`
- [ ] `src/panelcast/pipelines/manifest.py`
- [ ] `src/panelcast/pipelines/errors.py`
- [ ] `src/panelcast/evaluation/metrics.py`
- [ ] `src/panelcast/evaluation/calibration.py`
- [ ] `src/panelcast/evaluation/cv.py`
- [ ] `src/panelcast/reporting/figures.py`
- [ ] `src/panelcast/reporting/tables.py`
- [ ] `src/panelcast/reporting/model_card.py`
- [ ] `src/panelcast/utils/hashing.py`
- [ ] `src/panelcast/utils/git_state.py`
- [ ] `src/panelcast/utils/environment.py`
- [ ] `src/panelcast/utils/logging.py`
- [ ] `src/panelcast/utils/random.py`
- [ ] `src/panelcast/io/paths.py`
- [ ] `src/panelcast/io/readers.py`
- [ ] `src/panelcast/io/writers.py`

## Function/Class Verification

Key entities referenced in this document:

- [ ] `PipelineOrchestrator` class in `orchestrator.py`
- [ ] `PipelineConfig` dataclass in `orchestrator.py`
- [ ] `PipelineStage` dataclass in `stages.py`
- [ ] `StageContext` dataclass in `stages.py`
- [ ] `get_execution_order()` in `stages.py`
- [ ] `make_score_model()` in `model.py`
- [ ] `user_score_model` in `model.py`
- [ ] `critic_score_model` in `model.py`
- [ ] `compute_sigma_scaled()` in `model.py`
- [ ] `PriorConfig` dataclass in `priors.py`
- [ ] `get_default_priors()` in `priors.py`
- [ ] `MCMCConfig` dataclass in `fit.py`
- [ ] `FitResult` dataclass in `fit.py`
- [ ] `fit_model()` in `fit.py`
- [ ] `check_convergence()` in `diagnostics.py`
- [ ] `ConvergenceDiagnostics` dataclass in `diagnostics.py`
- [ ] `save_model()` in `io.py`
- [ ] `ModelManifest` dataclass in `io.py`
- [ ] `ModelsManifest` dataclass in `io.py`
- [ ] `RunManifest` Pydantic model in `manifest.py`
- [ ] `EnvironmentInfo` Pydantic model in `manifest.py`
- [ ] `GitStateModel` Pydantic model in `manifest.py`
- [ ] `prepare_datasets()` in `prepare_dataset.py`
- [ ] `PrepareConfig` dataclass in `prepare_dataset.py`
- [ ] `clean_albums()` in `cleaning.py`
- [ ] `RAW_TO_CANONICAL` dict in `cleaning.py`
- [ ] `CleaningConfig` dataclass in `cleaning.py`
- [ ] `filter_for_user_score_model()` in `cleaning.py`
- [ ] `filter_for_critic_score_model()` in `cleaning.py`
- [ ] `load_raw_albums()` in `ingest.py`
- [ ] `create_splits()` in `create_splits.py`
- [ ] `SplitConfig` dataclass in `create_splits.py`
- [ ] `within_entity_temporal_split()` in `split.py`
- [ ] `entity_disjoint_split()` in `split.py`
- [ ] `build_features()` in `build_features.py`
- [ ] `get_feature_blocks()` in `build_features.py`
- [ ] `FeaturePipeline` class in `pipeline.py`
- [ ] `BaseFeatureBlock` class in `base.py`
- [ ] `FeatureRegistry` class in `registry.py`
- [ ] `build_default_registry()` in `registry.py`
- [ ] `train_models()` in `train_bayes.py`
- [ ] `load_training_data()` in `train_bayes.py`
- [ ] `prepare_model_data()` in `train_bayes.py`
- [ ] `SensitivityResult` dataclass in `sensitivity.py`
- [ ] `PRIOR_CONFIGS` dict in `sensitivity.py`
- [ ] `run_prior_sensitivity()` in `sensitivity.py`
- [ ] `run_threshold_sensitivity()` in `sensitivity.py`
- [ ] `run_feature_ablation()` in `sensitivity.py`
- [ ] `AuditLogger` class in `lineage.py`
- [ ] `sha256_file()` in `hashing.py`
- [ ] `hash_dataframe()` in `hashing.py`

## No Old Artifact References

This document must NOT reference:
- [ ] `analyze_albums.py` (old monolithic script - does not exist)
- [ ] `bayesian_model.py` (old monolithic model - does not exist)
- [ ] PyMC (replaced by NumPyro)
- [ ] 4 model types (only 2: user_score_model, critic_score_model)
- [ ] `fit_retry_on_error()` (does not exist - replaced by `allow_divergences` flag)
- [ ] CSV outputs as primary format (replaced by Parquet throughout)
- [ ] CONFIG dict (replaced by Pydantic models + dataclasses)
- [ ] 14-step pipeline (replaced by 6-stage modular pipeline)
- [ ] Cross-validation branching (CV is separate, not a pipeline stage)
