# Extensibility Guide

Goal
- Add, remove, or modify features — or retarget the whole pipeline to a new
  dataset — without touching model code or pipelines.

## The three extension surfaces

1. **Dataset descriptor** (`src/panelcast/config/descriptor.py`): every
   domain-specific name — column names, date format, target bounds,
   posterior-site prefix, the feature-block list — comes from a
   `DatasetDescriptor`. One YAML under `configs/datasets/` retargets the
   pipeline; omitted keys keep their AOTY defaults ("default-equals-AOTY").
   Worked example: `configs/datasets/aero.yaml` + `docs/PORTING.md`.
2. **Feature registry** (`src/panelcast/features/registry.py`):
   `build_default_registry(descriptor)` registers the generic blocks
   (`temporal`, `entity_history`) closed over the descriptor's column names,
   plus any domain packs the descriptor lists.
3. **Feature packs** (`src/panelcast/features/packs/`): a pack is a module
   with a `register(registry)` function contributing domain-specific blocks.
   The music blocks (genre PCA, album type, collaboration) live in
   `packs/aoty.py`; a descriptor with `feature_packs: []` never sees them.

## Feature block pattern

1. Implement a block following `FeatureBlock` in
   `src/panelcast/features/base.py` (subclass `BaseFeatureBlock`).
2. Register it: in a feature pack's `register()` for domain-specific blocks,
   or in `build_default_registry()` for generic (descriptor-driven) blocks.
3. List it under `feature_blocks:` in the dataset descriptor with its params:

   ```yaml
   feature_blocks:
     - name: temporal
     - name: entity_history
     - name: my_block
       params:
         n_components: 10
   ```

4. Map any ablation flag onto it via the descriptor's `ablation_groups`
   (the CLI `--no-genre/--no-artist/--no-temporal` flags disable the groups
   named `genre` / `artist` / `temporal`).
5. Add unit tests for fit/transform behavior and leakage checks.

Required behavior
- Fit uses train data only.
- Transform works on any split (train/val/test) without re-fitting; the
  pipeline masks held-out target labels before history-based transforms.
- Block declares any dependencies in `requires`.
- Output column names are canonical (domain-independent) where possible;
  per-target columns derive from the descriptor's `model_prefix`
  (e.g. `user_prior_mean` for AOTY, `perf_prior_mean` for the aero example).

Current block files
- temporal (generic): `src/panelcast/features/temporal.py`
- entity_history (generic): `src/panelcast/features/history.py`
  (`artist_history` is its AOTY-pinned subclass in `features/artist.py`)
- core_numeric (generic): `src/panelcast/features/core.py` — pass-through
  for explicitly listed numeric columns with train-fitted imputation;
  pure-YAML feature selection (no Python needed for extra covariates)
- genre / genre_pca (aoty pack): `src/panelcast/features/genre.py`
- album_type (aoty pack): `src/panelcast/features/album_type.py`
- collaboration (aoty pack): `src/panelcast/features/collaboration.py`
- descriptor_pca: stub, deliberately unregistered

## Guard rails

- `tests/unit/test_no_domain_literals.py` fails the build if AOTY column
  literals creep into shared code paths outside the sanctioned whitelist.
- `tests/integration/test_feature_golden_hashes.py` pins the AOTY feature
  matrices byte-for-byte: a registry/block refactor must not change outputs.
- `tests/e2e/test_domain_portability.py` proves the descriptor surface:
  the aero domain runs the full pipeline with zero source changes, and
  `--dataset aoty_full` is byte-identical to running with no flag.

## Rebuilding features

- Full pipeline: `panelcast run` (stages data → splits → features → ...)
- Features only: `panelcast stage features`
- Outputs: `data/features/<split>/{train,validation,test}_features.parquet`
  and `data/features/manifest.json`

## Robustness checks

- Add a small fixture dataset (see `tests/e2e/conftest.py` and
  `tests/helpers/aero_data.py` for the two bundled domains).
- Confirm no target leakage by comparing train-only stats to full-data stats.
- Validate outputs are stable under seed changes.
