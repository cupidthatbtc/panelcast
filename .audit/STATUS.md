# Audit Status

Baseline tag: `audit-baseline`
Started: 2026-03-23

## Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| 0 (Baseline) | done | Tag created, data hashed, mypy=0 errors |
| 1 (Bug fixes) | done | 1A-1F all complete, all tests pass |
| 2 (Figures) | done | Predictions + reliability plots wired into publication pipeline |
| 3 (CRPS) | done | Vectorized via sorted order statistics, properscoring import removed |
| 4A (prev_score) | done | Pre-split global mean from dataset_stats.json, fallback to train mean |
| 4B (n_exponent) | done (no change) | Posterior = 0.002±0.001 — data overwhelms any prior. Scale=1.0 is fine. |
| 4C (HDI) | done | _hdi_per_observation via sliding window, interval_type param added |
| 4D (max_albums doc) | done | Docstring explains random-walk motivation for hard cap |
| 5 (Artist viz) | done | save_artist_prediction_plot + select_artist_subsets (4 categories, deduped) |
| 6 (Tests/docs) | done | 6C existed already, 6D increased to 500 samples+@slow, 6E design comment+TODO |
| 7 (Monitoring) | done | 7A prediction bounds logging + stats in summary JSON |

## Deferred Items

(none yet)

## Baseline Data

- Data hash: `data/raw/all_albums_full.csv` → see `baseline_data_hashes.txt`
- Baseline metrics: no metrics.json available (pipeline not recently run)
- mypy error count: 0 (with --check-untyped-defs). Enabled check_untyped_defs=true and removed dict-item from disabled codes.
