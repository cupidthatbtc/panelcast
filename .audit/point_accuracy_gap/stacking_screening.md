# gbm_offset stacking screening + confirmation (#76 → #86)

2026-07-03. E2: paired diagnostic runs (seed 42, 2×2000), identical settings,
only `gbm_offset: true` differing — one extra covariate in X (GBM-predicted
target over the other blocks' outputs, out-of-fold on train rows). Off arm =
[`../genre_pooling/metrics_gate_off.json`](../genre_pooling/metrics_gate_off.json);
on arm = `stacking_block_on_metrics.json` (this directory).

## Within-entity temporal (n=653)

| metric | off | on | delta |
|---|---:|---:|---:|
| MAE | 5.653 | **5.305** | −0.348 |
| R² | 0.429 | **0.501** | +0.072 |
| CRPS | 4.127 | 3.867 | −0.260 |
| 80% width | 20.0 | 18.0 | −2.0 (coverage held) |

MAE beats the standalone GBM (5.41, [`gbm_feature_ablation.md`](gbm_feature_ablation.md));
R² clears the ridge/GBM gap that motivated #76; intervals tighten at nominal
coverage. Publication-scale confirmation (combined fit with genre pooling):
within-entity MAE 5.302 / R² 0.501 — the screening effect replicates almost
exactly — paired held-out ELPD +224.2 (z +17.9) vs the 0.5.0 baseline.

**Promoted to default-on in 0.6.0** (#95); goldens regenerated for the new
default roster. Related negative results closing #76: column-drop
(`column_drop_metrics.json` — dropping `*_prior_*` breaks cold-start
calibration; the redundancy is conditional on a fitted entity effect) and the
posterior-median estimator ([`estimator_comparison.md`](estimator_comparison.md)
— keep the mean). Full verdicts on the issue threads.
