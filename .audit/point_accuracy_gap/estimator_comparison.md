# Point estimator: posterior median vs posterior mean (#76 step 5)

**Date:** 2026-07-02 · **Posterior:** archived 0.5.0 publication baseline
(`2026-07-02_084303`, 4×5000 draws, offset_logit default) · **Split:**
within_entity_temporal test (n=653), 20 000 predictive draws per point.

## Sanity anchor

The regenerated world is bit-for-bit the baseline's: `load_training_data`
reproduces the archived `data_hash` (`761a432d90d46d7f…`), and the recomputed
posterior-mean metrics match `evaluation/metrics.json` to 6 decimals
(MAE 5.660254, R² 0.42855).

## Result

| estimator | MAE | median AE | RMSE | R² | mean bias |
|---|---|---|---|---|---|
| posterior mean (shipped) | 5.660 | 4.210 | 8.188 | 0.4286 | +0.008 |
| posterior median | 5.584 | 4.013 | 8.183 | 0.4294 | **−0.611** |

## Verdict

The median estimator buys ~0.076 MAE (−1.3%) and ~0.20 median-AE, with RMSE
and R² unchanged — exactly the L1-vs-L2 textbook trade, realized through the
skewed Student-t predictive (median sits below the mean, hence the −0.61
systematic bias). This is not the point-accuracy lever: the GBM gap is ~0.4
MAE. **Keep the posterior mean as the shipped point estimate** (unbiasedness
and R²/RMSE-comparability with the baselines table matter more than a 1.3%
MAE cosmetic), and note the trade in BASELINES.md: part of the model-vs-GBM
MAE gap is the robust-loss/L2-estimator combination, not missing signal.

Raw numbers: `e4_results.json` (this directory).
