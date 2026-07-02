# Multi-seed stability check (#40)

The bake-off's selection decisions rode on a single seed (42). This sweep
re-fits the two contending cells at seeds 43 and 44 (same diagnostic preset,
4 chains × 1000, same on-disk splits/features; `multiseed/<cell>_seed<N>/`
snapshots) and reports the headline metrics as mean ± across-seed SD over
seeds {42, 43, 44}. The seed changes only the MCMC/PPC randomness, not the
data — exactly the Monte-Carlo noise the single-seed decisions were exposed
to. All six fits: 0 divergences.

## Within-entity temporal split, per cell

| metric | identity_rw | offset_logit_rw |
| --- | --- | --- |
| MAE | 5.637 ± 0.002 | 5.662 ± 0.002 |
| RMSE | 8.267 ± 0.003 | 8.191 ± 0.005 |
| R² | 0.417 ± 0.000 | 0.428 ± 0.001 |
| CRPS | 4.193 ± 0.001 | 4.133 ± 0.002 |
| 95% coverage | 0.957 ± 0.001 | 0.959 ± 0.000 |
| LOO elpd | −2485.6 ± 0.4 | −2426.1 ± 1.8 |

## Paired per-seed deltas (offset_logit_rw − identity_rw)

| metric | seed 42 | seed 43 | seed 44 | mean ± SD |
| --- | --- | --- | --- | --- |
| MAE | +0.025 | +0.021 | +0.029 | +0.025 ± 0.004 |
| RMSE | −0.078 | −0.078 | −0.074 | −0.077 ± 0.002 |
| CRPS | −0.060 | −0.061 | −0.059 | −0.060 ± 0.001 |
| LOO elpd | +60.1 | +61.3 | +57.0 | +59.5 ± 2.2 |

## Reading

Every headline delta reproduces with the sign and near-exact magnitude at all
three seeds. The LOO advantage of offset_logit (+57 to +61) is ~27× its
across-seed SD; the CRPS and RMSE advantages and the small MAE cost are
likewise stable. Seed noise (MAE SD ≈ 0.002 per cell) sits an order of
magnitude below the 0.03-MAE decision surface the original issue worried
about, so the bake-off's verdicts were made above the seed-noise floor.

Caveat: the LOO magnitudes inherit the held-out-LOO estimator concern (#63);
the across-seed *stability* shown here is unaffected, but the absolute +60
should be re-derived under the corrected estimator before the #43 decision.

Reproduce: `panelcast run --preset diagnostic --stages train,evaluate
--likelihood-family studentt --target-transform {identity,offset_logit}
--latent-process rw --seed {43,44}` with `PANELCAST_SAVE_LOG_LIKELIHOOD=1`,
snapshotting `outputs/evaluation/` per run.
