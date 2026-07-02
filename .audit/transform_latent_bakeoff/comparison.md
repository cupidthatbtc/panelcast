# Transform x latent-process bake-off (real subset, likelihood=studentt)

| cell | transform | latent | conv | rhat | ess | div | ppc_pin | pinned | mae | rmse | cov95 | pit_dev | crps | elpd | se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity_rw | identity | rw | FAIL | 1.010 | 787 | 0 | 3 | max,q50,q90 | 5.64 | 8.27 | 0.957 | 0.056 | 4.19 | -2227.7 | 24.2 |
| identity_ar1 | identity | ar1 | FAIL | 1.010 | 635 | 0 | 4 | skewness,max,q50,q90 | 5.63 | 8.27 | 0.954 | 0.056 | 4.20 | -2229.9 | 24.4 |
| offset_logit_rw | offset_logit | rw | FAIL | 1.010 | 615 | 0 | 4 | skewness,max,q10,q90 | 5.66 | 8.19 | 0.959 | 0.049 | 4.13 | -2205.5 | 21.9 |
| offset_logit_ar1 | offset_logit | ar1 | FAIL | 1.010 | 477 | 0 | 4 | skewness,max,q10,q90 | 5.64 | 8.17 | 0.954 | 0.048 | 4.13 | - | - |

## Pairwise held-out elpd vs kept default (identity_rw)

elpd_diff is cell minus default (positive = beats the default); dse is the
*paired* difference SE from per-point elpd diffs on identical test data (#63).

| cell | elpd_diff | dse | z (diff/dse) |
| --- | --- | --- | --- |
| identity_ar1 | -2.2 | 0.9 | -2.39 |
| offset_logit_rw | +22.2 | 4.5 | +4.91 |
