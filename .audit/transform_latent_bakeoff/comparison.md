# Transform x latent-process bake-off (real subset, likelihood=studentt)

| cell | transform | latent | conv | rhat | ess | div | ppc_pin | pinned | mae | rmse | cov95 | pit_dev | crps | loo | se | p_loo | waic | k_max | k>0.7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity_rw | identity | rw | FAIL | 1.010 | 787 | 0 | 3 | max,q50,q90 | 5.64 | 8.27 | 0.957 | 0.056 | 4.19 | -2485.8 | 29.7 | 258.0 | -2485.4 | 0.54 | 0 |
| identity_ar1 | identity | ar1 | FAIL | 1.010 | 635 | 0 | 4 | skewness,max,q50,q90 | 5.63 | 8.27 | 0.954 | 0.056 | 4.20 | -2491.0 | 29.1 | 261.1 | -2488.4 | 0.55 | 0 |
| offset_logit_rw | offset_logit | rw | FAIL | 1.010 | 615 | 0 | 4 | skewness,max,q10,q90 | 5.66 | 8.19 | 0.959 | 0.049 | 4.13 | -2425.6 | 26.9 | 220.1 | -2423.7 | 0.74 | 1 |
| offset_logit_ar1 | offset_logit | ar1 | FAIL | 1.010 | 477 | 0 | 4 | skewness,max,q10,q90 | 5.64 | 8.17 | 0.954 | 0.048 | 4.13 | -2431.7 | 26.3 | 224.4 | -2428.3 | 0.63 | 0 |

## Pairwise LOO vs kept default (identity_rw)

elpd_diff is cell minus default (positive = beats the default); dse is the
*paired* difference SE from `az.compare` on identical test data.

| cell | elpd_diff | dse | z (diff/dse) |
| --- | --- | --- | --- |
| identity_ar1 | -5.2 | 2.6 | -2.00 |
| offset_logit_rw | +60.1 | 4.6 | +12.96 |

_No current pointwise log-likelihood snapshot for offset_logit_ar1, so the pairwise table omits them._
