# Transform x latent-process bake-off (real subset, likelihood=studentt)

These are short **cell-selection** runs for comparing transform × latent process,
not the publication flagship. `conv=FAIL` here is a boundary R-hat miss (1.010 vs
the strict `< 1.01` gate) with **0 divergences** and bulk ESS comfortably above
the 400 floor (477–802) — the kind of margin that clears at the publication
configuration. The selected cell is validated separately at 4 × 5,000 (warmup
3,000); see *Real-data subset validation* in `MODEL_CARD.md` (R-hat 1.00, bulk
ESS 3,134, 0 divergences).

| cell | transform | latent | conv | rhat | ess | div | ppc_pin | pinned | mae | rmse | cov95 | pit_dev | crps | k_max | k>0.7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity_rw | identity | rw | FAIL | 1.010 | 802 | 0 | 3 | max,q50,q90 | 5.64 | 8.27 | 0.957 | 0.057 | 4.19 | 0.65 | 0 |
| identity_ar1 | identity | ar1 | FAIL | 1.010 | 577 | 0 | 4 | skewness,max,q50,q90 | 5.63 | 8.27 | 0.956 | 0.057 | 4.20 | 0.55 | 0 |
| offset_logit_rw | offset_logit | rw | FAIL | 1.010 | 649 | 0 | 4 | skewness,max,q10,q90 | 5.66 | 8.19 | 0.960 | 0.051 | 4.13 | 0.57 | 0 |
| offset_logit_ar1 | offset_logit | ar1 | FAIL | 1.010 | 477 | 0 | 4 | skewness,max,q10,q90 | 5.64 | 8.17 | 0.954 | 0.048 | 4.13 | 0.63 | 0 |
