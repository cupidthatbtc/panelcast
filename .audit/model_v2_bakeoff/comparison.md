# model-v2 gate bake-off (v1 vs v2) — real subset, publication config

`v1` = both opt-in gates **OFF** (the shipped default). `v2` = `errors_in_variables` + `propagate_rw_horizon` **both ON** (`configs/bakeoff_v2_gates.yaml`), layered over `--preset publication` (4 chains × 5,000, warmup 3,000, Student-t). Both fits clear the convergence gate. ~5k-album AOTY subset; full-corpus validation is #15.

## Within-entity temporal holdout (N = 653)

| cell | gates | R-hat | bulk ESS | div | MAE | RMSE | R² | CRPS | 95% cov | LOO elpd | LOO se | p_loo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | off | 1.00 | 3141 | 0 | 5.638 | 8.266 | 0.4177 | 4.194 | 0.957 | -2486.2 | 29.6 | 258.6 |
| v2 | eiv+rw_horizon | 1.00 | 3280 | 0 | 5.638 | 8.266 | 0.4177 | 4.194 | 0.956 | -2485.8 | 29.6 | 258.3 |

## Cold-start (artist-disjoint, N = 799)

| cell | MAE | RMSE | R² | 95% cov |
| --- | --- | --- | --- | --- |
| v1 | 7.193 | 10.818 | 0.0028 | 0.896 |
| v2 | 7.193 | 10.819 | 0.0026 | 0.896 |

## Verdict: null — keep both gates default-off

Turning on both model-v2 gates changes nothing measurable. LOO moves **+0.4** (v2 over v1) against a per-model SE of ~29.6 (|z| ≈ 0.01); every point and calibration metric is identical to 3–4 significant figures. This is consistent with the `n_exponent ≈ 0` result — the EIV gate corrects measurement noise the data says is negligible — and confirms the gates are **parity-safe to ship default-off**.

Caveat: the long-horizon RW-variance gate only acts on predictions *past the longest training trajectory* (deep extrapolation), which this within-horizon holdout does not exercise, so its value can only be measured at the full corpus / longer horizons (#15).

Estimator note (#63): the LOO columns were computed under the pre-#63 estimator (PSIS-LOO on held-out data). The null verdict is unaffected — the two cells' log-likelihoods are near-identical, so any estimator puts |z| ≈ 0.
