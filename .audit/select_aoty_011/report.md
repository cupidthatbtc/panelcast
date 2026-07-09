# panelcast select — focused 0.11 bake-off (horseshoe, impute_missing)

elpd_diff is arm minus reference (shipped defaults (reference arm)); positive = beats it. dse is
the *paired* difference SE from per-point elpd diffs on identical test data (#63).
Arms without a pointwise log-likelihood snapshot show "-" — no other estimator is
substituted.

| arm | elpd_diff | dse | z | d_cov80 | d_cov95 | pit_dev | ppc_pin | pinned | conv | wall_s | peak_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 79853b4bcf64 | +0.0 | 0.0 | +1.13 | +0.053 | +0.010 | 0.034 | 4 | skewness,max,q10,q90 | FAIL | 5157 | 1.48 |
| 750f957a8c71 | +0.0 | 0.0 | +0.00 | +0.053 | +0.010 | 0.034 | 4 | skewness,max,q10,q90 | FAIL | 5339 | 1.48 |
| d68348ed63f3 | -0.3 | 1.5 | -0.17 | +0.048 | +0.010 | 0.033 | 4 | skewness,max,q10,q90 | FAIL | 1945 | 1.48 |

**Verdict:** 79853b4bcf64 leads: +0.0 +/- 0.0 held-out ELPD (z +1.13) vs shipped defaults (reference arm). 79853b4bcf64 failed the convergence gate (rhat 1.010, ess 693, div 0) — treat its score as diagnostic-scale. 750f957a8c71 failed the convergence gate (rhat 1.010, ess 725, div 0) — treat its score as diagnostic-scale. d68348ed63f3 failed the convergence gate (rhat 1.010, ess 829, div 53) — treat its score as diagnostic-scale.
