# panelcast — entity-obs publication-scale confirmation (stage A)

elpd_diff is arm minus reference (shipped defaults (reference arm)); positive = beats it. dse is
the *paired* difference SE from per-point elpd diffs on identical test data (#63).
Arms without a pointwise log-likelihood snapshot show "-" — no other estimator is
substituted.

| arm | elpd_diff | dse | z | d_cov80 | d_cov95 | pit_dev | ppc_pin | pinned | conv | wall_s | peak_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2fee043e3e62 | +29.8 | 7.0 | +4.25 | +0.028 | +0.019 | 0.028 | 2 | skewness,max | PASS | 26751 | 7.59 |
| 750f957a8c71 | +0.0 | 0.0 | +0.00 | +0.050 | +0.010 | 0.031 | 4 | skewness,max,q10,q90 | PASS | 25467 | 7.39 |

**Verdict:** 2fee043e3e62 leads: +29.8 +/- 7.0 held-out ELPD (z +4.25) vs shipped defaults (reference arm).
