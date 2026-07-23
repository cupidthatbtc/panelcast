# 0.13.1 fair-evaluation pass for the max-events cap fix

2026-07-22. This is the required follow-up to #247 after the estimator fix
merged in #252. Both sides of the 0.13.0 promotion comparison were re-evaluated
under the fixed train-only cap coordinate frame; no model was refit.

## Locked inputs

| role | run | model |
|---|---|---|
| 0.13.0 entity-obs default | `sel_aoty-entityobs-s42_2fee043e3e62_20260711T132510220130` | `user_score_20260712_014147.nc` |
| archived incumbent | `sel_aoty-entityobs-s42_750f957a8c71_20260711T061249536771` | `user_score_20260711_172330.nc` |

Both are the seed-42, 4-chain, 5,000 warmup + 5,000 draw confirmation
fits on the same 653-row within-entity holdout. The test-feature hashes match
the source manifests exactly (`6b2fb5…` primary, `1922e0…` entity-disjoint),
and the resolved feature input hash is unchanged (`4ec61a…`). Evaluation ran
on CPU with seed 42 and `PANELCAST_SAVE_LOG_LIKELIHOOD=1`; pointwise
score-scale log likelihoods were retained for the paired comparison.

## Old estimator vs fixed estimator

| model | estimator | MAE | RMSE | R² | CRPS | cov80 | cov95 | width95 | held-out ELPD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entity-obs | 0.13.0 archived | 5.277020 | 7.672906 | 0.498242 | 3.805345 | 0.830015 | 0.967841 | 31.033033 | -2133.736310 |
| entity-obs | fixed | 5.277020 | 7.672906 | 0.498242 | 3.805345 | 0.830015 | 0.967841 | 31.033033 | -2133.736492 |
| incumbent | 0.13.0 archived | 5.301650 | 7.649735 | 0.501268 | 3.867142 | 0.852986 | 0.963247 | 31.520517 | -2163.506286 |
| incumbent | fixed | 5.301651 | 7.649735 | 0.501268 | 3.867142 | 0.852986 | 0.963247 | 31.520517 | -2163.506461 |

The estimator correction is null at published precision. The largest headline
movement is `4.8e-7` in incumbent MAE/RMSE; coverage and interval widths are
unchanged; the ELPD shifts are `-1.82e-4` and `-1.75e-4`.

This null is mechanically expected on this holdout. The two exposed entities
have one test event after 50 training events. The legacy combined-count offset
placed that event at sequence 50; the fixed train-only frame places it at 51,
which the shipped `propagate_rw_horizon: false` policy then clamps back to the
trained horizon 50. The fix still restores the correct coordinate frame,
horizon-clamp count, and strict-mode guard, and it matters when the random walk
is propagated or a domain has a deeper test horizon.

## Promotion comparison under the fixed estimator

Using the retained pointwise held-out log likelihoods:

- ELPD difference (entity-obs − incumbent): **+29.769968**
- paired SE: **6.998596**
- z: **+4.253706**

This reproduces the 0.13.0 promotion result (+29.77 ± 7.00, z +4.25). The
entity-obs verdict and every published model headline therefore stand.

## Standardized baseline regeneration

`panelcast compare --baselines` was rerun against the fixed entity-obs metrics
with the matching feature stamp. The standardized within-entity ridge row is:

| MAE | RMSE | R² | CRPS | cov80 | cov95 | width95 |
|---:|---:|---:|---:|---:|---:|---:|
| 5.39 | 7.71 | 0.494 | 4.06 | 0.884 | 0.962 | 33.27 |

The generated CSV/Markdown/JSON remain reproducible artifacts under
`fair-eval-0131-entityobs-s42/reports/baselines/`; the curated values and
interpretation are updated in `docs/BASELINES.md`.

## Verdict

**PASS.** #247 changes no released conclusion on the AOTY subset, both promotion
arms were scored fairly under the corrected estimator, and the standardized
baseline row has been regenerated. This clears the estimator gate for 0.13.1.
