# n_exponent Posterior Analysis (Phase 4B)

Date: 2026-03-23
Model: user_score_20260323_190303.nc
Config: 1 chain, 500 warmup, 500 samples, seed=42, learn_n_exponent=True, n_ref=53

## Posterior Summary

| Statistic | Value |
|-----------|-------|
| mean | 0.0016 |
| std | 0.0005 |
| median | 0.0015 |
| 90% HDI | [0.0008, 0.0025] |
| 95% HDI | [0.0008, 0.0028] |
| min | 0.0005 |
| max | 0.0034 |

## Interpretation

The posterior is concentrated near zero (0.002 ± 0.001), meaning the data
strongly favors homoscedastic noise — review count has essentially no effect
on observation-level variance.

## Prior Comparison

| Prior scale | 95% CI (induced) | Prior mean |
|------------|-------------------|------------|
| 0.5 | [0.272, 0.728] | 0.499 |
| 0.7 | [0.201, 0.799] | 0.498 |
| 1.0 | [0.122, 0.878] | 0.498 |
| 1.5 | [0.049, 0.950] | 0.497 |

All priors center mass around 0.5, but the posterior is at 0.002. The data
completely overwhelms any of these priors.

## Decision

**No change to prior scale.** Prior scale=1.0 is fine — the data dominates
regardless. Tightening the prior (e.g., to 0.7) would have zero practical
effect on the posterior or predictions.

This also suggests that for this dataset, the heteroscedastic mode provides
negligible benefit over the homoscedastic mode. The default `learn_n_exponent=False`
with `n_exponent=0.0` is the appropriate configuration.
