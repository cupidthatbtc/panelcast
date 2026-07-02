# n_exponent Posterior Analysis

## Re-run at diagnostic config (#44)

Date: 2026-07-02
Model: user_score_20260702_033937.nc
Config: 4 chains, 1000 warmup, 1000 samples, seed=42, learn_n_exponent=True,
prior=logit-normal(0, 1.0), diagnostic preset

| Statistic | Value |
|-----------|-------|
| mean | 0.0099 |
| std | 0.0048 |
| median | 0.0090 |
| 90% HDI | [0.0035, 0.0189] |
| 95% HDI | [0.0027, 0.0212] |
| R-hat | 1.00 |
| bulk ESS | 6224 |
| tail ESS | 2640 |
| divergences | 0 |

The collapse replicates with a real between-chain diagnostic: against a
diffuse prior centered near 0.5, the posterior concentrates at ≈0.01. The
mean sits ~6× above the original single-chain figure (an underdispersed
500-draw chain, and the beta-vs-logit-normal prior change), but the
substantive conclusion is unchanged — review count carries essentially no
within-observation variance signal, and the homoscedastic default
(`learn_n_exponent=False`, `n_exponent=0.0`) stands, now on 4000 draws
instead of 500.

Tree-depth saturation for this run: 0.0% of transitions hit
max_tree_depth=10 (first measurement through the #45 logging).

---

## Original analysis (Phase 4B, superseded evidence, same conclusion)

Date: 2026-03-23
Model: user_score_20260323_190303.nc
Config: 1 chain, 500 warmup, 500 samples, seed=42, learn_n_exponent=True, n_ref=53

### Posterior Summary

| Statistic | Value |
|-----------|-------|
| mean | 0.0016 |
| std | 0.0005 |
| median | 0.0015 |
| 90% HDI | [0.0008, 0.0025] |
| 95% HDI | [0.0008, 0.0028] |
| min | 0.0005 |
| max | 0.0034 |

### Interpretation

The posterior is concentrated near zero (0.002 ± 0.001), meaning the data
strongly favors homoscedastic noise — review count has essentially no effect
on observation-level variance.

### Prior Comparison

| Prior scale | 95% CI (induced) | Prior mean |
|------------|-------------------|------------|
| 0.5 | [0.272, 0.728] | 0.499 |
| 0.7 | [0.201, 0.799] | 0.498 |
| 1.0 | [0.122, 0.878] | 0.498 |
| 1.5 | [0.049, 0.950] | 0.497 |

All priors center mass around 0.5, but the posterior is at 0.002. The data
completely overwhelms any of these priors.

### Decision

**No change to prior scale.** Prior scale=1.0 is fine — the data dominates
regardless. Tightening the prior (e.g., to 0.7) would have zero practical
effect on the posterior or predictions.

This also suggests that for this dataset, the heteroscedastic mode provides
negligible benefit over the homoscedastic mode. The default `learn_n_exponent=False`
with `n_exponent=0.0` is the appropriate configuration.
