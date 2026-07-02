# What carries the signal

Two negative results, documented separately, are one finding: **the predictive
mass lives in the per-entity intercept.** The model is structurally "the
entity's own history ± calibrated noise" — neither measurement-noise modeling
nor covariates add much on top.

## The two halves

**Measurement-noise modeling is ~null (within observations).** The
heteroscedastic ablation let the data choose how strongly observation noise
should shrink with review count (`sigma ∝ n^-exponent`, diffuse prior centered
near 0.5). The posterior collapses to zero: 0.010 ± 0.005, 95% HDI
[0.003, 0.021] at 4 chains × 1000 (R-hat 1.00, bulk ESS 6224, 0 divergences;
[`.audit/n_exponent_posterior_analysis.md`](../.audit/n_exponent_posterior_analysis.md)).
Review count carries essentially no within-observation variance signal; the
homoscedastic default stands.

**Covariates are ~null (across entities).** On the entity-disjoint split —
never-seen artists, where only covariates can help — R² is 0.003 (RMSE 10.8,
n = 799), against 0.418 within-entity
([`.audit/baseline_metrics.json`](../.audit/baseline_metrics.json)). This is
not a model deficiency: every baseline (ridge, GBM) collapses the same way on
the same split ([`BASELINES.md`](BASELINES.md)). The features simply do not
transfer across entities.

## Why it is one result

Both ablations point at the same place. Remove the entity's own history and
almost nothing is left; model the noise on top of that history more carefully
and almost nothing changes. The variance the model explains enters through the
per-entity intercept and its random-walk trajectory — the entity's track
record — with the AR(1) term and covariates as small corrections.

Corollaries the thesis ties together:

- **The EIV gate defaulting off is principled, not just parity-cautious.**
  Errors-in-variables corrects measurement noise in the lagged regressor — noise
  the n_exponent posterior says is negligible. The v1-vs-v2 bake-off confirmed
  it: LOO +0.4 against an SE of ~29.6
  ([`.audit/model_v2_bakeoff/comparison.md`](../.audit/model_v2_bakeoff/comparison.md)).
- **Cold-start is the open frontier, not a tuning problem** (#41). With
  covariates ~null, improving never-seen-entity prediction needs new signal
  (a second pooling level, richer features), not a better likelihood.
- **It pairs with the durable PPC pins** ([`LIKELIHOOD_CANDIDATES.md`](LIKELIHOOD_CANDIDATES.md)):
  six likelihood families moved neither the upper-tail pins nor the predictive
  ledger much — again, the likelihood is not where the signal is.

## Evidence status

The original ablation was the thinnest fit in the repo (1 chain × 500, no
between-chain R-hat; posterior 0.0016 ± 0.0005). The #44 re-run at the
diagnostic config (4 chains × 1000, seed 42, logit-normal prior) replicates
the collapse with a real convergence diagnostic — 0.010 ± 0.005, R-hat 1.00,
bulk ESS 6224, tail ESS 2640, 0 divergences. The mean sits slightly higher
than the single-chain figure (an underdispersed 500-draw chain, and a
different prior parameterization), but both are ≈0 against a prior centered
near 0.5: the data rejects review-count-scaled noise either way.
