# Likelihood candidates for the skewed, bounded target

The review flagged a structural mismatch: the symmetric Student-t likelihood
against a left-skewed (skewness ≈ −1.79), bounded ([0, 100]) target, which pins
several posterior-predictive-check (PPC) p-values at the extremes (sd, skewness,
q50, q90, max). Two candidate likelihoods are implemented to address it,
selectable with `--likelihood-family`:

| Family | Idea | Sample sites added |
|--------|------|--------------------|
| `studentt` (default) | Symmetric Student-t, soft-clipped location | — |
| `normal` | Symmetric Gaussian | — |
| `skew_studentt` | sinh-arcsinh skew-t: Student-t base pushed through a Jones–Pewsey sinh-arcsinh transform with a learned skewness, then located/scaled | `{prefix}skewness` |
| `skew_normal` | sinh-arcsinh skew-**normal**: same transform on a *Normal* base — the skew of `skew_studentt` without the heavy tail that exploded it | `{prefix}skewness` |
| `split_normal` | Two-piece (Fechner) normal: separate left/right scales about `mu`; skew comes from σ_L ≠ σ_R, light tails on both sides | `{prefix}split_log_ratio` |
| `beta` | Score rescaled to (0, 1) (boundary squeeze) and modeled with a mean-precision Beta, affine-mapped back to the score bounds | `{prefix}phi` |
| `beta_binomial` | Score as the mean of `n` aggregated integer ratings — a score-scale `BetaBinomial(total = 100·n, a, b)`; inherently discrete (subsumes `--discretize-observation`). Gated to true rater-count domains | `{prefix}bb_phi` |

`{prefix}y` stays on the natural score scale under every family, so evaluation,
prediction, and the saved inference data are untouched by the choice.

### Plug-and-play registry

Families are defined once in `src/panelcast/models/bayes/likelihoods.py` as a
`LikelihoodSpec` (one entry per family in `REGISTRY`): how it contributes the
observation likelihood (`sample_obs`), how it draws cold-start predictive samples
(`predict_draws`), and — for location-scale families — its CDF (`cdf`). The
model's likelihood seam and the new-artist prediction dispatch both resolve a
family by name through `REGISTRY`, and the CLI/orchestrator validate against
`tuple(REGISTRY)`, so **adding a family is a single new entry** rather than edits
scattered across the model and prediction code. The four original families are
moved verbatim (a `slow` parity test pins their posterior draws bit-identical to
the pre-registry code).

### Discretization toggle (`--discretize-observation`)

Orthogonal to the family: AOTY scores are integer-valued, so a continuous
likelihood's replicated draws never land exactly on the observed integer
quantiles and the q50/q90 PPC p-values pin as an artifact. With this flag the
observation is made integer-aware by **dequantization** — inference conditions
the continuous base on `y + u` with a single fixed jitter `u ~ Uniform(−0.5,
0.5)` (held constant across every leapfrog step), and replicated/predictive
draws are rounded so `y_rep` stays integer. This keeps the gradient finite
everywhere; the marginalized interval-CDF alternative — integer `k` contributes
`log(F(k+0.5) − F(k−0.5))` via the family's CDF (`RoundedDistribution`) —
underflows to a flat-gradient cliff in the tails and walled the sampler
([#4](https://github.com/cupidthatbtc/panelcast/issues/4)), so it is kept
dormant behind `_DISCRETIZE_MODE` as a validated fallback. Either way one wrapper
handles both the inference `log_prob` and the PPC/predictive `sample`, so the
known-artist and cold-start paths stay consistent. It composes with any
location-scale family (`studentt`, `normal`, `skew_normal`, `split_normal`);
`beta` and `skew_studentt` reject it (no usable CDF). Default off ⇒ the
continuous likelihood is byte-identical to before.

## Controlled synthetic experiment

Before the GPU run below, the headline AOTY-scale PPC numbers were
compute-bound, so the *mechanism* the review flagged was first exercised on a
synthetic panel deliberately built to be left-skewed and bounded (real
entity/AR structure, a long left tail clipped to [0, 100]). That synthetic
experiment is retained here as the original evidence; the real-data result that
supersedes it is in the next section.

```bash
python scripts/experiment_likelihood_ppc.py
```

It fits each family at validation scale (2 chains × 500) and reports the PPC
p-values plus the raw predictive range.

### Result

On a synthetic target with observed skewness ≈ −0.58 (the entity + AR structure
absorbs much of the raw skew, so this is milder than AOTY's −1.79):

| Family | R-hat (max) | ESS bulk (min) | Predictive range | Verdict |
|--------|-------------|----------------|------------------|---------|
| `studentt` | 1.02 | 469 | **[−579, 580]** | Symmetric, heavy-tailed, **ignores the bounds** — the source of the max/min PPC pinning |
| `skew_studentt` | 1.01 | 376 | **[−3, 2784]** | Converges, but the skew on heavy Student-t tails **explodes the right tail** — a documented negative result |
| `beta` | 1.02 | 277 | **[0, 100]** | **Bounded by construction**; converges — the direct structural fix |

**Takeaways**

- Both new families **mix** at validation settings (R-hat ≤ 1.02), unlike the
  offset-logit transform, which failed to mix (R-hat 1.27–1.37).
- `beta` was the **synthetic-data recommended candidate**: its predictions
  cannot leave the score bounds, eliminating the out-of-bounds predictive mass
  that drives the symmetric likelihood's pinned max/min statistics. **This
  recommendation was overturned by the real-data result below** — the synthetic
  edge did not transfer to the real, strongly left-skewed AOTY scores.
- `skew_studentt` is **held** as a negative result: sinh-arcsinh skew on a
  heavy-tailed base produces extreme, unbounded draws and worsens tail
  statistics.
- The PPC p-values do **not** all reach the interior on this mild-skew,
  small-`n` synthetic; the definitive evaluation needed real AOTY data, which
  the **real-data result below** now provides (on a subset, at diagnostic
  scale). It reversed this section's recommendation.

## Real-data result (AOTY subset) — decisive

The compute-bound step above is now closed. The two families were fit on a real
**subset of the full AOTY corpus** — ~800 whole artists sampled with their full
discographies (`scripts/make_aoty_subset.py`), 5,182 albums with ≥10 user
ratings across 653 multi-album artists, observed user-score skewness **−2.08**
(matching the full corpus's −2.06). Both were run on GPU at the diagnostic
configuration (4 chains × 1000, warmup 1000; user-score model). This is a
subset, not the full corpus — but it is real, strongly left-skewed data at the
scale the synthetic experiment was standing in for.

| Family | R-hat (max) | ESS bulk (min) | Divergences | PPC pinned (p<0.01 or >0.99) | LOO Pareto-k > 0.7 |
|--------|-------------|----------------|-------------|------------------------------|--------------------|
| `studentt` | 1.01 | **795** | **0** | skewness, max, q50, q90 (**4**) | **0** |
| `beta` | 1.01 | 304 | 1 | skewness, min, max, q10, q50, q90 (**6**) | **35** |

Point/calibration were close (studentt MAE 5.64 / RMSE 8.27 / R² 0.42, 95%
coverage 0.957; beta MAE 5.67 / RMSE 8.08 / R² 0.44, 95% coverage 0.965; CRPS
≈ 4.2 for both).

**Takeaways**

- **`beta` does not fix the PPC mismatch on real data — it makes it worse.** It
  pinned *more* statistics than Student-t (six vs four), newly pinning `min`,
  `q10` and tightening `skewness`/`q50`. The bounded support does not, on these
  scores, pull the tail/quantile statistics back to the interior; the boundary
  squeeze concentrates mass and the pins sharpen. More MCMC samples sharpen
  pinned p-values further, so this is not a sample-count artifact.
- **`beta` mixes worse and breaks LOO.** Bulk ESS 304 vs 795 with one
  divergence, and 35 observations with Pareto-k > 0.7 (vs none for Student-t),
  which makes its LOO estimate unreliable.
- **Decision: keep `studentt` as the publication default.** The unilateral
  switch to `beta` in `configs/publication.yaml` (made on the synthetic evidence
  alone) has been reverted. `beta` remains available via `--likelihood-family`
  for anyone who wants to reproduce this comparison, but it is not adopted.
- **The likelihood mismatch is real and unresolved.** Even Student-t still pins
  `skewness`/`max`/`q50`/`q90` on real data; the bounded-skew misspecification
  is confirmed (not just a synthetic artifact) and remains an open limitation —
  none of the implemented candidates resolves it.

## Real-subset bake-off (new families + discretization)

The `beta` real-data result above closed the *first* wave. The skew-light
families (`skew_normal`, `split_normal`) and the discretization toggle are the
*second* wave — the two highest-value, lowest-risk levers against the remaining
pins: discretization attacks the integer-heaping q50/q90 pins directly, and the
skew-light families attack the `skewness`/`max` pins without the heavy-tail
blow-up that sank `skew_studentt`.

`scripts/bakeoff_likelihoods.py` runs each family (× discretize on/off) through
`panelcast run --preset diagnostic --stages train,evaluate` on the subset and
emits one comparison table (convergence, PPC pinned-count, point/calibration,
LOO Pareto-k):

```bash
AOTY_DATASET_PATH=data/raw/aoty_subset.csv \
    python scripts/bakeoff_likelihoods.py \
    --combos studentt,studentt+discretize,skew_normal,skew_normal+discretize,split_normal
```

Run from the GPU venv (`~/aoty-gpu`) after the data/splits/features stages exist
on disk. The continuous families were fit at diagnostic scale (4 chains × 1000);
the result is a **clean negative**:

| combo | rhat | ess | div | ppc_pin | pinned | skew p | mae | rmse | cov95 | crps | k_max |
|-------|------|-----|-----|---------|--------|--------|-----|------|-------|------|-------|
| `studentt` | 1.01 | **819** | 0 | **3** | max,q50,q90 | **0.99** | **5.64** | **8.27** | 0.957 | **4.19** | **0.55** |
| `skew_normal` | 1.04 | 123 | 1 | 5 | mean,skewness,max,q10,q50 | **1.00** | 6.48 | 8.94 | 0.974 | 4.56 | — |
| `split_normal` | 1.02 | 206 | 0 | 6 | mean,skewness,min,max,q10,q50 | **1.00** | 6.22 | 8.62 | 0.965 | 4.42 | 15.45 |

**Neither skew-light family helps.** The whole point of the second wave was to
move the `skewness` PPC p-value off Student-t's 0.99 pin. Both candidates pin it
*harder* — `skew_normal` and `split_normal` each land at exactly **1.00** and
newly trip `skewness` as an extreme statistic (Student-t's 0.99 sits just under
the >0.99 flag). They also pin *more* statistics overall (5 and 6 vs 3), mix worse
(bulk ESS 123 / 206 vs 819), cost ~0.6–0.8 MAE, and — for `split_normal` — break
LOO (Pareto-k 15.45). The learned skewness parameter does not relieve the
left-skew misspecification on these bounded, integer scores; it concentrates the
upper cluster and sharpens the pins, mirroring what `beta` did in the first wave.

**Dequantization fixes the toggle ([#4](https://github.com/cupidthatbtc/panelcast/issues/4)).**
The interval-censored `log(F(k+0.5) − F(k−0.5))` underflowed to a flat-gradient
cliff — `studentt+discretize` posted 1000/1000 divergences (R-hat 3.05, MAE 47),
`skew_normal+discretize` an R-hat of ~1.4e7. It is replaced by **dequantization**:
condition the continuous base on `y + u`, `u ~ Uniform(−0.5, 0.5)` a single fixed
jitter, and round on generation (the marginalized interval-CDF stays dormant
behind `_DISCRETIZE_MODE`). The `+discretize` combos now run (fresh diagnostic
4×1000; splits regenerated, so the Student-t baseline recomputes to ESS 779 / 4
pins — within sampling noise of the published run, same integer-quantile pins):

| combo | rhat | ess | div | ppc_pin | pinned | mae | rmse | cov95 | crps | k_max |
|-------|------|-----|-----|---------|--------|-----|------|-------|------|-------|
| `studentt` | 1.01 | 779 | 0 | 4 | skewness,max,q50,q90 | 5.64 | 8.27 | 0.957 | 4.19 | 0.67 |
| `studentt+discretize` | 1.01 | 753 | **0** | **3** | skewness,max,q90 | **5.64** | 8.28 | 0.957 | 4.20 | 0.61 |
| `skew_normal+discretize` | 1.03 | 86 | 0 | 6 | mean,skewness,max,q10,q50,q90 | 6.46 | 8.92 | 0.975 | 4.55 | — |
| `split_normal+discretize` | 1.02 | 191 | 0 | 7 | mean,skewness,min,max,q10,q50,q90 | 6.21 | 8.61 | 0.969 | 4.42 | 15.91 |

The numerics pathology is gone — **0 divergences** everywhere, R-hat ≤ 1.03, finite
gradients (a `jax.grad` finiteness test guards the tails). And on `studentt` the
median integer-heaping pin is relieved: **q50's p-value moves 0.009 → 0.082** (off
the extreme list) and q10 0.062 → 0.167. But `q90` (1.00), `max` (1.00) and
`skewness` (0.99) do not budge — they are the bounded left-skew misspecification,
not an integer-grid artifact, so dequantization cannot touch them. The skew
families re-pin everything (6–7 stats) at worse MAE/ESS, mirroring their continuous
behavior. So heaping drives `q50` specifically (hypothesis confirmed there); the
residual pins remain the deferred candidates' (Beta-Binomial, mixture) target.

**Verdict: `studentt` remains the default.** Across both waves (`beta`,
`skew_normal`, `split_normal`, `discretize`) no implemented candidate moves the
`skewness`/`max` pins toward the interior; every one trades worse convergence or
point accuracy for the same or sharper pins. The bounded-skew misspecification is
confirmed and remains an open limitation. Beta-Binomial (issue #2) has since been
implemented and tried (intractable uncapped — see below); the two-component
mixture is the remaining lever.

### Publication-scale confirmation

At the full publication configuration (4 chains × 5000, warmup 3000) the
Student-t model **passes the convergence gate** on the subset — R-hat 1.00, bulk
ESS **3,134**, 0 divergences, and a reliable LOO (Pareto-k max 0.43, none > 0.7).
It is well-calibrated (95% coverage 0.957, 80% 0.856) and accurate (MAE 5.64,
RMSE 8.27, CRPS 4.19). It nonetheless **still pins the same four PPC statistics**
— skewness (p 0.99), max (1.00), q50 (0.006), q90 (1.00). Five times the posterior
samples *sharpens* the pins rather than relaxing them, which confirms the
bounded-skew mismatch is a structural property of the likelihood, not a
sample-count or convergence artifact. The remaining open item is full-corpus
scale; the likelihood mismatch itself is the deferred candidates' target.

## Beta-Binomial (issue #2) — implemented, intractable uncapped

The first deferred candidate is now implemented as the `beta_binomial` family:
`User_Score` is modeled as the mean of `n = User_Ratings` integer ratings, the
rating sum being `BetaBinomial(total = 100·n, a, b)` with a mean-precision
parameterization kept on the score scale (`BetaBinomialScore`). Bounded support,
left skew, and n-dependent noise all follow from one generative story, and it is
gated behind a descriptor `n_obs_is_aggregation_count` flag so it is not offered
where the observation count is not a rater count (e.g. the aerospace example's
`Sensor_Samples`). It was put through the same subset bake-off (4 chains × 1000)
against the `studentt` baseline and the bounded `beta`:

| combo | conv | rhat | ess | div | ppc_pin | pinned | mae | k_max |
|-------|------|------|-----|-----|---------|--------|-----|-------|
| `studentt` | FAIL | 1.01 | 754 | 0 | 3 | max,q50,q90 | 5.64 | 0.65 |
| `beta` | FAIL | 1.02 | 236 | 0 | 6 | skewness,min,max,q10,q50,q90 | 5.67 | 2.75 |
| `beta_binomial` | **did not converge** | — | — | — | — | — | — | — |

**`beta_binomial` is intractable on real review-count data at this scale.** The
sampler maxes out the NUTS tree depth (1023 leapfrog steps) on *every* iteration
with the step size collapsed to ~1e-4, running ~5× slower than the other families
(projected ~2 h for the four chains vs ~9 min each for `studentt`/`beta`) and never
leaving warmup. The cause is the **over-confidence / float32-precision** failure the
roadmap flagged: subset albums carry up to ~23k ratings, so `total_count = 100·n`
reaches ~2.3M and `BetaBinomial.log_prob` differences `gammaln` values of order ~3e7
in float32 — the catastrophic cancellation leaves a noisy gradient NUTS cannot
follow. It is the same *kind* of negative as `beta`/`skew_normal` (the aggregated
story does not, on these scores, relieve the bounded-skew pins) but it surfaces as
non-convergence rather than sharper pins.

The open lever is a likelihood-side **`n_reviews` cap** — bound `total_count` to a
precision-safe range (≈ ≤ 2e4, so the bulk of albums keep their exact aggregation
count while only the mega-reviewed tail is clipped) — or a float64 log-density.
Tracked on [#2](https://github.com/cupidthatbtc/panelcast/issues/2); the family
ships available (`--likelihood-family beta_binomial`) but is **not adopted**.

## Deferred candidates

One heavier candidate remains deferred for follow-up; the registry makes it a
single new `LikelihoodSpec`:

- **Two-component mixture**
  ([#3](https://github.com/cupidthatbtc/panelcast/issues/3)) — a dense 65–85
  cluster plus a thin flop tail (`MixtureSameFamily` with ordering/label-switching
  handling).

## Adopting a candidate

```bash
# The publication default is the symmetric Student-t (see the real-data result
# above). To test a skew-light candidate or the integer discretization:
panelcast run --likelihood-family skew_normal
panelcast run --likelihood-family studentt --discretize-observation

# To reproduce the bounded-Beta comparison instead:
panelcast run --likelihood-family beta

# Beta-Binomial (aggregated ratings) is available but intractable uncapped at
# scale (issue #2); it requires a true rater-count descriptor:
panelcast run --likelihood-family beta_binomial

# Re-present the convergence + PPC of any run:
panelcast diagnose
```
