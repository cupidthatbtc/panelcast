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
observation is **interval-censored to integers** — integer `k` contributes
`log(F(k+0.5) − F(k−0.5))` via the family's CDF, and replicated/predictive draws
are rounded (one `RoundedDistribution` wrapper handles both the inference
`log_prob` and the PPC/predictive `sample`, so the known-artist and cold-start
paths stay consistent). It composes with any location-scale family (`studentt`,
`normal`, `skew_normal`, `split_normal`); `beta` and `skew_studentt` reject it
(no usable CDF). Default off ⇒ the continuous likelihood is byte-identical to
before.

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

**Discretization diverges.** The `+discretize` combos are excluded from the table
above because the interval-censored integer likelihood produces a pathological
geometry: `studentt+discretize` posts 1000/1000 divergences (R-hat 3.05, MAE 47),
`skew_normal+discretize` an R-hat of ~1.4e7. The tail log-difference
`log(F(k+0.5) − F(k−0.5))` underflows and its gradient goes flat, and the
`betainc`-based Student-t CDF compounds it. Tracked as
[#4](https://github.com/cupidthatbtc/panelcast/issues/4) (log-space CDF / Student-t
JVP, or switch to dequantization). The cheap, universal half — rounding `y_rep` in
the PPC — is unaffected and stays on.

**Verdict: `studentt` remains the default.** Across both waves (`beta`,
`skew_normal`, `split_normal`, `discretize`) no implemented candidate moves the
`skewness`/`max` pins toward the interior; every one trades worse convergence or
point accuracy for the same or sharper pins. The bounded-skew misspecification is
confirmed and remains an open limitation, now with the deferred candidates below
(Beta-Binomial, mixture) as the next levers to try.

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

## Deferred candidates

Two heavier candidates are deferred for follow-up (tracked as GitHub issues)
rather than implemented now; the registry makes each a single new
`LikelihoodSpec`:

- **Beta-Binomial / aggregated-ratings likelihood**
  ([#2](https://github.com/cupidthatbtc/panelcast/issues/2)) — model `User_Score`
  as the mean of `n = User_Ratings` discrete ratings, so bounded support, left
  skew, and n-dependent noise fall out of one generative story (`n_reviews` is
  already threaded into `sample_obs`).
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

# Re-present the convergence + PPC of any run:
panelcast diagnose
```
