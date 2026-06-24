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
| `beta` | Score rescaled to (0, 1) (boundary squeeze) and modeled with a mean-precision Beta, affine-mapped back to the score bounds | `{prefix}phi` |

`{prefix}y` stays on the natural score scale under every family, so evaluation,
prediction, and the saved inference data are untouched by the choice.

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

## Adopting a candidate

```bash
# The publication default is the symmetric Student-t (see the real-data result
# above). To reproduce the bounded-Beta comparison instead:
panelcast run --likelihood-family beta

# Re-present the convergence + PPC of any run:
panelcast diagnose
```
