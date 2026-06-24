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

The flagship AOTY dataset is not bundled (only a tiny fixture) and there is no
GPU available, so the headline AOTY-scale PPC numbers remain compute-bound (see
`MODEL_CARD.md`). To exercise the *mechanism* the review flagged, run a
synthetic panel deliberately built to be left-skewed and bounded (real
entity/AR structure, a long left tail clipped to [0, 100]):

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
- `beta` is the **recommended candidate**: its predictions cannot leave the
  score bounds, eliminating the out-of-bounds predictive mass that drives the
  symmetric likelihood's pinned max/min statistics.
- `skew_studentt` is **held** as a negative result: sinh-arcsinh skew on a
  heavy-tailed base produces extreme, unbounded draws and worsens tail
  statistics.
- The PPC p-values do **not** all reach the interior on this mild-skew,
  small-`n` synthetic; the definitive evaluation needs the real AOTY data at the
  publication configuration (4 chains × 5000), which is the remaining
  compute-bound step.

## Adopting a candidate

```bash
# Run the pipeline under the bounded Beta likelihood
panelcast run --likelihood-family beta

# Then re-present the convergence + PPC of that run
panelcast diagnose
```
