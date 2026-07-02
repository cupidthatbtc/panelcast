# beta_ceiling likelihood: screening result (#42)

**Hypothesis.** The plain `beta` family's support spans the nominal [0, 100],
but scores only occupy [low, ~88]; maybe the bounded-skew PPC pins (skewness,
max, q90) come from the unoccupied upper range rather than the distribution's
shape. `beta_ceiling` rescales the Beta onto [low, train-max + 0.5] to test
exactly that.

**Setup.** Diagnostic preset (4 × 2000), seed 42, `--likelihood-family
beta_ceiling --target-transform identity` (the family requires the raw score
scale — note this explicitly now that the default transform is
`offset_logit`), ~5k AOTY subset, 2026-07-02. Raw metrics:
`screening_metrics.json`.

## Result

| | studentt @ identity (pre-0.5.0 default) | beta_ceiling @ identity | studentt @ offset_logit (shipped default) |
|---|---|---|---|
| skewness / max / q90 pins | pinned | **pinned (1.000 / 0.995 / 1.000)** | pinned |
| lower tail | q50 pinned | **q10, q50, min re-pinned (0.000 / 0.003 / 0.999)** | q10 pinned only |
| total pinned stats | 4–5 | **6** | **4** |
| convergence | clean | R-hat 1.01, ESS 435, 0 div | clean |
| point (this run) | — | MAE 5.70, RMSE 8.11, R² 0.440 | — |

1. **The tested hypothesis is refuted.** Matching the support to the occupied
   range moves none of the upper-tail pins — skewness/max/q90 stay at the
   extremes. The pins are the shape mismatch itself, not a support artifact,
   consistent with every prior family attempt.
2. **It re-pins the lower tail.** The squeezed support concentrates mass and
   pins q10/q50/min — the statistics the shipped `offset_logit` default
   relieves. Net: 6 pinned stats vs the default's 4.
3. **One real finding: the ceiling fixes plain `beta`'s mixing.** ESS 435 vs
   plain beta's 304-with-a-divergence — the unoccupied range was beta's
   *sampling* pathology. Useful if a bounded family is ever needed for a
   domain whose data genuinely fills a sub-range.

**Verdict: not adopted.** The family stays available
(`--likelihood-family beta_ceiling`), non-default. Closes #42.
