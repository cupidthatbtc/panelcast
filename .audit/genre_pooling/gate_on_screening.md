# Genre pooling gate: paired screening result (#41)

**Setup.** Two identical runs on the ~5k AOTY subset, differing only in
`entity_group_pooling` (off vs on via `--config`): diagnostic preset at
**2 chains × (1000 warmup + 1000 samples), seed 42**, `offset_logit` default
transform, 2026-07-02. Screening settings — both arms sit below the strict
ESS gate (bulk 303 / 335, R-hat 1.01, 0 divergences) but are comparable to
each other, which is what a paired comparison needs. Raw metrics:
`metrics_gate_{off,on}.json` in this directory.

## Cold-start (entity_disjoint, n = 799) — the split the gate targets

| metric | gate off | gate on | delta |
|---|---:|---:|---:|
| MAE | 7.023 | 6.888 | **−0.135** |
| RMSE | 10.325 | 10.128 | **−0.198** |
| R² | 0.092 | 0.126 | **+0.034** |
| CRPS | 5.593 | 5.485 | **−0.108** |
| 80% cov / width | 0.841 / 28.6 | 0.854 / 28.7 | +0.013 / +0.2 |
| 95% cov / width | 0.959 / 42.9 | 0.959 / 43.1 | 0 / +0.1 |

## Within-entity temporal (n = 653) — should be unaffected, and is

Deltas: MAE −0.005, R² +0.002, CRPS −0.009, coverage ±0.002 — Monte-Carlo
noise scale. The gate only touches new-entity initialization; known-entity
prediction is untouched, as designed.

## Reading

1. **The gate delivers real cold-start signal.** Every cold-start metric
   moves the right way at once, coverage holds at nominal without widening,
   and the deltas are ~25× the within-split noise scale.
2. **The gain matches the predicted ceiling almost exactly.** The
   covariates-only diagnostic ([`covariates_only_r2.md`](covariates_only_r2.md))
   put the genre-block headroom at **R² ≈ 0.034**; the measured paired gain is
   **+0.034**. The pooling tier captures essentially all of the genre signal
   the features contain.
3. **Verdict: positive screening result; not yet a default flip.** Single
   seed, 2-chain screening scale. Promotion to default-on follows the usual
   playbook — publication-scale bake-off + multi-seed — tracked as the
   "genre-pooling promotion decision" under #75 (0.6.0). The gate ships in
   0.5.0 default-off.
