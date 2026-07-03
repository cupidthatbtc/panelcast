# Verdict ledger — re-verification of the recorded selection history (#98)

**Date:** 2026-07-03. **Why:** several model-selection verdicts were recorded while the
invalid PSIS-LOO-on-held-out estimator was live (replaced by the direct held-out lppd in
PR #63, shipped 0.5.0 on 2026-07-02). Before `panelcast select` can claim to reproduce
the manual history (#78 acceptance item 2), the history itself needs an evidence audit.
This ledger is that audit, and the ground truth the AOTY reproduction sweep is judged
against.

**Scope decision (2026-07-03):** verdicts with persisted per-point `log_likelihood.nc`
snapshots are recomputed via `scripts/bakeoff_transform_latent.py --reassemble` — no
refits. Snapshot-less pre-#63 verdicts are marked **unverified (pre-#63 evidence)**
rather than re-run: the select sweep re-tries every one of those options anyway.

**Classifications**

- **verified-solid** — the evidence never involved the retired estimator (PPC pins,
  ESS/R-hat, posterior concentration, point metrics, or post-#63 paired ELPD).
- **recomputed** — re-derived today from persisted per-point snapshots with the
  corrected estimator; the number below is the fresh recompute.
- **unverified (pre-#63 evidence)** — rests at least partly on a pre-#63 LOO figure
  and has no snapshots to recompute from. The verdict may well be right; it is simply
  not re-derivable without a refit. `select` re-tries these options on AOTY.

## Recompute receipt

`pixi run python scripts/bakeoff_transform_latent.py --reassemble` (2026-07-03, CPU,
no refits). The reassembled `comparison.{md,json}` came out **byte-identical** to the
committed artifacts — the decision journal's numbers reproduce exactly from the
persisted snapshots:

| cell vs identity_rw | elpd_diff | dse | z |
|---|---|---|---|
| identity_ar1 | −2.2 | 0.9 | −2.39 |
| offset_logit_rw | +22.2 | 4.5 | +4.91 |

Multi-seed pairs recomputed directly from the persisted per-point snapshots with the
same `_pointwise_elpd` pairing (n = 653 held-out points per cell):

| offset_logit_rw vs identity_rw | elpd_diff | dse | z |
|---|---|---|---|
| seed 43 | +22.3 | 4.5 | +4.93 |
| seed 44 | +22.0 | 4.5 | +4.86 |

Both match the values recorded in `.audit/transform_latent_bakeoff/MULTISEED.md`.

## A. Transform / latent process

| # | Option | Verdict | Classification | Evidence |
|---|--------|---------|----------------|----------|
| 1 | `target_transform=offset_logit` | promoted (default since 0.5.0) | **recomputed** | paired held-out ELPD vs identity_rw re-derived from `.audit/transform_latent_bakeoff/{identity_rw,offset_logit_rw}/log_likelihood.nc`: **+22.2 ± 4.5, z +4.91**, reproduced at seeds 43/44 (above). |
| 2 | `latent_process=ar1` | held (gate never triggered) | **recomputed** | paired ELPD identity_ar1 vs identity_rw re-derived from snapshots: **−2.2 ± 0.9, z −2.39** — a small resolved decrement, not the clear win the gate requires. The `offset_logit_ar1` cell has **no snapshot** (its ~8 h/chain geometry was never re-run post-#63); that single cell stays unmeasurable without a refit and renders `-` in the reassembled comparison. |
| 3 | `ar_center=global` | adopted | **verified-solid** | geometry, not ELPD: corr(rho, mu_artist) −0.997 → +0.016, debut AR terms exactly zero, prior-predictive flipped to pass (docs/decisions/DECISIONS.md). |

## B. Likelihood families (all vs `studentt`; PPC-pin / ESS evidence — no LOO involved)

| # | Option | Verdict | Classification | Evidence |
|---|--------|---------|----------------|----------|
| 4 | `studentt` (default) | kept | **verified-solid** | publication-scale 4×5000: R-hat 1.00, ESS 3134, 0 div; still pins skewness/max/q90 — structural, not sample-count (docs/LIKELIHOOD_CANDIDATES.md). |
| 5 | `beta` | rejected on real data | **verified-solid** | 6 pins vs studentt's 4, bulk ESS 304 vs 795, 1 divergence; the synthetic recommendation did not transfer. |
| 6 | `skew_studentt` | held (documented negative) | **verified-solid** | synthetic predictive range [−3, 2784]; sinh-arcsinh on heavy tails explodes the right tail. |
| 7 | `skew_normal` | rejected | **verified-solid** | skew p pins at 1.00, 5 pins vs 3, ESS 123, MAE 6.48. |
| 8 | `split_normal` | rejected | **verified-solid** | 6 pins vs 3, ESS 206, MAE 6.22. |
| 9 | `discretize_observation` | default-off; dequantization mechanism adopted over interval-CDF | **verified-solid** | relieves q50 heaping pin (p 0.009→0.082); skewness/max/q90 unmoved; interval-CDF alternative diverged 1000/1000. |
| 10 | `beta_binomial` | convergent negative | **verified-solid** | with rater cap 100: R-hat 1.01, ESS 499, 0 div, but re-pins the same 6 as beta at MAE 5.65; uncapped intractable in float32. |
| 11 | `mixture` | measured negative | **verified-solid** | continuous form R-hat 1.53 / ESS 7; +discretize converges but re-pins — mismatch structural across 5 families. |
| 12 | `beta_ceiling` | rejected (#42) | **verified-solid** | 2026-07-02 screening: upper-tail pins unmoved (skewness 1.000/max 0.995/q90 1.000), 6 pins vs 4; genuine finding: fixes plain beta's mixing (ESS 435, 0 div). `.audit/beta_ceiling/screening.md`. |

## C. Noise / observation-model gates

| # | Option | Verdict | Classification | Evidence |
|---|--------|---------|----------------|----------|
| 13 | `learn_n_exponent` (+ `n_exponent=0`) | null result, default-off | **verified-solid** | posterior concentration against the prior: #44 re-run 2026-07-02, mean 0.0099 ± 0.0048, HDI [0.0027, 0.0212], ESS 6224; no LOO involved. |
| 14 | `n_exponent_prior=logit-normal` | adopted (secondary knob) | **verified-solid** | prior-parameterization change with #44; conclusion (≈0) unchanged under either prior. |
| 15 | `artist_effect_param=noncentered`, `sigma_artist_prior_type=halfnormal` | adopted | **verified-solid** | won the 4-variant mixing bake-off (mixing metrics, estimator-independent). Artifact lives at `outputs/experiments/sigma_artist_mixing.json` — outside `.audit/`; gap noted below. |
| 16 | `heteroscedastic_entity_obs` (C1) | rejected | **unverified (pre-#63 evidence)** | the IMDb half of the reject rests on a "LOO ELPD −435 ± ~75" regression in exactly the pre-#63 formulation, undated, no post-#63 note, no snapshots (`docs/decisions/entity_overdispersion.md`). The econ half (no variant converges, ESS 4-5) is estimator-independent and stands. Composite classification: unverified — the calibration-vs-sharpness trade needs re-measurement under the corrected estimator. The select sweep re-tries C1 on AOTY. |
| 17 | `sigma_obs_prior_type=lognormal` (C2) | rejected | **verified-solid** | econ: lognormal alone leaves ESS at 4 (convergence failure, estimator-independent); c1c2 pathological (172 divergences). The overflowed-LOO detail is moot given the convergence evidence. |

## D. Model-v2 gates

| # | Option | Verdict | Classification | Evidence |
|---|--------|---------|----------------|----------|
| 18 | `errors_in_variables` | null, default-off | **unverified (pre-#63 evidence)** | `.audit/model_v2_bakeoff/comparison.md` self-flags its LOO columns as pre-#63; no `log_likelihood.nc` persisted, so the paired diff cannot be re-derived without a refit. The null is *very likely robust* (the two cells' log-likelihoods are near-identical; any estimator puts \|z\| ≈ 0, and all point/calibration metrics agree to 3-4 s.f.) — but formally the ELPD figure is pre-#63. `select` re-tries EIV. |
| 19 | `propagate_rw_horizon` | null on subset, default-off | **verified-solid** | within-horizon holdout cannot exercise the gate (acts only past max_seq_train); metrics unchanged by construction. Real measurement deferred to full corpus (#15). |

## E. 0.6.0 point-accuracy promotions (post-#63 by construction)

| # | Option | Verdict | Classification | Evidence |
|---|--------|---------|----------------|----------|
| 20 | `entity_group_pooling` | promoted (tri-state auto, 0.6.0) | **verified-solid** | #85 screening 2026-07-02: cold-start MAE −0.135 / R² +0.034 (matches Ridge headroom exactly); multi-seed direction holds; publication confirmation ELPD +224.2 ± 12.6 (z +17.9, corrected estimator, combined with gbm_offset). |
| 21 | `gbm_offset` | promoted (0.6.0) | **verified-solid** | #86 screening 2026-07-03: MAE 5.653→5.305, R² +0.072, CRPS −0.260 at held coverage; publication confirmation shares the z +17.9 paired ELPD above. |
| 22 | posterior-mean point estimator | kept (median rejected) | **verified-solid** | e4 on archived 0.5.0 baseline: median buys ~0.076 MAE but −0.611 systematic bias; point metrics only. |
| 23 | drop `*_prior_*` columns from X | rejected | **verified-solid** | breaks cold-start calibration; redundancy is conditional on a fitted entity effect. |
| 24 | GBM feature ablation | diagnostic only | **verified-solid** | model already extracts the history signal (GBM history-only R² 0.412 ≈ model 0.417); genre PCA dead weight for GBM point accuracy. No adoption to audit. |

## F. Lower-evidence defaults (recorded for completeness)

| # | Option | Verdict | Classification | Evidence |
|---|--------|---------|----------------|----------|
| 25 | `debut_prev_score_source=train_mean` | adopted | **verified-solid** | leakage principle (train-split-only mean), not a metric bake-off; `dataset_stats` kept as rollback gate. |
| 26 | `exclude_rw_raw_from_collection=true` (publication.yaml) | adopted | **verified-solid** | resource config, not a selection verdict: 94% peak-GPU cut with bit-identical posterior for all other sites (parity-tested). |

## Gaps list — what a future re-run would need

No persisted per-point `log_likelihood.nc` exists for these comparisons; they cannot be
recomputed under the corrected estimator without refits (deliberately out of scope here):

| Comparison | What exists instead | Refit scale if ever needed |
|---|---|---|
| `offset_logit_ar1` cell of the transform×latent grid | metrics.json + diagnostics.json | ~8 h/chain (pathological geometry — why it was never redone) |
| model_v2 EIV / rw_horizon (v1 vs v2) | metrics/diagnostics/calibration/predictions JSONs | 2 publication fits (4×5000) |
| genre pooling (#41/#85) | paired point/calibration metrics JSONs | screening was 2×2000; ELPD verdict already post-#63 via the publication confirmation |
| gbm_offset (#76/#86) | screening + confirmation metrics JSONs | same as above |
| beta_ceiling (#42) | screening_metrics.json | 1 diagnostic fit |
| n_exponent (#44) | posterior summary | posterior-concentration evidence; ELPD not the deciding axis |
| entity_overdispersion C1/C2 | per-domain JSONs under `outputs/experiments/` (not in `.audit/`) | the one **genuinely suspect** entry (row 16); IMDb + econ cheap fits |
| sigma_artist mixing bake-off | `outputs/experiments/sigma_artist_mixing.json` (not in `.audit/`) | mixing evidence, estimator-independent |

Two artifacts referenced by decisions live outside `.audit/` (rows 15, C1/C2) — from
0.7.0 on, sweep reports land in `.audit/` by construction (`select` A6), which closes
this class of gap.

## Bottom line

24 of 26 recorded verdicts are **verified-solid** or **recomputed** under the corrected
estimator. Two are **unverified (pre-#63 evidence)**: `heteroscedastic_entity_obs` (the
only genuinely suspect number — its IMDb LOO regression) and `errors_in_variables`
(self-flagged, almost certainly robust). Both options remain live candidates in
`panelcast select`, so the AOTY reproduction sweep re-adjudicates them from scratch.
