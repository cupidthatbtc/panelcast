# Where the GBM's point-accuracy edge comes from

**Question.** The baseline GBM beats the Bayesian model on within-entity point
accuracy (R² 0.486 vs 0.417). Which feature blocks carry that edge — i.e. what
signal does the structured model leave unconsumed?

**Method.** The exact leaderboard estimator
(`HistGradientBoostingRegressor(random_state=0)`, panels via
`load_panel_pair`, same NaN→0 fill), refit with each feature block dropped and
isolated, on both evaluation splits of the ~5k AOTY subset (within-entity
train 4235 / test 653; entity-disjoint train 3614 / test 799). Blocks:
temporal (6 cols), album_type (4), user_history (`user_prior_*`,
`user_trajectory`, `is_debut`; 5), critic_history (`critic_prior_*`,
`critic_trajectory`; 4), genre (10 PCA), collaboration (3). Full-feature fit
reproduces the published row exactly (MAE 5.413 / R² 0.4864). Permutation
importance: n_repeats=5 on the test panel, R² scoring, summed within block.

## Within-entity temporal (test R²)

| config | R² | MAE |
|---|---:|---:|
| full | 0.486 | 5.41 |
| **only user_history** | **0.412** | 5.79 |
| only user+critic history | 0.428 | 5.64 |
| drop user_history | 0.336 | 6.13 |
| drop critic_history | 0.456 | 5.57 |
| drop temporal | 0.447 | 5.58 |
| drop album_type | 0.453 | 5.54 |
| drop genre | **0.501** | 5.47 |
| drop collaboration | 0.474 | 5.47 |
| drop all history | 0.177 | 7.06 |

Permutation importance by block: user_history 0.60, temporal 0.038,
album_type 0.034, critic_history 0.023, genre 0.012, collaboration 0.001.

## Reading

1. **The model already extracts the user-history signal in full.** A GBM given
   *only* the user-history block scores R² 0.412 — statistically the Bayesian
   model's 0.417. The hierarchical entity effects are not leaving
   entity-history signal on the table; flexible feature extraction adds
   nothing over the structured treatment of the same information.
2. **The gap is NOT missing predictors — the model already regresses on every
   column.** The mean function carries an always-on linear term
   (`mu_raw = obs_artist_effect + X @ beta + ar_term`, `model.py`; `X` = all 32
   z-scored feature columns, `beta ~ Normal(0, 1)`). The GBM's edge
   (0.486 − 0.412 ≈ +0.07 R²) decomposes as temporal ≈ +0.04, album_type
   ≈ +0.03, critic_history ≈ +0.03 — signal the *linear term fails to convert*,
   not signal it never sees. Two mechanisms, separable by metric: on MAE the
   model (5.64) ties ridge (5.62) — the linearly-available signal is captured,
   and the R² shortfall vs ridge (0.417 / 0.455) is largely the Student-t
   likelihood trading squared-error for tail-robustness; the residual GBM edge
   (MAE 5.41) is nonlinearity/interactions no linear form reaches.
3. **Genre PCA features are dead weight for point accuracy here.** Dropping
   them *improves* the GBM (0.501 vs 0.486) once user history is present —
   consistent with the small genre-only headroom in
   [`../genre_pooling/covariates_only_r2.md`](../genre_pooling/covariates_only_r2.md).

## Entity-disjoint (cold-start)

Full-feature R² 0.037; every single-block config sits in [−0.08, +0.07] and
block deltas flip sign (dropping critic_history *helps*, +0.05), i.e. noise at
n = 799 with no entity history to anchor on. Cold-start point signal is weak
for every method; the within-entity split is where the actionable gap lives.

## Implication for 0.6.0

Adding covariates is off the table — they are already in the mean function.
The actionable questions (tracked in #76) are, in order: (1) inspect the
fitted `{prefix}beta` posterior from the 0.5.0 re-baseline — the cold-start
anomaly (covariates-only headroom ≈ 0.083 vs the model's ≈ 0.003 on the same
features) suggests the coefficients are diluted, plausibly by collinearity
between `user_prior_*` columns and the entity effects that encode the same
information; (2) if diluted, test dropping the redundant history columns from
`X` so `beta` can work on the genuinely exogenous blocks; (3) treat the
remaining GBM edge (MAE 5.41 vs 5.64) as a nonlinearity question, and the R²
gap to ridge as partly the Student-t robustness trade, not a defect.
