> **STATUS — software validation complete; domain-model research open**
>
> The package claim — feature-complete infrastructure for specifying, fitting,
> evaluating, and auditing hierarchical panel models — is validated end-to-end
> on real data. The items below track the *AOTY domain model*, which is
> research in progress, not a package-completeness prerequisite.
>
> - **Convergence is now demonstrated on real data.** A representative subset of
>   the AOTY corpus (~800 artists / 5,182 albums, user-score skewness −2.08)
>   **passes** the convergence gate at the publication configuration (4×5000,
>   warmup 5,000, target_accept 0.90) with the 0.13.0 entity-obs default:
>   R-hat 1.00, bulk ESS 1,119, 0 divergences. See *Real-data subset
>   re-baseline* below.
> - **The likelihood is still misspecified, less than before.** On the same real
>   data the transform relieved the mean/sd/q50 pins and the 0.13.0 entity-obs
>   default cleared q10 and q90; only skewness and max stay pinned — an open
>   modeling limitation, not a convergence problem.
> - **Scale caveat.** Those numbers are from the ~5,182-album validated SUBSET,
>   **not** the full eligible corpus (~62k albums with ≥10 user ratings).
>   Full-corpus publication-scale validation is **still pending** — it needs
>   more GPU than is available locally (>24 GB even with
>   `--exclude-rw-raw-from-collection`), so it is tracked separately as **#15**
>   (cloud A100/H100 80 GB). The *Hyperparameters* and *Evaluation Results*
>   sections below now describe the current published fit — the 0.13.0 entity-obs
>   re-baseline (`.audit/baseline_metrics.json`); the *Model Architecture* notes
>   are general. Canonical headline numbers:
>   [`.audit/release_results.json`](.audit/release_results.json).

# Model Card: panelcast — Hierarchical Bayesian Score Prediction

> panelcast is a general, domain-agnostic Bayesian tool for predicting bounded
> scores over entity histories. This card documents a model trained on the
> flagship **example domain, Album of the Year (AOTY)** — artists releasing
> albums scored 0–100. Each pipeline run generates its own card via
> `reporting/model_card.py` under `outputs/<run_id>/reports/`; this repo-root
> card is curated by hand. The snapshot below is from a validation-scale run
> and has been corrected where flagged.

## Model Details

- **Model type:** Bayesian Hierarchical Regression with Time-Varying Effects
- **Version:** 0.13.1
- **Authors:** panelcast project
- **Created:** 2026-06-11
- **Last updated:** 2026-07-22

## Intended Use

Intended use is illustrated for the AOTY example domain; analogous
considerations apply to any domain the pipeline is retargeted to. For AOTY this
model is intended for:

- Academic research on music industry trends and career trajectories
- Personal exploration of album score patterns and artist development
- Understanding factors that influence critical and user reception
- Educational demonstration of Bayesian hierarchical modeling
- Reproducibility research in music information retrieval

### Out-of-Scope Use

For the AOTY example domain, this model should NOT be used for:

- Commercial artist evaluation or signing decisions
- Real-time prediction systems in production environments
- Automated content moderation or recommendation without human review
- High-stakes decisions affecting artists' careers or livelihoods
- Marketing claims about album quality or artist potential

## Training Data

- **Dataset:** Album of the Year (AOTY)
- **Scale — three distinct tiers, often conflated:**
  - *Raw corpus:* ~130k scraped rows (`all_albums_full.csv`), including albums
    without user scores.
  - *Eligible corpus:* ~62k albums with a user score and ≥10 user ratings
    across ~11.6k artists — the population the subset was sampled from and the
    full-corpus target of #15.
  - *Validated subset (the published fit):* ~800 whole artists / 5,182 albums
    (`scripts/make_aoty_subset.py`); the within-entity temporal test evaluates
    653 held-out next albums.
- **Description:** Music album metadata and scores from Album of the Year, including artist information, release dates, genres, and both critic and user scores.
- **Preprocessing:** Leak-safe within-artist temporal splitting with artist-disjoint secondary checks, minimum-ratings filtering, features standardized to zero mean and unit variance.

## Model Architecture

Bayesian hierarchical regression with four key components:

1. **Hierarchical artist effects**: Partial pooling across artists for robust estimation of artist quality. Non-centered parameterization via LocScaleReparam avoids funnel geometry.

2. **Time-varying slopes**: Artist quality modeled as a random walk, allowing career trajectories to evolve over time.

3. **AR(1) structure**: Album-to-album dependencies captured via autoregressive term, modeling momentum effects where consecutive albums tend to have correlated scores.

4. **Heteroscedastic observation noise** (sigma_ref parameterization): Albums with more reviews have lower observation noise. The model samples sigma_ref (noise at the median review count n_ref) and derives per-observation noise as: sigma_obs = sigma_ref * n_ref^n_exponent, then sigma_i = sigma_obs / n_reviews_i^n_exponent. This reparameterization breaks the multiplicative funnel between sigma_obs and n_exponent that causes divergent transitions in MCMC sampling.

Mathematical form:
- y_ij ~ StudentT(df=4, mu_ij, sigma_i)  (family configurable; see below)
- mu_ij = artist_effect_jt + X_ij @ beta + rho * (prev_score_ij - ar_center)
- artist_effect_jt evolves via random walk from initial effect
- sigma_i = sigma_obs / n_reviews_i^n_exponent (heteroscedastic mode)

Configurable variants (defaults shown):
- **Latent process** (`--latent-process`, default `rw`): the artist-effect
  trajectory is a non-centered random walk, or `ar1` for stationary deviations
  with persistence phi (nests the random walk at phi = 1).
- **AR centering** (`--ar-center`, default `global`): the previous score is
  centered by the train-mean before the AR(1) term, so debut AR terms are zero
  and rho decouples from the artist-effect level (`none` recovers the legacy
  uncentered form).
- **Per-entity overdispersion** (`heteroscedastic_entity_obs`, default on since
  0.13.0, #238): a multiplicative log-normal noise factor per entity, inflating
  sigma_i for entities with noisier histories. Domains whose bake-offs rejected
  it pin it off in their run configs.

### Prior Distributions

Prior distributions are weakly informative, chosen to regularize inference while allowing the data to dominate:

- **mu_artist** ~ Normal(0, 1): Population mean of artist effects (deviations from feature-based predictions). Scale of 1.0 permits the population center to shift by ~1.0 SD on the standardized score scale.
- **sigma_artist** ~ HalfNormal(0.5): Scale of 0.5 encourages moderate partial pooling. Implies most artist effects within +/-1.0, consistent with observed between-artist spread.
- **sigma_rw** ~ HalfNormal(0.1): Scale of 0.1 produces smooth career trajectories where album-to-album quality changes are small relative to overall artist variation.
- **rho** ~ TruncatedNormal(0.0, 0.3, -0.99, 0.99): Centered at 0.0 with scale 0.3, allowing moderate autoregressive momentum without strong prior commitment to direction.
- **beta** ~ Normal(0.0, 1.0): Scale of 1.0 is weakly informative for standardized features, allowing data to determine effect sizes.
- **sigma_obs** ~ HalfNormal(1.0): Scale of 1.0 allows data to determine observation-level noise.


**Prior Predictive Check**: Prior predictive simulation (n_samples=500) shows 99.9% of prior-implied predictions fall within [0.0, 100.0]. Summary: mean=70.2, sd=6.6, range=[-0.3, 115.3].

### Hyperparameters

> The published publication-configuration fit (seed 42): 4 chains × 5,000 draws,
> warmup 5,000, target_accept 0.90 — the preset amended for the 0.13.0 entity-obs
> re-baseline. Values are the resolved priors and data dimensions of that fit
> (`training_summary.json`).

| Parameter | Value |
|-----------|-------|
| mu_artist_loc | 0.843 |
| mu_artist_scale | 1.0 |
| sigma_artist_scale | 0.5 |
| sigma_rw_scale | 0.1 |
| rho_loc | 0.0 |
| rho_scale | 0.3 |
| beta_loc | 0.0 |
| beta_scale | 1.0 |
| sigma_obs_scale | 0.5 |
| sigma_ref_scale | 0.5 |
| n_exponent_default | 0.0 |
| n_features | 33 |
| n_artists | 653 |
| max_albums | 50 |
| num_chains | 4 |
| num_warmup | 5000 |
| num_samples | 5000 |
| chain_method | sequential |
| target_accept_prob | 0.9 |
| max_tree_depth | 10 |

## Evaluation Results

### Real-data subset re-baseline (2026-07-12, entity-obs default)

The published baseline is the seed-42 confirmation-arm fit adopted as the 0.13.0
re-baseline: the new `heteroscedastic_entity_obs` default (#238) at the amended
publication configuration — 4 chains × 5,000, **warmup 5,000, target_accept 0.90**.
The preset changed because with the per-entity overdispersion factor on, the old
target_accept 0.95 failed the convergence gate (sigma_artist R-hat ≥ 1.01, bulk
ESS ~650–690) at both warmup 3,000 and 5,000, while 0.90/5,000 passed cleanly on
all three confirmation seeds. The fit was produced on 0.12.x code with the flag
set through config; the likelihood-registry parity lock makes it bit-identical to
the flipped 0.13.0 default. Data: a representative subset of the full AOTY corpus
— ~800 whole artists sampled with their full discographies
(`scripts/make_aoty_subset.py`), albums with ≥10 user ratings; 653 of the ~800
artists are multi-album and contribute one held-out next album each (the 653
evaluated test rows), observed user-score skewness −2.08 — fit on GPU (RTX 5090),
Student-t likelihood on the transformed `offset_logit` scale.

- **Convergence gate: PASS** — R-hat (max) 1.00, bulk ESS 1,119 (≥ 400), 0
  divergences.
- **Predictive** (within-entity temporal test, n = 653): MAE 5.28, RMSE 7.67,
  R² 0.498, CRPS 3.81; held-out ELPD (test lppd) −2133.7 (SE 23.7),
  −3.27/obs. Against the 0.6.0-default incumbent (entity-obs off): paired
  held-out ELPD **+29.8 ± 7.0 (z +4.25)** — a distributional + calibration win,
  point accuracy a wash (MAE 5.30 → 5.28, RMSE 7.65 → 7.67, R² 0.501 → 0.498).
  Three-seed confirmed (42/43/44); evidence in `.audit/select_entityobs_confirm/`.
- **Calibration:** 80% coverage 0.830 (width 17.0), 95% coverage 0.968 (width
  31.0). The 80% interval is now *closer to nominal* than the incumbent's 0.853;
  it misses the ±0.03 tolerance by 1.5e-5 (one album out of 653, below the
  subset's 1/653 grid), where the incumbent misses by ~1500× that margin — the
  #237 coverage non-inferiority axis on which the gate was promoted.
- **PPC — two of four structural pins resolved:** mean (p = 0.72), sd (0.37),
  q50 (0.18), min (0.83), **q10 (0.028)** and **q90 (0.97)** are now interior;
  only skewness (1.00) and max (1.00) stay pinned. Clearing q10 and q90 is the
  first movement on the bounded-skew tail misfit that six likelihood families
  never touched. See Limitations.
- **Baselines on the same real splits:** the model still leads the within-entity
  table on MAE and CRPS and roughly ties ridge on RMSE and R² (0.498 each); its
  80% intervals are now closer to nominal. Cold-start (artist-disjoint) leads at
  R² 0.117 (MAE 6.82), though it stays the model's weakest split in absolute
  terms. Regenerated table: [`docs/BASELINES.md`](docs/BASELINES.md).
- **Likelihood decision (historical):** the bounded Beta candidate was tested
  on this real subset and did **not** win — more pinned PPC statistics and
  worse mixing (bulk ESS 304, 1 divergence). Student-t remains the default.
  See `docs/LIKELIHOOD_CANDIDATES.md`.

This demonstrates the mechanism on **real, strongly left-skewed data**; it is a
~5k-album subset, not the final full-corpus result (#15). The raw metrics
snapshot is committed at `.audit/baseline_metrics.json`.

Two convergent negative results from the identity era — the learned
heteroscedastic exponent collapsing to zero and the entity-disjoint R² ≈ 0 —
are one finding: the predictive mass lives in the per-entity intercept, with
measurement-noise modeling and covariates both ~null. See
[`docs/decisions/what_carries_the_signal.md`](docs/decisions/what_carries_the_signal.md). The
`offset_logit` re-baseline softened the second half, and the 0.6.0 promotions
revise the first: with the `gbm_offset` covariate in the mean the model now
extracts the exogenous point signal too (cold-start R² 0.117, within-entity
R² 0.498) — the intercept still carries the largest single share, but
"covariates ~null" no longer describes the shipped default.

### Convergence Diagnostics (publication-scale snapshot, 2026-07-12)

Convergence status: **PASS**

- R-hat (max): 1.00 (threshold: < 1.01)
- ESS bulk (min): 1,119 (threshold: ≥ 400)
- Divergent transitions: 0

### Calibration

Credible interval coverage (within-entity temporal test):
- 80% CI: 83.0% empirical coverage, mean width = 16.99
- 95% CI: 96.8% empirical coverage, mean width = 31.03

**Posterior Predictive Checks:**
- mean: T(y_obs)=68.79, p=0.721
- sd: T(y_obs)=10.83, p=0.365
- skewness: T(y_obs)=-2.08, p=0.999
- min: T(y_obs)=3.00, p=0.828
- max: T(y_obs)=88.00, p=1.000
- q10: T(y_obs)=58.00, p=0.028
- q50: T(y_obs)=71.00, p=0.178
- q90: T(y_obs)=79.00, p=0.974

### Predictive Performance

Point prediction metrics (within-entity temporal test, n = 653):

- MAE: 5.28
- RMSE: 7.67
- R-squared: 0.498
- **Held-out ELPD (test lppd):** −2133.7 (SE: 23.7), −3.27 per observation

## Limitations

- **Convergence (compute-bounded, geometry fixed).** The historical sigma_artist ESS deficit was traced to a sampling-geometry confound: the uncentered AR(1) term absorbed the score level, ridge-coupling rho and mu_artist (corr -0.997). AR centering with a level-located mu_artist prior removed it (corr +0.016, debut AR terms exactly zero). Remaining R-hat/ESS shortfalls at cheap validation settings (2 chains x 500) were compute-bounded; this is now **confirmed on real data**: at the pre-0.13.0 publication configuration (4 chains x 5000, warmup 3000, target_accept 0.95, with the rw_raw collection exclusion required for 24 GB GPUs) the ~5k-album subset passed the gate under both transforms — R-hat 1.00, 0 divergences, bulk ESS 3,134 (identity) / 2,333 (offset_logit) — an earlier geometry confirmation that pre-dates the entity-obs default. Under the 0.13.0 preset (warmup 5000, target_accept 0.90; amended because 0.95 fails the gate with the entity-obs default) the published fit passes at R-hat 1.00 / bulk ESS 1,119 / 0 divergences (see Evaluation Results). The full-corpus run remains future work.
- **Symmetric likelihood vs. left-skewed target.** The Student-t likelihood is symmetric, but observed user-score distribution has skewness ~= -1.79 (long left tail of poorly-received albums). This is a structural mismatch, not a fitting issue. PPC p-values pinned at 0.000/1.000 for sd, skewness, q50, q90, and max were the expected signature of this mismatch at the identity-transform baseline; on the published fit only skewness and max remain pinned. The lightest candidate fix — an `offset_logit` target transform — is now the **default** (adopted in 0.5.0 on the corrected #63 estimator: paired held-out elpd +22.2 ± 4.5 over identity, seed-stable, plus R² 0.428 vs 0.417 and PIT 0.049 vs 0.056; [`.audit/transform_latent_bakeoff/`](.audit/transform_latent_bakeoff/)), yet it does not move the `skewness`/`max`/`q90` PPC pins — it only trades the `q50` pin for a new `q10` pin — so it leaves this structural mismatch unaddressed. **Six likelihood families have since been implemented and tried against the mismatch, selectable via `--likelihood-family`:** `beta`, `skew_studentt`, `skew_normal`, `split_normal`, `beta_binomial`, and a two-component `mixture` — plus an integer-aware dequantization toggle (`--discretize-observation`). **None moves the `skewness`/`max` PPC pins toward the interior**; each trades worse convergence or point accuracy for the same or sharper pins, confirmed on real AOTY data across five of them (the synthetic edge that favored `beta` did not survive real, strongly left-skewed scores). Dequantization does relieve the integer-heaping `q50` pin specifically (p 0.009 → 0.082), but `skewness`/`max`/`q90` are the bounded-skew misspecification itself, not an integer-grid artifact, and more posterior samples sharpen them rather than relaxing them. The mismatch is therefore **confirmed structural and unresolved**, so Student-t remains the publication default (full evidence in [`docs/LIKELIHOOD_CANDIDATES.md`](docs/LIKELIHOOD_CANDIDATES.md)). The symmetric likelihood's point accuracy and 95% interval calibration are unaffected. **First movement on these pins (now the AOTY default):** the gated `heteroscedastic_entity_obs` factor — surfaced by the `panelcast select` rung ladder, three-seed confirmed on the ~5k-album subset (held-out ELPD +29.8 ± 7.0, z +4.25; point accuracy a wash) — is the first change to move the bounded-skew pins, clearing **q10 and q90** while leaving `skewness`/`max`. **Promoted to the AOTY default in 0.13.0** (#238): #237 amended the coverage gate to clear on tolerance *or* non-inferiority to the reference, and the arm's 1.53e-5 80%-coverage miss is ~1500× smaller than the incumbent's, so it re-scores promotable on all three seeds; the full-corpus run (#15) still resolves the subset grid. Evidence in [`.audit/select_entityobs_confirm/`](.audit/select_entityobs_confirm/) and [`docs/decisions/entity_overdispersion.md`](docs/decisions/entity_overdispersion.md).
- **Soft-clip at [0, 100] under `--target-transform identity`.** With the 0.5.0 `offset_logit` default the bounds hold by construction and no clip is applied. Under the `identity` transform (the former default, still selectable) the bounded target meets a symmetric likelihood and soft_clip compresses both tails simultaneously.
- **Errors-in-variables in the AR(1) predictor (addressable; opt-in `errors_in_variables`).** The album-to-album term regresses on the *observed* previous user score as if it were noise-free (`ar_term = rho * (prev_score - ar_center)`), yet that same quantity is modeled as review-count-noisy when it is the response (the heteroscedastic `sigma_obs / n^exponent`). Conditioning on a noisy regressor attenuates `rho` toward zero, worst for sparse-review entities whose lagged score is least certain. The model-v2 fix (issue #30) ships as a gated option, default **off** so the published numbers stay byte-identical: rather than a second latent-state AR (which would duplicate the existing random-walk artist trajectory and reintroduce the `sigma_rw ↔ rho ↔ level` ridge), it de-noises the *regressor* with a fixed, data-derived measurement-error latent — `prev_latent = prev_score + (global_std/√prev_n_reviews)·z`, `z ~ Normal(0, 1)`, debuts pinned to zero — and forms `ar_term = rho·(prev_latent - ar_center)`. When a non-identity target transform is active, `global_std` is measured on the model scale, so the per-observation measurement-error scale is an approximation under the transform's nonlinearity — acceptable for a fixed regularization prior, but a consideration before default adoption. Synthetic-recovery tests confirm `rho` de-attenuates with the gate on; the quantitative subset bake-off (v1 vs v2 at the publication configuration, both clearing the gate at R-hat 1.00 / bulk ESS > 3,100 / 0 divergences) is now complete and the gate is **immaterial on this subset** — LOO moves +0.4 against an SE of ~29.6 and every point/calibration metric is unchanged, consistent with the `n_exponent ≈ 0` result. It stays default-off, parity-safe; see [`.audit/model_v2_bakeoff/comparison.md`](.audit/model_v2_bakeoff/comparison.md).
- **Trained on English-language reviews; may not generalize to other markets.**
- **Domain portability is structural, not predictive.** The apparatus retargets to a new domain (e.g. the bundled aerospace example) with one YAML and zero source changes, proven end-to-end by `tests/e2e/test_domain_portability.py` — but that test asserts the pipeline *runs* and the output *structure*, not predictive accuracy. Accuracy on any non-AOTY domain is untested by construction; the numbers in this card are AOTY-only.
- Dynamic artist trajectories are learned only when an artist has at least 2 training albums (configurable via `min_train_albums` / `--min-train-albums`).
- Less reliable for genre-crossing artists due to sparse data.
- Historical biases in music criticism may be reflected in predictions.
- Does not account for album-specific factors (production, label influence).
- Assumes gradual career evolution; sudden style changes poorly predicted.
- **Long-horizon predictive variance is understated (addressable; opt-in `propagate_rw_horizon`).** The latent artist effect is indexed by album sequence clipped to the longest training trajectory (`seq_idx = clip(album_seq - 1, 0, max_seq - 1)`), and prediction appends no further random-walk innovations beyond `max_seq`. For an album `h` steps past that horizon the forecast reuses the final latent step and omits the `(h - max_seq)·sigma_rw²` of accumulated random-walk variance a true multi-step-ahead forecast would carry, so deep-extrapolation intervals are too narrow. The model-v2 fix (issue #30) ships as a gated, default-**off** prediction-path option: with it on, the evaluate/predict stages drop the clamp and pass `max_seq = album_seq.max()`, so the re-sampled `rw_raw` trajectory (always excluded from the saved posterior and re-drawn from its prior at prediction) accumulates the full `h - 1` innovations and deep-extrapolation intervals widen by ~`sqrt(h - max_seq)·sigma_rw`. Training and within-horizon draws are unchanged; no `model.py` change. The flagship one-step-ahead use (next album) is unaffected either way; `--strict` still blocks unflagged horizon extrapolation. The v1-vs-v2 subset bake-off leaves this gate's metrics unchanged too, but the within-horizon holdout does not exercise deep extrapolation, so its value is only measurable at the full corpus / longer horizons (#15).
- Score predictions are probabilistic and should not be treated as ground truth.

## Ethical Considerations

- Predictions should not gatekeep artists or influence career decisions
- Aggregated scores may not reflect artistic merit or listener preferences
- Care should be taken when interpreting genre-based effects
- Model may perpetuate historical biases present in music criticism
- Predictions are for research and exploration, not commercial evaluation
- Artists and labels should not be ranked solely based on predicted scores

## How to Use

### Loading the Model

```python
from panelcast.models.bayes.io import load_manifest, load_model
from panelcast.paths import resolve_latest

# Models are run-scoped (0.6.0+): resolve the latest successful run via
# outputs/latest.json, then follow that run's models/manifest.json
models_dir = resolve_latest() / "models"
manifest = load_manifest(models_dir)
model_name = manifest.current["user_score"]
idata = load_model(models_dir / model_name)
```

### Making Predictions

```python
from panelcast.models.bayes.predict import (
    extract_posterior_samples,
    predict_new_entity,
)
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.training_summary import (
    ar_center_on_model_scale,
    load_training_summary,
)
import jax.numpy as jnp

# Build posterior sample dict from InferenceData
posterior_samples = extract_posterior_samples(idata)

# The training summary records the transform and AR centering the model
# was fit with (offset_logit is the 0.5.0+ default)
summary = load_training_summary(models_dir / "training_summary.json").to_json_dict()
target_transform = summary.get("target_transform") or "identity"
logit_offset = float(summary.get("logit_offset") or 0.5)
transform = get_transform(target_transform, offset=logit_offset)

# Predict one new album using standardized feature vector
n_features = int(posterior_samples["user_beta"].shape[-1])
X_new = jnp.zeros((1, n_features), dtype=jnp.float32)

# prev_score goes in on the model scale (forward-transformed), matching the
# model-scale ar_center; the returned draws come back on the [0, 100] score scale
pred = predict_new_entity(
    posterior_samples=posterior_samples,
    X_new=X_new,
    prev_score=transform.forward(jnp.array([72.5], dtype=jnp.float32)),
    n_reviews_new=jnp.array([300.0], dtype=jnp.float32),
    prefix="user_",
    target_transform=target_transform,
    logit_offset=logit_offset,
    ar_center=ar_center_on_model_scale(summary),
)
```

### Interpreting Results

```python
import numpy as np

# Extract prediction statistics from posterior predictive draws
y_samples = np.asarray(pred['y']).ravel()
pred_mean = float(np.mean(y_samples))
pred_std = float(np.std(y_samples))
ci_95 = np.percentile(y_samples, [2.5, 97.5])

print(f"Predicted score: {pred_mean:.1f} +/- {pred_std:.1f}")
print(f"95% CI: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]")
```
