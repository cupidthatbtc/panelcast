# Glossary

What everything in this project is, what it does, and why we use it.

The [full reference](#full-reference) follows below. This section covers the
things you'd actually bring up when explaining the project.

---

## The Stack

**NumPyro** — Probabilistic programming library that defines and fits our Bayesian model. Like Stan or PyMC but runs on JAX, so it gets GPU acceleration for free. We write our model as a Python function with `numpyro.sample()` calls and NumPyro handles all the MCMC machinery underneath.

**JAX** — The numerical backend underneath NumPyro. "NumPy but with GPU support and automatic differentiation." We never call JAX to do ML directly — NumPyro does that. We use JAX for array math inside the model (`jnp.einsum`, `jnp.clip`) and its explicit-key random number generator.

**ArviZ** — Diagnostics and visualization library for Bayesian models. Sits between NumPyro (fitting) and our evaluation code. Gives us R-hat, ESS, summary tables, and the InferenceData container that stores everything about a fitted model.

**pandas / NumPy / sklearn** — Standard data stack. pandas for data wrangling, NumPy for metrics computation, sklearn for PCA (genre features) and GroupShuffleSplit (artist-disjoint evaluation). We don't use sklearn for the model itself.

---

## The Model

**Hierarchical Bayesian regression** — Not a neural network. Each artist gets their own quality level, but those levels are drawn from a shared population distribution. Artists with few albums get "shrunk" toward the population mean (partial pooling). This is the core advantage over fitting each artist separately.

**numpyro.sample()** — Declares a random variable. Every parameter (mu_artist, sigma_obs, beta, rho) is introduced with `sample("name", distribution)`. During fitting, NUTS explores the joint posterior over all these. During prediction, they become draws from the fitted posterior.

**dist.StudentT(df=4)** — Our likelihood distribution. Heavier tails than Normal, so extreme album scores (a 12/100 or 98/100) don't dominate the fit. df=4 is heavy enough to be robust but not so heavy that we ignore the data. When df >= 100 it's just Normal.

**dist.HalfNormal** — The standard weakly-informative prior for scale parameters (sigma_artist, sigma_obs, sigma_rw). Forces them positive; doesn't impose strong opinions about magnitude.

**LocScaleReparam(centered=0)** — Non-centered parameterization. Instead of sampling `artist ~ Normal(mu, sigma)` directly (funnel geometry that traps NUTS), we sample `z ~ Normal(0,1)` and compute `artist = mu + sigma * z`. Same math, much better sampling.

**Random walk artist effects** — An artist's quality changes over time: `effect_t = effect_{t-1} + Normal(0, sigma_rw)`. This models career trajectories — an artist who made great albums in 2015 might have declined by 2024. Controlled by `sigma_rw`: small values = stable careers, large = volatile.

**AR(1) / rho** — Autoregressive term: `rho * prev_score`. Captures album-to-album momentum. Positive rho means hot streaks tend to continue. Bounded to (-0.99, 0.99) via TruncatedNormal to keep it stationary.

**soft_clip(x, 0, 100)** — Differentiable function that keeps predictions in (0, 100). Uses double-softplus so NUTS gradients still flow. A hard `jnp.clip` would create gradient discontinuities that break HMC.

**compute_sigma_scaled()** — Heteroscedastic noise: albums with more reviews have lower observation noise (more reliable aggregate scores). `sigma_obs / n_reviews^exponent`. Single-review albums get a 2x penalty. exponent=0 falls back to constant noise.

**sigma-ref reparameterization** — Instead of sampling sigma_obs directly (which creates a multiplicative funnel with n_exponent), we sample sigma_ref at the median review count and derive sigma_obs deterministically. Breaks the funnel, fewer divergences.

---

## Fitting

**NUTS** — No-U-Turn Sampler. Hamiltonian Monte Carlo with automatic trajectory length. Our settings: `max_tree_depth=10`, `target_accept_prob=0.90`. We run 4 chains, 1000 warmup + 1000 samples each = 4000 total posterior draws. These are standard publication settings.

**Divergent transitions** — NUTS's way of saying "the posterior geometry is too curved here." A few are OK; many mean the model needs reparameterization. This is why we use non-centered parameterization and switched from Beta to logit-normal for n_exponent.

**R-hat** — Convergence diagnostic: do the 4 chains agree? R-hat < 1.01 = converged. R-hat > 1.05 = problem. Checked for every parameter.

**ESS (Effective Sample Size)** — 4000 raw samples with autocorrelation might only give ~800 independent ones. Target: > 400 per chain. Low ESS = noisy posterior summaries, unreliable credible intervals.

**InferenceData** — ArviZ's container for everything about a fitted model. Posterior samples, diagnostics, observed data, constant data — all stored as xarray Datasets, saved to disk as NetCDF (.nc). This is the artifact that flows from fitting to evaluation to reporting.

---

## Prediction

**Predictive(model, posterior_samples, batch_ndims=1)** — NumPyro's posterior predictive sampler. Runs the model forward with `y=None` to generate predictions. Each posterior draw produces a different prediction, giving us a full predictive distribution per observation.

**predict_new_artist()** — The hierarchical payoff. For artists NOT in training, we sample a new artist effect from `Normal(mu_artist, sigma_artist)` — the population distribution. New artists automatically get wider uncertainty. Returns full predictive draws (`y`), mean predictions (`mu`), and the sampled artist effect.

**Epistemic vs. aleatoric uncertainty** — The spread of `mu` across posterior samples = epistemic (what we don't know about parameters). The gap between `y` and `mu` = aleatoric (irreducible observation noise). Our PredictionResult separates these.

---

## Evaluation

**CRPS (Continuous Ranked Probability Score)** — The primary metric. Generalizes MAE to full predictive distributions. Rewards both accuracy AND calibration. Lower is better. If you gave it a point prediction, CRPS = MAE. This is why we use a Bayesian model — CRPS measures something a point estimator can't.

**Coverage** — "Does the 95% credible interval actually contain the truth 95% of the time?" We check at 50%, 80%, 95%. A model with 99% empirical coverage at 95% nominal is well-calibrated but too uncertain — its intervals are too wide.

**Interval Score** — Combines coverage and sharpness (interval width) into one number. Penalizes both wide intervals (uninformative) and missed observations (miscalibrated). The decomposition tells you which problem you have.

**Reliability diagram** — Plot of nominal quantile vs. empirical hit-rate. Perfect calibration = diagonal. Above diagonal = overconfident. Below = underconfident. The most direct visual test of calibration.

**within_artist_temporal_split()** — Primary evaluation: hold out each artist's last album(s) for test. Tests the actual use case: "given an artist's history, predict their next album."

**artist_disjoint_split()** — Secondary evaluation: no artist overlap between train and test. Tests cold-start: "can we predict for artists we've never seen?" Uses `predict_new_artist()` under the hood.

---

## Pipeline

**8 stages** — prepare_dataset -> create_splits -> build_features -> train_bayes -> evaluate -> predict_next -> publication -> sensitivity. Run all with `panelcast run`, or one at a time with `panelcast stage <name>`.

**FeaturePipeline** — Orchestrates feature blocks with fit/transform separation. PCA, vocabularies, and scalers are always learned from training data only. This prevents leakage.

**GenreBlock + PCA(30)** — Genres as comma-separated strings -> multi-hot encoding -> PCA to 30 components. Captures genre similarity without a 200-column sparse matrix.

**training_summary.json** — The bridge between stages. train_bayes saves scaler params, artist mapping, feature names. evaluate and predict_next load it to ensure identical preprocessing.

**NetCDF (.nc)** — How we save fitted models. ArviZ's InferenceData.to_netcdf() writes the entire posterior to one HDF5-backed file. Reload with `az.from_netcdf()` for analysis without refitting.

---

---

# Full Reference

Everything in the project, organized by layer. The entries above are the
highlights; this section is the complete inventory.

---

## Inference Engine

**NumPyro** — The probabilistic programming library that defines and fits our model. It's like Stan or PyMC but runs on JAX, so it gets GPU acceleration for free. We write our model as a Python function with `numpyro.sample()` calls and NumPyro handles all the MCMC machinery.

**JAX** — The numerical computing backend underneath NumPyro. Think of it as "NumPy but with GPU support and automatic differentiation." We never call JAX to do machine learning directly — NumPyro does that. We use JAX for array math (`jnp.einsum`, `jnp.clip`, `jnp.where`) and its random number generator.

**jax.numpy (jnp)** — JAX's drop-in replacement for NumPy. All the array operations inside the model function use `jnp` instead of `np` because JAX needs to trace through them for automatic differentiation. Outside the model (metrics, data wrangling), we use regular `np`.

**jax.random.key()** — JAX's random number generator. Unlike NumPy's global RNG state, JAX uses explicit PRNG keys that you split and pass around. Every `fit_model()` and `predict_new_artist()` call takes a seed that becomes a JAX key. This is what makes our runs reproducible.

**jax.nn.softplus()** — A smooth approximation to `max(0, x)`. We use it inside `soft_clip()` to keep predictions in the (0, 100) range without hard boundaries that would break NUTS's gradient computation.

**jax.devices()** — How we detect whether we're running on GPU or CPU. Called once before fitting to log the hardware for reproducibility.

---

## Model Definition

**numpyro.sample()** — Declares a random variable in the model. Every parameter (mu_artist, sigma_obs, beta, rho, etc.) is introduced with `numpyro.sample("name", distribution)`. During fitting, NUTS explores the joint posterior over all these sites. During prediction, they become draws from the posterior.

**numpyro.deterministic()** — Records a computed value without adding to the log-density. We use it for `sigma_obs` in sigma-ref mode: `sigma_obs = sigma_ref * n_ref^n_exponent`. It's just bookkeeping — it lets us track sigma_obs in the posterior without double-counting it in the likelihood.

**numpyro.plate()** — Declares independent observations or groups. `plate("obs", n_obs)` tells NumPyro that the n_obs likelihood terms are conditionally independent given the parameters. `plate("artists", n_artists)` does the same for per-artist effects. This enables vectorized computation instead of Python loops.

**dist.Normal(loc, scale)** — Gaussian distribution. Used for artist effects (partial pooling), fixed-effect coefficients (beta), and random walk innovations. The default choice when we have no reason for anything fancier.

**dist.HalfNormal(scale)** — Normal distribution truncated to positive values. Used for all variance/scale parameters: sigma_artist, sigma_obs, sigma_rw, sigma_ref. We need these to be positive, and HalfNormal is the standard weakly-informative choice.

**dist.StudentT(df, loc, scale)** — Our likelihood distribution. Heavier tails than Normal, controlled by `df` (degrees of freedom). We use df=4, which means extreme album scores (a 12/100 or a 98/100) don't dominate the fit the way they would with a Normal likelihood. When df >= 100, it's numerically identical to Normal.

**dist.TruncatedNormal(loc, scale, low, high)** — Normal distribution with hard bounds. Used for the AR(1) coefficient `rho`, bounded to (-0.99, 0.99) to ensure stationarity. Without truncation, rho could wander outside [-1, 1] and make the autoregressive process explode.

**dist.LogNormal(loc, sigma)** — Positive distribution with a long right tail. Used for `sigma_rw` (random walk innovation). Unlike HalfNormal, LogNormal can't pile up mass at zero, which prevents NUTS from getting stuck exploring near-zero innovation scales.

**dist.TransformedDistribution(Normal, SigmoidTransform)** — Logit-normal distribution. Used for the learned `n_exponent` (heteroscedastic noise scaling power). We sample in unbounded space, then push through sigmoid to get a value in (0, 1). This avoids the funnel geometry that the old Beta prior caused, which was producing divergences.

**dist.Beta(alpha, beta)** — Legacy prior for `n_exponent`. Beta(2, 4) has mode at 0.25 and mean at 0.33. We switched to logit-normal because Beta caused divergences — the posterior geometry near 0 and 1 created funnels that NUTS couldn't navigate efficiently.

---

## Model Structure

**make_score_model(score_type)** — Factory function that builds either `user_score_model` or `critic_score_model`. The only difference is the parameter name prefix ("user_" vs "critic_"). This lets us fit two independent models with distinct posteriors from a single model definition.

**soft_clip(x, low=0, high=100, sharpness=5)** — Differentiable function that squashes predictions into (0, 100). Uses a double-softplus trick: acts as identity for values well within bounds, smoothly saturates near edges. Sharpness=5 means it's effectively a hard clip for x in [1, 99] but NUTS gradients still flow cleanly.

**compute_sigma_scaled(sigma_obs, n_reviews, exponent)** — Per-observation noise scaling. The core idea: an album with 500 reviews has a more reliable aggregate score than one with 3 reviews, so we shrink its observation noise. Formula: `sigma_obs / n_reviews^exponent`. Single-review albums get a 2x penalty. When exponent=0, it's just constant noise (homoscedastic).

**PriorConfig** — Frozen dataclass holding all hyperparameters. Every prior's location and scale is configurable for sensitivity analysis. Defaults are weakly informative: centers at zero, moderate scales, designed for standardized features.

**LocScaleReparam(centered=0)** — Non-centered parameterization for the initial artist effects. Instead of sampling `artist ~ Normal(mu, sigma)` directly (which creates a funnel between mu, sigma, and artist), it samples `z ~ Normal(0, 1)` and computes `artist = mu + sigma * z`. Same math, much better geometry for NUTS.

**numpyro.handlers.reparam()** — The mechanism that applies LocScaleReparam to the model. Wraps the centered model function and rewrites the specified sample sites to use non-centered parameterization.

---

## MCMC Fitting

**NUTS (No-U-Turn Sampler)** — The MCMC algorithm that explores our posterior. It's Hamiltonian Monte Carlo with automatic trajectory length tuning. We configure it with `max_tree_depth=10` (limits computation per step) and `target_accept_prob=0.90` (controls step size adaptation — higher means more conservative but fewer divergences).

**MCMC(kernel, num_warmup, num_samples, num_chains, chain_method)** — NumPyro's MCMC runner. Orchestrates running NUTS across 4 chains. We use `chain_method="sequential"` (one chain at a time) because it's the most stable on GPU. "vectorized" is faster but uses more memory; "parallel" requires multiple GPUs.

**mcmc.run(rng_key, extra_fields=("diverging", "num_steps"), **model_args)** — Kicks off sampling. The `extra_fields` argument tells it to also record whether each sample was a divergent transition and how many leapfrog steps it took. Model args are the data arrays: artist_idx, X, y, etc.

**mcmc.get_samples(group_by_chain=True)** — Extracts posterior samples after fitting. With `group_by_chain=True`, returns shape `(num_chains, num_samples, *param_shape)` — needed for ArviZ diagnostics. Without it, flattens to `(num_chains * num_samples, *param_shape)` — used for Predictive.

**mcmc.get_extra_fields()["diverging"]** — Boolean array flagging divergent transitions. Divergences mean NUTS hit a region of posterior geometry it couldn't navigate (usually a funnel). We log the count but don't fail — the evaluate stage checks against thresholds.

**MCMCConfig** — Frozen dataclass for MCMC settings. Default: 1000 warmup + 1000 samples across 4 chains = 4000 total posterior draws. These are standard publication settings. The config is saved to JSON inside the model manifest for reproducibility.

**FitResult** — Container returned by `fit_model()`. Holds the MCMC object, ArviZ InferenceData, divergence count, runtime in seconds, and GPU info string. Everything downstream (predict, evaluate, save) takes a FitResult.

**fit_model(model, model_args, config)** — The main fitting entry point. Creates NUTS kernel, runs MCMC, converts results to ArviZ InferenceData, logs GPU info and divergences. Returns a FitResult.

---

## Prediction

**Predictive(model, posterior_samples, batch_ndims=1)** — NumPyro's posterior predictive sampler. Takes the model function and a dict of posterior samples, then runs the model forward with `y=None` to generate predictions instead of conditioning on observed data. `batch_ndims=1` tells it the samples are flattened across chains.

**generate_posterior_predictive(model, mcmc, model_args)** — Generates predictions on the training data. Used for posterior predictive checks: "do the model's predictions look like the data it was trained on?" Sets `y=None` internally so the model samples from the likelihood rather than conditioning.

**predict_out_of_sample(model, mcmc, new_model_args)** — Same as posterior predictive but for held-out data (val/test set). The artists must have been seen during training — their fitted artist effects are reused.

**predict_new_artist(posterior_samples, X_new, prev_score)** — Prediction for artists NOT in the training set. Samples a new artist effect from `Normal(mu_artist, sigma_artist)` — the population distribution. This is the hierarchical Bayesian payoff: new artists get predictions with appropriate extra uncertainty, automatically. Returns a dict with `y` (full predictive draws), `mu` (mean without noise), `artist_effect`, and `sigma_scaled`.

**PredictionResult** — Container with `y` (predicted scores with observation noise, shape `(n_samples, n_obs)`) and optionally `mu` (mean predictions without noise). The spread of `mu` across samples captures epistemic uncertainty (what we don't know about the parameters); the difference between `y` and `mu` captures aleatoric uncertainty (irreducible noise).

**jnp.einsum("sf,af->sa", beta, X_new)** — Einstein summation for the matrix multiply in prediction. `s` = samples, `f` = features, `a` = albums. Computes `beta @ X_new.T` across all posterior samples at once. More explicit about dimensions than `@` operator when shapes get tricky.

---

## Diagnostics & Convergence

**ArviZ (az)** — The diagnostics and visualization library for Bayesian models. Sits between NumPyro (fitting) and our evaluation code. Provides R-hat, ESS, summary tables, and the InferenceData container that everything else consumes.

**az.InferenceData** — The central data container. Holds `posterior` (parameter samples), `sample_stats` (divergences, tree depth), `observed_data` (y values), and `constant_data` (X, artist_idx, etc.) as xarray Datasets. Saved to disk as NetCDF. Everything after fitting works with this object.

**az.summary()** — Computes a table of posterior statistics for each parameter: mean, std, HDI (highest density interval), MCSE (Monte Carlo standard error), ESS (effective sample size), and R-hat. This is the "is my model healthy?" check.

**R-hat (az.rhat())** — Split-rank normalized Gelman-Rubin statistic. Measures whether the 4 chains agree with each other. R-hat < 1.01 means convergence. R-hat > 1.05 means the chains are exploring different regions — something is wrong. We check this for every parameter.

**ESS-bulk (az.ess())** — Effective sample size for the bulk of the posterior. 4000 raw samples might only give 800 effective samples if there's autocorrelation. Target: > 400 per chain (> 1600 total). Low ESS means your posterior summaries are noisy.

**ESS-tail** — Same as ESS-bulk but focused on the tails (extreme quantiles). Can be much lower than ESS-bulk if the sampler struggles in low-density regions. Important for credible intervals — you need good tail ESS for reliable 95% CIs.

**Divergent transitions** — NUTS's way of saying "the posterior geometry is too curved for my step size here." A few divergences might be OK; many indicate the model needs reparameterization (which is why we use non-centered parameterization and logit-normal priors).

**xr.Dataset / xr.DataArray** — xarray's labeled multi-dimensional arrays. ArviZ uses them internally for posterior storage: dimensions are `chain`, `draw`, and parameter-specific dims. We construct these manually in `fit_model()` to control dimension naming and avoid ArviZ's shape-guessing heuristics (which can OOM on large datasets).

---

## Evaluation Metrics

**CRPS (Continuous Ranked Probability Score)** — The primary evaluation metric. Generalizes MAE to full predictive distributions. Lower is better. Rewards both accuracy (predicting near the true value) AND calibration (appropriate uncertainty). If you gave it a deterministic point prediction, CRPS equals MAE. Computed using the Gini mean difference form for speed: `E|Y - y| - 0.5 * E|Y - Y'|`.

**compute_crps(y_true, y_samples)** — Takes true values and posterior predictive samples, returns a CRPSResult with `mean_crps` and per-observation `crps_values`. The per-observation values help identify which albums the model struggles with.

**compute_point_metrics(y_true, y_pred_mean)** — Traditional regression metrics on the posterior mean: MAE, RMSE, R-squared, Median AE, and mean bias. These don't capture uncertainty quality, but they're useful for comparing against non-Bayesian baselines and for audiences familiar with frequentist metrics.

**MAE (Mean Absolute Error)** — Average absolute prediction error. Same units as the target (album score points). More robust to outliers than RMSE. In this project, a MAE of 8 means we're off by 8 points on average on a 0-100 scale.

**RMSE (Root Mean Squared Error)** — Square root of average squared error. Penalizes large errors more than MAE. If RMSE is much larger than MAE, you have a few really bad predictions dragging it up.

**R-squared (R2)** — Proportion of variance explained. 1.0 is perfect, 0.0 is "just predict the mean." Can go negative if the model is actively worse than the mean. For album scores with high inherent variability, even R2 = 0.3 means the model is capturing meaningful signal.

**Median AE** — Median of absolute errors. Even more robust to outliers than MAE. If Median AE is much lower than MAE, the model is good for most albums but terrible for a few.

**Mean Bias** — Average of (prediction - truth). Positive = overprediction, negative = underprediction. Should be near zero for a well-calibrated model.

**posterior_mean(y_samples)** — Averages posterior predictive samples across the sample dimension to get a single point prediction per observation. Shape: `(n_samples, n_obs)` -> `(n_obs,)`. This is the standard Bayesian point estimate (the posterior predictive mean).

---

## Calibration

**Coverage** — "Does the 95% credible interval actually contain the true value 95% of the time?" We compute this at multiple levels: 50%, 80%, 95%. A well-calibrated model hits all three. Overcoverage (e.g., 99% empirical at 95% nominal) means intervals are too wide — the model is underconfident.

**compute_coverage(y_true, y_samples, prob=0.95)** — Computes credible intervals from posterior samples and checks what fraction of true values fall inside. Returns CoverageResult with nominal, empirical coverage, interval bounds, and mean interval width (sharpness). Supports both equal-tailed and HDI intervals.

**Equal-tailed interval** — The default credible interval type. For 95%: lower = 2.5th percentile, upper = 97.5th percentile. Simple and symmetric, but wastes width for skewed posteriors.

**HDI (Highest Density Interval)** — The narrowest interval containing prob% of the posterior mass. Better than equal-tailed for skewed distributions. Computed via a sliding-window scan on sorted samples: find the window of size `ceil(prob * n_samples)` with minimum width.

**Sharpness (interval_width)** — Mean width of credible intervals. Narrower is better IF coverage is maintained. A model that gives 95% coverage with 20-point-wide intervals is well-calibrated but not very useful. We want tight intervals that still cover.

**Interval Score** — A proper scoring rule that combines calibration and sharpness into one number. Formula: `IS = width + (2/alpha) * penalty_for_misses`. The penalty is asymmetric: missing the interval costs `2/alpha` times the distance to the nearest bound. From Gneiting & Raftery (2007). Lower is better.

**compute_interval_score(y_true, y_samples, prob=0.95)** — Returns IntervalScoreResult with the mean score decomposed into sharpness (interval width) and calibration penalty (missed observations). If the calibration penalty dominates, your intervals are too narrow.

**WIS (Weighted Interval Score)** — Approximates CRPS using multiple interval levels (50%, 80%, 95%) plus the median. Follows Bracher et al. (2021). Useful when you want CRPS-like evaluation but from interval summaries rather than full posterior draws. Formula weighs narrower intervals more.

**Reliability diagram** — Plot of nominal quantile level (x-axis) vs. empirical hit-rate (y-axis). A perfectly calibrated model follows the diagonal. Points above the diagonal = overconfident (intervals too narrow). Points below = underconfident (intervals too wide).

**compute_reliability_data(y_true, y_samples, n_bins=10)** — Evaluates calibration across quantile levels from 0.05 to 0.95. For each level p, computes the predictive quantile q_p per observation, then checks what fraction of true values fall below q_p. Returns predicted_probs and observed_freq arrays for plotting.

---

## Data Pipeline

**within_artist_temporal_split()** — The PRIMARY evaluation strategy. Holds out each artist's last N albums for test, second-to-last N for validation, rest for training. Tests the core use case: "given an artist's history, predict their next album." Artists with too few albums are excluded entirely.

**artist_disjoint_split() / GroupShuffleSplit** — The SECONDARY evaluation strategy. Ensures no artist appears in both train and test. Tests cold-start prediction: "how well can we predict for artists we've never seen?" Uses sklearn's GroupShuffleSplit with Artist as the group key.

**clean_albums()** — Standardizes raw CSV data: parses dates, encodes collaborations, normalizes column names. The entry point for the data pipeline.

**validate_raw_dataframe()** — Schema validation on the raw data. Checks required columns exist, types are correct, and values are in expected ranges. Catches data quality issues early.

**ExclusionRecord / FilterStats** — Data lineage tracking. Every time we drop rows (missing scores, too few reviews, etc.), we log exactly what was dropped and why. ExclusionRecord captures per-row detail; FilterStats captures aggregate counts.

---

## Features

**FeaturePipeline** — Orchestrates multiple feature blocks with proper fit/transform separation. Fits all blocks on training data, then transforms any split using the fitted state. This prevents information leakage: PCA components, vocabularies, and scalers are always learned from training data only.

**BaseFeatureBlock** — Abstract base class for feature blocks. Each block has a `name`, a `requires` list (dependencies on other blocks), and `fit()`/`transform()` methods. New features are added by subclassing this.

**GenreBlock / sklearn.decomposition.PCA** — Converts comma-separated genre strings into numeric features. Pipeline: parse genres -> filter rare genres (< 20 occurrences) -> multi-hot encode -> PCA to 30 components. PCA is fitted on training data only. The 30 components capture genre similarity without a 200-column sparse matrix.

**ArtistBlock** — Computes per-artist features: album count (`n_albums`), appearance rank (`artist_appearance`), and the previous album's score (`prev_score`) for the AR(1) term. Previous score is 0.0 for debut albums.

**CollaborationBlock** — Encodes whether an album is a collaboration: `is_collaboration` (binary), `num_artists` (count), `collab_type_ordinal` (solo/duo/group/supergroup). Collaborations are noisier because the "artist identity" is split.

**TemporalBlock** — Extracts `Release_Year` as a numeric feature, standardized. Captures secular trends in album scoring over time.

---

## I/O and Serialization

**NetCDF (.nc)** — The file format for saved models. ArviZ's `InferenceData.to_netcdf()` writes the entire posterior (samples, diagnostics, observed data, constant data) to a single HDF5-backed file. Can be reloaded with `az.from_netcdf()` for post-hoc analysis without refitting.

**save_model() / load_model()** — Thin wrappers around ArviZ's NetCDF I/O plus our ModelManifest. `save_model()` writes the .nc file and updates the manifest JSON. `load_model()` reads them back. The manifest captures everything needed for reproducibility: MCMC config, priors, data hash, git commit, GPU info.

**ModelManifest** — Frozen dataclass tracking a single model's provenance. Includes: model type, filename, MCMC config, prior config, SHA256 hash of training data, git commit hash, GPU used, runtime, divergence count. Serialized as JSON alongside the NetCDF file.

**ModelsManifest** — Tracks current models (user + critic) plus full history of all models ever fitted. Enables rollback and comparison across runs.

**training_summary.json** — Saved by `train_bayes.py` after fitting. Contains scaler parameters, artist mapping (name -> index), feature names, data hash, and MCMC diagnostics. Loaded by `evaluate.py` and `predict_next.py` to ensure they use the exact same preprocessing as training.

**sha256_path() / sha256_file()** — Hashes data files for reproducibility. If the data hash in the manifest doesn't match the current file, you know the data changed since the model was fitted.

---

## Pipeline Orchestration

**PipelineOrchestrator** — Runs the full pipeline end-to-end. Resolves stage dependencies (topological sort), manages a Rich progress bar, implements hash-based skip logic (don't re-run stages whose inputs haven't changed), and writes a RunManifest capturing everything.

**StageContext** — The shared context passed to every stage's `run()` function. Contains: run directory, seed, MCMC settings, convergence thresholds, feature toggles, calibration parameters. This is how configuration flows from CLI to stages without globals.

**PipelineStage** — Dataclass defining a single stage: name, dependencies, run function, input/output paths for skip detection. The pipeline has 8 stages in order: prepare_dataset -> create_splits -> build_features -> train_bayes -> evaluate -> predict_next -> publication -> sensitivity.

**get_execution_order()** — Topological sort of pipeline stages. Ensures train_bayes runs after build_features, evaluate runs after train_bayes, etc. Returns a flat list of stages in valid execution order.

**Rich progress bar** — The terminal progress display during pipeline runs. Shows a spinner, stage name, elapsed time, and completion status. Uses SpinnerColumn, TextColumn, BarColumn, and TimeElapsedColumn from the Rich library.

**RunManifest** — JSON record of a complete pipeline run: timestamp, git state, environment info (Python version, library versions, pixi.lock hash), stage-by-stage timings and outcomes. Written to `outputs/{run_id}/manifest.json`.

---

## CLI

**Typer** — The CLI framework. Builds the `panelcast` command with subcommands (`run`, `stage`). Handles argument parsing, help text, and type validation via Python type hints. We chose Typer over Click for less boilerplate and over argparse for subcommand ergonomics.

**panelcast run** — The primary CLI entry point. Runs all pipeline stages in order. Key flags: `--seed` (reproducibility), `--dry-run` (show what would run), `--verbose` (debug logging), `--num-chains`, `--num-samples`, `--num-warmup` (MCMC overrides).

**panelcast stage <name>** — Runs a single pipeline stage in isolation. Useful during development to re-run just evaluate or predict_next without re-fitting the model.

---

## Visualization

**Plotly (go.Figure, make_subplots)** — Interactive HTML chart library. We use it for trace plots, posterior distributions, forest plots, predictions, and reliability diagrams. Plotly figures support hover, zoom, and pan — useful for exploring posteriors interactively. Exported as standalone HTML files.

**Matplotlib** — Static chart library for publication-quality figures. Used when we need PDF vector output at 300 DPI for academic manuscripts. Same chart types as Plotly but without interactivity.

**create_trace_plot(samples, var_name)** — 60/40 split: left panel shows MCMC trace (parameter value vs. draw number per chain), right panel shows density histogram. If all 4 chain traces look like "hairy caterpillars" overlapping each other, the sampler converged. If they wander to different levels, it didn't.

**create_posterior_plot(samples, var_name)** — Density plot of posterior distribution with HDI shading. Shows where the parameter probably lives. A tight peak means the data is informative about that parameter; a wide spread means substantial uncertainty.

**create_forest_plot()** — One row per parameter, showing point estimate (posterior mean) and HDI bars. Lets you compare effect sizes across parameters at a glance. If a parameter's HDI contains zero, its effect isn't clearly distinguishable from nothing.

**create_predictions_plot()** — Scatter plot of predicted vs. observed album scores with credible bands. Points near the diagonal are accurate predictions. The band width shows uncertainty — wider bands for albums we're less sure about (fewer reviews, new artists).

**create_reliability_plot()** — Plots the reliability curve (nominal vs. empirical quantile hit-rate). The 45-degree diagonal is perfect calibration. Deviations show where the model is overconfident or underconfident. This is the most direct visual test of calibration quality.

**aoty_light / aoty_dark templates** — Custom Plotly templates registered in `theme.py`. Consistent colorblind-safe color palette across all charts. The light template is for documents; the dark template is for presentations.

---

## Reporting

**create_coefficient_table(idata)** — Extracts a publication-ready table from the posterior: parameter name, posterior mean, SD, HDI bounds, ESS, R-hat. Exported as both CSV (for computation) and LaTeX (for manuscripts).

**create_diagnostics_table(idata)** — Summarizes convergence: R-hat, ESS-bulk, ESS-tail, and pass/fail status per parameter. The "is my model trustworthy?" table.

**export_table(df, path, caption)** — Dual-format export: writes `{path}.csv` and `{path}.tex`. The LaTeX output includes a `\caption{}` and `\label{}` for direct inclusion in papers.

**generate_model_card()** — Markdown document describing the fitted model: structure, hyperparameters, training data, convergence diagnostics, performance metrics, known limitations. Following the model card framework for ML transparency.

---

## Configuration

**YAML (configs/base.yaml)** — The project configuration file. Controls MCMC settings, prior hyperparameters, feature toggles, evaluation thresholds, and output paths. Loaded by `config/loader.py` with environment variable expansion (so secrets stay out of version control).

**Pydantic (config/schema.py)** — Validates the YAML config against a typed schema. Catches typos and invalid values before the pipeline runs. Nested dataclass-like structure: AppConfig contains MCMCConfig, PriorConfig, FeatureConfig, etc.

---

## Logging & Utilities

**structlog** — Structured logging library. Emits JSON-formatted log lines with key=value context (stage name, runtime, divergence count). Easier to parse programmatically than `print()` debugging. Used throughout the pipeline for production-grade logging.

**Rich** — Terminal formatting library. Used for the pipeline progress bar and formatted error output. Makes the CLI output readable with colors, spinners, and progress tracking.

**set_seeds(seed)** — Sets JAX PRNG state, NumPy random state, and Python's random module to the same seed. Called once at pipeline start. Ensures reproducibility across all sources of randomness.

**pixi** — The environment manager (like conda but faster). Manages Python version, JAX/CUDA versions, and all dependencies. `pixi.lock` is hashed into the RunManifest to ensure the exact environment is recorded.

**pyproject.toml** — Python project metadata: package name, version, dependencies, entry points. The `panelcast` CLI is registered here as a console script pointing to `cli.py:app`.
