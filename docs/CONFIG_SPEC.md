# Config Specification

Dataset
- dataset.raw_csv: string path
- dataset.encoding: string (default utf-8-sig)
- dataset.min_ratings: int (default 10)

Splits
- splits.strategy: string (within_entity_temporal | entity_disjoint)
- splits.group_col: string (default Artist)
- splits.seed: int (default 42)
- splits.train_frac: float (default 0.65)
- splits.val_frac: float (default 0.15)
- splits.test_frac: float (default 0.20)
- splits.k_folds: int (default 5)
- splits.time_holdout: bool

Imputation
- imputation.strategy: string (hierarchical)
- imputation.hierarchy: [artist, genre, decade, global]
- imputation.min_counts: mapping (artist, genre, decade)

Features
- features.include_genre: bool
- features.include_artist: bool
- features.include_temporal: bool
- features.include_album_type: bool
- features.pca.core: bool
- features.pca.genres_components: int
- features.pca.descriptors_components: int
- features.blocks: list of feature block specs (name + params)

Model
- model.sampler: numpyro
- model.num_warmup: int
- model.num_samples: int
- model.num_chains: int
- model.target_accept_prob: float
- model.max_tree_depth: int
- model.target_scale: string (standardized | raw)
- model.likelihood_df: float (default 4.0; Student-t degrees of freedom, >=100 behaves as Normal)
- model.likelihood_family: string (studentt | normal | skew_studentt | skew_normal | split_normal | beta; default studentt)
- model.discretize_observation: bool (default false; interval-censor integer observations — location-scale families only, rejected for beta/skew_studentt)
- model.priors.intercept_sd: float
- model.priors.slope_sd: float
- model.priors.group_sd: float
- model.priors.sigma_sd: float
- model.dynamic.enabled: bool
- model.dynamic.min_albums: int

Diagnostics
- diagnostics.rhat_max: float (<= 1.01)
- diagnostics.ess_min: int
- diagnostics.max_divergences: int

Evaluation
- evaluation.metrics: list (r2, rmse, mae, crps)
- evaluation.calibration_intervals: list (e.g., 0.8, 0.95)
- evaluation.coverage_tolerance: float
- evaluation.model_comparison: list (waic, loo)

Sensitivity
- sensitivity.min_ratings: list
- sensitivity.dynamic_min_albums: list
- sensitivity.prior_slope_sd: list
- sensitivity.feature_ablations: list
- sensitivity.splits: list

Outputs
- outputs.run_dir: string
- outputs.save_traces: bool
- outputs.processed_path: string
- outputs.features_dir: string
- outputs.features_manifest: string

Config merging
- Multiple config files can be passed to the CLI via repeated `-c` options.
- Merge behavior is deep-merge for dicts; lists are replaced.
- Environment variables in string values are expanded (e.g., `${AOTY_DATASET_PATH}`).
