# MCMC Comparison Configuration

Bake-off and comparison runs (`scripts/bakeoff_transform_latent.py`,
`scripts/bakeoff_likelihoods.py`) invoke `panelcast run --preset diagnostic`,
so the settings that actually run are the diagnostic preset
(`configs/diagnostic.yaml`) plus the base seed:

- seed: 42 (`configs/base.yaml`)
- warmup: 1000
- samples: 1000
- chains: 4
- target_accept: 0.90

Every cell's `metrics.json` confirms this (`ppc.n_samples: 4000` = 4 chains
× 1000 draws).

The original Phase 4 pipeline comparisons predating the bake-offs used
2 chains at otherwise identical settings; artifacts from that era were
superseded by the diagnostic-preset runs snapshotted under
`.audit/transform_latent_bakeoff/` and `.audit/model_v2_bakeoff/`.

Baseline data hash: see `baseline_data_hashes.txt`
