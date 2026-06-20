"""Examine n_exponent posterior to inform prior scale decision (Phase 4B)."""

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
from scipy.special import expit

# Load the most recent model
model_dir = Path("models")
manifest_path = model_dir / "manifest.json"
if not manifest_path.exists():
    print("ERROR: models/manifest.json not found")
    sys.exit(1)

with open(manifest_path) as f:
    manifest = json.load(f)

model_file = manifest.get("current", {}).get("user_score")
if model_file is None:
    print("ERROR: no current user_score model in manifest")
    sys.exit(1)

model_path = model_dir / model_file
print(f"Loading model: {model_path}")
idata = az.from_netcdf(model_path)

# Check if n_exponent is in posterior
posterior_vars = list(idata.posterior.data_vars)
n_exp_var = None
for v in posterior_vars:
    if "n_exponent" in v:
        n_exp_var = v
        break

if n_exp_var is None:
    print("n_exponent NOT in posterior (model uses fixed exponent)")
    print(f"Available vars: {posterior_vars[:15]}")
    sys.exit(0)

print(f"\nFound: {n_exp_var}")
samples = idata.posterior[n_exp_var].values.flatten()
print(f"  n_samples: {len(samples)}")
print(f"  mean:   {np.mean(samples):.4f}")
print(f"  std:    {np.std(samples):.4f}")
print(f"  median: {np.median(samples):.4f}")
print(f"  HDI 90%: [{np.percentile(samples, 5):.4f}, {np.percentile(samples, 95):.4f}]")
print(f"  HDI 95%: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
print(f"  min:    {np.min(samples):.4f}")
print(f"  max:    {np.max(samples):.4f}")

# Compare prior vs posterior
print("\n--- Prior comparison ---")
for scale in [0.5, 0.7, 1.0, 1.5]:
    z = np.random.default_rng(42).normal(0, scale, 10000)
    prior_samples = expit(z)
    lo, hi = np.percentile(prior_samples, 2.5), np.percentile(prior_samples, 97.5)
    print(
        f"  LogitNormal(0, {scale}): 95% CI = [{lo:.3f}, {hi:.3f}], "
        f"mean={np.mean(prior_samples):.3f}"
    )

# Recommendation
post_mean = np.mean(samples)
post_std = np.std(samples)
print("\n--- Recommendation ---")
print(f"Posterior: N_exp ~ {post_mean:.3f} ± {post_std:.3f}")
if post_std < 0.05:
    print("Posterior is very concentrated. Prior scale=1.0 is already overwhelmed by data.")
    print("No change needed — data dominates regardless of prior.")
elif post_std < 0.15:
    print("Posterior is moderately concentrated.")
    print("Consider scale=0.7 (95% CI [0.20, 0.80]) to match posterior spread.")
else:
    print("Posterior is diffuse — prior matters. Keep scale=1.0 or investigate model.")
