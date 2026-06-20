"""Standalone experiment: does a logit target-transform fix the PPC mismatch?

The production model fits Student-t on raw scores in [0, 100] with a soft-clip.
Observed scores are left-skewed (skewness ~= -1.79); the symmetric likelihood
cannot match that skew, producing PPC p-values pinned at 0.000 / 1.000 for
sd, skewness, q50, q90, and max.

This script tests the smallest-possible intervention: transform y onto an
unbounded scale before fitting, so a symmetric likelihood is appropriate,
then back-transform for PPC comparison on the original scale.

It is INTENTIONALLY standalone. It does not import the production model,
does not write to models/, does not mutate configs/base.yaml. Two minimal
MCMC fits, a side-by-side PPC table, and a comparison plot.

Run:
    pixi run -- python scripts/experiment_logit_transform.py

Outputs:
    reports/experiments/logit_transform/ppc_comparison.csv
    reports/experiments/logit_transform/ppc_comparison.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import scipy.stats
from numpyro.infer import MCMC, NUTS, Predictive

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("logit_experiment")

# --- Paths -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "user_score_minratings_10.parquet"
OUT_DIR = REPO_ROOT / "reports" / "experiments" / "logit_transform"


# --- Target transform --------------------------------------------------------
# Design choice: the naive logit(y / 100) explodes at y=0 and y=100, and AOTY
# has albums at both extremes. We use the Smithson-Verkuilen style offset:
#     p = (y + 0.5) / 101
#     z = logit(p) = log(p / (1 - p))
# The +0.5 / +1 offset keeps inputs safely inside (0, 1) with minimal
# distortion (nudges each observation by one half-point on a 100-point scale).
# Using the same constants in both directions guarantees exact invertibility.
#
# Alternatives worth experimenting with if logit underperforms:
#   - probit((y + 0.5) / 101): similar behavior, slightly thinner tails
#   - larger offset (y + 1) / 102: stronger regularization at extremes
#   - asinh(y - 50): smooth, no bounded domain, but doesn't use the [0, 100]
#     structure

_OFFSET = 0.5
_SCALE = 101.0  # == 100 + 2 * _OFFSET


def logit_transform(y: np.ndarray) -> np.ndarray:
    """Map scores in [0, 100] onto the real line via offset-logit."""
    p = (np.asarray(y, dtype=np.float64) + _OFFSET) / _SCALE
    return np.log(p / (1.0 - p)).astype(np.float32)


def inv_logit_transform(z: np.ndarray) -> np.ndarray:
    """Inverse: logit-inverse, undo offset. Exact inverse of logit_transform."""
    z = np.asarray(z, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    return (p * _SCALE - _OFFSET).astype(np.float32)


# --- Minimal models ----------------------------------------------------------
# Two intentionally-simple models, just enough to isolate the likelihood
# question. No hierarchy, no random walk, no AR(1). The production pipeline
# already demonstrates those; the question here is purely about the target
# distribution's interaction with a symmetric likelihood.


def model_raw(X: jnp.ndarray, y: jnp.ndarray | None = None) -> None:
    """Fit Student-t on raw y with soft-clip to [0, 100] (baseline)."""
    n_features = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(70.0, 10.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([n_features]).to_event(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))
    # Soft-clip to mirror production behavior. softplus-based, identity in
    # the interior, saturates near 0 and 100.
    mu_raw = alpha + X @ beta
    s = 0.5  # sharpness
    mu = 0.0 + jax.nn.softplus(s * (mu_raw - 0.0)) / s
    mu = 100.0 - jax.nn.softplus(s * (100.0 - mu)) / s
    with numpyro.plate("obs", X.shape[0]):
        numpyro.sample("y", dist.StudentT(4.0, mu, sigma), obs=y)


def model_logit(X: jnp.ndarray, z: jnp.ndarray | None = None) -> None:
    """Fit Student-t on logit-transformed target z (unbounded scale)."""
    n_features = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([n_features]).to_event(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    mu = alpha + X @ beta  # no clip — target is unbounded here
    with numpyro.plate("obs", X.shape[0]):
        numpyro.sample("z", dist.StudentT(4.0, mu, sigma), obs=z)


# --- PPC helpers -------------------------------------------------------------

PPC_STATS = {
    "mean": np.mean,
    "sd": np.std,
    "skewness": lambda x: float(scipy.stats.skew(x, nan_policy="omit")),
    "q10": lambda x: float(np.percentile(x, 10)),
    "q50": lambda x: float(np.percentile(x, 50)),
    "q90": lambda x: float(np.percentile(x, 90)),
    "min": np.min,
    "max": np.max,
}


def bayesian_p(t_obs: float, t_rep: np.ndarray) -> float:
    """P(T(y_rep) >= T(y_obs)). Under well-specified model, ~= 0.5."""
    return float(np.mean(t_rep >= t_obs))


def ppc_table(y_obs: np.ndarray, y_rep: np.ndarray, label: str) -> pd.DataFrame:
    """Compute PPC stats; y_rep is (n_samples, n_obs)."""
    rows = []
    for name, fn in PPC_STATS.items():
        t_obs = float(fn(y_obs))
        t_rep = np.array([fn(y_rep[i]) for i in range(y_rep.shape[0])])
        rows.append(
            {
                "statistic": name,
                "t_obs": t_obs,
                "p_value": bayesian_p(t_obs, t_rep),
                "model": label,
            }
        )
    return pd.DataFrame(rows)


# --- Main --------------------------------------------------------------------


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from the cleaned dataset. Tiny fixed feature set."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected processed data at {DATA_PATH}. Run `panelcast stage data` first."
        )
    df = pd.read_parquet(DATA_PATH)
    y = df["User_Score"].to_numpy(dtype=np.float32)
    # Minimal features: year z-scored + log review count. The experiment
    # isolates the likelihood question, not feature engineering.
    year = df["Year"].astype(float).to_numpy()
    year_z = (year - year.mean()) / year.std()
    reviews = df["User_Ratings"].astype(float).to_numpy()
    log_reviews_z = (np.log1p(reviews) - np.log1p(reviews).mean()) / np.log1p(reviews).std()
    X = np.stack([year_z, log_reviews_z], axis=1).astype(np.float32)
    return X, y


def run_mcmc(model, X, obs, seed=0, num_warmup=500, num_samples=500):
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.asarray(X), **obs)
    return mcmc


def posterior_predictive(model, mcmc, X, site, seed=1):
    predictive = Predictive(model, posterior_samples=mcmc.get_samples())
    draws = predictive(jax.random.PRNGKey(seed), X=jnp.asarray(X))
    return np.asarray(draws[site])


def plot_comparison(y_obs, y_rep_raw, y_rep_logit_backtransformed, out_path: Path) -> None:
    """Overlay observed KDE with both posterior-predictive KDEs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 100, 60)
    ax.hist(y_obs, bins=bins, density=True, alpha=0.35, label="observed", color="black")
    ax.hist(y_rep_raw[0], bins=bins, density=True, alpha=0.45, label="raw-y model", color="tab:red")
    ax.hist(
        y_rep_logit_backtransformed[0],
        bins=bins,
        density=True,
        alpha=0.45,
        label="logit-transformed model",
        color="tab:blue",
    )
    ax.set_xlabel("User Score")
    ax.set_ylabel("density")
    ax.set_title("Posterior-predictive density vs. observed — one posterior draw per model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading data from %s", DATA_PATH)
    X, y = load_data()
    log.info(
        "n=%d, n_features=%d, y range=[%.1f, %.1f], skew=%.3f",
        len(y),
        X.shape[1],
        y.min(),
        y.max(),
        scipy.stats.skew(y),
    )

    # --- Baseline: fit on raw y ---
    log.info("Fitting baseline model on raw y (symmetric Student-t + soft clip)...")
    mcmc_raw = run_mcmc(model_raw, X, obs={"y": jnp.asarray(y)}, seed=0)
    y_rep_raw = posterior_predictive(model_raw, mcmc_raw, X, site="y", seed=1)
    raw_table = ppc_table(y, y_rep_raw, label="raw")

    # --- Variant: fit on logit-transformed y ---
    log.info("Transforming target and fitting logit-variant...")
    z = logit_transform(y)
    mcmc_logit = run_mcmc(model_logit, X, obs={"z": jnp.asarray(z)}, seed=0)
    z_rep = posterior_predictive(model_logit, mcmc_logit, X, site="z", seed=1)
    y_rep_logit = inv_logit_transform(z_rep)
    logit_table = ppc_table(y, y_rep_logit, label="logit")

    # --- Report ---
    out = pd.concat([raw_table, logit_table], ignore_index=True)
    out_path_csv = OUT_DIR / "ppc_comparison.csv"
    out.to_csv(out_path_csv, index=False)
    log.info("Wrote %s", out_path_csv)

    pivot = out.pivot(index="statistic", columns="model", values="p_value")
    log.info("\nBayesian p-values (closer to 0.5 is better):\n%s", pivot.to_string())

    plot_comparison(y, y_rep_raw, y_rep_logit, OUT_DIR / "ppc_comparison.png")
    log.info("Wrote %s", OUT_DIR / "ppc_comparison.png")

    pinned_raw = ((pivot["raw"] <= 0.05) | (pivot["raw"] >= 0.95)).sum()
    pinned_logit = ((pivot["logit"] <= 0.05) | (pivot["logit"] >= 0.95)).sum()
    log.info(
        "PPC statistics outside [0.05, 0.95]: raw=%d/%d, logit=%d/%d",
        pinned_raw,
        len(pivot),
        pinned_logit,
        len(pivot),
    )


if __name__ == "__main__":
    main()
