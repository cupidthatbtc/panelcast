"""Controlled PPC comparison of the likelihood families on a skewed, bounded target.

The flagship AOTY dataset is not bundled (only a tiny fixture), and there is no
GPU here, so the headline AOTY PPC numbers stay compute-bound (see MODEL_CARD).
This experiment instead demonstrates the *mechanism* the review flagged on a
synthetic panel deliberately built to be left-skewed and bounded (skewness
~ -1.8 on [0, 100]) — the same pathology as AOTY user scores. It fits the model
under each likelihood family at validation scale and reports the posterior-
predictive p-values, showing whether the skew/bounded candidates move the pinned
statistics (sd, skewness, min, max, q-tails) off the extremes.

    python scripts/experiment_likelihood_ppc.py

Writes outputs/experiments/likelihood_ppc.json and prints a comparison table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402

jax.config.update("jax_platform_name", "cpu")

from panelcast.evaluation.ppc import compute_ppc_statistics  # noqa: E402
from panelcast.models.bayes.fit import MCMCConfig, fit_model  # noqa: E402
from panelcast.models.bayes.model import make_score_model  # noqa: E402
from panelcast.models.bayes.predict import generate_posterior_predictive  # noqa: E402
from panelcast.models.bayes.priors import PriorConfig  # noqa: E402

BOUNDS = (0.0, 100.0)
FAMILIES = ("studentt", "skew_studentt", "skew_normal", "split_normal", "beta")
# The statistics the review reported as pinned (sd, skewness, q50, q90, max).
WATCH = ("sd", "skewness", "min", "max", "q50", "q90")


def make_skewed_panel(seed: int = 0):
    """A left-skewed, bounded panel with real entity structure."""
    rng = np.random.default_rng(seed)
    n_artists = 40
    n_features = 3
    rows = []
    for a in range(n_artists):
        n_albums = int(rng.integers(3, 7))
        artist_level = float(rng.normal(80.0, 6.0))  # high baseline -> ceiling
        effect = 0.0
        prev = artist_level
        for t in range(1, n_albums + 1):
            effect += float(rng.normal(0.0, 1.5))
            x = rng.standard_normal(n_features)
            # Long LEFT tail: subtract an exponential from a high mean, clip.
            score = artist_level + effect + 2.0 * x[0] - float(rng.exponential(7.0))
            score = float(np.clip(score, 0.0, 100.0))
            rows.append(
                {
                    "artist": a,
                    "seq": t,
                    "prev": prev,
                    "x": x,
                    "y": score,
                }
            )
            prev = score
    artist_idx = np.array([r["artist"] for r in rows], dtype=np.int32)
    album_seq = np.array([r["seq"] for r in rows], dtype=np.int32)
    prev_score = np.array([r["prev"] for r in rows], dtype=np.float32)
    X = np.stack([r["x"] for r in rows]).astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    return artist_idx, album_seq, prev_score, X, y


def run_family(family: str, data, seed: int = 0) -> dict:
    artist_idx, album_seq, prev_score, X, y = data
    n_artists = int(artist_idx.max()) + 1
    train_mean = float(np.mean(y))
    model = make_score_model("user")
    model_args = dict(
        artist_idx=artist_idx,
        album_seq=album_seq,
        prev_score=prev_score,
        X=X,
        y=y,
        n_artists=n_artists,
        max_seq=int(album_seq.max()),
        priors=PriorConfig(likelihood_family=family),
        target_bounds=BOUNDS,
        likelihood_df=4.0,
        ar_center=train_mean,
    )
    config = MCMCConfig(
        num_warmup=500,
        num_samples=500,
        num_chains=2,
        seed=seed,
        target_accept_prob=0.9,
    )
    result = fit_model(model=model, model_args=model_args, config=config)
    ppc = generate_posterior_predictive(model, result.mcmc, model_args, seed=seed + 1)
    y_rep = np.asarray(ppc.y)
    stats = compute_ppc_statistics(y, y_rep)
    # Convergence summary (max R-hat / min ESS) from the MCMC.
    import arviz as az

    idata = az.from_numpyro(result.mcmc)
    summ = az.summary(idata)
    return {
        "family": family,
        "ppc": {k: v["p_value"] for k, v in stats.summary.items()},
        "rhat_max": float(summ["r_hat"].max()),
        "ess_bulk_min": float(summ["ess_bulk"].min()),
        "yrep_min": float(y_rep.min()),
        "yrep_max": float(y_rep.max()),
    }


def main() -> None:
    data = make_skewed_panel()
    y = data[4]
    import scipy.stats as ss

    obs_skew = float(ss.skew(y))
    print(f"Synthetic target: n={len(y)}, skewness={obs_skew:.2f}, range=[{y.min():.1f},{y.max():.1f}]\n")

    results = [run_family(fam, data) for fam in FAMILIES]

    header = f"{'family':<14} {'rhat':>6} {'ess':>6} " + " ".join(f"{s:>9}" for s in WATCH)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['family']:<14} {r['rhat_max']:>6.3f} {r['ess_bulk_min']:>6.0f} "
        row += " ".join(f"{r['ppc'].get(s, float('nan')):>9.3f}" for s in WATCH)
        print(row)
    print("\n(p-values near 0.000/1.000 are 'pinned'; interior values ~0.1-0.9 are healthy.)")

    out_dir = Path("outputs/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "likelihood_ppc.json"
    out_path.write_text(
        json.dumps({"observed_skewness": obs_skew, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
