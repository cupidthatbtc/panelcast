"""Full-length (1000-sample) MCMC validation for sigma-ref reparameterization.

Tests verify:
1. 1000-sample MCMC on synthetic data completes without errors
2. Divergence rate stays below generous 50% threshold (synthetic data)
3. R-hat < 1.05 for sigma_ref and n_exponent parameters
4. ESS-bulk > 100 for sigma_ref and n_exponent parameters
5. sigma_obs deterministic site is present with positive values
6. Convergence reference JSON is saved for regression comparison

Manual CLI command for real-data validation (target: divergence_rate < 0.05):
    python -m panelcast.pipelines.orchestrator \\
        --learn-n-exponent --samples 1000 --warmup 1000 --chains 4 \\
        --allow-divergences
    # Check models/training_summary.json for diagnostics
"""

import datetime
import json
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from panelcast.models.bayes.fit import MCMCConfig, fit_model
from panelcast.models.bayes.model import user_score_model
from panelcast.models.bayes.priors import PriorConfig

# Path to convergence reference output
CONVERGENCE_REF_PATH = Path(__file__).resolve().parents[4] / "models" / "convergence_reference.json"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_sample_data():
    """Create sample data dict for model tests (matches test_sigma_ref_training.py pattern)."""
    key = random.PRNGKey(42)
    n_obs = 50
    n_artists = 10
    n_features = 3

    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
        "prev_score": jnp.zeros(n_obs),
        "X": random.normal(key, (n_obs, n_features)),
        "n_artists": n_artists,
        "max_seq": (n_obs // n_artists),
        "n_reviews": jnp.array([10 + i * 5 for i in range(n_obs)], dtype=jnp.float32),
    }


def _make_model_args_for_fit():
    """Create full model_args dict suitable for fit_model().

    Adds observed y, priors, and heteroscedastic config to base sample data.
    Computes n_ref as median(n_reviews) matching the train_models convention.
    """
    data = _make_sample_data()
    data["y"] = np.asarray(random.normal(random.PRNGKey(1), (50,)) * 5 + 70)
    data["learn_n_exponent"] = True
    data["n_exponent"] = 0.0
    data["n_exponent_prior"] = "logit-normal"
    data["n_ref"] = float(np.median(np.asarray(data["n_reviews"])))
    data["priors"] = PriorConfig()
    return data


# ---------------------------------------------------------------------------
# Module-scope fixture: shared across BOTH test classes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_fit_result():
    """Run fit_model with 1000 samples, 2 chains (shared across module).

    This is the expensive MCMC run -- module scope ensures it runs only once.
    """
    model_args = _make_model_args_for_fit()
    config = MCMCConfig(num_warmup=1000, num_samples=1000, num_chains=2)
    return fit_model(user_score_model, model_args, config=config, progress_bar=False)


# ===========================================================================
# Class 1: Full validation diagnostics
# ===========================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
class TestFullValidation:
    """Full-length MCMC validation checking convergence diagnostics.

    Verifies that sigma_ref reparameterization produces well-converged
    posteriors on synthetic data with 1000 samples and 2 chains.
    """

    def test_divergence_rate_below_threshold(self, full_fit_result):
        """Divergence rate should be below 50% on synthetic data."""
        num_samples = 1000
        num_chains = 2
        divergence_rate = full_fit_result.divergences / (num_samples * num_chains)
        print(
            f"\nDivergence rate: {divergence_rate:.4f} "
            f"({full_fit_result.divergences}/{num_samples * num_chains})"
        )
        assert divergence_rate < 0.50, (
            f"Divergence rate {divergence_rate:.4f} >= 0.50 threshold. "
            f"Got {full_fit_result.divergences} divergences in {num_samples * num_chains} samples."
        )

    def test_sigma_ref_rhat(self, full_fit_result):
        """R-hat for sigma_ref should be below 1.05."""
        summary = az.summary(
            full_fit_result.idata, var_names=["user_sigma_ref"], kind="diagnostics"
        )
        r_hat = float(summary["r_hat"].iloc[0])
        print(f"\nsigma_ref R-hat: {r_hat:.4f}")
        assert r_hat < 1.05, f"sigma_ref R-hat {r_hat:.4f} >= 1.05"

    def test_sigma_ref_ess_bulk(self, full_fit_result):
        """ESS-bulk for sigma_ref should exceed 100."""
        summary = az.summary(
            full_fit_result.idata, var_names=["user_sigma_ref"], kind="diagnostics"
        )
        ess_bulk = float(summary["ess_bulk"].iloc[0])
        print(f"\nsigma_ref ESS-bulk: {ess_bulk:.0f}")
        assert ess_bulk > 100, f"sigma_ref ESS-bulk {ess_bulk:.0f} <= 100"

    def test_n_exponent_rhat(self, full_fit_result):
        """R-hat for n_exponent should be below 1.05."""
        summary = az.summary(
            full_fit_result.idata, var_names=["user_n_exponent"], kind="diagnostics"
        )
        r_hat = float(summary["r_hat"].iloc[0])
        print(f"\nn_exponent R-hat: {r_hat:.4f}")
        assert r_hat < 1.05, f"n_exponent R-hat {r_hat:.4f} >= 1.05"

    def test_n_exponent_ess_bulk(self, full_fit_result):
        """ESS-bulk for n_exponent should exceed 100."""
        summary = az.summary(
            full_fit_result.idata, var_names=["user_n_exponent"], kind="diagnostics"
        )
        ess_bulk = float(summary["ess_bulk"].iloc[0])
        print(f"\nn_exponent ESS-bulk: {ess_bulk:.0f}")
        assert ess_bulk > 100, f"n_exponent ESS-bulk {ess_bulk:.0f} <= 100"

    def test_sigma_obs_deterministic_present(self, full_fit_result):
        """sigma_obs deterministic site should be present with positive values."""
        assert (
            "user_sigma_obs" in full_fit_result.idata.posterior
        ), "user_sigma_obs missing from posterior"
        sigma_obs_values = full_fit_result.idata.posterior["user_sigma_obs"].values
        assert np.all(
            sigma_obs_values > 0
        ), f"sigma_obs has non-positive values: min={np.min(sigma_obs_values):.6f}"


# ===========================================================================
# Class 2: Convergence reference JSON
# ===========================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
class TestConvergenceReference:
    """Save convergence diagnostics to JSON for regression comparison.

    Uses the same module-scope MCMC run as TestFullValidation.
    """

    def test_save_convergence_reference(self, full_fit_result):
        """Build and save convergence reference JSON with diagnostic values."""
        num_samples = 1000
        num_chains = 2
        num_warmup = 1000

        # Compute divergence rate
        divergence_rate = float(full_fit_result.divergences / (num_samples * num_chains))

        # Extract sigma_ref diagnostics
        sigma_ref_summary = az.summary(
            full_fit_result.idata, var_names=["user_sigma_ref"], kind="diagnostics"
        )
        sigma_ref_rhat = float(sigma_ref_summary["r_hat"].iloc[0])
        sigma_ref_ess_bulk = int(sigma_ref_summary["ess_bulk"].iloc[0])
        sigma_ref_ess_tail = int(sigma_ref_summary["ess_tail"].iloc[0])

        # Extract n_exponent diagnostics
        n_exp_summary = az.summary(
            full_fit_result.idata, var_names=["user_n_exponent"], kind="diagnostics"
        )
        n_exp_rhat = float(n_exp_summary["r_hat"].iloc[0])
        n_exp_ess_bulk = int(n_exp_summary["ess_bulk"].iloc[0])
        n_exp_ess_tail = int(n_exp_summary["ess_tail"].iloc[0])

        # Determine pass/fail
        passed = (
            divergence_rate < 0.50
            and sigma_ref_rhat < 1.05
            and n_exp_rhat < 1.05
            and sigma_ref_ess_bulk > 100
            and n_exp_ess_bulk > 100
        )

        reference = {
            "validation_type": "full",
            "data_type": "synthetic",
            "num_samples": num_samples,
            "num_chains": num_chains,
            "num_warmup": num_warmup,
            "divergence_rate": divergence_rate,
            "diagnostics": {
                "sigma_ref": {
                    "r_hat": sigma_ref_rhat,
                    "ess_bulk": sigma_ref_ess_bulk,
                    "ess_tail": sigma_ref_ess_tail,
                },
                "n_exponent": {
                    "r_hat": n_exp_rhat,
                    "ess_bulk": n_exp_ess_bulk,
                    "ess_tail": n_exp_ess_tail,
                },
            },
            "passed": passed,
            "recommend_grid_search": divergence_rate >= 0.05,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Write to models/convergence_reference.json
        CONVERGENCE_REF_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONVERGENCE_REF_PATH.write_text(json.dumps(reference, indent=2) + "\n")

        # Verify the file was written and is valid JSON
        assert CONVERGENCE_REF_PATH.exists(), "convergence_reference.json was not created"
        loaded = json.loads(CONVERGENCE_REF_PATH.read_text())
        assert "divergence_rate" in loaded
        assert "diagnostics" in loaded
        assert "passed" in loaded
        print(
            f"\nConvergence reference saved: passed={loaded['passed']}, "
            f"div_rate={loaded['divergence_rate']:.4f}"
        )
