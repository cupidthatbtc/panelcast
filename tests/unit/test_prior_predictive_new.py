"""New coverage tests for evaluation/prior_predictive.py.

Targets uncovered paths:
- run_prior_predictive: 2D array subsampling, n_obs_original detection via different
  array keys (album_seq, prev_score, X), priors=None using model_args priors,
  summary computation with actual statistics, fraction_threshold edge at 1.0
- generate_prior_justification_text: sensitivity df without 'elpd_delta' column,
  sensitivity with only default variants after filtering
"""

import numpy as np
import pandas as pd
import pytest

from panelcast.evaluation.prior_predictive import (
    PriorPredictiveResult,
    generate_prior_justification_text,
    run_prior_predictive,
)
from panelcast.models.bayes.priors import PriorConfig

# ---------------------------------------------------------------------------
# Helpers for mocking JAX/numpyro inside run_prior_predictive
# ---------------------------------------------------------------------------


def _setup_mocks(monkeypatch, predictive_cls):
    """Wire up fake jax.random and numpyro.infer modules."""
    import sys

    fake_jax_random = type(sys)("fake_jax_random")
    fake_jax_random.key = lambda seed: seed
    monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

    fake_numpyro_infer = type(sys)("fake_numpyro_infer")
    fake_numpyro_infer.Predictive = predictive_cls
    monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)


class _SimplePredictive:
    """Predictive that returns constant obs_y = 50."""

    def __init__(self, model, num_samples):
        self._n = num_samples

    def __call__(self, rng_key, **kwargs):
        # Determine n_obs from any array key present
        for key in ("artist_idx", "album_seq", "prev_score", "X"):
            if key in kwargs and hasattr(kwargs[key], "__len__"):
                n = len(kwargs[key])
                return {"obs_y": np.ones((self._n, n)) * 50.0}
        return {"obs_y": np.ones((self._n, 1)) * 50.0}


# ===========================================================================
# run_prior_predictive: subsampling with 2D arrays
# ===========================================================================


class TestSubsamplingWith2DArrays:
    """Cover the 2D array subsampling path (val.ndim == 2)."""

    def test_2d_X_array_subsampled(self, monkeypatch):
        """2D feature matrix X is subsampled on axis=0."""
        n_obs = 200
        max_obs = 30

        captured = {}

        class CapturePredictive:
            def __init__(self, model, num_samples):
                self._n = num_samples

            def __call__(self, rng_key, **kwargs):
                captured.update(kwargs)
                n = kwargs["X"].shape[0]
                return {"obs_y": np.ones((self._n, n)) * 50.0}

        _setup_mocks(monkeypatch, CapturePredictive)

        model_args = {
            "artist_idx": np.arange(n_obs),
            "album_seq": np.ones(n_obs),
            "prev_score": np.zeros(n_obs),
            "X": np.random.default_rng(0).normal(0, 1, (n_obs, 5)),
            "y": np.ones(n_obs) * 60,
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=3,
            max_obs=max_obs,
            seed=7,
        )

        assert result.sampled_indices is not None
        assert len(result.sampled_indices) == max_obs
        assert captured["X"].shape == (max_obs, 5)
        assert captured["artist_idx"].shape == (max_obs,)


class TestNObsDetectedFromDifferentKeys:
    """Cover n_obs_original detection from album_seq, prev_score, X."""

    def test_n_obs_from_album_seq_only(self, monkeypatch):
        """Detect n_obs from album_seq when artist_idx is absent."""
        _setup_mocks(monkeypatch, _SimplePredictive)

        model_args = {
            "album_seq": np.array([1, 2, 3, 4]),
            "y": np.array([50, 60, 70, 80]),
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=3,
        )
        assert result.n_obs_original == 4

    def test_n_obs_from_prev_score_only(self, monkeypatch):
        """Detect n_obs from prev_score when artist_idx/album_seq absent."""
        _setup_mocks(monkeypatch, _SimplePredictive)

        model_args = {
            "prev_score": np.array([0.0, 55.0, 60.0]),
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=3,
        )
        assert result.n_obs_original == 3

    def test_n_obs_from_X_only(self, monkeypatch):
        """Detect n_obs from X when other array keys absent."""

        class XPredictive:
            def __init__(self, model, num_samples):
                self._n = num_samples

            def __call__(self, rng_key, **kwargs):
                n = kwargs["X"].shape[0] if "X" in kwargs else 1
                return {"obs_y": np.ones((self._n, n)) * 50.0}

        _setup_mocks(monkeypatch, XPredictive)

        model_args = {
            "X": np.ones((7, 3)),
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=3,
        )
        assert result.n_obs_original == 7


class TestPriorsNoneUsesModelArgsDefault:
    """Cover the priors=None path (uses model_args['priors'] if present)."""

    def test_priors_none_keeps_model_args_priors(self, monkeypatch):
        captured = {}

        class CapturePredictive:
            def __init__(self, model, num_samples):
                self._n = num_samples

            def __call__(self, rng_key, **kwargs):
                captured.update(kwargs)
                return {"obs_y": np.ones((self._n, 2)) * 50.0}

        _setup_mocks(monkeypatch, CapturePredictive)

        custom = PriorConfig(mu_artist_loc=77.0)
        model_args = {
            "artist_idx": np.array([0, 1]),
            "y": np.array([50.0, 60.0]),
            "priors": custom,
        }

        run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            priors=None,
            n_samples=3,
        )

        # priors=None should NOT replace model_args['priors']
        assert captured["priors"].mu_artist_loc == 77.0


class TestSummaryStatistics:
    """Cover the summary statistics computation path."""

    def test_summary_values_match_computed(self, monkeypatch):
        """Verify summary values match numpy computations on y_samples."""
        rng = np.random.default_rng(99)
        samples = rng.normal(50, 15, (20, 10))

        class FixedPredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"test_y": samples}

        _setup_mocks(monkeypatch, FixedPredictive)

        model_args = {"artist_idx": np.arange(10)}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=20,
        )

        flat = samples.ravel()
        assert result.summary["mean"] == pytest.approx(float(np.mean(flat)), rel=1e-5)
        assert result.summary["sd"] == pytest.approx(float(np.std(flat)), rel=1e-5)
        assert result.summary["min"] == pytest.approx(float(np.min(flat)), rel=1e-5)
        assert result.summary["max"] == pytest.approx(float(np.max(flat)), rel=1e-5)


class TestFractionThresholdEdge:
    """Cover fraction_threshold = 1.0 boundary."""

    def test_fraction_threshold_one_requires_all_in_bounds(self, monkeypatch):
        """fraction_threshold=1.0 requires 100% of samples in bounds."""

        class AllInBoundsPredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"obs_y": np.ones((5, 3)) * 50.0}

        _setup_mocks(monkeypatch, AllInBoundsPredictive)

        model_args = {"artist_idx": np.array([0, 1, 2])}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
            fraction_threshold=1.0,
        )

        assert result.reasonable is True
        assert result.fraction_in_bounds == 1.0


class TestRunPriorPredictiveCustomSeed:
    """Cover seed propagation and reproducibility."""

    def test_different_seeds_propagated(self, monkeypatch):
        """Different seeds should be passed to rng_key."""
        captured_keys = []

        class SeedCapturePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                captured_keys.append(rng_key)
                return {"obs_y": np.ones((3, 2)) * 50.0}

        _setup_mocks(monkeypatch, SeedCapturePredictive)

        model_args = {"artist_idx": np.array([0, 1])}

        run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=3,
            seed=100,
        )
        run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=3,
            seed=200,
        )

        # Seeds should be different (our mock just stores the seed integer)
        assert captured_keys[0] != captured_keys[1]


# ===========================================================================
# generate_prior_justification_text: sensitivity edge cases
# ===========================================================================


class TestSensitivityWithoutElpdDelta:
    """Cover sensitivity DataFrame that lacks 'elpd_delta' column."""

    def test_no_elpd_delta_column_no_sensitivity_section(self):
        priors = PriorConfig()
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(priors, sensitivity_summary=df)
        assert "Sensitivity" not in text


class TestSensitivityAllDefaultsAfterFilter:
    """Cover case where eligible variants are all baseline/default after filter."""

    def test_only_default_variant_after_eligible_filter(self):
        priors = PriorConfig()
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -5.0],
                "eligible_for_ranking": [True, False],
            }
        )
        text = generate_prior_justification_text(priors, sensitivity_summary=df)
        # sigma_rw is ineligible, only baseline remains -> no sensitivity section
        assert "Sensitivity" not in text


class TestJustificationTextStructure:
    """Verify text structure with all three sections."""

    def test_all_sections_ordering(self):
        priors = PriorConfig()
        ppr = PriorPredictiveResult(
            y_samples=np.ones((10, 5)) * 50,
            summary={
                "mean": 50.0,
                "sd": 10.0,
                "q2.5": 30.0,
                "q97.5": 70.0,
                "min": 10.0,
                "max": 90.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.95,
            n_samples=10,
        )
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -3.0],
                "eligible_for_ranking": [True, True],
            }
        )

        text = generate_prior_justification_text(
            priors,
            prior_predictive_result=ppr,
            sensitivity_summary=df,
        )

        # Verify ordering: domain text first, then PPC, then sensitivity
        ppc_idx = text.index("Prior Predictive Check")
        sens_idx = text.index("Sensitivity")
        assert ppc_idx < sens_idx

    def test_ppc_section_includes_bounds(self):
        priors = PriorConfig()
        ppr = PriorPredictiveResult(
            y_samples=np.ones((10, 5)) * 50,
            summary={
                "mean": 50.0,
                "sd": 10.0,
                "q2.5": 30.0,
                "q97.5": 70.0,
                "min": 10.0,
                "max": 90.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.88,
            n_samples=200,
        )

        text = generate_prior_justification_text(priors, prior_predictive_result=ppr)

        assert "[0, 100]" in text
        assert "88.0%" in text
        assert "n_samples=200" in text
