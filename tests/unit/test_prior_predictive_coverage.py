"""Coverage-targeted tests for prior predictive module.

Targets missed lines and branches in evaluation/prior_predictive.py:
- run_prior_predictive execution with mocked JAX/numpyro
- Subsampling logic for large datasets
- n_reviews subsampling path
- Missing observed site error path
- fraction_threshold validation
- Sensitivity analysis edge cases in generate_prior_justification_text
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


@pytest.fixture
def default_priors():
    """Default PriorConfig for testing."""
    return PriorConfig()


class TestRunPriorPredictiveValidation:
    """Tests for input validation edge cases in run_prior_predictive."""

    def test_rejects_negative_n_samples(self):
        """Negative n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args={},
                n_samples=-5,
            )

    def test_rejects_negative_max_obs(self):
        """Negative max_obs raises ValueError."""
        with pytest.raises(ValueError, match="max_obs must be >= 1"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args={},
                max_obs=-1,
            )

    def test_rejects_equal_score_bounds(self):
        """Equal lower and upper score bounds raises ValueError."""
        with pytest.raises(ValueError, match="score_bounds must be"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args={},
                score_bounds=(50, 50),
            )

    def test_rejects_fraction_threshold_above_one(self):
        """fraction_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="fraction_threshold must be in"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args={},
                fraction_threshold=1.5,
            )

    def test_rejects_fraction_threshold_below_zero(self):
        """fraction_threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="fraction_threshold must be in"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args={},
                fraction_threshold=-0.1,
            )

    def test_rejects_single_element_score_bounds(self):
        """Single-element score_bounds raises ValueError."""
        with pytest.raises(ValueError, match="score_bounds must be"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args={},
                score_bounds=(50,),
            )


class TestRunPriorPredictiveExecution:
    """Tests for the full run_prior_predictive execution path with mocked JAX."""

    def test_basic_execution_with_mock(self, monkeypatch, default_priors):
        """run_prior_predictive completes with mocked Predictive."""
        n_samples = 10
        n_obs = 5
        y_out = np.random.default_rng(42).normal(50, 10, (n_samples, n_obs))

        class FakePredictive:
            def __init__(self, model, num_samples):
                self._n = num_samples

            def __call__(self, rng_key, **kwargs):
                return {"obs_y": y_out}

        class FakeRandom:
            @staticmethod
            def key(seed):
                return seed

        fake_random_module = FakeRandom()

        import panelcast.evaluation.prior_predictive as pp_mod

        # Monkeypatch the imports inside the function
        monkeypatch.setattr(pp_mod, "__name__", pp_mod.__name__)  # no-op to ensure module loaded

        # We need to monkeypatch the lazy imports inside run_prior_predictive.
        # The function does `from jax import random` and `from numpyro.infer import Predictive`
        # at call time. We monkeypatch the modules themselves.
        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)
        monkeypatch.setattr("jax.random", fake_jax_random, raising=False)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {
            "artist_idx": np.arange(n_obs),
            "album_seq": np.ones(n_obs),
            "prev_score": np.zeros(n_obs),
            "X": np.ones((n_obs, 3)),
            "y": np.random.default_rng(0).normal(50, 10, n_obs),
            "priors": default_priors,
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=n_samples,
            seed=42,
        )

        assert result.reasonable is True or result.reasonable is False
        assert result.y_samples.shape == (n_samples, n_obs)
        assert result.n_samples == n_samples
        assert result.n_obs_original == n_obs
        assert result.sampled_indices is None
        assert "mean" in result.summary
        assert "sd" in result.summary
        assert "q2.5" in result.summary
        assert "q97.5" in result.summary
        assert "min" in result.summary
        assert "max" in result.summary
        assert 0.0 <= result.fraction_in_bounds <= 1.0

    def test_uses_custom_priors_when_provided(self, monkeypatch):
        """When priors kwarg is given, it overrides model_args['priors']."""
        captured_args = {}

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                captured_args.update(kwargs)
                n = len(kwargs.get("artist_idx", [1]))
                return {"obs_y": np.ones((5, n)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        custom = PriorConfig(mu_artist_loc=99.0)
        model_args = {
            "artist_idx": np.array([0, 1]),
            "y": np.array([50.0, 60.0]),
            "priors": PriorConfig(),
        }

        run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            priors=custom,
            n_samples=5,
        )

        assert captured_args["priors"].mu_artist_loc == 99.0

    def test_sets_y_to_none(self, monkeypatch):
        """run_prior_predictive sets y=None in model_args for Predictive."""
        captured_args = {}

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                captured_args.update(kwargs)
                return {"obs_y": np.ones((5, 2)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {
            "artist_idx": np.array([0, 1]),
            "y": np.array([50.0, 60.0]),
        }

        run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
        )

        assert captured_args["y"] is None

    def test_missing_observed_site_raises(self, monkeypatch):
        """ValueError when Predictive output has no key ending with '_y'."""

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"mu": np.ones((5, 2)), "sigma": np.ones((5, 2))}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {"artist_idx": np.array([0, 1])}

        with pytest.raises(ValueError, match="Unable to locate observed site"):
            run_prior_predictive(
                model=lambda **kw: None,
                model_args=model_args,
                n_samples=5,
            )


class TestRunPriorPredictiveSubsampling:
    """Tests for subsampling logic when n_obs > max_obs."""

    def test_subsamples_when_over_max_obs(self, monkeypatch):
        """Arrays are subsampled when n_obs exceeds max_obs."""
        n_obs = 200
        max_obs = 50

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                n = len(kwargs["artist_idx"])
                return {"obs_y": np.ones((5, n)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {
            "artist_idx": np.arange(n_obs),
            "album_seq": np.ones(n_obs),
            "prev_score": np.zeros(n_obs),
            "X": np.ones((n_obs, 3)),
            "y": np.ones(n_obs) * 50,
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
            max_obs=max_obs,
            seed=42,
        )

        assert result.n_obs_original == n_obs
        assert result.max_obs == max_obs
        assert result.sampled_indices is not None
        assert len(result.sampled_indices) == max_obs
        assert result.y_samples.shape[1] == max_obs

    def test_subsamples_n_reviews(self, monkeypatch):
        """n_reviews is subsampled along with other arrays."""
        n_obs = 100
        max_obs = 20
        captured_args = {}

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                captured_args.update(kwargs)
                n = len(kwargs["artist_idx"])
                return {"obs_y": np.ones((5, n)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {
            "artist_idx": np.arange(n_obs),
            "album_seq": np.ones(n_obs),
            "prev_score": np.zeros(n_obs),
            "X": np.ones((n_obs, 3)),
            "y": np.ones(n_obs) * 50,
            "n_reviews": np.arange(n_obs, dtype=float),
        }

        run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
            max_obs=max_obs,
            seed=42,
        )

        assert len(captured_args["n_reviews"]) == max_obs

    def test_no_subsampling_when_under_max_obs(self, monkeypatch):
        """No subsampling when n_obs <= max_obs."""
        n_obs = 10

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                n = len(kwargs["artist_idx"])
                return {"obs_y": np.ones((5, n)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {
            "artist_idx": np.arange(n_obs),
            "y": np.ones(n_obs) * 50,
        }

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
            max_obs=2000,
        )

        assert result.sampled_indices is None
        assert result.n_obs_original == n_obs

    def test_no_array_keys_yields_zero_n_obs(self, monkeypatch):
        """When no array keys are present, n_obs_original is 0."""

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"obs_y": np.ones((5, 1)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {"scalar_param": 42}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
        )

        assert result.n_obs_original == 0
        assert result.sampled_indices is None


class TestRunPriorPredictiveReasonableness:
    """Tests for the reasonableness check logic."""

    def test_reasonable_when_all_in_bounds(self, monkeypatch):
        """Result is reasonable when all samples are within bounds."""

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"obs_y": np.ones((10, 5)) * 50}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {"artist_idx": np.array([0, 1, 2, 3, 4])}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=10,
            score_bounds=(0, 100),
            fraction_threshold=0.90,
        )

        assert result.reasonable is True
        assert result.fraction_in_bounds == pytest.approx(1.0)

    def test_not_reasonable_when_out_of_bounds(self, monkeypatch):
        """Result is not reasonable when too many samples are outside bounds."""

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                # All values at 200, well outside (0, 100)
                return {"obs_y": np.ones((10, 5)) * 200}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {"artist_idx": np.array([0, 1, 2, 3, 4])}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=10,
            score_bounds=(0, 100),
            fraction_threshold=0.90,
        )

        assert result.reasonable is False
        assert result.fraction_in_bounds == pytest.approx(0.0)

    def test_boundary_fraction_threshold(self, monkeypatch):
        """At exactly the threshold, result should be reasonable."""

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                # 90 values in bounds (50), 10 values out (200)
                in_bounds = np.ones((1, 90)) * 50
                out_bounds = np.ones((1, 10)) * 200
                return {"obs_y": np.concatenate([in_bounds, out_bounds], axis=1)}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {"artist_idx": np.arange(100)}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=1,
            fraction_threshold=0.90,
        )

        assert result.fraction_in_bounds == pytest.approx(0.90)
        assert result.reasonable is True

    def test_fraction_threshold_zero_always_reasonable(self, monkeypatch):
        """fraction_threshold=0 means any result is reasonable."""

        class FakePredictive:
            def __init__(self, model, num_samples):
                pass

            def __call__(self, rng_key, **kwargs):
                return {"obs_y": np.ones((5, 2)) * 999}

        import sys

        fake_jax_random = type(sys)("fake_jax_random")
        fake_jax_random.key = lambda seed: seed
        monkeypatch.setitem(sys.modules, "jax.random", fake_jax_random)

        fake_numpyro_infer = type(sys)("fake_numpyro_infer")
        fake_numpyro_infer.Predictive = FakePredictive
        monkeypatch.setitem(sys.modules, "numpyro.infer", fake_numpyro_infer)

        model_args = {"artist_idx": np.array([0, 1])}

        result = run_prior_predictive(
            model=lambda **kw: None,
            model_args=model_args,
            n_samples=5,
            fraction_threshold=0.0,
        )

        assert result.reasonable is True


class TestJustificationSensitivityEdgeCases:
    """Tests for sensitivity analysis edge cases in generate_prior_justification_text."""

    def test_empty_sensitivity_df_no_section(self, default_priors):
        """Empty DataFrame produces no sensitivity section."""
        empty_df = pd.DataFrame(
            columns=["variant", "parameter", "elpd_delta", "eligible_for_ranking"]
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=empty_df)
        assert "Sensitivity" not in text

    def test_none_sensitivity_no_section(self, default_priors):
        """None sensitivity_summary produces no sensitivity section."""
        text = generate_prior_justification_text(default_priors, sensitivity_summary=None)
        assert "Sensitivity" not in text

    def test_non_dataframe_sensitivity_ignored(self, default_priors):
        """Non-DataFrame sensitivity_summary is ignored."""
        text = generate_prior_justification_text(
            default_priors, sensitivity_summary="not a dataframe"
        )
        assert "Sensitivity" not in text

    def test_all_ineligible_no_sensitivity_section(self, default_priors):
        """All ineligible variants produce no sensitivity section."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -3.0],
                "eligible_for_ranking": [False, False],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "Sensitivity" not in text

    def test_only_baseline_eligible_no_sensitivity_section(self, default_priors):
        """Only baseline eligible (filtered out) produces no sensitivity section."""
        df = pd.DataFrame(
            {
                "variant": ["default"],
                "parameter": ["baseline"],
                "elpd_delta": [0.0],
                "eligible_for_ranking": [True],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "Sensitivity" not in text

    def test_nan_elpd_delta_dropped(self, default_priors):
        """Rows with NaN elpd_delta are dropped before finding max."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2", "beta_x5"],
                "parameter": ["baseline", "sigma_rw_scale", "beta_scale"],
                "elpd_delta": [0.0, float("nan"), -2.0],
                "eligible_for_ranking": [True, True, True],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "Sensitivity" in text
        assert "beta_scale" in text

    def test_moderate_sensitivity_label(self, default_priors):
        """Large |delta_elpd| >= 5 produces 'moderate' sensitivity label."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -10.0],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "moderate" in text

    def test_minimal_sensitivity_label(self, default_priors):
        """Small |delta_elpd| < 5 produces 'minimal' sensitivity label."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -2.0],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "minimal" in text

    def test_sensitivity_with_missing_variant_column(self, default_priors):
        """DataFrame without 'variant' column still works."""
        df = pd.DataFrame(
            {
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -3.0],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "Sensitivity" in text
        assert "sigma_rw_scale" in text

    def test_all_elpd_delta_nan_no_sensitivity(self, default_priors):
        """All NaN elpd_delta produces no sensitivity section after dropna."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [float("nan"), float("nan")],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors, sensitivity_summary=df)
        assert "Sensitivity" not in text


class TestJustificationWithBothPPCAndSensitivity:
    """Tests for justification text with both PPC and sensitivity data."""

    def test_both_sections_present(self, default_priors):
        """Both PPC and sensitivity sections appear when data is provided."""
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
        sensitivity_df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -3.5],
                "eligible_for_ranking": [True, True],
            }
        )

        text = generate_prior_justification_text(
            default_priors,
            prior_predictive_result=ppr,
            sensitivity_summary=sensitivity_df,
        )

        assert "Prior Predictive Check" in text
        assert "Sensitivity" in text
        assert "sigma_rw_scale" in text


class TestPriorPredictiveResultDefaults:
    """Tests for PriorPredictiveResult default field values."""

    def test_default_n_samples(self):
        """Default n_samples is 0."""
        result = PriorPredictiveResult(
            y_samples=np.zeros((1, 1)),
            summary={},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
        )
        assert result.n_samples == 0

    def test_default_max_obs(self):
        """Default max_obs is 2000."""
        result = PriorPredictiveResult(
            y_samples=np.zeros((1, 1)),
            summary={},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
        )
        assert result.max_obs == 2000

    def test_default_seed(self):
        """Default seed is 42."""
        result = PriorPredictiveResult(
            y_samples=np.zeros((1, 1)),
            summary={},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
        )
        assert result.seed == 42

    def test_default_sampled_indices_is_none(self):
        """Default sampled_indices is None."""
        result = PriorPredictiveResult(
            y_samples=np.zeros((1, 1)),
            summary={},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
        )
        assert result.sampled_indices is None
