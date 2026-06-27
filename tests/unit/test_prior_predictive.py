"""Unit tests for prior predictive module."""

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
    return PriorConfig()


@pytest.fixture
def custom_priors():
    return PriorConfig(
        mu_artist_loc=0.5,
        mu_artist_scale=2.0,
        sigma_artist_scale=1.0,
        sigma_rw_scale=0.2,
        rho_loc=0.1,
        rho_scale=0.5,
        beta_loc=0.0,
        beta_scale=2.0,
        sigma_obs_scale=1.5,
    )


@pytest.fixture
def mock_prior_predictive_result():
    return PriorPredictiveResult(
        y_samples=np.random.default_rng(42).normal(50, 15, (100, 50)),
        summary={
            "mean": 50.0,
            "sd": 15.0,
            "q2.5": 20.0,
            "q97.5": 80.0,
            "min": 5.0,
            "max": 95.0,
        },
        reasonable=True,
        bounds=(0, 100),
        fraction_in_bounds=0.95,
        n_samples=100,
        n_obs_original=50,
        max_obs=2000,
        sampled_indices=None,
        seed=42,
    )


class TestPriorPredictiveResultFields:
    def test_prior_predictive_result_fields(self, mock_prior_predictive_result):
        """Dataclass fields are accessible."""
        result = mock_prior_predictive_result
        assert result.y_samples.shape == (100, 50)
        assert isinstance(result.summary, dict)
        assert isinstance(result.reasonable, bool)
        assert result.bounds == (0, 100)
        assert 0 <= result.fraction_in_bounds <= 1
        assert result.n_samples == 100
        assert result.n_obs_original == 50
        assert result.seed == 42


class TestPriorPredictiveSubsampling:
    def test_prior_predictive_subsampling(self):
        """When n_obs > max_obs, arrays should be subsampled."""
        n_obs = 5000
        max_obs = 100
        rng = np.random.default_rng(42)
        sampled_indices = np.sort(rng.choice(n_obs, size=max_obs, replace=False))

        result = PriorPredictiveResult(
            y_samples=np.zeros((10, max_obs)),
            summary={
                "mean": 0.0,
                "sd": 1.0,
                "q2.5": -2.0,
                "q97.5": 2.0,
                "min": -5.0,
                "max": 5.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.5,
            n_samples=10,
            n_obs_original=n_obs,
            max_obs=max_obs,
            sampled_indices=sampled_indices,
            seed=42,
        )
        assert result.n_obs_original == 5000
        assert result.max_obs == 100
        assert result.sampled_indices is not None
        assert len(result.sampled_indices) == max_obs
        assert result.y_samples.shape[1] == max_obs


class TestJustificationUsesActualValues:
    def test_justification_uses_actual_values(self, custom_priors):
        """Non-default PriorConfig -> actual numeric values appear in text."""
        text = generate_prior_justification_text(custom_priors)
        # Custom values should appear, not defaults
        assert "0.5" in text  # mu_artist_loc
        assert "2.0" in text  # mu_artist_scale
        assert "1.0" in text  # sigma_artist_scale
        assert "0.2" in text  # sigma_rw_scale
        assert "0.1" in text  # rho_loc

    def test_justification_default_values(self, default_priors):
        """Default priors produce valid text with default values."""
        text = generate_prior_justification_text(default_priors)
        assert "mu_artist" in text
        assert "sigma_artist" in text
        assert "sigma_rw" in text
        assert "rho" in text
        assert "beta" in text
        assert "sigma_obs" in text


class TestJustificationWithPPC:
    def test_justification_with_ppc(self, default_priors, mock_prior_predictive_result):
        """Mock PriorPredictiveResult -> 'Prior Predictive Check' in text."""
        text = generate_prior_justification_text(
            default_priors,
            prior_predictive_result=mock_prior_predictive_result,
        )
        assert "Prior Predictive Check" in text
        assert "95.0%" in text  # fraction_in_bounds
        assert "n_samples=100" in text


class TestJustificationWithSensitivity:
    def test_justification_with_sensitivity(self, default_priors):
        """Mock sensitivity DataFrame -> 'Sensitivity' in text."""
        sensitivity_df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_scale_x2", "beta_scale_x5"],
                "parameter": ["baseline", "sigma_rw_scale", "beta_scale"],
                "elpd_delta": [0.0, -3.5, -1.2],
                "eligible_for_ranking": [True, True, True],
            }
        )
        text = generate_prior_justification_text(
            default_priors,
            sensitivity_summary=sensitivity_df,
        )
        assert "Sensitivity" in text
        assert "sigma_rw_scale" in text  # most sensitive

    def test_justification_sensitivity_convergence_aware(self, default_priors):
        """Only converged variants referenced in text."""
        sensitivity_df = pd.DataFrame(
            {
                "variant": [
                    "default",
                    "sigma_rw_scale_x2",
                    "beta_scale_x5",
                ],
                "parameter": ["baseline", "sigma_rw_scale", "beta_scale"],
                "elpd_delta": [0.0, -50.0, -1.2],
                "eligible_for_ranking": [
                    True,
                    False,
                    True,
                ],  # sigma_rw FAILED
            }
        )
        text = generate_prior_justification_text(
            default_priors,
            sensitivity_summary=sensitivity_df,
        )
        # sigma_rw_scale has largest delta but is NOT eligible
        # So beta_scale should be cited as most sensitive
        assert "beta_scale" in text
        assert "most sensitive parameter is beta_scale" in text


class TestPriorPredictiveInputValidation:
    def test_rejects_nonpositive_n_samples(self):
        """run_prior_predictive should reject n_samples < 1."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            run_prior_predictive(
                model=lambda **kwargs: None,
                model_args={},
                n_samples=0,
            )

    def test_rejects_nonpositive_max_obs(self):
        """run_prior_predictive should reject max_obs < 1."""
        with pytest.raises(ValueError, match="max_obs must be >= 1"):
            run_prior_predictive(
                model=lambda **kwargs: None,
                model_args={},
                max_obs=0,
            )

    def test_rejects_invalid_score_bounds(self):
        """run_prior_predictive should reject malformed score bounds."""
        with pytest.raises(ValueError, match="score_bounds must be"):
            run_prior_predictive(
                model=lambda **kwargs: None,
                model_args={},
                score_bounds=(100, 0),
            )


# --- from unit/test_prior_predictive_coverage.py ---


@pytest.fixture
def default_priors_coverage():
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

    def test_basic_execution_with_mock(self, monkeypatch, default_priors_coverage):
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
            "priors": default_priors_coverage,
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

    def test_empty_sensitivity_df_no_section(self, default_priors_coverage):
        """Empty DataFrame produces no sensitivity section."""
        empty_df = pd.DataFrame(
            columns=["variant", "parameter", "elpd_delta", "eligible_for_ranking"]
        )
        text = generate_prior_justification_text(
            default_priors_coverage, sensitivity_summary=empty_df
        )
        assert "Sensitivity" not in text

    def test_none_sensitivity_no_section(self, default_priors_coverage):
        """None sensitivity_summary produces no sensitivity section."""
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=None)
        assert "Sensitivity" not in text

    def test_non_dataframe_sensitivity_ignored(self, default_priors_coverage):
        """Non-DataFrame sensitivity_summary is ignored."""
        text = generate_prior_justification_text(
            default_priors_coverage, sensitivity_summary="not a dataframe"
        )
        assert "Sensitivity" not in text

    def test_all_ineligible_no_sensitivity_section(self, default_priors_coverage):
        """All ineligible variants produce no sensitivity section."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -3.0],
                "eligible_for_ranking": [False, False],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "Sensitivity" not in text

    def test_only_baseline_eligible_no_sensitivity_section(self, default_priors_coverage):
        """Only baseline eligible (filtered out) produces no sensitivity section."""
        df = pd.DataFrame(
            {
                "variant": ["default"],
                "parameter": ["baseline"],
                "elpd_delta": [0.0],
                "eligible_for_ranking": [True],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "Sensitivity" not in text

    def test_nan_elpd_delta_dropped(self, default_priors_coverage):
        """Rows with NaN elpd_delta are dropped before finding max."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2", "beta_x5"],
                "parameter": ["baseline", "sigma_rw_scale", "beta_scale"],
                "elpd_delta": [0.0, float("nan"), -2.0],
                "eligible_for_ranking": [True, True, True],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "Sensitivity" in text
        assert "beta_scale" in text

    def test_moderate_sensitivity_label(self, default_priors_coverage):
        """Large |delta_elpd| >= 5 produces 'moderate' sensitivity label."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -10.0],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "moderate" in text

    def test_minimal_sensitivity_label(self, default_priors_coverage):
        """Small |delta_elpd| < 5 produces 'minimal' sensitivity label."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -2.0],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "minimal" in text

    def test_sensitivity_with_missing_variant_column(self, default_priors_coverage):
        """DataFrame without 'variant' column still works."""
        df = pd.DataFrame(
            {
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [0.0, -3.0],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "Sensitivity" in text
        assert "sigma_rw_scale" in text

    def test_all_elpd_delta_nan_no_sensitivity(self, default_priors_coverage):
        """All NaN elpd_delta produces no sensitivity section after dropna."""
        df = pd.DataFrame(
            {
                "variant": ["default", "sigma_rw_x2"],
                "parameter": ["baseline", "sigma_rw_scale"],
                "elpd_delta": [float("nan"), float("nan")],
                "eligible_for_ranking": [True, True],
            }
        )
        text = generate_prior_justification_text(default_priors_coverage, sensitivity_summary=df)
        assert "Sensitivity" not in text


class TestJustificationWithBothPPCAndSensitivity:
    """Tests for justification text with both PPC and sensitivity data."""

    def test_both_sections_present(self, default_priors_coverage):
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
            default_priors_coverage,
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


# --- from unit/test_prior_predictive_expanded.py ---


class TestPriorPredictiveResultDataclass:
    """Tests for PriorPredictiveResult dataclass."""

    def test_y_samples_shape(self):
        y = np.random.default_rng(42).normal(50, 10, (50, 30))
        result = PriorPredictiveResult(
            y_samples=y,
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
            fraction_in_bounds=0.98,
            n_samples=50,
            n_obs_original=30,
            max_obs=2000,
            sampled_indices=None,
            seed=42,
        )
        assert result.y_samples.shape == (50, 30)

    def test_reasonable_true(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=0,
        )
        assert result.reasonable is True

    def test_reasonable_false(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=False,
            bounds=(0, 100),
            fraction_in_bounds=0.2,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=0,
        )
        assert result.reasonable is False

    def test_bounds_tuple(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=0,
        )
        assert result.bounds == (0, 100)
        assert result.bounds[0] == 0
        assert result.bounds[1] == 100

    def test_summary_keys(self):
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={
                "mean": 50.0,
                "sd": 15.0,
                "q2.5": 20.0,
                "q97.5": 80.0,
                "min": 5.0,
                "max": 95.0,
            },
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=0.95,
            n_samples=10,
            n_obs_original=5,
            max_obs=2000,
            sampled_indices=None,
            seed=42,
        )
        assert "mean" in result.summary
        assert "sd" in result.summary
        assert "q2.5" in result.summary
        assert "q97.5" in result.summary

    def test_with_sampled_indices(self):
        indices = np.array([0, 5, 10, 15, 20])
        result = PriorPredictiveResult(
            y_samples=np.zeros((10, 5)),
            summary={"mean": 0.0, "sd": 0.0, "q2.5": 0.0, "q97.5": 0.0, "min": 0.0, "max": 0.0},
            reasonable=True,
            bounds=(0, 100),
            fraction_in_bounds=1.0,
            n_samples=10,
            n_obs_original=100,
            max_obs=5,
            sampled_indices=indices,
            seed=42,
        )
        assert result.sampled_indices is not None
        assert len(result.sampled_indices) == 5


class TestGeneratePriorJustificationText:
    """Extended tests for generate_prior_justification_text."""

    def test_returns_string(self):
        text = generate_prior_justification_text(PriorConfig())
        assert isinstance(text, str)

    def test_contains_all_prior_names(self):
        text = generate_prior_justification_text(PriorConfig())
        for name in ["mu_artist", "sigma_artist", "sigma_rw", "rho", "beta", "sigma_obs"]:
            assert name in text

    def test_custom_priors_reflected(self):
        config = PriorConfig(mu_artist_loc=70.0, mu_artist_scale=5.0)
        text = generate_prior_justification_text(config)
        assert "70.0" in text
        assert "5.0" in text

    def test_no_ppc_section_without_result(self):
        text = generate_prior_justification_text(PriorConfig())
        assert "Prior Predictive Check" not in text

    def test_no_sensitivity_section_without_data(self):
        text = generate_prior_justification_text(PriorConfig())
        assert "Sensitivity" not in text

    def test_length_reasonable(self):
        text = generate_prior_justification_text(PriorConfig())
        # Should be a meaningful length, not empty
        assert len(text) > 100


# --- from unit/test_prior_predictive_new.py ---


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
