"""Tests for next-event prediction pipeline.

Exercises real data transformation logic in predict_next.py while only mocking
external dependencies (JAX devices, NumPyro Predictive, file I/O, idata loading).
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.predict_next import (
    SCENARIOS_KNOWN,
    SCENARIOS_NEW,
    _extract_posterior_samples,
    _predict_known_entities,
    _predict_new_entities,
    predict_entity_next,
    predict_next_events,
)


def test_scenario_constants():
    """Scenario name constants are defined correctly."""
    assert SCENARIOS_KNOWN == ["same", "population_mean", "entity_mean"]
    assert SCENARIOS_NEW == ["population", "debut_defaults"]


class TestExtractPosteriorSamples:
    """Tests for _extract_posterior_samples with mock InferenceData."""

    def test_extracts_and_flattens(self):
        """Samples are flattened from (chains, draws, ...) to (n_samples, ...)."""

        class MockDataArray:
            def __init__(self, values):
                self._values = values

            @property
            def values(self):
                return self._values

        class MockPosterior:
            def __init__(self, data_vars_dict):
                self._data_vars = data_vars_dict

            @property
            def data_vars(self):
                return self._data_vars

            def __getitem__(self, key):
                return MockDataArray(self._data_vars[key])

        class MockIData:
            def __init__(self, posterior_dict):
                self.posterior = MockPosterior(posterior_dict)

        mock_data = {
            "user_mu_artist": np.random.randn(2, 3),
            "user_beta": np.random.randn(2, 3, 5),
        }
        idata = MockIData(mock_data)

        result = _extract_posterior_samples(idata)

        assert result["user_mu_artist"].shape == (6,)
        assert result["user_beta"].shape == (6, 5)


@pytest.fixture
def mock_summary():
    """Realistic training summary dict."""
    return {
        "artist_to_idx": {"ArtistA": 0, "ArtistB": 1, "ArtistC": 2},
        "n_artists": 3,
        "max_seq": 5,
        "global_mean_score": 72.5,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {"mean": [1.0, 2.0], "std": [0.5, 1.0]},
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "n_ref": None,
        "n_reviews_stats": {"median": 100, "min": 10, "max": 500, "mean": 150.0},
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
    }


@pytest.fixture
def mock_posterior_samples():
    """Dict of JAX arrays simulating flattened posterior samples (10 total)."""
    rng = np.random.RandomState(42)
    return {
        "user_mu_artist": jnp.array(rng.randn(10).astype(np.float32)),
        "user_sigma_artist": jnp.array(np.abs(rng.randn(10)).astype(np.float32) + 0.1),
        "user_beta": jnp.array(rng.randn(10, 2).astype(np.float32)),
        "user_rho": jnp.array((rng.randn(10) * 0.3).astype(np.float32)),
        "user_sigma_obs": jnp.array(np.abs(rng.randn(10)).astype(np.float32) + 0.1),
        "user_rw_effects": jnp.array((rng.randn(10, 3, 5) * 0.1).astype(np.float32)),
        "user_sigma_rw": jnp.array(np.abs(rng.randn(10) * 0.1).astype(np.float32) + 0.01),
    }


@pytest.fixture
def mock_last_album_info():
    """DataFrame indexed by Artist with last album info."""
    return pd.DataFrame(
        {
            "album_seq": [3, 2, 4],
            "User_Score": [75.0, 80.0, 65.0],
            "feat_1": [1.5, 2.0, 0.5],
            "feat_2": [3.0, 4.0, 1.0],
            "median_n_reviews": [100, 200, 50],
            "n_albums": [3, 2, 4],
        },
        index=pd.Index(["ArtistA", "ArtistB", "ArtistC"], name="Artist"),
    )


@pytest.fixture
def mock_artist_mean_features():
    """DataFrame indexed by Artist with mean feature values."""
    return pd.DataFrame(
        {
            "feat_1": [1.2, 1.8, 0.6],
            "feat_2": [2.8, 3.5, 1.2],
        },
        index=pd.Index(["ArtistA", "ArtistB", "ArtistC"], name="Artist"),
    )


def _make_jax_mock():
    """Create a mock for the jax module used in predict_next."""
    mock_jax = MagicMock()
    mock_device = MagicMock()
    mock_jax.devices.return_value = [mock_device]
    # Make default_device a context manager
    mock_jax.default_device.return_value.__enter__ = MagicMock(return_value=None)
    mock_jax.default_device.return_value.__exit__ = MagicMock(return_value=False)
    return mock_jax


def _make_predictive_mock(n_samples, n_obs):
    """Create a Predictive mock returning random predictions."""
    mock_predictive_cls = MagicMock()
    rng = np.random.RandomState(99)
    mock_predictive_instance = MagicMock()
    mock_predictive_instance.return_value = {
        "user_y": rng.randn(n_samples, n_obs).astype(np.float32)
    }
    mock_predictive_cls.return_value = mock_predictive_instance
    return mock_predictive_cls


class TestPredictKnownEntities:
    """Tests for _predict_known_entities exercising real data transformations."""

    def _run(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
        *,
        predictive_mock=None,
        strict=False,
    ):
        """Helper to call _predict_known_entities with standard mocking."""
        n_artists = len(
            [a for a in mock_summary["artist_to_idx"] if a in mock_last_album_info.index]
        )
        n_samples = next(iter(mock_posterior_samples.values())).shape[0]

        mock_jax = _make_jax_mock()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        if predictive_mock is None:
            predictive_mock = _make_predictive_mock(n_samples, n_artists)

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", predictive_mock),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            return _predict_known_entities(
                mock_posterior_samples,
                mock_summary,
                mock_last_album_info,
                mock_artist_mean_features,
                strict=strict,
            )

    def test_returns_dataframe(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """Result is a DataFrame with expected columns."""
        result = self._run(
            mock_posterior_samples,
            mock_summary,
            mock_last_album_info,
            mock_artist_mean_features,
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "entity",
            "scenario",
            "pred_mean",
            "pred_std",
            "pred_q05",
            "pred_q25",
            "pred_q50",
            "pred_q75",
            "pred_q95",
            "last_score",
            "n_training_events",
            "horizon_clamped",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_three_scenarios(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """Result has 3 * n_artists rows (3 scenarios x 3 artists = 9 rows)."""
        result = self._run(
            mock_posterior_samples,
            mock_summary,
            mock_last_album_info,
            mock_artist_mean_features,
        )
        assert len(result) == 9
        assert set(result["scenario"].unique()) == set(SCENARIOS_KNOWN)

    def test_feature_standardization(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """For 'same' scenario, features are z-scored using training scaler."""
        mock_jax = _make_jax_mock()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        # Capture call args from Predictive
        predictive_mock = _make_predictive_mock(10, 3)
        call_args_list = []
        original_return = predictive_mock.return_value

        def capture_call(*args, **kwargs):
            call_args_list.append(kwargs)
            return original_return.return_value

        original_return.side_effect = capture_call

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", predictive_mock),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            _predict_known_entities(
                mock_posterior_samples,
                mock_summary,
                mock_last_album_info,
                mock_artist_mean_features,
            )

        # First call is "same" scenario.
        # ArtistA: feat_1=1.5, scaler mean=1.0, std=0.5 => (1.5-1.0)/0.5 = 1.0
        # ArtistA: feat_2=3.0, scaler mean=2.0, std=1.0 => (3.0-2.0)/1.0 = 1.0
        same_args = call_args_list[0]
        X = same_args["X"]
        np.testing.assert_allclose(X[0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(X[0, 1], 1.0, atol=1e-5)

    def test_population_mean_zeros(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """For 'population_mean' scenario, X is all zeros."""
        mock_jax = _make_jax_mock()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        predictive_mock = _make_predictive_mock(10, 3)
        call_args_list = []
        original_return = predictive_mock.return_value

        def capture_call(*args, **kwargs):
            call_args_list.append(kwargs)
            return original_return.return_value

        original_return.side_effect = capture_call

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", predictive_mock),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            _predict_known_entities(
                mock_posterior_samples,
                mock_summary,
                mock_last_album_info,
                mock_artist_mean_features,
            )

        # Second call is "population_mean" scenario.
        pop_mean_args = call_args_list[1]
        X = pop_mean_args["X"]
        np.testing.assert_allclose(X, 0.0, atol=1e-7)

    def test_min_albums_filter_clamps_sequence_to_static_effect(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """Artists below training-album threshold should predict at sequence 1."""
        summary = dict(mock_summary)
        summary["min_albums_filter"] = 2

        last_album_info = mock_last_album_info.copy()
        last_album_info.loc["ArtistB", "n_albums"] = 1

        mock_jax = _make_jax_mock()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        predictive_mock = _make_predictive_mock(10, 3)
        call_args_list = []
        original_return = predictive_mock.return_value

        def capture_call(*args, **kwargs):
            call_args_list.append(kwargs)
            return original_return.return_value

        original_return.side_effect = capture_call

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", predictive_mock),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            _predict_known_entities(
                mock_posterior_samples,
                summary,
                last_album_info,
                mock_artist_mean_features,
            )

        # First call is "same" scenario with artists in order A, B, C.
        same_args = call_args_list[0]
        album_seq = same_args["album_seq"]
        assert int(album_seq[1]) == 1

    def test_propagate_rw_horizon_keeps_deep_sequence(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """propagate_rw_horizon keeps an artist's true next sequence instead of
        clamping at max_seq, while below-threshold artists still fall to seq 1."""
        summary = dict(mock_summary)
        summary["max_seq"] = 3  # ArtistA (seq 3) and ArtistC (seq 4) exceed the horizon
        summary["min_albums_filter"] = 2
        summary["priors"] = {**mock_summary["priors"], "propagate_rw_horizon": True}

        last_album_info = mock_last_album_info.copy()
        last_album_info.loc["ArtistA", "n_albums"] = 1  # below threshold -> static seq 1

        mock_jax = _make_jax_mock()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        predictive_mock = _make_predictive_mock(10, 3)
        call_args_list = []
        original_return = predictive_mock.return_value

        def capture_call(*args, **kwargs):
            call_args_list.append(kwargs)
            return original_return.return_value

        original_return.side_effect = capture_call

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", predictive_mock),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            _predict_known_entities(
                mock_posterior_samples,
                summary,
                last_album_info,
                mock_artist_mean_features,
            )

        album_seq = call_args_list[0]["album_seq"]
        assert int(album_seq[0]) == 1  # ArtistA below threshold -> static effect
        assert int(album_seq[2]) == 5  # ArtistC kept at 4 + 1, not clamped to max_seq=3

    def test_eiv_injects_prev_meas_sigma(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """errors_in_variables injects a finite, positive prev_meas_sigma into the
        known-entity model args, scaled by each entity's last-album review count."""
        summary = dict(mock_summary)
        summary["global_std_score"] = 8.0
        summary["priors"] = {**mock_summary["priors"], "errors_in_variables": True}

        mock_jax = _make_jax_mock()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        predictive_mock = _make_predictive_mock(10, 3)
        call_args_list = []
        original_return = predictive_mock.return_value

        def capture_call(*args, **kwargs):
            call_args_list.append(kwargs)
            return original_return.return_value

        original_return.side_effect = capture_call

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", predictive_mock),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            _predict_known_entities(
                mock_posterior_samples,
                summary,
                mock_last_album_info,
                mock_artist_mean_features,
            )

        sigma = call_args_list[0]["prev_meas_sigma"]
        assert sigma.shape[0] == 3
        assert np.isfinite(sigma).all()
        assert (sigma > 0).all()

    def test_eiv_missing_global_std_warns(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """A legacy summary without global_std_score yields a zero scale (EIV no-op)
        rather than failing; the call site logs the legacy-summary warning."""
        summary = dict(mock_summary)
        summary.pop("global_std_score", None)
        summary["priors"] = {**mock_summary["priors"], "errors_in_variables": True}

        result = self._run(
            mock_posterior_samples,
            summary,
            mock_last_album_info,
            mock_artist_mean_features,
        )
        assert not result.empty

    def test_skips_missing_artists(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """Artists in artist_to_idx but NOT in last_album_info are skipped."""
        summary = dict(mock_summary)
        summary["artist_to_idx"] = {
            **mock_summary["artist_to_idx"],
            "ArtistD": 3,
        }
        summary["n_artists"] = 4

        result = self._run(
            mock_posterior_samples,
            summary,
            mock_last_album_info,
            mock_artist_mean_features,
        )
        # Only 3 artists in last_album_info => 9 rows (3 artists x 3 scenarios)
        assert len(result) == 9
        assert "ArtistD" not in result["entity"].values

    def test_strict_raises_on_horizon_clamp(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """Strict mode should fail when next_seq exceeds training max_seq."""
        summary = dict(mock_summary)
        summary["max_seq"] = 2

        with pytest.raises(ValueError, match="extrapolation beyond training sequence horizon"):
            self._run(
                mock_posterior_samples,
                summary,
                mock_last_album_info,
                mock_artist_mean_features,
                strict=True,
            )

    def test_below_threshold_entity_excluded_from_horizon_clamp(
        self,
        mock_posterior_samples,
        mock_summary,
        mock_last_album_info,
        mock_artist_mean_features,
    ):
        """A below-min-albums entity is pinned to seq 1 (static effect) and never
        extrapolates, so it must not count as horizon-clamped: no strict failure
        and no per-row flag."""
        summary = dict(mock_summary)
        summary["max_seq"] = 4  # only ArtistC (seq 4 -> 5) would exceed the horizon
        summary["min_albums_filter"] = 2

        last_album_info = mock_last_album_info.copy()
        last_album_info.loc["ArtistC", "n_albums"] = 1  # below threshold

        # Before the fix the sub-threshold entity was counted and strict raised here.
        result = self._run(
            mock_posterior_samples,
            summary,
            last_album_info,
            mock_artist_mean_features,
            strict=True,
        )
        assert not result.loc[result["entity"] == "ArtistC", "horizon_clamped"].any()


class TestPredictNewEntities:
    """Tests for _predict_new_entities with mocked JAX and predict_new_entity."""

    def _run(self, mock_posterior_samples, summary, *, predict_return=None):
        """Helper to call _predict_new_entities with standard mocking."""
        mock_jax = _make_jax_mock()
        if predict_return is None:
            rng = np.random.RandomState(77)
            predict_return = {"y": rng.randn(10).astype(np.float32)}

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch(
                "panelcast.pipelines.predict_next.predict_new_entity",
                return_value=predict_return,
            ) as mock_predict,
        ):
            result = _predict_new_entities(mock_posterior_samples, summary)
            return result, mock_predict

    def test_returns_dataframe(self, mock_posterior_samples, mock_summary):
        """Result is a DataFrame with expected columns and 2 rows."""
        result, _ = self._run(mock_posterior_samples, mock_summary)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        expected_cols = {
            "scenario",
            "pred_mean",
            "pred_std",
            "pred_q05",
            "pred_q25",
            "pred_q50",
            "pred_q75",
            "pred_q95",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_scenarios(self, mock_posterior_samples, mock_summary):
        """Both scenario names are correct."""
        result, _ = self._run(mock_posterior_samples, mock_summary)
        assert list(result["scenario"]) == ["population", "debut_defaults"]

    def test_homoscedastic_no_n_reviews(self, mock_posterior_samples, mock_summary):
        """Homoscedastic (n_exponent=0.0) does not pass n_reviews_new."""
        _, mock_predict = self._run(mock_posterior_samples, mock_summary)
        for call in mock_predict.call_args_list:
            assert "n_reviews_new" not in call.kwargs

    def test_heteroscedastic_passes_n_reviews(self, mock_posterior_samples, mock_summary):
        """Heteroscedastic (n_exponent=0.5) passes n_reviews_new."""
        summary = dict(mock_summary)
        summary["n_exponent"] = 0.5
        _, mock_predict = self._run(mock_posterior_samples, summary)
        for call in mock_predict.call_args_list:
            assert "n_reviews_new" in call.kwargs

    def test_seed_passed_through(self, mock_posterior_samples, mock_summary):
        """Specific seed value is forwarded to predict_new_entity."""
        mock_jax = _make_jax_mock()
        rng = np.random.RandomState(77)
        predict_return = {"y": rng.randn(10).astype(np.float32)}

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch(
                "panelcast.pipelines.predict_next.predict_new_entity",
                return_value=predict_return,
            ) as mock_predict,
        ):
            _predict_new_entities(mock_posterior_samples, mock_summary, seed=99)

        # Both scenario calls should receive seed=99
        for call in mock_predict.call_args_list:
            assert call.kwargs["seed"] == 99

    def test_uses_global_mean_prev_score_for_cold_start(self, mock_posterior_samples, mock_summary):
        """Cold-start scenarios should use training global mean as prev_score baseline."""
        summary = dict(mock_summary)
        summary["global_mean_score"] = 74.25

        _, mock_predict = self._run(mock_posterior_samples, summary)

        for call in mock_predict.call_args_list:
            assert call.kwargs["prev_score"] == pytest.approx(74.25)

    def test_cold_start_honors_trained_likelihood_family(
        self, mock_posterior_samples, mock_summary
    ):
        """Cold-start must predict under the trained family, not the studentt default.

        Regression for the bug where _predict_new_entities never forwarded
        likelihood_family / discretize_observation, so a beta (or any
        non-studentt) model silently predicted new entities under Student-t.
        """
        summary = dict(mock_summary)
        summary["likelihood_family"] = "beta"
        summary["discretize_observation"] = True

        _, mock_predict = self._run(mock_posterior_samples, summary)

        assert mock_predict.call_args_list  # both scenarios called
        for call in mock_predict.call_args_list:
            assert call.kwargs["likelihood_family"] == "beta"
            assert call.kwargs["discretize_observation"] is True

    def test_cold_start_defaults_to_studentt_when_unset(self, mock_posterior_samples, mock_summary):
        """Legacy summaries (no family keys) default to studentt, no discretization."""
        # mock_summary has neither likelihood_family nor discretize_observation.
        _, mock_predict = self._run(mock_posterior_samples, mock_summary)

        for call in mock_predict.call_args_list:
            assert call.kwargs["likelihood_family"] == "studentt"
            assert call.kwargs["discretize_observation"] is False


class TestPredictNextEvents:
    """Integration-level test for predict_next_events with mocked I/O."""

    def test_end_to_end(self, tmp_path, mock_summary, mock_posterior_samples):
        """predict_next_events returns dict with expected keys."""
        # Build mock manifest
        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}

        # Build mock idata
        class MockDataArray:
            def __init__(self, values):
                self._values = values

            @property
            def values(self):
                return self._values

        class MockPosterior:
            def __init__(self, data_vars_dict):
                self._data_vars = data_vars_dict

            @property
            def data_vars(self):
                return self._data_vars

            def __getitem__(self, key):
                return MockDataArray(self._data_vars[key])

        mock_idata = MagicMock()
        rng = np.random.RandomState(42)
        posterior_data = {
            "user_mu_artist": rng.randn(2, 5).astype(np.float32),
            "user_sigma_artist": np.abs(rng.randn(2, 5)).astype(np.float32) + 0.1,
            "user_beta": rng.randn(2, 5, 2).astype(np.float32),
            "user_rho": (rng.randn(2, 5) * 0.3).astype(np.float32),
            "user_sigma_obs": np.abs(rng.randn(2, 5)).astype(np.float32) + 0.1,
            "user_rw_effects": (rng.randn(2, 5, 3, 5) * 0.1).astype(np.float32),
            "user_sigma_rw": np.abs(rng.randn(2, 5) * 0.1).astype(np.float32) + 0.01,
        }
        mock_idata.posterior = MockPosterior(posterior_data)

        # Build mock train data
        train_df = pd.DataFrame(
            {
                "Artist": ["ArtistA", "ArtistA", "ArtistB", "ArtistC"],
                "User_Score": [75.0, 80.0, 65.0, 70.0],
                "User_Ratings": [100, 200, 50, 150],
            }
        )
        train_features = pd.DataFrame(
            {
                "feat_1": [1.5, 2.0, 0.5, 1.0],
                "feat_2": [3.0, 4.0, 1.0, 2.5],
                "n_reviews": [100, 200, 50, 150],
            },
            index=train_df.index,
        )

        # Small DataFrames for known/new predictions
        known_df = pd.DataFrame(
            {
                "entity": ["ArtistA"],
                "scenario": ["same"],
                "pred_mean": [75.0],
                "pred_std": [5.0],
                "pred_q05": [65.0],
                "pred_q25": [70.0],
                "pred_q50": [75.0],
                "pred_q75": [80.0],
                "pred_q95": [85.0],
                "last_score": [80.0],
                "n_training_events": [2],
            }
        )
        new_df = pd.DataFrame(
            {
                "scenario": ["population"],
                "pred_mean": [72.0],
                "pred_std": [8.0],
                "pred_q05": [58.0],
                "pred_q25": [66.0],
                "pred_q50": [72.0],
                "pred_q75": [78.0],
                "pred_q95": [86.0],
            }
        )

        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.seed = 42

        with (
            patch(
                "panelcast.pipelines.predict_next.load_manifest",
                return_value=mock_manifest,
            ),
            patch(
                "panelcast.pipelines.predict_next.load_model",
                return_value=mock_idata,
            ),
            patch(
                "builtins.open",
                MagicMock(
                    return_value=MagicMock(
                        __enter__=MagicMock(return_value=MagicMock()),
                        __exit__=MagicMock(return_value=False),
                    )
                ),
            ),
            patch("json.load", return_value=mock_summary),
            patch(
                "panelcast.pipelines.predict_next._extract_posterior_samples",
                return_value=mock_posterior_samples,
            ),
            patch(
                "panelcast.pipelines.predict_next._predict_known_entities",
                return_value=known_df,
            ),
            patch(
                "panelcast.pipelines.predict_next._predict_new_entities",
                return_value=new_df,
            ),
            patch(
                "panelcast.pipelines.predict_next.pd.read_parquet",
                side_effect=[train_df, train_features],
            ),
            patch("panelcast.pipelines.predict_next.Path") as MockPath,
        ):
            # Make Path("outputs/predictions") return our tmp path
            mock_output_dir = MagicMock()
            mock_output_dir.__truediv__ = lambda self, other: tmp_path / other

            def path_side_effect(p):
                if p == "outputs/predictions":
                    return mock_output_dir
                return MagicMock()

            MockPath.side_effect = path_side_effect
            # But Path("models") also needs to work -- just delegate to MagicMock
            # The load_manifest and load_model are already mocked

            result = predict_next_events(mock_ctx)

        assert isinstance(result, dict)
        assert "known_predictions_path" in result
        assert "new_predictions_path" in result
        assert result["known_predictions_path"].endswith("next_event_known_entities.csv")
        assert result["new_predictions_path"].endswith("next_event_new_entity.csv")
        assert "known_predictions_legacy_path" not in result
        assert "new_predictions_legacy_path" not in result
        assert "summary_path" in result
        assert "pred_summary" in result

    def test_index_alignment_raises_on_mismatch(self, mock_summary, mock_posterior_samples):
        """ValueError raised when train_df and train_features have mismatched indices."""
        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}

        mock_idata = MagicMock()
        # Satisfy the posterior-prefix guard so the join-alignment path runs.
        mock_idata.posterior.data_vars = ["user_sigma_obs"]

        # train_df and train_features with different indices
        train_df = pd.DataFrame(
            {
                "Artist": ["ArtistA", "ArtistB"],
                "User_Score": [75.0, 80.0],
                "User_Ratings": [100, 200],
            },
            index=[0, 1],
        )
        train_features = pd.DataFrame(
            {
                "feat_1": [1.5, 2.0],
                "feat_2": [3.0, 4.0],
                "n_reviews": [100, 200],
            },
            index=[10, 11],  # Mismatched indices
        )

        mock_ctx = MagicMock()
        mock_ctx.seed = 42

        with (
            patch(
                "panelcast.pipelines.predict_next.load_manifest",
                return_value=mock_manifest,
            ),
            patch(
                "panelcast.pipelines.predict_next.load_model",
                return_value=mock_idata,
            ),
            patch("builtins.open", MagicMock()),
            patch("json.load", return_value=mock_summary),
            patch(
                "panelcast.pipelines.predict_next._extract_posterior_samples",
                return_value=mock_posterior_samples,
            ),
            patch(
                "panelcast.pipelines.predict_next.pd.read_parquet",
                side_effect=[train_df, train_features],
            ),
            patch("panelcast.pipelines.predict_next.Path") as MockPath,
        ):
            MockPath.side_effect = lambda p: MagicMock()

            with pytest.raises(ValueError, match="different indices"):
                predict_next_events(mock_ctx)


class TestExtractPosteriorSamplesEdgeCases:
    """Additional tests for _extract_posterior_samples."""

    def test_scalar_parameter(self):
        """Scalar parameters are flattened to 1-D."""

        class MockDataArray:
            def __init__(self, values):
                self._values = values

            @property
            def values(self):
                return self._values

        class MockPosterior:
            def __init__(self, data):
                self._data = data

            @property
            def data_vars(self):
                return self._data

            def __getitem__(self, key):
                return MockDataArray(self._data[key])

        class MockIData:
            def __init__(self, data):
                self.posterior = MockPosterior(data)

        # Scalar param: shape (chains=2, draws=3)
        idata = MockIData({"user_sigma_obs": np.ones((2, 3))})
        result = _extract_posterior_samples(idata)
        assert result["user_sigma_obs"].shape == (6,)

    def test_extracts_all_params(self):
        """All posterior data_vars are extracted."""

        class MockDataArray:
            def __init__(self, values):
                self._values = values

            @property
            def values(self):
                return self._values

        class MockPosterior:
            def __init__(self, data):
                self._data = data

            @property
            def data_vars(self):
                return self._data

            def __getitem__(self, key):
                return MockDataArray(self._data[key])

        class MockIData:
            def __init__(self, data):
                self.posterior = MockPosterior(data)

        data = {
            "user_beta": np.ones((1, 5, 2)),
            "other_param": np.ones((1, 5)),
        }
        idata = MockIData(data)
        result = _extract_posterior_samples(idata)
        assert "user_beta" in result
        assert "other_param" in result
        assert result["other_param"].shape == (5,)


class TestScenarioConstants:
    """Additional tests for scenario constants."""

    def test_known_scenarios_are_strings(self):
        """All SCENARIOS_KNOWN are strings."""
        for s in SCENARIOS_KNOWN:
            assert isinstance(s, str)

    def test_new_scenarios_are_strings(self):
        """All SCENARIOS_NEW are strings."""
        for s in SCENARIOS_NEW:
            assert isinstance(s, str)

    def test_no_overlap(self):
        """Known and new scenarios do not overlap."""
        assert not set(SCENARIOS_KNOWN) & set(SCENARIOS_NEW)


# --- from unit/pipelines/test_predict_next_branches.py ---


@pytest.fixture
def base_summary():
    return {
        "artist_to_idx": {"ArtistA": 0, "ArtistB": 1},
        "n_artists": 2,
        "max_seq": 5,
        "global_mean_score": 72.5,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {"mean": [1.0, 2.0], "std": [0.5, 1.0]},
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "n_ref": None,
        "n_reviews_stats": {"median": 100, "min": 10, "max": 500, "mean": 150.0},
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
    }


@pytest.fixture
def small_posterior():
    rng = np.random.RandomState(0)
    return {
        "user_mu_artist": jnp.array(rng.randn(6).astype(np.float32)),
        "user_sigma_artist": jnp.array(np.abs(rng.randn(6)).astype(np.float32) + 0.1),
        "user_beta": jnp.array(rng.randn(6, 2).astype(np.float32)),
        "user_rho": jnp.array((rng.randn(6) * 0.3).astype(np.float32)),
        "user_sigma_obs": jnp.array(np.abs(rng.randn(6)).astype(np.float32) + 0.1),
        "user_rw_effects": jnp.array((rng.randn(6, 2, 5) * 0.1).astype(np.float32)),
        "user_sigma_rw": jnp.array(np.abs(rng.randn(6) * 0.1).astype(np.float32) + 0.01),
    }


@pytest.fixture
def last_album_info():
    return pd.DataFrame(
        {
            "album_seq": [2, 3],
            "User_Score": [75.0, 80.0],
            "feat_1": [1.5, 2.0],
            "feat_2": [3.0, 4.0],
            "median_n_reviews": [100, 200],
            "n_albums": [2, 3],
        },
        index=pd.Index(["ArtistA", "ArtistB"], name="Artist"),
    )


@pytest.fixture
def artist_mean_features():
    return pd.DataFrame(
        {"feat_1": [1.2, 1.8], "feat_2": [2.8, 3.5]},
        index=pd.Index(["ArtistA", "ArtistB"], name="Artist"),
    )


def _mock_jax():
    m = MagicMock()
    m.devices.return_value = [MagicMock()]
    m.default_device.return_value.__enter__ = MagicMock(return_value=None)
    m.default_device.return_value.__exit__ = MagicMock(return_value=False)
    return m


def _mock_predictive(n_samples, n_obs):
    cls = MagicMock()
    inst = MagicMock()
    rng = np.random.RandomState(7)
    inst.return_value = {"user_y": rng.randn(n_samples, n_obs).astype(np.float32)}
    cls.return_value = inst
    return cls


class TestKnownEntitiesBranches:
    def test_missing_feature_scaler_raises(
        self, small_posterior, base_summary, last_album_info, artist_mean_features
    ):
        summary = {**base_summary, "feature_scaler": None}
        with pytest.raises(ValueError, match="feature_scaler"):
            _predict_known_entities(small_posterior, summary, last_album_info, artist_mean_features)

    def test_horizon_clamp_warning_non_strict(
        self, small_posterior, base_summary, last_album_info, artist_mean_features
    ):
        # album_seq=5, max_seq=5 => next_seq=6 > 5 => clamped
        info = last_album_info.copy()
        info.loc["ArtistA", "album_seq"] = 5
        summary = {**base_summary, "max_seq": 5}

        mock_jax = _mock_jax()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()
        pred_cls = _mock_predictive(6, 2)

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", pred_cls),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            result = _predict_known_entities(
                small_posterior, summary, info, artist_mean_features, strict=False
            )

        assert isinstance(result, pd.DataFrame)
        assert any(result["horizon_clamped"])

    def test_missing_artist_mean_cols_warning_and_fallback(
        self, small_posterior, base_summary, last_album_info
    ):
        # artist_mean_features missing feat_2 => L140-141 warning + None, L204 zeros fallback
        partial_mean = pd.DataFrame(
            {"feat_1": [1.2, 1.8]},
            index=pd.Index(["ArtistA", "ArtistB"], name="Artist"),
        )

        mock_jax = _mock_jax()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()
        pred_cls = _mock_predictive(6, 2)

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", pred_cls),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            result = _predict_known_entities(
                small_posterior, base_summary, last_album_info, partial_mean
            )

        entity_mean_rows = result[result["scenario"] == "entity_mean"]
        assert len(entity_mean_rows) == 2

    def test_valid_artists_empty_batch_continue(
        self, small_posterior, base_summary, artist_mean_features
    ):
        # All artists absent from last_album_info => no valid_artists => L207 continue
        empty_info = pd.DataFrame(
            columns=["album_seq", "User_Score", "feat_1", "feat_2", "median_n_reviews", "n_albums"],
            index=pd.Index([], name="Artist"),
        )

        mock_jax = _mock_jax()
        mock_random = MagicMock()
        pred_cls = _mock_predictive(6, 0)

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", pred_cls),
            patch("panelcast.pipelines.predict_next.random", mock_random),
        ):
            result = _predict_known_entities(
                small_posterior, base_summary, empty_info, artist_mean_features
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_non_identity_transform_forward_and_inverse(
        self, small_posterior, base_summary, last_album_info, artist_mean_features
    ):
        # Uses a mocked transform with name != "identity" to hit L213 and L258
        summary = {
            **base_summary,
            "target_transform": "logit",
            "dataset": {"target_bounds": [0.0, 100.0], "model_prefix": "user"},
        }

        mock_jax = _mock_jax()
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()
        pred_cls = _mock_predictive(6, 2)

        mock_transform = MagicMock()
        mock_transform.name = "logit"
        mock_transform.forward.return_value = np.array([0.5, 0.6], dtype=np.float32)
        mock_transform.inverse.return_value = np.ones((6, 2), dtype=np.float32) * 50.0

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.Predictive", pred_cls),
            patch("panelcast.pipelines.predict_next.random", mock_random),
            patch("panelcast.pipelines.predict_next.get_transform", return_value=mock_transform),
        ):
            result = _predict_known_entities(
                small_posterior, summary, last_album_info, artist_mean_features
            )

        assert mock_transform.forward.called
        assert mock_transform.inverse.called
        assert isinstance(result, pd.DataFrame)


class TestNewEntitiesBranches:
    def test_non_identity_transform_in_new_artists(self, small_posterior, base_summary):
        # target_transform != "identity" => L323-324
        summary = {
            **base_summary,
            "target_transform": "logit",
            "dataset": {"target_bounds": [0.0, 100.0]},
        }

        mock_jax = _mock_jax()
        mock_transform = MagicMock()
        mock_transform.name = "logit"
        mock_transform.forward.return_value = 0.5

        rng = np.random.RandomState(1)
        pred_return = {"y": rng.randn(6).astype(np.float32)}

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch("panelcast.pipelines.predict_next.get_transform", return_value=mock_transform),
            patch("panelcast.pipelines.predict_next.predict_new_entity", return_value=pred_return),
        ):
            result = _predict_new_entities(small_posterior, summary)

        mock_transform.forward.assert_called()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


def _make_minimal_dfs():
    train_df = pd.DataFrame(
        {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0], "User_Ratings": [100, 200]},
        index=[0, 1],
    )
    train_features = pd.DataFrame(
        {"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]},
        index=[0, 1],
    )
    return train_df, train_features


def _make_known_new_dfs():
    known_df = pd.DataFrame(
        {
            "entity": ["ArtistA", "ArtistA", "ArtistA"],
            "scenario": ["same", "population_mean", "entity_mean"],
            "pred_mean": [75.0, 74.0, 73.0],
            "pred_std": [5.0, 5.0, 5.0],
            "pred_q05": [65.0, 64.0, 63.0],
            "pred_q25": [70.0, 69.0, 68.0],
            "pred_q50": [75.0, 74.0, 73.0],
            "pred_q75": [80.0, 79.0, 78.0],
            "pred_q95": [85.0, 84.0, 83.0],
            "last_score": [80.0, 80.0, 80.0],
            "n_training_events": [2, 2, 2],
            "horizon_clamped": [False, False, False],
        }
    )
    new_df = pd.DataFrame(
        {
            "scenario": ["population", "debut_defaults"],
            "pred_mean": [72.0, 70.0],
            "pred_std": [8.0, 9.0],
            "pred_q05": [58.0, 55.0],
            "pred_q25": [66.0, 63.0],
            "pred_q50": [72.0, 70.0],
            "pred_q75": [78.0, 77.0],
            "pred_q95": [86.0, 85.0],
        }
    )
    return known_df, new_df


def _predict_next_events_stack(
    stack,
    base_summary,
    small_posterior,
    train_df,
    train_features,
    known_df,
    new_df,
    *,
    model_key_in_manifest=True,
    path_side_effect=None,
):
    """Enter all standard patches for predict_next_events into an ExitStack."""
    mock_manifest = MagicMock()
    if model_key_in_manifest:
        mock_manifest.current = {"user_score": "model.nc"}
    else:
        mock_manifest.current = {}

    mock_idata = MagicMock()
    mock_idata.posterior.data_vars = ["user_sigma_obs"]

    mock_summary_obj = MagicMock()
    mock_summary_obj.to_json_dict.return_value = base_summary

    stack.enter_context(
        patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest)
    )
    stack.enter_context(
        patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata)
    )
    stack.enter_context(
        patch(
            "panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj
        )
    )
    stack.enter_context(
        patch(
            "panelcast.pipelines.predict_next._extract_posterior_samples",
            return_value=small_posterior,
        )
    )
    stack.enter_context(
        patch(
            "panelcast.pipelines.predict_next.join_splits_with_features",
            side_effect=lambda df, feat, **kw: df.join(feat, how="left"),
        )
    )
    stack.enter_context(
        patch(
            "panelcast.pipelines.predict_next.pd.read_parquet",
            side_effect=[train_df, train_features],
        )
    )
    known_mock = stack.enter_context(
        patch("panelcast.pipelines.predict_next._predict_known_entities", return_value=known_df)
    )
    new_mock = stack.enter_context(
        patch("panelcast.pipelines.predict_next._predict_new_entities", return_value=new_df)
    )
    stack.enter_context(
        patch(
            "panelcast.pipelines.predict_next.resolve_split_dir",
            return_value=MagicMock(__truediv__=lambda s, o: MagicMock()),
        )
    )

    if path_side_effect is not None:
        stack.enter_context(
            patch("panelcast.pipelines.predict_next.Path", side_effect=path_side_effect)
        )
    else:
        stack.enter_context(
            patch("panelcast.pipelines.predict_next.Path", side_effect=lambda p: MagicMock())
        )

    # Hand back the prediction-helper mocks so callers can assert on the
    # last_album_info / summary that the upstream column logic actually built.
    return known_mock, new_mock


class TestPredictNextEventsBranches:
    def _ctx(self):
        ctx = MagicMock()
        ctx.seed = 1
        ctx.strict = False
        ctx.predictive_batch_size = 500
        ctx.predict_artist_batch_size = 50
        return ctx

    def test_manifest_missing_model_key_raises(self, tmp_path, base_summary, small_posterior):
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                model_key_in_manifest=False,
            )
            with pytest.raises(ValueError, match="No trained"):
                predict_next_events(ctx)

    def test_posterior_prefix_mismatch_raises(self, tmp_path, base_summary, small_posterior):
        train_df, train_features = _make_minimal_dfs()
        ctx = self._ctx()

        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}
        mock_idata = MagicMock()
        mock_idata.posterior.data_vars = ["other_sigma_obs"]
        mock_summary_obj = MagicMock()
        mock_summary_obj.to_json_dict.return_value = base_summary

        with (
            patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest),
            patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata),
            patch(
                "panelcast.pipelines.predict_next.load_training_summary",
                return_value=mock_summary_obj,
            ),
            patch(
                "panelcast.pipelines.predict_next.pd.read_parquet",
                side_effect=[train_df, train_features],
            ),
            patch("panelcast.pipelines.predict_next.Path", side_effect=lambda p: MagicMock()),
        ):
            with pytest.raises(ValueError, match="no sites with expected prefix"):
                predict_next_events(ctx)

    def test_n_reviews_col_uses_n_obs_col_fallback(self, tmp_path, base_summary, small_posterior):
        # train_df has User_Ratings not n_reviews => median review count must be
        # taken from the n_obs_col (User_Ratings: median(100, 200) = 150).
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        ctx = self._ctx()
        with ExitStack() as stack:
            known_mock, _ = _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_events(ctx)
        last_album_info = known_mock.call_args.args[2]
        assert last_album_info.loc["ArtistA", "median_n_reviews"] == 150.0
        assert "known_predictions_path" in result

    def test_no_n_reviews_col_uses_summary_fallback(self, tmp_path, base_summary, small_posterior):
        # train_df has neither n_reviews nor User_Ratings => L463 + L477
        train_df = pd.DataFrame(
            {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0]},
            index=[0, 1],
        )
        train_features = pd.DataFrame(
            {"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]},
            index=[0, 1],
        )
        known_df, new_df = _make_known_new_dfs()
        ctx = self._ctx()
        with ExitStack() as stack:
            known_mock, _ = _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_events(ctx)
        # No review column at all => fall back to the summary's median (100).
        last_album_info = known_mock.call_args.args[2]
        assert last_album_info.loc["ArtistA", "median_n_reviews"] == 100
        assert "known_predictions_path" in result

    def test_n_reviews_col_explicit(self, tmp_path, base_summary, small_posterior):
        # train_df has an explicit n_reviews column => median(100, 200) = 150.
        train_df = pd.DataFrame(
            {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0], "n_reviews": [100, 200]},
            index=[0, 1],
        )
        train_features = pd.DataFrame(
            {"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]},
            index=[0, 1],
        )
        known_df, new_df = _make_known_new_dfs()
        ctx = self._ctx()
        with ExitStack() as stack:
            known_mock, _ = _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_events(ctx)
        last_album_info = known_mock.call_args.args[2]
        assert last_album_info.loc["ArtistA", "median_n_reviews"] == 150.0
        assert "known_predictions_path" in result

    def test_out_of_bounds_warning_triggered(self, tmp_path, base_summary, small_posterior):
        # pred_mean > 100 => L529 warning
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        known_df = known_df.copy()
        known_df["pred_mean"] = 110.0
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_events(ctx)
        assert "pred_summary" in result

    def test_wide_interval_warning_triggered(self, tmp_path, base_summary, small_posterior):
        # CI width > 80 => L541 warning
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        known_df = known_df.copy()
        known_df["pred_q05"] = 0.0
        known_df["pred_q95"] = 90.0
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_events(ctx)
        assert "pred_summary" in result

    def test_horizon_clamped_count_in_summary(self, tmp_path, base_summary, small_posterior):
        # known_df with horizon_clamped=True => L550
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        known_df = known_df.copy()
        known_df["horizon_clamped"] = True
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_events_stack(
                stack,
                base_summary,
                small_posterior,
                train_df,
                train_features,
                known_df,
                new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_events(ctx)
        assert result["pred_summary"]["n_horizon_clamped_artists"] == 1


class TestPredictEntityNext:
    def _make_summary_obj(self, base_summary):
        obj = MagicMock()
        obj.to_json_dict.return_value = base_summary
        return obj

    def _base_stack(self, stack, base_summary, small_posterior, train_df, train_features):
        mock_summary_obj = self._make_summary_obj(base_summary)
        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}
        mock_idata = MagicMock()

        known_df = pd.DataFrame(
            {
                "entity": ["ArtistA"],
                "scenario": ["same"],
                "pred_mean": [75.0],
                "pred_std": [5.0],
                "pred_q05": [65.0],
                "pred_q25": [70.0],
                "pred_q50": [75.0],
                "pred_q75": [80.0],
                "pred_q95": [85.0],
                "last_score": [80.0],
                "n_training_events": [2],
                "horizon_clamped": [False],
            }
        )

        stack.enter_context(
            patch(
                "panelcast.pipelines.predict_next.load_training_summary",
                return_value=mock_summary_obj,
            )
        )
        stack.enter_context(
            patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest)
        )
        stack.enter_context(
            patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata)
        )
        stack.enter_context(
            patch(
                "panelcast.pipelines.predict_next._extract_posterior_samples",
                return_value=small_posterior,
            )
        )
        stack.enter_context(
            patch(
                "panelcast.pipelines.predict_next.pd.read_parquet",
                side_effect=[train_df, train_features],
            )
        )
        stack.enter_context(
            patch(
                "panelcast.pipelines.predict_next.join_splits_with_features",
                side_effect=lambda df, feat, **kw: df.join(feat, how="left"),
            )
        )
        stack.enter_context(
            patch("panelcast.pipelines.predict_next._predict_known_entities", return_value=known_df)
        )
        return known_df

    def test_known_artist_default_splits_path(self, base_summary, small_posterior):
        # splits_path=None => L628-632 resolve_split_dir path
        train_df = pd.DataFrame(
            {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0], "n_reviews": [100, 200]},
            index=[0, 1],
        )
        train_features = pd.DataFrame({"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]}, index=[0, 1])

        with ExitStack() as stack:
            self._base_stack(stack, base_summary, small_posterior, train_df, train_features)
            stack.enter_context(
                patch(
                    "panelcast.pipelines.predict_next.resolve_split_dir",
                    return_value=MagicMock(__truediv__=lambda s, o: MagicMock()),
                )
            )
            result = predict_entity_next("ArtistA", models_dir="models", splits_path=None)

        assert isinstance(result, pd.DataFrame)

    def test_explicit_splits_path(self, base_summary, small_posterior):
        # splits_path provided => L628 skipped
        train_df = pd.DataFrame(
            {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0], "n_reviews": [100, 200]},
            index=[0, 1],
        )
        train_features = pd.DataFrame({"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]}, index=[0, 1])

        with ExitStack() as stack:
            self._base_stack(stack, base_summary, small_posterior, train_df, train_features)
            result = predict_entity_next(
                "ArtistA", models_dir="models", splits_path="dummy.parquet"
            )

        assert isinstance(result, pd.DataFrame)

    def test_unknown_artist_raises(self, base_summary, small_posterior):
        mock_summary_obj = self._make_summary_obj(base_summary)
        with (
            patch(
                "panelcast.pipelines.predict_next.load_training_summary",
                return_value=mock_summary_obj,
            ),
        ):
            with pytest.raises(KeyError, match="is not part of the trained model"):
                predict_entity_next("UnknownArtist", models_dir="models")

    def test_manifest_missing_raises(self, base_summary, small_posterior):
        mock_summary_obj = self._make_summary_obj(base_summary)
        mock_manifest = MagicMock()
        mock_manifest.current = {}
        with (
            patch(
                "panelcast.pipelines.predict_next.load_training_summary",
                return_value=mock_summary_obj,
            ),
            patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest),
        ):
            with pytest.raises(ValueError, match="No trained"):
                predict_entity_next("ArtistA", models_dir="models")

    def test_empty_train_rows_raises(self, base_summary, small_posterior):
        # ArtistA in summary but training split only has ArtistB => empty filter
        train_df = pd.DataFrame(
            {"Artist": ["ArtistB"], "User_Score": [80.0], "n_reviews": [100]},
            index=[0],
        )
        train_features = pd.DataFrame({"feat_1": [1.5], "feat_2": [3.0]}, index=[0])

        mock_summary_obj = self._make_summary_obj(base_summary)
        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}
        mock_idata = MagicMock()

        with (
            patch(
                "panelcast.pipelines.predict_next.load_training_summary",
                return_value=mock_summary_obj,
            ),
            patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest),
            patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata),
            patch(
                "panelcast.pipelines.predict_next._extract_posterior_samples",
                return_value=small_posterior,
            ),
            patch(
                "panelcast.pipelines.predict_next.pd.read_parquet",
                side_effect=[train_df, train_features],
            ),
            patch(
                "panelcast.pipelines.predict_next.join_splits_with_features",
                side_effect=lambda df, feat, **kw: df.join(feat, how="left"),
            ),
        ):
            with pytest.raises(KeyError, match="has no rows"):
                predict_entity_next("ArtistA", models_dir="models", splits_path="dummy.parquet")

    def test_n_reviews_uses_n_obs_col(self, base_summary, small_posterior):
        # train_df has User_Ratings not n_reviews => L673-674
        train_df = pd.DataFrame(
            {
                "Artist": ["ArtistA", "ArtistA"],
                "User_Score": [75.0, 80.0],
                "User_Ratings": [100, 200],
            },
            index=[0, 1],
        )
        train_features = pd.DataFrame({"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]}, index=[0, 1])

        with ExitStack() as stack:
            self._base_stack(stack, base_summary, small_posterior, train_df, train_features)
            result = predict_entity_next(
                "ArtistA", models_dir="models", splits_path="dummy.parquet"
            )

        assert isinstance(result, pd.DataFrame)

    def test_no_n_reviews_col_uses_summary_median(self, base_summary, small_posterior):
        # train_df has neither => L675-676
        train_df = pd.DataFrame(
            {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0]},
            index=[0, 1],
        )
        train_features = pd.DataFrame({"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]}, index=[0, 1])

        with ExitStack() as stack:
            self._base_stack(stack, base_summary, small_posterior, train_df, train_features)
            result = predict_entity_next(
                "ArtistA", models_dir="models", splits_path="dummy.parquet"
            )

        assert isinstance(result, pd.DataFrame)


class TestGbmOffsetColdStart:
    """The gbm_offset covariate zero-fills for hypothetical entities (#86).

    Zero on the standardized scale is the train-mean offset, so cold-start
    predictions degrade to the population-level effect rather than crashing
    or extrapolating.
    """

    def test_x_new_zero_fills_gbm_offset_slot(self, mock_posterior_samples, mock_summary):
        summary = dict(mock_summary)
        summary["feature_cols"] = [*summary["feature_cols"], "gbm_offset"]
        mock_jax = _make_jax_mock()
        rng = np.random.RandomState(77)
        predict_return = {"y": rng.randn(10).astype(np.float32)}

        with (
            patch("panelcast.pipelines.predict_next.jax", mock_jax),
            patch(
                "panelcast.pipelines.predict_next.predict_new_entity",
                return_value=predict_return,
            ) as mock_predict,
        ):
            _predict_new_entities(mock_posterior_samples, summary)

        for call in mock_predict.call_args_list:
            x_new = np.asarray(call.kwargs["X_new"])
            assert x_new.shape == (3,)
            assert (x_new == 0.0).all()
