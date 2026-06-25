"""Branch-coverage tests for predict_next.py.

Covers lines missed by the primary test file:
- L100  _predict_known_artists: missing feature_scaler raises ValueError
- L128  horizon clamp warning (non-strict)
- L140-141  artist_mean missing features warning + None assignment
- L204  artist_mean fallback zeros when artist_mean_scaled is None
- L207  continue when no valid artists in batch
- L213  transform.forward on prev_score when transform != identity
- L258  transform.inverse on y_pred when transform != identity
- L323-324  _predict_new_artists: transform.forward path
- L416  predict_next_albums: manifest missing model_key raises
- L428-429  predict_next_albums: posterior prefix mismatch raises
- L460-463  predict_next_albums: n_reviews_col detection branches
- L477  predict_next_albums: no n_reviews_col fallback
- L529  predict_next_albums: out-of-bounds warning
- L541  predict_next_albums: wide-interval warning
- L550  predict_next_albums: horizon_clamped count
- L628-681  predict_artist_next: all branches
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.predict_next import (
    _predict_known_artists,
    _predict_new_artists,
    predict_artist_next,
    predict_next_albums,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _predict_known_artists branch tests
# ---------------------------------------------------------------------------


class TestKnownArtistsBranches:
    def test_missing_feature_scaler_raises(self, small_posterior, base_summary, last_album_info, artist_mean_features):
        summary = {**base_summary, "feature_scaler": None}
        with pytest.raises(ValueError, match="feature_scaler"):
            _predict_known_artists(small_posterior, summary, last_album_info, artist_mean_features)

    def test_horizon_clamp_warning_non_strict(self, small_posterior, base_summary, last_album_info, artist_mean_features):
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
            result = _predict_known_artists(small_posterior, summary, info, artist_mean_features, strict=False)

        assert isinstance(result, pd.DataFrame)
        assert any(result["horizon_clamped"])

    def test_missing_artist_mean_cols_warning_and_fallback(self, small_posterior, base_summary, last_album_info):
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
            result = _predict_known_artists(small_posterior, base_summary, last_album_info, partial_mean)

        artist_mean_rows = result[result["scenario"] == "artist_mean"]
        assert len(artist_mean_rows) == 2

    def test_valid_artists_empty_batch_continue(self, small_posterior, base_summary, artist_mean_features):
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
            result = _predict_known_artists(small_posterior, base_summary, empty_info, artist_mean_features)

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
            result = _predict_known_artists(small_posterior, summary, last_album_info, artist_mean_features)

        assert mock_transform.forward.called
        assert mock_transform.inverse.called
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _predict_new_artists branch tests
# ---------------------------------------------------------------------------


class TestNewArtistsBranches:
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
            patch("panelcast.pipelines.predict_next.predict_new_artist", return_value=pred_return),
        ):
            result = _predict_new_artists(small_posterior, summary)

        mock_transform.forward.assert_called()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Helpers for predict_next_albums integration tests
# ---------------------------------------------------------------------------


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
    known_df = pd.DataFrame({
        "artist": ["ArtistA", "ArtistA", "ArtistA"],
        "scenario": ["same", "population_mean", "artist_mean"],
        "pred_mean": [75.0, 74.0, 73.0],
        "pred_std": [5.0, 5.0, 5.0],
        "pred_q05": [65.0, 64.0, 63.0],
        "pred_q25": [70.0, 69.0, 68.0],
        "pred_q50": [75.0, 74.0, 73.0],
        "pred_q75": [80.0, 79.0, 78.0],
        "pred_q95": [85.0, 84.0, 83.0],
        "last_score": [80.0, 80.0, 80.0],
        "n_training_albums": [2, 2, 2],
        "horizon_clamped": [False, False, False],
    })
    new_df = pd.DataFrame({
        "scenario": ["population", "debut_defaults"],
        "pred_mean": [72.0, 70.0],
        "pred_std": [8.0, 9.0],
        "pred_q05": [58.0, 55.0],
        "pred_q25": [66.0, 63.0],
        "pred_q50": [72.0, 70.0],
        "pred_q75": [78.0, 77.0],
        "pred_q95": [86.0, 85.0],
    })
    return known_df, new_df


def _predict_next_albums_stack(stack, base_summary, small_posterior, train_df, train_features, known_df, new_df, *, model_key_in_manifest=True, path_side_effect=None):
    """Enter all standard patches for predict_next_albums into an ExitStack."""
    mock_manifest = MagicMock()
    if model_key_in_manifest:
        mock_manifest.current = {"user_score": "model.nc"}
    else:
        mock_manifest.current = {}

    mock_idata = MagicMock()
    mock_idata.posterior.data_vars = ["user_sigma_obs"]

    mock_summary_obj = MagicMock()
    mock_summary_obj.to_json_dict.return_value = base_summary

    stack.enter_context(patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest))
    stack.enter_context(patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata))
    stack.enter_context(patch("panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj))
    stack.enter_context(patch("panelcast.pipelines.predict_next._extract_posterior_samples", return_value=small_posterior))
    stack.enter_context(patch("panelcast.pipelines.predict_next.join_splits_with_features", side_effect=lambda df, feat, **kw: df.join(feat, how="left")))
    stack.enter_context(patch("panelcast.pipelines.predict_next.pd.read_parquet", side_effect=[train_df, train_features]))
    stack.enter_context(patch("panelcast.pipelines.predict_next._predict_known_artists", return_value=known_df))
    stack.enter_context(patch("panelcast.pipelines.predict_next._predict_new_artists", return_value=new_df))
    stack.enter_context(patch("panelcast.pipelines.predict_next.resolve_split_dir", return_value=MagicMock(__truediv__=lambda s, o: MagicMock())))

    if path_side_effect is not None:
        stack.enter_context(patch("panelcast.pipelines.predict_next.Path", side_effect=path_side_effect))
    else:
        stack.enter_context(patch("panelcast.pipelines.predict_next.Path", side_effect=lambda p: MagicMock()))


# ---------------------------------------------------------------------------
# predict_next_albums guard / branch tests
# ---------------------------------------------------------------------------


class TestPredictNextAlbumsBranches:
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
            _predict_next_albums_stack(stack, base_summary, small_posterior, train_df, train_features, known_df, new_df, model_key_in_manifest=False)
            with pytest.raises(ValueError, match="No trained"):
                predict_next_albums(ctx)

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
            patch("panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj),
            patch("panelcast.pipelines.predict_next.pd.read_parquet", side_effect=[train_df, train_features]),
            patch("panelcast.pipelines.predict_next.Path", side_effect=lambda p: MagicMock()),
        ):
            with pytest.raises(ValueError, match="no sites with expected prefix"):
                predict_next_albums(ctx)

    def test_n_reviews_col_uses_n_obs_col_fallback(self, tmp_path, base_summary, small_posterior):
        # train_df has User_Ratings not n_reviews => L461 branch
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_albums_stack(
                stack, base_summary, small_posterior, train_df, train_features, known_df, new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_albums(ctx)
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
            _predict_next_albums_stack(
                stack, base_summary, small_posterior, train_df, train_features, known_df, new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_albums(ctx)
        assert "known_predictions_path" in result

    def test_n_reviews_col_explicit(self, tmp_path, base_summary, small_posterior):
        # train_df has n_reviews => L459 branch
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
            _predict_next_albums_stack(
                stack, base_summary, small_posterior, train_df, train_features, known_df, new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_albums(ctx)
        assert "known_predictions_path" in result

    def test_out_of_bounds_warning_triggered(self, tmp_path, base_summary, small_posterior):
        # pred_mean > 100 => L529 warning
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        known_df = known_df.copy()
        known_df["pred_mean"] = 110.0
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_albums_stack(
                stack, base_summary, small_posterior, train_df, train_features, known_df, new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_albums(ctx)
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
            _predict_next_albums_stack(
                stack, base_summary, small_posterior, train_df, train_features, known_df, new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_albums(ctx)
        assert "pred_summary" in result

    def test_horizon_clamped_count_in_summary(self, tmp_path, base_summary, small_posterior):
        # known_df with horizon_clamped=True => L550
        train_df, train_features = _make_minimal_dfs()
        known_df, new_df = _make_known_new_dfs()
        known_df = known_df.copy()
        known_df["horizon_clamped"] = True
        ctx = self._ctx()
        with ExitStack() as stack:
            _predict_next_albums_stack(
                stack, base_summary, small_posterior, train_df, train_features, known_df, new_df,
                path_side_effect=lambda p: tmp_path if p == "outputs/predictions" else MagicMock(),
            )
            result = predict_next_albums(ctx)
        assert result["pred_summary"]["n_horizon_clamped_artists"] == 1


# ---------------------------------------------------------------------------
# predict_artist_next (L628-681)
# ---------------------------------------------------------------------------


class TestPredictArtistNext:
    def _make_summary_obj(self, base_summary):
        obj = MagicMock()
        obj.to_json_dict.return_value = base_summary
        return obj

    def _base_stack(self, stack, base_summary, small_posterior, train_df, train_features):
        mock_summary_obj = self._make_summary_obj(base_summary)
        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}
        mock_idata = MagicMock()

        known_df = pd.DataFrame({
            "artist": ["ArtistA"],
            "scenario": ["same"],
            "pred_mean": [75.0],
            "pred_std": [5.0],
            "pred_q05": [65.0],
            "pred_q25": [70.0],
            "pred_q50": [75.0],
            "pred_q75": [80.0],
            "pred_q95": [85.0],
            "last_score": [80.0],
            "n_training_albums": [2],
            "horizon_clamped": [False],
        })

        stack.enter_context(patch("panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj))
        stack.enter_context(patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest))
        stack.enter_context(patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata))
        stack.enter_context(patch("panelcast.pipelines.predict_next._extract_posterior_samples", return_value=small_posterior))
        stack.enter_context(patch("panelcast.pipelines.predict_next.pd.read_parquet", side_effect=[train_df, train_features]))
        stack.enter_context(patch("panelcast.pipelines.predict_next.join_splits_with_features", side_effect=lambda df, feat, **kw: df.join(feat, how="left")))
        stack.enter_context(patch("panelcast.pipelines.predict_next._predict_known_artists", return_value=known_df))
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
            stack.enter_context(patch(
                "panelcast.pipelines.predict_next.resolve_split_dir",
                return_value=MagicMock(__truediv__=lambda s, o: MagicMock()),
            ))
            result = predict_artist_next("ArtistA", models_dir="models", splits_path=None)

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
            result = predict_artist_next("ArtistA", models_dir="models", splits_path="dummy.parquet")

        assert isinstance(result, pd.DataFrame)

    def test_unknown_artist_raises(self, base_summary, small_posterior):
        mock_summary_obj = self._make_summary_obj(base_summary)
        with (
            patch("panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj),
        ):
            with pytest.raises(KeyError, match="is not part of the trained model"):
                predict_artist_next("UnknownArtist", models_dir="models")

    def test_manifest_missing_raises(self, base_summary, small_posterior):
        mock_summary_obj = self._make_summary_obj(base_summary)
        mock_manifest = MagicMock()
        mock_manifest.current = {}
        with (
            patch("panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj),
            patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest),
        ):
            with pytest.raises(ValueError, match="No trained"):
                predict_artist_next("ArtistA", models_dir="models")

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
            patch("panelcast.pipelines.predict_next.load_training_summary", return_value=mock_summary_obj),
            patch("panelcast.pipelines.predict_next.load_manifest", return_value=mock_manifest),
            patch("panelcast.pipelines.predict_next.load_model", return_value=mock_idata),
            patch("panelcast.pipelines.predict_next._extract_posterior_samples", return_value=small_posterior),
            patch("panelcast.pipelines.predict_next.pd.read_parquet", side_effect=[train_df, train_features]),
            patch("panelcast.pipelines.predict_next.join_splits_with_features", side_effect=lambda df, feat, **kw: df.join(feat, how="left")),
        ):
            with pytest.raises(KeyError, match="has no rows"):
                predict_artist_next("ArtistA", models_dir="models", splits_path="dummy.parquet")

    def test_n_reviews_uses_n_obs_col(self, base_summary, small_posterior):
        # train_df has User_Ratings not n_reviews => L673-674
        train_df = pd.DataFrame(
            {"Artist": ["ArtistA", "ArtistA"], "User_Score": [75.0, 80.0], "User_Ratings": [100, 200]},
            index=[0, 1],
        )
        train_features = pd.DataFrame({"feat_1": [1.5, 2.0], "feat_2": [3.0, 4.0]}, index=[0, 1])

        with ExitStack() as stack:
            self._base_stack(stack, base_summary, small_posterior, train_df, train_features)
            result = predict_artist_next("ArtistA", models_dir="models", splits_path="dummy.parquet")

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
            result = predict_artist_next("ArtistA", models_dir="models", splits_path="dummy.parquet")

        assert isinstance(result, pd.DataFrame)
