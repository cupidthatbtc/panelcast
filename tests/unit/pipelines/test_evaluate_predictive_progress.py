"""Regression test for #274: the horizon-clamped primary-eval predictive.

On a large beta_binomial domain the primary-split posterior-predictive ran for
minutes with no log output between the ``primary_eval_horizon_clamped`` warning
and completion, so a slow run was indistinguishable from a hang. The predictive
now emits a per-batch progress heartbeat. This drives the exact clamp path
(``n_rows_over_horizon > 0``) on a tiny beta_binomial panel and asserts the
predictive both COMPLETES and reports progress for every posterior batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from panelcast.pipelines.evaluate import (
    _prepare_test_model_args,
    _run_known_artist_predictive,
)


@pytest.fixture
def beta_binomial_summary() -> dict:
    return {
        "artist_to_idx": {"Artist_A": 0, "Artist_B": 1},
        "n_artists": 2,
        "max_seq": 2,
        "max_albums": 10,
        "min_albums_filter": 1,
        "global_mean_score": 0.25,
        "global_std_score": 0.05,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "feature_cols": ["feat_1", "feat_2"],
        },
        "dataset": {
            "entity_col": "Artist",
            "event_col": "Album",
            "target_col": "BA",
            "n_obs_col": "AB",
            "model_prefix": "user",
            "target_bounds": (0.0, 1.0),
        },
        "priors": {
            "likelihood_family": "beta_binomial",
            "target_transform": "identity",
            "rho_scale": 0.02,
        },
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "likelihood_df": 4.0,
        "n_ref": None,
        "target_transform": "identity",
    }


def _clamped_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A panel where Artist_A's later test rows extend past max_seq_train=2."""
    train_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B", "Artist_B"],
            "BA": [0.25, 0.27, 0.30, 0.28],
            "AB": [400, 420, 380, 360],
            "Release_Date_Parsed": pd.to_datetime(
                ["2017-01-01", "2018-01-01", "2017-01-01", "2018-01-01"]
            ),
        }
    )
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B"],
            "BA": [0.31, 0.24, 0.29],
            "AB": [410, 300, 350],
            "Release_Date_Parsed": pd.to_datetime(
                ["2019-01-01", "2020-01-01", "2019-01-01"]
            ),
        }
    )
    test_features = pd.DataFrame(
        {"feat_1": [1.0, 1.2, 0.8], "feat_2": [2.0, 1.8, 2.2], "n_reviews": [410, 300, 350]},
        index=test_df.index,
    )
    return train_df, test_df, test_features


def test_clamped_primary_predictive_completes_and_reports_progress(beta_binomial_summary):
    train_df, test_df, test_features = _clamped_panel()

    with capture_logs() as prep_logs:
        model_args, y_true, _ = _prepare_test_model_args(
            test_df, test_features, beta_binomial_summary, train_df=train_df, strict=False
        )
    # The clamp path is what wedged in #274 — make sure this panel exercises it.
    assert any(e.get("event") == "primary_eval_horizon_clamped" for e in prep_logs)
    assert int(model_args["album_seq"].max()) <= beta_binomial_summary["max_seq"]

    n_draws, batch_size = 7, 3
    rng = np.random.default_rng(0)
    posterior_samples = {
        "user_bb_phi": np.full(n_draws, 20.0, dtype=np.float32),
        "user_beta": rng.normal(scale=0.1, size=(n_draws, 2)).astype(np.float32),
    }

    with capture_logs() as pred_logs:
        y_samples = _run_known_artist_predictive(
            posterior_samples,
            model_args,
            prefix="user",
            batch_size=batch_size,
            progress_label="primary",
        )

    assert y_samples.shape == (n_draws, len(y_true))
    assert np.all(np.isfinite(y_samples))

    # The heartbeat that makes slow-vs-hung decidable: one event per batch.
    expected_batches = (n_draws + batch_size - 1) // batch_size
    progress = [e for e in pred_logs if e.get("event") == "predictive_progress"]
    assert len(progress) == expected_batches
    assert any(e.get("event") == "predictive_start" for e in pred_logs)
    assert [e["batch"] for e in progress] == list(range(1, expected_batches + 1))
