"""Ranking metrics (#182): top-K quality, rank distributions, calibration."""

from __future__ import annotations

import numpy as np
import pytest

from panelcast.evaluation.ranking import compute_ranking_metrics


def _slate(n=30, n_draws=200, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    latent = np.linspace(90, 60, n)
    y_true = latent + rng.normal(0, 2, size=n)
    samples = latent[None, :] + rng.normal(0, noise, size=(n_draws, n))
    entities = np.array([f"artist_{i}" for i in range(n)], dtype=object)
    return y_true, samples, entities


class TestComputeRankingMetrics:
    def test_good_model_scores_high(self):
        y_true, samples, entities = _slate()
        metrics, slate = compute_ranking_metrics(y_true, samples, entities, ks=(5, 10))
        assert metrics["spearman"] > 0.8
        assert metrics["kendall_tau"] > 0.6
        assert metrics["top_k"]["5"]["precision"] >= 0.6
        assert len(slate) == len(y_true)

    def test_p_topk_mass_sums_to_k(self):
        y_true, samples, entities = _slate()
        _, slate = compute_ranking_metrics(y_true, samples, entities, ks=(10,))
        # Every draw places exactly 10 rows in the top 10.
        assert slate["p_top10"].sum() == pytest.approx(10.0)
        assert ((slate["p_top10"] >= 0) & (slate["p_top10"] <= 1)).all()

    def test_k_beyond_slate_is_null_not_crash(self):
        y_true, samples, entities = _slate(n=8)
        metrics, slate = compute_ranking_metrics(y_true, samples, entities, ks=(5, 25))
        assert metrics["top_k"]["25"] is None
        assert "p_top25" not in slate.columns
        assert metrics["top_k"]["5"] is not None

    def test_slate_sorted_by_predicted_rank_with_entities(self):
        y_true, samples, entities = _slate()
        _, slate = compute_ranking_metrics(y_true, samples, entities, ks=(5,))
        assert slate["predicted_rank"].tolist() == list(range(1, len(y_true) + 1))
        assert slate.loc[0, "entity"].startswith("artist_")
        assert {"y_true", "pred_mean", "expected_rank", "realized_rank"} <= set(slate.columns)

    def test_ties_break_deterministically(self):
        y_true = np.array([70.0, 70.0, 70.0, 60.0])
        samples = np.tile(np.array([80.0, 80.0, 80.0, 50.0]), (50, 1))
        m1, s1 = compute_ranking_metrics(y_true, samples, ["a", "b", "c", "d"], ks=(2,))
        m2, s2 = compute_ranking_metrics(y_true, samples, ["a", "b", "c", "d"], ks=(2,))
        assert s1["realized_rank"].tolist() == s2["realized_rank"].tolist()
        assert s1["realized_rank"].tolist()[:3] == [1, 2, 3]  # row order breaks ties
        assert "tie_break" in m1 and m1["tie_break"] == m2["tie_break"]

    def test_calibration_block_shape(self):
        y_true, samples, entities = _slate()
        metrics, _ = compute_ranking_metrics(y_true, samples, entities, ks=(10,))
        cal = metrics["top_k"]["10"]["calibration"]
        assert len(cal["predicted_probs"]) == len(cal["observed_freq"]) == len(cal["counts"])
        assert sum(cal["counts"]) == len(y_true)

    def test_empty_slate_raises(self):
        with pytest.raises(ValueError, match="empty slate"):
            compute_ranking_metrics(np.array([]), np.zeros((10, 0)), [])
