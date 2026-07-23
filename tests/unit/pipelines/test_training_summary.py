"""Tests for the typed TrainingSummary contract."""

from __future__ import annotations

import json

import pytest

from panelcast.pipelines.training_summary import (
    SCHEMA_VERSION,
    DatasetSummaryBlock,
    TrainingSummary,
    load_training_summary,
    upgrade_training_summary,
)

# Key order produced by the legacy dict literal in train_bayes.train_models.
LEGACY_KEY_ORDER = [
    "model_type",
    "model_path",
    "mcmc_config",
    "convergence_thresholds",
    "min_albums_filter",
    "n_artists_below_threshold",
    "priors",
    "data_hash",
    "n_observations",
    "n_artists",
    "n_features",
    "feature_scaler",
    "artist_to_idx",
    "max_seq",
    "max_albums",
    "global_mean_score",
    "feature_cols",
    "n_exponent",
    "learn_n_exponent",
    "n_exponent_prior",
    "likelihood_df",
    "n_ref",
    "n_reviews_stats",
    "divergences",
    "divergence_rate",
    "runtime_seconds",
    "diagnostics",
    "heteroscedastic_mode",
]


def _legacy_summary_dict() -> dict:
    """A v0-shaped summary as train_models historically wrote it."""
    return {
        "model_type": "user_score",
        "model_path": "models/user_score_abc.nc",
        "mcmc_config": {"num_chains": 2, "num_samples": 500},
        "convergence_thresholds": {
            "rhat_threshold": 1.01,
            "ess_threshold": 400,
            "allow_divergences": False,
        },
        "min_albums_filter": 2,
        "n_artists_below_threshold": 3,
        "priors": {"beta_scale": 1.0},
        "data_hash": "abc123",
        "n_observations": 100,
        "n_artists": 10,
        "n_features": 5,
        "feature_scaler": {"mean": [0.0], "std": [1.0]},
        "artist_to_idx": {"A": 0},
        "max_seq": 7,
        "max_albums": 50,
        "global_mean_score": 70.3,
        "feature_cols": ["f1"],
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "likelihood_df": 4.0,
        "n_ref": None,
        "n_reviews_stats": {"min": 10, "max": 100, "median": 30, "mean": 35.0},
        "divergences": 0,
        "divergence_rate": 0.0,
        "runtime_seconds": 12.3,
        "diagnostics": {"passed": True, "rhat_max": 1.002},
        "heteroscedastic_mode": {"mode": "homoscedastic"},
    }


class TestKeyOrderRegression:
    def test_legacy_keys_are_ordered_prefix(self):
        """Serialized output must keep historical keys as an ordered prefix."""
        raw = _legacy_summary_dict()
        raw["dataset"] = DatasetSummaryBlock().model_dump()
        summary = TrainingSummary(**raw)
        dumped_keys = list(summary.to_json_dict().keys())
        assert dumped_keys[: len(LEGACY_KEY_ORDER)] == LEGACY_KEY_ORDER
        # New keys append strictly after.
        assert dumped_keys[len(LEGACY_KEY_ORDER) :] == ["schema_version", "dataset"]

    def test_basis_model_provenance_is_retained(self):
        provenance = {
            "split": "within_entity_temporal",
            "curves": {
                "age_curve": {
                    "standardization": {
                        "feature_names": ["age_curve__basis_00"],
                        "feature_indices": [2],
                        "mean": [0.25],
                        "std": [0.75],
                    }
                }
            },
        }
        dumped = TrainingSummary(basis_curves=provenance).to_json_dict()
        assert dumped["basis_curves"] == provenance

    def test_unset_dataset_omitted(self):
        """A summary built without a dataset block must not emit dataset: null."""
        dumped = TrainingSummary(**_legacy_summary_dict()).to_json_dict()
        assert "dataset" not in dumped
        assert dumped["schema_version"] == SCHEMA_VERSION

    def test_round_trips_through_json(self):
        summary = TrainingSummary(**_legacy_summary_dict())
        text = json.dumps(summary.to_json_dict())
        reloaded = TrainingSummary(**json.loads(text))
        assert reloaded.global_mean_score == 70.3
        assert reloaded.schema_version == SCHEMA_VERSION

    def test_extra_keys_preserved(self):
        raw = _legacy_summary_dict()
        raw["some_future_gate"] = "value"
        summary = TrainingSummary(**raw)
        assert summary.to_json_dict()["some_future_gate"] == "value"

    def test_unset_heteroscedastic_mode_omitted(self):
        raw = _legacy_summary_dict()
        del raw["heteroscedastic_mode"]
        dumped = TrainingSummary(**raw).to_json_dict()
        assert "heteroscedastic_mode" not in dumped


class TestLegacyUpgrade:
    def test_v0_upgrades_with_aoty_defaults(self):
        summary = upgrade_training_summary(_legacy_summary_dict())
        assert summary.schema_version == SCHEMA_VERSION
        assert summary.dataset is not None
        assert summary.dataset.model_prefix == "user"
        assert summary.dataset.target_bounds == [0.0, 100.0]

    def test_v1_passes_through_without_upgrade(self):
        raw = _legacy_summary_dict()
        raw["schema_version"] = SCHEMA_VERSION
        raw["dataset"] = DatasetSummaryBlock(name="aero", model_prefix="perf").model_dump()
        summary = upgrade_training_summary(raw)
        assert summary.dataset is not None
        assert summary.dataset.model_prefix == "perf"

    def test_load_from_file(self, tmp_path):
        path = tmp_path / "training_summary.json"
        path.write_text(json.dumps(_legacy_summary_dict()), encoding="utf-8")
        summary = load_training_summary(path)
        assert summary.model_type == "user_score"
        assert summary.dataset is not None
        assert summary.dataset.name == "aoty"

    def test_partial_summary_loads_without_phantom_keys(self):
        """Minimal summaries (test fixtures, old artifacts) must round-trip
        without gaining null keys that would break summary.get defaults."""
        raw = {"artist_to_idx": {"A": 0}, "feature_cols": ["f1"], "global_mean_score": 70.0}
        summary = upgrade_training_summary(raw)
        dumped = summary.to_json_dict()
        assert "n_exponent" not in dumped
        assert dumped["artist_to_idx"] == {"A": 0}
        assert dumped["schema_version"] == SCHEMA_VERSION
        assert dumped["dataset"]["model_prefix"] == "user"
