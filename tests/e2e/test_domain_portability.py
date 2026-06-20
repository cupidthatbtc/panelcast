"""Domain-portability proof: one descriptor YAML, zero core-code changes.

Three layers:

(a) stages-only — data -> splits -> features on the synthetic aero domain
    (CI; deliberately not marked e2e/slow: this is the portability contract).
(b) tiny-MCMC — the full pipeline through train/evaluate/predict at
    1 chain x 50 draws (slow marker; local/nightly only).
(c) AOTY equivalence — running with ``--dataset aoty_full`` is byte-identical
    to running with no dataset flag at all (CI).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator
from panelcast.utils.hashing import hash_dataframe
from tests.e2e.conftest import create_minimal_dataset
from tests.helpers.aero_data import make_aero_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
AERO_DESCRIPTOR = REPO_ROOT / "configs" / "datasets" / "aero.yaml"
AOTY_DESCRIPTOR = REPO_ROOT / "configs" / "datasets" / "aoty_full.yaml"


def _run_pipeline_in(workdir: Path, output_base: Path, **config_kwargs):
    """Run the orchestrator with cwd=workdir and environment checks mocked."""
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        config = PipelineConfig(seed=42, strict=False, **config_kwargs)
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash",
                warnings=[],
            )
            orchestrator = PipelineOrchestrator(config, output_base=output_base)
            exit_code = orchestrator.run()
    finally:
        os.chdir(cwd)
    return exit_code, orchestrator


def _write_aero_raw(workdir: Path) -> None:
    raw_dir = workdir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    make_aero_dataset(seed=42).to_csv(raw_dir / "test_flights.csv", index=False, encoding="utf-8")


MUSIC_FEATURE_COLUMNS = (
    "user_prior_mean",
    "critic_prior_mean",
    "is_album",
    "is_ep",
    "is_collaboration",
    "num_artists",
    "collab_type_ordinal",
)


# ============================================================================
# (a) Aero stages-only: data -> splits -> features
# ============================================================================


@pytest.fixture(scope="module")
def aero_stages_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("aero_stages")
    _write_aero_raw(tmp)
    exit_code, _ = _run_pipeline_in(
        tmp,
        tmp / "outputs",
        stages=["data", "splits", "features"],
        dataset=str(AERO_DESCRIPTOR),
        # The aero descriptor's primary_min_obs; pin a per-domain pipeline
        # config (or --min-ratings) to the descriptor's primary threshold.
        min_ratings=5,
    )
    assert exit_code == 0, "Aero stages-only pipeline failed"
    return tmp


class TestAeroStagesOnly:
    def test_processed_dataset_named_and_shaped_by_descriptor(self, aero_stages_run):
        processed = aero_stages_run / "data" / "processed" / "perf_minobs_5.parquet"
        assert processed.exists()
        df = pd.read_parquet(processed)
        for col in ("Airframe", "Flight_ID", "Perf_Score", "Sensor_Samples", "Flight_Date_Parsed"):
            assert col in df.columns, f"missing {col}"
        assert "User_Score" not in df.columns
        assert "Artist" not in df.columns
        assert df["Perf_Score"].between(0, 10).all()

    def test_dataset_stats_use_descriptor_target(self, aero_stages_run):
        stats = json.loads(
            (aero_stages_run / "data" / "processed" / "dataset_stats.json").read_text(
                encoding="utf-8"
            )
        )
        assert stats["source_dataset"] == "perf_minobs_5"
        assert 0.0 < stats["global_mean_score"] < 10.0

    def test_temporal_split_orders_within_airframe(self, aero_stages_run):
        split_dir = aero_stages_run / "data" / "splits" / "within_artist_temporal"
        train = pd.read_parquet(split_dir / "train.parquet")
        test = pd.read_parquet(split_dir / "test.parquet")
        assert len(test) > 0
        for airframe, test_rows in test.groupby("Airframe"):
            train_dates = train.loc[train["Airframe"] == airframe, "Flight_Date_Parsed"].dropna()
            if train_dates.empty:
                continue
            assert (
                test_rows["Flight_Date_Parsed"].min() >= train_dates.max()
            ), f"temporal violation for {airframe}"

    def test_features_have_entity_history_and_no_music_blocks(self, aero_stages_run):
        features = pd.read_parquet(
            aero_stages_run
            / "data"
            / "features"
            / "within_artist_temporal"
            / "train_features.parquet"
        )
        for col in (
            "perf_prior_mean",
            "perf_prior_std",
            "perf_prior_count",
            "perf_trajectory",
            "is_debut",
            "album_sequence",
            "n_reviews",
            # core_numeric pass-through of the domain's own covariates,
            # selected purely in the descriptor YAML.
            "Thrust_Margin",
            "Payload_Fraction",
        ):
            assert col in features.columns, f"missing {col}"
        for col in MUSIC_FEATURE_COLUMNS:
            assert col not in features.columns, f"unexpected music-domain column {col}"
        assert not any(c.startswith("genre_pc") for c in features.columns)

    def test_feature_manifest_records_aero_blocks(self, aero_stages_run):
        manifest = json.loads(
            (aero_stages_run / "data" / "features" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["blocks"] == ["temporal", "entity_history", "core_numeric"]
        assert manifest["target_label_leakage_prevention"]["masked_score_columns"] == ["Perf_Score"]


# ============================================================================
# (b) Aero tiny-MCMC full pipeline (slow; local/nightly)
# ============================================================================


@pytest.mark.slow
class TestAeroTinyMcmc:
    @pytest.fixture(scope="class")
    def aero_full_run(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        tmp = tmp_path_factory.mktemp("aero_full")
        _write_aero_raw(tmp)
        exit_code, _ = _run_pipeline_in(
            tmp,
            tmp / "outputs",
            stages=["data", "splits", "features", "train", "evaluate", "predict"],
            dataset=str(AERO_DESCRIPTOR),
            min_ratings=5,
            num_chains=1,
            num_samples=50,
            num_warmup=50,
            allow_divergences=True,
            rhat_threshold=1.1,
            ess_threshold=100,
        )
        assert exit_code == 0, "Aero full pipeline failed"
        return tmp

    def test_training_summary_carries_aero_domain(self, aero_full_run):
        summary_path = aero_full_run / "models" / "training_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset = summary["dataset"]
        assert dataset["model_prefix"] == "perf"
        assert dataset["target_bounds"] == [0.0, 10.0]
        assert dataset["entity_col"] == "Airframe"

    def test_posterior_sites_use_perf_prefix(self, aero_full_run):
        import arviz as az

        nc_files = sorted((aero_full_run / "models").rglob("*.nc"))
        assert nc_files, "no posterior netcdf saved"
        idata = az.from_netcdf(nc_files[0])
        site_names = list(idata.posterior.data_vars)
        assert any(name.startswith("perf_") for name in site_names), site_names
        assert not any(name.startswith("user_") for name in site_names), site_names

    def test_known_airframe_predictions_in_bounds(self, aero_full_run):
        preds = pd.read_csv(
            aero_full_run / "outputs" / "predictions" / "next_album_known_artists.csv"
        )
        assert len(preds) > 0
        # Keyed by airframe names from the aero fixture.
        assert preds["artist"].str.contains("-").all()
        for col in ("pred_mean", "pred_q05", "pred_q95"):
            assert preds[col].between(0.0, 10.0).all(), f"{col} outside aero bounds"
        assert preds["pred_mean"].notna().all()


# ============================================================================
# (c) AOTY equivalence: --dataset aoty_full == no flag, byte-identical
# ============================================================================


def _run_aoty_stages(tmp: Path, dataset: str | None) -> dict[str, str]:
    """Run data->splits->features on the minimal AOTY fixture; hash outputs."""
    create_minimal_dataset(tmp)
    exit_code, _ = _run_pipeline_in(
        tmp,
        tmp / "outputs",
        stages=["data", "splits", "features"],
        dataset=dataset,
    )
    assert exit_code == 0
    hashes: dict[str, str] = {}
    for parquet in sorted((tmp / "data").rglob("*.parquet")):
        rel = parquet.relative_to(tmp / "data").as_posix()
        hashes[rel] = hash_dataframe(pd.read_parquet(parquet))
    return hashes


class TestAotyDescriptorEquivalence:
    def test_aoty_full_descriptor_is_byte_identical_to_defaults(
        self, tmp_path_factory: pytest.TempPathFactory
    ):
        default_run = _run_aoty_stages(tmp_path_factory.mktemp("aoty_default"), dataset=None)
        descriptor_run = _run_aoty_stages(
            tmp_path_factory.mktemp("aoty_descriptor"), dataset=str(AOTY_DESCRIPTOR)
        )
        assert set(default_run) == set(descriptor_run)
        mismatches = {
            rel: (default_run[rel], descriptor_run[rel])
            for rel in default_run
            if default_run[rel] != descriptor_run[rel]
        }
        assert not mismatches, f"--dataset aoty_full diverged from defaults: {mismatches}"
