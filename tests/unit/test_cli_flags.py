"""CLI flag -> PipelineConfig dispatch matrix (pure CLI coverage).

Each test invokes ``run --dry-run`` with run_pipeline mocked and asserts the
flag lands on the right PipelineConfig field. Covers the gate flags added in
the descriptor/statistics phases plus their defaults.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.pipelines.orchestrator import PipelineConfig

runner = CliRunner()


@pytest.fixture
def captured_config(monkeypatch):
    captured: dict = {}

    def fake_run_pipeline(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", fake_run_pipeline)
    return captured


def _invoke(captured: dict, *args: str) -> PipelineConfig:
    result = runner.invoke(app, ["run", "--dry-run", *args])
    assert result.exit_code == 0, result.output
    return captured["config"]


class TestGateFlagDefaults:
    def test_defaults(self, captured_config):
        config = _invoke(captured_config)
        assert config.target_transform == "identity"
        assert config.ar_center == "global"
        assert config.latent_process == "rw"
        assert config.debut_prev_score_source == "train_mean"
        assert config.exclude_rw_raw_from_collection is False
        assert config.dataset is None


class TestGateFlagDispatch:
    def test_target_transform(self, captured_config):
        config = _invoke(captured_config, "--target-transform", "offset_logit")
        assert config.target_transform == "offset_logit"

    def test_ar_center(self, captured_config):
        config = _invoke(captured_config, "--ar-center", "none")
        assert config.ar_center == "none"

    def test_latent_process(self, captured_config):
        config = _invoke(captured_config, "--latent-process", "ar1")
        assert config.latent_process == "ar1"

    def test_debut_prev_score_source(self, captured_config):
        config = _invoke(captured_config, "--debut-prev-score-source", "dataset_stats")
        assert config.debut_prev_score_source == "dataset_stats"

    def test_exclude_rw_raw_from_collection(self, captured_config):
        config = _invoke(captured_config, "--exclude-rw-raw-from-collection")
        assert config.exclude_rw_raw_from_collection is True

    def test_dataset(self, captured_config):
        config = _invoke(captured_config, "--dataset", "aoty_full")
        assert config.dataset == "aoty_full"

    def test_invalid_gate_value_rejected(self, captured_config):
        result = runner.invoke(app, ["run", "--dry-run", "--latent-process", "bogus"])
        assert result.exit_code != 0


class TestMcmcFlagDispatch:
    def test_sampling_flags(self, captured_config):
        config = _invoke(
            captured_config,
            "--num-chains",
            "3",
            "--num-samples",
            "750",
            "--num-warmup",
            "250",
            "--target-accept",
            "0.93",
        )
        assert config.num_chains == 3
        assert config.num_samples == 750
        assert config.num_warmup == 250
        assert config.target_accept == 0.93

    def test_ablation_flags(self, captured_config):
        config = _invoke(captured_config, "--no-genre", "--no-temporal")
        assert config.enable_genre is False
        assert config.enable_temporal is False
        assert config.enable_artist is True

    def test_split_flags(self, captured_config):
        config = _invoke(captured_config, "--val-albums", "1", "--min-train-albums", "3")
        assert config.val_albums == 1
        assert config.min_train_albums == 3


class TestConfigAndCliPrecedence:
    def test_yaml_then_explicit_cli_gate_flag(self, captured_config, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("ar_center: none\ntarget_transform: offset_logit\n", encoding="utf-8")
        config = _invoke(captured_config, "--config", str(cfg), "--ar-center", "artist_running")
        # explicit CLI wins; non-overridden YAML key applies
        assert config.ar_center == "artist_running"
        assert config.target_transform == "offset_logit"
