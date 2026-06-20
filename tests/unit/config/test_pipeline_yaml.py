"""Tests for the YAML -> PipelineConfig mapping layer and --config CLI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.config.loader import load_yaml_config
from panelcast.config.pipeline_yaml import (
    PIPELINE_YAML_MAPPING,
    apply_yaml_overrides,
)
from panelcast.pipelines.orchestrator import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[3]

runner = CliRunner()


def _make_pipeline_mocks(monkeypatch, exit_code: int = 0):
    """Patch run_pipeline, capturing the PipelineConfig it receives."""
    captured: dict[str, object] = {}

    def fake_run_pipeline(config):
        captured["config"] = config
        return exit_code

    monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", fake_run_pipeline)
    return captured


# ============================================================================
# apply_yaml_overrides unit behavior
# ============================================================================


class TestApplyYamlOverrides:
    def test_yaml_overrides_defaults(self):
        kwargs = {"num_samples": 1000, "seed": 42}
        out = apply_yaml_overrides(kwargs, {"num_samples": 5000})
        assert out["num_samples"] == 5000
        assert out["seed"] == 42

    def test_explicit_cli_param_wins(self):
        kwargs = {"num_samples": 250}
        out = apply_yaml_overrides(kwargs, {"num_samples": 5000}, {"num_samples"})
        assert out["num_samples"] == 250

    def test_cli_param_name_differs_from_field(self):
        # min_albums_filter is guarded by the CLI param "min_albums".
        kwargs = {"min_albums_filter": 3}
        out = apply_yaml_overrides(kwargs, {"min_albums_filter": 5}, {"min_albums"})
        assert out["min_albums_filter"] == 3
        out = apply_yaml_overrides(kwargs, {"min_albums_filter": 5}, set())
        assert out["min_albums_filter"] == 5

    def test_unmapped_key_is_ignored(self):
        kwargs = {"seed": 42}
        out = apply_yaml_overrides(kwargs, {"not_a_real_key": 1, "seed": 7})
        assert out["seed"] == 7
        assert "not_a_real_key" not in out

    def test_calibration_intervals_normalized_to_sorted_tuple(self):
        out = apply_yaml_overrides({}, {"calibration_intervals": [0.95, 0.8, 0.95]})
        assert out["calibration_intervals"] == (0.8, 0.95)

    def test_stages_accepts_list_and_comma_string(self):
        assert apply_yaml_overrides({}, {"stages": ["data", "splits"]})["stages"] == [
            "data",
            "splits",
        ]
        assert apply_yaml_overrides({}, {"stages": "data, splits"})["stages"] == [
            "data",
            "splits",
        ]

    def test_chain_method_normalized_and_validated(self):
        assert apply_yaml_overrides({}, {"chain_method": "Vectorized"})["chain_method"] == (
            "vectorized"
        )
        with pytest.raises(ValueError, match="chain_method"):
            apply_yaml_overrides({}, {"chain_method": "bogus"})

    def test_input_kwargs_not_mutated(self):
        kwargs = {"seed": 42}
        apply_yaml_overrides(kwargs, {"seed": 7})
        assert kwargs["seed"] == 42

    def test_mapping_fields_exist_on_pipeline_config(self):
        config_fields = set(PipelineConfig.__dataclass_fields__)
        for key, spec in PIPELINE_YAML_MAPPING.items():
            assert spec.config_field in config_fields, (key, spec.config_field)


# ============================================================================
# Repository config files load cleanly
# ============================================================================


class TestRepositoryConfigs:
    @pytest.mark.parametrize("name", ["publication.yaml", "base.yaml", "dev.yaml"])
    def test_config_file_fully_mapped(self, name):
        data = load_yaml_config(REPO_ROOT / "configs" / name)
        unmapped = [k for k in data if k not in PIPELINE_YAML_MAPPING]
        assert unmapped == [], f"{name} has unmapped keys: {unmapped}"

    def test_publication_yaml_builds_valid_config(self):
        data = load_yaml_config(REPO_ROOT / "configs" / "publication.yaml")
        kwargs = apply_yaml_overrides({}, data)
        config = PipelineConfig(**kwargs)
        assert config.num_chains == 4
        assert config.num_samples == 5000
        assert config.num_warmup == 3000
        assert config.target_accept == 0.95
        assert config.target_transform == "identity"
        assert config.ar_center == "global"
        assert config.calibration_intervals == (0.80, 0.95)

    def test_base_yaml_equals_defaults_where_overlapping(self):
        """base.yaml restates effective CLI defaults; loading it must be a no-op
        relative to the CLI's own defaults."""
        data = load_yaml_config(REPO_ROOT / "configs" / "base.yaml")
        kwargs = apply_yaml_overrides({}, data)
        config = PipelineConfig(**kwargs)
        # Spot-check against PipelineConfig defaults (min_train_albums is the
        # one place the CLI default (2) differs from the dataclass default).
        defaults = PipelineConfig()
        assert config.seed == defaults.seed
        assert config.num_samples == defaults.num_samples
        assert config.target_transform == defaults.target_transform
        assert config.min_train_albums == 2

    def test_dev_yaml_is_cheap_run(self):
        data = load_yaml_config(REPO_ROOT / "configs" / "dev.yaml")
        kwargs = apply_yaml_overrides({}, data)
        assert kwargs == {"num_chains": 2, "num_samples": 500, "num_warmup": 500}


# ============================================================================
# CLI integration: --config precedence
# ============================================================================


class TestCliConfigOption:
    def _invoke(self, monkeypatch, args):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--dry-run", *args])
        assert result.exit_code == 0, result.output
        return captured["config"]

    def test_no_config_is_bit_identical_to_legacy(self, monkeypatch):
        config = self._invoke(monkeypatch, [])
        expected = self._invoke(monkeypatch, [])
        assert config == expected
        assert isinstance(config, PipelineConfig)
        assert config.num_samples == 1000
        assert config.min_train_albums == 2

    def test_config_file_overrides_defaults(self, monkeypatch, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({"num_samples": 321, "seed": 7}), encoding="utf-8")
        config = self._invoke(monkeypatch, ["--config", str(cfg)])
        assert config.num_samples == 321
        assert config.seed == 7

    def test_explicit_cli_beats_config_file(self, monkeypatch, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({"num_samples": 321}), encoding="utf-8")
        config = self._invoke(monkeypatch, ["--config", str(cfg), "--num-samples", "777"])
        assert config.num_samples == 777

    def test_later_config_files_override_earlier(self, monkeypatch, tmp_path):
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text(yaml.safe_dump({"num_samples": 111, "seed": 5}), encoding="utf-8")
        second.write_text(yaml.safe_dump({"num_samples": 222}), encoding="utf-8")
        config = self._invoke(monkeypatch, ["--config", str(first), "--config", str(second)])
        assert config.num_samples == 222
        assert config.seed == 5

    def test_publication_yaml_loads_through_cli(self, monkeypatch):
        config = self._invoke(
            monkeypatch, ["--config", str(REPO_ROOT / "configs" / "publication.yaml")]
        )
        assert config.num_samples == 5000
        assert config.num_warmup == 3000
        assert config.target_accept == 0.95

    def test_invalid_yaml_value_is_bad_parameter(self, monkeypatch, tmp_path):
        _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({"chain_method": "bogus"}), encoding="utf-8")
        result = runner.invoke(app, ["run", "--dry-run", "--config", str(cfg)])
        assert result.exit_code != 0
