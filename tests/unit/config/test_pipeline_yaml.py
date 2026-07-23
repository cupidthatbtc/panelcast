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

    def test_unknown_key_is_fatal(self):
        with pytest.raises(ValueError, match="not_a_real_key"):
            apply_yaml_overrides({"seed": 42}, {"not_a_real_key": 1, "seed": 7})

    def test_typo_error_suggests_nearest_key(self):
        with pytest.raises(ValueError, match="did you mean: num_samples"):
            apply_yaml_overrides({}, {"num_sample": 5000})

    def test_unknown_nested_block_is_fatal(self):
        with pytest.raises(ValueError, match="mcmc"):
            apply_yaml_overrides({}, {"mcmc": {"num_samples": 5000}})

    def test_allow_unknown_preserves_keys_and_applies_mapped(self):
        out = apply_yaml_overrides(
            {"seed": 42}, {"not_a_real_key": 1, "seed": 7}, allow_unknown=True
        )
        assert out["seed"] == 7
        assert out["unknown_config_keys"] == {"not_a_real_key": 1}

    def test_allow_unknown_with_no_unknowns_adds_nothing(self):
        out = apply_yaml_overrides({}, {"seed": 7}, allow_unknown=True)
        assert "unknown_config_keys" not in out

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

    def test_prior_and_init_knobs_map_onto_valid_config(self):
        out = apply_yaml_overrides(
            {},
            {
                "sigma_artist_prior_type": "lognormal",
                "artist_effect_param": "zerosum",
                "init_strategy": "median",
            },
        )
        config = PipelineConfig(**out)
        assert config.sigma_artist_prior_type == "lognormal"
        assert config.artist_effect_param == "zerosum"
        assert config.init_strategy == "median"

    def test_prior_and_init_knobs_survive_resolved_roundtrip(self):
        from panelcast.config.pipeline_yaml import (
            dump_resolved_config,
            load_resolved_config,
        )

        config = PipelineConfig(
            sigma_artist_prior_type="lognormal",
            artist_effect_param="zerosum",
            init_strategy="feasible",
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "resolved.yaml"
            p.write_text(dump_resolved_config(config), encoding="utf-8")
            restored = PipelineConfig(**load_resolved_config(p))
        assert restored.sigma_artist_prior_type == "lognormal"
        assert restored.artist_effect_param == "zerosum"
        assert restored.init_strategy == "feasible"

    def test_lognormal_prior_params_map_onto_valid_config(self):
        out = apply_yaml_overrides(
            {},
            {
                "sigma_rw_lognormal_loc": -6.0,
                "sigma_rw_lognormal_sigma": 0.4,
                "sigma_artist_lognormal_loc": -3.7,
                "sigma_artist_lognormal_sigma": 0.5,
            },
        )
        config = PipelineConfig(**out)
        assert config.sigma_rw_lognormal_loc == -6.0
        assert config.sigma_rw_lognormal_sigma == 0.4
        assert config.sigma_artist_lognormal_loc == -3.7
        assert config.sigma_artist_lognormal_sigma == 0.5

    def test_lognormal_prior_params_survive_resolved_roundtrip(self):
        from panelcast.config.pipeline_yaml import (
            dump_resolved_config,
            load_resolved_config,
        )

        config = PipelineConfig(
            sigma_rw_lognormal_loc=-6.0,
            sigma_rw_lognormal_sigma=0.4,
            sigma_artist_lognormal_loc=-3.7,
            sigma_artist_lognormal_sigma=0.5,
        )
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "resolved.yaml"
            p.write_text(dump_resolved_config(config), encoding="utf-8")
            restored = PipelineConfig(**load_resolved_config(p))
        assert restored.sigma_rw_lognormal_loc == -6.0
        assert restored.sigma_rw_lognormal_sigma == 0.4
        assert restored.sigma_artist_lognormal_loc == -3.7
        assert restored.sigma_artist_lognormal_sigma == 0.5

    def test_rho_prior_params_map_onto_valid_config(self):
        out = apply_yaml_overrides(
            {},
            {"rho_loc": 0.2, "rho_scale": 0.02},
        )
        config = PipelineConfig(**out)
        assert config.rho_loc == 0.2
        assert config.rho_scale == 0.02

    def test_rho_prior_params_survive_resolved_roundtrip(self):
        from panelcast.config.pipeline_yaml import (
            dump_resolved_config,
            load_resolved_config,
        )

        config = PipelineConfig(rho_loc=0.2, rho_scale=0.02)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "resolved.yaml"
            p.write_text(dump_resolved_config(config), encoding="utf-8")
            restored = PipelineConfig(**load_resolved_config(p))
        assert restored.rho_loc == 0.2
        assert restored.rho_scale == 0.02

    def test_select_knobs_are_yaml_mapped(self):
        # select writes every knob into arm run-configs; an unmapped knob is
        # silently dropped and the arm fits as a mislabeled reference (#158's
        # impute_missing did exactly this on GPU).
        from panelcast.select.space import KNOBS

        unmapped = [k.name for k in KNOBS if k.name not in PIPELINE_YAML_MAPPING]
        assert unmapped == [], f"select knobs missing from PIPELINE_YAML_MAPPING: {unmapped}"


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
        assert config.num_warmup == 5000
        assert config.target_accept == 0.90
        assert config.target_transform == "offset_logit"
        assert config.ar_center == "global"
        assert config.calibration_intervals == (0.80, 0.95)

    def test_base_yaml_equals_defaults_where_overlapping(self):
        """base.yaml restates effective CLI defaults; loading it must be a no-op
        relative to the CLI's own defaults."""
        data = load_yaml_config(REPO_ROOT / "configs" / "base.yaml")
        kwargs = apply_yaml_overrides({}, data)
        config = PipelineConfig(**kwargs)
        # Spot-check against PipelineConfig defaults (min_train_albums used to
        # diverge between the CLI (2) and the dataclass (1); both are 2 now).
        defaults = PipelineConfig()
        assert config.seed == defaults.seed
        assert config.num_samples == defaults.num_samples
        assert config.target_transform == defaults.target_transform
        assert config.min_train_albums == defaults.min_train_albums == 2

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
        assert config.num_warmup == 5000
        assert config.target_accept == 0.90

    def test_invalid_yaml_value_is_bad_parameter(self, monkeypatch, tmp_path):
        _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({"chain_method": "bogus"}), encoding="utf-8")
        result = runner.invoke(app, ["run", "--dry-run", "--config", str(cfg)])
        assert result.exit_code != 0

    def test_unknown_key_fails_the_cli_with_suggestion(self, monkeypatch, tmp_path):
        _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(yaml.safe_dump({"num_sample": 5000}), encoding="utf-8")
        result = runner.invoke(app, ["run", "--dry-run", "--config", str(cfg)])
        assert result.exit_code != 0
        assert "num_sample" in result.output
        assert "num_samples" in result.output

    def test_unknown_key_from_merged_second_file_fails(self, monkeypatch, tmp_path):
        _make_pipeline_mocks(monkeypatch)
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text(yaml.safe_dump({"num_samples": 111}), encoding="utf-8")
        second.write_text(yaml.safe_dump({"num_sample": 222}), encoding="utf-8")
        result = runner.invoke(
            app, ["run", "--dry-run", "--config", str(first), "--config", str(second)]
        )
        assert result.exit_code != 0
        assert "num_sample" in result.output

    def test_allow_unknown_config_keys_escape(self, monkeypatch, tmp_path):
        captured = _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            yaml.safe_dump({"num_sample": 5000, "seed": 9}), encoding="utf-8"
        )
        result = runner.invoke(
            app,
            ["run", "--dry-run", "--config", str(cfg), "--allow-unknown-config-keys"],
        )
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.seed == 9
        assert config.unknown_config_keys == {"num_sample": 5000}
        assert config.num_samples == 1000  # the typo key was NOT applied


class TestUnknownKeysReachTheManifest:
    def test_setup_records_unknown_config_keys(self, tmp_path, monkeypatch):
        from panelcast.pipelines.orchestrator import PipelineOrchestrator
        from panelcast.utils.git_state import GitState

        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_git_state",
            lambda: GitState(commit="abc", branch="t", dirty=False, untracked_count=0),
        )
        orch = PipelineOrchestrator(
            PipelineConfig(unknown_config_keys={"num_sample": 5000}),
            output_base=tmp_path / "outputs",
        )
        orch._setup_run()
        assert orch.manifest.flags["unknown_config_keys"] == {"num_sample": 5000}

    def test_unknown_keys_do_not_invalidate_skip_detection(self):
        from panelcast.pipelines.manifest import flag_differences
        from panelcast.pipelines.orchestrator import (
            PipelineOrchestrator,
            _get_default_config,
        )

        assert "unknown_config_keys" in PipelineOrchestrator.SKIP_FLAG_IGNORE
        diffs = flag_differences(
            {"seed": 42, "unknown_config_keys": {"num_sample": 5000}},
            {"seed": 42, "unknown_config_keys": {}},
            _get_default_config(),
            ignore=PipelineOrchestrator.SKIP_FLAG_IGNORE,
        )
        assert diffs == []
