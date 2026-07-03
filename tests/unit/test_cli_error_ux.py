"""CLI error-UX coverage: compare --metrics, bad --config paths, actionable
"no trained model" errors, and the --no-progress flag plumbing."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.models.bayes.fit import resolve_progress_bar
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

runner = CliRunner()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_pipeline_mocks(monkeypatch, exit_code: int = 0):
    captured: dict[str, object] = {}

    def fake_config(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
    monkeypatch.setattr(
        "panelcast.pipelines.orchestrator.run_pipeline", lambda config: exit_code
    )
    return captured


class TestCompareMetricsOption:
    """compare --metrics plumbs a custom metrics.json into the comparison."""

    def _mock_comparison(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_comparison(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                table=SimpleNamespace(to_string=lambda index: "TABLE"), artifacts=[]
            )

        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.run_baseline_comparison", fake_comparison
        )
        return captured

    def test_metrics_default(self, monkeypatch):
        captured = self._mock_comparison(monkeypatch)
        result = runner.invoke(app, ["compare", "--baselines"])
        assert result.exit_code == 0
        # None defers resolution to run_baseline_comparison (latest run or flat fallback).
        assert captured["metrics_path"] is None

    def test_metrics_custom_path(self, monkeypatch):
        captured = self._mock_comparison(monkeypatch)
        result = runner.invoke(
            app, ["compare", "--baselines", "--metrics", "outputs/old_run/metrics.json"]
        )
        assert result.exit_code == 0
        assert captured["metrics_path"] == Path("outputs/old_run/metrics.json")

    def test_metrics_in_help(self):
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--metrics" in strip_ansi(result.output)


class TestConfigFileNotFound:
    """A bad --config path is a clean parameter error, not a traceback."""

    def test_run_missing_config(self):
        result = runner.invoke(app, ["run", "--config", "does/not/exist.yaml"])
        output = strip_ansi(result.output)
        assert result.exit_code == 2
        assert "Config file not found" in output
        assert "does/not/exist.yaml" in output
        assert "Traceback" not in output

    def test_stage_missing_config(self):
        result = runner.invoke(app, ["stage", "data", "--config", "does/not/exist.yaml"])
        output = strip_ansi(result.output)
        assert result.exit_code == 2
        assert "Config file not found" in output
        assert "does/not/exist.yaml" in output
        assert "Traceback" not in output

    def test_run_missing_second_config(self, tmp_path):
        """The error names the missing file even when other layers exist."""
        good = tmp_path / "good.yaml"
        good.write_text("seed: 1\n", encoding="utf-8")
        result = runner.invoke(
            app, ["run", "--config", str(good), "--config", "missing.yaml"]
        )
        output = strip_ansi(result.output)
        assert result.exit_code == 2
        assert "missing.yaml" in output


class TestNoTrainedModelMessages:
    """'No trained model' errors name the real model_dir and a remediation."""

    def test_predict_message_names_dir_and_hint(self, monkeypatch, tmp_path):
        import pytest

        from panelcast.pipelines import predict_next

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            predict_next,
            "load_training_summary",
            lambda path: SimpleNamespace(to_json_dict=lambda: {}),
        )
        monkeypatch.setattr(predict_next, "load_manifest", lambda model_dir: None)
        ctx = SimpleNamespace(seed=42)
        with pytest.raises(ValueError, match="No trained user_score model") as exc:
            predict_next.predict_next_events(ctx)
        message = str(exc.value)
        assert "manifest.json" in message
        assert "panelcast stage train" in message

    def test_evaluate_message_names_dir_and_hint(self, monkeypatch, tmp_path):
        import pytest

        from panelcast.pipelines import evaluate

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(evaluate, "load_manifest", lambda model_dir: None)
        ctx = SimpleNamespace(seed=42)
        with pytest.raises(ValueError, match="No trained user_score model") as exc:
            evaluate.evaluate_models(ctx)
        message = str(exc.value)
        assert "manifest.json" in message
        assert "panelcast stage train" in message


class TestResolveProgressBar:
    """None means auto (stderr TTY); explicit values pass through."""

    def test_none_uses_tty_detection(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", SimpleNamespace(isatty=lambda: True))
        assert resolve_progress_bar(None) is True
        monkeypatch.setattr(sys, "stderr", SimpleNamespace(isatty=lambda: False))
        assert resolve_progress_bar(None) is False

    def test_explicit_values_pass_through(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", SimpleNamespace(isatty=lambda: True))
        assert resolve_progress_bar(False) is False
        monkeypatch.setattr(sys, "stderr", SimpleNamespace(isatty=lambda: False))
        assert resolve_progress_bar(True) is True


class TestNoProgressFlag:
    """--no-progress wiring: CLI -> PipelineConfig -> StageContext."""

    def test_run_default_is_auto(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["progress_bar"] is None

    def test_run_no_progress_sets_false(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--no-progress"])
        assert result.exit_code == 0
        assert captured["kwargs"]["progress_bar"] is False

    def test_stage_train_no_progress(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "train", "--no-progress"])
        assert result.exit_code == 0
        assert captured["kwargs"]["progress_bar"] is False

    def test_stage_sensitivity_no_progress(self, monkeypatch):
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "sensitivity", "--no-progress"])
        assert result.exit_code == 0
        assert captured["kwargs"]["progress_bar"] is False

    def test_yaml_progress_bar_key(self, monkeypatch, tmp_path):
        captured = _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text("progress_bar: false\n", encoding="utf-8")
        result = runner.invoke(app, ["run", "--config", str(cfg)])
        assert result.exit_code == 0
        assert captured["kwargs"]["progress_bar"] is False

    def test_explicit_no_progress_beats_yaml(self, monkeypatch, tmp_path):
        captured = _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text("progress_bar: true\n", encoding="utf-8")
        result = runner.invoke(app, ["run", "--no-progress", "--config", str(cfg)])
        assert result.exit_code == 0
        assert captured["kwargs"]["progress_bar"] is False

    def test_stage_context_receives_progress_bar(self, tmp_path):
        config = PipelineConfig(progress_bar=False)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        ctx = orchestrator._create_stage_context()
        assert ctx.progress_bar is False

    def test_command_string_records_no_progress(self, tmp_path):
        orchestrator = PipelineOrchestrator(
            PipelineConfig(progress_bar=False), output_base=tmp_path
        )
        assert "--no-progress" in orchestrator._build_command_string()
        orchestrator = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        assert "--no-progress" not in orchestrator._build_command_string()

    def test_skip_and_resume_treatment(self):
        # Execution mechanics: never invalidates skip detection, never restored
        # on resume.
        assert "progress_bar" in PipelineOrchestrator.SKIP_FLAG_IGNORE
        assert "progress_bar" not in PipelineOrchestrator.RESUME_CONFIG_KEYS


class TestSensitivityProgressThreading:
    """The sensitivity fit helpers forward progress_bar to fit_model."""

    def _patch_fit(self, monkeypatch):
        from panelcast.pipelines import sensitivity

        captured: dict[str, object] = {}

        def fake_fit(model, args, config=None, progress_bar=True):
            captured["progress_bar"] = progress_bar
            return SimpleNamespace(idata="idata")

        monkeypatch.setattr(sensitivity, "fit_model", fake_fit)
        monkeypatch.setattr(
            sensitivity,
            "check_convergence",
            lambda idata, allow_divergences: SimpleNamespace(
                passed=True, rhat_max=1.0, divergences=0
            ),
        )
        monkeypatch.setattr(
            sensitivity,
            "extract_coefficient_summary",
            lambda idata, var_names=None: pd.DataFrame(),
        )
        return captured

    def test_prior_sensitivity_forwards_progress_bar(self, monkeypatch):
        from panelcast.pipelines import sensitivity

        captured = self._patch_fit(monkeypatch)
        sensitivity.run_prior_sensitivity(
            model=lambda: None,
            model_args={},
            configs={"default": SimpleNamespace()},
            compute_loo_cv=False,
            progress_bar=False,
        )
        assert captured["progress_bar"] is False

    def test_feature_ablation_forwards_progress_bar(self, monkeypatch):
        import numpy as np

        from panelcast.pipelines import sensitivity

        captured = self._patch_fit(monkeypatch)
        sensitivity.run_feature_ablation(
            model=lambda: None,
            model_args={"X": np.zeros((3, 2))},
            feature_groups={},
            compute_loo_cv=False,
            progress_bar=False,
        )
        assert captured["progress_bar"] is False
