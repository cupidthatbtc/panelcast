"""Comprehensive CLI coverage tests.

Tests uncovered paths in cli.py including stage subcommand wiring,
flag passthrough to PipelineConfig, exit codes from pipeline failures,
preflight/resume flags, and error message formatting.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from panelcast.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# Helpers for mocking PipelineConfig + run_pipeline
# ---------------------------------------------------------------------------


def _make_pipeline_mocks(monkeypatch, exit_code: int = 0):
    """Patch PipelineConfig and run_pipeline, returning capture dict.

    The captured dict will contain:
      - "kwargs": keyword arguments passed to PipelineConfig
      - "config": the SimpleNamespace config object
    """
    captured: dict[str, object] = {}

    def fake_config(**kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    def fake_run_pipeline(config):
        captured["config"] = config
        return exit_code

    monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
    monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", fake_run_pipeline)
    return captured


# ============================================================================
# Stage Subcommand Wiring
# ============================================================================


class TestStageDataWiring:
    """Tests for stage data subcommand config wiring."""

    def test_stage_data_default_config(self, monkeypatch):
        """Stage data passes correct defaults to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["stages"] == ["data"]
        assert kwargs["verbose"] is False

    def test_stage_data_custom_seed(self, monkeypatch):
        """Stage data passes custom seed through."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data", "--seed", "99"])
        assert result.exit_code == 0
        assert captured["kwargs"]["seed"] == 99

    def test_stage_data_verbose(self, monkeypatch):
        """Stage data passes verbose flag through."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data", "--verbose"])
        assert result.exit_code == 0
        assert captured["kwargs"]["verbose"] is True


class TestStageSplitsWiring:
    """Tests for stage splits subcommand config wiring."""

    def test_stage_splits_default_config(self, monkeypatch):
        """Stage splits passes correct defaults to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "splits"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["stages"] == ["splits"]
        assert kwargs["verbose"] is False

    def test_stage_splits_custom_seed_verbose(self, monkeypatch):
        """Stage splits passes custom seed and verbose through."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "splits", "--seed", "7", "-v"])
        assert result.exit_code == 0
        assert captured["kwargs"]["seed"] == 7
        assert captured["kwargs"]["verbose"] is True


class TestStageFeaturesWiring:
    """Tests for stage features subcommand config wiring."""

    def test_stage_features_default_config(self, monkeypatch):
        """Stage features passes correct defaults to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "features"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["stages"] == ["features"]
        assert kwargs["verbose"] is False


class TestStageTrainWiring:
    """Tests for stage train subcommand config wiring."""

    def test_stage_train_default_config(self, monkeypatch):
        """Stage train passes correct defaults to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "train"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["stages"] == ["train"]
        assert kwargs["strict"] is False
        assert kwargs["verbose"] is False
        assert kwargs["rhat_threshold"] == 1.01
        assert kwargs["ess_threshold"] == 400
        assert kwargs["allow_divergences"] is False

    def test_stage_train_strict_with_thresholds(self, monkeypatch):
        """Stage train passes strict mode and custom thresholds."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            [
                "stage",
                "train",
                "--strict",
                "--rhat-threshold",
                "1.05",
                "--ess-threshold",
                "200",
                "--allow-divergences",
            ],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["strict"] is True
        assert kwargs["rhat_threshold"] == 1.05
        assert kwargs["ess_threshold"] == 200
        assert kwargs["allow_divergences"] is True


class TestStageEvaluateWiring:
    """Tests for stage evaluate subcommand config wiring."""

    def test_stage_evaluate_default_config(self, monkeypatch):
        """Stage evaluate passes correct defaults to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "evaluate"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["stages"] == ["evaluate"]
        assert kwargs["verbose"] is False


class TestStagePredictWiring:
    """Tests for stage predict subcommand config wiring."""

    def test_stage_predict_default_config(self, monkeypatch):
        """Stage predict passes correct defaults to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "predict"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["stages"] == ["predict"]
        assert kwargs["verbose"] is False

    def test_stage_predict_verbose(self, monkeypatch):
        """Stage predict passes verbose flag through."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "predict", "-v"])
        assert result.exit_code == 0
        assert captured["kwargs"]["verbose"] is True


class TestStageReportWiring:
    """Tests for stage report subcommand config wiring (beyond existing tests)."""

    def test_stage_report_nonzero_exit(self, monkeypatch):
        """Stage report propagates non-zero exit code from run_pipeline."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "report"])
        assert result.exit_code == 1


class TestStageDatasetConfigPresetWiring:
    """Stage commands honor --dataset / --config / --preset like `run` (issue 2d)."""

    def test_stage_dataset_passthrough(self, monkeypatch):
        """--dataset on a stage command reaches PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data", "--dataset", "aero"])
        assert result.exit_code == 0
        assert captured["kwargs"]["dataset"] == "aero"
        assert captured["kwargs"]["stages"] == ["data"]

    def test_stage_config_overrides_apply(self, monkeypatch, tmp_path):
        """A --config YAML value overlays onto the stage config."""
        captured = _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text("min_ratings: 25\n", encoding="utf-8")
        result = runner.invoke(app, ["stage", "splits", "--config", str(cfg)])
        assert result.exit_code == 0
        assert captured["kwargs"]["min_ratings"] == 25
        assert captured["kwargs"]["stages"] == ["splits"]

    def test_stage_explicit_flag_beats_config(self, monkeypatch, tmp_path):
        """An explicit stage flag wins over the same key in --config."""
        captured = _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text("seed: 999\n", encoding="utf-8")
        result = runner.invoke(app, ["stage", "data", "--seed", "7", "--config", str(cfg)])
        assert result.exit_code == 0
        assert captured["kwargs"]["seed"] == 7

    def test_stage_config_cannot_redirect_stage(self, monkeypatch, tmp_path):
        """A YAML `stages` key never redirects an explicit `stage <name>`."""
        captured = _make_pipeline_mocks(monkeypatch)
        cfg = tmp_path / "c.yaml"
        cfg.write_text("stages: [train]\n", encoding="utf-8")
        result = runner.invoke(app, ["stage", "data", "--config", str(cfg)])
        assert result.exit_code == 0
        assert captured["kwargs"]["stages"] == ["data"]

    def test_stage_unknown_preset_errors(self, monkeypatch):
        """An unknown --preset exits non-zero with a clear message."""
        _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["stage", "data", "--preset", "bogus"])
        assert result.exit_code == 1
        assert "unknown --preset" in strip_ansi(result.stdout)


# ============================================================================
# Run Command: Full Config Passthrough
# ============================================================================


class TestRunConfigPassthrough:
    """Tests that ALL run command flags pass through to PipelineConfig."""

    def test_default_config_values(self, monkeypatch):
        """Run with no flags produces expected default config."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 42
        assert kwargs["skip_existing"] is False
        assert kwargs["stages"] is None
        assert kwargs["dry_run"] is False
        assert kwargs["strict"] is False
        assert kwargs["enforce_lockfile"] is True
        assert kwargs["verbose"] is False
        assert kwargs["resume"] is None
        assert kwargs["max_albums"] == 50
        assert kwargs["num_chains"] == 4
        assert kwargs["num_samples"] == 1000
        assert kwargs["num_warmup"] == 1000
        assert kwargs["target_accept"] == 0.90
        assert kwargs["max_tree_depth"] == 10
        assert kwargs["chain_method"] == "sequential"
        assert kwargs["rhat_threshold"] == 1.01
        assert kwargs["ess_threshold"] == 400
        assert kwargs["allow_divergences"] is False
        # min_ratings defaults to None at the CLI; the orchestrator resolves it
        # from the descriptor's primary_min_obs (10 for AOTY).
        assert kwargs["min_ratings"] is None
        assert kwargs["min_albums_filter"] == 2
        assert kwargs["enable_genre"] is True
        assert kwargs["enable_artist"] is True
        assert kwargs["enable_temporal"] is True
        assert kwargs["n_exponent"] == 0.0
        assert kwargs["learn_n_exponent"] is False
        assert kwargs["n_exponent_alpha"] == 2.0
        assert kwargs["n_exponent_beta"] == 4.0
        assert kwargs["n_exponent_prior"] == "logit-normal"
        assert kwargs["calibration_intervals"] == (0.80, 0.95)
        assert kwargs["coverage_tolerance"] == 0.03
        assert kwargs["prediction_interval"] == 0.95
        assert kwargs["evaluate_secondary_split"] is True

    def test_mcmc_flags(self, monkeypatch):
        """MCMC-related flags pass through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            [
                "run",
                "--num-chains",
                "8",
                "--num-samples",
                "2000",
                "--num-warmup",
                "500",
                "--target-accept",
                "0.95",
                "--max-tree-depth",
                "12",
                "--chain-method",
                "vectorized",
            ],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["num_chains"] == 8
        assert kwargs["num_samples"] == 2000
        assert kwargs["num_warmup"] == 500
        assert kwargs["target_accept"] == 0.95
        assert kwargs["max_tree_depth"] == 12
        assert kwargs["chain_method"] == "vectorized"

    def test_convergence_thresholds(self, monkeypatch):
        """Convergence threshold flags pass through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            [
                "run",
                "--rhat-threshold",
                "1.05",
                "--ess-threshold",
                "200",
                "--allow-divergences",
            ],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["rhat_threshold"] == 1.05
        assert kwargs["ess_threshold"] == 200
        assert kwargs["allow_divergences"] is True

    def test_data_filtering_flags(self, monkeypatch):
        """Data filtering flags pass through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            [
                "run",
                "--min-ratings",
                "20",
                "--min-albums",
                "5",
                "--max-albums",
                "100",
            ],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["min_ratings"] == 20
        assert kwargs["min_albums_filter"] == 5
        assert kwargs["max_albums"] == 100

    def test_feature_ablation_flags(self, monkeypatch):
        """Feature ablation flags pass through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            ["run", "--no-genre", "--no-artist", "--no-temporal"],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["enable_genre"] is False
        assert kwargs["enable_artist"] is False
        assert kwargs["enable_temporal"] is False

    def test_heteroscedastic_noise_flags(self, monkeypatch):
        """Heteroscedastic noise flags pass through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            [
                "run",
                "--n-exponent",
                "0.5",
                "--learn-n-exponent",
                "--n-exponent-alpha",
                "3.0",
                "--n-exponent-beta",
                "5.0",
                "--n-exponent-prior",
                "beta",
            ],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["n_exponent"] == 0.5
        assert kwargs["learn_n_exponent"] is True
        assert kwargs["n_exponent_alpha"] == 3.0
        assert kwargs["n_exponent_beta"] == 5.0
        assert kwargs["n_exponent_prior"] == "beta"

    def test_calibration_and_evaluation_flags(self, monkeypatch):
        """Calibration and evaluation flags pass through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            [
                "run",
                "--calibration-intervals",
                "0.50,0.80,0.90,0.95",
                "--coverage-tolerance",
                "0.05",
                "--prediction-interval",
                "0.90",
                "--no-secondary-split",
            ],
        )
        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["calibration_intervals"] == (0.50, 0.80, 0.90, 0.95)
        assert kwargs["coverage_tolerance"] == 0.05
        assert kwargs["prediction_interval"] == 0.90
        assert kwargs["evaluate_secondary_split"] is False

    def test_stages_filter_passthrough(self, monkeypatch):
        """Stages filter is parsed and passed as list."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            ["run", "--stages", "data,splits,features"],
        )
        assert result.exit_code == 0
        assert captured["kwargs"]["stages"] == ["data", "splits", "features"]

    def test_stages_filter_with_whitespace(self, monkeypatch):
        """Stages filter strips whitespace around stage names."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(
            app,
            ["run", "--stages", " data , splits "],
        )
        assert result.exit_code == 0
        assert captured["kwargs"]["stages"] == ["data", "splits"]

    def test_skip_existing_flag(self, monkeypatch):
        """Skip-existing flag passes through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--skip-existing"])
        assert result.exit_code == 0
        assert captured["kwargs"]["skip_existing"] is True

    def test_strict_flag(self, monkeypatch):
        """Strict flag passes through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--strict", "--num-chains", "2"])
        assert result.exit_code == 0
        assert captured["kwargs"]["strict"] is True

    def test_allow_unlocked_env_sets_enforce_lockfile_false(self, monkeypatch):
        """--allow-unlocked-env sets enforce_lockfile to False."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--allow-unlocked-env"])
        assert result.exit_code == 0
        assert captured["kwargs"]["enforce_lockfile"] is False

    def test_verbose_flag(self, monkeypatch):
        """Verbose flag passes through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--verbose"])
        assert result.exit_code == 0
        assert captured["kwargs"]["verbose"] is True

    def test_chain_method_parallel(self, monkeypatch):
        """Chain method 'parallel' passes through correctly."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--chain-method", "parallel"])
        assert result.exit_code == 0
        assert captured["kwargs"]["chain_method"] == "parallel"


# ============================================================================
# Exit Code Paths
# ============================================================================


class TestRunExitCodes:
    """Tests for exit code propagation from run_pipeline."""

    def test_run_propagates_nonzero_exit_code(self, monkeypatch):
        """Run command propagates non-zero exit code from pipeline."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1

    def test_run_propagates_exit_code_2(self, monkeypatch):
        """Run command propagates exit code 2 from pipeline."""
        _make_pipeline_mocks(monkeypatch, exit_code=2)
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 2

    def test_stage_data_propagates_nonzero_exit(self, monkeypatch):
        """Stage data propagates non-zero exit code."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "data"])
        assert result.exit_code == 1

    def test_stage_splits_propagates_nonzero_exit(self, monkeypatch):
        """Stage splits propagates non-zero exit code."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "splits"])
        assert result.exit_code == 1

    def test_stage_features_propagates_nonzero_exit(self, monkeypatch):
        """Stage features propagates non-zero exit code."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "features"])
        assert result.exit_code == 1

    def test_stage_train_propagates_nonzero_exit(self, monkeypatch):
        """Stage train propagates non-zero exit code."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "train"])
        assert result.exit_code == 1

    def test_stage_evaluate_propagates_nonzero_exit(self, monkeypatch):
        """Stage evaluate propagates non-zero exit code."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "evaluate"])
        assert result.exit_code == 1

    def test_stage_predict_propagates_nonzero_exit(self, monkeypatch):
        """Stage predict propagates non-zero exit code."""
        _make_pipeline_mocks(monkeypatch, exit_code=1)
        result = runner.invoke(app, ["stage", "predict"])
        assert result.exit_code == 1


# ============================================================================
# Resume Flag Wiring
# ============================================================================


class TestResumeFlag:
    """Tests for --resume flag passthrough."""

    def test_resume_passes_run_id(self, monkeypatch):
        """--resume passes run ID string to PipelineConfig."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--resume", "2026-01-19_143052"])
        assert result.exit_code == 0
        assert captured["kwargs"]["resume"] == "2026-01-19_143052"

    def test_resume_default_is_none(self, monkeypatch):
        """Default resume value is None."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 0
        assert captured["kwargs"]["resume"] is None


# ============================================================================
# Preflight Flag Wiring
# ============================================================================


class TestPreflightFlags:
    """Tests for preflight flag wiring (quick preflight path)."""

    def test_preflight_runs_check_and_continues(self, monkeypatch):
        """--preflight runs memory check then continues to pipeline."""
        captured = _make_pipeline_mocks(monkeypatch)

        # Mock the preflight imports inside the run function
        fake_dimensions = SimpleNamespace(n_observations=500, n_artists=50)
        fake_result = SimpleNamespace(
            status="pass",
            exit_code=0,
        )

        monkeypatch.setattr(
            "panelcast.data.ingest.extract_data_dimensions",
            lambda **kwargs: fake_dimensions,
        )
        monkeypatch.setattr(
            "panelcast.preflight.run_preflight_check",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "panelcast.preflight.render_preflight_result",
            lambda result, verbose, dimensions: None,
        )
        monkeypatch.setattr(
            "panelcast.preflight.PreflightStatus",
            SimpleNamespace(FAIL="fail"),
        )

        result = runner.invoke(app, ["run", "--preflight"])
        assert result.exit_code == 0
        # Pipeline should have been called
        assert "kwargs" in captured

    def test_preflight_only_exits_after_check(self, monkeypatch):
        """--preflight-only exits after memory check without running pipeline."""
        fake_dimensions = SimpleNamespace(n_observations=500, n_artists=50)
        fake_result = SimpleNamespace(
            status="pass",
            exit_code=0,
        )

        monkeypatch.setattr(
            "panelcast.data.ingest.extract_data_dimensions",
            lambda **kwargs: fake_dimensions,
        )
        monkeypatch.setattr(
            "panelcast.preflight.run_preflight_check",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "panelcast.preflight.render_preflight_result",
            lambda result, verbose, dimensions: None,
        )
        monkeypatch.setattr(
            "panelcast.preflight.PreflightStatus",
            SimpleNamespace(FAIL="fail"),
        )

        result = runner.invoke(app, ["run", "--preflight-only"])
        assert result.exit_code == 0

    def test_preflight_fail_aborts_without_force(self, monkeypatch):
        """Preflight failure aborts pipeline without --force-run."""
        fake_dimensions = SimpleNamespace(n_observations=500, n_artists=50)
        fake_result = SimpleNamespace(
            status="fail",
            exit_code=1,
        )

        monkeypatch.setattr(
            "panelcast.data.ingest.extract_data_dimensions",
            lambda **kwargs: fake_dimensions,
        )
        monkeypatch.setattr(
            "panelcast.preflight.run_preflight_check",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "panelcast.preflight.render_preflight_result",
            lambda result, verbose, dimensions: None,
        )
        monkeypatch.setattr(
            "panelcast.preflight.PreflightStatus",
            SimpleNamespace(FAIL="fail"),
        )

        result = runner.invoke(app, ["run", "--preflight"])
        assert result.exit_code == 1
        assert "Aborting" in result.output or "force-run" in result.output

    def test_preflight_fail_continues_with_force_run(self, monkeypatch):
        """Preflight failure continues pipeline with --force-run."""
        captured = _make_pipeline_mocks(monkeypatch)

        fake_dimensions = SimpleNamespace(n_observations=500, n_artists=50)
        fake_result = SimpleNamespace(
            status="fail",
            exit_code=1,
        )

        monkeypatch.setattr(
            "panelcast.data.ingest.extract_data_dimensions",
            lambda **kwargs: fake_dimensions,
        )
        monkeypatch.setattr(
            "panelcast.preflight.run_preflight_check",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "panelcast.preflight.render_preflight_result",
            lambda result, verbose, dimensions: None,
        )
        monkeypatch.setattr(
            "panelcast.preflight.PreflightStatus",
            SimpleNamespace(FAIL="fail"),
        )

        result = runner.invoke(app, ["run", "--preflight", "--force-run"])
        assert result.exit_code == 0
        assert "kwargs" in captured

    def test_preflight_only_fail_returns_exit_code(self, monkeypatch):
        """--preflight-only returns the preflight exit code on failure."""
        fake_dimensions = SimpleNamespace(n_observations=500, n_artists=50)
        fake_result = SimpleNamespace(
            status="fail",
            exit_code=1,
        )

        monkeypatch.setattr(
            "panelcast.data.ingest.extract_data_dimensions",
            lambda **kwargs: fake_dimensions,
        )
        monkeypatch.setattr(
            "panelcast.preflight.run_preflight_check",
            lambda **kwargs: fake_result,
        )
        monkeypatch.setattr(
            "panelcast.preflight.render_preflight_result",
            lambda result, verbose, dimensions: None,
        )
        monkeypatch.setattr(
            "panelcast.preflight.PreflightStatus",
            SimpleNamespace(FAIL="fail"),
        )

        result = runner.invoke(app, ["run", "--preflight-only"])
        assert result.exit_code == 1


# ============================================================================
# Error Message Formatting
# ============================================================================


class TestErrorMessageFormatting:
    """Tests that user-facing errors are clean (no tracebacks)."""

    def test_invalid_chain_method_no_traceback(self):
        """Invalid chain method produces clean error, no traceback."""
        result = runner.invoke(app, ["run", "--chain-method", "badmethod"])
        output = strip_ansi(result.output)
        assert "Invalid --chain-method" in output
        assert "Traceback" not in output

    def test_invalid_n_exponent_prior_no_traceback(self):
        """Invalid n-exponent-prior produces clean error, no traceback."""
        result = runner.invoke(app, ["run", "--n-exponent-prior", "badprior"])
        output = strip_ansi(result.output)
        assert "Invalid --n-exponent-prior" in output
        assert "Traceback" not in output

    def test_invalid_calibration_intervals_no_traceback(self):
        """Invalid calibration intervals produces clean error, no traceback."""
        result = runner.invoke(app, ["run", "--calibration-intervals", "not_a_number"])
        output = strip_ansi(result.output)
        assert result.exit_code != 0
        assert "Traceback" not in output

    def test_pipeline_config_value_error_no_traceback(self, monkeypatch):
        """ValueError from PipelineConfig produces clean error, no traceback."""

        def fake_config(**kwargs):
            raise ValueError("strict mode requires num_chains >= 2")

        monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
        monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", lambda _: 0)

        result = runner.invoke(app, ["run"])
        output = strip_ansi(result.output)
        assert result.exit_code != 0
        assert "Traceback" not in output

    def test_out_of_range_calibration_no_traceback(self):
        """Calibration interval value at boundary produces clean error."""
        result = runner.invoke(app, ["run", "--calibration-intervals", "1.0"])
        output = strip_ansi(result.output)
        assert result.exit_code != 0
        assert "Traceback" not in output


# ============================================================================
# Calibration Intervals Parsing
# ============================================================================


class TestCalibrationIntervalsParsing:
    """Tests for calibration intervals parsing edge cases."""

    def test_duplicate_intervals_deduplicated(self, monkeypatch):
        """Duplicate calibration intervals are deduplicated."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--calibration-intervals", "0.80,0.80,0.95"])
        assert result.exit_code == 0
        assert captured["kwargs"]["calibration_intervals"] == (0.80, 0.95)

    def test_intervals_are_sorted(self, monkeypatch):
        """Calibration intervals are sorted in ascending order."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--calibration-intervals", "0.95,0.50,0.80"])
        assert result.exit_code == 0
        assert captured["kwargs"]["calibration_intervals"] == (0.50, 0.80, 0.95)

    def test_single_interval_accepted(self, monkeypatch):
        """A single calibration interval is accepted."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--calibration-intervals", "0.90"])
        assert result.exit_code == 0
        assert captured["kwargs"]["calibration_intervals"] == (0.90,)


# ============================================================================
# Chain Method Normalization
# ============================================================================


class TestChainMethodNormalization:
    """Tests for chain method case normalization."""

    @pytest.mark.parametrize(
        "input_method,expected",
        [
            ("sequential", "sequential"),
            ("SEQUENTIAL", "sequential"),
            ("Vectorized", "vectorized"),
            ("PARALLEL", "parallel"),
        ],
    )
    def test_chain_method_normalized_to_lowercase(self, monkeypatch, input_method, expected):
        """Chain method is normalized to lowercase before passing to config."""
        captured = _make_pipeline_mocks(monkeypatch)
        result = runner.invoke(app, ["run", "--chain-method", input_method])
        assert result.exit_code == 0
        assert captured["kwargs"]["chain_method"] == expected


# ============================================================================
# Setup Guide
# ============================================================================


class TestSetupGuideOutput:
    """Tests for --setup-guide output content."""

    def test_setup_guide_mentions_url(self):
        """--setup-guide shows the GitHub URL."""
        result = runner.invoke(app, ["--setup-guide"])
        assert result.exit_code == 0
        assert "github.com" in result.output

    def test_setup_guide_mentions_topics(self):
        """--setup-guide mentions key topics like GPU config."""
        result = runner.invoke(app, ["--setup-guide"])
        output = strip_ansi(result.output)
        assert "GPU" in output
        assert "troubleshooting" in output.lower()


# ============================================================================
# Export-Figures Command
# ============================================================================


class TestExportFiguresCommand:
    """Tests for export-figures subcommand."""

    def test_export_figures_help(self):
        """Export-figures command shows help with expected options."""
        result = runner.invoke(app, ["export-figures", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--output" in output
        assert "--formats" in output
        assert "--width" in output
        assert "--height" in output
        assert "--scale" in output
        assert "--run" in output
