"""Tests for CLI pipeline commands.

Tests the CLI entry points for pipeline execution, including the main
'run' command and individual stage subcommands.
"""

import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from panelcast.cli import __version__, app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_main_help_shows_commands(self):
        """Main help shows run and stage commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "stage" in result.stdout

    def test_run_help_shows_all_options(self):
        """Run command help shows all expected options."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

        # Check all flags are documented (strip ANSI codes for CI)
        output = strip_ansi(result.stdout)
        assert "--seed" in output
        assert "--skip-existing" in output
        assert "--stages" in output
        assert "--dry-run" in output
        assert "--strict" in output
        assert "--verbose" in output
        assert "--resume" in output

    def test_stage_help_shows_subcommands(self):
        """Stage help shows all individual stages."""
        result = runner.invoke(app, ["stage", "--help"])
        assert result.exit_code == 0
        assert "data" in result.stdout
        assert "splits" in result.stdout
        assert "features" in result.stdout
        assert "train" in result.stdout
        assert "evaluate" in result.stdout
        assert "report" in result.stdout

    def test_stage_data_help_works(self):
        """Individual stage help works."""
        result = runner.invoke(app, ["stage", "data", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "--seed" in output
        assert "--verbose" in output


class TestCLIVersion:
    """Tests for version flag."""

    def test_version_flag_shows_version(self):
        """--version flag shows version string."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout
        assert "panelcast version" in result.stdout

    def test_version_short_flag(self):
        """Short -V flag also shows version."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestRunCommand:
    """Tests for main run command."""

    def test_run_dry_run_succeeds(self):
        """Run with --dry-run completes without executing stages."""
        result = runner.invoke(app, ["run", "--dry-run"])
        # Exit code 0 means success (or early exit from dry-run)
        assert result.exit_code == 0

    def test_run_dry_run_verbose(self):
        """Run with --dry-run --verbose works."""
        result = runner.invoke(app, ["run", "--dry-run", "--verbose"])
        assert result.exit_code == 0

    def test_run_with_seed(self):
        """Run accepts custom seed."""
        result = runner.invoke(app, ["run", "--dry-run", "--seed", "123"])
        assert result.exit_code == 0

    def test_run_with_stages_filter(self):
        """Run accepts stages filter."""
        result = runner.invoke(app, ["run", "--dry-run", "--stages", "data,splits"])
        assert result.exit_code == 0


class TestStageCommands:
    """Tests for individual stage commands."""

    @pytest.mark.parametrize(
        "stage_name",
        ["data", "splits", "features", "train", "evaluate", "predict", "report"],
    )
    def test_stage_command_exists(self, stage_name):
        """Each stage command exists and shows help."""
        result = runner.invoke(app, ["stage", stage_name, "--help"])
        assert result.exit_code == 0

    def test_stage_train_has_strict_option(self):
        """Train stage has --strict option."""
        result = runner.invoke(app, ["stage", "train", "--help"])
        assert result.exit_code == 0
        assert "--strict" in strip_ansi(result.stdout)

    def test_stage_report_has_strict_option(self):
        """Report stage has --strict option for publication fail-fast checks."""
        result = runner.invoke(app, ["stage", "report", "--help"])
        assert result.exit_code == 0
        assert "--strict" in strip_ansi(result.stdout)

    def test_stage_predict_help_has_options(self):
        """Predict stage help shows expected options."""
        result = runner.invoke(app, ["stage", "predict", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "--seed" in output
        assert "--verbose" in output


class TestStageReportCommandFlow:
    """Tests for report-stage config wiring."""

    @pytest.mark.parametrize("strict_flag", [False, True])
    def test_stage_report_passes_strict_to_pipeline_config(self, monkeypatch, strict_flag):
        captured: dict[str, object] = {}

        def fake_config(**kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(**kwargs)

        def fake_run_pipeline(config):
            captured["config"] = config
            return 0

        monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
        monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", fake_run_pipeline)

        args = ["stage", "report", "--seed", "123"]
        if strict_flag:
            args.append("--strict")
        result = runner.invoke(app, args)

        assert result.exit_code == 0
        kwargs = captured["kwargs"]
        assert kwargs["seed"] == 123
        assert kwargs["stages"] == ["report"]
        assert kwargs["strict"] is strict_flag
        assert kwargs["verbose"] is False
        assert captured["config"].strict is strict_flag

    def test_stage_report_passes_verbose_flag(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_config(**kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr("panelcast.pipelines.orchestrator.PipelineConfig", fake_config)
        monkeypatch.setattr("panelcast.pipelines.orchestrator.run_pipeline", lambda _config: 0)

        result = runner.invoke(app, ["stage", "report", "--verbose"])

        assert result.exit_code == 0
        assert captured["kwargs"]["verbose"] is True


class TestPackageExports:
    """Tests for package exports."""

    def test_run_pipeline_import(self):
        """run_pipeline can be imported from pipelines package."""
        from panelcast.pipelines import run_pipeline

        assert callable(run_pipeline)

    def test_pipeline_config_import(self):
        """PipelineConfig can be imported from pipelines package."""
        from panelcast.pipelines import PipelineConfig

        # Can instantiate with defaults
        config = PipelineConfig()
        assert config.seed == 42
        assert config.dry_run is False

    def test_pipeline_orchestrator_import(self):
        """PipelineOrchestrator can be imported."""
        from panelcast.pipelines import PipelineOrchestrator

        assert PipelineOrchestrator is not None

    def test_stages_import(self):
        """Stage-related exports work."""
        from panelcast.pipelines import (
            build_pipeline_stages,
            get_execution_order,
            get_stage,
        )

        assert len(build_pipeline_stages()) > 0
        assert callable(get_execution_order)
        assert callable(get_stage)

    def test_errors_import(self):
        """Error classes can be imported."""
        from panelcast.pipelines import (
            ConvergenceError,
            DataValidationError,
            PipelineError,
            StageError,
        )

        # Check error hierarchy
        assert issubclass(ConvergenceError, PipelineError)
        assert issubclass(DataValidationError, PipelineError)
        assert issubclass(StageError, PipelineError)

    def test_manifest_import(self):
        """Manifest-related exports work."""
        from panelcast.pipelines import (
            generate_run_id,
        )

        # generate_run_id should return string
        run_id = generate_run_id()
        assert isinstance(run_id, str)
        assert len(run_id) == 17  # YYYY-MM-DD_HHMMSS


class TestCLIExitCodes:
    """Tests for CLI exit code behavior."""

    def test_no_command_shows_help(self):
        """Running with no command shows help and exits cleanly."""
        result = runner.invoke(app, [])
        # With invoke_without_command=True, shows help
        assert result.exit_code == 0

    def test_invalid_stage_in_stages_fails(self):
        """Invalid stage name in --stages causes failure."""
        result = runner.invoke(app, ["run", "--dry-run", "--stages", "invalid_stage"])
        assert result.exit_code != 0

    def test_strict_single_chain_fails_with_config_error(self):
        """Strict mode should reject single-chain configs before pipeline execution."""
        result = runner.invoke(
            app,
            ["run", "--strict", "--num-chains", "1", "--num-samples", "100", "--num-warmup", "100"],
        )
        output = strip_ansi(result.output)
        assert result.exit_code != 0
        assert "strict mode requires num_chains >= 2" in output
        assert "Traceback" not in output


# ============================================================================
# CLI Input Validation Tests
# ============================================================================


class TestCLIInputValidation:
    """Tests for CLI input validation."""

    def test_invalid_chain_method_fails(self):
        """Invalid --chain-method exits with error."""
        result = runner.invoke(app, ["run", "--dry-run", "--chain-method", "invalid"])
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "Invalid --chain-method" in output

    def test_chain_method_case_insensitive(self):
        """Chain method is case-insensitive."""
        result = runner.invoke(app, ["run", "--dry-run", "--chain-method", "SEQUENTIAL"])
        assert result.exit_code == 0

    def test_invalid_n_exponent_prior_fails(self):
        """Invalid --n-exponent-prior exits with error."""
        result = runner.invoke(app, ["run", "--dry-run", "--n-exponent-prior", "invalid"])
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "Invalid --n-exponent-prior" in output

    def test_invalid_calibration_intervals_fails(self):
        """Invalid --calibration-intervals exits with error."""
        result = runner.invoke(app, ["run", "--dry-run", "--calibration-intervals", "abc"])
        assert result.exit_code != 0

    def test_empty_calibration_intervals_fails(self):
        """Empty --calibration-intervals exits with error."""
        result = runner.invoke(app, ["run", "--dry-run", "--calibration-intervals", ""])
        assert result.exit_code != 0

    def test_out_of_range_calibration_intervals_fails(self):
        """Out-of-range --calibration-intervals exits with error."""
        result = runner.invoke(app, ["run", "--dry-run", "--calibration-intervals", "0.0,0.5"])
        assert result.exit_code != 0

    def test_valid_calibration_intervals_accepted(self):
        """Valid --calibration-intervals accepted."""
        result = runner.invoke(
            app, ["run", "--dry-run", "--calibration-intervals", "0.50,0.80,0.95"]
        )
        assert result.exit_code == 0


class TestCLIRunOptions:
    """Tests for various run command option combinations."""

    def test_run_with_max_albums(self):
        """Run accepts --max-albums."""
        result = runner.invoke(app, ["run", "--dry-run", "--max-albums", "100"])
        assert result.exit_code == 0

    def test_run_with_num_chains(self):
        """Run accepts --num-chains."""
        result = runner.invoke(app, ["run", "--dry-run", "--num-chains", "2"])
        assert result.exit_code == 0

    def test_run_with_allow_divergences(self):
        """Run accepts --allow-divergences."""
        result = runner.invoke(app, ["run", "--dry-run", "--allow-divergences"])
        assert result.exit_code == 0

    def test_run_with_feature_ablation(self):
        """Run accepts feature ablation flags."""
        result = runner.invoke(app, ["run", "--dry-run", "--no-genre"])
        assert result.exit_code == 0

    def test_run_with_learn_n_exponent(self):
        """Run accepts --learn-n-exponent."""
        result = runner.invoke(app, ["run", "--dry-run", "--learn-n-exponent"])
        assert result.exit_code == 0

    def test_run_with_allow_unlocked_env(self):
        """Run accepts --allow-unlocked-env."""
        result = runner.invoke(app, ["run", "--dry-run", "--allow-unlocked-env"])
        assert result.exit_code == 0

    def test_run_with_no_secondary_split(self):
        """Run accepts --no-secondary-split."""
        result = runner.invoke(app, ["run", "--dry-run", "--no-secondary-split"])
        assert result.exit_code == 0


class TestCLISetupGuide:
    """Tests for --setup-guide option."""

    def test_setup_guide_shows_path(self):
        """--setup-guide shows path to setup guide."""
        result = runner.invoke(app, ["--setup-guide"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "GETTING_STARTED.md" in output

    def test_setup_guide_shows_topics(self):
        """--setup-guide lists covered topics."""
        result = runner.invoke(app, ["--setup-guide"])
        output = strip_ansi(result.output)
        assert "installation" in output.lower()


class TestCLIStageTrainOptions:
    """Tests for stage train subcommand options."""

    def test_train_has_rhat_option(self):
        """Train stage has --rhat-threshold option."""
        result = runner.invoke(app, ["stage", "train", "--help"])
        output = strip_ansi(result.output)
        assert "--rhat-threshold" in output

    def test_train_has_ess_option(self):
        """Train stage has --ess-threshold option."""
        result = runner.invoke(app, ["stage", "train", "--help"])
        output = strip_ansi(result.output)
        assert "--ess-threshold" in output

    def test_train_has_allow_divergences(self):
        """Train stage has --allow-divergences option."""
        result = runner.invoke(app, ["stage", "train", "--help"])
        output = strip_ansi(result.output)
        assert "--allow-divergences" in output


class TestCLIRunHelp:
    """Tests for run command help details."""

    def test_run_help_shows_mcmc_options(self):
        """Run help shows MCMC options."""
        result = runner.invoke(app, ["run", "--help"])
        output = strip_ansi(result.output)
        assert "--num-chains" in output
        assert "--num-samples" in output
        assert "--num-warmup" in output
        assert "--target-accept" in output
        assert "--max-tree-depth" in output

    def test_run_help_shows_convergence_options(self):
        """Run help shows convergence threshold options."""
        result = runner.invoke(app, ["run", "--help"])
        output = strip_ansi(result.output)
        assert "--rhat-threshold" in output
        assert "--ess-threshold" in output
        assert "--allow-divergences" in output

    def test_run_help_shows_feature_ablation_options(self):
        """Run help shows feature ablation flags."""
        result = runner.invoke(app, ["run", "--help"])
        output = strip_ansi(result.output)
        assert "--no-genre" in output
        assert "--no-artist" in output
        assert "--no-temporal" in output

    def test_run_help_shows_noise_options(self):
        """Run help shows heteroscedastic noise options."""
        result = runner.invoke(app, ["run", "--help"])
        output = strip_ansi(result.output)
        assert "--n-exponent" in output
        # --learn-n-exponent may be truncated in help, check prefix
        assert "--learn-n-expone" in output

    def test_run_help_shows_preflight_options(self):
        """Run help shows preflight options."""
        result = runner.invoke(app, ["run", "--help"])
        output = strip_ansi(result.output)
        assert "--preflight" in output
        assert "--preflight-only" in output
        assert "--force-run" in output
