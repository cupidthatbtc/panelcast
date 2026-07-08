"""Tests for pipeline orchestrator."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from panelcast.pipelines.errors import (
    ConvergenceError,
    EnvironmentError,
    StageError,
    StageSkipped,
)
from panelcast.pipelines.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    run_pipeline,
)

_TINY_DESCRIPTOR_YAML = "name: tiny\nmin_obs_thresholds: [3, 7, 9]\nprimary_min_obs: 7\n"


class TestMinRatingsResolution:
    """min_ratings resolves from the descriptor when left unset (issue 2c)."""

    def test_unset_resolves_to_aoty_primary_min_obs(self):
        """No dataset + unset min_ratings -> AOTY descriptor primary_min_obs (10)."""
        config = PipelineConfig()
        assert config.min_ratings is None
        PipelineOrchestrator(config)
        assert config.min_ratings == 10

    def test_unset_resolves_to_custom_descriptor_primary_min_obs(self, tmp_path):
        """A descriptor with primary_min_obs=7 drives the unset default."""
        descriptor_yaml = tmp_path / "tiny.yaml"
        descriptor_yaml.write_text(_TINY_DESCRIPTOR_YAML, encoding="utf-8")
        config = PipelineConfig(dataset=str(descriptor_yaml))
        PipelineOrchestrator(config)
        assert config.min_ratings == 7

    def test_explicit_min_ratings_wins_over_descriptor(self, tmp_path):
        """An explicit (materialized) min_ratings is never overridden by the
        descriptor's primary_min_obs."""
        descriptor_yaml = tmp_path / "tiny.yaml"
        descriptor_yaml.write_text(_TINY_DESCRIPTOR_YAML, encoding="utf-8")
        config = PipelineConfig(dataset=str(descriptor_yaml), min_ratings=9)
        PipelineOrchestrator(config)
        assert config.min_ratings == 9

    def test_min_ratings_in_resume_config_keys(self):
        """min_ratings is restored from the manifest on resume."""
        assert "min_ratings" in PipelineOrchestrator.RESUME_CONFIG_KEYS

    def test_resume_restores_min_ratings_from_manifest(self, tmp_path):
        """Resume restores the manifest's min_ratings, overriding __init__'s guess."""
        descriptor_yaml = tmp_path / "tiny.yaml"
        descriptor_yaml.write_text(_TINY_DESCRIPTOR_YAML, encoding="utf-8")
        config = PipelineConfig(resume="some-run")
        orch = PipelineOrchestrator(config)
        # __init__ resolved against AOTY (the restored dataset isn't applied yet).
        assert config.min_ratings == 10
        orch.manifest = MagicMock(flags={"dataset": str(descriptor_yaml), "min_ratings": 7})
        orch._restore_config_from_manifest()
        assert config.dataset == str(descriptor_yaml)
        assert config.min_ratings == 7

    def test_resume_redrives_min_ratings_when_unpinned(self, tmp_path):
        """A manifest lacking a pinned threshold re-derives it from the restored descriptor."""
        descriptor_yaml = tmp_path / "tiny.yaml"
        descriptor_yaml.write_text(_TINY_DESCRIPTOR_YAML, encoding="utf-8")
        config = PipelineConfig(resume="some-run")
        orch = PipelineOrchestrator(config)
        orch.manifest = MagicMock(flags={"dataset": str(descriptor_yaml), "min_ratings": None})
        orch._restore_config_from_manifest()
        assert config.min_ratings == 7  # re-derived from the tiny descriptor primary_min_obs

    def test_restore_config_noop_without_manifest(self, tmp_path):
        """_restore_config_from_manifest returns immediately when no manifest is loaded."""
        config = PipelineConfig()
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch.manifest = None
        orch._restore_config_from_manifest()  # no manifest -> no-op, no error
        assert config.min_ratings == 10


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """PipelineConfig has sensible defaults."""
        config = PipelineConfig()
        assert config.seed == 42
        assert config.skip_existing is False
        assert config.stages is None
        assert config.dry_run is False
        assert config.strict is False
        assert config.verbose is False
        assert config.resume is None

    def test_custom_values(self):
        """PipelineConfig accepts custom values."""
        config = PipelineConfig(
            seed=123,
            skip_existing=True,
            stages=["data", "splits"],
            dry_run=True,
            strict=True,
            verbose=True,
            resume="2026-01-19_143052",
        )
        assert config.seed == 123
        assert config.skip_existing is True
        assert config.stages == ["data", "splits"]
        assert config.dry_run is True
        assert config.strict is True
        assert config.verbose is True
        assert config.resume == "2026-01-19_143052"

    def test_max_tree_depth_validation(self):
        """PipelineConfig validates max_tree_depth range."""
        import pytest

        # Valid values at boundaries
        PipelineConfig(max_tree_depth=5)
        PipelineConfig(max_tree_depth=15)

        # Invalid: too low
        with pytest.raises(ValueError, match="max_tree_depth"):
            PipelineConfig(max_tree_depth=4)

        # Invalid: too high
        with pytest.raises(ValueError, match="max_tree_depth"):
            PipelineConfig(max_tree_depth=16)

    def test_strict_requires_two_or_more_chains(self):
        """Strict mode should fail fast when R-hat is not computable."""
        import pytest

        with pytest.raises(ValueError, match="num_chains >= 2"):
            PipelineConfig(strict=True, num_chains=1)

    def test_strict_requires_samples_meeting_ess_threshold(self):
        """Strict mode should fail fast when ESS threshold is unattainable."""
        import pytest

        with pytest.raises(ValueError, match="num_samples >= ess_threshold"):
            PipelineConfig(strict=True, num_chains=4, num_samples=100, ess_threshold=400)


class TestPipelineOrchestratorInit:
    """Tests for PipelineOrchestrator initialization."""

    def test_basic_init(self):
        """Orchestrator initializes with config."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.output_base == Path("outputs")
        assert orchestrator.run_dir is None
        assert orchestrator.manifest is None

    def test_custom_output_base(self, tmp_path: Path):
        """Orchestrator accepts custom output base."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        assert orchestrator.output_base == tmp_path


class TestDryRunMode:
    """Tests for dry_run mode."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_dry_run_does_not_execute_stages(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Dry run mode logs but doesn't execute stage functions."""
        # Set up mock environment verification
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123def456",
            warnings=[],
        )

        # Create a mock run_fn that would fail if called for real execution
        was_called = {"value": False}

        def failing_run_fn():
            was_called["value"] = True
            raise RuntimeError("Should not be called in dry run")

        # Patch the stages to have a controlled run_fn
        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "test_stage"
            mock_stage.description = "Test stage"
            mock_stage.run_fn = failing_run_fn
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(dry_run=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            assert was_called["value"] is False  # run_fn was NOT called

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_dry_run_records_nothing_executed(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Dry run marks itself in flags but claims no completed stages,
        stage hashes, or latest pointer (it executed nothing)."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = None
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(dry_run=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            orchestrator.run()

            assert orchestrator.manifest is not None
            assert orchestrator.manifest.stages_completed == []
            assert orchestrator.manifest.stage_hashes == {}
            assert orchestrator.manifest.flags["dry_run"] is True
            assert not (tmp_path / "latest.json").exists()
            assert not (tmp_path / "latest").exists()


class TestSkipExisting:
    """Tests for skip_existing mode."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_skip_existing_uses_manifest_hashes(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Skip existing checks hash from previous manifest."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = None
            # Make should_skip return True to simulate unchanged inputs
            mock_stage.should_skip.return_value = True
            mock_stage.compute_input_hash.return_value = "same_hash"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(skip_existing=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            orchestrator.run()

            # Stage should have been checked for skip
            mock_stage.should_skip.assert_called()

    def test_skip_flag_differences_detects_output_affecting_changes(self, tmp_path: Path):
        """Changed modeling flags should disable hash-based skip reuse."""
        config = PipelineConfig(min_ratings=25)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.manifest = MagicMock(flags={"seed": 42, "min_ratings": 25})

        previous_manifest = MagicMock(flags={"seed": 42, "min_ratings": 10})

        assert orchestrator._skip_flag_differences(previous_manifest) == ["min_ratings"]

    def test_skip_flag_differences_ignores_execution_only_flags(self, tmp_path: Path):
        """Execution-only flags should not invalidate skip reuse."""
        config = PipelineConfig(skip_existing=True)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.manifest = MagicMock(flags={"seed": 42, "skip_existing": True})

        previous_manifest = MagicMock(flags={"seed": 42, "skip_existing": False})

        assert orchestrator._skip_flag_differences(previous_manifest) == []


class TestErrorHandling:
    """Tests for error handling."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_error_returns_correct_exit_code(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed runs return correct error exit code."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Should return error exit code
            assert exit_code == 4  # StageError exit code

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_error_updates_manifest_success_false(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed run updates manifest success to False."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "train"
            mock_stage.description = "Train model"
            mock_stage.run_fn = MagicMock(side_effect=ConvergenceError("R-hat exceeded", "train"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(strict=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 2  # ConvergenceError exit code

            # Manifest should record failure
            assert orchestrator.manifest is not None
            assert orchestrator.manifest.success is False
            assert "R-hat exceeded" in orchestrator.manifest.error

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_error_attempts_to_move_to_failed(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed runs attempt to move to outputs/failed/."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            with patch("panelcast.pipelines.orchestrator.shutil.move") as mock_move:
                config = PipelineConfig()
                orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                orchestrator.run()

                # Should attempt to move
                mock_move.assert_called()


class TestManifestSaving:
    """Tests for manifest persistence."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_manifest_saved_after_each_stage(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Manifest is saved incrementally after each stage."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        save_count = {"value": 0}

        def count_saves(manifest, run_dir):
            save_count["value"] += 1
            # Actually save the manifest
            from panelcast.pipelines.manifest import save_run_manifest as real_save

            return real_save(manifest, run_dir)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            stage1 = MagicMock()
            stage1.name = "stage1"
            stage1.description = "Stage 1"
            stage1.run_fn = None
            stage1.compute_input_hash.return_value = "hash1"

            stage2 = MagicMock()
            stage2.name = "stage2"
            stage2.description = "Stage 2"
            stage2.run_fn = None
            stage2.compute_input_hash.return_value = "hash2"

            mock_order.return_value = [stage1, stage2]

            with patch(
                "panelcast.pipelines.orchestrator.save_run_manifest",
                side_effect=count_saves,
            ):
                config = PipelineConfig()
                orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                orchestrator.run()

            # Initial save + 2 stages + final success = at least 4 saves
            assert save_count["value"] >= 3


class TestEnvironmentVerification:
    """Tests for environment verification."""

    def test_environment_verified_at_startup(self, tmp_path: Path):
        """Environment verification is called at pipeline start."""
        with patch("panelcast.pipelines.orchestrator.ensure_environment_locked") as mock_ensure:
            with patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify:
                mock_verify.return_value = MagicMock(
                    is_reproducible=True,
                    pixi_lock_hash="abc123",
                    warnings=[],
                )

                with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                    mock_order.return_value = []

                    config = PipelineConfig()
                    orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                    orchestrator.run()

                    # ensure_environment_locked should be called
                    mock_ensure.assert_called_once()

    def test_strict_mode_fails_when_pixi_lock_missing(self, tmp_path: Path):
        """Strict mode fails if pixi.lock is not found."""
        with patch(
            "panelcast.pipelines.orchestrator.ensure_environment_locked",
            side_effect=EnvironmentError("pixi.lock not found"),
        ):
            config = PipelineConfig(strict=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 5  # EnvironmentError exit code


class TestRunPipeline:
    """Tests for run_pipeline convenience function."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_run_pipeline_returns_exit_code(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """run_pipeline returns orchestrator exit code."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []

            config = PipelineConfig()
            exit_code = run_pipeline(config, output_base=tmp_path)

            assert exit_code == 0


class TestStageSkipped:
    """Tests for StageSkipped control flow."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_stage_skipped_is_not_error(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """StageSkipped exception doesn't cause pipeline failure."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageSkipped("Inputs unchanged"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Should succeed (not fail on StageSkipped)
            assert exit_code == 0
            # Stage should be in skipped list
            assert "data" in orchestrator.manifest.stages_skipped


class TestLatestSymlink:
    """Tests for outputs/latest symlink/junction creation."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_latest_link_created_on_success(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Successful run creates outputs/latest link."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0

            latest_link = tmp_path / "latest"
            # On Windows, this might be a junction (appears as dir)
            # On Unix, this is a symlink
            assert latest_link.exists() or latest_link.is_symlink()

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_latest_link_not_created_on_failure(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed run does not create/update outputs/latest link."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 4

            # latest link should NOT exist (or point to previous successful run, not this failed one)
            latest_link = tmp_path / "latest"
            # For a fresh run that fails, latest should not exist
            assert not latest_link.exists() and not latest_link.is_symlink()


class TestBuildCommandString:
    """Tests for command string building."""

    def test_default_command(self, tmp_path: Path):
        """Default config produces simple command."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert cmd == "panelcast run"

    def test_command_with_options(self, tmp_path: Path):
        """Options are included in command string."""
        config = PipelineConfig(
            seed=123,
            skip_existing=True,
            stages=["data", "splits"],
            dry_run=True,
            strict=True,
            verbose=True,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--seed 123" in cmd
        assert "--skip-existing" in cmd
        assert "--stages data,splits" in cmd
        assert "--dry-run" in cmd
        assert "--strict" in cmd
        assert "--verbose" in cmd


class TestResumeConfigRestoration:
    """Tests for resume config restoration from manifest."""

    def test_chain_method_in_resume_config_keys(self):
        """chain_method is recorded in the manifest and re-emitted by the resume
        command, so a resumed run must restore it rather than fall back to the default."""
        assert "chain_method" in PipelineOrchestrator.RESUME_CONFIG_KEYS

    def test_resume_restores_chain_method_from_manifest(self):
        """Resume adopts the manifest's chain_method instead of the CLI default."""
        config = PipelineConfig(resume="some-run")
        orch = PipelineOrchestrator(config)
        assert config.chain_method == "sequential"  # CLI default
        orch.manifest = MagicMock(flags={"chain_method": "parallel"})
        orch._restore_config_from_manifest()
        assert config.chain_method == "parallel"

    def test_seed_in_resume_config_keys(self):
        """The RNG seed governs the whole MCMC draw, so a resumed run must restore
        it rather than silently re-fit under the CLI default (#seed-resume)."""
        assert "seed" in PipelineOrchestrator.RESUME_CONFIG_KEYS

    def test_resume_restores_seed_from_manifest(self):
        """Resume adopts the manifest's seed instead of reverting to the CLI default."""
        config = PipelineConfig(resume="some-run")
        orch = PipelineOrchestrator(config)
        assert config.seed == 42  # CLI default
        orch.manifest = MagicMock(flags={"seed": 7})
        orch._restore_config_from_manifest()
        assert config.seed == 7

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    @patch("panelcast.pipelines.orchestrator.get_execution_order")
    def test_resume_restores_config_from_manifest(
        self,
        mock_order: MagicMock,
        mock_verify: MagicMock,
        _mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Resume restores target_accept and max_tree_depth from manifest."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )
        mock_order.return_value = []

        # Create run directory with manifest containing old config values
        run_id = "2026-01-20_120000"
        run_dir = tmp_path / run_id
        run_dir.mkdir()

        manifest_data = {
            "run_id": run_id,
            "created_at": "2026-01-20T12:00:00",
            "command": "panelcast run",
            "flags": {
                "seed": 42,
                "skip_existing": False,
                "stages": None,
                "dry_run": False,
                "strict": False,
                "verbose": False,
                "resume": None,
                "max_albums": 50,
                "num_chains": 4,
                "num_samples": 1000,
                "num_warmup": 1000,
                "target_accept": 0.75,  # Old value (different from current default 0.90)
                "max_tree_depth": 8,  # Old value (different from current default 10)
                "chain_method": "sequential",
                "rhat_threshold": 1.01,
                "ess_threshold": 400,
                "allow_divergences": False,
                "min_ratings": 10,
                "min_albums_filter": 2,
                "enable_genre": True,
                "enable_artist": True,
                "enable_temporal": True,
                "n_exponent": 0.0,
                "learn_n_exponent": False,
                "n_exponent_alpha": 2.0,
                "n_exponent_beta": 4.0,
                "n_exponent_prior": "logit-normal",
            },
            "seed": 42,
            "git": {
                "commit": "abc123",
                "branch": "main",
                "dirty": False,
                "untracked_count": 0,
            },
            "environment": {
                "python_version": "3.11.0",
                "jax_version": "0.4.26",
                "numpyro_version": "0.15.0",
                "arviz_version": "0.18.0",
                "platform": "Linux",
                "pixi_lock_hash": "abc123",
            },
            "input_hashes": {},
            "stage_hashes": {},
            "stages_completed": [],
            "stages_skipped": [],
            "outputs": {},
            "success": False,
            "error": None,
            "duration_seconds": 0.0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest_data))

        # Create config with current defaults (0.90, 10)
        config = PipelineConfig(resume=run_id)
        assert config.target_accept == 0.90  # Current default
        assert config.max_tree_depth == 10  # Current default

        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.run()

        # After resume, config should have manifest values
        assert orchestrator.config.target_accept == 0.75
        assert orchestrator.config.max_tree_depth == 8

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    @patch("panelcast.pipelines.orchestrator.get_execution_order")
    def test_resume_warns_on_missing_config_keys(
        self,
        mock_order: MagicMock,
        mock_verify: MagicMock,
        _mock_ensure: MagicMock,
        tmp_path: Path,
        caplog,
    ):
        """Resume warns when manifest is missing MCMC config keys."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )
        mock_order.return_value = []

        # Create run directory with manifest missing target_accept and max_tree_depth
        run_id = "2026-01-20_120000"
        run_dir = tmp_path / run_id
        run_dir.mkdir()

        # Manifest without target_accept and max_tree_depth (simulating old manifest)
        manifest_data = {
            "run_id": run_id,
            "created_at": "2026-01-20T12:00:00",
            "command": "panelcast run",
            "flags": {
                "seed": 42,
                "skip_existing": False,
                "stages": None,
                "dry_run": False,
                "strict": False,
                "verbose": False,
                "resume": None,
                "max_albums": 50,
                "num_chains": 4,
                "num_samples": 1000,
                "num_warmup": 1000,
                # target_accept missing
                # max_tree_depth missing
                "chain_method": "sequential",
            },
            "seed": 42,
            "git": {
                "commit": "abc123",
                "branch": "main",
                "dirty": False,
                "untracked_count": 0,
            },
            "environment": {
                "python_version": "3.11.0",
                "jax_version": "0.4.26",
                "numpyro_version": "0.15.0",
                "arviz_version": "0.18.0",
                "platform": "Linux",
                "pixi_lock_hash": "abc123",
            },
            "input_hashes": {},
            "stage_hashes": {},
            "stages_completed": [],
            "stages_skipped": [],
            "outputs": {},
            "success": False,
            "error": None,
            "duration_seconds": 0.0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest_data))

        config = PipelineConfig(resume=run_id)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with caplog.at_level(logging.WARNING):
            orchestrator.run()

        # Should have logged warnings about missing keys
        log_messages = caplog.text
        assert "target_accept" in log_messages
        assert "max_tree_depth" in log_messages
        assert "resume_config_missing" in log_messages

        # Config should still have current defaults
        assert orchestrator.config.target_accept == 0.90
        assert orchestrator.config.max_tree_depth == 10

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    @patch("panelcast.pipelines.orchestrator.get_execution_order")
    def test_resume_partial_config_restoration(
        self,
        mock_order: MagicMock,
        mock_verify: MagicMock,
        _mock_ensure: MagicMock,
        tmp_path: Path,
        caplog,
    ):
        """Resume restores present keys and warns on missing keys."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )
        mock_order.return_value = []

        # Create run directory with manifest containing only target_accept
        run_id = "2026-01-20_120000"
        run_dir = tmp_path / run_id
        run_dir.mkdir()

        manifest_data = {
            "run_id": run_id,
            "created_at": "2026-01-20T12:00:00",
            "command": "panelcast run",
            "flags": {
                "seed": 42,
                "skip_existing": False,
                "stages": None,
                "dry_run": False,
                "strict": False,
                "verbose": False,
                "resume": None,
                "max_albums": 50,
                "num_chains": 4,
                "num_samples": 1000,
                "num_warmup": 1000,
                "target_accept": 0.85,  # Present with non-default value
                # max_tree_depth missing
                "chain_method": "sequential",
            },
            "seed": 42,
            "git": {
                "commit": "abc123",
                "branch": "main",
                "dirty": False,
                "untracked_count": 0,
            },
            "environment": {
                "python_version": "3.11.0",
                "jax_version": "0.4.26",
                "numpyro_version": "0.15.0",
                "arviz_version": "0.18.0",
                "platform": "Linux",
                "pixi_lock_hash": "abc123",
            },
            "input_hashes": {},
            "stage_hashes": {},
            "stages_completed": [],
            "stages_skipped": [],
            "outputs": {},
            "success": False,
            "error": None,
            "duration_seconds": 0.0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest_data))

        config = PipelineConfig(resume=run_id)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with caplog.at_level(logging.WARNING):
            orchestrator.run()

        # target_accept should be restored to 0.85
        assert orchestrator.config.target_accept == 0.85

        # Warning logged for missing max_tree_depth
        assert "max_tree_depth" in caplog.text

        # max_tree_depth remains at default 10
        assert orchestrator.config.max_tree_depth == 10


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestPipelineConfigValidation:
    """Tests for PipelineConfig validation."""

    def test_invalid_n_exponent_prior_raises(self):
        """Invalid n_exponent_prior raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="n_exponent_prior"):
            PipelineConfig(n_exponent_prior="invalid_prior")

    def test_empty_calibration_intervals_raises(self):
        """Empty calibration_intervals raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="calibration_intervals"):
            PipelineConfig(calibration_intervals=())

    def test_invalid_calibration_interval_raises(self):
        """Calibration interval outside (0, 1) raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="calibration interval"):
            PipelineConfig(calibration_intervals=(0.0,))

        with pytest.raises(ValueError, match="calibration interval"):
            PipelineConfig(calibration_intervals=(1.0,))

    def test_negative_coverage_tolerance_raises(self):
        """Negative coverage_tolerance raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="coverage_tolerance"):
            PipelineConfig(coverage_tolerance=-0.01)

    def test_invalid_prediction_interval_raises(self):
        """prediction_interval outside (0, 1) raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="prediction_interval"):
            PipelineConfig(prediction_interval=0.0)

        with pytest.raises(ValueError, match="prediction_interval"):
            PipelineConfig(prediction_interval=1.0)

    def test_num_chains_zero_raises(self):
        """num_chains < 1 raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="num_chains"):
            PipelineConfig(num_chains=0)

    def test_num_samples_zero_raises(self):
        """num_samples < 1 raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="num_samples"):
            PipelineConfig(num_samples=0)

    def test_ess_threshold_zero_raises(self):
        """ess_threshold < 1 raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="ess_threshold"):
            PipelineConfig(ess_threshold=0)

    def test_valid_prior_accepted(self):
        """Valid prior names are accepted."""
        PipelineConfig(n_exponent_prior="logit-normal")
        PipelineConfig(n_exponent_prior="beta")

    def test_valid_calibration_intervals(self):
        """Valid calibration intervals accepted."""
        config = PipelineConfig(calibration_intervals=(0.5, 0.8, 0.95))
        assert config.calibration_intervals == (0.5, 0.8, 0.95)

    def test_invalid_target_transform_raises(self):
        import pytest

        with pytest.raises(ValueError, match="target_transform"):
            PipelineConfig(target_transform="bogus")

    def test_invalid_likelihood_family_raises(self):
        import pytest

        with pytest.raises(ValueError, match="likelihood_family"):
            PipelineConfig(likelihood_family="bogus")

    def test_discretize_unsupported_family_raises(self):
        import pytest

        with pytest.raises(ValueError, match="discretize_observation"):
            PipelineConfig(likelihood_family="beta", discretize_observation=True)

    def test_discretize_with_non_identity_transform_raises(self):
        import pytest

        with pytest.raises(ValueError, match="target_transform"):
            PipelineConfig(discretize_observation=True, target_transform="offset_logit")

    def test_invalid_debut_prev_score_source_raises(self):
        import pytest

        with pytest.raises(ValueError, match="debut_prev_score_source"):
            PipelineConfig(debut_prev_score_source="bogus")

    def test_invalid_latent_process_raises(self):
        import pytest

        with pytest.raises(ValueError, match="latent_process"):
            PipelineConfig(latent_process="bogus")

    def test_invalid_sigma_obs_prior_type_raises(self):
        import pytest

        with pytest.raises(ValueError, match="sigma_obs_prior_type"):
            PipelineConfig(sigma_obs_prior_type="bogus")

    def test_nonpositive_tau_entity_scale_raises(self):
        import pytest

        with pytest.raises(ValueError, match="tau_entity_scale"):
            PipelineConfig(tau_entity_scale=0.0)

    def test_identity_required_family_with_default_transform_raises(self):
        """Bounded families die inside train under offset_logit; validation
        must reject the combination up front."""
        import pytest

        for family in ("beta", "beta_ceiling", "beta_binomial"):
            with pytest.raises(ValueError, match="target_transform='identity'"):
                PipelineConfig(likelihood_family=family)

    def test_identity_required_family_with_identity_passes(self):
        config = PipelineConfig(likelihood_family="beta", target_transform="identity")
        assert config.likelihood_family == "beta"

    def test_sigma_knobs_rejected_for_sigma_ignoring_family(self):
        """learn_n_exponent / heteroscedastic_entity_obs / fixed n_exponent are
        silently inert for the Beta families; validation rejects them."""
        import pytest

        with pytest.raises(ValueError, match="learn_n_exponent"):
            PipelineConfig(
                likelihood_family="beta",
                target_transform="identity",
                learn_n_exponent=True,
            )
        with pytest.raises(ValueError, match="heteroscedastic_entity_obs"):
            PipelineConfig(
                likelihood_family="beta_ceiling",
                target_transform="identity",
                heteroscedastic_entity_obs=True,
            )
        with pytest.raises(ValueError, match="n_exponent"):
            PipelineConfig(
                likelihood_family="beta",
                target_transform="identity",
                n_exponent=0.5,
            )

    def test_sigma_knobs_still_valid_for_sigma_using_families(self):
        config = PipelineConfig(likelihood_family="studentt", learn_n_exponent=True)
        assert config.learn_n_exponent is True
        config = PipelineConfig(likelihood_family="normal", heteroscedastic_entity_obs=True)
        assert config.heteroscedastic_entity_obs is True

    def test_min_train_albums_default_matches_cli(self):
        """The dataclass default matches the documented `run` CLI default (2),
        so `stage splits` / `demo` build the same split population as `run`."""
        assert PipelineConfig().min_train_albums == 2


class TestMinRatingsThresholdValidation:
    """The orchestrator rejects min_ratings values the data stage never writes."""

    def test_unmaterialized_threshold_raises(self, tmp_path):
        import pytest

        config = PipelineConfig(min_ratings=7)
        with pytest.raises(ValueError, match=r"min_ratings=7.*\[5, 10, 25\]"):
            PipelineOrchestrator(config, output_base=tmp_path)

    def test_descriptor_thresholds_accepted(self, tmp_path):
        for value in (5, 10, 25):
            orch = PipelineOrchestrator(
                PipelineConfig(min_ratings=value), output_base=tmp_path
            )
            assert orch.config.min_ratings == value

    def test_none_resolves_to_descriptor_primary(self, tmp_path):
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        assert orch.config.min_ratings == 10


class TestOrchestratorCommandString:
    """Additional tests for command string building."""

    def test_command_with_mcmc_options(self, tmp_path):
        """MCMC options included in command string."""
        config = PipelineConfig(
            num_chains=8,
            num_samples=2000,
            num_warmup=500,
            target_accept=0.95,
            max_tree_depth=12,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--num-chains 8" in cmd
        assert "--num-samples 2000" in cmd
        assert "--num-warmup 500" in cmd
        assert "--target-accept 0.95" in cmd
        assert "--max-tree-depth 12" in cmd

    def test_command_with_feature_flags(self, tmp_path):
        """Feature ablation flags in command string."""
        config = PipelineConfig(
            enable_genre=False,
            enable_artist=False,
            enable_temporal=False,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--no-genre" in cmd
        assert "--no-artist" in cmd
        assert "--no-temporal" in cmd

    def test_command_with_allow_divergences(self, tmp_path):
        """Allow divergences flag in command string."""
        config = PipelineConfig(allow_divergences=True)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--allow-divergences" in cmd

    def test_command_with_allow_unlocked_env(self, tmp_path):
        """Allow unlocked env flag in command string."""
        config = PipelineConfig(enforce_lockfile=False)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--allow-unlocked-env" in cmd

    def test_command_with_learn_n_exponent(self, tmp_path):
        """Learn n_exponent flag in command string."""
        config = PipelineConfig(learn_n_exponent=True)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--learn-n-exponent" in cmd

    def test_command_with_no_secondary_split(self, tmp_path):
        """Disabled secondary split flag in command string."""
        config = PipelineConfig(evaluate_secondary_split=False)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--no-secondary-split" in cmd

    def test_command_default_values_not_included(self, tmp_path):
        """Default values are not included in command string."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        # Default values should NOT appear
        assert "--seed" not in cmd
        assert "--num-chains" not in cmd
        assert "--max-albums" not in cmd
        assert "--no-genre" not in cmd

    def test_command_with_likelihood_and_dataset(self, tmp_path):
        """Likelihood and dataset overrides appear in the command string."""
        config = PipelineConfig(
            likelihood_df=8.0,
            likelihood_family="skew_normal",
            discretize_observation=True,
            # discretization requires the raw score scale (identity)
            target_transform="identity",
            dataset="aero",
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--likelihood-df 8.0" in cmd
        assert "--likelihood-family skew_normal" in cmd
        assert "--discretize-observation" in cmd
        assert "--dataset aero" in cmd


class TestOrchestratorResumeErrors:
    """Tests for resume error paths."""

    def test_resume_nonexistent_run_raises(self, tmp_path):
        """Resume with nonexistent run ID raises PipelineError."""
        import pytest

        from panelcast.pipelines.errors import PipelineError as PE

        config = PipelineConfig(resume="nonexistent_run_id")
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch(
                "panelcast.pipelines.orchestrator.verify_environment",
                return_value=MagicMock(is_reproducible=True, pixi_lock_hash="abc123", warnings=[]),
            ),
        ):
            with pytest.raises(PE, match="Cannot find run to resume"):
                orchestrator.run()

    def test_resume_missing_manifest_raises(self, tmp_path):
        """Resume with missing manifest.json raises PipelineError."""
        import pytest

        from panelcast.pipelines.errors import PipelineError as PE

        run_id = "2026-01-20_120000"
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        # No manifest.json in directory

        config = PipelineConfig(resume=run_id)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch(
                "panelcast.pipelines.orchestrator.verify_environment",
                return_value=MagicMock(is_reproducible=True, pixi_lock_hash="abc123", warnings=[]),
            ),
        ):
            with pytest.raises(PE, match="No manifest.json"):
                orchestrator.run()


class TestStageContextCreation:
    """Tests for _create_stage_context."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_create_stage_context_propagates_config(self, mock_verify, mock_ensure, tmp_path):
        """StageContext receives all config values."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        config = PipelineConfig(
            seed=99,
            strict=True,
            verbose=True,
            max_albums=100,
            num_chains=8,
            n_exponent=0.3,
            enable_genre=False,
            warmup_export_path="outputs/select/x/warmup_reference.pkl",
            warmup_import_path="outputs/select/y/warmup_reference.pkl",
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []
            orchestrator.run()

        ctx = orchestrator._create_stage_context()
        assert ctx.seed == 99
        assert ctx.strict is True
        assert ctx.verbose is True
        assert ctx.max_albums == 100
        assert ctx.num_chains == 8
        assert ctx.n_exponent == 0.3
        assert ctx.enable_genre is False
        # The warm-start paths MUST reach the context: train_bayes reads them
        # via getattr, so a dropped field silently disables warmup transfer
        # (exactly the 0.8.0 bug the #138 GPU validation exposed).
        assert ctx.warmup_export_path == "outputs/select/x/warmup_reference.pkl"
        assert ctx.warmup_import_path == "outputs/select/y/warmup_reference.pkl"


class TestCloseLogHandlers:
    """Tests for _close_log_handlers."""

    def test_close_log_handlers_removes_file_handlers(self, tmp_path):
        """_close_log_handlers closes and removes file handlers."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        # Add a test file handler to root logger
        root_logger = logging.getLogger()
        test_file = tmp_path / "test.log"
        handler = logging.FileHandler(str(test_file))
        root_logger.addHandler(handler)

        orchestrator._close_log_handlers()

        # Handler should be removed
        assert handler not in root_logger.handlers

    def test_close_log_handlers_noop_when_no_file_handlers(self, tmp_path):
        """_close_log_handlers is safe when no file handlers exist."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        # Should not raise
        orchestrator._close_log_handlers()


class TestRecordStageOutputs:
    """Tests for _record_stage_outputs."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_records_existing_output_paths(self, mock_verify, mock_ensure, tmp_path):
        """_record_stage_outputs records paths that exist."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        # Set up manifest
        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []
            orchestrator.run()

        # Create a fake output file
        output_file = tmp_path / "output.csv"
        output_file.write_text("data")

        stage = MagicMock()
        stage.name = "test"
        stage.output_paths = [output_file, tmp_path / "nonexistent.csv"]

        orchestrator._record_stage_outputs(stage, run_result=None)

        # Should record the existing file
        assert f"test:{output_file.as_posix()}" in orchestrator.manifest.outputs
        # Should NOT record nonexistent file
        nonexistent_key = f"test:{(tmp_path / 'nonexistent.csv').as_posix()}"
        assert nonexistent_key not in orchestrator.manifest.outputs

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_records_dict_run_result(self, mock_verify, mock_ensure, tmp_path):
        """_record_stage_outputs records paths from dict run_result."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []
            orchestrator.run()

        output_file = tmp_path / "dynamic_output.json"
        output_file.write_text("{}")

        stage = MagicMock()
        stage.name = "test"
        stage.output_paths = []

        run_result = {"dynamic_key": str(output_file)}
        orchestrator._record_stage_outputs(stage, run_result=run_result)

        assert "test:dynamic_key" in orchestrator.manifest.outputs

    def test_noop_when_manifest_none(self, tmp_path):
        """_record_stage_outputs does nothing when manifest is None."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.manifest = None

        stage = MagicMock()
        stage.name = "test"
        stage.output_paths = []

        # Should not raise
        orchestrator._record_stage_outputs(stage, run_result=None)


class TestConfigConflictHandling:
    """Tests for config conflict resolution."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_learn_n_exponent_overrides_fixed(self, mock_verify, mock_ensure, tmp_path):
        """When both learn_n_exponent and n_exponent set, n_exponent is cleared."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        config = PipelineConfig(n_exponent=0.5, learn_n_exponent=True)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []
            orchestrator.run()

        # n_exponent should be cleared to 0.0
        assert orchestrator.config.n_exponent == 0.0


class TestConvergenceHandling:
    """Tests for convergence error handling in strict vs non-strict mode."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_convergence_error_strict_returns_exit_code_2(self, mock_verify, mock_ensure, tmp_path):
        """Strict mode returns exit code 2 for convergence errors."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "train"
            mock_stage.description = "Train model"
            mock_stage.run_fn = MagicMock(side_effect=ConvergenceError("R-hat exceeded", "train"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(strict=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 2
            assert orchestrator.manifest.success is False
            assert "R-hat exceeded" in orchestrator.manifest.error

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_unexpected_exception_wrapped_as_pipeline_error(
        self, mock_verify, mock_ensure, tmp_path
    ):
        """Unexpected exceptions are wrapped as PipelineError."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=RuntimeError("unexpected"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Generic exit code 1 for unexpected errors
            assert exit_code == 1
            assert orchestrator.manifest.success is False


class TestDefaultConfigCache:
    """Tests for the _DEFAULT_CONFIG cache."""

    def test_reset_default_config(self):
        """_reset_default_config clears the cache."""
        from panelcast.pipelines.orchestrator import (
            _get_default_config,
            _reset_default_config,
        )

        # Get a default config to populate cache
        config1 = _get_default_config()
        assert config1 is not None

        # Reset cache
        _reset_default_config()

        # Get again - should create new instance
        config2 = _get_default_config()
        assert config2 is not None
        # They should be equal but not the same object
        assert config2.seed == config1.seed


class TestResumeDescriptorDrift:
    """Resume aborts when the descriptor changed since the original run."""

    def test_resume_descriptor_hash_mismatch_raises(self, tmp_path):
        import pytest

        from panelcast.pipelines.errors import PipelineError

        descriptor_yaml = tmp_path / "tiny.yaml"
        descriptor_yaml.write_text(_TINY_DESCRIPTOR_YAML, encoding="utf-8")
        config = PipelineConfig(resume="some-run")
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch.manifest = MagicMock(
            flags={"dataset": str(descriptor_yaml), "dataset_descriptor_hash": "stale-hash"}
        )
        with pytest.raises(PipelineError, match="descriptor changed"):
            orch._restore_config_from_manifest()


class TestCreateLatestLinkBranches:
    """Exercises the symlink/junction branches of _create_latest_link directly."""

    def _orch_with_run_dir(self, tmp_path):
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        orch.run_dir = tmp_path / "run"
        orch.run_dir.mkdir()
        return orch

    def test_noop_without_run_dir(self, tmp_path):
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        orch.run_dir = None
        orch._create_latest_link()  # returns immediately, no link created
        assert not (tmp_path / "latest").exists()

    def test_exists_oserror_then_unlink_failure_returns(self, tmp_path, monkeypatch):
        """A broken NTFS link (OSError on exists) is treated as removable; unlink failure logs."""
        import pathlib

        orch = self._orch_with_run_dir(tmp_path)

        def raise_oserror(self):
            raise OSError("WinError 1920")

        def raise_filenotfound(self):
            raise FileNotFoundError("gone")

        monkeypatch.setattr(pathlib.Path, "exists", raise_oserror)
        monkeypatch.setattr(pathlib.Path, "unlink", raise_filenotfound)
        orch._create_latest_link()  # OSError -> removable; unlink raises -> logged + return

    def test_win32_symlink_path(self, tmp_path, monkeypatch):
        orch = self._orch_with_run_dir(tmp_path)
        monkeypatch.setattr("panelcast.pipelines.orchestrator.sys.platform", "win32")
        calls = {}
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.os.symlink",
            lambda *a, **k: calls.setdefault("symlink", True),
        )
        orch._create_latest_link()
        assert calls.get("symlink")

    def test_win32_junction_fallback(self, tmp_path, monkeypatch):
        orch = self._orch_with_run_dir(tmp_path)
        monkeypatch.setattr("panelcast.pipelines.orchestrator.sys.platform", "win32")

        def no_privilege(*a, **k):
            raise OSError("symlink privilege required")

        calls = {}
        monkeypatch.setattr("panelcast.pipelines.orchestrator.os.symlink", no_privilege)
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.subprocess.run",
            lambda *a, **k: calls.setdefault("mklink", True),
        )
        orch._create_latest_link()
        assert calls.get("mklink")  # fell back to the directory junction

    def test_win32_removes_existing_junction(self, tmp_path, monkeypatch):
        orch = self._orch_with_run_dir(tmp_path)
        (tmp_path / "latest").mkdir()  # an existing junction appears as a directory
        monkeypatch.setattr("panelcast.pipelines.orchestrator.sys.platform", "win32")
        calls = {}
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.os.rmdir",
            lambda p: calls.setdefault("rmdir", True),
        )
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.os.symlink",
            lambda *a, **k: calls.setdefault("symlink", True),
        )
        orch._create_latest_link()
        assert calls.get("rmdir") and calls.get("symlink")

    def test_win32_junction_rejects_unsafe_path(self, tmp_path, monkeypatch):
        """A junction target with shell metacharacters is refused (no mklink)."""
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        orch.run_dir = tmp_path / "ru$n"  # '$' is a rejected metacharacter
        orch.run_dir.mkdir()
        monkeypatch.setattr("panelcast.pipelines.orchestrator.sys.platform", "win32")

        def no_privilege(*a, **k):
            raise OSError("symlink privilege required")

        calls = {}
        monkeypatch.setattr("panelcast.pipelines.orchestrator.os.symlink", no_privilege)
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.subprocess.run",
            lambda *a, **k: calls.setdefault("mklink", True),
        )
        orch._create_latest_link()
        assert "mklink" not in calls  # rejected before invoking mklink


class TestExecuteStagesSkipDetection:
    """Covers the skip-detection IO error handlers in _execute_stages."""

    def test_latest_link_exists_oserror_is_swallowed(self, tmp_path, monkeypatch):
        import pathlib

        config = PipelineConfig(skip_existing=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch.run_dir = tmp_path / "run"
        orch.run_dir.mkdir()

        def raise_oserror(self):
            raise OSError("WinError 1920")

        monkeypatch.setattr(pathlib.Path, "exists", raise_oserror)
        orch._execute_stages([])  # OSError on exists() -> link_exists False, no crash

    def test_previous_manifest_load_oserror_is_swallowed(self, tmp_path, monkeypatch):
        config = PipelineConfig(skip_existing=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch.run_dir = tmp_path / "run"
        orch.run_dir.mkdir()
        latest = tmp_path / "latest"
        latest.mkdir()
        (latest / "manifest.json").write_text("{}", encoding="utf-8")

        def boom(path):
            raise OSError("locked")

        monkeypatch.setattr("panelcast.pipelines.orchestrator.load_run_manifest", boom)
        orch._execute_stages([])  # load raises OSError -> swallowed, previous stays None

    def test_previous_manifest_load_generic_error_is_swallowed(self, tmp_path, monkeypatch):
        config = PipelineConfig(skip_existing=True)
        orch = PipelineOrchestrator(config, output_base=tmp_path)
        orch.run_dir = tmp_path / "run"
        orch.run_dir.mkdir()
        latest = tmp_path / "latest"
        latest.mkdir()
        (latest / "manifest.json").write_text("{}", encoding="utf-8")

        def boom(path):
            raise ValueError("corrupt manifest")

        monkeypatch.setattr("panelcast.pipelines.orchestrator.load_run_manifest", boom)
        orch._execute_stages([])  # generic error -> swallowed, previous stays None


class TestHandleFailureExistingFailedDir:
    """Covers replacing a pre-existing failed/<run> directory on re-failure."""

    def test_existing_failed_dir_is_replaced(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.save_run_manifest", lambda m, d: None
        )
        orch = PipelineOrchestrator(PipelineConfig(), output_base=tmp_path)
        orch.run_dir = tmp_path / "run"
        orch.run_dir.mkdir()
        (orch.run_dir / "new.txt").write_text("new", encoding="utf-8")
        orch.manifest = MagicMock()
        orch._start_time = 0.0
        stale = tmp_path / "failed" / "run"
        stale.mkdir(parents=True)
        (stale / "old.txt").write_text("old", encoding="utf-8")

        orch._handle_failure(RuntimeError("boom"), stage="data")

        moved = tmp_path / "failed" / "run"
        assert moved.exists()
        assert (moved / "new.txt").exists()  # the fresh run replaced the stale failed dir
        assert not (moved / "old.txt").exists()


class TestRunTagAndVersionProvenance:
    """--tag and the package version round-trip through the run manifest."""

    def _setup(self, tmp_path, monkeypatch, **config_kwargs):
        from panelcast.pipelines.manifest import EnvironmentInfo
        from panelcast.utils.git_state import GitState

        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_git_state",
            lambda: GitState(commit="abc", branch="test", dirty=False, untracked_count=0),
        )
        monkeypatch.setattr(
            "panelcast.pipelines.orchestrator.capture_environment",
            lambda: EnvironmentInfo(
                python_version="3.11.0",
                jax_version="0.4.0",
                numpyro_version=None,
                arviz_version=None,
                platform="Test",
                pixi_lock_hash=None,
            ),
        )
        orch = PipelineOrchestrator(PipelineConfig(**config_kwargs), output_base=tmp_path)
        orch._setup_run()
        return orch

    def test_tag_and_version_saved_and_loaded(self, tmp_path, monkeypatch):
        from panelcast import __version__
        from panelcast.pipelines.manifest import load_run_manifest

        orch = self._setup(tmp_path, monkeypatch, tag="exp-1")
        loaded = load_run_manifest(orch.run_dir / "manifest.json")
        assert loaded.tag == "exp-1"
        assert loaded.version == __version__
        assert "--tag exp-1" in loaded.command

    def test_tag_defaults_to_none_and_stays_out_of_command(self, tmp_path, monkeypatch):
        from panelcast.pipelines.manifest import load_run_manifest

        orch = self._setup(tmp_path, monkeypatch)
        loaded = load_run_manifest(orch.run_dir / "manifest.json")
        assert loaded.tag is None
        assert "--tag" not in loaded.command

    def test_legacy_manifest_without_version_loads(self, tmp_path, monkeypatch):
        from panelcast.pipelines.manifest import load_run_manifest

        orch = self._setup(tmp_path, monkeypatch)
        manifest_path = orch.run_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        del payload["version"]
        del payload["tag"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_run_manifest(manifest_path)
        assert loaded.version is None
        assert loaded.tag is None


class TestStageWeights:
    """Duration-weighted progress (presentation only, #161)."""

    def _orch(self, tmp_path, monkeypatch):
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        monkeypatch.chdir(tmp_path)  # keep the flat features cache out of reach
        return PipelineOrchestrator(PipelineConfig(), output_base=tmp_path / "outputs")

    def _stages(self, *names):
        from types import SimpleNamespace

        return [SimpleNamespace(name=n) for n in names]

    def test_no_history_degrades_to_equal_weights(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)
        weights = orch._stage_weights(self._stages("data", "train"), {})
        assert weights == {"data": 1.0, "train": 1.0}

    def test_previous_durations_weight_stages(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)
        weights = orch._stage_weights(
            self._stages("data", "train", "evaluate"), {"data": 4.0, "train": 7200.0}
        )
        assert weights["data"] == 4.0
        assert weights["train"] == 7200.0
        assert weights["evaluate"] == orch._FALLBACK_STAGE_SECONDS

    def test_predictor_overrides_previous_train_duration(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)
        monkeypatch.setattr(orch, "_predicted_train_seconds", lambda: 3600.0)
        weights = orch._stage_weights(self._stages("data", "train"), {"train": 60.0})
        assert weights["train"] == 3600.0

    def test_predicted_train_seconds_none_without_features(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)
        assert orch._predicted_train_seconds() is None
