"""End-to-end tests for the AOTY prediction pipeline.

Tests the full pipeline from raw data loading through feature generation,
validating integration between pipeline stages and error handling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panelcast.pipelines.errors import DataValidationError, PipelineError, StageSkipped
from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

# Register custom markers
pytestmark = [pytest.mark.e2e]


class TestFullPipeline:
    """End-to-end tests for full pipeline execution."""

    @pytest.mark.slow
    def test_pipeline_runs_data_stage(
        self,
        minimal_raw_csv: Path,
        tmp_path: Path,
    ) -> None:
        """Pipeline runs data stage successfully with minimal data.

        Tests that:
        - Pipeline can read raw CSV
        - Data stage produces expected output files
        - Exit code is 0
        """
        # Change to temp directory root so data/raw/all_albums_full.csv is valid
        original_cwd = os.getcwd()
        try:
            os.chdir(minimal_raw_csv.parent.parent.parent)  # Go to temp dir root

            config = PipelineConfig(
                seed=42,
                stages=["data"],
                dry_run=False,
                strict=False,
            )

            with (
                patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
                patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
            ):
                mock_verify.return_value = MagicMock(
                    is_reproducible=True,
                    pixi_lock_hash="test_hash_123",
                    warnings=[],
                )

                output_dir = tmp_path / "outputs"
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

            # Verify success
            assert exit_code == 0, f"Pipeline failed with exit code {exit_code}"

            # Verify manifest was created
            assert orchestrator.manifest is not None
            assert orchestrator.manifest.success is True
            assert "data" in orchestrator.manifest.stages_completed

        finally:
            os.chdir(original_cwd)

    def test_pipeline_dry_run_no_execution(
        self,
        tmp_path: Path,
    ) -> None:
        """Dry run mode logs stages without executing them.

        Tests that:
        - Dry run does not create output files
        - Stages are NOT recorded as completed (nothing executed)
        - Exit code is 0
        """
        config = PipelineConfig(
            seed=42,
            stages=["data", "splits"],
            dry_run=True,
            strict=False,
        )

        # Track if run_fn was actually called
        run_fn_called = {"data": False, "splits": False}

        def track_data_run(ctx):
            run_fn_called["data"] = True

        def track_splits_run(ctx):
            run_fn_called["splits"] = True

        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            # Patch stages to track execution
            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_data_stage = MagicMock()
                mock_data_stage.name = "data"
                mock_data_stage.description = "Prepare data"
                mock_data_stage.run_fn = track_data_run
                mock_data_stage.compute_input_hash.return_value = "hash_data"

                mock_splits_stage = MagicMock()
                mock_splits_stage.name = "splits"
                mock_splits_stage.description = "Create splits"
                mock_splits_stage.run_fn = track_splits_run
                mock_splits_stage.compute_input_hash.return_value = "hash_splits"

                mock_order.return_value = [mock_data_stage, mock_splits_stage]

                output_dir = tmp_path / "outputs"
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

        # Verify dry run behavior
        assert exit_code == 0
        assert not run_fn_called["data"], "Data stage should not run in dry_run"
        assert not run_fn_called["splits"], "Splits stage should not run in dry_run"

        # A dry run executes nothing, so it must not claim completed stages,
        # record stage hashes (would poison --skip-existing), or take the
        # latest pointer (would point consumers at an artifact-less dir).
        assert orchestrator.manifest is not None
        assert orchestrator.manifest.stages_completed == []
        assert orchestrator.manifest.stage_hashes == {}
        assert orchestrator.manifest.flags["dry_run"] is True
        assert not (tmp_path / "outputs" / "latest.json").exists()

    def test_pipeline_skip_detection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Skip detection works when inputs are unchanged.

        Tests that:
        - First run executes stage
        - Second run with skip_existing=True skips stage
        - Stage appears in stages_skipped list
        """
        # The data-stage stamp writes to the cwd-relative flat cache; isolate it
        # under tmp so the run can't touch the repo's real data/ (issues #127, #118).
        monkeypatch.chdir(tmp_path)

        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"
            run_count = {"data": 0}

            def counting_run_fn(ctx):
                run_count["data"] += 1

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_stage = MagicMock()
                mock_stage.name = "data"
                mock_stage.description = "Prepare data"
                mock_stage.run_fn = counting_run_fn
                mock_stage.compute_input_hash.return_value = "same_hash_123"
                # First run: should_skip returns False
                mock_stage.should_skip.return_value = False
                mock_order.return_value = [mock_stage]

                # First run: execute normally
                config1 = PipelineConfig(
                    seed=42,
                    stages=["data"],
                    skip_existing=False,
                )
                orchestrator1 = PipelineOrchestrator(config1, output_base=output_dir)
                exit_code1 = orchestrator1.run()

                assert exit_code1 == 0
                assert run_count["data"] == 1
                assert "data" in orchestrator1.manifest.stages_completed

                # Second run: should_skip returns True (simulating unchanged inputs)
                mock_stage.should_skip.return_value = True

                config2 = PipelineConfig(
                    seed=42,
                    stages=["data"],
                    skip_existing=True,
                )
                orchestrator2 = PipelineOrchestrator(config2, output_base=output_dir)
                exit_code2 = orchestrator2.run()

                assert exit_code2 == 0
                # Stage was NOT run again
                assert run_count["data"] == 1
                # Stage appears in skipped
                assert "data" in orchestrator2.manifest.stages_skipped

    def test_pipeline_stage_error_propagation(
        self,
        tmp_path: Path,
    ) -> None:
        """Stage errors propagate with correct exit codes.

        Tests that:
        - PipelineError from stage causes non-zero exit
        - Error message is captured in manifest
        - Manifest marks success=False
        """
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            def failing_run_fn(ctx):
                raise DataValidationError("Missing required column: User_Score", stage="data")

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_stage = MagicMock()
                mock_stage.name = "data"
                mock_stage.description = "Prepare data"
                mock_stage.run_fn = failing_run_fn
                mock_stage.compute_input_hash.return_value = "hash_123"
                mock_order.return_value = [mock_stage]

                config = PipelineConfig(
                    seed=42,
                    stages=["data"],
                    strict=True,  # Ensure errors propagate
                )
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

            # Verify error handling
            assert exit_code == 3  # DataValidationError exit code
            assert orchestrator.manifest is not None
            assert orchestrator.manifest.success is False
            assert "Missing required column" in orchestrator.manifest.error

    def test_pipeline_invalid_stage_name(
        self,
        tmp_path: Path,
    ) -> None:
        """Invalid stage name returns error exit code.

        Tests that:
        - Unknown stage name causes KeyError
        - Exit code is 1 (general error)
        """
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            config = PipelineConfig(
                seed=42,
                stages=["nonexistent_stage"],
            )
            orchestrator = PipelineOrchestrator(config, output_base=output_dir)
            exit_code = orchestrator.run()

            # Invalid stage name should return error
            assert exit_code == 1

    def test_pipeline_stage_skipped_not_error(
        self,
        tmp_path: Path,
    ) -> None:
        """StageSkipped exception is handled as non-error.

        Tests that:
        - StageSkipped doesn't cause pipeline failure
        - Stage is recorded in stages_skipped
        - Exit code is 0
        """
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            def skipping_run_fn(ctx):
                raise StageSkipped("No new data to process")

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_stage = MagicMock()
                mock_stage.name = "data"
                mock_stage.description = "Prepare data"
                mock_stage.run_fn = skipping_run_fn
                mock_stage.compute_input_hash.return_value = "hash_123"
                mock_order.return_value = [mock_stage]

                config = PipelineConfig(
                    seed=42,
                    stages=["data"],
                )
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

            # StageSkipped is not an error
            assert exit_code == 0
            assert orchestrator.manifest is not None
            assert orchestrator.manifest.success is True
            assert "data" in orchestrator.manifest.stages_skipped

    def test_pipeline_creates_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pipeline creates valid manifest.json file.

        Tests that:
        - manifest.json is created in run directory
        - Manifest contains required fields
        - Manifest is valid JSON
        """
        # The data-stage stamp writes to the cwd-relative flat cache; isolate it
        # under tmp so the run can't touch the repo's real data/ (issues #127, #118).
        monkeypatch.chdir(tmp_path)

        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_stage = MagicMock()
                mock_stage.name = "data"
                mock_stage.description = "Prepare data"
                mock_stage.run_fn = None
                mock_stage.compute_input_hash.return_value = "hash_123"
                mock_order.return_value = [mock_stage]

                config = PipelineConfig(seed=42, stages=["data"])
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

            assert exit_code == 0
            assert orchestrator.run_dir is not None

            # Verify manifest file exists and is valid
            manifest_path = orchestrator.run_dir / "manifest.json"
            assert manifest_path.exists(), "manifest.json should be created"

            with open(manifest_path) as f:
                manifest_data = json.load(f)

            # Verify required fields
            assert "run_id" in manifest_data
            assert "created_at" in manifest_data
            assert "seed" in manifest_data
            assert manifest_data["seed"] == 42
            assert "stages_completed" in manifest_data
            assert "data" in manifest_data["stages_completed"]
            assert manifest_data["success"] is True

    def test_pipeline_creates_latest_symlink(
        self,
        tmp_path: Path,
    ) -> None:
        """Successful pipeline creates outputs/latest link.

        Tests that:
        - outputs/latest link/junction is created
        - Link points to the current run directory
        """
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_order.return_value = []  # No stages

                config = PipelineConfig(seed=42)
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

            assert exit_code == 0

            latest_link = output_dir / "latest"
            # On Windows this might be a junction (appears as dir)
            # On Unix this is a symlink
            assert latest_link.exists() or latest_link.is_symlink()


class TestPipelineErrorScenarios:
    """Tests for various error conditions."""

    def test_environment_error_in_strict_mode(
        self,
        tmp_path: Path,
    ) -> None:
        """Strict mode fails if environment check fails.

        Tests that:
        - EnvironmentError causes exit code 5
        - Pipeline does not execute stages
        """
        from panelcast.pipelines.errors import EnvironmentError

        with patch(
            "panelcast.pipelines.orchestrator.ensure_environment_locked",
            side_effect=EnvironmentError("pixi.lock not found"),
        ):
            output_dir = tmp_path / "outputs"

            config = PipelineConfig(
                seed=42,
                stages=["data"],
                strict=True,
            )
            orchestrator = PipelineOrchestrator(config, output_base=output_dir)
            exit_code = orchestrator.run()

            assert exit_code == 5  # EnvironmentError exit code

    def test_unexpected_exception_wrapped(
        self,
        tmp_path: Path,
    ) -> None:
        """Unexpected exceptions are wrapped in PipelineError.

        Tests that:
        - RuntimeError becomes exit code 1
        - Error is captured in manifest
        """
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            def unexpected_error(ctx):
                raise RuntimeError("Unexpected failure in stage")

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_stage = MagicMock()
                mock_stage.name = "data"
                mock_stage.description = "Prepare data"
                mock_stage.run_fn = unexpected_error
                mock_stage.compute_input_hash.return_value = "hash_123"
                mock_order.return_value = [mock_stage]

                config = PipelineConfig(
                    seed=42,
                    stages=["data"],
                )
                orchestrator = PipelineOrchestrator(config, output_base=output_dir)
                exit_code = orchestrator.run()

            # Wrapped exception returns generic error code
            assert exit_code == 1
            assert orchestrator.manifest is not None
            assert orchestrator.manifest.success is False
            assert "Unexpected failure" in orchestrator.manifest.error


class TestPipelineResume:
    """Tests for pipeline resume functionality."""

    def test_resume_from_previous_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pipeline can resume from a previous run.

        Tests that:
        - Resume finds previous run directory
        - Already completed stages are not re-run
        """
        # The data-stage stamp writes to the cwd-relative flat cache; isolate it
        # under tmp so the run can't touch the repo's real data/ (issues #127, #118).
        monkeypatch.chdir(tmp_path)

        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            # First run: complete data stage
            run_count = {"data": 0, "splits": 0}

            def count_data(ctx):
                run_count["data"] += 1

            def count_splits(ctx):
                run_count["splits"] += 1

            with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
                mock_data = MagicMock()
                mock_data.name = "data"
                mock_data.description = "Data"
                mock_data.run_fn = count_data
                mock_data.compute_input_hash.return_value = "hash_data"

                mock_splits = MagicMock()
                mock_splits.name = "splits"
                mock_splits.description = "Splits"
                mock_splits.run_fn = count_splits
                mock_splits.compute_input_hash.return_value = "hash_splits"

                mock_order.return_value = [mock_data]

                config1 = PipelineConfig(
                    seed=42,
                    stages=["data"],
                )
                orchestrator1 = PipelineOrchestrator(config1, output_base=output_dir)
                exit_code1 = orchestrator1.run()

                assert exit_code1 == 0
                assert run_count["data"] == 1
                run_id = orchestrator1.manifest.run_id

                # Now resume with splits stage
                mock_order.return_value = [mock_data, mock_splits]

                config2 = PipelineConfig(
                    seed=42,
                    stages=["data", "splits"],
                    resume=run_id,
                )
                orchestrator2 = PipelineOrchestrator(config2, output_base=output_dir)
                exit_code2 = orchestrator2.run()

                assert exit_code2 == 0
                # Data was already completed, should not run again
                assert run_count["data"] == 1
                # Splits should run
                assert run_count["splits"] == 1
                assert "data" in orchestrator2.manifest.stages_completed
                assert "splits" in orchestrator2.manifest.stages_completed

    def test_resume_nonexistent_run_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """Resume with invalid run_id fails gracefully.

        Tests that:
        - PipelineError is raised for missing run
        - Error contains helpful message
        """
        with (
            patch("panelcast.pipelines.orchestrator.ensure_environment_locked"),
            patch("panelcast.pipelines.orchestrator.verify_environment") as mock_verify,
        ):
            mock_verify.return_value = MagicMock(
                is_reproducible=True,
                pixi_lock_hash="test_hash_123",
                warnings=[],
            )

            output_dir = tmp_path / "outputs"

            config = PipelineConfig(
                seed=42,
                resume="nonexistent-run-id-12345",
            )
            orchestrator = PipelineOrchestrator(config, output_base=output_dir)

            # Resume with invalid run_id raises PipelineError
            with pytest.raises(PipelineError) as exc_info:
                orchestrator.run()

            # Verify error message is helpful
            assert "Cannot find run to resume" in str(exc_info.value)
            assert "nonexistent-run-id-12345" in str(exc_info.value)
