"""Comprehensive coverage tests for cli.py, orchestrator.py, and stages.py.

Targets uncovered lines in:
- cli.py (34% -> higher): export-figures execution, preflight-full
- orchestrator.py (87% -> higher): resume from failed, skip-existing manifest loading,
  Windows junction, command string with beta prior params, capture_stage_input_hashes
- stages.py (92% -> higher): stage run function wrappers
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from panelcast.cli import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_mocks(monkeypatch, exit_code: int = 0):
    """Patch PipelineConfig and run_pipeline, returning capture dict."""
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
# CLI: export-figures command execution
# ============================================================================


class TestExportFiguresExecution:
    """Tests for export-figures command execution paths."""

    def test_export_figures_no_data_exits_with_error(self, monkeypatch):
        """Export-figures with no dashboard data exits with code 1."""
        mock_data = SimpleNamespace(
            predictions=None,
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        result = runner.invoke(app, ["export-figures"])
        assert result.exit_code == 1
        assert "No data available" in result.output

    def test_export_figures_with_predictions(self, monkeypatch, tmp_path):
        """Export-figures renders predictions when available."""
        mock_data = SimpleNamespace(
            predictions={
                "y_true": [1, 2, 3],
                "y_pred_mean": [1.1, 2.1, 3.1],
                "y_pred_lower": [0.5, 1.5, 2.5],
                "y_pred_upper": [1.5, 2.5, 3.5],
            },
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        mock_fig = MagicMock()
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: mock_fig,
        )

        export_results = {"predictions": [tmp_path / "pred.svg"]}
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda **kwargs: export_results,
        )

        result = runner.invoke(
            app, ["export-figures", "--output", str(tmp_path), "--formats", "svg"]
        )
        assert result.exit_code == 0
        assert "Exported" in result.output

    def test_export_figures_with_coefficients_and_reliability(self, monkeypatch, tmp_path):
        """Export-figures renders coefficients and reliability when available."""
        mock_data = SimpleNamespace(
            predictions=None,
            coefficients={"param": "beta", "mean": 1.0},
            reliability={
                "predicted_probs": [0.1, 0.5],
                "observed_freq": [0.12, 0.48],
                "counts": [10, 20],
            },
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        mock_fig = MagicMock()
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_forest_plot",
            lambda coef: mock_fig,
        )
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_reliability_plot",
            lambda probs, freq, counts: mock_fig,
        )

        export_results = {
            "coefficients": [tmp_path / "coef.svg"],
            "reliability": [tmp_path / "rel.svg"],
        }
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda **kwargs: export_results,
        )

        result = runner.invoke(
            app, ["export-figures", "--output", str(tmp_path), "--formats", "svg"]
        )
        assert result.exit_code == 0

    def test_export_figures_with_idata_trace_plot(self, monkeypatch, tmp_path):
        """Export-figures renders trace plot from idata when available."""
        import numpy as np

        # Mock idata with posterior
        mock_posterior_var = MagicMock()
        mock_posterior_var.values = np.random.randn(2, 100)

        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["mu_artist"]
        mock_posterior.__getitem__ = lambda self, key: mock_posterior_var

        mock_idata = MagicMock()
        mock_idata.posterior = mock_posterior

        mock_data = SimpleNamespace(
            predictions=None,
            coefficients=None,
            reliability=None,
            idata=mock_idata,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        mock_fig = MagicMock()
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_trace_plot",
            lambda samples, var_name: mock_fig,
        )

        export_results = {"trace": [tmp_path / "trace.svg"]}
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda **kwargs: export_results,
        )

        result = runner.invoke(
            app, ["export-figures", "--output", str(tmp_path), "--formats", "svg"]
        )
        assert result.exit_code == 0

    def test_export_figures_idata_trace_error_handled(self, monkeypatch, tmp_path):
        """Export-figures handles trace plot errors gracefully."""
        mock_idata = MagicMock()
        mock_idata.posterior = MagicMock(side_effect=Exception("bad idata"))

        mock_data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1],
                "y_pred_lower": [0],
                "y_pred_upper": [2],
            },
            coefficients=None,
            reliability=None,
            idata=mock_idata,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        mock_fig = MagicMock()
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: mock_fig,
        )

        export_results = {"predictions": [tmp_path / "pred.svg"]}
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda **kwargs: export_results,
        )

        result = runner.invoke(
            app, ["export-figures", "--output", str(tmp_path), "--formats", "svg"]
        )
        # Should succeed despite idata error
        assert result.exit_code == 0

    def test_export_figures_kaleido_warning_for_png(self, monkeypatch, tmp_path):
        """Export-figures warns when Kaleido not available for PNG."""
        mock_data = SimpleNamespace(
            predictions=None,
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: False,
        )

        result = runner.invoke(
            app, ["export-figures", "--output", str(tmp_path), "--formats", "png"]
        )
        # Should still attempt but warn
        # Exit code 1 because no data
        assert result.exit_code == 1

    def test_export_figures_with_custom_dimensions(self, monkeypatch, tmp_path):
        """Export-figures passes custom width/height/scale."""
        captured = {}
        mock_data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1],
                "y_pred_lower": [0],
                "y_pred_upper": [2],
            },
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        mock_fig = MagicMock()
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: mock_fig,
        )

        def fake_export(**kwargs):
            captured.update(kwargs)
            return {"predictions": [tmp_path / "pred.svg"]}

        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            fake_export,
        )

        result = runner.invoke(
            app,
            [
                "export-figures",
                "--output",
                str(tmp_path),
                "--formats",
                "svg",
                "--width",
                "1200",
                "--height",
                "800",
                "--scale",
                "3.0",
            ],
        )
        assert result.exit_code == 0
        assert captured["width"] == 1200
        assert captured["height"] == 800
        assert captured["scale"] == 3.0

    def test_export_figures_with_run_dir(self, monkeypatch, tmp_path):
        """Export-figures passes --run to load_dashboard_data."""
        captured = {}
        mock_data = SimpleNamespace(
            predictions=None,
            coefficients=None,
            reliability=None,
            idata=None,
        )

        def fake_load(run_path):
            captured["run_path"] = run_path
            return mock_data

        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            fake_load,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            lambda: True,
        )

        result = runner.invoke(
            app, ["export-figures", "--run", str(tmp_path / "myrun"), "--formats", "svg"]
        )
        # Exit 1 because no data, but that's fine - we check the run_path was passed
        assert result.exit_code == 1
        assert captured["run_path"] == tmp_path / "myrun"

    def test_export_figures_svg_only_no_kaleido_check(self, monkeypatch, tmp_path):
        """Export-figures with svg-only format does not check Kaleido."""
        kaleido_checked = {"called": False}

        def fake_ensure():
            kaleido_checked["called"] = True
            return True

        mock_data = SimpleNamespace(
            predictions={
                "y_true": [1],
                "y_pred_mean": [1],
                "y_pred_lower": [0],
                "y_pred_upper": [2],
            },
            coefficients=None,
            reliability=None,
            idata=None,
        )
        monkeypatch.setattr(
            "panelcast.visualization.dashboard.load_dashboard_data",
            lambda run_path: mock_data,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.ensure_kaleido_chrome",
            fake_ensure,
        )
        mock_fig = MagicMock()
        monkeypatch.setattr(
            "panelcast.visualization.charts.create_predictions_plot",
            lambda *args: mock_fig,
        )
        monkeypatch.setattr(
            "panelcast.visualization.export.export_all_figures",
            lambda **kwargs: {"predictions": [tmp_path / "pred.svg"]},
        )

        result = runner.invoke(
            app, ["export-figures", "--output", str(tmp_path), "--formats", "svg"]
        )
        assert result.exit_code == 0
        # Kaleido should NOT have been checked since no raster formats
        assert kaleido_checked["called"] is False


# ============================================================================
# CLI: preflight-full execution
# ============================================================================


class TestPreflightFullExecution:
    """Tests for --preflight-full execution path in cli.py."""

    def test_preflight_full_missing_data_exits_2(self, monkeypatch, tmp_path):
        """--preflight-full exits with code 2 when data files are missing."""
        # Use monkeypatch to change cwd to a directory without required data files
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["run", "--preflight-full"])
        assert result.exit_code == 2
        output = strip_ansi(result.output)
        assert "MISSING" in output or "Error" in output


# ============================================================================
# Orchestrator: resume from failed directory
# ============================================================================


class TestOrchestratorResumeFromFailed:
    """Tests for resuming from outputs/failed/ directory."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    @patch("panelcast.pipelines.orchestrator.get_execution_order")
    def test_resume_from_failed_directory(self, mock_order, mock_verify, mock_ensure, tmp_path):
        """Resume moves run back from failed/ and continues."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )
        mock_order.return_value = []

        run_id = "2026-01-20_120000"
        failed_dir = tmp_path / "failed" / run_id
        failed_dir.mkdir(parents=True)

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
                "target_accept": 0.90,
                "max_tree_depth": 10,
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
                "calibration_intervals": [0.80, 0.95],
                "coverage_tolerance": 0.03,
                "prediction_interval": 0.95,
                "evaluate_secondary_split": True,
                "enforce_lockfile": True,
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
            "stages_completed": ["data"],
            "stages_skipped": [],
            "outputs": {},
            "success": False,
            "error": "previous error",
            "duration_seconds": 0.0,
        }
        (failed_dir / "manifest.json").write_text(json.dumps(manifest_data))

        config = PipelineConfig(resume=run_id)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        exit_code = orchestrator.run()

        assert exit_code == 0
        # Should have moved from failed/ back to normal dir
        assert (tmp_path / run_id).exists()
        assert not failed_dir.exists()


# ============================================================================
# Orchestrator: skip_existing with previous manifest loading
# ============================================================================


class TestOrchestratorSkipExistingManifest:
    """Tests for skip-existing manifest loading edge cases."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_skip_existing_loads_latest_manifest(self, mock_verify, mock_ensure, tmp_path):
        """Skip-existing loads manifest from outputs/latest symlink."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        # Create previous run with manifest
        prev_run = tmp_path / "prev_run"
        prev_run.mkdir()
        prev_manifest = {
            "run_id": "prev_run",
            "created_at": "2026-01-19T12:00:00",
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
                "target_accept": 0.90,
                "max_tree_depth": 10,
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
                "calibration_intervals": [0.80, 0.95],
                "coverage_tolerance": 0.03,
                "prediction_interval": 0.95,
                "evaluate_secondary_split": True,
                "enforce_lockfile": True,
            },
            "seed": 42,
            "git": {
                "commit": "abc",
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
                "pixi_lock_hash": "abc",
            },
            "input_hashes": {},
            "stage_hashes": {"data": "hash123"},
            "stages_completed": ["data"],
            "stages_skipped": [],
            "outputs": {},
            "success": True,
            "error": None,
            "duration_seconds": 1.0,
        }
        (prev_run / "manifest.json").write_text(json.dumps(prev_manifest))

        # Create symlink latest -> prev_run
        import os

        latest = tmp_path / "latest"
        os.symlink(prev_run, latest, target_is_directory=True)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.should_skip.return_value = True
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(skip_existing=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            # Stage should have been checked for skip
            mock_stage.should_skip.assert_called()

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_skip_existing_broken_symlink_handled(self, mock_verify, mock_ensure, tmp_path):
        """Skip-existing handles broken latest symlink gracefully."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        # Create a broken symlink
        import os

        latest = tmp_path / "latest"
        os.symlink(tmp_path / "nonexistent_run", latest, target_is_directory=True)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(return_value=None)
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(skip_existing=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Should succeed despite broken symlink
            assert exit_code == 0

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_skip_existing_disabled_on_flag_change(self, mock_verify, mock_ensure, tmp_path):
        """Skip-existing is disabled when flags differ from previous manifest."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        # Create previous run with different min_ratings
        prev_run = tmp_path / "prev_run"
        prev_run.mkdir()
        prev_manifest = {
            "run_id": "prev_run",
            "created_at": "2026-01-19T12:00:00",
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
                "target_accept": 0.90,
                "max_tree_depth": 10,
                "chain_method": "sequential",
                "rhat_threshold": 1.01,
                "ess_threshold": 400,
                "allow_divergences": False,
                "min_ratings": 5,  # Different from current
                "min_albums_filter": 2,
                "enable_genre": True,
                "enable_artist": True,
                "enable_temporal": True,
                "n_exponent": 0.0,
                "learn_n_exponent": False,
                "n_exponent_alpha": 2.0,
                "n_exponent_beta": 4.0,
                "n_exponent_prior": "logit-normal",
                "calibration_intervals": [0.80, 0.95],
                "coverage_tolerance": 0.03,
                "prediction_interval": 0.95,
                "evaluate_secondary_split": True,
                "enforce_lockfile": True,
            },
            "seed": 42,
            "git": {
                "commit": "abc",
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
                "pixi_lock_hash": "abc",
            },
            "input_hashes": {},
            "stage_hashes": {"data": "hash123"},
            "stages_completed": ["data"],
            "stages_skipped": [],
            "outputs": {},
            "success": True,
            "error": None,
            "duration_seconds": 1.0,
        }
        (prev_run / "manifest.json").write_text(json.dumps(prev_manifest))

        import os

        latest = tmp_path / "latest"
        os.symlink(prev_run, latest, target_is_directory=True)

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(return_value=None)
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(skip_existing=True, min_ratings=10)  # Different!
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            # should_skip is called with None (since manifest was cleared due to flag diff)
            mock_stage.should_skip.assert_called_with(None, force=False)


# ============================================================================
# Orchestrator: command string with beta prior and n_exponent
# ============================================================================


class TestOrchestratorCommandStringAdvanced:
    """Tests for command string building with advanced options."""

    def test_command_with_beta_prior_params(self, tmp_path):
        """Command string includes beta prior params when using beta prior."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(
            learn_n_exponent=True,
            n_exponent_prior="beta",
            n_exponent_alpha=3.0,
            n_exponent_beta=5.0,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--learn-n-exponent" in cmd
        assert "--n-exponent-prior beta" in cmd
        assert "--n-exponent-alpha 3.0" in cmd
        assert "--n-exponent-beta 5.0" in cmd

    def test_command_omits_beta_params_for_logit_normal(self, tmp_path):
        """Command string omits beta params when using logit-normal prior."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(
            learn_n_exponent=True,
            n_exponent_prior="logit-normal",
            n_exponent_alpha=3.0,  # Non-default but should NOT appear
            n_exponent_beta=5.0,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--learn-n-exponent" in cmd
        assert "--n-exponent-alpha" not in cmd
        assert "--n-exponent-beta" not in cmd

    def test_command_with_n_exponent_fixed(self, tmp_path):
        """Command string includes fixed n_exponent when not learning."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(
            n_exponent=0.5,
            learn_n_exponent=False,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--n-exponent 0.5" in cmd
        assert "--learn-n-exponent" not in cmd

    def test_command_omits_n_exponent_when_learning(self, tmp_path):
        """Command string omits --n-exponent when --learn-n-exponent is set."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(
            n_exponent=0.5,  # Should be omitted because learn is True
            learn_n_exponent=True,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--n-exponent 0.5" not in cmd
        assert "--learn-n-exponent" in cmd

    def test_command_with_calibration_intervals(self, tmp_path):
        """Command string includes non-default calibration intervals."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(
            calibration_intervals=(0.50, 0.80, 0.90, 0.95),
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--calibration-intervals" in cmd

    def test_command_with_coverage_tolerance(self, tmp_path):
        """Command string includes non-default coverage tolerance."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(coverage_tolerance=0.05)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--coverage-tolerance 0.05" in cmd

    def test_command_with_prediction_interval(self, tmp_path):
        """Command string includes non-default prediction interval."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(prediction_interval=0.90)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--prediction-interval 0.9" in cmd

    def test_command_with_min_ratings_and_albums(self, tmp_path):
        """Command string includes non-default data filtering."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(min_ratings=20, min_albums_filter=5)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--min-ratings 20" in cmd
        assert "--min-albums 5" in cmd

    def test_command_with_chain_method(self, tmp_path):
        """Command string includes non-default chain method."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(chain_method="parallel")
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--chain-method parallel" in cmd

    def test_command_with_rhat_and_ess(self, tmp_path):
        """Command string includes non-default convergence thresholds."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig(rhat_threshold=1.05, ess_threshold=200)
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--rhat-threshold 1.05" in cmd
        assert "--ess-threshold 200" in cmd


# ============================================================================
# Orchestrator: _capture_stage_input_hashes
# ============================================================================


class TestCaptureStageInputHashes:
    """Tests for _capture_stage_input_hashes method."""

    def test_captures_existing_file_hashes(self, tmp_path):
        """Captures hashes for files that exist."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        input_file = tmp_path / "input.csv"
        input_file.write_text("data")

        stage = MagicMock()
        stage.name = "test"
        stage.input_paths = [input_file]

        hashes = orchestrator._capture_stage_input_hashes(stage)
        assert str(input_file) in hashes
        assert len(hashes[str(input_file)]) == 64

    def test_skips_nonexistent_files(self, tmp_path):
        """Skips files that don't exist."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        stage = MagicMock()
        stage.name = "test"
        stage.input_paths = [tmp_path / "nonexistent.csv"]

        hashes = orchestrator._capture_stage_input_hashes(stage)
        assert len(hashes) == 0

    def test_handles_hash_error(self, tmp_path):
        """Handles errors during hashing gracefully."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        input_file = tmp_path / "input.csv"
        input_file.write_text("data")

        stage = MagicMock()
        stage.name = "test"
        stage.input_paths = [input_file]

        with patch("panelcast.pipelines.orchestrator.sha256_path", side_effect=OSError("perm")):
            hashes = orchestrator._capture_stage_input_hashes(stage)
            assert len(hashes) == 0


# ============================================================================
# Orchestrator: _create_latest_link edge cases
# ============================================================================


class TestCreateLatestLinkEdgeCases:
    """Tests for _create_latest_link edge cases."""

    def test_latest_link_replaces_existing(self, tmp_path):
        """Creating latest link replaces existing symlink."""
        import os

        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        # Create initial "old" run dir and symlink
        old_run = tmp_path / "old_run"
        old_run.mkdir()
        latest = tmp_path / "latest"
        os.symlink(old_run, latest, target_is_directory=True)

        # Now create new run and link to it
        new_run = tmp_path / "new_run"
        new_run.mkdir()
        orchestrator.run_dir = new_run
        orchestrator._create_latest_link()

        assert latest.resolve() == new_run.resolve()

    def test_latest_link_no_run_dir_noop(self, tmp_path):
        """_create_latest_link does nothing when run_dir is None."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.run_dir = None

        # Should not raise
        orchestrator._create_latest_link()
        assert not (tmp_path / "latest").exists()


# ============================================================================
# Orchestrator: convergence error non-strict mode
# ============================================================================


class TestConvergenceNonStrict:
    """Tests for convergence errors in non-strict mode."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_convergence_warning_in_non_strict_logs_warning(
        self, mock_verify, mock_ensure, tmp_path
    ):
        """Non-strict mode logs convergence warning (catches ConvergenceError)."""
        from panelcast.pipelines.errors import ConvergenceError
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "train"
            mock_stage.description = "Train"
            mock_stage.run_fn = MagicMock(side_effect=ConvergenceError("R-hat too high", "train"))
            mock_stage.compute_input_hash.return_value = "hash"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(strict=False)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Non-strict mode must *catch* the ConvergenceError (not let it
            # propagate) and report a non-zero exit rather than crashing.
            assert exit_code == 1
            assert mock_stage.run_fn.called


# ============================================================================
# Orchestrator: _execute_stages with already-completed stages (resume)
# ============================================================================


class TestExecuteStagesResume:
    """Tests for skipping already-completed stages on resume."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_already_completed_stages_skipped(
        self, mock_verify, mock_ensure, tmp_path, monkeypatch
    ):
        """Already-completed stages from manifest are skipped on resume."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        monkeypatch.chdir(tmp_path)  # a completed stage stamps the repo's data dirs
        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

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
                "target_accept": 0.90,
                "max_tree_depth": 10,
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
                "calibration_intervals": [0.80, 0.95],
                "coverage_tolerance": 0.03,
                "prediction_interval": 0.95,
                "evaluate_secondary_split": True,
                "enforce_lockfile": True,
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
            "stage_hashes": {"data": "hash1"},
            "stages_completed": ["data"],
            "stages_skipped": [],
            "outputs": {},
            "success": False,
            "error": "previous error",
            "duration_seconds": 0.0,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest_data))

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            # Two stages: data (completed) and splits (new)
            data_stage = MagicMock()
            data_stage.name = "data"
            data_stage.description = "Prepare data"
            data_stage.run_fn = MagicMock(side_effect=RuntimeError("should not be called"))
            data_stage.compute_input_hash.return_value = "hash1"

            splits_stage = MagicMock()
            splits_stage.name = "splits"
            splits_stage.description = "Create splits"
            splits_stage.run_fn = MagicMock(return_value=None)
            splits_stage.compute_input_hash.return_value = "hash2"

            mock_order.return_value = [data_stage, splits_stage]

            config = PipelineConfig(resume=run_id)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            # data stage run_fn should NOT have been called (already completed)
            data_stage.run_fn.assert_not_called()
            # splits stage run_fn SHOULD have been called
            splits_stage.run_fn.assert_called_once()


# ============================================================================
# Orchestrator: stage with no run_fn
# ============================================================================


class TestStageNoRunFn:
    """Test execution of stages with no run function."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_stage_without_run_fn_logs_warning(self, mock_verify, mock_ensure, tmp_path):
        """Stage with run_fn=None logs warning and continues."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "placeholder"
            mock_stage.description = "Placeholder"
            mock_stage.run_fn = None
            mock_stage.compute_input_hash.return_value = ""
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            assert "placeholder" in orchestrator.manifest.stages_completed


# ============================================================================
# Orchestrator: environment not reproducible warning
# ============================================================================


class TestEnvironmentWarning:
    """Test environment warning path."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_non_reproducible_environment_warns(self, mock_verify, mock_ensure, tmp_path):
        """Non-reproducible environment logs warning but continues."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=False,
            pixi_lock_hash=None,
            warnings=["pixi.lock not found"],
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []

            config = PipelineConfig(enforce_lockfile=False)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0


# ============================================================================
# Orchestrator: _handle_failure with PermissionError on move
# ============================================================================


class TestHandleFailurePermissionError:
    """Tests for failure handling when file move fails."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_permission_error_on_move_to_failed(self, mock_verify, mock_ensure, tmp_path):
        """PermissionError during move-to-failed is handled gracefully."""
        from panelcast.pipelines.errors import StageError
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            with patch(
                "panelcast.pipelines.orchestrator.shutil.move",
                side_effect=PermissionError("locked"),
            ):
                config = PipelineConfig()
                orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                exit_code = orchestrator.run()

                # Should return error code but not crash
                assert exit_code == 4


# ============================================================================
# Orchestrator: _skip_flag_differences edge cases
# ============================================================================


class TestSkipFlagDifferencesEdgeCases:
    """Tests for _skip_flag_differences edge cases."""

    def test_manifest_none_returns_empty(self, tmp_path):
        """Returns empty list when manifest is None."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.manifest = None

        prev = MagicMock(flags={"seed": 42})
        assert orchestrator._skip_flag_differences(prev) == []

    def test_new_keys_in_current_detected(self, tmp_path):
        """New flags present in current but not previous are detected."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        orchestrator.manifest = MagicMock(flags={"seed": 42, "new_flag": True})
        prev = MagicMock(flags={"seed": 42})

        diffs = orchestrator._skip_flag_differences(prev)
        assert "new_flag" in diffs


# ============================================================================
# Stages: stage run function wrappers
# ============================================================================


class TestStageRunFunctions:
    """Tests for stage run function wrappers in stages.py."""

    def _make_ctx(self, tmp_path):
        """Create minimal StageContext."""
        from panelcast.pipelines.stages import StageContext

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return StageContext(
            run_dir=run_dir,
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )

    def test_run_splits_stage(self, tmp_path, monkeypatch):
        """_run_splits_stage calls create_splits correctly."""
        from panelcast.pipelines.stages import _run_splits_stage

        ctx = self._make_ctx(tmp_path)
        captured = {}

        def fake_create_splits(config):
            captured["config"] = config
            return {"splits": "ok"}

        monkeypatch.setattr(
            "panelcast.pipelines.create_splits.create_splits",
            fake_create_splits,
        )
        monkeypatch.setattr(
            "panelcast.pipelines.create_splits.SplitConfig",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        result = _run_splits_stage(ctx)
        assert result == {"splits": "ok"}
        assert captured["config"].random_state == 42
        assert captured["config"].min_ratings == 10

    def test_run_features_stage(self, tmp_path, monkeypatch):
        """_run_features_stage calls build_features correctly."""
        from panelcast.pipelines.stages import _run_features_stage

        ctx = self._make_ctx(tmp_path)
        captured = {}

        def fake_build_features(ctx_arg):
            captured["ctx"] = ctx_arg
            return {"features": "ok"}

        monkeypatch.setattr(
            "panelcast.pipelines.build_features.build_features",
            fake_build_features,
        )

        result = _run_features_stage(ctx)
        assert result == {"features": "ok"}
        assert captured["ctx"] is ctx

    def test_run_train_stage(self, tmp_path, monkeypatch):
        """_run_train_stage calls train_models correctly."""
        from panelcast.pipelines.stages import _run_train_stage

        ctx = self._make_ctx(tmp_path)
        captured = {}

        def fake_train_models(ctx_arg):
            captured["ctx"] = ctx_arg
            return {"model": "trained"}

        monkeypatch.setattr(
            "panelcast.pipelines.train_bayes.train_models",
            fake_train_models,
        )

        result = _run_train_stage(ctx)
        assert result == {"model": "trained"}

    def test_run_evaluate_stage(self, tmp_path, monkeypatch):
        """_run_evaluate_stage calls evaluate_models correctly."""
        from panelcast.pipelines.stages import _run_evaluate_stage

        ctx = self._make_ctx(tmp_path)
        captured = {}

        def fake_evaluate_models(ctx_arg):
            captured["ctx"] = ctx_arg
            return {"metrics": "computed"}

        monkeypatch.setattr(
            "panelcast.pipelines.evaluate.evaluate_models",
            fake_evaluate_models,
        )

        result = _run_evaluate_stage(ctx)
        assert result == {"metrics": "computed"}

    def test_run_predict_stage(self, tmp_path, monkeypatch):
        """_run_predict_stage calls predict_next_events correctly."""
        from panelcast.pipelines.stages import _run_predict_stage

        ctx = self._make_ctx(tmp_path)
        captured = {}

        def fake_predict(ctx_arg):
            captured["ctx"] = ctx_arg
            return {"predictions": "done"}

        monkeypatch.setattr(
            "panelcast.pipelines.predict_next.predict_next_events",
            fake_predict,
        )

        result = _run_predict_stage(ctx)
        assert result == {"predictions": "done"}

    def test_run_report_stage(self, tmp_path, monkeypatch):
        """_run_report_stage calls generate_publication_artifacts correctly."""
        from panelcast.pipelines.stages import _run_report_stage

        ctx = self._make_ctx(tmp_path)
        captured = {}

        def fake_generate(ctx_arg):
            captured["ctx"] = ctx_arg
            return {"report": "generated"}

        monkeypatch.setattr(
            "panelcast.pipelines.publication.generate_publication_artifacts",
            fake_generate,
        )

        result = _run_report_stage(ctx)
        assert result == {"report": "generated"}


# ============================================================================
# Orchestrator: invalid stage in get_execution_order
# ============================================================================


class TestOrchestratorInvalidStage:
    """Test orchestrator handles invalid stage from config."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_invalid_stage_returns_exit_code_1(self, mock_verify, mock_ensure, tmp_path):
        """Invalid stage name returns exit code 1."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        config = PipelineConfig(stages=["nonexistent_stage"])
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        exit_code = orchestrator.run()

        assert exit_code == 1


# ============================================================================
# Orchestrator: empty stages list
# ============================================================================


class TestOrchestratorEmptyStages:
    """Test orchestrator with empty stage list."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_empty_stages_succeeds(self, mock_verify, mock_ensure, tmp_path):
        """Empty stages list completes successfully."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

        mock_verify.return_value = MagicMock(
            is_reproducible=True, pixi_lock_hash="abc123", warnings=[]
        )

        with patch("panelcast.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            assert orchestrator.manifest.success is True


# ============================================================================
# Orchestrator: _record_stage_outputs with Path value in run_result
# ============================================================================


class TestRecordStageOutputsPath:
    """Test _record_stage_outputs with Path values."""

    @patch("panelcast.pipelines.orchestrator.ensure_environment_locked")
    @patch("panelcast.pipelines.orchestrator.verify_environment")
    def test_records_path_object_in_run_result(self, mock_verify, mock_ensure, tmp_path):
        """_record_stage_outputs records Path objects from run_result."""
        from panelcast.pipelines.orchestrator import PipelineConfig, PipelineOrchestrator

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

        # Pass Path object instead of string
        run_result = {"path_key": output_file}
        orchestrator._record_stage_outputs(stage, run_result=run_result)

        assert "test:path_key" in orchestrator.manifest.outputs


# ============================================================================
# CLI: main() entry point
# ============================================================================


class TestMainEntryPoint:
    """Tests for main() CLI entry point."""

    def test_main_callable(self):
        """main() function exists and is callable."""
        from panelcast.cli import main

        assert callable(main)
