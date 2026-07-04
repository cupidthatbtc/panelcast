"""The `panelcast select` command: dry-run plan and argument handling."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from panelcast.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = str(REPO_ROOT / "configs" / "select.yaml")


class TestDryRun:
    def test_aoty_dry_run_prints_full_space(self):
        result = runner.invoke(app, ["select", "--dry-run", "--config", CONFIG])
        assert result.exit_code == 0
        out = result.stdout
        # The frozen options must appear — proof they are re-tried, not pruned.
        assert "ar1" in out
        assert "beta_ceiling" in out
        assert "errors_in_variables" in out
        assert "one-factor-at-a-time" in out

    def test_effort_tier_shown(self):
        result = runner.invoke(
            app, ["select", "--dry-run", "--effort", "quick", "--config", CONFIG]
        )
        assert result.exit_code == 0
        assert "effort=quick" in result.stdout

    def test_aero_dry_run_shows_pruning(self):
        descriptor = str(REPO_ROOT / "examples" / "aerospace" / "descriptor.yaml")
        result = runner.invoke(
            app, ["select", "--dry-run", "--dataset", descriptor, "--config", CONFIG]
        )
        assert result.exit_code == 0
        assert "pruned:" in result.stdout
        assert "beta_binomial" in result.stdout


class TestArgs:
    def test_unknown_effort_errors(self):
        result = runner.invoke(
            app, ["select", "--dry-run", "--effort", "turbo", "--config", CONFIG]
        )
        assert result.exit_code == 1
        assert "Unknown effort tier" in result.stdout

    def test_max_fits_caps_plan(self):
        result = runner.invoke(
            app, ["select", "--dry-run", "--max-fits", "5", "--config", CONFIG]
        )
        assert result.exit_code == 0
        assert "planned fits: " in result.stdout
