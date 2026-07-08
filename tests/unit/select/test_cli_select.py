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


class TestConfigHandling:
    def test_explicit_missing_config_errors(self):
        # A typo'd --config must not silently run under shipped defaults.
        result = runner.invoke(
            app, ["select", "--dry-run", "--config", "definitely/missing.yaml"]
        )
        assert result.exit_code == 2
        assert "config file not found" in result.output

    def test_malformed_config_errors(self, tmp_path):
        bad = tmp_path / "select.yaml"
        bad.write_text("rules: {promote_z: 4.0", encoding="utf-8")
        result = runner.invoke(app, ["select", "--dry-run", "--config", str(bad)])
        assert result.exit_code == 1
        assert "malformed select config" in result.output

    def test_effective_rules_echoed_in_plan(self):
        result = runner.invoke(app, ["select", "--dry-run", "--config", CONFIG])
        assert result.exit_code == 0
        assert "promote_z=" in result.stdout
        assert "coverage tolerance" in result.stdout


class TestAvailableColumns:
    def _capture(self, monkeypatch, frame):
        import panelcast.cli.select_cmd as sc
        import panelcast.select.orchestrate as orch

        captured: dict = {}
        monkeypatch.setattr(sc, "_prepared_paths", lambda descriptor: None)
        monkeypatch.setattr(sc, "_load_prepared_frame", lambda descriptor: frame)

        def fake_run_select(*a, **k):
            captured.update(k)
            return {"report_dir": "rd", "winner_arm": None, "promotable": [],
                    "n_arms_scored": 0, "ledger": "l"}

        monkeypatch.setattr(orch, "run_select", fake_run_select)
        return captured

    def test_columns_passed_when_frame_loads(self, monkeypatch):
        # Data-aware structural pruning is only live when the CLI hands the
        # prepared frame's columns to the sweep.
        import pandas as pd

        df = pd.DataFrame({"Artist": ["a"], "User_Score": [70.0], "genre_group": ["g"]})
        captured = self._capture(monkeypatch, (df, ["genre_group"]))
        result = runner.invoke(app, ["select", "--effort", "quick", "--config", CONFIG])
        assert result.exit_code == 0
        assert captured["available_columns"] == frozenset(df.columns)

    def test_columns_none_without_frame(self, monkeypatch):
        captured = self._capture(monkeypatch, (None, None))
        result = runner.invoke(app, ["select", "--config", CONFIG])
        assert result.exit_code == 0
        assert captured["available_columns"] is None


class TestRealRun:
    def _patch(self, monkeypatch, run_result):
        import panelcast.cli.select_cmd as sc
        import panelcast.select.orchestrate as orch

        monkeypatch.setattr(sc, "_prepared_paths", lambda descriptor: None)
        monkeypatch.setattr(sc, "_load_prepared_frame", lambda descriptor: (None, None))
        monkeypatch.setattr(orch, "run_select", lambda *a, **k: run_result)

    def test_winner_recommended_message(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"report_dir": "rd", "winner_arm": "abc123", "promotable": ["abc123"],
             "n_arms_scored": 3, "ledger": "l"},
        )
        result = runner.invoke(app, ["select", "--effort", "quick", "--config", CONFIG])
        assert result.exit_code == 0
        assert "Recommended" in result.stdout
        assert "manual PR" in result.stdout

    def test_no_winner_message(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"report_dir": "rd", "winner_arm": None, "promotable": [],
             "n_arms_scored": 3, "ledger": "l"},
        )
        result = runner.invoke(app, ["select", "--config", CONFIG])
        assert result.exit_code == 0
        assert "defaults hold" in result.stdout

    def test_missing_data_note(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"report_dir": "rd", "winner_arm": None, "promotable": [],
             "n_arms_scored": 0, "ledger": "l"},
        )
        result = runner.invoke(app, ["select", "--config", CONFIG])
        assert "prior-predictive screen and data diagnostics are skipped" in result.stdout

    def test_cleared_rules_but_unconfirmed(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"report_dir": "rd", "winner_arm": None, "promotable": ["abc123"],
             "confirmed": False, "n_arms_scored": 3, "ledger": "l"},
        )
        result = runner.invoke(app, ["select", "--config", CONFIG])
        assert result.exit_code == 0
        assert "did not" in result.stdout
        assert "confirmation" in result.stdout


class TestFrameLoading:
    def test_prepared_paths_and_frame(self, tmp_path, monkeypatch):
        import pandas as pd

        from panelcast.cli.select_cmd import _load_prepared_frame, _prepared_paths
        from panelcast.config.descriptor import DatasetDescriptor

        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "features").mkdir(parents=True)
        (tmp_path / "data" / "splits" / "within_entity_temporal").mkdir(parents=True)
        pd.DataFrame({"original_row_id": [0, 1], "f": [1.0, 2.0]}).to_parquet(
            tmp_path / "data" / "features" / "train_features.parquet"
        )
        pd.DataFrame(
            {"original_row_id": [0, 1], "Artist": ["a", "b"], "User_Score": [70.0, 80.0]}
        ).to_parquet(tmp_path / "data" / "splits" / "within_entity_temporal" / "train.parquet")

        hint = _prepared_paths(DatasetDescriptor())
        assert hint["n_artists"] == 2
        df, cols = _load_prepared_frame(DatasetDescriptor())
        assert cols == ["f"]
        assert len(df) == 2

    def test_prepared_paths_none_without_features(self, tmp_path, monkeypatch):
        from panelcast.cli.select_cmd import _load_prepared_frame, _prepared_paths
        from panelcast.config.descriptor import DatasetDescriptor

        monkeypatch.chdir(tmp_path)
        assert _prepared_paths(DatasetDescriptor()) is None
        assert _load_prepared_frame(DatasetDescriptor()) == (None, None)

    def test_foreign_domain_frame_falls_back(self, tmp_path, monkeypatch, capsys):
        # The flat data/ artifacts are shared across domains: a frame left by
        # another domain's run must be ignored, not crash the prior screen on
        # a missing target column.
        import pandas as pd

        from panelcast.cli.select_cmd import _load_prepared_frame
        from panelcast.config.descriptor import DatasetDescriptor

        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "features").mkdir(parents=True)
        (tmp_path / "data" / "splits" / "within_entity_temporal").mkdir(parents=True)
        pd.DataFrame({"original_row_id": [0, 1], "f": [1.0, 2.0]}).to_parquet(
            tmp_path / "data" / "features" / "train_features.parquet"
        )
        pd.DataFrame(
            {"original_row_id": [0, 1], "unit_id": ["u1", "u2"], "failure_rate": [0.1, 0.2]}
        ).to_parquet(tmp_path / "data" / "splits" / "within_entity_temporal" / "train.parquet")

        assert _load_prepared_frame(DatasetDescriptor()) == (None, None)
        out = capsys.readouterr().out
        assert "do not match dataset 'aoty'" in out
        assert "Artist" in out and "User_Score" in out
