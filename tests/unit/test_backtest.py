"""Rolling-origin backtest (#179): offset splits, ledger resume, aggregation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from panelcast.data.split import within_entity_temporal_split
from panelcast.pipelines import backtest as bt


def _df():
    rows = []
    for artist, n in (("A", 6), ("B", 4), ("C", 3)):
        for i in range(n):
            rows.append(
                {
                    "Artist": artist,
                    "Album": f"{artist}{i}",
                    "Release_Date_Parsed": pd.Timestamp(f"20{10 + i}-01-01"),
                    "User_Score": 60.0 + i,
                }
            )
    return pd.DataFrame(rows)


class TestOriginOffsetSplit:
    def test_offset_zero_is_the_standard_split(self):
        df = _df()
        base = within_entity_temporal_split(df, test_albums=1, val_albums=1)
        offset = within_entity_temporal_split(df, test_albums=1, val_albums=1, origin_offset=0)
        for a, b in zip(base, offset):
            pd.testing.assert_frame_equal(a, b)

    def test_offset_one_holds_out_second_to_last(self):
        df = _df()
        train, val, test = within_entity_temporal_split(
            df, test_albums=1, val_albums=0, min_train_albums=1, origin_offset=1
        )
        held = test.set_index("Artist")["Album"].to_dict()
        # Last event per entity is dropped as future; (last-1)-th is the test.
        assert held == {"A": "A4", "B": "B2", "C": "C1"}
        all_events = set(train["Album"]) | set(val["Album"]) | set(test["Album"])
        assert {"A5", "B3", "C2"}.isdisjoint(all_events)

    def test_deeper_origins_shrink_entity_set(self):
        df = _df()
        _, _, test = within_entity_temporal_split(
            df, test_albums=1, val_albums=0, min_train_albums=1, origin_offset=2
        )
        # C has 3 events: 2 future + 1 test + 0 train fails min_train_albums=1.
        assert set(test["Artist"]) == {"A", "B"}

    def test_negative_offset_raises(self):
        with pytest.raises(ValueError, match="origin_offset"):
            within_entity_temporal_split(_df(), origin_offset=-1)


class TestBacktestLedger:
    def test_roundtrip_and_resume_set(self, tmp_path):
        ledger = bt.BacktestLedger(tmp_path / "ledger.json")
        ledger.upsert(bt.OriginRecord(origin=0, status="completed", run_dir="x"))
        ledger.upsert(bt.OriginRecord(origin=1, status="failed", error="boom"))

        reloaded = bt.BacktestLedger(tmp_path / "ledger.json")
        assert reloaded.completed_origins() == {0}
        assert reloaded.records[1].error == "boom"


class TestAggregation:
    def _records(self):
        def rec(origin, mae, cov):
            return bt.OriginRecord(
                origin=origin,
                status="completed",
                run_dir=f"outputs/r{origin}",
                n_test=100 - origin,
                n_entities=90 - origin,
                metrics={
                    "mae": mae,
                    "rmse": mae + 1,
                    "r2": 0.5,
                    "crps": None,
                    "coverage_0.80": cov,
                    "coverage_0.95": None,
                    "wis": None,
                    "elpd_per_obs": None,
                },
            )

        return [rec(0, 5.0, 0.80), rec(1, 6.0, 0.84), bt.OriginRecord(origin=2, status="failed")]

    def test_mean_se_minmax(self):
        agg = bt.aggregate_backtest(self._records())
        mae = agg["metrics"]["mae"]
        assert mae["mean"] == pytest.approx(5.5)
        assert mae["se"] == pytest.approx(np.std([5.0, 6.0], ddof=1) / np.sqrt(2))
        assert (mae["min"], mae["max"]) == (5.0, 6.0)
        assert agg["n_origins_completed"] == 2
        assert agg["n_origins_requested"] == 3
        assert agg["metrics"]["crps"] is None

    def test_markdown_reports_populations(self):
        md = bt.render_backtest_markdown(bt.aggregate_backtest(self._records()))
        assert "n_test" in md and "n_entities" in md
        assert "| 0 | completed | 100 | 90 |" in md
        assert "| mae | 5.5000 |" in md


class TestRunBacktestResume:
    def test_completed_origins_are_skipped(self, tmp_path, monkeypatch):
        cfg = bt.BacktestConfig(origins=2, backtest_id="t", output_root=tmp_path)
        ledger_path = cfg.backtest_dir / "ledger.json"
        pre = bt.BacktestLedger(ledger_path)
        pre.upsert(
            bt.OriginRecord(
                origin=0,
                status="completed",
                run_dir="outputs/prior",
                metrics={name: 1.0 for name, _ in bt._AGGREGATED_METRICS},
            )
        )

        launched: list[int] = []

        def fake_launch(config_path, panelcast_bin, timeout_seconds):
            payload = json.loads(json.dumps(config_path.read_text()))
            assert "origin_offset: 1" in payload
            launched.append(1)
            return 1, "simulated failure"

        monkeypatch.setattr(bt, "_launch_origin", fake_launch)
        monkeypatch.setattr(bt, "_default_panelcast_bin", lambda: "panelcast", raising=False)

        agg = bt.run_backtest(cfg)
        assert launched == [1]  # origin 0 skipped, only origin 1 launched
        assert agg["n_origins_completed"] == 1
        reloaded = bt.BacktestLedger(ledger_path)
        assert reloaded.records[1].status == "failed"
        assert (cfg.backtest_dir / "backtest_metrics.json").exists()
        assert (cfg.backtest_dir / "backtest_report.md").exists()


def test_dig_paths_match_the_real_metrics_shape():
    # Pin every aggregation path against the actual metrics.json nesting the
    # evaluate stage writes — the elpd path was silently wrong once already.
    payload = {
        "n_test": 500,
        "point_metrics": {"mae": 5.3, "rmse": 7.1, "r2": 0.5},
        "crps": {"mean_crps": 3.9},
        "calibration": {
            "coverages": {"0.80": {"empirical": 0.79}, "0.95": {"empirical": 0.94}},
            "wis": 4.2,
        },
        "info_criteria": {
            "heldout_elpd": {"elpd": -1000.0, "se": 30.0, "elpd_per_obs": -4.115}
        },
    }
    harvested = {name: bt._dig(payload, path) for name, path in bt._AGGREGATED_METRICS}
    assert harvested == {
        "mae": 5.3,
        "rmse": 7.1,
        "r2": 0.5,
        "crps": 3.9,
        "coverage_0.80": 0.79,
        "coverage_0.95": 0.94,
        "wis": 4.2,
        "elpd_per_obs": -4.115,
    }
