"""Coverage tests for panelcast.pipelines.compare_baselines.

Drives ``run_baseline_comparison`` end-to-end over the five baselines on a tiny
synthetic panel written to on-disk split/feature parquet artifacts (no Bayesian
fit, so it stays fast), plus the module's helper functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelcast.data.alignment import ROW_ID_COL
from panelcast.data.split_types import SplitType, split_dir_name
from panelcast.pipelines.compare_baselines import (
    ComparisonResult,
    _bayes_rows_from_metrics,
    _build_panel,
    _entity_last_score,
    _feature_cols,
    _json_safe,
    _render_markdown,
    load_panel_pair,
    run_baseline_comparison,
)
from tests.helpers.aero_data import make_aero_descriptor

FEATURE_COLS = ["feat_a", "feat_b"]


def _panel(n_entities: int = 4, per: int = 6, seed: int = 0) -> pd.DataFrame:
    """A small entity/event/date/score/feature panel with stable row ids."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    rid = 0
    for e in range(n_entities):
        base = float(rng.normal(6.0, 1.0))
        date = pd.Timestamp("2021-01-01")
        for k in range(per):
            date = date + pd.Timedelta(days=30)
            score = float(np.clip(base + rng.normal(0.0, 0.5), 0.05, 9.95))
            rows.append(
                {
                    ROW_ID_COL: rid,
                    "Airframe": f"AF{e}",
                    "Flight_ID": f"AF{e}-F{k}",
                    "Flight_Date_Parsed": date,
                    "Perf_Score": round(score, 2),
                    "feat_a": float(rng.normal()),
                    "feat_b": float(rng.normal()),
                    "n_reviews": int(rng.integers(5, 50)),
                }
            )
            rid += 1
    return pd.DataFrame(rows)


def _write_split_artifacts(
    root_splits: Path,
    root_features: Path,
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    val_df: pd.DataFrame | None = None,
) -> None:
    """Persist train/test split + feature parquets for one split directory."""
    split_dir = root_splits / split_name
    feat_dir = root_features / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    split_keep = [ROW_ID_COL, "Airframe", "Flight_ID", "Flight_Date_Parsed", "Perf_Score"]
    feat_keep = [ROW_ID_COL, *feature_cols, "n_reviews"]
    train_df[split_keep].to_parquet(split_dir / "train.parquet")
    test_df[split_keep].to_parquet(split_dir / "test.parquet")
    if val_df is not None:
        val_df[split_keep].to_parquet(split_dir / "validation.parquet")
    train_df[feat_keep].to_parquet(feat_dir / "train_features.parquet")
    test_df[feat_keep].to_parquet(feat_dir / "test_features.parquet")


def _seed_both_splits(tmp_path: Path) -> tuple[Path, Path]:
    """Write artifacts for both default splits under tmp_path/data."""
    root_splits = tmp_path / "data" / "splits"
    root_features = tmp_path / "data" / "features"
    full = _panel()
    # Train = all but the last event per entity; test = the last event per entity.
    last = full.groupby("Airframe").tail(1).index
    train_df = full.drop(index=last).reset_index(drop=True)
    test_df = full.loc[last].reset_index(drop=True)
    for split in (SplitType.WITHIN_ENTITY_TEMPORAL, SplitType.ENTITY_DISJOINT):
        _write_split_artifacts(
            root_splits, root_features, split_dir_name(split), train_df, test_df
        )
    return root_splits, root_features


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestJsonSafe:
    def test_replaces_non_finite_with_none(self):
        out = _json_safe({"a": float("nan"), "b": float("inf"), "c": 1.5})
        assert out["a"] is None
        assert out["b"] is None
        assert out["c"] == 1.5

    def test_recurses_into_lists_and_casts_numpy(self):
        out = _json_safe([np.int64(3), {"x": np.float64(2.0)}, (np.float64("nan"),)])
        assert out == [3, {"x": 2.0}, [None]]
        # round-trips through json
        json.dumps(out)

    def test_passes_through_plain_objects(self):
        assert _json_safe("hello") == "hello"
        assert _json_safe(7) == 7


class TestFeatureCols:
    def test_excludes_n_reviews_and_row_id(self):
        df = pd.DataFrame(
            {ROW_ID_COL: [0], "n_reviews": [1], "feat_a": [0.1], "feat_b": [0.2]}
        )
        assert _feature_cols(df) == ["feat_a", "feat_b"]


class TestEntityLastScore:
    def test_maps_entity_to_chronologically_last_score(self):
        df = _panel(n_entities=2, per=3)
        desc = make_aero_descriptor()
        last = _entity_last_score(df, desc)
        # Last event per entity has the largest date.
        for ent, grp in df.groupby("Airframe"):
            expected = grp.sort_values("Flight_Date_Parsed")["Perf_Score"].iloc[-1]
            assert last[ent] == pytest.approx(expected)

    def test_drops_nan_scores(self):
        df = _panel(n_entities=1, per=2)
        df.loc[df.index[-1], "Perf_Score"] = float("nan")
        desc = make_aero_descriptor()
        last = _entity_last_score(df, desc)
        # The only remaining non-NaN score is the first event.
        assert last["AF0"] == pytest.approx(df["Perf_Score"].iloc[0])


class TestRenderMarkdown:
    def test_empty_table_message(self):
        md = _render_markdown(pd.DataFrame())
        assert "No rows" in md

    def test_renders_header_and_rows(self):
        table = pd.DataFrame({"model": ["global_mean"], "mae": [1.2]})
        md = _render_markdown(table)
        assert "| model | mae |" in md
        assert "global_mean" in md


class TestBayesRowsFromMetrics:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _bayes_rows_from_metrics(tmp_path / "nope.json", (0.80, 0.95)) == []

    def test_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "metrics.json"
        p.write_text("{not json", encoding="utf-8")
        assert _bayes_rows_from_metrics(p, (0.80, 0.95)) == []

    def test_extracts_row_with_nan_fallbacks(self, tmp_path):
        p = tmp_path / "metrics.json"
        payload = {
            "primary_split": "within_entity_temporal",
            "point_metrics": {"n_observations": 100, "mae": 1.0, "rmse": 1.4, "r2": 0.6},
            "calibration": {
                "coverages": {
                    "0.80": {"empirical": 0.79},
                    "0.95": {"empirical": 0.94, "interval_width": 5.0},
                }
            },
            "crps": {"mean_crps": 0.7},
            "ppc": {"summary": {"skewness": {"p_value": 0.3}}},
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        rows = _bayes_rows_from_metrics(p, (0.80, 0.95))
        assert len(rows) == 1
        row = rows[0]
        assert row["model"] == "bayes (current)"
        assert row["mae"] == 1.0
        assert row["cov80"] == 0.79
        assert row["width95"] == 5.0
        assert row["ppc_skew_p"] == 0.3

    def test_missing_keys_render_nan(self, tmp_path):
        p = tmp_path / "metrics.json"
        p.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        rows = _bayes_rows_from_metrics(p, (0.80, 0.95))
        assert len(rows) == 1
        assert np.isnan(rows[0]["mae"])


# ---------------------------------------------------------------------------
# load_panel_pair
# ---------------------------------------------------------------------------


class TestLoadPanelPair:
    def test_builds_train_and_test_panels(self, tmp_path):
        root_splits, root_features = _seed_both_splits(tmp_path)
        desc = make_aero_descriptor()
        train_panel, test_panel = load_panel_pair(
            SplitType.WITHIN_ENTITY_TEMPORAL, desc, root_splits, root_features
        )
        assert train_panel.X.shape[1] == len(FEATURE_COLS)
        assert train_panel.bounds == (0.0, 10.0)
        # prev_score is filled for every train row.
        assert train_panel.prev_score is not None
        assert np.isfinite(train_panel.prev_score).all()
        # test prev_score comes from the entity's last train score map.
        assert test_panel.prev_score is not None

    def test_raises_when_no_feature_columns(self, tmp_path):
        root_splits = tmp_path / "data" / "splits"
        root_features = tmp_path / "data" / "features"
        full = _panel()
        last = full.groupby("Airframe").tail(1).index
        train_df = full.drop(index=last).reset_index(drop=True)
        test_df = full.loc[last].reset_index(drop=True)
        # feature_cols empty -> only row id + n_reviews columns survive
        _write_split_artifacts(
            root_splits,
            root_features,
            split_dir_name(SplitType.WITHIN_ENTITY_TEMPORAL),
            train_df,
            test_df,
            feature_cols=[],
        )
        desc = make_aero_descriptor()
        with pytest.raises(ValueError, match="No predictor features"):
            load_panel_pair(
                SplitType.WITHIN_ENTITY_TEMPORAL, desc, root_splits, root_features
            )


class TestBuildPanelTestBranch:
    def test_unseen_entity_uses_train_mean(self, tmp_path):
        desc = make_aero_descriptor()
        split_df = pd.DataFrame(
            {
                ROW_ID_COL: [0, 1],
                "Airframe": ["NEW", "AF0"],
                "Flight_ID": ["NEW-F0", "AF0-F9"],
                "Flight_Date_Parsed": pd.to_datetime(["2022-01-01", "2022-02-01"]),
                "Perf_Score": [5.0, 6.0],
            }
        )
        feat_df = pd.DataFrame(
            {ROW_ID_COL: [0, 1], "feat_a": [0.1, 0.2], "feat_b": [0.3, 0.4]}
        )
        panel = _build_panel(
            split_df,
            feat_df,
            desc,
            FEATURE_COLS,
            train_mean=7.5,
            prev_score_map={"AF0": 6.5},
            is_train=False,
        )
        # NEW entity -> train_mean; AF0 -> its mapped prev score.
        assert panel.prev_score[0] == pytest.approx(7.5)
        assert panel.prev_score[1] == pytest.approx(6.5)


def _event_row(
    rid: int, entity: str, event: str, date: str, score: float, n_reviews: float = 20
) -> dict:
    return {
        ROW_ID_COL: rid,
        "Airframe": entity,
        "Flight_ID": event,
        "Flight_Date_Parsed": pd.Timestamp(date),
        "Perf_Score": score,
        "feat_a": 0.1 * rid,
        "feat_b": -0.2 * rid,
        "n_reviews": n_reviews,
    }


def _handmade_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-entity panel: AF0 has a val event and two test events, AF1 neither."""
    train = pd.DataFrame(
        [
            _event_row(0, "AF0", "AF0-F0", "2021-01-01", 5.0),
            _event_row(1, "AF0", "AF0-F1", "2021-02-01", 6.0),
            _event_row(2, "AF1", "AF1-F0", "2021-01-15", 4.0),
            _event_row(3, "AF1", "AF1-F1", "2021-02-15", 4.5),
        ]
    )
    val = pd.DataFrame([_event_row(20, "AF0", "AF0-FV", "2021-03-01", 8.0)])
    test = pd.DataFrame(
        [
            _event_row(10, "AF0", "AF0-F2", "2021-04-01", 7.0),
            _event_row(11, "AF0", "AF0-F3", "2021-05-01", 3.0),
            _event_row(12, "AF1", "AF1-F2", "2021-04-15", 5.5),
        ]
    )
    return train, val, test


class TestSequentialPrevScore:
    """Test-panel prev_score must mirror evaluate.py's sequential protocol."""

    @staticmethod
    def _load(tmp_path, split_type, train, test, val=None):
        root_splits = tmp_path / "data" / "splits"
        root_features = tmp_path / "data" / "features"
        _write_split_artifacts(
            root_splits, root_features, split_dir_name(split_type), train, test, val_df=val
        )
        desc = make_aero_descriptor()
        return load_panel_pair(split_type, desc, root_splits, root_features)

    def test_val_score_preferred_over_train(self, tmp_path):
        train, val, test = _handmade_frames()
        _, test_panel = self._load(
            tmp_path, SplitType.WITHIN_ENTITY_TEMPORAL, train, test, val
        )
        # AF0's first test event conditions on its val score, not the last
        # train score; AF1 has no val event and keeps its last train score.
        assert test_panel.prev_score[0] == pytest.approx(8.0)
        assert test_panel.prev_score[2] == pytest.approx(4.5)

    def test_teacher_forces_preceding_test_labels(self, tmp_path):
        train, val, test = _handmade_frames()
        _, test_panel = self._load(
            tmp_path, SplitType.WITHIN_ENTITY_TEMPORAL, train, test, val
        )
        # AF0's second test event conditions on the first test event's label.
        assert test_panel.prev_score[1] == pytest.approx(7.0)

    def test_without_validation_first_event_uses_last_train_score(self, tmp_path):
        train, _, test = _handmade_frames()
        _, test_panel = self._load(tmp_path, SplitType.WITHIN_ENTITY_TEMPORAL, train, test)
        assert test_panel.prev_score[0] == pytest.approx(6.0)
        assert test_panel.prev_score[1] == pytest.approx(7.0)
        assert test_panel.prev_score[2] == pytest.approx(4.5)

    def test_empty_validation_is_ignored(self, tmp_path):
        train, val, test = _handmade_frames()
        _, test_panel = self._load(
            tmp_path, SplitType.WITHIN_ENTITY_TEMPORAL, train, test, val.iloc[0:0]
        )
        assert test_panel.prev_score[0] == pytest.approx(6.0)

    def test_disjoint_split_keeps_cold_start_protocol(self, tmp_path):
        # Entity-disjoint mirrors the model's cold-start evaluation: no val
        # conditioning, no teacher forcing — every unseen-entity row gets the
        # train mean.
        train, _, _ = _handmade_frames()
        test = pd.DataFrame(
            [
                _event_row(30, "AF9", "AF9-F0", "2021-04-01", 7.0),
                _event_row(31, "AF9", "AF9-F1", "2021-05-01", 3.0),
            ]
        )
        val = pd.DataFrame([_event_row(40, "AF9", "AF9-FV", "2021-03-01", 9.0)])
        _, test_panel = self._load(tmp_path, SplitType.ENTITY_DISJOINT, train, test, val)
        train_mean = train["Perf_Score"].mean()
        assert test_panel.prev_score == pytest.approx([train_mean, train_mean])


class TestNReviewsValidityFilter:
    def test_invalid_test_rows_dropped(self, tmp_path):
        train, _, test = _handmade_frames()
        test.loc[0, "n_reviews"] = 0
        test.loc[1, "n_reviews"] = float("nan")
        root_splits = tmp_path / "data" / "splits"
        root_features = tmp_path / "data" / "features"
        _write_split_artifacts(
            root_splits,
            root_features,
            split_dir_name(SplitType.WITHIN_ENTITY_TEMPORAL),
            train,
            test,
        )
        desc = make_aero_descriptor()
        train_panel, test_panel = load_panel_pair(
            SplitType.WITHIN_ENTITY_TEMPORAL, desc, root_splits, root_features
        )
        # Only the valid-n_reviews test row survives; train is untouched.
        assert test_panel.y.shape[0] == 1
        assert test_panel.y[0] == pytest.approx(5.5)
        assert test_panel.entity[0] == "AF1"
        assert train_panel.y.shape[0] == len(train)


class TestNObsAlignmentCheck:
    @staticmethod
    def _run(tmp_path, monkeypatch, n_observations):
        import structlog

        _seed_both_splits(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.load_descriptor",
            lambda dataset=None: make_aero_descriptor(),
        )
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "primary_split": "within_entity_temporal",
                    "point_metrics": {"n_observations": n_observations, "mae": 1.0},
                }
            ),
            encoding="utf-8",
        )
        with structlog.testing.capture_logs() as logs:
            run_baseline_comparison(
                dataset=None,
                splits=(SplitType.WITHIN_ENTITY_TEMPORAL,),
                n_samples=64,
                output_dir=tmp_path / "reports" / "baselines",
                include_bayes=True,
                metrics_path=metrics_path,
            )
        return [e for e in logs if e["event"] == "baseline_n_obs_mismatch"]

    def test_warns_when_bayes_n_obs_differs(self, tmp_path, monkeypatch):
        warnings = self._run(tmp_path, monkeypatch, n_observations=999)
        assert len(warnings) == 1
        assert warnings[0]["bayes_n_obs"] == 999

    def test_silent_when_n_obs_match(self, tmp_path, monkeypatch):
        # _seed_both_splits holds out one test event per entity (4 entities).
        assert self._run(tmp_path, monkeypatch, n_observations=4) == []


# ---------------------------------------------------------------------------
# run_baseline_comparison (end-to-end)
# ---------------------------------------------------------------------------


class TestRunBaselineComparison:
    def test_end_to_end_with_bayes_row(self, tmp_path, monkeypatch):
        root_splits, root_features = _seed_both_splits(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.load_descriptor",
            lambda dataset=None: make_aero_descriptor(),
        )
        metrics_path = tmp_path / "outputs" / "evaluation" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "primary_split": "within_entity_temporal",
                    "point_metrics": {"n_observations": 10, "mae": 1.0, "rmse": 1.2, "r2": 0.4},
                    "calibration": {"coverages": {"0.95": {"empirical": 0.95, "interval_width": 4.0}}},
                    "crps": {"mean_crps": 0.5},
                }
            ),
            encoding="utf-8",
        )
        result = run_baseline_comparison(
            dataset=None,
            n_samples=64,
            output_dir=tmp_path / "reports" / "baselines",
            include_bayes=True,
            metrics_path=metrics_path,
        )
        assert isinstance(result, ComparisonResult)
        # 5 baselines x 2 splits + 1 bayes row
        models = {r["model"] for r in result.rows}
        assert {"global_mean", "entity_mean", "last_score", "ridge", "gbm"} <= models
        assert "bayes (current)" in models
        assert not result.table.empty
        # Artifacts written: csv, md, json.
        suffixes = {p.suffix for p in result.artifacts}
        assert {".csv", ".md", ".json"} <= suffixes
        for p in result.artifacts:
            assert p.exists()
        # JSON artifact is valid (non-finite scrubbed).
        json_art = next(p for p in result.artifacts if p.suffix == ".json")
        json.loads(json_art.read_text(encoding="utf-8"))

    def test_without_bayes_row(self, tmp_path, monkeypatch):
        _seed_both_splits(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.load_descriptor",
            lambda dataset=None: make_aero_descriptor(),
        )
        result = run_baseline_comparison(
            dataset=None,
            splits=(SplitType.WITHIN_ENTITY_TEMPORAL,),
            n_samples=64,
            output_dir=tmp_path / "reports" / "baselines",
            include_bayes=False,
        )
        assert "bayes (current)" not in {r["model"] for r in result.rows}
        assert all(r["split"] == "within_entity_temporal" for r in result.rows)

    def test_default_output_dir_is_run_scoped(self, tmp_path, monkeypatch):
        """output_dir=None writes under the latest run's reports/baselines."""
        _seed_both_splits(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.load_descriptor",
            lambda dataset=None: make_aero_descriptor(),
        )
        run_dir = tmp_path / "outputs" / "runA"
        run_dir.mkdir(parents=True)
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_id": "runA", "run_dir": "runA"}), encoding="utf-8"
        )
        result = run_baseline_comparison(
            dataset=None,
            splits=(SplitType.WITHIN_ENTITY_TEMPORAL,),
            n_samples=64,
            include_bayes=False,
        )
        expected = (run_dir / "reports" / "baselines").resolve()
        assert all(p.resolve().parent == expected for p in result.artifacts)
        assert (expected / "baseline_comparison.csv").exists()

    def test_explicit_output_dir_overrides_run_scoping(self, tmp_path, monkeypatch):
        """An explicit output_dir wins even when a latest run exists."""
        _seed_both_splits(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.load_descriptor",
            lambda dataset=None: make_aero_descriptor(),
        )
        run_dir = tmp_path / "outputs" / "runA"
        run_dir.mkdir(parents=True)
        (tmp_path / "outputs" / "latest.json").write_text(
            json.dumps({"run_id": "runA", "run_dir": "runA"}), encoding="utf-8"
        )
        custom = tmp_path / "custom_out"
        result = run_baseline_comparison(
            dataset=None,
            splits=(SplitType.WITHIN_ENTITY_TEMPORAL,),
            n_samples=64,
            output_dir=custom,
            include_bayes=False,
        )
        assert all(p.resolve().parent == custom.resolve() for p in result.artifacts)
        assert not (run_dir / "reports" / "baselines").exists()

    def test_default_output_dir_flat_without_latest(self, tmp_path, monkeypatch):
        """Without a latest pointer the default falls back to flat reports/baselines."""
        _seed_both_splits(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "panelcast.pipelines.compare_baselines.load_descriptor",
            lambda dataset=None: make_aero_descriptor(),
        )
        result = run_baseline_comparison(
            dataset=None,
            splits=(SplitType.WITHIN_ENTITY_TEMPORAL,),
            n_samples=64,
            include_bayes=False,
        )
        expected = (tmp_path / "reports" / "baselines").resolve()
        assert all(p.resolve().parent == expected for p in result.artifacts)
