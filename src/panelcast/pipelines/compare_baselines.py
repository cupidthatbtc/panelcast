"""Baseline comparison driver: fit baselines on existing splits, emit a table.

Assembles the per-split panels straight from the split + feature parquet
artifacts (no model fit required), scores every baseline through the shared
evaluation toolkit, optionally appends the current Bayesian model's metrics from
an evaluation run, and writes the benchmark table as CSV + Markdown.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from panelcast.config.descriptor import DatasetDescriptor, load_descriptor
from panelcast.data.alignment import ROW_ID_COL, join_splits_with_features
from panelcast.data.split_types import SplitType, resolve_split_dir
from panelcast.models.baselines import PanelData, benchmark_baselines
from panelcast.models.baselines.core import BaselineScore
from panelcast.pipelines.stamps import verify_stamps
from panelcast.reporting.tables import create_baseline_benchmark_table, export_table

log = structlog.get_logger()

DEFAULT_SPLITS: tuple[SplitType, ...] = (
    SplitType.WITHIN_ENTITY_TEMPORAL,
    SplitType.ENTITY_DISJOINT,
)


@dataclass
class ComparisonResult:
    """Outputs of a baseline comparison run."""

    rows: list[dict]
    scores: list[BaselineScore]
    table: pd.DataFrame
    artifacts: list[Path]


def _json_safe(obj: object) -> object:
    """Recursively replace non-finite floats with None so json.dumps emits valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _feature_cols(features_df: pd.DataFrame) -> list[str]:
    """Predictor columns: everything except n_reviews and the row-id key."""
    return [c for c in features_df.columns if c not in ("n_reviews", ROW_ID_COL)]


def _entity_last_train_score(
    train_df: pd.DataFrame, descriptor: DatasetDescriptor
) -> dict[object, float]:
    """Map each train entity to its chronologically-last training score."""
    entity_col = descriptor.entity_col
    target_col = descriptor.target_col
    sort_cols = [entity_col]
    if descriptor.parsed_date_col in train_df.columns:
        sort_cols.append(descriptor.parsed_date_col)
    if descriptor.event_col in train_df.columns:
        sort_cols.append(descriptor.event_col)
    ordered = train_df.sort_values(sort_cols, na_position="first")
    last = ordered.groupby(entity_col)[target_col].last()
    return {e: float(v) for e, v in last.items() if pd.notna(v)}


def _build_panel(
    split_df: pd.DataFrame,
    features_df: pd.DataFrame,
    descriptor: DatasetDescriptor,
    feature_cols: list[str],
    *,
    train_mean: float,
    prev_score_map: dict[object, float] | None,
    is_train: bool,
) -> PanelData:
    """Assemble a PanelData from a joined split/feature frame."""
    entity_col = descriptor.entity_col
    target_col = descriptor.target_col
    merged = join_splits_with_features(split_df, features_df, name="baseline_panel")
    merged[feature_cols] = merged[feature_cols].fillna(0)

    X = merged[feature_cols].to_numpy(dtype=float)
    y = pd.to_numeric(merged[target_col], errors="coerce").to_numpy(dtype=float)
    entity = merged[entity_col].to_numpy()

    if is_train:
        # Order chronologically within entity before shift(1) so prev_score is
        # the true predecessor, not a random row (mirrors _entity_last_train_score).
        sort_cols = [entity_col]
        if descriptor.parsed_date_col in merged.columns:
            sort_cols.append(descriptor.parsed_date_col)
        if descriptor.event_col in merged.columns:
            sort_cols.append(descriptor.event_col)
        ordered = merged.sort_values(sort_cols, kind="stable", na_position="first")
        prev = ordered.groupby(entity_col)[target_col].shift(1)
        prev = prev.reindex(merged.index).fillna(train_mean)
        prev_score = pd.to_numeric(prev, errors="coerce").to_numpy(dtype=float)
    else:
        pmap = prev_score_map or {}
        prev_score = np.array([pmap.get(e, train_mean) for e in entity], dtype=float)

    return PanelData(
        X=X,
        y=y,
        entity=entity,
        prev_score=prev_score,
        bounds=tuple(descriptor.target_bounds),
    )


def load_panel_pair(
    split_type: SplitType,
    descriptor: DatasetDescriptor,
    splits_root: Path = Path("data/splits"),
    features_root: Path = Path("data/features"),
) -> tuple[PanelData, PanelData]:
    """Load (train, test) panels for one split from on-disk artifacts."""
    split_dir = resolve_split_dir(splits_root, split_type)
    feature_dir = resolve_split_dir(features_root, split_type)

    train_split = pd.read_parquet(split_dir / "train.parquet")
    test_split = pd.read_parquet(split_dir / "test.parquet")
    train_feat = pd.read_parquet(feature_dir / "train_features.parquet")
    test_feat = pd.read_parquet(feature_dir / "test_features.parquet")

    feature_cols = _feature_cols(train_feat)
    if not feature_cols:
        raise ValueError(
            f"No predictor features for split '{split_type}'. Feature parquet at "
            f"{feature_dir} must include at least one predictor column."
        )

    target_col = descriptor.target_col
    train_mean = float(pd.to_numeric(train_split[target_col], errors="coerce").mean())
    prev_map = _entity_last_train_score(train_split, descriptor)

    train_panel = _build_panel(
        train_split,
        train_feat,
        descriptor,
        feature_cols,
        train_mean=train_mean,
        prev_score_map=None,
        is_train=True,
    )
    test_panel = _build_panel(
        test_split,
        test_feat,
        descriptor,
        feature_cols,
        train_mean=train_mean,
        prev_score_map=prev_map,
        is_train=False,
    )
    return train_panel, test_panel


def _bayes_rows_from_metrics(
    metrics_path: Path, levels: tuple[float, ...]
) -> list[dict]:
    """Pull the current Bayesian model's row(s) from an evaluation metrics.json.

    Best-effort: the evaluation artifact schema varies, so anything missing is
    rendered as NaN (an em-dash in the table) rather than dropped.
    """
    if not metrics_path.exists():
        return []
    try:
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    def _nan_get(mapping: object, *keys: object, default: float = float("nan")):
        cur = mapping
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    point = metrics.get("point_metrics") or metrics.get("metrics") or {}
    coverages = _nan_get(metrics, "calibration", "coverages", default={})
    ppc_summary = _nan_get(metrics, "ppc", "summary", default={})
    primary_split = metrics.get("primary_split", str(SplitType.WITHIN_ENTITY_TEMPORAL.value))

    row: dict[str, float | str] = {
        "model": "bayes (current)",
        "split": str(primary_split),
        "n_obs": _nan_get(point, "n_observations"),
        "mae": _nan_get(point, "mae"),
        "rmse": _nan_get(point, "rmse"),
        "r2": _nan_get(point, "r2"),
        "crps": _nan_get(metrics, "crps", "mean_crps"),
    }
    for level in levels:
        row[f"cov{int(round(level * 100))}"] = _nan_get(
            coverages, f"{level:.2f}", "empirical"
        )
    row["width95"] = _nan_get(coverages, "0.95", "interval_width")
    row["ppc_skew_p"] = _nan_get(ppc_summary, "skewness", "p_value")
    row["runtime_s"] = float("nan")
    return [row]


def _verify_features_match_metrics(metrics_path: Path) -> None:
    """Fail fast when data/features was regenerated since the evaluation ran.

    The bayes row comes from metrics.json while baselines fit on the current
    feature parquets; mixing provenances would make the table incomparable.
    """
    try:
        with open(metrics_path, encoding="utf-8") as f:
            recorded = json.load(f).get("feature_stamp")
    except (OSError, json.JSONDecodeError):
        return
    if recorded:
        verify_stamps({"features": recorded}, "compare")


def run_baseline_comparison(
    dataset: str | None = None,
    splits: tuple[SplitType, ...] = DEFAULT_SPLITS,
    levels: tuple[float, ...] = (0.80, 0.95),
    n_samples: int = 1000,
    seed: int = 0,
    output_dir: Path = Path("reports/baselines"),
    include_bayes: bool = True,
    metrics_path: Path = Path("outputs/evaluation/metrics.json"),
) -> ComparisonResult:
    """Fit and score every baseline on each split; write the benchmark table."""
    if include_bayes:
        _verify_features_match_metrics(metrics_path)
    descriptor = load_descriptor(dataset)
    all_rows: list[dict] = []
    all_scores: list[BaselineScore] = []

    for split_type in splits:
        log.info("baseline_split_start", split=str(split_type.value))
        train_panel, test_panel = load_panel_pair(split_type, descriptor)
        scores = benchmark_baselines(
            train_panel,
            test_panel,
            split=str(split_type.value),
            levels=levels,
            n_samples=n_samples,
            seed=seed,
        )
        all_scores.extend(scores)
        all_rows.extend(s.to_row(levels) for s in scores)

    if include_bayes:
        all_rows.extend(_bayes_rows_from_metrics(metrics_path, levels))

    table = create_baseline_benchmark_table(all_rows, levels=levels)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = export_table(
        table,
        output_dir / "baseline_comparison",
        formats=("csv",),
        caption="Baseline vs. model benchmark (point accuracy, calibration, CRPS).",
    )
    md_path = output_dir / "baseline_comparison.md"
    md_path.write_text(_render_markdown(table), encoding="utf-8")
    artifacts.append(md_path)
    json_path = output_dir / "baseline_comparison.json"
    json_path.write_text(json.dumps(_json_safe(all_rows), indent=2), encoding="utf-8")
    artifacts.append(json_path)

    log.info("baseline_comparison_complete", rows=len(all_rows), artifacts=len(artifacts))
    return ComparisonResult(rows=all_rows, scores=all_scores, table=table, artifacts=artifacts)


def _render_markdown(table: pd.DataFrame) -> str:
    """Render the benchmark table as a GitHub Markdown table (no extra deps)."""
    header = "# Baseline comparison\n\n"
    if table.empty:
        return header + "_No rows — run the splits and features stages first._\n"
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return header + "\n".join(lines) + "\n"
