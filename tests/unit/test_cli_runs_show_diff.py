"""`panelcast runs show` / `runs diff` / enriched `runs list` (#160)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    save_run_manifest,
)

runner = CliRunner()


def _write_run(
    base: Path,
    run_id: str,
    flags: dict | None = None,
    mae: float = 5.30,
    commit: str = "a" * 40,
) -> Path:
    run_dir = base / run_id
    metrics = {
        "point_metrics": {"mae": mae, "rmse": mae * 1.3, "r2": 0.5},
        "calibration": {"coverages": {"0.95": {"empirical": 0.94, "nominal": 0.95}}},
    }
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    manifest = RunManifest(
        run_id=run_id,
        created_at="2026-07-08T00:00:00Z",
        command="panelcast run",
        flags={"dataset_descriptor_hash": "abc123", **(flags or {})},
        seed=42,
        git=GitStateModel(commit=commit, branch="main", dirty=False, untracked_count=0),
        environment=EnvironmentInfo(
            python_version="3.14",
            jax_version="0.8.2",
            numpyro_version="0.19",
            arviz_version=None,
            platform="Linux",
            pixi_lock_hash="f" * 64,
            fingerprint="deadbeef00000000",
        ),
        input_hashes={},
        stage_hashes={},
        stages_completed=["train", "evaluate"],
        stages_skipped=[],
        outputs={},
        success=True,
    )
    save_run_manifest(manifest, run_dir)
    return run_dir


class TestRunsShow:
    def test_renders_provenance_and_metrics(self, tmp_path):
        _write_run(tmp_path, "run_a")
        result = runner.invoke(app, ["runs", "show", "run_a", "--output-base", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "run_a" in result.output
        assert "seed      42" in result.output
        assert "aaaaaaa on main" in result.output
        assert "fingerprint deadbeef00000000" in result.output
        assert "mae 5.300" in result.output
        assert "0.95: 0.940" in result.output


class TestRunsDiff:
    def test_flag_metric_and_fact_deltas(self, tmp_path):
        _write_run(tmp_path, "run_a", flags={"num_samples": 1000}, mae=5.30)
        _write_run(tmp_path, "run_b", flags={"num_samples": 2000}, mae=5.10)
        result = runner.invoke(
            app, ["runs", "diff", "run_a", "run_b", "--output-base", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert "num_samples: 1000 -> 2000" in result.output
        assert "point_metrics.mae: 5.3 -> 5.1" in result.output
        assert "identical:" in result.output
        assert "WARNING" not in result.output

    def test_descriptor_mismatch_warns(self, tmp_path):
        _write_run(tmp_path, "run_a")
        _write_run(tmp_path, "run_b", flags={"dataset_descriptor_hash": "zzz999"})
        result = runner.invoke(
            app, ["runs", "diff", "run_a", "run_b", "--output-base", str(tmp_path)]
        )
        assert "not a like-for-like comparison" in result.output

    def test_git_commit_difference_surfaces(self, tmp_path):
        _write_run(tmp_path, "run_a", commit="a" * 40)
        _write_run(tmp_path, "run_b", commit="b" * 40)
        result = runner.invoke(
            app, ["runs", "diff", "run_a", "run_b", "--output-base", str(tmp_path)]
        )
        assert "aaaaaaa vs bbbbbbb" in result.output

    def test_absent_flag_falls_back_to_default(self, tmp_path):
        # run_a predates the flag entirely; run_b pins the default value — no delta.
        _write_run(tmp_path, "run_a")
        _write_run(tmp_path, "run_b", flags={"num_samples": 1000})
        result = runner.invoke(
            app, ["runs", "diff", "run_a", "run_b", "--output-base", str(tmp_path)]
        )
        assert "num_samples" not in result.output


class TestRunsListEnriched:
    def test_list_shows_seed_git_and_metrics(self, tmp_path):
        _write_run(tmp_path, "run_a")
        result = runner.invoke(app, ["runs", "list", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "mae" in result.output
        assert "5.30" in result.output
        assert "aaaaaaa" in result.output
        assert "42" in result.output
