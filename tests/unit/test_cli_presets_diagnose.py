"""Tests for the config presets and the diagnose driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from panelcast.config.loader import load_yaml_config
from panelcast.config.pipeline_yaml import PIPELINE_YAML_MAPPING, apply_yaml_overrides
from panelcast.pipelines.diagnose import build_report, render_markdown, run_diagnose

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
PRESETS = ("quick", "dev", "diagnostic", "publication")


class TestPresets:
    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset_config_exists_and_maps(self, preset):
        path = CONFIGS / f"{preset}.yaml"
        assert path.exists(), f"missing preset config {path}"
        data = load_yaml_config([str(path)])
        assert data, f"{preset}.yaml is empty"
        # Every key must be a known PipelineConfig mapping (no silent typos).
        unknown = [k for k in data if k not in PIPELINE_YAML_MAPPING]
        assert not unknown, f"{preset}.yaml has unmapped keys: {unknown}"

    def test_quick_is_single_chain(self):
        data = load_yaml_config([str(CONFIGS / "quick.yaml")])
        assert data["num_chains"] == 1

    def test_cli_overrides_preset(self):
        # Preset layered first, then an explicit CLI param wins.
        data = load_yaml_config([str(CONFIGS / "quick.yaml")])
        kwargs = {"num_chains": 4, "num_samples": 999}
        out = apply_yaml_overrides(kwargs, data, explicit_cli_params={"num_samples"})
        assert out["num_chains"] == 1  # from the preset
        assert out["num_samples"] == 999  # explicit CLI flag wins


class TestDiagnose:
    def _write_eval(self, tmp_path: Path, *, passed: bool, skew_p: float) -> Path:
        eval_dir = tmp_path / "evaluation"
        eval_dir.mkdir()
        (eval_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "divergences": 0,
                    "ess_bulk_min": 158.0,
                    "ess_threshold": 400,
                    "passed": passed,
                    "rhat_max": 1.03,
                    "rhat_threshold": 1.01,
                }
            ),
            encoding="utf-8",
        )
        (eval_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "ppc": {
                        "summary": {
                            "mean": {"observed": 68.0, "p_value": 0.5, "mc_se": 0.0},
                            "skewness": {"observed": -1.8, "p_value": skew_p, "mc_se": 0.0},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return eval_dir

    def test_pinned_ppc_flagged(self, tmp_path):
        eval_dir = self._write_eval(tmp_path, passed=False, skew_p=0.998)
        report = build_report(eval_dir)
        assert "skewness" in report.extreme_ppc
        assert report.convergence["passed"] is False
        assert "FAILED" in report.verdict
        flags = {row["statistic"]: row["flag"] for row in report.ppc}
        assert flags["skewness"] == "pinned"
        assert flags["mean"] == "ok"

    def test_clean_run_verdict(self, tmp_path):
        eval_dir = self._write_eval(tmp_path, passed=True, skew_p=0.42)
        report = build_report(eval_dir)
        assert not report.extreme_ppc
        assert "no PPC statistics pinned" in report.verdict
        assert "## Convergence" in render_markdown(report)

    def test_missing_artifacts_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_report(tmp_path / "nope")

    def test_run_diagnose_writes_artifacts(self, tmp_path):
        eval_dir = self._write_eval(tmp_path, passed=False, skew_p=0.998)
        out_dir = tmp_path / "report"
        report = run_diagnose(eval_dir=eval_dir, output_dir=out_dir)
        assert (out_dir / "diagnostics_report.md").exists()
        assert (out_dir / "diagnostics_report.json").exists()
        assert report.artifacts


class TestErrorDecompositionCommand:
    """diagnose --errors over the identified predictions artifact (#180)."""

    def _write_predictions(self, tmp_path: Path, *, identified: bool) -> Path:
        eval_dir = tmp_path / "evaluation"
        split_dir = eval_dir / "within_entity_temporal"
        split_dir.mkdir(parents=True)
        payload = {
            "y_true": [70.0, 80.0],
            "y_pred_mean": [72.0, 75.0],
            "y_pred_lower": [60.0, 65.0],
            "y_pred_upper": [84.0, 85.0],
            "residuals": [-2.0, 5.0],
            "interval_level": 0.8,
        }
        if identified:
            payload |= {
                "entity": ["A", "B"],
                "event": ["a1", "b1"],
                "n_reviews": [10, 20],
                "train_history": [2, 0],
                "y_pred_sd": [2.0, 5.0],
                "pit": [0.2, 0.99],
                "covered": {"0.80": [True, False]},
            }
        (split_dir / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")
        return eval_dir

    def test_writes_csvs_and_top25(self, tmp_path):
        from panelcast.pipelines.diagnose import run_error_decomposition

        eval_dir = self._write_predictions(tmp_path, identified=True)
        out_dir = tmp_path / "reports"
        artifacts = run_error_decomposition(eval_dir=eval_dir, output_dir=out_dir)
        assert (out_dir / "error_decomposition_within_entity_temporal.csv").exists()
        assert (out_dir / "error_rollup_entity_within_entity_temporal.csv").exists()
        md = (out_dir / "error_top25_within_entity_temporal.md").read_text(encoding="utf-8")
        assert "| entity |" in md or "| event |" in md
        assert len(artifacts) >= 3

    def test_pre_feature_payload_degrades_clearly(self, tmp_path):
        from panelcast.pipelines.diagnose import run_error_decomposition

        eval_dir = self._write_predictions(tmp_path, identified=False)
        with pytest.raises(ValueError, match="identity fields"):
            run_error_decomposition(eval_dir=eval_dir, output_dir=tmp_path / "reports")

    def test_no_predictions_raises_file_not_found(self, tmp_path):
        from panelcast.pipelines.diagnose import run_error_decomposition

        empty = tmp_path / "evaluation"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            run_error_decomposition(eval_dir=empty, output_dir=tmp_path / "reports")
