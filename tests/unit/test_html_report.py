"""Run HTML dashboard (#159): composition, fallbacks, portability."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from panelcast.reporting.html_report import build_run_report, write_run_report

_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def run_dir(tmp_path):
    rd = tmp_path / "2026-07-08_000000_000000_abcd"
    (rd / "evaluation").mkdir(parents=True)
    (rd / "reports" / "figures").mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "2026-07-08_000000_000000_abcd",
                "command": "panelcast run --preset quick",
                "seed": 42,
                "success": True,
                "duration_seconds": 123.4,
                "version": "0.10.0",
                "git": {"commit": "abcdef1234567890", "dirty": False},
                "stage_durations": {"train": 100.0, "evaluate": 20.0},
            }
        ),
        encoding="utf-8",
    )
    (rd / "evaluation" / "metrics.json").write_text(
        json.dumps(
            {
                "calibration": {"within_tolerance": True},
                "splits": {
                    "within_entity_temporal": {
                        "n_test": 500,
                        "point_metrics": {"mae": 5.3, "rmse": 7.1, "r2": 0.5},
                        "crps": {"mean_crps": 3.9},
                        "calibration": {
                            "coverages": {
                                "0.80": {"empirical": 0.79},
                                "0.95": {"empirical": 0.94},
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (rd / "evaluation" / "diagnostics.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (rd / "reports" / "publication_readiness.json").write_text(
        json.dumps({"ready": False}), encoding="utf-8"
    )
    (rd / "reports" / "figures" / "reliability_primary.png").write_bytes(_PNG_1PX)
    return rd


class TestBuildRunReport:
    def test_composes_header_verdicts_and_metrics(self, run_dir):
        page = build_run_report(run_dir, interactive=False)
        assert "2026-07-08_000000_000000_abcd" in page
        assert "abcdef123456" in page
        assert "convergence: PASS" in page
        assert "calibration: PASS" in page
        assert "readiness: FAIL" in page
        assert "<td>5.300</td>" in page
        assert "Stage durations" in page and "train" in page

    def test_no_interactive_embeds_pngs(self, run_dir):
        page = build_run_report(run_dir, interactive=False)
        assert "data:image/png;base64," in page
        assert "reliability_primary" in page
        assert "plotly" not in page.lower()

    def test_degrades_on_bare_run_dir(self, tmp_path):
        bare = tmp_path / "bare_run"
        bare.mkdir()
        page = build_run_report(bare, interactive=False)
        assert "no manifest.json" in page
        assert "No metrics.json found." in page
        assert "No figures available." in page

    def test_missing_latest_raises(self, tmp_path, monkeypatch):
        import panelcast.paths as paths

        monkeypatch.setattr(paths, "resolve_latest", lambda *a, **k: None)
        with pytest.raises(FileNotFoundError, match="no latest run"):
            build_run_report(None)


class TestWriteRunReport:
    def test_writes_index_html_into_run_reports(self, run_dir):
        path = write_run_report(run_dir, interactive=False)
        assert path == run_dir / "reports" / "index.html"
        assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_explicit_output_path(self, run_dir, tmp_path):
        out = tmp_path / "elsewhere" / "summary.html"
        path = write_run_report(run_dir, output_path=out, interactive=False)
        assert path == out and out.exists()
