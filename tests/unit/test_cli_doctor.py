"""`panelcast doctor` (#162): read-only preflight, never crashes, exit codes."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from panelcast.cli import app
from panelcast.doctor import CheckResult, run_doctor

runner = CliRunner()


class TestRunDoctor:
    def test_never_raises_and_covers_all_checks(self):
        results = run_doctor(None)
        names = [r.name for r in results]
        assert names == [
            "pixi.lock", "versions", "accelerator", "compile cache", "git",
            "dataset", "data stamps", "calibration", "disk",
        ]
        assert all(r.status in ("PASS", "WARN", "FAIL") for r in results)

    def test_broken_probe_becomes_fail_row(self, monkeypatch):
        import panelcast.doctor as doctor_mod

        def boom():
            raise RuntimeError("nvml exploded")

        result = doctor_mod._check("accelerator", boom)
        assert result.status == "FAIL"
        assert "nvml exploded" in result.detail

    def test_missing_dataset_fails_with_hint(self, tmp_path, monkeypatch):
        import panelcast.doctor as doctor_mod

        monkeypatch.chdir(tmp_path)  # descriptor default CSV not present here
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        result = doctor_mod._check("dataset", lambda: doctor_mod._dataset(None))
        assert result.status == "FAIL"
        assert result.hint and "AOTY_DATASET_PATH" in result.hint


class TestDoctorCli:
    def test_json_output_and_exit_code(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no lockfile, no dataset here
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        result = runner.invoke(app, ["doctor", "--json"])
        # Stray log lines may precede the payload; parse from the JSON start.
        start = result.output.index("[" + "\n")
        payload = json.loads(result.output[start:])
        assert {r["name"] for r in payload} >= {"versions", "dataset", "disk"}
        by_name = {r["name"]: r for r in payload}
        assert by_name["dataset"]["status"] == "FAIL"
        assert result.exit_code == 1

    def test_human_output_summarizes(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["doctor"])
        assert "pass" in result.output and "fail" in result.output
