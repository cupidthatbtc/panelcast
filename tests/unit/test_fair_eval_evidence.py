from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / ".audit" / "fair_eval_0131"


def test_machine_record_recomputes_paired_elpd() -> None:
    record = json.loads((AUDIT / "fair_eval.json").read_text(encoding="utf-8"))
    paired = record["paired_elpd"]
    pointwise = np.asarray(paired["pointwise_difference"], dtype=float)

    assert pointwise.size == paired["n"] == 653
    assert pointwise.sum() == pytest.approx(paired["difference"], abs=1e-10)
    paired_se = math.sqrt(pointwise.size * np.var(pointwise, ddof=1))
    assert paired_se == pytest.approx(paired["paired_se"], abs=1e-10)
    assert pointwise.sum() / paired_se == pytest.approx(paired["z"], abs=1e-10)


def test_machine_record_pins_generated_baselines() -> None:
    record = json.loads((AUDIT / "fair_eval.json").read_text(encoding="utf-8"))
    baseline_bytes = (AUDIT / "baseline_comparison.json").read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest() == record["baseline_comparison_sha256"]

    rows = json.loads(baseline_bytes)
    ridge = next(
        row
        for row in rows
        if row["model"] == "ridge" and row["split"] == "within_entity_temporal"
    )
    assert ridge["mae"] == pytest.approx(5.388237231499237)
    assert ridge["r2"] == pytest.approx(0.4938053844226804)
