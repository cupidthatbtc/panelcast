from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import arviz as az
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / ".audit" / "fair_eval_0131"
_SPEC = importlib.util.spec_from_file_location("fair_eval_builder", AUDIT / "build_evidence.py")
assert _SPEC is not None and _SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILDER)


def test_machine_record_recomputes_paired_elpd() -> None:
    record = json.loads((AUDIT / "fair_eval.json").read_text(encoding="utf-8"))
    paired = record["paired_elpd"]
    pointwise = np.asarray(paired["pointwise_difference"], dtype=float)

    assert pointwise.size == paired["n"] == 653
    assert pointwise.sum() == pytest.approx(paired["difference"], abs=1e-10)
    paired_se = math.sqrt(pointwise.size * np.var(pointwise, ddof=1))
    assert paired_se == pytest.approx(paired["paired_se"], abs=1e-10)
    assert pointwise.sum() / paired_se == pytest.approx(paired["z"], abs=1e-10)


def test_builder_derives_paired_elpd_from_netcdf(tmp_path) -> None:
    entity_values = np.array([[[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]]])
    incumbent_values = np.array([[[-1.1, -2.2, -3.4], [-1.1, -2.2, -3.4]]])
    entity_path = tmp_path / "entity.nc"
    incumbent_path = tmp_path / "incumbent.nc"
    az.from_dict(log_likelihood={"y": entity_values}).to_netcdf(entity_path)
    az.from_dict(log_likelihood={"y": incumbent_values}).to_netcdf(incumbent_path)

    paired = _BUILDER.paired_elpd(entity_path, incumbent_path)

    np.testing.assert_allclose(paired["pointwise_difference"], [0.1, 0.2, 0.4])
    assert paired["difference"] == pytest.approx(0.7)
    expected_se = math.sqrt(3 * np.var([0.1, 0.2, 0.4], ddof=1))
    assert paired["paired_se"] == pytest.approx(expected_se)
    assert paired["z"] == pytest.approx(0.7 / expected_se)


def test_builder_rejects_unpaired_outcomes(tmp_path) -> None:
    entity = tmp_path / "entity"
    incumbent = tmp_path / "incumbent"
    relative = Path("evaluation/within_entity_temporal/predictions.json")
    (entity / relative).parent.mkdir(parents=True)
    (incumbent / relative).parent.mkdir(parents=True)
    (entity / relative).write_text(json.dumps({"y_true": [1.0, 2.0]}), encoding="utf-8")
    (incumbent / relative).write_text(
        json.dumps({"y_true": [1.0, 3.0]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="same ordered outcomes"):
        _BUILDER.validate_pairing(entity, incumbent, {"rows": 2})


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
