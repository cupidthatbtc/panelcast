"""Preflight-ladder variant grid (#104).

Covers the GPU-free pieces of scripts/experiment_preflight_validation.py:
variant-grid construction and the gbm column-drop width probe. The script
lives in ``scripts/`` (not an installed package), so it is loaded by path.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "experiment_preflight_validation.py"
_spec = importlib.util.spec_from_file_location("experiment_preflight_validation", _SCRIPT)
ladder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ladder)


def test_variant_grid_full():
    variants = ladder.build_variants("user", ["a", "gbm_offset"], pooling_available=True)
    names = [v["name"] for v in variants]
    assert names == [
        "identity_transform",
        "exclude_rw_raw",
        "entity_group_pooling",
        "gbm_column_dropped",
        "vectorized_chains",
    ]


def test_identity_variant_flips_transform():
    variants = ladder.build_variants("user", [], pooling_available=False)
    identity = next(v for v in variants if v["name"] == "identity_transform")
    assert identity["target_transform"] == "identity"


def test_exclude_variant_uses_prefixed_site():
    variants = ladder.build_variants("perf", [], pooling_available=False)
    excl = next(v for v in variants if v["name"] == "exclude_rw_raw")
    assert excl["exclude_collection"] == ("perf_rw_raw",)


def test_pooling_variant_skipped_when_unavailable():
    names = [v["name"] for v in ladder.build_variants("user", [], pooling_available=False)]
    assert "entity_group_pooling" not in names


def test_gbm_variant_skipped_without_column():
    names = [v["name"] for v in ladder.build_variants("user", ["a", "b"], pooling_available=True)]
    assert "gbm_column_dropped" not in names


def test_gbm_variant_marks_drop_column():
    variants = ladder.build_variants("user", ["a", "gbm_offset"], pooling_available=False)
    gbm = next(v for v in variants if v["name"] == "gbm_column_dropped")
    assert gbm["drop_column"] == "gbm_offset"


def test_baseline_is_production_transform():
    assert ladder.BASELINE_TARGET_TRANSFORM == "offset_logit"
    assert ladder.ANCHOR_RUNG in {r["name"] for r in ladder.RUNGS}


def test_drop_feature_column_removes_only_named():
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = ladder.drop_feature_column(X, ["a", "gbm_offset", "b", "c"], "gbm_offset")
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, X[:, [0, 2, 3]])


def test_drop_feature_column_missing_raises():
    with pytest.raises(ValueError, match="gbm_offset"):
        ladder.drop_feature_column(np.zeros((2, 2)), ["a", "b"], "gbm_offset")
