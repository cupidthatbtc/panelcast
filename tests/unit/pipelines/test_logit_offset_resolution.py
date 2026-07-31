"""A recorded logit_offset of zero is a configuration, not a missing value.

Training accepts and records ``logit_offset=0.0`` (the plain logit, valid when
observations sit strictly inside the bounds). Consumers used to resolve it with
``float(summary.get("logit_offset") or 0.5)``, which cannot tell numeric zero
from an absent key, so evaluation, prediction and rollout applied a different
forward transform, inverse and Jacobian than the fit used. Every consumer now
reads through :func:`logit_offset_from_summary`; this module pins the resolver's
semantics and guards against the two idioms drifting apart again.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.evaluate import _transform_from_summary
from panelcast.pipelines.pipeline_config import PipelineConfig
from panelcast.pipelines.training_summary import (
    DEFAULT_LOGIT_OFFSET,
    ar_center_on_model_scale,
    logit_offset_from_summary,
)

SRC = Path(__file__).resolve().parents[3] / "src" / "panelcast"

# Modules that resolve the recorded offset from a training summary.
CONSUMERS = (
    "pipelines/evaluate.py",
    "pipelines/predict_next.py",
    "pipelines/sensitivity.py",
)


def _summary(**overrides) -> dict:
    summary: dict = {
        "target_transform": "offset_logit",
        "dataset": {"target_bounds": [0.0, 100.0]},
        "priors": {"ar_center": "global"},
        "ar_center_value": 70.0,
    }
    summary.update(overrides)
    return summary


class TestResolver:
    def test_missing_key_falls_back_to_the_default(self):
        assert logit_offset_from_summary({}) == DEFAULT_LOGIT_OFFSET

    def test_explicit_null_falls_back_to_the_default(self):
        """Legacy/pre-gate summaries serialize the field as null, not absent."""
        assert logit_offset_from_summary({"logit_offset": None}) == DEFAULT_LOGIT_OFFSET

    def test_recorded_zero_is_propagated(self):
        assert logit_offset_from_summary({"logit_offset": 0.0}) == 0.0

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 1.0])
    def test_recorded_value_is_returned_verbatim(self, value):
        assert logit_offset_from_summary({"logit_offset": value}) == value

    def test_survives_a_json_round_trip(self):
        """The summary reaches consumers as a file, not a live dict."""
        raw = json.loads(json.dumps({"logit_offset": 0.0}))
        assert logit_offset_from_summary(raw) == 0.0


class TestConsumersUseTheRecordedOffset:
    def test_evaluate_transform_keeps_zero(self):
        transform = _transform_from_summary(_summary(logit_offset=0.0))
        assert transform.name == "offset_logit"
        assert transform.offset == 0.0

    def test_evaluate_transform_matches_the_fitted_forward_map(self):
        """A zero offset is the plain logit; the default 0.5 is a different map."""
        transform = _transform_from_summary(_summary(logit_offset=0.0))
        expected = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.0)
        y = np.array([10.0, 50.0, 90.0])
        np.testing.assert_allclose(np.asarray(transform.forward(y)), np.asarray(expected.forward(y)))
        default = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.5)
        assert not np.allclose(np.asarray(transform.forward(y)), np.asarray(default.forward(y)))

    def test_ar_center_uses_the_recorded_offset(self):
        center = ar_center_on_model_scale(_summary(logit_offset=0.0))
        expected = get_transform("offset_logit", target_bounds=(0.0, 100.0), offset=0.0)
        assert center == pytest.approx(float(expected.forward(70.0)))

    def test_legacy_summary_still_gets_the_default_offset(self):
        transform = _transform_from_summary(_summary())
        assert transform.offset == DEFAULT_LOGIT_OFFSET


class TestSingleResolver:
    @pytest.mark.parametrize("relative", CONSUMERS)
    def test_no_consumer_re_derives_the_offset(self, relative):
        """``summary.get("logit_offset")`` only ever appears inside the resolver.

        Both historical idioms (``or 0.5`` and ``.get(..., 0.5)``) disagreed on
        zero and on an explicit null respectively; one call site is the only way
        they cannot drift apart again.
        """
        tree = ast.parse((SRC / relative).read_text(encoding="utf-8"))
        offending = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "logit_offset"
        ]
        assert offending == [], (
            f"{relative} resolves logit_offset inline at lines {offending}; "
            "use logit_offset_from_summary instead."
        )


class TestConfigValidation:
    def test_zero_is_a_supported_configuration(self):
        config = PipelineConfig(run_id="offset-zero", logit_offset=0.0)
        assert config.logit_offset == 0.0

    @pytest.mark.parametrize("value", [-0.5, math.nan, math.inf])
    def test_out_of_range_offsets_are_rejected(self, value):
        with pytest.raises(ValueError, match="logit_offset"):
            PipelineConfig(run_id="offset-bad", logit_offset=value)
