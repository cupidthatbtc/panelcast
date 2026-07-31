"""A recorded logit_offset of zero is a configuration, not a missing value.

Training accepts and records ``logit_offset=0.0`` (the plain logit, valid when
observations sit strictly inside the bounds). Consumers used to resolve it with
``float(summary.get("logit_offset") or 0.5)``, which cannot tell numeric zero
from an absent key, so evaluation, prediction and rollout applied a different
forward transform, inverse and Jacobian than the fit used. Every consumer now
reads through :func:`logit_offset_from_summary`; this module pins the resolver's
semantics and guards against the two idioms drifting apart again. The sibling
``target_transform`` field, read the same two disagreeing ways, is covered here
too.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

import panelcast
from panelcast.config.descriptor import DEFAULT_DESCRIPTOR
from panelcast.models.bayes.transforms import get_transform
from panelcast.pipelines.evaluate import _transform_from_summary
from panelcast.pipelines.pipeline_config import PipelineConfig, resolve_model_facts
from panelcast.pipelines.training_summary import (
    DEFAULT_LOGIT_OFFSET,
    ar_center_on_model_scale,
    logit_offset_from_summary,
    target_transform_from_summary,
)

SRC = Path(panelcast.__file__).resolve().parent

GUARDED_KEYS = ("logit_offset", "target_transform")

# The resolvers own the recorded keys. The select modules read a
# target_transform off sweep arms and the enumerated space, which is a
# configuration axis rather than anything a fit recorded.
EXEMPT_MODULES = frozenset(
    {
        "pipelines/training_summary.py",
        "select/prior_screen.py",
        "select/runner.py",
        "select/space.py",
    }
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

    @pytest.mark.parametrize("value", [-0.5, math.nan, math.inf, -math.inf])
    def test_out_of_range_recorded_offsets_are_rejected(self, value):
        """Every summary on disk predates the config-side guard, so the read
        path is the one that actually meets an unvalidated offset."""
        with pytest.raises(ValueError, match="logit_offset"):
            logit_offset_from_summary({"logit_offset": value})

    @pytest.mark.parametrize("value", [True, "wide", [0.5]])
    def test_non_numeric_recorded_offsets_are_rejected(self, value):
        with pytest.raises(ValueError, match="logit_offset"):
            logit_offset_from_summary({"logit_offset": value})

    @pytest.mark.parametrize("value", ["0.5", np.float32(0.25)])
    def test_the_read_path_accepts_everything_the_write_path_records(self, value):
        """Both sides coerce through the same helper, so a config that trains
        can never produce a summary that crashes every consumer."""
        assert logit_offset_from_summary({"logit_offset": value}) == pytest.approx(float(value))


class TestTargetTransformResolver:
    def test_missing_key_resolves_to_identity(self):
        assert target_transform_from_summary({}) == "identity"

    def test_explicit_null_resolves_to_identity(self):
        """The .get(..., "identity") idiom returned None here, which reached
        get_transform as a transform name."""
        assert target_transform_from_summary({"target_transform": None}) == "identity"

    def test_recorded_name_is_returned(self):
        recorded = {"target_transform": "offset_logit"}
        assert target_transform_from_summary(recorded) == "offset_logit"

    def test_the_write_path_records_a_resolved_name(self):
        """The identity fallback is only correct because a null never reaches
        the summary from a post-gate run: resolve_model_facts fills it in."""
        config = PipelineConfig(run_id="transform-unset")
        assert config.target_transform is None
        resolve_model_facts(config, DEFAULT_DESCRIPTOR)
        assert config.target_transform is not None


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
    """No module outside the resolvers re-derives either recorded field.

    Scans the whole package with an explicit exemption list rather than a
    roster of today's consumers: a new pipeline, report writer or CLI path that
    starts reading the summary fails by default, whatever it names its variable.
    Both access forms count -- ``.get("logit_offset")`` and the subscript
    ``["logit_offset"]`` -- because the historical idioms disagreed on zero and
    on an explicit null respectively.
    """

    def _inline_reads(self, path: Path) -> list[tuple[int, str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in GUARDED_KEYS
            ):
                found.append((node.lineno, str(node.args[0].value)))
            # Reads only: train_bayes writes summary["logit_offset"] = ...
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in GUARDED_KEYS
            ):
                found.append((node.lineno, str(node.slice.value)))
        return found

    def test_the_resolvers_are_the_only_readers(self):
        offenders = {
            str(path.relative_to(SRC)): self._inline_reads(path)
            for path in sorted(SRC.rglob("*.py"))
            if str(path.relative_to(SRC)).replace("\\", "/") not in EXEMPT_MODULES
        }
        offenders = {k: v for k, v in offenders.items() if v}
        assert offenders == {}, (
            f"inline reads of {GUARDED_KEYS} outside {sorted(EXEMPT_MODULES)}: {offenders}; "
            "use logit_offset_from_summary / target_transform_from_summary."
        )

    def test_the_guard_sees_both_access_forms_and_any_receiver(self, tmp_path):
        """A guard keyed on `.get()` or on the variable being named `summary`
        would miss most of the ways the next consumer will write this."""
        sample = tmp_path / "sample.py"
        sample.write_text(
            "a = summary.get('logit_offset')\n"
            "b = loaded['target_transform']\n"
            "c = self.summary.get('logit_offset')\n"
            "d = ctx.summary['target_transform']\n"
            "summary['logit_offset'] = 1.0\n",
            encoding="utf-8",
        )
        assert self._inline_reads(sample) == [
            (1, "logit_offset"),
            (2, "target_transform"),
            (3, "logit_offset"),
            (4, "target_transform"),
        ]


class TestConfigValidation:
    def test_zero_is_a_supported_configuration(self):
        config = PipelineConfig(run_id="offset-zero", logit_offset=0.0)
        assert config.logit_offset == 0.0

    @pytest.mark.parametrize("value", [-0.5, math.nan, math.inf])
    def test_out_of_range_offsets_are_rejected(self, value):
        with pytest.raises(ValueError, match="logit_offset"):
            PipelineConfig(run_id="offset-bad", logit_offset=value)

    @pytest.mark.parametrize("value", ["wide", True, None])
    def test_non_numeric_offsets_raise_value_error_not_type_error(self, value):
        """Callers wrap config construction in `except ValueError`."""
        with pytest.raises(ValueError, match="logit_offset"):
            PipelineConfig(run_id="offset-bad-type", logit_offset=value)

    def test_a_parseable_offset_is_normalized_not_merely_checked(self):
        """A YAML string that only validated would reach the summary verbatim
        and then be rejected on every read, after training had already run."""
        config = PipelineConfig(run_id="offset-str", logit_offset="0.25")
        assert config.logit_offset == 0.25
        assert isinstance(config.logit_offset, float)

    def test_the_config_default_is_the_resolver_default(self):
        assert PipelineConfig(run_id="offset-default").logit_offset == DEFAULT_LOGIT_OFFSET
