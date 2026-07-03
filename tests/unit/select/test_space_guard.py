"""CI guard: the select knob table stays in sync with the code's own surfaces.

A new gate boolean or Literal-typed field on ``PipelineConfig``, a new value on
a ``config/gates.py`` Literal, or a new likelihood family must show up in the
candidate space (or be explicitly excluded with a reason) — otherwise these
tests fail and the addition can't silently escape the selection protocol.
"""

from __future__ import annotations

import types
from dataclasses import fields as dataclass_fields
from typing import Union, get_args, get_origin, get_type_hints

from panelcast.config import gates
from panelcast.models.bayes.likelihoods import REGISTRY
from panelcast.pipelines.orchestrator import PipelineConfig
from panelcast.select.space import EXCLUDED_FIELDS, KNOBS

HINTS = get_type_hints(PipelineConfig)
GATE_ALIASES = {name: getattr(gates, name) for name in gates.__all__}
KNOB_NAMES = {knob.name for knob in KNOBS}
COVERED = KNOB_NAMES | set(EXCLUDED_FIELDS)


def _is_bool_like(hint) -> bool:
    if hint is bool:
        return True
    if get_origin(hint) in (Union, types.UnionType):
        return set(get_args(hint)) == {bool, type(None)}
    return False


def _gate_literal_alias(hint):
    for name, alias in GATE_ALIASES.items():
        if hint == alias:
            return name
    return None


class TestFieldCoverage:
    def test_every_bool_gate_is_covered(self):
        missing = [
            f.name
            for f in dataclass_fields(PipelineConfig)
            if _is_bool_like(HINTS[f.name]) and f.name not in COVERED
        ]
        assert not missing, (
            f"PipelineConfig bool gate(s) {missing} are neither select knobs nor "
            "excluded with a reason in panelcast.select.space"
        )

    def test_every_literal_gate_is_covered(self):
        missing = [
            f.name
            for f in dataclass_fields(PipelineConfig)
            if _gate_literal_alias(HINTS[f.name]) and f.name not in COVERED
        ]
        assert not missing, (
            f"PipelineConfig Literal gate(s) {missing} are neither select knobs "
            "nor excluded with a reason in panelcast.select.space"
        )

    def test_every_gates_literal_reaches_a_config_field(self):
        used = {
            _gate_literal_alias(HINTS[f.name])
            for f in dataclass_fields(PipelineConfig)
        } - {None}
        orphaned = set(GATE_ALIASES) - used
        assert not orphaned, f"gates.py Literal(s) {orphaned} back no PipelineConfig field"

    def test_knobs_and_exclusions_are_disjoint(self):
        overlap = KNOB_NAMES & set(EXCLUDED_FIELDS)
        assert not overlap

    def test_every_knob_is_a_config_field(self):
        config_fields = {f.name for f in dataclass_fields(PipelineConfig)}
        assert KNOB_NAMES <= config_fields


class TestValueCoverage:
    def test_literal_knob_values_match_the_literal(self):
        for knob in KNOBS:
            alias_name = _gate_literal_alias(HINTS[knob.name])
            if alias_name is None or knob.name == "likelihood_family":
                continue
            assert set(knob.values) == set(get_args(GATE_ALIASES[alias_name])), knob.name

    def test_family_knob_values_match_the_registry(self):
        (knob,) = [k for k in KNOBS if k.name == "likelihood_family"]
        assert set(knob.values) == set(REGISTRY)

    def test_bool_knobs_offer_both_values(self):
        for knob in KNOBS:
            if knob.kind == "bool":
                assert set(knob.values) == {True, False}, knob.name
            if knob.kind == "tristate":
                assert set(knob.values) == {None, True, False}, knob.name

    def test_transform_registry_matches_the_literal(self):
        # The transform factories are a fourth enumeration surface; a transform
        # registered without a Literal value (or vice versa) must not slip past
        # the candidate space.
        from panelcast.models.bayes.transforms import _TRANSFORM_FACTORIES

        assert set(_TRANSFORM_FACTORIES) == set(get_args(gates.TargetTransform))
