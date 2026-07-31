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
    TARGET_TRANSFORMS,
    ar_center_on_model_scale,
    logit_offset_from_summary,
    target_transform_from_summary,
)

SRC = Path(panelcast.__file__).resolve().parent
# scripts/ is where the idiom is most likely to reappear -- written fast, from
# another script rather than from the resolver module. Anchored on the repo via
# this file, not on the installed package: scripts are not installed, so a
# non-editable install would otherwise skip the root and narrow the guard back
# to the package with nothing to signal it.
REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
SCAN_ROOTS = [("panelcast", SRC)] + [("scripts", p) for p in (SCRIPTS,) if p.is_dir()]

GUARDED_KEYS = ("logit_offset", "target_transform")

# Exemptions are (module, key) pairs, not whole modules: the select modules
# read a target_transform off sweep arms and the enumerated space, which is a
# configuration axis rather than anything a fit recorded -- but a select module
# that started reading a recorded logit_offset is still a defect.
EXEMPT_READS = frozenset(
    {
        ("panelcast/pipelines/training_summary.py", "logit_offset"),
        ("panelcast/pipelines/training_summary.py", "target_transform"),
        ("panelcast/select/prior_screen.py", "target_transform"),
        ("panelcast/select/runner.py", "target_transform"),
        ("panelcast/select/space.py", "target_transform"),
        # A validation-ladder variant spec, not a recorded summary.
        ("scripts/experiment_preflight_validation.py", "target_transform"),
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

    @pytest.mark.parametrize("value", [True, np.True_, "wide", [0.5]])
    def test_non_numeric_recorded_offsets_are_rejected(self, value):
        """np.bool_ is not a bool subclass but floats to 1.0, so it sat in the
        gap between "numpy scalars are supported" and "booleans are not"."""
        with pytest.raises(ValueError, match="logit_offset"):
            logit_offset_from_summary({"logit_offset": value})

    @pytest.mark.parametrize("configured", [0.0, 0.25, "0.5", None])
    def test_a_configured_offset_survives_the_round_trip_to_a_consumer(self, configured):
        """The property that matters end to end: whatever the config accepts,
        serializes and resolves back to the same number. Fails if either guard
        drifts from the other."""
        config = PipelineConfig(run_id="offset-round-trip", logit_offset=configured)
        recorded = json.loads(json.dumps({"logit_offset": config.logit_offset}))
        assert logit_offset_from_summary(recorded) == pytest.approx(config.logit_offset)

    def test_numpy_scalars_resolve(self):
        """An in-process summary (sensitivity takes one as a parameter) can
        carry numpy scalars that never pass through JSON."""
        assert logit_offset_from_summary({"logit_offset": np.float32(0.25)}) == pytest.approx(0.25)


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

    @pytest.mark.parametrize("value", ["", "   ", 42])
    def test_a_recorded_non_name_is_rejected(self, value):
        """The empty string pins that a falsy recorded value is not silently
        rewritten to the default -- the same shape as substituting a zero
        offset. Unknown *names* stay the registry's job, since it is extensible
        and already raises with the registered set."""
        with pytest.raises(ValueError, match="target_transform"):
            target_transform_from_summary({"target_transform": value})

    def test_an_unknown_name_reaches_the_registry_error(self):
        recorded = {"target_transform": "offset-logit"}
        assert target_transform_from_summary(recorded) == "offset-logit"
        with pytest.raises(ValueError, match="Unknown target_transform"):
            get_transform(
                target_transform_from_summary(recorded), target_bounds=(0.0, 100.0), offset=0.5
            )

    def test_a_padded_name_resolves_to_the_bare_one(self):
        """Otherwise the read path accepts a name the config side rejects, and
        a padded "identity" takes the non-identity branch in predict_next."""
        assert target_transform_from_summary({"target_transform": " identity "}) == "identity"

    def test_the_write_path_records_a_resolved_name(self):
        """The identity fallback is only correct because a null never reaches
        the summary from a post-gate run: resolve_model_facts fills it in."""
        config = PipelineConfig(run_id="transform-unset")
        assert config.target_transform is None
        resolve_model_facts(config, DEFAULT_DESCRIPTOR)
        assert config.target_transform in TARGET_TRANSFORMS

    def test_in_place_resolution_re_coerces_what_it_did_not_set(self):
        """resolve_model_facts is the one established in-place mutation path,
        and it ends with _validate(). Assigning past the constructor would
        otherwise re-open the write-then-crash-on-read gap: this fails if that
        trailing re-validation is ever dropped."""
        config = PipelineConfig(run_id="transform-unset")
        config.logit_offset = "0.25"
        config.target_transform = " offset_logit "
        resolve_model_facts(config, DEFAULT_DESCRIPTOR)
        assert config.logit_offset == 0.25
        assert isinstance(config.logit_offset, float)
        assert config.target_transform == "offset_logit"
        # _validate now mutates, so idempotence is load-bearing for its four
        # callers rather than incidental.
        config._validate()
        assert config.logit_offset == 0.25
        assert config.target_transform == "offset_logit"


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

    The guard cannot tell a recorded field from a configured one, so a
    legitimate config-axis read (a sweep arm, a YAML loader) needs an exemption.
    Adding one widens a hole rather than annotating a false positive: the pair
    is exempt for good, including for a summary read that module grows later.
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
        offenders: dict[str, list[tuple[int, str]]] = {}
        exercised: set[tuple[str, str]] = set()
        scanned_modules: set[str] = set()
        for label, root in SCAN_ROOTS:
            for path in sorted(root.rglob("*.py")):
                # Root-qualified: an exemption for scripts/foo.py must not also
                # exempt a package module that happens to share its basename.
                module = f"{label}/{path.relative_to(root).as_posix()}"
                scanned_modules.add(module)
                found = []
                for read in self._inline_reads(path):
                    if (module, read[1]) in EXEMPT_READS:
                        exercised.add((module, read[1]))
                    else:
                        found.append(read)
                if found:
                    offenders[module] = found
        # A source-stripped install would scan nothing and pass vacuously.
        assert scanned_modules >= {
            "panelcast/pipelines/evaluate.py",
            "panelcast/pipelines/predict_next.py",
            "panelcast/pipelines/sensitivity.py",
            "panelcast/pipelines/training_summary.py",
        }, f"the guard did not reach the consumer modules; it scanned {sorted(scanned_modules)}"
        # Gated on the repo, not on the root the assertion is checking: the
        # is_dir() skip must not be able to switch the scan off silently.
        if (REPO / "pyproject.toml").exists():
            assert "scripts/predict_entity.py" in scanned_modules
        assert offenders == {}, (
            f"inline reads of {GUARDED_KEYS} outside the resolvers: {offenders}; "
            "use logit_offset_from_summary / target_transform_from_summary."
        )
        # Only roots that were actually scanned can retire an exemption: under a
        # wheel install scripts/ is absent, and its entries are not stale.
        scanned_labels = {label for label, _ in SCAN_ROOTS}
        stale = {
            entry
            for entry in EXEMPT_READS - exercised
            if entry[0].split("/", 1)[0] in scanned_labels
        }
        assert stale == set(), (
            f"exemptions no longer matched by any read: {sorted(stale)}; "
            "drop them rather than pre-blessing a future violation."
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
        assert sorted(self._inline_reads(sample)) == sorted(
            [
                (1, "logit_offset"),
                (2, "target_transform"),
                (3, "logit_offset"),
                (4, "target_transform"),
            ]
        )


class TestConfigValidation:
    def test_zero_is_a_supported_configuration(self):
        config = PipelineConfig(run_id="offset-zero", logit_offset=0.0)
        assert config.logit_offset == 0.0

    @pytest.mark.parametrize("value", [-0.5, math.nan, math.inf, -math.inf])
    def test_out_of_range_offsets_are_rejected(self, value):
        with pytest.raises(ValueError, match="logit_offset"):
            PipelineConfig(run_id="offset-bad", logit_offset=value)

    @pytest.mark.parametrize("value", ["wide", True, [0.5]])
    def test_non_numeric_offsets_raise_value_error_not_type_error(self, value):
        """Callers wrap config construction in `except ValueError`."""
        with pytest.raises(ValueError, match="logit_offset"):
            PipelineConfig(run_id="offset-bad-type", logit_offset=value)

    def test_null_is_the_unset_sentinel_on_both_paths(self):
        """`logit_offset: null` in YAML reaches the dataclass as None, and it
        resolved to the default before this change."""
        assert PipelineConfig(run_id="offset-null", logit_offset=None).logit_offset == (
            DEFAULT_LOGIT_OFFSET
        )

    def test_an_unrecognized_transform_is_rejected(self):
        with pytest.raises(ValueError, match="target_transform"):
            PipelineConfig(run_id="transform-bad", target_transform="offset-logit")

    def test_a_parseable_offset_is_normalized_not_merely_checked(self):
        """A YAML string that only validated would reach the summary verbatim
        and then be rejected on every read, after training had already run."""
        config = PipelineConfig(run_id="offset-str", logit_offset="0.25")
        assert config.logit_offset == 0.25
        assert isinstance(config.logit_offset, float)

    def test_the_config_default_is_the_resolver_default(self):
        assert PipelineConfig(run_id="offset-default").logit_offset == DEFAULT_LOGIT_OFFSET

    def test_manifest_flag_restoration_goes_through_the_constructor(self, tmp_path):
        """setattr past the constructor was the one mutation shape a search for
        `config.logit_offset` cannot find: `runs reproduce` rebuilt a pre-0.9.0
        config from a flags dict, so a recorded value reached the re-executed
        run neither validated nor normalized. Now built like the sibling
        branch, which has always been validated by construction."""
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config

        manifest = SimpleNamespace(flags={"logit_offset": "0.25", "seed": 7})
        config, provenance = _reproduce_config(tmp_path / "missing", manifest)
        assert config.logit_offset == 0.25
        assert isinstance(config.logit_offset, float)
        assert config.seed == 7
        assert "manifest flags" in provenance

    def test_manifest_flag_restoration_rejects_an_invalid_recorded_flag(self, tmp_path):
        """Re-executing a recorded run under an invalid config is not
        best-effort; the resolved_config branch has always refused it."""
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config

        manifest = SimpleNamespace(flags={"logit_offset": -1.0})
        with pytest.raises(ValueError, match="logit_offset"):
            _reproduce_config(tmp_path / "missing", manifest)

    def test_a_run_id_that_is_not_a_bare_directory_name_is_refused(self, tmp_path):
        """The restored run_id now goes through validate_run_id, which is a
        containment rule rather than a naming convention that narrowed: an id
        with a path separator was never a directory a run could live in, so
        reproducing it is refusing an artifact that was already unusable."""
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config
        from panelcast.paths import RunPathError

        manifest = SimpleNamespace(flags={"run_id": "../escape"})
        with pytest.raises(RunPathError, match="run_id"):
            _reproduce_config(tmp_path / "missing", manifest)

    def test_a_recorded_list_is_restored_as_the_tuple_the_field_declares(self, tmp_path):
        """JSON has no tuples, so the coercion is what keeps a restored config
        the same type as every other construction path."""
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config

        manifest = SimpleNamespace(flags={"calibration_intervals": [0.5, 0.9]})
        config, _ = _reproduce_config(tmp_path / "missing", manifest)
        assert config.calibration_intervals == (0.5, 0.9)

    def test_a_flag_whose_companion_gate_was_never_recorded_still_restores(self, tmp_path):
        """The fallback rebuilds a deliberately partial config: a knob recorded
        without the YAML-only gate it pairs with keeps the current default for
        that gate rather than failing the reproduction."""
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config

        manifest = SimpleNamespace(flags={"sigma_rw_lognormal_loc": -2.5, "seed": 11})
        config, _ = _reproduce_config(tmp_path / "missing", manifest)
        # Asserting the restored value, not just the unrelated one: without it
        # a renamed field would be dropped by the defaults filter and the test
        # would keep passing while covering nothing.
        assert config.sigma_rw_lognormal_loc == -2.5
        assert config.auto_priors is None
        assert config.seed == 11

    def test_a_pairing_todays_cross_field_rule_rejects_fails_the_reproduction(self, tmp_path):
        """The other half: a manifest recording both the auto toggle and an
        explicit sigma loc. Refusing is the intended outcome -- the two cannot
        both take effect, so re-executing would run an experiment that is not
        the recorded one."""
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config

        manifest = SimpleNamespace(
            flags={"auto_priors": True, "sigma_rw_lognormal_loc": -2.5}
        )
        with pytest.raises(ValueError, match="auto_priors"):
            _reproduce_config(tmp_path / "missing", manifest)

    def test_unrecorded_flags_keep_their_defaults(self, tmp_path):
        from types import SimpleNamespace

        from panelcast.cli.runs_cmd import _reproduce_config

        manifest = SimpleNamespace(flags={"not_a_config_field": 1, "seed": 3})
        config, _ = _reproduce_config(tmp_path / "missing", manifest)
        assert config.seed == 3
        assert config.logit_offset == DEFAULT_LOGIT_OFFSET
