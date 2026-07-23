"""Tests for publication artifact helpers."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.publication import (
    SECONDARY_SPLIT,
    _build_publication_readiness,
    _ConvergenceLike,
    _CoverageLike,
    _fan_plot_kwargs,
    _get_coefficient_var_names,
    _LooLike,
    _parse_convergence,
    _parse_coverage_results,
    _parse_loo_result,
    _parse_point_metrics,
    _PointMetricsLike,
    _render_publication_readiness_markdown,
    _resolve_primary_metrics,
    _safe_float,
    _safe_int,
    _uses_default_plot_presentation,
    generate_publication_artifacts,
)


class MockPosterior:
    def __init__(self, var_names):
        self._var_names = set(var_names)

    def __contains__(self, key):
        return key in self._var_names


class MockIData:
    def __init__(self, var_names):
        self.posterior = MockPosterior(var_names)


def _check_map(payload: dict) -> dict[str, dict]:
    return {check["name"]: check for check in payload["checks"]}


class TestSafeParsers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (1, 1.0),
            (-1, -1.0),
            (0.0, 0.0),
            (1.25, 1.25),
            ("1.5", 1.5),
            ("-0.125", -0.125),
            ("1e3", 1000.0),
            (True, 1.0),
            (False, 0.0),
        ],
    )
    def test_safe_float_valid_inputs(self, value, expected):
        assert _safe_float(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            " ",
            "nan-ish",
            object(),
            [],
            {},
            (1, 2),
        ],
    )
    def test_safe_float_invalid_inputs(self, value):
        assert _safe_float(value) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0),
            (1, 1),
            (-1, -1),
            (1.2, 1),
            ("2", 2),
            ("-12", -12),
            (True, 1),
            (False, 0),
        ],
    )
    def test_safe_int_valid_inputs(self, value, expected):
        assert _safe_int(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "abc",
            "1.5",
            object(),
            [],
            {},
            (1,),
        ],
    )
    def test_safe_int_invalid_inputs(self, value):
        assert _safe_int(value) is None


class TestGetCoefficientVarNames:
    def test_basic_vars_always_present(self):
        idata = MockIData(["user_beta", "user_mu_artist", "user_sigma_artist", "user_sigma_obs"])
        result = _get_coefficient_var_names(idata)
        assert "user_beta" in result
        assert "user_mu_artist" in result
        assert "user_sigma_artist" in result
        assert "user_sigma_obs" in result

    def test_includes_sigma_ref_when_present(self):
        idata = MockIData(
            [
                "user_beta",
                "user_mu_artist",
                "user_sigma_artist",
                "user_sigma_obs",
                "user_sigma_ref",
            ]
        )
        result = _get_coefficient_var_names(idata)
        assert "user_sigma_ref" in result

    def test_excludes_sigma_ref_when_absent(self):
        idata = MockIData(["user_beta", "user_mu_artist", "user_sigma_artist", "user_sigma_obs"])
        result = _get_coefficient_var_names(idata)
        assert "user_sigma_ref" not in result

    def test_includes_n_exponent_when_present(self):
        idata = MockIData(
            [
                "user_beta",
                "user_mu_artist",
                "user_sigma_artist",
                "user_sigma_obs",
                "user_n_exponent",
            ]
        )
        result = _get_coefficient_var_names(idata)
        assert "user_n_exponent" in result

    def test_excludes_n_exponent_when_absent(self):
        idata = MockIData(["user_beta", "user_mu_artist", "user_sigma_artist", "user_sigma_obs"])
        result = _get_coefficient_var_names(idata)
        assert "user_n_exponent" not in result

    def test_custom_prefix(self):
        idata = MockIData(
            [
                "critic_beta",
                "critic_mu_artist",
                "critic_sigma_artist",
                "critic_sigma_obs",
            ]
        )
        result = _get_coefficient_var_names(idata, prefix="critic_")
        assert "critic_beta" in result
        assert "critic_sigma_obs" in result

    def test_ordering_sigma_ref_before_sigma_obs(self):
        idata = MockIData(
            [
                "user_beta",
                "user_mu_artist",
                "user_sigma_artist",
                "user_sigma_obs",
                "user_sigma_ref",
            ]
        )
        result = _get_coefficient_var_names(idata)
        ref_idx = result.index("user_sigma_ref")
        obs_idx = result.index("user_sigma_obs")
        assert ref_idx < obs_idx


class TestResolvePrimaryMetrics:
    def test_resolve_split_aware_payload(self):
        primary = {"point_metrics": {"rmse": 1.2, "mae": 0.8, "r2": 0.5}}
        metrics = {
            "primary_split": "within_entity_temporal",
            "splits": {
                "within_entity_temporal": primary,
                SECONDARY_SPLIT: {"point_metrics": {"rmse": 2.0, "mae": 1.5, "r2": 0.1}},
            },
        }
        assert _resolve_primary_metrics(metrics) is primary

    @pytest.mark.parametrize(
        "metrics",
        [
            {},
            {"primary_split": "within_entity_temporal"},
            {"splits": {"within_entity_temporal": {"a": 1}}},
            {
                "primary_split": "missing",
                "splits": {"within_entity_temporal": {"point_metrics": {"rmse": 1}}},
            },
            {"primary_split": 123, "splits": {"within_entity_temporal": {"a": 1}}},
            {"primary_split": "x", "splits": []},
            {
                "primary_split": "within_entity_temporal",
                "splits": {"within_entity_temporal": []},
            },
        ],
    )
    def test_resolve_primary_metrics_falls_back(self, metrics):
        assert _resolve_primary_metrics(metrics) == metrics


class TestCoverageParsing:
    def test_parse_coverage_current_schema(self):
        result = _parse_coverage_results(
            {
                "calibration": {
                    "coverages": {
                        "0.8": {"nominal": 0.8, "empirical": 0.79, "interval_width": 12.5},
                        "0.95": {"nominal": 0.95, "empirical": 0.93},
                    }
                }
            }
        )
        assert result is not None
        assert result[0.8].empirical == pytest.approx(0.79)
        assert result[0.8].interval_width == pytest.approx(12.5)
        assert result[0.95].empirical == pytest.approx(0.93)
        assert result[0.95].interval_width is None

    def test_parse_coverage_legacy_schema(self):
        result = _parse_coverage_results(
            {"calibration": {"coverage_80": 0.77, "coverage_95": 0.91}}
        )
        assert result is not None
        assert result[0.8].empirical == pytest.approx(0.77)
        assert result[0.95].empirical == pytest.approx(0.91)

    def test_parse_coverage_prefers_current_over_legacy_duplicates(self):
        result = _parse_coverage_results(
            {
                "calibration": {
                    "coverages": {"0.8": {"nominal": 0.8, "empirical": 0.79}},
                    "coverage_80": 0.75,
                }
            }
        )
        assert result is not None
        assert result[0.8].empirical == pytest.approx(0.79)

    @pytest.mark.parametrize(
        "calibration",
        [
            None,
            42,
            "bad",
            [],
            {"coverages": []},
            {"coverages": {"0.8": None}},
            {"coverages": {"0.8": {"nominal": 1.2, "empirical": 0.9}}},
            {"coverages": {"0.8": {"nominal": 0.8}}},
            {"coverages": {"0.8": {"nominal": "bad", "empirical": 0.9}}},
            {"coverage_a": 0.9},
            {"coverage_200": 0.9},
            {"coverage_80": "bad"},
        ],
    )
    def test_parse_coverage_invalid_payloads_return_none(self, calibration):
        result = _parse_coverage_results({"calibration": calibration})
        assert result is None


class TestPointMetricParsing:
    def test_parse_point_metrics_prefers_point_metrics_dict(self):
        result = _parse_point_metrics(
            {
                "point_metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.3},
                "mae": 9.0,
                "rmse": 9.0,
                "r2": -1.0,
            }
        )
        assert result is not None
        assert result.mae == pytest.approx(1.0)
        assert result.rmse == pytest.approx(2.0)
        assert result.r2 == pytest.approx(0.3)

    def test_parse_point_metrics_falls_back_to_root(self):
        result = _parse_point_metrics({"mae": 0.8, "rmse": 1.2, "r2": 0.5})
        assert result is not None
        assert result.mae == pytest.approx(0.8)
        assert result.rmse == pytest.approx(1.2)
        assert result.r2 == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"point_metrics": None},
            {"point_metrics": {"mae": 1.0, "rmse": 2.0}},
            {"point_metrics": {"mae": 1.0, "r2": 0.5}},
            {"point_metrics": {"rmse": 2.0, "r2": 0.5}},
            {"point_metrics": {"mae": "bad", "rmse": 2.0, "r2": 0.5}},
            {"mae": 1.0, "rmse": "bad", "r2": 0.5},
            {"mae": 1.0, "rmse": 2.0, "r2": None},
        ],
    )
    def test_parse_point_metrics_invalid_payloads(self, payload):
        assert _parse_point_metrics(payload) is None


class TestLooParsing:
    def test_parse_heldout_elpd_valid(self):
        result = _parse_loo_result(
            {"info_criteria": {"heldout_elpd": {"elpd": -123.4, "se": 5.6, "n_obs": 100}}}
        )
        assert result is not None
        assert result.elpd_loo == pytest.approx(-123.4)
        assert result.se_elpd == pytest.approx(5.6)

    def test_parse_legacy_loo_fallback(self):
        # Pre-#63 metrics.json payloads still parse.
        result = _parse_loo_result({"info_criteria": {"loo": {"elpd": -123.4, "se": 5.6}}})
        assert result is not None
        assert result.elpd_loo == pytest.approx(-123.4)
        assert result.se_elpd == pytest.approx(5.6)

    def test_heldout_elpd_preferred_over_legacy_loo(self):
        result = _parse_loo_result(
            {
                "info_criteria": {
                    "heldout_elpd": {"elpd": -100.0, "se": 4.0},
                    "loo": {"elpd": -200.0, "se": 8.0},
                }
            }
        )
        assert result is not None
        assert result.elpd_loo == pytest.approx(-100.0)

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"info_criteria": None},
            {"info_criteria": {"heldout_elpd": None}},
            {"info_criteria": {"heldout_elpd": {"elpd": -123.4}}},
            {"info_criteria": {"heldout_elpd": {"se": 5.6}}},
            {"info_criteria": {"heldout_elpd": {"elpd": "bad", "se": 5.6}}},
            {"info_criteria": {"heldout_elpd": {"elpd": -123.4, "se": "bad"}}},
            {"info_criteria": {"loo": {"elpd": -123.4}}},
        ],
    )
    def test_parse_loo_result_invalid_payload(self, payload):
        assert _parse_loo_result(payload) is None


class TestConvergenceParsing:
    def test_parse_convergence_uses_diagnostics_first(self):
        diagnostics = {"passed": True, "rhat_max": 1.002, "ess_bulk_min": 900}
        metrics = {"diagnostics": {"passed": False, "rhat_max": 2.0, "ess_bulk_min": 1}}
        parsed = _parse_convergence(diagnostics, metrics)
        assert parsed is not None
        assert parsed.passed is True
        assert parsed.rhat_max == pytest.approx(1.002)
        assert parsed.ess_bulk_min == pytest.approx(900.0)

    def test_parse_convergence_falls_back_to_metrics(self):
        parsed = _parse_convergence(
            {},
            {"diagnostics": {"passed": False, "rhat_max": 1.03, "ess_bulk_min": 150}},
        )
        assert parsed is not None
        assert parsed.passed is False
        assert parsed.rhat_max == pytest.approx(1.03)
        assert parsed.ess_bulk_min == pytest.approx(150.0)

    @pytest.mark.parametrize(
        ("payload", "expected_none"),
        [
            ({}, True),
            ({"divergences": 0}, True),
            ({"passed": False}, False),
            ({"rhat_max": 1.05}, False),
            ({"ess_bulk_min": 500}, False),
            ({"ess_tail_min": 450}, False),
            ({"divergences": 2}, False),
            ({"rhat_max": "bad", "ess_bulk_min": "bad"}, True),
            ({"rhat_max": None, "ess_bulk_min": None, "passed": False}, False),
        ],
    )
    def test_parse_convergence_presence_rules(self, payload, expected_none):
        parsed = _parse_convergence(payload, {})
        if expected_none:
            assert parsed is None
        else:
            assert parsed is not None

    @pytest.mark.parametrize(
        ("divergences_raw", "expected"),
        [
            (0, 0),
            (5, 5),
            ("7", 7),
            ("bad", 0),
            (None, 0),
        ],
    )
    def test_parse_convergence_divergence_casting(self, divergences_raw, expected):
        parsed = _parse_convergence(
            {"passed": True, "rhat_max": 1.0, "ess_bulk_min": 1000, "divergences": divergences_raw},
            {},
        )
        assert parsed is not None
        assert parsed.divergences == expected

    @pytest.mark.parametrize(
        ("ess_tail_min", "expected_tail"),
        [
            (1200, 1200.0),
            ("1100", 1100.0),
            (None, None),
            ("bad", None),
        ],
    )
    def test_parse_convergence_ess_tail_absent_stays_none(self, ess_tail_min, expected_tail):
        # Regression: an absent ess_tail_min must NOT be silently substituted
        # with ess_bulk_min (a fabricated diagnostic under the tail label).
        parsed = _parse_convergence(
            {
                "passed": True,
                "rhat_max": 1.001,
                "ess_bulk_min": 1000,
                "ess_tail_min": ess_tail_min,
            },
            {},
        )
        assert parsed is not None
        if expected_tail is None:
            assert parsed.ess_tail_min is None
        else:
            assert parsed.ess_tail_min == pytest.approx(expected_tail)

    def test_parse_convergence_ess_tail_key_missing_entirely(self):
        # Current evaluate.py payload shape (pre ess_tail_min serialization).
        parsed = _parse_convergence(
            {"passed": True, "rhat_max": 1.001, "ess_bulk_min": 1000},
            {},
        )
        assert parsed is not None
        assert parsed.ess_tail_min is None
        assert parsed.failing_params == []


class TestPublicationReadiness:
    def test_ready_when_all_critical_checks_pass(self):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": {"within_tolerance": True}},
                    "entity_disjoint": {"calibration": {"within_tolerance": True}},
                },
            },
            diagnostics={"passed": True, "rhat_max": 1.003, "ess_bulk_min": 1800},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=True,
        )
        assert payload["ready"] is True
        assert payload["critical_failed"] == []

    def test_ess_threshold_is_total_not_per_chain(self):
        # Regression: the gate multiplied the total bulk-ESS floor by num_chains,
        # so a healthy 623 (>= the 400 floor) failed with 4 chains (623 < 1600).
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": {"within_tolerance": True}},
                    "entity_disjoint": {"calibration": {"within_tolerance": True}},
                },
            },
            diagnostics={
                "passed": True,
                "rhat_max": 1.003,
                "ess_bulk_min": 623,
                "ess_threshold": 400,
            },
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=True,
        )
        checks = _check_map(payload)
        assert checks["ess_within_threshold"]["passed"] is True
        assert payload["ready"] is True

    def test_not_ready_for_single_chain_and_missing_secondary(self):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": {"within_tolerance": False}},
                },
            },
            diagnostics={"passed": False, "rhat_max": None, "ess_bulk_min": 1},
            training_summary={"mcmc_config": {"num_chains": 1}},
            artifact_errors=[{"artifact": "metrics_table", "error": "boom"}],
            require_secondary_split=True,
        )
        assert payload["ready"] is False
        assert "mcmc_min_2_chains" in payload["critical_failed"]
        assert "primary_calibration_within_tolerance" in payload["critical_failed"]
        assert "secondary_split_evaluated" in payload["critical_failed"]
        assert "publication_artifact_errors" in payload["critical_failed"]

    def test_markdown_render_includes_table(self):
        payload = {
            "ready": False,
            "critical_failed": ["example_critical"],
            "recommended_failed": ["example_recommended"],
            "checks": [
                {
                    "name": "example_critical",
                    "severity": "critical",
                    "passed": False,
                    "detail": "failed",
                },
                {
                    "name": "example_recommended",
                    "severity": "recommended",
                    "passed": False,
                    "detail": "recommended fail",
                },
            ],
        }
        md = _render_publication_readiness_markdown(payload)
        assert "# Publication Readiness" in md
        assert "| Check | Severity | Passed | Detail |" in md
        assert "example_critical" in md

    @pytest.mark.parametrize(
        (
            "num_chains",
            "diag_passed",
            "rhat_max",
            "ess_bulk_min",
            "primary_within",
            "secondary_state",
        ),
        list(
            product(
                [1, 2, 4],
                [False, True],
                [None, 1.005, 1.02],
                [None, 700, 2000],
                [False, True],
                ["missing", "fail", "pass"],
            )
        ),
    )
    def test_readiness_matrix_invariants(
        self,
        num_chains,
        diag_passed,
        rhat_max,
        ess_bulk_min,
        primary_within,
        secondary_state,
    ):
        splits = {
            "within_entity_temporal": {"calibration": {"within_tolerance": primary_within}},
        }
        if secondary_state == "pass":
            splits[SECONDARY_SPLIT] = {"calibration": {"within_tolerance": True}}
        elif secondary_state == "fail":
            splits[SECONDARY_SPLIT] = {"calibration": {"within_tolerance": False}}

        payload = _build_publication_readiness(
            metrics={"primary_split": "within_entity_temporal", "splits": splits},
            diagnostics={
                "passed": diag_passed,
                "rhat_max": rhat_max,
                "ess_bulk_min": ess_bulk_min,
            },
            training_summary={"mcmc_config": {"num_chains": num_chains}},
            artifact_errors=[],
            require_secondary_split=True,
        )
        checks = _check_map(payload)
        critical_failed = set(payload["critical_failed"])

        assert checks["mcmc_min_2_chains"]["passed"] is (num_chains >= 2)
        assert checks["mcmc_recommended_4_chains"]["passed"] is (num_chains >= 4)
        assert checks["convergence_passed"]["passed"] is (diag_passed is True)
        assert checks["primary_calibration_within_tolerance"]["passed"] is primary_within
        assert checks["publication_artifact_errors"]["passed"] is True

        if rhat_max is None:
            assert checks["rhat_available"]["passed"] is False
            assert "rhat_within_threshold" not in checks
            assert "rhat_available" in critical_failed
        else:
            assert checks["rhat_available"]["passed"] is True
            assert checks["rhat_within_threshold"]["passed"] is (rhat_max < 1.01)
            if rhat_max >= 1.01:
                assert "rhat_within_threshold" in critical_failed

        if ess_bulk_min is None:
            assert checks["ess_available"]["passed"] is False
            assert "ess_within_threshold" not in checks
            assert "ess_available" in critical_failed
        else:
            assert checks["ess_available"]["passed"] is True
            assert checks["ess_within_threshold"]["passed"] is (ess_bulk_min >= 400)
            if ess_bulk_min < 400:
                assert "ess_within_threshold" in critical_failed

        if secondary_state == "missing":
            assert checks["secondary_split_evaluated"]["passed"] is False
            assert checks["secondary_calibration_within_tolerance"]["passed"] is False
            assert "secondary_split_evaluated" in critical_failed
            assert "secondary_calibration_within_tolerance" in critical_failed
        elif secondary_state == "fail":
            assert checks["secondary_split_evaluated"]["passed"] is True
            assert checks["secondary_calibration_within_tolerance"]["passed"] is False
            assert "secondary_calibration_within_tolerance" in critical_failed
        else:
            assert checks["secondary_split_evaluated"]["passed"] is True
            assert checks["secondary_calibration_within_tolerance"]["passed"] is True

        assert payload["ready"] is (len(payload["critical_failed"]) == 0)

    @pytest.mark.parametrize(
        ("secondary_state", "expected_recommended_failed"),
        [
            ("missing", True),
            ("present", False),
        ],
    )
    def test_readiness_with_secondary_disabled_uses_recommended_check(
        self, secondary_state, expected_recommended_failed
    ):
        splits = {
            "within_entity_temporal": {"calibration": {"within_tolerance": True}},
        }
        if secondary_state == "present":
            splits[SECONDARY_SPLIT] = {"calibration": {"within_tolerance": False}}

        payload = _build_publication_readiness(
            metrics={"primary_split": "within_entity_temporal", "splits": splits},
            diagnostics={"passed": True, "rhat_max": 1.001, "ess_bulk_min": 4000},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=False,
        )
        checks = _check_map(payload)
        secondary_check = checks["secondary_split_evaluated"]

        assert secondary_check["severity"] == "recommended"
        assert secondary_check["passed"] is (secondary_state == "present")
        assert ("secondary_split_evaluated" in payload["recommended_failed"]) is (
            expected_recommended_failed
        )
        assert "secondary_split_evaluated" not in payload["critical_failed"]
        assert payload["ready"] is True

    @pytest.mark.parametrize("available", [True, False])
    def test_prior_predictive_artifact_check(self, available):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": {"within_tolerance": True}},
                    SECONDARY_SPLIT: {"calibration": {"within_tolerance": True}},
                },
            },
            diagnostics={"passed": True, "rhat_max": 1.003, "ess_bulk_min": 1800},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=True,
            prior_predictive_available=available,
        )
        checks = _check_map(payload)
        check = checks["prior_predictive_artifact_present"]
        assert check["severity"] == "recommended"
        assert check["passed"] is available
        assert ("prior_predictive_artifact_present" in payload["recommended_failed"]) is (
            not available
        )
        # Recommended severity: absence never blocks readiness.
        assert payload["ready"] is True

    def test_readiness_flags_artifact_errors(self):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": {"within_tolerance": True}},
                    SECONDARY_SPLIT: {"calibration": {"within_tolerance": True}},
                },
            },
            diagnostics={"passed": True, "rhat_max": 1.0, "ess_bulk_min": 5000},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[
                {"artifact": "trace_plot", "error": "failed to render"},
                {"artifact": "model_card", "error": "invalid key"},
            ],
            require_secondary_split=True,
        )
        checks = _check_map(payload)
        assert checks["publication_artifact_errors"]["passed"] is False
        assert "publication_artifact_errors" in payload["critical_failed"]
        assert payload["ready"] is False


class TestReadinessMarkdown:
    def test_markdown_handles_empty_payload(self):
        md = _render_publication_readiness_markdown({})
        assert "- **Status:** FAIL" in md
        assert "- **Critical failures:** 0" in md
        assert "- **Recommended failures:** 0" in md

    def test_markdown_handles_non_bool_passed_values(self):
        payload = {
            "ready": True,
            "checks": [
                {"name": "a", "severity": "critical", "passed": 1, "detail": "ok"},
                {"name": "b", "severity": "critical", "passed": 0, "detail": "bad"},
            ],
            "critical_failed": ["b"],
            "recommended_failed": [],
        }
        md = _render_publication_readiness_markdown(payload)
        assert "| a | critical | yes | ok |" in md
        assert "| b | critical | no | bad |" in md

    @pytest.mark.parametrize(
        ("detail", "expected_fragment"),
        [
            ("line1\nline2", "line1 line2"),
            ("value | with pipe", "value \\| with pipe"),
        ],
    )
    def test_markdown_sanitizes_cell_content(self, detail, expected_fragment):
        payload = {
            "ready": False,
            "checks": [
                {
                    "name": "check",
                    "severity": "critical",
                    "passed": False,
                    "detail": detail,
                }
            ],
            "critical_failed": ["check"],
            "recommended_failed": [],
        }
        md = _render_publication_readiness_markdown(payload)
        assert expected_fragment in md


class TestHelperInterop:
    def test_parse_helpers_round_trip_into_model_card_like_shape(self):
        primary_metrics = _resolve_primary_metrics(
            {
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {
                        "calibration": {
                            "coverages": {"0.95": {"nominal": 0.95, "empirical": 0.94}}
                        },
                        "point_metrics": {"mae": 0.8, "rmse": 1.1, "r2": 0.4},
                        "info_criteria": {"heldout_elpd": {"elpd": -42.5, "se": 1.2}},
                    }
                },
            }
        )
        coverage = _parse_coverage_results(primary_metrics)
        point = _parse_point_metrics(primary_metrics)
        loo = _parse_loo_result(primary_metrics)
        convergence = _parse_convergence(
            {"passed": True, "rhat_max": 1.003, "ess_bulk_min": 2000},
            {},
        )

        assert coverage is not None and 0.95 in coverage
        assert point is not None
        assert loo is not None
        assert convergence is not None
        assert point.rmse == pytest.approx(1.1)
        assert loo.elpd_loo == pytest.approx(-42.5)
        assert convergence.rhat_max == pytest.approx(1.003)


# --- from unit/pipelines/test_publication_coverage.py ---


def _write_json(path: Path, data: dict) -> None:
    """Write a JSON file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_metrics(
    *,
    primary_within_tolerance: bool = True,
    secondary_within_tolerance: bool | None = True,
    include_ppc: bool = False,
    include_wis: bool = False,
) -> dict:
    """Build a metrics payload."""
    primary: dict[str, Any] = {
        "point_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
        "calibration": {
            "within_tolerance": primary_within_tolerance,
            "coverages": {
                "0.80": {"nominal": 0.80, "empirical": 0.78, "interval_width": 12.0},
                "0.95": {"nominal": 0.95, "empirical": 0.93},
            },
        },
    }
    if include_wis:
        primary["calibration"]["wis"] = 4.5
    if include_ppc:
        primary["ppc"] = {
            "summary": {
                "mean": {"observed": 75.0, "p_value": 0.45, "mc_se": 0.02},
                "sd": {"observed": 8.0, "p_value": 0.52, "mc_se": 0.03},
            },
            "n_obs": 100,
            "n_samples": 200,
        }
    splits = {"within_entity_temporal": primary}
    if secondary_within_tolerance is not None:
        splits["entity_disjoint"] = {
            "calibration": {"within_tolerance": secondary_within_tolerance}
        }
    return {
        "primary_split": "within_entity_temporal",
        "splits": splits,
    }


def _make_diagnostics(*, passed: bool = True, rhat_max: float = 1.003) -> dict:
    """Build a diagnostics payload."""
    return {
        "passed": passed,
        "rhat_max": rhat_max,
        "ess_bulk_min": 2000,
        "ess_tail_min": 1800,
        "divergences": 0,
    }


def _make_training_summary(*, num_chains: int = 4) -> dict:
    """Build a training summary payload."""
    return {
        "n_observations": 500,
        "n_features": 10,
        "n_artists": 100,
        "max_albums": 50,
        "mcmc_config": {
            "num_chains": num_chains,
            "num_warmup": 500,
            "num_samples": 750,
            "chain_method": "sequential",
            "target_accept_prob": 0.9,
            "max_tree_depth": 10,
        },
    }


def _fake_export_table(df: pd.DataFrame, base_path: str, caption: str) -> None:
    """Write stub CSV and TeX files."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text("% fake\n", encoding="utf-8")


def _fake_plot(*_args, output_dir: Path = None, filename_base: str = "", **_kw):
    """Create stub PDF and PNG files."""
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{filename_base}.pdf"
    png = output_dir / f"{filename_base}.png"
    pdf.write_text("pdf", encoding="utf-8")
    png.write_text("png", encoding="utf-8")
    return pdf, png


def _fake_write_model_card(_data, path: Path) -> None:
    """Write a stub model card."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Model Card\n", encoding="utf-8")


def _make_fake_idata():
    """Create a mock InferenceData with minimal posterior keys."""
    idata = MagicMock()
    idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }
    return idata


def _setup_ctx(*, strict: bool = False, run_dir=None, evaluate_secondary_split=True):
    """Create a ctx SimpleNamespace for publication tests."""
    return SimpleNamespace(
        run_dir=run_dir,
        strict=strict,
        evaluate_secondary_split=evaluate_secondary_split,
    )


def _base_patches(tmp_path, idata=None, **overrides):
    """Build a dictionary of standard patches for generate_publication_artifacts.

    Returns a dict of (target, mock) pairs suitable for use in contextmanager stacking.
    Callers can override individual patches via keyword arguments.
    """
    if idata is None:
        idata = _make_fake_idata()
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})

    defaults = {
        "load_manifest": patch(
            "panelcast.pipelines.publication.load_manifest",
            return_value=fake_manifest,
        ),
        "load_model": patch(
            "panelcast.pipelines.publication.load_model",
            return_value=idata,
        ),
        "create_coefficient_table": patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        "create_diagnostics_table": patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        "export_table": patch(
            "panelcast.pipelines.publication.export_table",
            side_effect=_fake_export_table,
        ),
        "save_trace_plot": patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *a, **kw: _fake_plot(
                output_dir=kw["output_dir"], filename_base=kw["filename_base"]
            ),
        ),
        "save_posterior_plot": patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *a, **kw: _fake_plot(
                output_dir=kw["output_dir"], filename_base=kw["filename_base"]
            ),
        ),
        "create_default_model_card_data": patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        "update_model_card_with_results": patch(
            "panelcast.pipelines.publication.update_model_card_with_results",
            side_effect=lambda data, **kw: data,
        ),
        "write_model_card": patch(
            "panelcast.pipelines.publication.write_model_card",
            side_effect=_fake_write_model_card,
        ),
        "Path": patch(
            "panelcast.pipelines.publication.Path",
            side_effect=lambda p: tmp_path / p,
        ),
    }
    defaults.update(overrides)
    return defaults


def _run_with_patches(tmp_path, ctx, patches_dict):
    """Enter all patch context managers and run generate_publication_artifacts."""
    managers = {k: v.__enter__() for k, v in patches_dict.items()}
    try:
        return generate_publication_artifacts(ctx)
    finally:
        for v in patches_dict.values():
            v.__exit__(None, None, None)


class TestPublicationMissingModel:
    def test_raises_when_no_manifest(self, tmp_path):
        """Should raise ValueError when model manifest is None."""
        ctx = _setup_ctx()
        with (
            patch("panelcast.pipelines.publication.load_manifest", return_value=None),
            patch(
                "panelcast.pipelines.publication.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ValueError, match="No trained user_score model"):
                generate_publication_artifacts(ctx)

    def test_raises_when_user_score_missing_from_manifest(self, tmp_path):
        """Should raise ValueError when manifest has no user_score entry."""
        ctx = _setup_ctx()
        fake_manifest = SimpleNamespace(current={"critic_score": "model.nc"})
        with (
            patch(
                "panelcast.pipelines.publication.load_manifest",
                return_value=fake_manifest,
            ),
            patch(
                "panelcast.pipelines.publication.Path",
                side_effect=lambda p: tmp_path / p,
            ),
        ):
            with pytest.raises(ValueError, match="No trained user_score model"):
                generate_publication_artifacts(ctx)


class TestPublicationMissingInputFiles:
    def test_missing_metrics_records_error(self, tmp_path):
        """Missing metrics.json should record an error but not crash in non-strict."""
        # No metrics.json written
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "metrics_input" for e in artifacts["errors"])

    def test_missing_diagnostics_records_error(self, tmp_path):
        """Missing diagnostics.json should record an error but not crash in non-strict."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        # No diagnostics.json
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "diagnostics_input" for e in artifacts["errors"])

    def test_missing_training_summary_records_error(self, tmp_path):
        """Missing training_summary.json should record an error but not crash."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        # No training_summary.json
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "training_summary_input" for e in artifacts["errors"])

    def test_corrupt_metrics_json_records_error(self, tmp_path):
        """Corrupt (invalid JSON) metrics file should record error."""
        metrics_path = tmp_path / "outputs/evaluation/metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("{bad json!", encoding="utf-8")
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "metrics_input" for e in artifacts["errors"])


class TestPublicationTableFailures:
    def test_diagnostics_table_failure_recorded(self, tmp_path):
        """Diagnostics table generation failure should be recorded, not fatal."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            create_diagnostics_table=patch(
                "panelcast.pipelines.publication.create_diagnostics_table",
                side_effect=RuntimeError("diag table boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "diagnostics_table" for e in artifacts["errors"])

    def test_metrics_summary_table_failure_recorded(self, tmp_path):
        """Metrics summary table failure should be recorded."""
        # Use metrics with WIS and point metrics that will trigger the table code
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(include_wis=True),
        )
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        # Make export_table raise only for metrics_summary (third call)
        call_count = {"n": 0}
        original_export = _fake_export_table

        def _counting_export(df, base_path, caption):
            call_count["n"] += 1
            if "metrics" in caption.lower():
                raise RuntimeError("metrics export boom")
            return original_export(df, base_path, caption)

        patches = _base_patches(
            tmp_path,
            export_table=patch(
                "panelcast.pipelines.publication.export_table",
                side_effect=_counting_export,
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "metrics_summary_table" for e in artifacts["errors"])


class TestDiagnosticsTableThresholds:
    """The exported diagnostics table must use the run's gate thresholds."""

    def _run_and_capture(self, tmp_path, ctx, diagnostics):
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", diagnostics)
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        captured = {}

        def _capture_diag_table(idata, rhat_threshold=None, ess_threshold=None):
            captured["rhat_threshold"] = rhat_threshold
            captured["ess_threshold"] = ess_threshold
            return pd.DataFrame({"x": [1]})

        patches = _base_patches(
            tmp_path,
            create_diagnostics_table=patch(
                "panelcast.pipelines.publication.create_diagnostics_table",
                side_effect=_capture_diag_table,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)
        return captured

    def test_thresholds_from_diagnostics_json(self, tmp_path):
        diagnostics = _make_diagnostics()
        diagnostics["rhat_threshold"] = 1.02
        diagnostics["ess_threshold"] = 150
        captured = self._run_and_capture(tmp_path, _setup_ctx(strict=False), diagnostics)
        assert captured["rhat_threshold"] == pytest.approx(1.02)
        assert captured["ess_threshold"] == 150

    def test_thresholds_fall_back_to_ctx_then_defaults(self, tmp_path):
        ctx = _setup_ctx(strict=False)
        ctx.rhat_threshold = 1.05
        ctx.ess_threshold = 200
        captured = self._run_and_capture(tmp_path, ctx, _make_diagnostics())
        assert captured["rhat_threshold"] == pytest.approx(1.05)
        assert captured["ess_threshold"] == 200

        captured = self._run_and_capture(tmp_path, _setup_ctx(strict=False), _make_diagnostics())
        assert captured["rhat_threshold"] == pytest.approx(1.01)
        assert captured["ess_threshold"] == 400


class TestPublicationFigureFailures:
    def test_trace_plot_failure_recorded(self, tmp_path):
        """Trace plot failure should be recorded, not fatal."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_trace_plot=patch(
                "panelcast.pipelines.publication.save_trace_plot",
                side_effect=RuntimeError("trace boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "trace_plot" for e in artifacts["errors"])

    def test_posterior_plot_failure_recorded(self, tmp_path):
        """Posterior plot failure should be recorded, not fatal."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_posterior_plot=patch(
                "panelcast.pipelines.publication.save_posterior_plot",
                side_effect=RuntimeError("posterior boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "posterior_plot" for e in artifacts["errors"])


class TestPublicationModelCardFailure:
    def test_model_card_failure_recorded(self, tmp_path):
        """Model card write failure should be recorded."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            write_model_card=patch(
                "panelcast.pipelines.publication.write_model_card",
                side_effect=RuntimeError("model card boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "model_card" for e in artifacts["errors"])


class TestPublicationStrictMode:
    def test_strict_raises_on_artifact_errors(self, tmp_path):
        """Strict mode should raise ValueError when artifacts have errors."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=True)
        patches = _base_patches(
            tmp_path,
            create_coefficient_table=patch(
                "panelcast.pipelines.publication.create_coefficient_table",
                side_effect=RuntimeError("coef boom"),
            ),
        )
        with pytest.raises(ValueError, match="Publication artifact generation failed"):
            _run_with_patches(tmp_path, ctx, patches)

    def test_strict_raises_on_readiness_failure(self, tmp_path):
        """Strict mode should raise ValueError when publication readiness fails."""
        # diagnostics.passed=False triggers readiness failure
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(primary_within_tolerance=False),
        )
        _write_json(
            tmp_path / "outputs/evaluation/diagnostics.json",
            _make_diagnostics(passed=False, rhat_max=1.05),
        )
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(num_chains=1),
        )
        ctx = _setup_ctx(strict=True)
        patches = _base_patches(tmp_path)
        with pytest.raises(ValueError, match="readiness checks failed"):
            _run_with_patches(tmp_path, ctx, patches)

    def test_non_strict_does_not_raise_on_readiness_failure(self, tmp_path):
        """Non-strict mode should not raise on readiness failure."""
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(primary_within_tolerance=False),
        )
        _write_json(
            tmp_path / "outputs/evaluation/diagnostics.json",
            _make_diagnostics(passed=False, rhat_max=1.05),
        )
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(num_chains=1),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # Should still produce readiness file
        readiness_path = tmp_path / "reports/publication_readiness.json"
        assert readiness_path.exists()
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        assert readiness["ready"] is False


class TestPublicationPredictionTable:
    def test_prediction_table_generated_when_csv_exists(self, tmp_path):
        """Prediction scenario table should be generated when CSV exists."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        # Create the prediction CSV
        known_csv = tmp_path / "outputs/predictions/next_event_known_entities.csv"
        known_csv.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame(
            {
                "scenario": ["optimistic", "optimistic", "pessimistic", "pessimistic"],
                "pred_mean": [80.0, 82.0, 60.0, 62.0],
                "entity": ["A", "B", "A", "B"],
            }
        )
        pred_df.to_csv(known_csv, index=False)

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # Should include prediction scenario table
        pred_table_paths = [p for p in artifacts["tables"] if "prediction_scenarios" in p]
        assert len(pred_table_paths) > 0

    def test_prediction_table_skipped_when_no_csv(self, tmp_path):
        """No prediction table should be generated when CSV does not exist."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        pred_table_paths = [p for p in artifacts["tables"] if "prediction_scenarios" in p]
        assert len(pred_table_paths) == 0

    def test_prediction_table_failure_recorded(self, tmp_path):
        """Prediction table failure should be recorded as error."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        # Create a malformed CSV
        known_csv = tmp_path / "outputs/predictions/next_event_known_entities.csv"
        known_csv.parent.mkdir(parents=True, exist_ok=True)
        # Missing required columns for groupby
        pd.DataFrame({"bad_column": [1, 2]}).to_csv(known_csv, index=False)

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "prediction_scenarios_table" for e in artifacts["errors"])


class TestPublicationPPCDensityPlot:
    """The PPC density plot is intentionally not generated -- evaluate.py persists
    only the PPC summary (p-values), never the replicated draws. These guard the
    no-plot contract: a PPC summary in the metrics runs cleanly and still feeds the
    model card, but produces no ppc_density figure and no ppc_density_plot error."""

    def test_ppc_summary_present_produces_no_density_plot(self, tmp_path):
        metrics = _make_metrics()
        metrics["splits"]["within_entity_temporal"]["ppc"] = {
            "summary": {
                "mean": {"observed": 75.0, "p_value": 0.45, "mc_se": 0.03},
                "sd": {"observed": 8.0, "p_value": 0.52, "mc_se": 0.03},
            },
            "n_obs": 100,
            "n_samples": 200,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert not any(e["artifact"] == "ppc_density_plot" for e in artifacts["errors"])
        assert not any("ppc_density" in Path(p).name for p in artifacts["figures"])

    def test_ppc_summary_missing_fields_does_not_crash(self, tmp_path):
        metrics = _make_metrics()
        metrics["splits"]["within_entity_temporal"]["ppc"] = {
            "summary": {"mean": {"p_value": 0.45}},  # missing observed / mc_se
            "n_obs": 100,
            "n_samples": 0,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert not any(e["artifact"] == "ppc_density_plot" for e in artifacts["errors"])
        assert not any("ppc_density" in Path(p).name for p in artifacts["figures"])


class TestPublicationRunDirCopy:
    def test_artifacts_copied_to_run_dir(self, tmp_path):
        """Artifacts should be copied to run directory when it exists."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )

        run_dir = tmp_path / "outputs" / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = _setup_ctx(strict=False, run_dir=run_dir)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)

        # Run dir should have reports subdirectory with copied artifacts
        run_reports = run_dir / "reports"
        assert run_reports.exists()
        # Status files should be copied
        assert (run_reports / "artifact_status.json").exists()
        assert (run_reports / "publication_readiness.json").exists()
        assert (run_reports / "PUBLICATION_READINESS.md").exists()

    def test_no_copy_when_run_dir_none(self, tmp_path):
        """Should not attempt copies when run_dir is None."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False, run_dir=None)
        patches = _base_patches(tmp_path)
        # Should not raise
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert isinstance(artifacts, dict)


class TestPublicationOutputFiles:
    def test_readiness_json_and_md_written(self, tmp_path):
        """Publication readiness JSON and Markdown files should be written."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)

        readiness_json = tmp_path / "reports/publication_readiness.json"
        readiness_md = tmp_path / "reports/PUBLICATION_READINESS.md"
        assert readiness_json.exists()
        assert readiness_md.exists()
        payload = json.loads(readiness_json.read_text(encoding="utf-8"))
        assert "ready" in payload
        assert "checks" in payload

        md_text = readiness_md.read_text(encoding="utf-8")
        assert "# Publication Readiness" in md_text

    @pytest.mark.parametrize("write_pp", [True, False])
    def test_readiness_reports_prior_predictive_presence(self, tmp_path, write_pp):
        """The readiness payload reflects whether evaluate wrote prior_predictive.json."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())
        if write_pp:
            _write_json(
                tmp_path / "outputs/evaluation/prior_predictive.json",
                {"reasonable": True, "fraction_in_bounds": 0.99},
            )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        _run_with_patches(tmp_path, ctx, patches)

        readiness = json.loads(
            (tmp_path / "reports/publication_readiness.json").read_text(encoding="utf-8")
        )
        checks = {c["name"]: c for c in readiness["checks"]}
        assert checks["prior_predictive_artifact_present"]["passed"] is write_pp

    def test_artifact_status_json_written(self, tmp_path):
        """artifact_status.json should always be written with summary counts."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)

        status_path = tmp_path / "reports/artifact_status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert "n_tables" in status
        assert "n_figures" in status
        assert "n_docs" in status
        assert "n_errors" in status
        assert "publication_ready" in status

    def test_metrics_table_includes_wis(self, tmp_path):
        """Metrics table should include WIS row when present."""
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(include_wis=True),
        )
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)

        # Capture what export_table receives
        captured_dfs = []
        original_export = _fake_export_table

        def _capturing_export(df, base_path, caption):
            captured_dfs.append((caption, df))
            return original_export(df, base_path, caption)

        patches = _base_patches(
            tmp_path,
            export_table=patch(
                "panelcast.pipelines.publication.export_table",
                side_effect=_capturing_export,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        # Find the metrics summary table
        metrics_tables = [(cap, df) for cap, df in captured_dfs if "performance" in cap.lower()]
        assert len(metrics_tables) == 1
        metrics_df = metrics_tables[0][1]
        assert "WIS" in metrics_df["Metric"].values

    def test_metrics_table_with_missing_point_metrics(self, tmp_path):
        """Metrics table should skip RMSE/MAE/R2 when point metrics unavailable."""
        metrics = _make_metrics()
        # Remove point_metrics
        del metrics["splits"]["within_entity_temporal"]["point_metrics"]
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)

        captured_dfs = []
        original_export = _fake_export_table

        def _capturing_export(df, base_path, caption):
            captured_dfs.append((caption, df))
            return original_export(df, base_path, caption)

        patches = _base_patches(
            tmp_path,
            export_table=patch(
                "panelcast.pipelines.publication.export_table",
                side_effect=_capturing_export,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        metrics_tables = [(cap, df) for cap, df in captured_dfs if "performance" in cap.lower()]
        assert len(metrics_tables) == 1
        metrics_df = metrics_tables[0][1]
        # Should not have RMSE/MAE/R2 rows
        if not metrics_df.empty:
            assert "RMSE" not in metrics_df["Metric"].values


class TestPublicationModelCardMetadata:
    def test_training_summary_populates_model_card_data(self, tmp_path):
        """Model card should receive training summary metadata."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(num_chains=4),
        )
        ctx = _setup_ctx(strict=False)

        captured = {}

        def _capture_update(data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return data

        patches = _base_patches(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_capture_update,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        data = captured["data"]
        assert data.dataset_size == 500
        assert data.hyperparameters["n_features"] == 10
        assert data.hyperparameters["n_artists"] == 100
        assert data.hyperparameters["max_albums"] == 50
        assert data.hyperparameters["num_chains"] == 4

    def test_model_config_rows_recorded_from_training_summary(self, tmp_path):
        """Per-run model configuration must land in the hyperparameters table."""
        training = _make_training_summary()
        training["target_transform"] = "offset_logit"
        training["likelihood_family"] = "studentt"
        training["entity_group_pooling"] = True
        training["feature_cols"] = ["prev_score", "gbm_offset"]
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", training)
        ctx = _setup_ctx(strict=False)

        captured = {}

        def _capture_update(data, **kwargs):
            captured["data"] = data
            return data

        patches = _base_patches(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_capture_update,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        hyp = captured["data"].hyperparameters
        assert hyp["target_transform"] == "offset_logit"
        assert hyp["likelihood_family"] == "studentt"
        assert hyp["entity_group_pooling"] is True
        assert hyp["gbm_offset"] is True

    def test_model_config_rows_omitted_for_legacy_summary(self, tmp_path):
        """Older summaries without the config keys must not fabricate rows."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())
        ctx = _setup_ctx(strict=False)

        captured = {}

        def _capture_update(data, **kwargs):
            captured["data"] = data
            return data

        patches = _base_patches(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_capture_update,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        hyp = captured["data"].hyperparameters
        for key in ("target_transform", "likelihood_family", "entity_group_pooling", "gbm_offset"):
            assert key not in hyp

    def test_model_card_with_prior_justification(self, tmp_path):
        """Model card should receive prior justification when priors exist in training summary."""
        training = _make_training_summary()
        training["priors"] = {
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", training)

        ctx = _setup_ctx(strict=False)

        captured = {}

        def _capture_update(data, **kwargs):
            captured["kwargs"] = kwargs
            return data

        patches = _base_patches(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_capture_update,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        # prior_justification should be passed to update_model_card_with_results
        # (it may be None if the generate function fails, but it shouldn't crash)
        assert "prior_justification" in captured["kwargs"]


class TestPublicationEvaluateSecondaryDisabled:
    def test_secondary_split_disabled_via_ctx(self, tmp_path):
        """When ctx.evaluate_secondary_split=False, secondary check should be recommended."""
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(secondary_within_tolerance=None),
        )
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False, evaluate_secondary_split=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)

        readiness_path = tmp_path / "reports/publication_readiness.json"
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        # Secondary split should not be critical
        assert "secondary_split_evaluated" not in readiness.get("critical_failed", [])


class TestBuildReadinessEdgeCases:
    def test_missing_mcmc_config_key(self):
        """Missing mcmc_config in training_summary should flag num_chains missing."""
        payload = _build_publication_readiness(
            metrics={},
            diagnostics={"passed": True, "rhat_max": 1.001},
            training_summary={},
            artifact_errors=[],
            require_secondary_split=False,
        )
        assert "mcmc_num_chains_present" in payload["critical_failed"]

    def test_non_dict_mcmc_config_handled(self):
        """Non-dict mcmc_config should be treated as empty."""
        payload = _build_publication_readiness(
            metrics={},
            diagnostics={"passed": True, "rhat_max": 1.001},
            training_summary={"mcmc_config": "bad"},
            artifact_errors=[],
            require_secondary_split=False,
        )
        assert "mcmc_num_chains_present" in payload["critical_failed"]

    def test_rhat_with_custom_threshold(self):
        """Custom rhat_threshold from diagnostics should be used."""
        payload = _build_publication_readiness(
            metrics={},
            diagnostics={
                "passed": True,
                "rhat_max": 1.005,
                "ess_bulk_min": 5000,
                "rhat_threshold": 1.02,
            },
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=False,
        )
        # rhat_max=1.005 < threshold 1.02 should pass
        checks = {c["name"]: c for c in payload["checks"]}
        assert checks["rhat_within_threshold"]["passed"] is True

    def test_ess_with_custom_threshold(self):
        """Custom ess_threshold from diagnostics should be used."""
        payload = _build_publication_readiness(
            metrics={},
            diagnostics={
                "passed": True,
                "rhat_max": 1.001,
                "ess_bulk_min": 500,
                "ess_threshold": 100,
            },
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=False,
        )
        # ess_bulk_min=500 >= 100*4=400 should pass
        checks = {c["name"]: c for c in payload["checks"]}
        assert checks["ess_within_threshold"]["passed"] is True


class TestRenderReadinessMarkdownEdgeCases:
    def test_markdown_with_none_detail(self):
        """None values in check detail should render as empty string."""
        payload = {
            "ready": True,
            "checks": [
                {
                    "name": "check",
                    "severity": "critical",
                    "passed": True,
                    "detail": None,
                }
            ],
            "critical_failed": [],
            "recommended_failed": [],
        }
        md = _render_publication_readiness_markdown(payload)
        assert "| check | critical | yes |  |" in md

    def test_markdown_with_carriage_return(self):
        """Carriage returns in detail should be replaced with spaces."""
        payload = {
            "ready": True,
            "checks": [
                {
                    "name": "check",
                    "severity": "critical",
                    "passed": True,
                    "detail": "line1\r\nline2",
                }
            ],
            "critical_failed": [],
            "recommended_failed": [],
        }
        md = _render_publication_readiness_markdown(payload)
        assert "\r" not in md
        assert "line1" in md
        assert "line2" in md


# --- from unit/pipelines/test_publication_more.py ---


def _write_json_more(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_metrics_more(
    *,
    primary_within_tolerance: bool = True,
    secondary_within_tolerance: bool | None = True,
    include_ppc: bool = False,
    include_wis: bool = False,
    ppc_override: dict | None = None,
) -> dict:
    primary: dict[str, Any] = {
        "point_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
        "calibration": {
            "within_tolerance": primary_within_tolerance,
            "coverages": {
                "0.80": {"nominal": 0.80, "empirical": 0.78, "interval_width": 12.0},
                "0.95": {"nominal": 0.95, "empirical": 0.93},
            },
        },
    }
    if include_wis:
        primary["calibration"]["wis"] = 4.5
    if include_ppc:
        primary["ppc"] = {
            "summary": {
                "mean": {"observed": 75.0, "p_value": 0.45, "mc_se": 0.02},
                "sd": {"observed": 8.0, "p_value": 0.52, "mc_se": 0.03},
            },
            "n_obs": 100,
            "n_samples": 200,
        }
    if ppc_override is not None:
        primary["ppc"] = ppc_override
    splits = {"within_entity_temporal": primary}
    if secondary_within_tolerance is not None:
        splits["entity_disjoint"] = {
            "calibration": {"within_tolerance": secondary_within_tolerance}
        }
    return {
        "primary_split": "within_entity_temporal",
        "splits": splits,
    }


def _make_diagnostics_more(*, passed: bool = True, rhat_max: float = 1.003) -> dict:
    return {
        "passed": passed,
        "rhat_max": rhat_max,
        "ess_bulk_min": 2000,
        "ess_tail_min": 1800,
        "divergences": 0,
    }


def _make_training_summary_more(*, num_chains: int = 4, priors: dict | None = None) -> dict:
    summary: dict[str, Any] = {
        "n_observations": 500,
        "n_features": 10,
        "n_artists": 100,
        "max_albums": 50,
        "mcmc_config": {
            "num_chains": num_chains,
            "num_warmup": 500,
            "num_samples": 750,
            "chain_method": "sequential",
            "target_accept_prob": 0.9,
            "max_tree_depth": 10,
        },
    }
    if priors is not None:
        summary["priors"] = priors
    return summary


def _fake_export_table_more(df: pd.DataFrame, base_path: str, caption: str) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text("% fake\n", encoding="utf-8")


def _fake_plot_more(*_args, output_dir: Path = None, filename_base: str = "", **_kw):
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{filename_base}.pdf"
    png = output_dir / f"{filename_base}.png"
    pdf.write_text("pdf", encoding="utf-8")
    png.write_text("png", encoding="utf-8")
    return pdf, png


def _fake_write_model_card_more(_data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Model Card\n", encoding="utf-8")


def _make_fake_idata_more():
    idata = MagicMock()
    idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }
    return idata


def _setup_ctx_more(*, strict: bool = False, run_dir=None, evaluate_secondary_split=True):
    return SimpleNamespace(
        run_dir=run_dir,
        strict=strict,
        evaluate_secondary_split=evaluate_secondary_split,
    )


def _base_patches_more(tmp_path, idata=None, **overrides):
    if idata is None:
        idata = _make_fake_idata_more()
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})

    defaults = {
        "load_manifest": patch(
            "panelcast.pipelines.publication.load_manifest",
            return_value=fake_manifest,
        ),
        "load_model": patch(
            "panelcast.pipelines.publication.load_model",
            return_value=idata,
        ),
        "create_coefficient_table": patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        "create_diagnostics_table": patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        "export_table": patch(
            "panelcast.pipelines.publication.export_table",
            side_effect=_fake_export_table_more,
        ),
        "save_trace_plot": patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *a, **kw: _fake_plot_more(
                output_dir=kw["output_dir"], filename_base=kw["filename_base"]
            ),
        ),
        "save_posterior_plot": patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *a, **kw: _fake_plot_more(
                output_dir=kw["output_dir"], filename_base=kw["filename_base"]
            ),
        ),
        "create_default_model_card_data": patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        "update_model_card_with_results": patch(
            "panelcast.pipelines.publication.update_model_card_with_results",
            side_effect=lambda data, **kw: data,
        ),
        "write_model_card": patch(
            "panelcast.pipelines.publication.write_model_card",
            side_effect=_fake_write_model_card_more,
        ),
        "Path": patch(
            "panelcast.pipelines.publication.Path",
            side_effect=lambda p: tmp_path / p,
        ),
    }
    defaults.update(overrides)
    return defaults


def _run_with_patches_more(tmp_path, ctx, patches_dict):
    managers = {k: v.__enter__() for k, v in patches_dict.items()}
    try:
        return generate_publication_artifacts(ctx)
    finally:
        for v in patches_dict.values():
            v.__exit__(None, None, None)


class TestParseCoverageResultsLegacySkip:
    def test_key_named_coverages_in_legacy_loop_is_skipped(self):
        # calibration dict has a "coverages" key that starts with "coverage_"
        # but the code explicitly skips it (line 135-136).
        calibration = {
            "coverages": {"0.80": {"nominal": 0.80, "empirical": 0.78}},
            "coverage_90": 0.89,
        }
        result = _parse_coverage_results({"calibration": calibration})
        assert result is not None
        # The 0.90 key should come from the legacy coverage_90 entry, not "coverages"
        prob = 90.0 / 100.0
        assert prob in result


class TestPlotPresentationSelection:
    def test_unrelated_descriptor_override_keeps_default_axes(self):
        descriptor = DatasetDescriptor(raw_path_default="data/raw/alternate.csv")
        assert _uses_default_plot_presentation(descriptor) is True

    def test_target_presentation_override_uses_portable_axes(self):
        descriptor = DatasetDescriptor(target_col="g_mag")
        assert _uses_default_plot_presentation(descriptor) is False

    def test_external_default_looking_domain_still_caps_ticks(self):
        descriptor = DatasetDescriptor(name="my_reviews")
        assert _uses_default_plot_presentation(descriptor) is True
        assert _fan_plot_kwargs(descriptor) == {"max_x_ticks": 20}

    def test_aoty_data_override_keeps_legacy_ticks(self):
        descriptor = DatasetDescriptor(raw_path_default="data/raw/alternate.csv")
        assert _fan_plot_kwargs(descriptor) == {}


class TestPredictionsScatterPlot:
    def _write_predictions_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_true": [80.0, 75.0, 90.0],
            "y_pred_mean": [79.0, 76.0, 88.0],
            "y_pred_lower": [70.0, 65.0, 80.0],
            "y_pred_upper": [88.0, 85.0, 96.0],
            "interval_level": 0.90,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_predictions_scatter_plot_saved(self, tmp_path):
        """save_predictions_plot should be called when predictions.json exists."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        # Write predictions.json at the primary split path
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        self._write_predictions_json(pred_path)

        saved = {}

        def _fake_pred_plot(
            y_true,
            y_pred_mean,
            y_pred_lower,
            y_pred_upper,
            output_dir,
            filename_base,
            ci_label="",
            **kwargs,
        ):
            saved["called"] = True
            saved.update(kwargs)
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        ctx = _setup_ctx_more(strict=False)
        ctx.descriptor = DatasetDescriptor(
            name="magnitude",
            target_col="g_mag",
            invert_target_axis=True,
            secondary_target_col=None,
            secondary_prefix=None,
            secondary_n_obs_col=None,
        )
        patches = _base_patches_more(
            tmp_path,
            save_predictions_plot=patch(
                "panelcast.pipelines.publication.save_predictions_plot",
                side_effect=_fake_pred_plot,
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        assert saved.get("called") is True
        assert saved["target_label"] == "g mag"
        assert saved["axis_padding"] is None
        assert saved["invert_axes"] is True
        pred_figs = [p for p in artifacts["figures"] if "predictions_primary" in p]
        assert len(pred_figs) == 2  # pdf + png

    def test_predictions_scatter_plot_error_recorded(self, tmp_path):
        """Exception in save_predictions_plot should record predictions_plot error."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        self._write_predictions_json(pred_path)

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            save_predictions_plot=patch(
                "panelcast.pipelines.publication.save_predictions_plot",
                side_effect=RuntimeError("plot boom"),
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "predictions_plot" in error_names


class TestReliabilityDiagram:
    def _write_calibration_json(self, path: Path, *, with_bin_edges: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "predicted_probs": [0.1, 0.3, 0.5, 0.7, 0.9],
            "observed_freq": [0.08, 0.28, 0.52, 0.68, 0.91],
            "counts": [50, 60, 55, 65, 70],
        }
        if with_bin_edges:
            payload["bin_edges"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_reliability_diagram_saved(self, tmp_path):
        """save_reliability_plot should be called when calibration.json exists."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        cal_path = tmp_path / "outputs/evaluation/within_entity_temporal/calibration.json"
        self._write_calibration_json(cal_path)

        saved = {}

        def _fake_reliability(reliability, output_dir, filename_base):
            saved["called"] = True
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            save_reliability_plot=patch(
                "panelcast.pipelines.publication.save_reliability_plot",
                side_effect=_fake_reliability,
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        assert saved.get("called") is True
        rel_figs = [p for p in artifacts["figures"] if "reliability_primary" in p]
        assert len(rel_figs) == 2

    def test_reliability_diagram_without_bin_edges(self, tmp_path):
        """Reliability plot should reconstruct bin_edges when absent from JSON."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        cal_path = tmp_path / "outputs/evaluation/within_entity_temporal/calibration.json"
        self._write_calibration_json(cal_path, with_bin_edges=False)

        saved = {}

        def _fake_reliability(reliability, output_dir, filename_base):
            saved["bin_edges"] = reliability.bin_edges
            saved["called"] = True
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            save_reliability_plot=patch(
                "panelcast.pipelines.publication.save_reliability_plot",
                side_effect=_fake_reliability,
            ),
        )
        _run_with_patches_more(tmp_path, ctx, patches)
        assert saved.get("called") is True
        # bin_edges reconstructed via linspace
        assert len(saved["bin_edges"]) == 6  # 5 probs → 6 edges

    def test_reliability_diagram_error_recorded(self, tmp_path):
        """Exception in save_reliability_plot should record reliability_plot error."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        cal_path = tmp_path / "outputs/evaluation/within_entity_temporal/calibration.json"
        self._write_calibration_json(cal_path)

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            save_reliability_plot=patch(
                "panelcast.pipelines.publication.save_reliability_plot",
                side_effect=RuntimeError("reliability boom"),
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "reliability_plot" in error_names


def _make_known_artists_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "entity": ["ArtistA", "ArtistA", "ArtistB", "ArtistB"],
            "scenario": ["same", "better", "same", "better"],
            "pred_mean": [78.0, 82.0, 70.0, 74.0],
            "pred_q05": [65.0, 70.0, 58.0, 62.0],
            "pred_q25": [72.0, 76.0, 65.0, 68.0],
            "pred_q50": [78.0, 82.0, 70.0, 74.0],
            "pred_q75": [84.0, 88.0, 75.0, 79.0],
            "pred_q95": [90.0, 94.0, 82.0, 85.0],
        }
    )
    df.to_csv(path, index=False)


def _make_train_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Column names must match DatasetDescriptor defaults: Artist, Album, User_Score
    df = pd.DataFrame(
        {
            "Artist": ["ArtistA", "ArtistA", "ArtistA", "ArtistB", "ArtistB", "ArtistB"],
            "Album": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "User_Score": [75.0, 78.0, 80.0, 68.0, 70.0, 72.0],
            "Release_Date_Parsed": [2010, 2013, 2016, 2011, 2014, 2017],
        }
    )
    df.to_parquet(path, index=False)


def _make_predictions_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": [80.0, 75.0],
        "y_pred_mean": [79.0, 76.0],
        "y_pred_lower": [70.0, 65.0],
        "y_pred_upper": [88.0, 85.0],
        "interval_level": 0.90,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestArtistFanCharts:
    def test_fan_charts_generated_when_artifacts_present(self, tmp_path):
        """Artist fan-charts block should run and log completion."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        _make_known_artists_csv(tmp_path / "outputs/predictions/next_event_known_entities.csv")
        _make_train_parquet(tmp_path / "data/splits/within_entity_temporal/train.parquet")
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)

        fan_calls = []
        fan_quantiles = {}
        fan_kwargs = {}

        def _fake_fan(
            artist,
            actual_scores,
            pred_samples,
            album_labels,
            output_dir,
            filename_base,
            categories=None,
            forecast_quantiles=None,
            **kwargs,
        ):
            fan_calls.append(artist)
            fan_quantiles[artist] = forecast_quantiles
            fan_kwargs[artist] = kwargs
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        ctx = _setup_ctx_more(strict=False)
        ctx.descriptor = DatasetDescriptor(name="magnitude", invert_target_axis=True)
        patches = _base_patches_more(
            tmp_path,
            save_artist_prediction_plot=patch(
                "panelcast.pipelines.publication.save_artist_prediction_plot",
                side_effect=_fake_fan,
            ),
            select_artist_subsets=patch(
                "panelcast.pipelines.publication.select_artist_subsets",
                return_value={"top": ["ArtistA", "ArtistB"]},
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        # Fan charts should have been attempted for both artists
        assert len(fan_calls) >= 1
        # The stored CSV quantiles pass through unmodified for the forecast point.
        assert np.allclose(fan_quantiles["ArtistA"], [65.0, 72.0, 78.0, 84.0, 90.0])
        assert fan_kwargs["ArtistA"] == {
            "event_label": "Album",
            "target_label": "User Score",
            "y_limits": None,
            "invert_y_axis": True,
            "max_x_ticks": 20,
        }
        # No outer error
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" not in error_names

    def test_fan_charts_skipped_when_no_known_csv(self, tmp_path):
        """Fan-chart block should log skip (not error) when known_csv is absent."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(tmp_path)
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" not in error_names

    def test_fan_charts_outer_error_recorded(self, tmp_path):
        """Outer exception in fan-chart block should record artist_fan_charts error."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        _make_known_artists_csv(tmp_path / "outputs/predictions/next_event_known_entities.csv")
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            # Crash inside select_artist_subsets to trigger the outer except
            select_artist_subsets=patch(
                "panelcast.pipelines.publication.select_artist_subsets",
                side_effect=RuntimeError("subset boom"),
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" in error_names

    def test_fan_charts_skipped_when_train_parquet_missing(self, tmp_path):
        """Fan-chart block should log skip (not error) when train.parquet is absent."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        _make_known_artists_csv(tmp_path / "outputs/predictions/next_event_known_entities.csv")
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)
        # No train.parquet written

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            select_artist_subsets=patch(
                "panelcast.pipelines.publication.select_artist_subsets",
                return_value={"top": ["ArtistA"]},
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" not in error_names

    def test_fan_charts_per_artist_failure_recorded(self, tmp_path):
        """Per-artist chart failures must reach artifacts['errors'] so the
        readiness gate cannot report ready while every chart silently failed."""
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", _make_training_summary_more())

        _make_known_artists_csv(tmp_path / "outputs/predictions/next_event_known_entities.csv")
        _make_train_parquet(tmp_path / "data/splits/within_entity_temporal/train.parquet")
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            save_artist_prediction_plot=patch(
                "panelcast.pipelines.publication.save_artist_prediction_plot",
                side_effect=RuntimeError("fan boom"),
            ),
            select_artist_subsets=patch(
                "panelcast.pipelines.publication.select_artist_subsets",
                return_value={"top": ["ArtistA", "ArtistB"]},
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        # The per-artist failure is recorded (not swallowed as a warning only)...
        assert any(name.startswith("artist_fan_chart:") for name in error_names)
        # ...while the outer block itself did not error.
        assert "artist_fan_charts" not in error_names


class TestPriorPredictiveLoadFailure:
    def test_corrupt_prior_predictive_json_logs_warning(self, tmp_path):
        """Corrupt prior_predictive.json should log a warning, not crash."""
        training = _make_training_summary_more(
            priors={
                "mu_artist_loc": 0.0,
                "mu_artist_scale": 1.0,
                "sigma_artist_scale": 0.5,
                "sigma_rw_scale": 0.1,
                "rho_loc": 0.0,
                "rho_scale": 0.3,
                "beta_loc": 0.0,
                "beta_scale": 1.0,
                "sigma_obs_scale": 1.0,
                "sigma_ref_scale": 1.0,
                "n_exponent_alpha": 2.0,
                "n_exponent_beta": 4.0,
                "n_exponent_loc": -2.2,
                "n_exponent_scale": 1.0,
                "betabinom_max_n_reviews": 100.0,
            }
        )
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", training)

        # Write corrupt (non-JSON) prior_predictive.json
        pp_path = tmp_path / "outputs/evaluation/prior_predictive.json"
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        pp_path.write_text("NOT VALID JSON{{{{", encoding="utf-8")

        captured = {}

        def _cap_update(data, **kwargs):
            captured["prior_justification"] = kwargs.get("prior_justification")
            return data

        ctx = _setup_ctx_more(strict=False)
        patches = _base_patches_more(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_cap_update,
            ),
        )
        artifacts = _run_with_patches_more(tmp_path, ctx, patches)
        # Pipeline should not crash; prior_justification may be None or a string
        # (generate_prior_justification_text can work without pp_result)
        assert "model_card" not in [e["artifact"] for e in artifacts["errors"]]


class TestOATSummaryLoadFailure:
    def test_corrupt_oat_summary_csv_logs_warning(self, tmp_path):
        """Unreadable oat_summary.csv should log a warning, not crash."""
        training = _make_training_summary_more(
            priors={
                "mu_artist_loc": 0.0,
                "mu_artist_scale": 1.0,
                "sigma_artist_scale": 0.5,
                "sigma_rw_scale": 0.1,
                "rho_loc": 0.0,
                "rho_scale": 0.3,
                "beta_loc": 0.0,
                "beta_scale": 1.0,
                "sigma_obs_scale": 1.0,
                "sigma_ref_scale": 1.0,
                "n_exponent_alpha": 2.0,
                "n_exponent_beta": 4.0,
                "n_exponent_loc": -2.2,
                "n_exponent_scale": 1.0,
                "betabinom_max_n_reviews": 100.0,
            }
        )
        _write_json_more(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_more())
        _write_json_more(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_more())
        _write_json_more(tmp_path / "models/training_summary.json", training)

        # Write a file that is there but not a valid CSV for pandas
        oat_path = tmp_path / "reports/sensitivity/oat_summary.csv"
        oat_path.parent.mkdir(parents=True, exist_ok=True)
        oat_path.write_bytes(b"\x00\x01\x02\x03")  # binary garbage

        ctx = _setup_ctx_more(strict=False)

        def _mock_read_csv(path, *a, **kw):
            if "oat_summary" in str(path):
                raise pd.errors.ParserError("bad csv")
            return pd.read_csv(path, *a, **kw)

        patches = _base_patches_more(tmp_path)
        # Patch pd.read_csv inside publication module so oat read fails
        with patch("panelcast.pipelines.publication.pd.read_csv", side_effect=_mock_read_csv):
            managers = {k: v.__enter__() for k, v in patches.items()}
            try:
                artifacts = generate_publication_artifacts(ctx)
            finally:
                for v in patches.values():
                    v.__exit__(None, None, None)

        assert "model_card" not in [e["artifact"] for e in artifacts["errors"]]


# --- from unit/pipelines/test_publication_new.py ---


def _write_json_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_metrics_new(**overrides) -> dict:
    primary = {
        "point_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
        "calibration": {
            "within_tolerance": True,
            "coverages": {
                "0.80": {"nominal": 0.80, "empirical": 0.78, "interval_width": 12.0},
                "0.95": {"nominal": 0.95, "empirical": 0.93},
            },
        },
    }
    primary.update(overrides.get("primary_extra", {}))
    splits = {
        "within_entity_temporal": primary,
        "entity_disjoint": {"calibration": {"within_tolerance": True}},
    }
    return {"primary_split": "within_entity_temporal", "splits": splits}


def _make_diagnostics_new(**overrides) -> dict:
    d = {
        "passed": True,
        "rhat_max": 1.003,
        "ess_bulk_min": 2000,
        "ess_tail_min": 1800,
        "divergences": 0,
    }
    d.update(overrides)
    return d


def _make_training_summary_new(**overrides) -> dict:
    d = {
        "n_observations": 500,
        "n_features": 10,
        "n_artists": 100,
        "max_albums": 50,
        "mcmc_config": {
            "num_chains": 4,
            "num_warmup": 500,
            "num_samples": 750,
            "chain_method": "sequential",
            "target_accept_prob": 0.9,
            "max_tree_depth": 10,
        },
    }
    d.update(overrides)
    return d


def _fake_export_table_new(df, base_path, caption):
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text("% fake\n", encoding="utf-8")


def _fake_plot_new(*_args, output_dir=None, filename_base="", **_kw):
    if output_dir is None:
        raise ValueError("output_dir required")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{filename_base}.pdf"
    png = output_dir / f"{filename_base}.png"
    pdf.write_text("pdf", encoding="utf-8")
    png.write_text("png", encoding="utf-8")
    return pdf, png


def _fake_write_model_card_new(_data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Model Card\n", encoding="utf-8")


def _make_fake_idata_new(extra_vars=None):
    idata = MagicMock()
    vars_set = {
        "user_beta",
        "user_mu_artist",
        "user_sigma_artist",
        "user_sigma_obs",
    }
    if extra_vars:
        vars_set.update(extra_vars)
    idata.posterior = {v: 1 for v in vars_set}
    return idata


def _setup_ctx_new(**overrides):
    defaults = {
        "run_dir": None,
        "strict": False,
        "evaluate_secondary_split": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _base_patches_new(tmp_path, idata=None, **overrides):
    if idata is None:
        idata = _make_fake_idata_new()
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    defaults = {
        "load_manifest": patch(
            "panelcast.pipelines.publication.load_manifest",
            return_value=fake_manifest,
        ),
        "load_model": patch(
            "panelcast.pipelines.publication.load_model",
            return_value=idata,
        ),
        "create_coefficient_table": patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        "create_diagnostics_table": patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        "export_table": patch(
            "panelcast.pipelines.publication.export_table",
            side_effect=_fake_export_table_new,
        ),
        "save_trace_plot": patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *a, **kw: _fake_plot_new(
                output_dir=kw["output_dir"], filename_base=kw["filename_base"]
            ),
        ),
        "save_posterior_plot": patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *a, **kw: _fake_plot_new(
                output_dir=kw["output_dir"], filename_base=kw["filename_base"]
            ),
        ),
        "create_default_model_card_data": patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        "update_model_card_with_results": patch(
            "panelcast.pipelines.publication.update_model_card_with_results",
            side_effect=lambda data, **kw: data,
        ),
        "write_model_card": patch(
            "panelcast.pipelines.publication.write_model_card",
            side_effect=_fake_write_model_card_new,
        ),
        "Path": patch(
            "panelcast.pipelines.publication.Path",
            side_effect=lambda p: tmp_path / p,
        ),
    }
    defaults.update(overrides)
    return defaults


def _run_with_patches_new(tmp_path, ctx, patches_dict):
    managers = {k: v.__enter__() for k, v in patches_dict.items()}
    try:
        return generate_publication_artifacts(ctx)
    finally:
        for v in patches_dict.values():
            v.__exit__(None, None, None)


class TestDataclassFields:
    def test_coverage_like_interval_width_default(self):
        c = _CoverageLike(empirical=0.85)
        assert c.empirical == 0.85
        assert c.interval_width is None

    def test_coverage_like_with_width(self):
        c = _CoverageLike(empirical=0.80, interval_width=12.5)
        assert c.interval_width == 12.5

    def test_point_metrics_like_fields(self):
        p = _PointMetricsLike(mae=0.8, rmse=1.2, r2=0.5)
        assert p.mae == 0.8
        assert p.rmse == 1.2
        assert p.r2 == 0.5

    def test_loo_like_fields(self):
        loo = _LooLike(elpd_loo=-42.0, se_elpd=3.5)
        assert loo.elpd_loo == -42.0
        assert loo.se_elpd == 3.5

    def test_convergence_like_fields(self):
        c = _ConvergenceLike(
            passed=True,
            rhat_max=1.003,
            ess_bulk_min=2000.0,
            ess_tail_min=1800.0,
            divergences=0,
            failing_params=[],
        )
        assert c.passed is True
        assert c.rhat_max == 1.003
        assert c.ess_bulk_min == 2000.0
        assert c.ess_tail_min == 1800.0
        assert c.divergences == 0
        assert c.failing_params == []

    def test_convergence_like_with_failing_params(self):
        c = _ConvergenceLike(
            passed=False,
            rhat_max=1.05,
            ess_bulk_min=100.0,
            ess_tail_min=50.0,
            divergences=15,
            failing_params=["user_beta", "user_sigma_obs"],
        )
        assert c.passed is False
        assert len(c.failing_params) == 2


class TestParseConvergenceFailingParams:
    def test_failing_params_preserved(self):
        parsed = _parse_convergence(
            {
                "passed": False,
                "rhat_max": 1.05,
                "ess_bulk_min": 100,
                "failing_params": ["user_beta[0]", "user_sigma_obs"],
            },
            {},
        )
        assert parsed is not None
        assert parsed.failing_params == ["user_beta[0]", "user_sigma_obs"]

    def test_failing_params_default_empty(self):
        parsed = _parse_convergence(
            {"passed": True, "rhat_max": 1.001, "ess_bulk_min": 2000},
            {},
        )
        assert parsed is not None
        assert parsed.failing_params == []


class TestParseCoverageEdgeCases:
    def test_nominal_inferred_from_key(self):
        """When nominal is missing from entry, prob_key is used."""
        result = _parse_coverage_results(
            {
                "calibration": {
                    "coverages": {
                        "0.5": {"empirical": 0.48},
                    }
                }
            }
        )
        assert result is not None
        assert 0.5 in result
        assert result[0.5].empirical == pytest.approx(0.48)

    def test_coverage_zero_and_one_rejected(self):
        """Nominal values of 0.0 and 1.0 should be rejected."""
        result = _parse_coverage_results(
            {
                "calibration": {
                    "coverages": {
                        "0.0": {"nominal": 0.0, "empirical": 0.0},
                        "1.0": {"nominal": 1.0, "empirical": 1.0},
                    }
                }
            }
        )
        assert result is None

    def test_legacy_and_current_merge(self):
        """Legacy entries fill in gaps not covered by current schema."""
        result = _parse_coverage_results(
            {
                "calibration": {
                    "coverages": {
                        "0.80": {"nominal": 0.80, "empirical": 0.79},
                    },
                    "coverage_50": 0.48,
                    "coverage_80": 0.75,  # Should NOT override current
                }
            }
        )
        assert result is not None
        assert result[0.8].empirical == pytest.approx(0.79)  # current wins
        assert result[0.5].empirical == pytest.approx(0.48)  # legacy fills gap


class TestCoefficientTableWithExtendedVars:
    def test_idata_with_sigma_ref_and_n_exponent(self, tmp_path):
        """Coefficient table should include sigma_ref and n_exponent vars."""
        idata = _make_fake_idata_new(extra_vars={"user_sigma_ref", "user_n_exponent"})
        _write_json_new(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_new())
        _write_json_new(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_new())
        _write_json_new(tmp_path / "models/training_summary.json", _make_training_summary_new())
        ctx = _setup_ctx_new()

        captured_var_names = {}

        def _capture_coef_table(idata_arg, var_names=None):
            captured_var_names["var_names"] = var_names
            return pd.DataFrame({"x": [1]})

        patches = _base_patches_new(
            tmp_path,
            idata=idata,
            create_coefficient_table=patch(
                "panelcast.pipelines.publication.create_coefficient_table",
                side_effect=_capture_coef_table,
            ),
        )
        _run_with_patches_new(tmp_path, ctx, patches)

        vn = captured_var_names["var_names"]
        assert "user_sigma_ref" in vn
        assert "user_n_exponent" in vn
        # sigma_ref should come before sigma_obs
        assert vn.index("user_sigma_ref") < vn.index("user_sigma_obs")


class TestMetricsTableSharpness:
    def test_sharpness_rows_from_coverage_width(self, tmp_path):
        """Metrics summary should include sharpness rows when interval_width is present."""
        metrics = _make_metrics_new(
            primary_extra={
                "calibration": {
                    "within_tolerance": True,
                    "coverages": {
                        "0.80": {
                            "nominal": 0.80,
                            "empirical": 0.78,
                            "interval_width": 12.5,
                        },
                        "0.95": {
                            "nominal": 0.95,
                            "empirical": 0.93,
                            "interval_width": 25.0,
                        },
                    },
                    "wis": 4.5,
                },
            }
        )
        _write_json_new(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json_new(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_new())
        _write_json_new(tmp_path / "models/training_summary.json", _make_training_summary_new())
        ctx = _setup_ctx_new()

        captured_dfs = []

        def _capturing_export(df, base_path, caption):
            captured_dfs.append((caption, df))
            _fake_export_table_new(df, base_path, caption)

        patches = _base_patches_new(
            tmp_path,
            export_table=patch(
                "panelcast.pipelines.publication.export_table",
                side_effect=_capturing_export,
            ),
        )
        _run_with_patches_new(tmp_path, ctx, patches)

        # Find metrics summary table
        perf_tables = [(c, d) for c, d in captured_dfs if "performance" in c.lower()]
        assert len(perf_tables) == 1
        df = perf_tables[0][1]
        metric_names = df["Metric"].tolist()
        assert any("Sharpness" in m for m in metric_names)
        assert "WIS" in metric_names


class TestRunDirCopyComplete:
    def test_figures_and_tables_copied_to_run_dir(self, tmp_path):
        """Run dir copy should include figures and tables subdirectories."""
        _write_json_new(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_new())
        _write_json_new(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_new())
        _write_json_new(tmp_path / "models/training_summary.json", _make_training_summary_new())

        run_dir = tmp_path / "outputs" / "run_test"
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = _setup_ctx_new(run_dir=run_dir)
        patches = _base_patches_new(tmp_path)
        _run_with_patches_new(tmp_path, ctx, patches)

        run_reports = run_dir / "reports"
        # Figures should be copied
        assert (run_reports / "figures").exists()
        fig_files = list((run_reports / "figures").iterdir())
        assert len(fig_files) > 0
        # Tables should be copied
        assert (run_reports / "tables").exists()
        table_files = list((run_reports / "tables").iterdir())
        assert len(table_files) > 0


class TestPriorJustificationLoading:
    def test_prior_predictive_json_loaded(self, tmp_path):
        """Prior predictive result JSON should be loaded when available."""
        training = _make_training_summary_new()
        training["priors"] = {
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
        }
        _write_json_new(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_new())
        _write_json_new(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_new())
        _write_json_new(tmp_path / "models/training_summary.json", training)

        # Write prior predictive result
        pp_data = {
            "summary": {"mean": 75.0, "std": 10.0},
            "reasonable": True,
            "bounds": [0, 100],
            "fraction_in_bounds": 0.95,
            "n_samples": 1000,
            "n_obs_original": 500,
            "max_obs": 2000,
            "seed": 42,
        }
        _write_json_new(tmp_path / "outputs/evaluation/prior_predictive.json", pp_data)

        ctx = _setup_ctx_new()

        captured = {}

        def _capture_update(data, **kwargs):
            captured["kwargs"] = kwargs
            return data

        patches = _base_patches_new(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_capture_update,
            ),
        )
        _run_with_patches_new(tmp_path, ctx, patches)

        assert "prior_justification" in captured["kwargs"]

    def test_oat_summary_loaded(self, tmp_path):
        """OAT sensitivity summary CSV should be loaded when available."""
        training = _make_training_summary_new()
        training["priors"] = {
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
        }
        _write_json_new(tmp_path / "outputs/evaluation/metrics.json", _make_metrics_new())
        _write_json_new(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics_new())
        _write_json_new(tmp_path / "models/training_summary.json", training)

        # The sensitivity stage writes under <reports>/sensitivity, not the
        # old hardcoded outputs/sensitivity path.
        oat_dir = tmp_path / "reports" / "sensitivity"
        oat_dir.mkdir(parents=True, exist_ok=True)
        oat_df = pd.DataFrame(
            {
                "parameter": ["sigma_beta"],
                "baseline": [1.0],
                "multiplier": [2.0],
                "rmse_change": [0.05],
            }
        )
        oat_df.to_csv(oat_dir / "oat_summary.csv", index=False)

        captured = {}

        def _capture_justification(priors, prior_predictive_result=None, sensitivity_summary=None):
            captured["sensitivity_summary"] = sensitivity_summary
            return "justification"

        ctx = _setup_ctx_new()
        patches = _base_patches_new(tmp_path)
        with patch(
            "panelcast.evaluation.prior_predictive.generate_prior_justification_text",
            side_effect=_capture_justification,
        ):
            artifacts = _run_with_patches_new(tmp_path, ctx, patches)
        assert isinstance(artifacts, dict)
        # The CSV at the real sensitivity-stage path was actually loaded.
        assert captured["sensitivity_summary"] is not None
        assert list(captured["sensitivity_summary"]["parameter"]) == ["sigma_beta"]


class TestBuildReadinessNew:
    def test_non_dict_calibration_in_primary(self):
        """Non-dict calibration in primary_metrics should be handled."""
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": "bad_type"},
                },
            },
            diagnostics={"passed": True, "rhat_max": 1.001, "ess_bulk_min": 5000},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=False,
        )
        checks = {c["name"]: c for c in payload["checks"]}
        assert checks["primary_calibration_within_tolerance"]["passed"] is False

    def test_non_dict_splits_payload(self):
        """Non-dict splits in metrics should not crash."""
        payload = _build_publication_readiness(
            metrics={"splits": "not_a_dict"},
            diagnostics={"passed": True, "rhat_max": 1.001, "ess_bulk_min": 5000},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=True,
        )
        checks = {c["name"]: c for c in payload["checks"]}
        assert checks["secondary_split_evaluated"]["passed"] is False


class TestReadinessConsistentWithRealGate:
    """End-to-end: a diagnostics payload produced by the REAL check_convergence
    (serialized the way evaluate.py writes diagnostics.json) must yield readiness
    verdicts consistent with the gate's passed flag. Post-#142 both apply the
    same TOTAL bulk-ESS floor, so ~600 total with threshold 400 passes both
    (a per-chain misreading would demand 1600 and flip the verdict)."""

    def test_real_check_convergence_payload_feeds_readiness(self):
        az = pytest.importorskip("arviz")
        from panelcast.models.bayes.diagnostics import check_convergence

        rng = np.random.default_rng(42)
        n_chains, n_draws = 4, 150
        idata = az.from_dict(
            posterior={"user_mu": rng.normal(size=(n_chains, n_draws))},
            sample_stats={"diverging": np.zeros((n_chains, n_draws), dtype=bool)},
        )

        diags = check_convergence(idata, ess_threshold=400)
        # iid draws: total bulk ESS ~ n_chains * n_draws = 600 — above the
        # total floor but below the per-chain misreading (400 * 4 = 1600).
        assert 400 < diags.ess_bulk_min < 1600
        assert diags.passed is True

        # Serialized exactly as evaluate.py writes diagnostics.json.
        payload = {
            "passed": diags.passed,
            "rhat_max": float(diags.rhat_max),
            "ess_bulk_min": float(diags.ess_bulk_min),
            "divergences": int(diags.divergences),
            "rhat_threshold": float(diags.rhat_threshold),
            "ess_threshold": int(diags.ess_threshold),
        }
        readiness = _build_publication_readiness(
            metrics={
                "primary_split": "within_entity_temporal",
                "splits": {
                    "within_entity_temporal": {"calibration": {"within_tolerance": True}},
                    SECONDARY_SPLIT: {"calibration": {"within_tolerance": True}},
                },
            },
            diagnostics=payload,
            training_summary={"mcmc_config": {"num_chains": n_chains}},
            artifact_errors=[],
            require_secondary_split=True,
            prior_predictive_available=True,
        )
        checks = _check_map(readiness)
        assert checks["ess_within_threshold"]["passed"] is (diags.ess_bulk_min >= 400)
        assert checks["convergence_passed"]["passed"] is diags.passed
        assert readiness["ready"] is diags.passed


class TestRepoRootModelCardUntouched:
    """#135 regression: publication must never write the curated repo-root
    MODEL_CARD.md. Runs without the Path reroute patch, in a temp cwd, so a
    reintroduced root write would actually hit the sentinel."""

    def test_generate_artifacts_leaves_cwd_model_card_alone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sentinel = "# Curated repo-root model card - do not overwrite\n"
        root_card = tmp_path / "MODEL_CARD.md"
        root_card.write_text(sentinel, encoding="utf-8")

        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        del patches["Path"]  # real cwd-relative paths; no tmp_path rerouting
        artifacts = _run_with_patches(tmp_path, ctx, patches)

        # The run-scoped card was written under reports/, not the repo root.
        assert (tmp_path / "reports" / "MODEL_CARD.md").exists()
        assert any(Path(d).name == "MODEL_CARD.md" for d in artifacts["docs"])
        assert root_card.read_text(encoding="utf-8") == sentinel


class TestRenderReadinessMarkdownNew:
    def test_pass_status(self):
        md = _render_publication_readiness_markdown(
            {"ready": True, "critical_failed": [], "recommended_failed": [], "checks": []}
        )
        assert "**Status:** PASS" in md

    def test_fail_status(self):
        md = _render_publication_readiness_markdown(
            {"ready": False, "critical_failed": ["x"], "recommended_failed": [], "checks": []}
        )
        assert "**Status:** FAIL" in md

    def test_check_rows_rendered(self):
        md = _render_publication_readiness_markdown(
            {
                "ready": True,
                "critical_failed": [],
                "recommended_failed": [],
                "checks": [
                    {
                        "name": "my_check",
                        "severity": "critical",
                        "passed": True,
                        "detail": "all good",
                    },
                ],
            }
        )
        assert "my_check" in md
        assert "all good" in md
        assert "yes" in md
