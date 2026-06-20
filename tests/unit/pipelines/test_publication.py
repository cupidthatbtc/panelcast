"""Tests for publication artifact helpers."""

from __future__ import annotations

from itertools import product

import pytest

from panelcast.pipelines.publication import (
    SECONDARY_SPLIT,
    _build_publication_readiness,
    _get_coefficient_var_names,
    _parse_convergence,
    _parse_coverage_results,
    _parse_loo_result,
    _parse_point_metrics,
    _render_publication_readiness_markdown,
    _resolve_primary_metrics,
    _safe_float,
    _safe_int,
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
            "primary_split": "within_artist_temporal",
            "splits": {
                "within_artist_temporal": primary,
                SECONDARY_SPLIT: {"point_metrics": {"rmse": 2.0, "mae": 1.5, "r2": 0.1}},
            },
        }
        assert _resolve_primary_metrics(metrics) is primary

    @pytest.mark.parametrize(
        "metrics",
        [
            {},
            {"primary_split": "within_artist_temporal"},
            {"splits": {"within_artist_temporal": {"a": 1}}},
            {
                "primary_split": "missing",
                "splits": {"within_artist_temporal": {"point_metrics": {"rmse": 1}}},
            },
            {"primary_split": 123, "splits": {"within_artist_temporal": {"a": 1}}},
            {"primary_split": "x", "splits": []},
            {
                "primary_split": "within_artist_temporal",
                "splits": {"within_artist_temporal": []},
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
    def test_parse_loo_result_valid(self):
        result = _parse_loo_result({"info_criteria": {"loo": {"elpd": -123.4, "se": 5.6}}})
        assert result is not None
        assert result.elpd_loo == pytest.approx(-123.4)
        assert result.se_elpd == pytest.approx(5.6)

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"info_criteria": None},
            {"info_criteria": {"loo": None}},
            {"info_criteria": {"loo": {"elpd": -123.4}}},
            {"info_criteria": {"loo": {"se": 5.6}}},
            {"info_criteria": {"loo": {"elpd": "bad", "se": 5.6}}},
            {"info_criteria": {"loo": {"elpd": -123.4, "se": "bad"}}},
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
            (None, 1000.0),
            ("bad", 1000.0),
        ],
    )
    def test_parse_convergence_ess_tail_fallback(self, ess_tail_min, expected_tail):
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
        assert parsed.ess_tail_min == pytest.approx(expected_tail)


class TestPublicationReadiness:
    def test_ready_when_all_critical_checks_pass(self):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_artist_temporal",
                "splits": {
                    "within_artist_temporal": {"calibration": {"within_tolerance": True}},
                    "artist_disjoint": {"calibration": {"within_tolerance": True}},
                },
            },
            diagnostics={"passed": True, "rhat_max": 1.003, "ess_bulk_min": 1800},
            training_summary={"mcmc_config": {"num_chains": 4}},
            artifact_errors=[],
            require_secondary_split=True,
        )
        assert payload["ready"] is True
        assert payload["critical_failed"] == []

    def test_not_ready_for_single_chain_and_missing_secondary(self):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_artist_temporal",
                "splits": {
                    "within_artist_temporal": {"calibration": {"within_tolerance": False}},
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
            "within_artist_temporal": {"calibration": {"within_tolerance": primary_within}},
        }
        if secondary_state == "pass":
            splits[SECONDARY_SPLIT] = {"calibration": {"within_tolerance": True}}
        elif secondary_state == "fail":
            splits[SECONDARY_SPLIT] = {"calibration": {"within_tolerance": False}}

        payload = _build_publication_readiness(
            metrics={"primary_split": "within_artist_temporal", "splits": splits},
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
            assert checks["ess_within_threshold"]["passed"] is (ess_bulk_min >= 400 * num_chains)
            if ess_bulk_min < 400 * num_chains:
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
            "within_artist_temporal": {"calibration": {"within_tolerance": True}},
        }
        if secondary_state == "present":
            splits[SECONDARY_SPLIT] = {"calibration": {"within_tolerance": False}}

        payload = _build_publication_readiness(
            metrics={"primary_split": "within_artist_temporal", "splits": splits},
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

    def test_readiness_flags_artifact_errors(self):
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_artist_temporal",
                "splits": {
                    "within_artist_temporal": {"calibration": {"within_tolerance": True}},
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
                "primary_split": "within_artist_temporal",
                "splits": {
                    "within_artist_temporal": {
                        "calibration": {
                            "coverages": {"0.95": {"nominal": 0.95, "empirical": 0.94}}
                        },
                        "point_metrics": {"mae": 0.8, "rmse": 1.1, "r2": 0.4},
                        "info_criteria": {"loo": {"elpd": -42.5, "se": 1.2}},
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
