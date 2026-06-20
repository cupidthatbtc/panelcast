"""Additional coverage tests for publication pipeline.

Targets uncovered lines/branches including:
- _CoverageLike / _PointMetricsLike / _LooLike / _ConvergenceLike dataclass fields
- PPC density plot with replicated distributions present
- Prior predictive result loading from JSON
- OAT sensitivity summary loading
- Coefficient table with sigma_ref + n_exponent present
- Sharpness rows in metrics table
- generate_publication_artifacts with run_dir copy of figures/tables
- _parse_coverage_results with mixed current + legacy (legacy-only new nominal)
- _build_publication_readiness with non-dict metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.publication import (
    _build_publication_readiness,
    _ConvergenceLike,
    _CoverageLike,
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
    generate_publication_artifacts,
)

# ============================================================================
# Helpers
# ============================================================================


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_metrics(**overrides) -> dict:
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
        "within_artist_temporal": primary,
        "artist_disjoint": {"calibration": {"within_tolerance": True}},
    }
    return {"primary_split": "within_artist_temporal", "splits": splits}


def _make_diagnostics(**overrides) -> dict:
    d = {
        "passed": True,
        "rhat_max": 1.003,
        "ess_bulk_min": 2000,
        "ess_tail_min": 1800,
        "divergences": 0,
    }
    d.update(overrides)
    return d


def _make_training_summary(**overrides) -> dict:
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


def _fake_export_table(df, base_path, caption):
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text("% fake\n", encoding="utf-8")


def _fake_plot(*_args, output_dir=None, filename_base="", **_kw):
    if output_dir is None:
        raise ValueError("output_dir required")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{filename_base}.pdf"
    png = output_dir / f"{filename_base}.png"
    pdf.write_text("pdf", encoding="utf-8")
    png.write_text("png", encoding="utf-8")
    return pdf, png


def _fake_write_model_card(_data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Model Card\n", encoding="utf-8")


def _make_fake_idata(extra_vars=None):
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


def _setup_ctx(**overrides):
    defaults = {
        "run_dir": None,
        "strict": False,
        "evaluate_secondary_split": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _base_patches(tmp_path, idata=None, **overrides):
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
    managers = {k: v.__enter__() for k, v in patches_dict.items()}
    try:
        return generate_publication_artifacts(ctx)
    finally:
        for v in patches_dict.values():
            v.__exit__(None, None, None)


# ============================================================================
# Tests: dataclass field access
# ============================================================================


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


# ============================================================================
# Tests: _parse_convergence with failing_params
# ============================================================================


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


# ============================================================================
# Tests: _parse_coverage_results edge cases
# ============================================================================


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


# ============================================================================
# Tests: coefficient table with sigma_ref and n_exponent
# ============================================================================


class TestCoefficientTableWithExtendedVars:
    def test_idata_with_sigma_ref_and_n_exponent(self, tmp_path):
        """Coefficient table should include sigma_ref and n_exponent vars."""
        idata = _make_fake_idata(extra_vars={"user_sigma_ref", "user_n_exponent"})
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())
        ctx = _setup_ctx()

        captured_var_names = {}

        def _capture_coef_table(idata_arg, var_names=None):
            captured_var_names["var_names"] = var_names
            return pd.DataFrame({"x": [1]})

        patches = _base_patches(
            tmp_path,
            idata=idata,
            create_coefficient_table=patch(
                "panelcast.pipelines.publication.create_coefficient_table",
                side_effect=_capture_coef_table,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        vn = captured_var_names["var_names"]
        assert "user_sigma_ref" in vn
        assert "user_n_exponent" in vn
        # sigma_ref should come before sigma_obs
        assert vn.index("user_sigma_ref") < vn.index("user_sigma_obs")


# ============================================================================
# Tests: metrics table with sharpness and WIS
# ============================================================================


class TestMetricsTableSharpness:
    def test_sharpness_rows_from_coverage_width(self, tmp_path):
        """Metrics summary should include sharpness rows when interval_width is present."""
        metrics = _make_metrics(
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
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())
        ctx = _setup_ctx()

        captured_dfs = []

        def _capturing_export(df, base_path, caption):
            captured_dfs.append((caption, df))
            _fake_export_table(df, base_path, caption)

        patches = _base_patches(
            tmp_path,
            export_table=patch(
                "panelcast.pipelines.publication.export_table",
                side_effect=_capturing_export,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)

        # Find metrics summary table
        perf_tables = [(c, d) for c, d in captured_dfs if "performance" in c.lower()]
        assert len(perf_tables) == 1
        df = perf_tables[0][1]
        metric_names = df["Metric"].tolist()
        assert any("Sharpness" in m for m in metric_names)
        assert "WIS" in metric_names


# ============================================================================
# Tests: run dir copy includes figures and tables
# ============================================================================


class TestRunDirCopyComplete:
    def test_figures_and_tables_copied_to_run_dir(self, tmp_path):
        """Run dir copy should include figures and tables subdirectories."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        run_dir = tmp_path / "outputs" / "run_test"
        run_dir.mkdir(parents=True, exist_ok=True)

        ctx = _setup_ctx(run_dir=run_dir)
        patches = _base_patches(tmp_path)
        _run_with_patches(tmp_path, ctx, patches)

        run_reports = run_dir / "reports"
        # Figures should be copied
        assert (run_reports / "figures").exists()
        fig_files = list((run_reports / "figures").iterdir())
        assert len(fig_files) > 0
        # Tables should be copied
        assert (run_reports / "tables").exists()
        table_files = list((run_reports / "tables").iterdir())
        assert len(table_files) > 0


# ============================================================================
# Tests: PPC with prior predictive and OAT sensitivity
# ============================================================================


class TestPriorJustificationLoading:
    def test_prior_predictive_json_loaded(self, tmp_path):
        """Prior predictive result JSON should be loaded when available."""
        training = _make_training_summary()
        training["priors"] = {
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", training)

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
        _write_json(tmp_path / "outputs/evaluation/prior_predictive.json", pp_data)

        ctx = _setup_ctx()

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

        assert "prior_justification" in captured["kwargs"]

    def test_oat_summary_loaded(self, tmp_path):
        """OAT sensitivity summary CSV should be loaded when available."""
        training = _make_training_summary()
        training["priors"] = {
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", training)

        # Write OAT summary CSV
        oat_dir = tmp_path / "outputs" / "sensitivity"
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

        ctx = _setup_ctx()
        patches = _base_patches(tmp_path)
        # Should not crash when loading OAT summary
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert isinstance(artifacts, dict)


# ============================================================================
# Tests: _build_publication_readiness additional edge cases
# ============================================================================


class TestBuildReadinessNew:
    def test_non_dict_calibration_in_primary(self):
        """Non-dict calibration in primary_metrics should be handled."""
        payload = _build_publication_readiness(
            metrics={
                "primary_split": "within_artist_temporal",
                "splits": {
                    "within_artist_temporal": {"calibration": "bad_type"},
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


# ============================================================================
# Tests: _render_publication_readiness_markdown
# ============================================================================


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
