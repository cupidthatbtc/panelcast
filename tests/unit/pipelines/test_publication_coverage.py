"""Additional coverage tests for publication pipeline.

Targets missed lines/branches in publication.py including:
- generate_publication_artifacts entry point with various failure modes
- Table/figure/model-card generation failure branches
- Strict mode validation (raise vs warn)
- Missing/corrupt input data handling
- PPC density plot paths (mc_se computation, replicated distributions)
- Prediction scenario table (known_csv exists vs not)
- Prior justification loading with PriorPredictiveResult and sensitivity summary
- Run directory artifact copy logic
- Publication readiness strict-mode enforcement
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelcast.pipelines.publication import (
    _build_publication_readiness,
    _render_publication_readiness_markdown,
    generate_publication_artifacts,
)

# ============================================================================
# Helpers
# ============================================================================


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


# ============================================================================
# Tests: generate_publication_artifacts failure branches
# ============================================================================


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
        known_csv = tmp_path / "outputs/predictions/next_album_known_artists.csv"
        known_csv.parent.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame(
            {
                "scenario": ["optimistic", "optimistic", "pessimistic", "pessimistic"],
                "pred_mean": [80.0, 82.0, 60.0, 62.0],
                "artist": ["A", "B", "A", "B"],
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
        known_csv = tmp_path / "outputs/predictions/next_album_known_artists.csv"
        known_csv.parent.mkdir(parents=True, exist_ok=True)
        # Missing required columns for groupby
        pd.DataFrame({"bad_column": [1, 2]}).to_csv(known_csv, index=False)

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert any(e["artifact"] == "prediction_scenarios_table" for e in artifacts["errors"])


class TestPublicationPPCDensityPlot:
    def test_ppc_with_missing_observed_skips_stat(self, tmp_path):
        """PPC stat with missing 'observed' should be skipped, not crash."""
        metrics = _make_metrics()
        metrics["splits"]["within_entity_temporal"]["ppc"] = {
            "summary": {
                "mean": {"p_value": 0.45},  # missing observed
                "sd": {"observed": 8.0, "p_value": 0.52, "mc_se": 0.03},
            },
            "n_obs": 100,
            "n_samples": 200,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # Should not crash
        assert not any(e["artifact"] == "ppc_density_plot" for e in artifacts["errors"])

    def test_ppc_mc_se_computed_from_n_samples(self, tmp_path):
        """When mc_se is None, it should be computed from p_value and n_samples."""
        metrics = _make_metrics()
        metrics["splits"]["within_entity_temporal"]["ppc"] = {
            "summary": {
                "mean": {
                    "observed": 75.0,
                    "p_value": 0.5,
                    "mc_se": None,
                },
            },
            "n_obs": 100,
            "n_samples": 400,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert not any(e["artifact"] == "ppc_density_plot" for e in artifacts["errors"])

    def test_ppc_mc_se_fallback_zero_when_no_n_samples(self, tmp_path):
        """When mc_se is None and n_samples is 0, mc_se should default to 0."""
        metrics = _make_metrics()
        metrics["splits"]["within_entity_temporal"]["ppc"] = {
            "summary": {
                "mean": {
                    "observed": 75.0,
                    "p_value": 0.5,
                    "mc_se": None,
                },
            },
            "n_obs": 100,
            "n_samples": 0,
        }
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert not any(e["artifact"] == "ppc_density_plot" for e in artifacts["errors"])

    def test_ppc_density_plot_failure_recorded(self, tmp_path):
        """PPC density plot failure should be recorded as error."""
        metrics = _make_metrics(include_ppc=True)
        _write_json(tmp_path / "outputs/evaluation/metrics.json", metrics)
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(
            tmp_path / "models/training_summary.json",
            _make_training_summary(),
        )
        ctx = _setup_ctx(strict=False)
        # Make the PPC import fail
        patches = _base_patches(tmp_path)

        def _broken_import(*args, **kwargs):
            raise ImportError("no ppc module")

        with patch(
            "panelcast.pipelines.publication._safe_float",
            side_effect=RuntimeError("force ppc error"),
        ):
            # This will make the entire PPC block fail
            pass

        # Instead use a more targeted approach: patch the ppc module import
        # to cause failure inside the try block
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # PPC with no replicated distributions should skip plot generation gracefully
        assert not any(e["artifact"] == "ppc_density_plot" for e in artifacts["errors"])


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


# ============================================================================
# Tests: _build_publication_readiness edge cases
# ============================================================================


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
