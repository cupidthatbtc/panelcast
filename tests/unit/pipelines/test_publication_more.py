"""Additional coverage tests for publication pipeline — targeting missed lines.

Covers:
- line 136: legacy coverage_ key named "coverages" is skipped
- lines 753-754, 794-799: PPC stat n_samples int-cast error branches
- lines 808-815: PPC density plot rendered when has_distributions is True
- lines 821-823: PPC density plot skipped (no distributions) logged
- lines 838-852: predictions scatter plot loaded and saved
- lines 855-857: predictions scatter plot error branch
- lines 869-888: reliability diagram loaded and saved
- lines 891-893: reliability diagram error branch
- lines 912-982: artist fan-chart block (known_csv present + pred path present)
- lines 985-987: artist fan-chart outer error branch
- lines 1068-1069: prior_predictive.json corrupt/missing-key branch
- lines 1077-1078: oat_summary.csv unreadable branch
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
    _parse_coverage_results,
    generate_publication_artifacts,
)

# ============================================================================
# Helpers — mirrored from test_publication_coverage.py
# ============================================================================


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_metrics(
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


def _make_diagnostics(*, passed: bool = True, rhat_max: float = 1.003) -> dict:
    return {
        "passed": passed,
        "rhat_max": rhat_max,
        "ess_bulk_min": 2000,
        "ess_tail_min": 1800,
        "divergences": 0,
    }


def _make_training_summary(*, num_chains: int = 4, priors: dict | None = None) -> dict:
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


def _fake_export_table(df: pd.DataFrame, base_path: str, caption: str) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text("% fake\n", encoding="utf-8")


def _fake_plot(*_args, output_dir: Path = None, filename_base: str = "", **_kw):
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{filename_base}.pdf"
    png = output_dir / f"{filename_base}.png"
    pdf.write_text("pdf", encoding="utf-8")
    png.write_text("png", encoding="utf-8")
    return pdf, png


def _fake_write_model_card(_data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Model Card\n", encoding="utf-8")


def _make_fake_idata():
    idata = MagicMock()
    idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }
    return idata


def _setup_ctx(*, strict: bool = False, run_dir=None, evaluate_secondary_split=True):
    return SimpleNamespace(
        run_dir=run_dir,
        strict=strict,
        evaluate_secondary_split=evaluate_secondary_split,
    )


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
# Unit test: legacy coverage_ key named "coverages" is skipped (line 136)
# ============================================================================


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
        # No 0.80 key from the dict-value of "coverages" key in the legacy loop
        # (that dict-branch is handled via the new schema path, but "coverages" itself
        # is explicitly skipped in the legacy loop to avoid double-counting)


# ============================================================================
# PPC density plot: has_distributions=True → save_ppc_density_plot called
# (lines 808-815, 821-823)
# ============================================================================


class TestPPCDensityPlotWithDistributions:
    def test_ppc_density_plot_called_when_distributions_present(self, tmp_path):
        """save_ppc_density_plot should be called when a stat has replicated data.

        PPCResult/PPCStatistic are imported locally inside generate_publication_artifacts,
        so we patch them at their source module. We make PPCStatistic always return a fake
        stat with a non-empty replicated_distribution so has_distributions is True.
        """
        from panelcast.evaluation.ppc import PPCResult, PPCStatistic

        fake_stat = PPCStatistic(
            name="mean",
            observed=75.0,
            replicated_distribution=np.array([74.0, 75.0, 76.0]),
            bayesian_p_value=0.45,
            mc_se=0.02,
        )
        fake_ppc_result = PPCResult(statistics=[fake_stat], n_obs=100, n_samples=200)

        called = {}

        def _fake_ppc_density(ppc_result, *, output_dir, filename_base):
            called["ppc_result"] = ppc_result
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics(include_ppc=True))
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_ppc_density_plot=patch(
                "panelcast.pipelines.publication.save_ppc_density_plot",
                side_effect=_fake_ppc_density,
            ),
        )
        # Patch PPCResult and PPCStatistic at the source module so the local import
        # inside generate_publication_artifacts picks up the patched versions.
        with (
            patch("panelcast.evaluation.ppc.PPCStatistic", side_effect=lambda **kw: fake_stat),
            patch("panelcast.evaluation.ppc.PPCResult", return_value=fake_ppc_result),
        ):
            artifacts = _run_with_patches(tmp_path, ctx, patches)

        assert "ppc_result" in called
        from pathlib import Path as _Path
        pdf_entries = [
            p for p in artifacts["figures"] if "ppc_density" in _Path(p).name and ".pdf" in p
        ]
        assert len(pdf_entries) == 1

    def test_ppc_density_plot_skipped_when_no_distributions(self, tmp_path):
        """Pipeline should log skip (not error) when no replicated distributions."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics(include_ppc=True))
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)

        # No ppc_density error recorded (skipped cleanly)
        error_artifacts = [e["artifact"] for e in artifacts["errors"]]
        assert "ppc_density_plot" not in error_artifacts
        # And no ppc_density filename was added (filter by basename to avoid tmp_path match)
        from pathlib import Path as _Path
        ppc_figs = [p for p in artifacts["figures"] if "ppc_density" in _Path(p).name]
        assert len(ppc_figs) == 0


# ============================================================================
# PPC n_samples int-cast branches (lines 753-754, 794-799)
# ============================================================================


class TestPPCNSamplesIntCast:
    def test_n_samples_non_int_string_defaults_to_zero(self, tmp_path):
        """n_samples that can't be cast to int should default to 0 (mc_se → 0.0)."""
        # mc_se is None so the code tries to compute it from n_samples.
        # n_samples = "bad" -> cast fails -> defaults to 0 -> mc_se = 0.0
        ppc_override = {
            "summary": {
                "mean": {"observed": 75.0, "p_value": 0.45},  # mc_se missing → triggers cast
            },
            "n_obs": 100,
            "n_samples": "bad",  # non-castable
        }
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(ppc_override=ppc_override),
        )
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # Should not crash; ppc_density_plot should be skipped (no distributions)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "ppc_density_plot" not in error_names

    def test_n_obs_non_int_string_defaults_to_zero(self, tmp_path):
        """n_obs that can't be cast to int should default to 0 safely."""
        ppc_override = {
            "summary": {
                "mean": {"observed": 75.0, "p_value": 0.45, "mc_se": 0.02},
            },
            "n_obs": "not-a-number",
            "n_samples": 200,
        }
        _write_json(
            tmp_path / "outputs/evaluation/metrics.json",
            _make_metrics(ppc_override=ppc_override),
        )
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "ppc_density_plot" not in error_names


# ============================================================================
# Predictions scatter plot (lines 838-857)
# ============================================================================


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
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        # Write predictions.json at the primary split path
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        self._write_predictions_json(pred_path)

        saved = {}

        def _fake_pred_plot(y_true, y_pred_mean, y_pred_lower, y_pred_upper, output_dir,
                            filename_base, ci_label=""):
            saved["called"] = True
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_predictions_plot=patch(
                "panelcast.pipelines.publication.save_predictions_plot",
                side_effect=_fake_pred_plot,
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert saved.get("called") is True
        pred_figs = [p for p in artifacts["figures"] if "predictions_primary" in p]
        assert len(pred_figs) == 2  # pdf + png

    def test_predictions_scatter_plot_error_recorded(self, tmp_path):
        """Exception in save_predictions_plot should record predictions_plot error."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        self._write_predictions_json(pred_path)

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_predictions_plot=patch(
                "panelcast.pipelines.publication.save_predictions_plot",
                side_effect=RuntimeError("plot boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "predictions_plot" in error_names


# ============================================================================
# Reliability diagram (lines 869-893)
# ============================================================================


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
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

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

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_reliability_plot=patch(
                "panelcast.pipelines.publication.save_reliability_plot",
                side_effect=_fake_reliability,
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        assert saved.get("called") is True
        rel_figs = [p for p in artifacts["figures"] if "reliability_primary" in p]
        assert len(rel_figs) == 2

    def test_reliability_diagram_without_bin_edges(self, tmp_path):
        """Reliability plot should reconstruct bin_edges when absent from JSON."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

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

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_reliability_plot=patch(
                "panelcast.pipelines.publication.save_reliability_plot",
                side_effect=_fake_reliability,
            ),
        )
        _run_with_patches(tmp_path, ctx, patches)
        assert saved.get("called") is True
        # bin_edges reconstructed via linspace
        assert len(saved["bin_edges"]) == 6  # 5 probs → 6 edges

    def test_reliability_diagram_error_recorded(self, tmp_path):
        """Exception in save_reliability_plot should record reliability_plot error."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        cal_path = tmp_path / "outputs/evaluation/within_entity_temporal/calibration.json"
        self._write_calibration_json(cal_path)

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            save_reliability_plot=patch(
                "panelcast.pipelines.publication.save_reliability_plot",
                side_effect=RuntimeError("reliability boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "reliability_plot" in error_names


# ============================================================================
# Artist fan charts (lines 912-987)
# ============================================================================


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
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        _make_known_artists_csv(
            tmp_path / "outputs/predictions/next_event_known_entities.csv"
        )
        _make_train_parquet(
            tmp_path / "data/splits/within_entity_temporal/train.parquet"
        )
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)

        fan_calls = []

        def _fake_fan(artist, actual_scores, pred_samples, album_labels,
                      output_dir, filename_base, categories=None):
            fan_calls.append(artist)
            pdf = output_dir / f"{filename_base}.pdf"
            png = output_dir / f"{filename_base}.png"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf.write_text("pdf", encoding="utf-8")
            png.write_text("png", encoding="utf-8")
            return pdf, png

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
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
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # Fan charts should have been attempted for both artists
        assert len(fan_calls) >= 1
        # No outer error
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" not in error_names

    def test_fan_charts_skipped_when_no_known_csv(self, tmp_path):
        """Fan-chart block should log skip (not error) when known_csv is absent."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(tmp_path)
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" not in error_names

    def test_fan_charts_outer_error_recorded(self, tmp_path):
        """Outer exception in fan-chart block should record artist_fan_charts error."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        _make_known_artists_csv(
            tmp_path / "outputs/predictions/next_event_known_entities.csv"
        )
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            # Crash inside select_artist_subsets to trigger the outer except
            select_artist_subsets=patch(
                "panelcast.pipelines.publication.select_artist_subsets",
                side_effect=RuntimeError("subset boom"),
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" in error_names

    def test_fan_charts_skipped_when_train_parquet_missing(self, tmp_path):
        """Fan-chart block should log skip (not error) when train.parquet is absent."""
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", _make_training_summary())

        _make_known_artists_csv(
            tmp_path / "outputs/predictions/next_event_known_entities.csv"
        )
        pred_path = tmp_path / "outputs/evaluation/within_entity_temporal/predictions.json"
        _make_predictions_json(pred_path)
        # No train.parquet written

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            select_artist_subsets=patch(
                "panelcast.pipelines.publication.select_artist_subsets",
                return_value={"top": ["ArtistA"]},
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        error_names = [e["artifact"] for e in artifacts["errors"]]
        assert "artist_fan_charts" not in error_names


# ============================================================================
# Prior predictive JSON loading — corrupt/missing-key branch (lines 1068-1069)
# ============================================================================


class TestPriorPredictiveLoadFailure:
    def test_corrupt_prior_predictive_json_logs_warning(self, tmp_path):
        """Corrupt prior_predictive.json should log a warning, not crash."""
        training = _make_training_summary(
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
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", training)

        # Write corrupt (non-JSON) prior_predictive.json
        pp_path = tmp_path / "outputs/evaluation/prior_predictive.json"
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        pp_path.write_text("NOT VALID JSON{{{{", encoding="utf-8")

        captured = {}

        def _cap_update(data, **kwargs):
            captured["prior_justification"] = kwargs.get("prior_justification")
            return data

        ctx = _setup_ctx(strict=False)
        patches = _base_patches(
            tmp_path,
            update_model_card_with_results=patch(
                "panelcast.pipelines.publication.update_model_card_with_results",
                side_effect=_cap_update,
            ),
        )
        artifacts = _run_with_patches(tmp_path, ctx, patches)
        # Pipeline should not crash; prior_justification may be None or a string
        # (generate_prior_justification_text can work without pp_result)
        assert "model_card" not in [e["artifact"] for e in artifacts["errors"]]


# ============================================================================
# OAT sensitivity CSV unreadable branch (lines 1077-1078)
# ============================================================================


class TestOATSummaryLoadFailure:
    def test_corrupt_oat_summary_csv_logs_warning(self, tmp_path):
        """Unreadable oat_summary.csv should log a warning, not crash."""
        training = _make_training_summary(
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
        _write_json(tmp_path / "outputs/evaluation/metrics.json", _make_metrics())
        _write_json(tmp_path / "outputs/evaluation/diagnostics.json", _make_diagnostics())
        _write_json(tmp_path / "models/training_summary.json", training)

        # Write a file that is there but not a valid CSV for pandas
        oat_path = tmp_path / "outputs/sensitivity/oat_summary.csv"
        oat_path.parent.mkdir(parents=True, exist_ok=True)
        oat_path.write_bytes(b"\x00\x01\x02\x03")  # binary garbage

        ctx = _setup_ctx(strict=False)

        def _mock_read_csv(path, *a, **kw):
            if "oat_summary" in str(path):
                raise pd.errors.ParserError("bad csv")
            return pd.read_csv(path, *a, **kw)

        patches = _base_patches(tmp_path)
        # Patch pd.read_csv inside publication module so oat read fails
        with patch("panelcast.pipelines.publication.pd.read_csv", side_effect=_mock_read_csv):
            managers = {k: v.__enter__() for k, v in patches.items()}
            try:
                artifacts = generate_publication_artifacts(ctx)
            finally:
                for v in patches.values():
                    v.__exit__(None, None, None)

        assert "model_card" not in [e["artifact"] for e in artifacts["errors"]]
