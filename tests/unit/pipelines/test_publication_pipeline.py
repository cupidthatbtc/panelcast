"""Tests for publication pipeline failure signaling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from panelcast.pipelines.publication import generate_publication_artifacts


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "primary_split": "within_artist_temporal",
        "splits": {
            "within_artist_temporal": {
                "point_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
                "calibration": {
                    "coverages": {
                        "0.80": {"nominal": 0.80, "empirical": 0.78},
                        "0.95": {"nominal": 0.95, "empirical": 0.93},
                    }
                },
            }
        },
    }
    path.write_text(json.dumps(metrics), encoding="utf-8")


def _fake_export_table(df: pd.DataFrame, base_path: str, caption: str) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".tex").write_text("% fake table\n", encoding="utf-8")


def _fake_plot(output_dir: Path, filename_base: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename_base}.pdf"
    png_path = output_dir / f"{filename_base}.png"
    pdf_path.write_text("pdf", encoding="utf-8")
    png_path.write_text("png", encoding="utf-8")
    return pdf_path, png_path


def _fake_write_model_card(_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Model Card\n", encoding="utf-8")


def test_publication_non_strict_records_errors_and_writes_status(tmp_path):
    """Non-strict mode should record failures in artifact_status.json."""
    _write_metrics(tmp_path / "outputs/evaluation/metrics.json")

    ctx = SimpleNamespace(run_dir=None, strict=False)
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = MagicMock()
    fake_idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }

    with (
        patch("panelcast.pipelines.publication.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.publication.load_model", return_value=fake_idata),
        patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch("panelcast.pipelines.publication.export_table", side_effect=_fake_export_table),
        patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        patch("panelcast.pipelines.publication.update_model_card_with_results", return_value={}),
        patch(
            "panelcast.pipelines.publication.write_model_card", side_effect=_fake_write_model_card
        ),
        patch("panelcast.pipelines.publication.Path", side_effect=lambda p: tmp_path / p),
    ):
        artifacts = generate_publication_artifacts(ctx)

    status_path = tmp_path / "reports/artifact_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["n_errors"] >= 1
    assert any(item["artifact"] == "coefficients_table" for item in payload["errors"])
    assert len(artifacts["errors"]) >= 1


def test_publication_strict_raises_when_any_artifact_fails(tmp_path):
    """Strict mode should fail if any publication artifact fails."""
    _write_metrics(tmp_path / "outputs/evaluation/metrics.json")

    ctx = SimpleNamespace(run_dir=None, strict=True)
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = MagicMock()
    fake_idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }

    with (
        patch("panelcast.pipelines.publication.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.publication.load_model", return_value=fake_idata),
        patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            side_effect=RuntimeError("boom"),
        ),
        patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch("panelcast.pipelines.publication.export_table", side_effect=_fake_export_table),
        patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        patch("panelcast.pipelines.publication.update_model_card_with_results", return_value={}),
        patch(
            "panelcast.pipelines.publication.write_model_card", side_effect=_fake_write_model_card
        ),
        patch("panelcast.pipelines.publication.Path", side_effect=lambda p: tmp_path / p),
    ):
        with pytest.raises(ValueError, match="Publication artifact generation failed"):
            generate_publication_artifacts(ctx)

    assert (tmp_path / "reports/artifact_status.json").exists()


def test_publication_model_card_includes_loaded_metrics_and_training_summary(tmp_path):
    """Model card update should receive parsed diagnostics/metrics payloads."""
    metrics_payload = {
        "primary_split": "within_artist_temporal",
        "splits": {
            "within_artist_temporal": {
                "point_metrics": {"rmse": 1.25, "mae": 0.75, "r2": 0.42},
                "calibration": {
                    "coverages": {
                        "0.80": {"nominal": 0.80, "empirical": 0.79},
                        "0.95": {"nominal": 0.95, "empirical": 0.94},
                    }
                },
                "info_criteria": {
                    "loo": {"elpd": -123.4, "se": 5.6},
                },
            }
        },
    }
    metrics_path = tmp_path / "outputs/evaluation/metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    diagnostics_payload = {
        "passed": True,
        "rhat_max": 1.005,
        "ess_bulk_min": 1200,
        "ess_tail_min": 1100,
        "divergences": 0,
    }
    diagnostics_path = tmp_path / "outputs/evaluation/diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics_payload), encoding="utf-8")

    training_summary_payload = {
        "n_observations": 1234,
        "n_features": 32,
        "n_artists": 456,
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
    training_summary_path = tmp_path / "models/training_summary.json"
    training_summary_path.parent.mkdir(parents=True, exist_ok=True)
    training_summary_path.write_text(json.dumps(training_summary_payload), encoding="utf-8")

    ctx = SimpleNamespace(run_dir=None, strict=False)
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = MagicMock()
    fake_idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }

    captured: dict[str, object] = {}

    def _capture_update(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return data

    with (
        patch("panelcast.pipelines.publication.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.publication.load_model", return_value=fake_idata),
        patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch("panelcast.pipelines.publication.export_table", side_effect=_fake_export_table),
        patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        patch(
            "panelcast.pipelines.publication.update_model_card_with_results",
            side_effect=_capture_update,
        ),
        patch(
            "panelcast.pipelines.publication.write_model_card", side_effect=_fake_write_model_card
        ),
        patch("panelcast.pipelines.publication.Path", side_effect=lambda p: tmp_path / p),
    ):
        generate_publication_artifacts(ctx)

    model_card_data = captured["data"]
    assert model_card_data.dataset_size == 1234
    assert model_card_data.hyperparameters["n_features"] == 32
    assert model_card_data.hyperparameters["num_samples"] == 750

    kwargs = captured["kwargs"]
    assert kwargs["convergence"] is not None
    assert kwargs["convergence"].passed is True
    assert kwargs["coverage_results"] is not None
    assert set(kwargs["coverage_results"].keys()) == {0.8, 0.95}
    assert kwargs["point_metrics"] is not None
    assert kwargs["point_metrics"].rmse == pytest.approx(1.25)
    assert kwargs["loo_result"] is not None
    assert kwargs["loo_result"].elpd_loo == pytest.approx(-123.4)


def test_publication_v1_metrics_backward_compat(tmp_path):
    """v1-style metrics payload (no schema_version/splits/point_metrics) should not crash."""
    metrics_payload = {
        "rmse": 1.3,
        "mae": 0.7,
        "r2": 0.4,
        "calibration": {"coverage_80": 0.79, "coverage_95": 0.93},
        "crps": {"mean_crps": 5.0, "n_obs": 100},
    }
    metrics_path = tmp_path / "outputs/evaluation/metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    ctx = SimpleNamespace(run_dir=None, strict=False)
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = MagicMock()
    fake_idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }

    with (
        patch("panelcast.pipelines.publication.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.publication.load_model", return_value=fake_idata),
        patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch("panelcast.pipelines.publication.export_table", side_effect=_fake_export_table),
        patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        patch("panelcast.pipelines.publication.update_model_card_with_results", return_value={}),
        patch(
            "panelcast.pipelines.publication.write_model_card", side_effect=_fake_write_model_card
        ),
        patch("panelcast.pipelines.publication.Path", side_effect=lambda p: tmp_path / p),
    ):
        artifacts = generate_publication_artifacts(ctx)

    assert not any(err["artifact"] == "metrics_summary_table" for err in artifacts["errors"])


def test_publication_ppc_summary_missing_fields_is_tolerated(tmp_path):
    """PPC summary with missing/null fields should not crash publication pipeline."""
    metrics_payload = {
        "primary_split": "within_artist_temporal",
        "splits": {
            "within_artist_temporal": {
                "point_metrics": {"rmse": 1.0, "mae": 0.8, "r2": 0.5},
                "calibration": {"coverages": {"0.80": {"nominal": 0.8, "empirical": 0.79}}},
                "ppc": {
                    "summary": {
                        "mean": {"p_value": 0.45},  # missing observed/mc_se
                        "sd": {"observed": None, "p_value": 0.52, "mc_se": None},
                    },
                    "n_samples": 200,
                },
            }
        },
    }
    metrics_path = tmp_path / "outputs/evaluation/metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

    ctx = SimpleNamespace(run_dir=None, strict=False)
    fake_manifest = SimpleNamespace(current={"user_score": "model.nc"})
    fake_idata = MagicMock()
    fake_idata.posterior = {
        "user_beta": 1,
        "user_mu_artist": 1,
        "user_sigma_artist": 1,
        "user_sigma_obs": 1,
    }

    with (
        patch("panelcast.pipelines.publication.load_manifest", return_value=fake_manifest),
        patch("panelcast.pipelines.publication.load_model", return_value=fake_idata),
        patch(
            "panelcast.pipelines.publication.create_coefficient_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch(
            "panelcast.pipelines.publication.create_diagnostics_table",
            return_value=pd.DataFrame({"x": [1]}),
        ),
        patch("panelcast.pipelines.publication.export_table", side_effect=_fake_export_table),
        patch(
            "panelcast.pipelines.publication.save_trace_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.save_posterior_plot",
            side_effect=lambda *args, **kwargs: _fake_plot(
                kwargs["output_dir"], kwargs["filename_base"]
            ),
        ),
        patch(
            "panelcast.pipelines.publication.create_default_model_card_data",
            return_value=SimpleNamespace(dataset_size=0, hyperparameters={}),
        ),
        patch("panelcast.pipelines.publication.update_model_card_with_results", return_value={}),
        patch(
            "panelcast.pipelines.publication.write_model_card", side_effect=_fake_write_model_card
        ),
        patch("panelcast.pipelines.publication.Path", side_effect=lambda p: tmp_path / p),
    ):
        artifacts = generate_publication_artifacts(ctx)

    assert not any(err["artifact"] == "ppc_density_plot" for err in artifacts["errors"])
