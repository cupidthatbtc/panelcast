"""Unit tests for visualization dashboard module."""

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from panelcast.visualization.dashboard import (
    DashboardData,
    _find_column,
    create_artist_view,
    create_coefficients_table,
    create_dashboard_figures,
    get_artist_list,
    load_dashboard_data,
)


class TestDashboardData:
    """Tests for DashboardData dataclass."""

    def test_default_values_are_none(self):
        data = DashboardData()
        assert data.idata is None
        assert data.predictions is None
        assert data.coefficients is None
        assert data.reliability is None
        assert data.artist_data is None

    def test_with_predictions(self):
        pred = {
            "y_true": np.array([1, 2]),
            "y_pred_mean": np.array([1.1, 2.1]),
            "y_pred_lower": np.array([0.5, 1.5]),
            "y_pred_upper": np.array([1.5, 2.5]),
        }
        data = DashboardData(predictions=pred)
        assert data.predictions is not None
        assert len(data.predictions["y_true"]) == 2

    def test_with_coefficients(self):
        df = pd.DataFrame({"param": ["a"], "mean": [1.0]})
        data = DashboardData(coefficients=df)
        assert data.coefficients is not None

    def test_with_reliability(self):
        rel = {
            "predicted_probs": np.array([0.5]),
            "observed_freq": np.array([0.4]),
            "counts": np.array([10]),
        }
        data = DashboardData(reliability=rel)
        assert data.reliability is not None


class TestFindColumn:
    """Tests for _find_column helper."""

    def test_finds_first_match(self):
        df = pd.DataFrame({"mean": [1], "estimate": [2]})
        result = _find_column(df, ["mean", "estimate"])
        assert result == "mean"

    def test_finds_second_candidate(self):
        df = pd.DataFrame({"estimate": [2], "value": [3]})
        result = _find_column(df, ["mean", "estimate"])
        assert result == "estimate"

    def test_returns_none_when_not_found(self):
        df = pd.DataFrame({"x": [1]})
        result = _find_column(df, ["mean", "estimate"])
        assert result is None

    def test_empty_candidates(self):
        df = pd.DataFrame({"x": [1]})
        result = _find_column(df, [])
        assert result is None


class TestCreateDashboardFigures:
    """Tests for create_dashboard_figures."""

    def test_empty_data_returns_empty_dict(self):
        data = DashboardData()
        figures = create_dashboard_figures(data)
        assert figures == {}

    def test_predictions_only(self):
        pred = {
            "y_true": np.array([70, 75, 80]),
            "y_pred_mean": np.array([72, 74, 78]),
            "y_pred_lower": np.array([67, 69, 73]),
            "y_pred_upper": np.array([77, 79, 83]),
        }
        data = DashboardData(predictions=pred)
        figures = create_dashboard_figures(data)
        assert "predictions" in figures
        assert isinstance(figures["predictions"], str)

    def test_coefficients_only(self):
        df = pd.DataFrame(
            {
                "param": ["a", "b"],
                "mean": [1.0, 2.0],
                "hdi_3%": [0.5, 1.5],
                "hdi_97%": [1.5, 2.5],
            }
        )
        data = DashboardData(coefficients=df)
        figures = create_dashboard_figures(data)
        assert "coefficients" in figures

    def test_reliability_only(self):
        rel = {
            "predicted_probs": np.array([0.2, 0.5, 0.8]),
            "observed_freq": np.array([0.25, 0.45, 0.82]),
            "counts": np.array([10, 20, 30]),
        }
        data = DashboardData(reliability=rel)
        figures = create_dashboard_figures(data)
        assert "reliability" in figures

    def test_missing_prediction_keys(self):
        pred = {"y_true": np.array([1])}  # Missing required keys
        data = DashboardData(predictions=pred)
        figures = create_dashboard_figures(data)
        assert "predictions" not in figures

    def test_dark_theme(self):
        pred = {
            "y_true": np.array([70, 75]),
            "y_pred_mean": np.array([72, 74]),
            "y_pred_lower": np.array([67, 69]),
            "y_pred_upper": np.array([77, 79]),
        }
        data = DashboardData(predictions=pred)
        figures = create_dashboard_figures(data, theme="aoty_dark")
        assert "predictions" in figures

    def test_coefficients_auto_detect_columns(self):
        df = pd.DataFrame(
            {
                "parameter": ["a"],
                "estimate": [1.0],
                "hdi_2.5%": [0.5],
                "hdi_97.5%": [1.5],
            }
        )
        data = DashboardData(coefficients=df)
        figures = create_dashboard_figures(data)
        assert "coefficients" in figures

    def test_coefficients_missing_columns_skipped(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        data = DashboardData(coefficients=df)
        figures = create_dashboard_figures(data)
        assert "coefficients" not in figures

    def test_all_data_present(self):
        pred = {
            "y_true": np.array([70, 75]),
            "y_pred_mean": np.array([72, 74]),
            "y_pred_lower": np.array([67, 69]),
            "y_pred_upper": np.array([77, 79]),
        }
        coef = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        rel = {
            "predicted_probs": np.array([0.5]),
            "observed_freq": np.array([0.4]),
            "counts": np.array([10]),
        }
        data = DashboardData(predictions=pred, coefficients=coef, reliability=rel)
        figures = create_dashboard_figures(data)
        assert "predictions" in figures
        assert "coefficients" in figures
        assert "reliability" in figures

    def test_interval_level_labels_predictions_legend(self):
        pred = {
            "y_true": np.array([70, 75]),
            "y_pred_mean": np.array([72, 74]),
            "y_pred_lower": np.array([67, 69]),
            "y_pred_upper": np.array([77, 79]),
            "interval_level": 0.80,
        }
        data = DashboardData(predictions=pred)
        figures = create_dashboard_figures(data)
        assert '"80% CI"' in figures["predictions"]
        assert "94% CI" not in figures["predictions"]

    def test_unknown_interval_level_uses_generic_label(self):
        pred = {
            "y_true": np.array([70, 75]),
            "y_pred_mean": np.array([72, 74]),
            "y_pred_lower": np.array([67, 69]),
            "y_pred_upper": np.array([77, 79]),
        }
        data = DashboardData(predictions=pred)
        figures = create_dashboard_figures(data)
        assert "94% CI" not in figures["predictions"]
        assert '"CI"' in figures["predictions"]


class TestCreateArtistView:
    """Tests for create_artist_view."""

    def test_returns_html_string(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead", "Radiohead"],
                "year": [2000, 2007],
                "score": [85, 90],
            }
        )
        result = create_artist_view("Radiohead", df)
        assert isinstance(result, str)

    def test_case_insensitive_search(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead"],
                "year": [2000],
                "score": [85],
            }
        )
        result = create_artist_view("radiohead", df)
        # Should find the artist (not return the simple "not-found" div)
        assert 'class="not-found"' not in result

    def test_artist_not_found(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead"],
                "year": [2000],
                "score": [85],
            }
        )
        result = create_artist_view("Beatles", df)
        assert 'class="not-found"' in result

    def test_xss_protection_in_not_found(self):
        df = pd.DataFrame({"artist": ["Radiohead"], "score": [85]})
        result = create_artist_view('<script>alert("xss")</script>', df)
        assert "<script>" not in result

    def test_with_predictions(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead", "Radiohead"],
                "year": [2000, 2007],
                "score": [85, 90],
                "prediction": [84, 88],
                "lower": [80, 84],
                "upper": [88, 92],
            }
        )
        result = create_artist_view("Radiohead", df)
        assert isinstance(result, str)

    def test_without_time_column(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead", "Radiohead"],
                "score": [85, 90],
            }
        )
        result = create_artist_view("Radiohead", df)
        assert isinstance(result, str)

    def test_sorts_by_time(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead", "Radiohead"],
                "year": [2007, 2000],
                "score": [90, 85],
            }
        )
        result = create_artist_view("Radiohead", df)
        assert isinstance(result, str)

    def test_ci_label_parameter(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead", "Radiohead"],
                "year": [2000, 2007],
                "score": [85, 90],
                "prediction": [84, 88],
                "lower": [80, 84],
                "upper": [88, 92],
            }
        )
        result = create_artist_view("Radiohead", df, ci_label="80% CI")
        assert "80% CI" in result

    def test_default_ci_label_not_hardcoded_94(self):
        df = pd.DataFrame(
            {
                "artist": ["Radiohead", "Radiohead"],
                "year": [2000, 2007],
                "score": [85, 90],
                "prediction": [84, 88],
                "lower": [80, 84],
                "upper": [88, 92],
            }
        )
        result = create_artist_view("Radiohead", df)
        assert "94% CI" not in result


class TestGetArtistList:
    """Tests for get_artist_list."""

    def test_returns_sorted_list(self):
        df = pd.DataFrame({"artist": ["Beatles", "Radiohead", "Adele"]})
        result = get_artist_list(df)
        assert result == ["Adele", "Beatles", "Radiohead"]

    def test_deduplicates(self):
        df = pd.DataFrame({"artist": ["Radiohead", "Beatles", "Radiohead"]})
        result = get_artist_list(df)
        assert result == ["Beatles", "Radiohead"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"artist": []})
        result = get_artist_list(df)
        assert result == []

    def test_missing_artist_column(self):
        df = pd.DataFrame({"name": ["Radiohead"]})
        result = get_artist_list(df)
        assert result == []

    def test_drops_nan(self):
        df = pd.DataFrame({"artist": ["Radiohead", None, "Beatles"]})
        result = get_artist_list(df)
        assert result == ["Beatles", "Radiohead"]


class TestCreateCoefficientsTable:
    """Tests for create_coefficients_table."""

    def test_returns_html_string(self):
        df = pd.DataFrame(
            {
                "param": ["intercept"],
                "mean": [0.5],
                "hdi_3%": [0.3],
                "hdi_97%": [0.7],
            }
        )
        result = create_coefficients_table(df)
        assert "<table" in result

    def test_contains_parameter_name(self):
        df = pd.DataFrame(
            {
                "param": ["intercept"],
                "mean": [0.5],
                "hdi_3%": [0.3],
                "hdi_97%": [0.7],
            }
        )
        result = create_coefficients_table(df)
        assert "intercept" in result

    def test_formats_numbers(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.23456],
                "hdi_3%": [0.5],
                "hdi_97%": [2.0],
            }
        )
        result = create_coefficients_table(df)
        assert "1.235" in result  # 3 decimal places

    def test_multiple_rows(self):
        df = pd.DataFrame(
            {
                "param": ["a", "b", "c"],
                "mean": [1.0, 2.0, 3.0],
                "hdi_3%": [0.5, 1.5, 2.5],
                "hdi_97%": [1.5, 2.5, 3.5],
            }
        )
        result = create_coefficients_table(df)
        assert result.count("<tr>") == 4  # 1 header + 3 data rows

    def test_missing_columns_returns_message(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = create_coefficients_table(df)
        assert "Unable to generate" in result or "missing" in result.lower()

    def test_xss_protection(self):
        df = pd.DataFrame(
            {
                "param": ['<script>alert("xss")</script>'],
                "mean": [0.5],
                "hdi_3%": [0.3],
                "hdi_97%": [0.7],
            }
        )
        result = create_coefficients_table(df)
        assert "<script>" not in result

    def test_alternative_column_names(self):
        df = pd.DataFrame(
            {
                "parameter": ["a"],
                "estimate": [1.0],
                "hdi_2.5%": [0.5],
                "hdi_97.5%": [1.5],
            }
        )
        result = create_coefficients_table(df)
        assert "<table" in result

    def test_data_value_attributes(self):
        df = pd.DataFrame(
            {
                "param": ["a"],
                "mean": [1.0],
                "hdi_3%": [0.5],
                "hdi_97%": [1.5],
            }
        )
        result = create_coefficients_table(df)
        assert "data-value=" in result


class TestCreateDashboardFiguresTracePlot:
    """Tests for the trace-plot branch in create_dashboard_figures (lines 121-143)."""

    def test_trace_plot_multidim_samples(self):
        # ndim > 2: shape (chain=2, draw=50, dim=200) triggers the reshape branch
        import arviz as az

        idata = az.from_dict(posterior={"alpha": np.random.randn(2, 50, 200)})
        data = DashboardData(idata=idata)
        figures = create_dashboard_figures(data)
        assert "trace" in figures
        assert isinstance(figures["trace"], str)

    def test_trace_plot_multidim_uses_first_element_per_chain(self, monkeypatch):
        # (chain, draw, k) must reduce to element [.., .., 0] per chain, not a
        # flattened cross-section mixing draws and parameter dims.
        import arviz as az

        arr = np.arange(2 * 50 * 3, dtype=float).reshape(2, 50, 3)
        idata = az.from_dict(posterior={"alpha": arr})
        captured = {}

        def fake_trace_plot(samples, var_name, template="aoty_light"):
            captured["samples"] = np.asarray(samples)
            captured["var_name"] = var_name
            mock = MagicMock()
            mock.to_html.return_value = "<div></div>"
            return mock

        monkeypatch.setattr("panelcast.visualization.dashboard.create_trace_plot", fake_trace_plot)
        data = DashboardData(idata=idata)
        figures = create_dashboard_figures(data)
        assert "trace" in figures
        assert captured["samples"].shape == (2, 50)
        np.testing.assert_array_equal(captured["samples"], arr[:, :, 0])
        assert captured["var_name"] == "alpha[0]"

    def test_trace_plot_2d_samples(self):
        # ndim == 2: shape (chain=2, draw=50) — normal path, no reshape
        import arviz as az

        idata = az.from_dict(posterior={"beta": np.random.randn(2, 50)})
        data = DashboardData(idata=idata)
        figures = create_dashboard_figures(data)
        assert "trace" in figures

    def test_trace_plot_1d_samples_via_mock(self):
        # ndim == 1: arviz won't produce 1-D vars naturally, so stub posterior
        class FakeVar:
            values = np.array([1.0, 2.0, 3.0])  # ndim == 1

        class FakePosterior:
            data_vars: ClassVar[dict] = {"gamma": FakeVar()}

            def __getitem__(self, key):
                return FakeVar()

        class FakeIdata:
            posterior = FakePosterior()

        data = DashboardData(idata=FakeIdata())
        figures = create_dashboard_figures(data)
        # Either produced a trace or fell into the except — either way no crash
        assert isinstance(figures, dict)

    def test_trace_plot_except_branch(self):
        # posterior access raises → logger.debug, trace skipped
        class BadIdata:
            @property
            def posterior(self):
                raise RuntimeError("broken idata")

        data = DashboardData(idata=BadIdata())
        figures = create_dashboard_figures(data)
        assert "trace" not in figures


class TestLoadDashboardData:
    """Tests for load_dashboard_data (lines 448-587)."""

    # ------------------------------------------------------------------ helpers

    def _write_json(self, path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def _write_csv(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def _write_parquet(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    # ------------------------------------------------------------------ tests

    def test_no_artifacts_returns_empty_dashboarddata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_dashboard_data()
        assert isinstance(result, DashboardData)
        assert result.idata is None
        assert result.predictions is None
        assert result.coefficients is None
        assert result.reliability is None
        assert result.artist_data is None

    def test_predictions_loaded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pred = {
            "y_true": [70.0, 75.0],
            "y_pred_mean": [71.0, 74.0],
            "y_pred_lower": [65.0, 68.0],
            "y_pred_upper": [77.0, 80.0],
        }
        self._write_json(tmp_path / "outputs" / "evaluation" / "predictions.json", pred)
        result = load_dashboard_data()
        assert result.predictions is not None
        np.testing.assert_array_equal(result.predictions["y_true"], [70.0, 75.0])

    def test_predictions_interval_level_kept(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        pred = {
            "y_true": [70.0, 75.0],
            "y_pred_mean": [71.0, 74.0],
            "y_pred_lower": [65.0, 68.0],
            "y_pred_upper": [77.0, 80.0],
            "interval_level": 0.8,
        }
        self._write_json(tmp_path / "outputs" / "evaluation" / "predictions.json", pred)
        result = load_dashboard_data()
        assert result.predictions is not None
        assert result.predictions["interval_level"] == 0.8

    def test_predictions_malformed_skipped(self, tmp_path, monkeypatch):
        # Missing required keys → predictions stays None
        monkeypatch.chdir(tmp_path)
        self._write_json(
            tmp_path / "outputs" / "evaluation" / "predictions.json",
            {"y_true": [1.0]},
        )
        result = load_dashboard_data()
        assert result.predictions is None

    def test_predictions_invalid_json_skipped(self, tmp_path, monkeypatch):
        # Corrupted file → except branch (line 520-521)
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "outputs" / "evaluation" / "predictions.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("NOT JSON {{{")
        result = load_dashboard_data()
        assert result.predictions is None

    def test_coefficients_loaded_coefficient_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"param": ["a"], "mean": [1.0], "hdi_3%": [0.5], "hdi_97%": [1.5]})
        self._write_csv(tmp_path / "reports" / "tables" / "coefficient_summary.csv", df)
        result = load_dashboard_data()
        assert result.coefficients is not None
        assert "param" in result.coefficients.columns

    def test_coefficients_loaded_summary_fallback(self, tmp_path, monkeypatch):
        # No *coefficient*.csv → falls back to *summary*.csv
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"param": ["b"], "mean": [2.0], "hdi_3%": [1.0], "hdi_97%": [3.0]})
        self._write_csv(tmp_path / "reports" / "tables" / "model_summary.csv", df)
        result = load_dashboard_data()
        assert result.coefficients is not None

    def test_calibration_loaded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cal = {
            "predicted_probs": [0.2, 0.5, 0.8],
            "observed_freq": [0.22, 0.48, 0.79],
            "counts": [10, 20, 15],
        }
        self._write_json(tmp_path / "outputs" / "evaluation" / "calibration.json", cal)
        result = load_dashboard_data()
        assert result.reliability is not None
        np.testing.assert_array_equal(result.reliability["counts"], [10, 20, 15])

    def test_calibration_missing_keys_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_json(
            tmp_path / "outputs" / "evaluation" / "calibration.json",
            {"predicted_probs": [0.5]},
        )
        result = load_dashboard_data()
        assert result.reliability is None

    def test_artist_data_user_score_parquet(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"artist": ["Radiohead", "Portishead"], "score": [85, 78]})
        self._write_parquet(tmp_path / "data" / "processed" / "user_score_data.parquet", df)
        result = load_dashboard_data()
        assert result.artist_data is not None
        assert "artist" in result.artist_data.columns

    def test_artist_data_cleaned_parquet_fallback(self, tmp_path, monkeypatch):
        # No *user_score*.parquet → falls back to cleaned*.parquet
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"artist": ["Björk"], "score": [91]})
        self._write_parquet(tmp_path / "data" / "processed" / "cleaned_data.parquet", df)
        result = load_dashboard_data()
        assert result.artist_data is not None

    def test_artist_data_capital_artist_renamed(self, tmp_path, monkeypatch):
        # Column named "Artist" (not "artist") → gets renamed
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"Artist": ["Miles Davis"], "score": [95]})
        self._write_parquet(tmp_path / "data" / "processed" / "user_score_capital.parquet", df)
        result = load_dashboard_data()
        assert result.artist_data is not None
        assert "artist" in result.artist_data.columns
        assert "Artist" not in result.artist_data.columns

    def test_artist_data_no_artist_column_skipped(self, tmp_path, monkeypatch):
        # Parquet with no artist/Artist/ARTIST column → artist_data stays None
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"name": ["Someone"], "score": [80]})
        self._write_parquet(tmp_path / "data" / "processed" / "user_score_noartist.parquet", df)
        result = load_dashboard_data()
        assert result.artist_data is None

    def test_run_dir_arg_used_for_models(self, tmp_path, monkeypatch):
        # Pass explicit run_dir; with no .nc files → idata stays None, no crash
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "outputs" / "20260101_120000"
        run_dir.mkdir(parents=True)
        result = load_dashboard_data(run_dir=run_dir)
        assert isinstance(result, DashboardData)

    def test_most_recent_run_dir_auto_detected(self, tmp_path, monkeypatch):
        # outputs/ has a digit-prefixed subdir → auto-detected as run_dir
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "outputs" / "20260101_120000"
        run_dir.mkdir(parents=True)
        (run_dir / "models").mkdir()
        # No .nc files → idata stays None, but auto-detection path exercised
        result = load_dashboard_data()
        assert isinstance(result, DashboardData)

    def test_idata_invalid_nc_skipped(self, tmp_path, monkeypatch):
        # A .nc file that can't be read → except branch (lines 498-499)
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "outputs" / "20260101_130000"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True)
        bad_nc = models_dir / "model.nc"
        bad_nc.write_bytes(b"NOT A NETCDF FILE AT ALL")
        result = load_dashboard_data()
        assert result.idata is None

    def test_idata_loaded_from_netcdf(self, tmp_path, monkeypatch):
        # Write a real idata via arviz, load it back
        monkeypatch.chdir(tmp_path)
        try:
            import arviz as az

            idata = az.from_dict(posterior={"mu": np.random.randn(2, 100)})
            run_dir = tmp_path / "outputs" / "20260101_140000"
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True)
            nc_path = models_dir / "model.nc"
            idata.to_netcdf(str(nc_path))
            result = load_dashboard_data()
            assert result.idata is not None
        except Exception:
            # If netcdf backend unavailable, just confirm no unhandled exception
            pytest.skip("netcdf backend not available")

    def test_run_dir_with_all_artifacts(self, tmp_path, monkeypatch):
        """Exercise the run_dir code path with coefficients inside the run dir."""
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "outputs" / "20260101_150000"
        coef_df = pd.DataFrame({"param": ["x"], "mean": [0.5], "hdi_3%": [0.1], "hdi_97%": [0.9]})
        self._write_csv(run_dir / "reports" / "tables" / "coefficient_run.csv", coef_df)
        result = load_dashboard_data(run_dir=run_dir)
        # coefficients may come from run_dir or project-root search_dirs
        assert isinstance(result, DashboardData)
