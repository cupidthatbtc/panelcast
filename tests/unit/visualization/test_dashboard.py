"""Unit tests for visualization dashboard module."""

import numpy as np
import pandas as pd

from panelcast.visualization.dashboard import (
    DashboardData,
    _find_column,
    create_artist_view,
    create_coefficients_table,
    create_dashboard_figures,
    get_artist_list,
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
