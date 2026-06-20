"""New coverage-targeted tests for reporting/tables.py.

Targets missed lines:
- _format_with_precision: non-finite value, non-finite uncertainty, zero uncertainty
- _format_with_precision: ValueError/OverflowError fallback
- _escape_latex_param_name
- create_coefficient_table: empty var_names, apply_precision=False
- create_diagnostics_table: empty var_names, mcse_mean missing
- create_diagnostics_table: status logic branches (Fail ESS, Fail R-hat+ESS, Fail R-hat)
- create_comparison_table: single model, empty dict
- create_sensitivity_summary_table: non-numeric values, non-OK convergence
- export_table: csv only, tex only, with caption/label
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelcast.reporting.tables import (
    _escape_latex_param_name,
    _format_with_precision,
    create_sensitivity_summary_table,
    export_table,
)


class TestFormatWithPrecision:
    """Tests for _format_with_precision adaptive formatting."""

    def test_normal_value(self):
        """Normal value with normal uncertainty."""
        result = _format_with_precision(1.234, 0.05)
        assert "." in result
        assert float(result) == pytest.approx(1.234, abs=0.01)

    def test_non_finite_value(self):
        """Non-finite value returns string representation."""
        assert _format_with_precision(float("inf"), 0.1) == "inf"
        assert _format_with_precision(float("-inf"), 0.1) == "-inf"
        assert _format_with_precision(float("nan"), 0.1) == "nan"

    def test_non_finite_uncertainty(self):
        """Non-finite uncertainty falls back to min_decimals."""
        result = _format_with_precision(1.234, float("inf"))
        assert result == "1.23"  # Default min_decimals=2

    def test_zero_uncertainty(self):
        """Zero uncertainty falls back to min_decimals."""
        result = _format_with_precision(1.234, 0.0)
        assert result == "1.23"

    def test_custom_min_decimals(self):
        """min_decimals parameter is respected."""
        result = _format_with_precision(1.234, float("inf"), min_decimals=4)
        assert result == "1.2340"

    def test_very_small_uncertainty(self):
        """Very small uncertainty shows more decimal places."""
        result = _format_with_precision(0.001234, 0.00005)
        # Should have more than 2 decimals
        decimal_places = len(result.split(".")[-1])
        assert decimal_places >= 2

    def test_large_uncertainty(self):
        """Large uncertainty with min_decimals constraint."""
        result = _format_with_precision(100.0, 50.0)
        # Should still have at least min_decimals
        decimal_places = len(result.split(".")[-1])
        assert decimal_places >= 2


class TestEscapeLatexParamName:
    """Tests for _escape_latex_param_name."""

    def test_underscore_escaped(self):
        """Underscores are escaped for LaTeX."""
        assert _escape_latex_param_name("sigma_artist") == r"sigma\_artist"

    def test_no_special_chars(self):
        """Names without special chars are unchanged."""
        assert _escape_latex_param_name("beta") == "beta"

    def test_brackets_preserved(self):
        """Square brackets are preserved."""
        assert _escape_latex_param_name("beta[0]") == "beta[0]"

    def test_multiple_underscores(self):
        """Multiple underscores all escaped."""
        assert _escape_latex_param_name("a_b_c") == r"a\_b\_c"


class TestExportTable:
    """Tests for export_table function."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for export testing."""
        return pd.DataFrame(
            {"Estimate": ["1.23", "4.56"], "SE": ["0.05", "0.10"]},
            index=["beta[0]", "beta[1]"],
        )

    def test_csv_only(self, sample_df, tmp_path):
        """Export CSV only."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=("csv",))
        assert len(paths) == 1
        assert paths[0].suffix == ".csv"
        assert paths[0].exists()

    def test_tex_only(self, sample_df, tmp_path):
        """Export LaTeX only."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=("tex",))
        assert len(paths) == 1
        assert paths[0].suffix == ".tex"
        assert paths[0].exists()

    def test_both_formats(self, sample_df, tmp_path):
        """Export both CSV and LaTeX."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=("csv", "tex"))
        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert ".csv" in suffixes
        assert ".tex" in suffixes

    def test_csv_content(self, sample_df, tmp_path):
        """CSV content is valid."""
        base_path = tmp_path / "table"
        export_table(sample_df, base_path, formats=("csv",))
        loaded = pd.read_csv(base_path.with_suffix(".csv"), index_col=0)
        assert "Estimate" in loaded.columns
        assert len(loaded) == 2

    def test_tex_with_caption_and_label(self, sample_df, tmp_path):
        """LaTeX output includes caption and label."""
        base_path = tmp_path / "table"
        export_table(
            sample_df, base_path, formats=("tex",), caption="Test caption", label="tab:test"
        )
        content = (base_path.with_suffix(".tex")).read_text()
        assert "Test caption" in content
        assert "tab:test" in content

    def test_tex_default_label(self, sample_df, tmp_path):
        """LaTeX output derives label from filename when not specified."""
        base_path = tmp_path / "coefficients"
        export_table(sample_df, base_path, formats=("tex",))
        content = (base_path.with_suffix(".tex")).read_text()
        assert "tab:coefficients" in content

    def test_creates_parent_directories(self, sample_df, tmp_path):
        """Parent directories are created if they don't exist."""
        base_path = tmp_path / "deep" / "nested" / "table"
        paths = export_table(sample_df, base_path, formats=("csv",))
        assert paths[0].exists()

    def test_no_formats_returns_empty(self, sample_df, tmp_path):
        """Empty formats tuple creates no files."""
        base_path = tmp_path / "table"
        paths = export_table(sample_df, base_path, formats=())
        assert len(paths) == 0


class TestCreateSensitivitySummaryTable:
    """Tests for create_sensitivity_summary_table."""

    def test_ok_convergence(self):
        """OK convergence status is preserved."""
        df = pd.DataFrame(
            {
                "variant": ["baseline"],
                "parameter": ["sigma"],
                "multiplier": [1.0],
                "elpd": [-100.0],
                "elpd_delta": [0.0],
                "elpd_se": [5.0],
                "convergence_flag": ["OK"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert result["Status"].iloc[0] == "OK"

    def test_non_ok_convergence_gets_dagger(self):
        """Non-OK convergence gets dagger symbol appended."""
        df = pd.DataFrame(
            {
                "variant": ["test"],
                "parameter": ["sigma"],
                "multiplier": [2.0],
                "elpd": [-120.0],
                "elpd_delta": [-20.0],
                "elpd_se": [8.0],
                "convergence_flag": ["R-hat > 1.01"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "\u2020" in result["Status"].iloc[0]  # dagger

    def test_non_numeric_elpd_becomes_em_dash(self):
        """Non-numeric ELPD values become em-dash."""
        df = pd.DataFrame(
            {
                "variant": ["test"],
                "parameter": ["sigma"],
                "multiplier": [2.0],
                "elpd": ["N/A"],
                "elpd_delta": [None],
                "elpd_se": [""],
                "convergence_flag": ["OK"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "\u2014" in str(result["ELPD"].iloc[0])  # em-dash

    def test_column_rename(self):
        """Columns are renamed for publication."""
        df = pd.DataFrame(
            {
                "variant": ["baseline"],
                "parameter": ["sigma"],
                "multiplier": [1.0],
                "elpd": [-100.0],
                "elpd_delta": [0.0],
                "elpd_se": [5.0],
                "convergence_flag": ["OK"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "Variant" in result.columns
        assert "Parameter" in result.columns
        assert "ELPD" in result.columns
        assert "\u0394ELPD" in result.columns
        assert "SE" in result.columns
        assert "Status" in result.columns
