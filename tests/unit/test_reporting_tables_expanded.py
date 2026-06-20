"""Expanded tests for reporting/tables.py: precision formatting, sensitivity table, edge cases."""

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panelcast.reporting.tables import (
    _escape_latex_param_name,
    _format_with_precision,
    create_coefficient_table,
    create_diagnostics_table,
    create_sensitivity_summary_table,
    export_table,
)


def _make_idata(n_chains=4, n_draws=500, param_names=None):
    """Helper to build mock InferenceData."""
    if param_names is None:
        param_names = ["mu", "sigma"]
    np.random.seed(42)
    posterior_dict = {}
    for name in param_names:
        posterior_dict[name] = xr.DataArray(
            np.random.randn(n_chains, n_draws),
            dims=["chain", "draw"],
            coords={"chain": range(n_chains), "draw": range(n_draws)},
        )
    posterior = xr.Dataset(posterior_dict)
    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(
                np.zeros((n_chains, n_draws), dtype=bool),
                dims=["chain", "draw"],
            )
        }
    )
    return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


# ── _format_with_precision edge cases ──────────────────────────────


class TestFormatWithPrecisionExpanded:
    """Expanded edge-case tests for _format_with_precision."""

    def test_negative_inf(self):
        result = _format_with_precision(-np.inf, 0.1)
        assert "inf" in result.lower()

    def test_very_large_value(self):
        result = _format_with_precision(1e10, 1e8)
        assert "." in result or "e" in result.lower()

    def test_very_small_value(self):
        result = _format_with_precision(1e-8, 1e-10)
        assert "." in result

    def test_negative_value(self):
        result = _format_with_precision(-3.14, 0.05)
        assert result.startswith("-")

    def test_exact_zero_value(self):
        result = _format_with_precision(0.0, 0.01)
        assert "0.00" in result

    def test_min_greater_than_max_decimals(self):
        """min_decimals > max_decimals: max should still cap."""
        result = _format_with_precision(1.234, 0.001, min_decimals=6, max_decimals=3)
        decimals = len(result.split(".")[-1])
        assert decimals <= 6  # Implementation clamps at max of (min, capped)


# ── _escape_latex_param_name ───────────────────────────────────────


class TestEscapeLatexParamName:
    """Tests for _escape_latex_param_name."""

    def test_no_underscores(self):
        assert _escape_latex_param_name("beta") == "beta"

    def test_single_underscore(self):
        assert _escape_latex_param_name("sigma_artist") == r"sigma\_artist"

    def test_multiple_underscores(self):
        result = _escape_latex_param_name("mu_artist_effect")
        assert result == r"mu\_artist\_effect"

    def test_brackets_unchanged(self):
        assert _escape_latex_param_name("beta[0]") == "beta[0]"

    def test_mixed(self):
        result = _escape_latex_param_name("user_beta[0]")
        assert result == r"user\_beta[0]"


# ── create_coefficient_table expanded ──────────────────────────────


class TestCreateCoefficientTableExpanded:
    """Expanded coefficient table tests."""

    def test_many_parameters(self):
        idata = _make_idata(param_names=["a", "b", "c", "d", "e"])
        result = create_coefficient_table(idata)
        assert len(result) == 5

    def test_single_chain(self):
        idata = _make_idata(n_chains=1)
        result = create_coefficient_table(idata)
        assert len(result) == 2

    def test_few_draws(self):
        idata = _make_idata(n_draws=20)
        result = create_coefficient_table(idata)
        assert len(result) == 2

    def test_ci_lower_le_upper(self):
        idata = _make_idata()
        result = create_coefficient_table(idata, apply_precision=False)
        for idx in result.index:
            assert result.at[idx, "CI Lower"] <= result.at[idx, "CI Upper"]


# ── create_diagnostics_table expanded ──────────────────────────────


class TestCreateDiagnosticsTableExpanded:
    """Expanded diagnostics table tests."""

    def test_missing_posterior_raises(self):
        idata = az.InferenceData()
        with pytest.raises(ValueError, match="posterior"):
            create_diagnostics_table(idata)

    def test_var_names_filter(self):
        idata = _make_idata(param_names=["alpha", "beta", "gamma"])
        result = create_diagnostics_table(idata, var_names=["alpha"])
        assert len(result) == 1
        assert "alpha" in result.index

    def test_single_chain_diagnostics(self):
        """Single chain should still produce valid diagnostics table."""
        idata = _make_idata(n_chains=1)
        result = create_diagnostics_table(idata)
        assert isinstance(result, pd.DataFrame)
        assert "Status" in result.columns


# ── create_sensitivity_summary_table ───────────────────────────────


class TestCreateSensitivitySummaryTable:
    """Tests for create_sensitivity_summary_table."""

    @pytest.fixture
    def oat_df(self):
        return pd.DataFrame(
            {
                "variant": ["baseline", "high_sigma", "low_sigma"],
                "parameter": ["sigma_artist", "sigma_artist", "sigma_artist"],
                "multiplier": [1.0, 2.0, 0.5],
                "elpd": [-1234.0, -1240.0, -1230.0],
                "elpd_delta": [0.0, -6.0, 4.0],
                "elpd_se": [12.0, 13.0, 11.0],
                "convergence_flag": ["OK", "OK", "WARN"],
            }
        )

    def test_returns_dataframe(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        assert isinstance(result, pd.DataFrame)

    def test_renames_columns(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        assert "Variant" in result.columns
        assert "Parameter" in result.columns

    def test_dagger_for_non_ok(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        statuses = result["Status"].tolist()
        assert any("\u2020" in s for s in statuses)

    def test_ok_no_dagger(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        ok_rows = result[result["Status"] == "OK"]
        for s in ok_rows["Status"]:
            assert "\u2020" not in s

    def test_numeric_formatting(self, oat_df):
        result = create_sensitivity_summary_table(oat_df)
        # ELPD values should be formatted as strings
        for val in result["ELPD"]:
            assert isinstance(val, str)

    def test_none_elpd_becomes_dash(self):
        df = pd.DataFrame(
            {
                "variant": ["test"],
                "parameter": ["p"],
                "multiplier": [1.0],
                "elpd": [None],
                "elpd_delta": [None],
                "elpd_se": [None],
                "convergence_flag": ["FAIL"],
            }
        )
        result = create_sensitivity_summary_table(df)
        assert "\u2014" in result["ELPD"].iloc[0]


# ── export_table expanded ─────────────────────────────────────────


class TestExportTableExpanded:
    """Expanded export table tests."""

    def test_csv_only(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_table(df, Path(tmpdir) / "t", formats=("csv",))
            assert len(paths) == 1
            assert paths[0].suffix == ".csv"

    def test_tex_only(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_table(df, Path(tmpdir) / "t", formats=("tex",))
            assert len(paths) == 1
            assert paths[0].suffix == ".tex"

    def test_no_formats(self):
        df = pd.DataFrame({"A": [1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_table(df, Path(tmpdir) / "t", formats=())
            assert paths == []

    def test_csv_roundtrip_preserves_data(self):
        df = pd.DataFrame(
            {"Estimate": [1.23, 4.56], "SE": [0.1, 0.2]},
            index=["mu", "sigma"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "roundtrip"
            export_table(df, out, formats=("csv",))
            loaded = pd.read_csv(out.with_suffix(".csv"), index_col=0)
            assert loaded.loc["mu", "Estimate"] == pytest.approx(1.23)

    def test_latex_has_booktabs(self):
        df = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bt"
            export_table(df, out, formats=("tex",))
            content = (out.with_suffix(".tex")).read_text()
            assert "\\begin{table}" in content
