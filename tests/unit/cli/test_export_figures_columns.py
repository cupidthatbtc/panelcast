"""_coefficient_columns resolves both arviz-summary and report coefficient tables."""

from __future__ import annotations

import pandas as pd

from panelcast.cli.commands import _coefficient_columns


def test_resolves_arviz_summary_columns():
    df = pd.DataFrame({"param": ["a"], "mean": [0.1], "hdi_3%": [-0.1], "hdi_97%": [0.3]})
    assert _coefficient_columns(df) == ("mean", "hdi_3%", "hdi_97%", "param")


def test_resolves_report_table_columns():
    # As read from reports/tables/coefficients.csv: the parameter name lands in
    # the unnamed first column; estimate/interval columns are Title Case.
    df = pd.DataFrame(
        {
            "Unnamed: 0": ["user_beta[0]"],
            "Estimate": [0.1],
            "SE": [0.7],
            "CI Lower": [-1.3],
            "CI Upper": [1.4],
        }
    )
    assert _coefficient_columns(df) == ("Estimate", "CI Lower", "CI Upper", "Unnamed: 0")


def test_returns_none_when_estimate_or_interval_missing():
    assert _coefficient_columns(pd.DataFrame({"foo": [1], "bar": [2]})) is None


def test_returns_none_for_non_dataframe():
    # export-figures can receive a plain dict (legacy/mocked data); must not crash.
    assert _coefficient_columns({"beta": [0.5, 0.3]}) is None
