"""Expanded tests for reporting/figures.py: styles, colors, get_trace_plot_vars."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from panelcast.reporting.figures import (
    COLORBLIND_COLORS,
    get_trace_plot_vars,
    set_publication_style,
)


@pytest.fixture
def idata_basic():
    """Minimal InferenceData with user-prefixed parameters."""
    np.random.seed(42)
    posterior = xr.Dataset(
        {
            "user_mu_artist": (["chain", "draw"], np.random.randn(2, 50)),
            "user_sigma_artist": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
            "user_sigma_rw": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
            "user_sigma_obs": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
            "user_rho": (["chain", "draw"], np.random.randn(2, 50) * 0.3),
        }
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def idata_with_sigma_ref(idata_basic):
    """InferenceData with sigma_ref parameter."""
    idata_basic.posterior["user_sigma_ref"] = xr.DataArray(
        np.abs(np.random.randn(2, 50)),
        dims=["chain", "draw"],
    )
    return idata_basic


@pytest.fixture
def idata_with_n_exponent(idata_basic):
    """InferenceData with n_exponent parameter."""
    idata_basic.posterior["user_n_exponent"] = xr.DataArray(
        np.random.rand(2, 50),
        dims=["chain", "draw"],
    )
    return idata_basic


class TestColorblindColors:
    """Tests for COLORBLIND_COLORS constant."""

    def test_is_list(self):
        assert isinstance(COLORBLIND_COLORS, list)

    def test_has_seven_colors(self):
        assert len(COLORBLIND_COLORS) == 7

    def test_all_hex(self):
        for color in COLORBLIND_COLORS:
            assert color.startswith("#")
            assert len(color) == 7

    def test_all_unique(self):
        assert len(set(COLORBLIND_COLORS)) == 7


class TestSetPublicationStyle:
    """Tests for set_publication_style context manager."""

    def test_restores_rcparams(self):
        original_fontsize = plt.rcParams["font.size"]
        with set_publication_style():
            assert plt.rcParams["font.size"] == 9
        assert plt.rcParams["font.size"] == original_fontsize

    def test_sets_serif_font(self):
        with set_publication_style():
            assert plt.rcParams["font.family"] == ["serif"]

    def test_sets_savefig_dpi(self):
        with set_publication_style():
            assert plt.rcParams["savefig.dpi"] == 300

    def test_removes_top_spine(self):
        with set_publication_style():
            assert plt.rcParams["axes.spines.top"] is False

    def test_removes_right_spine(self):
        with set_publication_style():
            assert plt.rcParams["axes.spines.right"] is False

    def test_pdf_fonttype_42(self):
        with set_publication_style():
            assert plt.rcParams["pdf.fonttype"] == 42

    def test_figure_creation_inside(self):
        with set_publication_style():
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            plt.close(fig)

    def test_nested_context(self):
        with set_publication_style():
            with set_publication_style():
                assert plt.rcParams["font.size"] == 9
            assert plt.rcParams["font.size"] == 9


class TestGetTracePlotVars:
    """Tests for get_trace_plot_vars."""

    def test_basic_vars(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert "user_sigma_obs" in vars
        assert "user_rho" in vars

    def test_includes_hyperpriors(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_", include_hyperpriors=True)
        assert "user_mu_artist" in vars
        assert "user_sigma_artist" in vars
        assert "user_sigma_rw" in vars

    def test_excludes_hyperpriors(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_", include_hyperpriors=False)
        assert "user_mu_artist" not in vars
        assert "user_sigma_artist" not in vars

    def test_includes_sigma_ref(self, idata_with_sigma_ref):
        vars = get_trace_plot_vars(idata_with_sigma_ref, prefix="user_")
        assert "user_sigma_ref" in vars
        # sigma_ref should appear before sigma_obs
        ref_idx = vars.index("user_sigma_ref")
        obs_idx = vars.index("user_sigma_obs")
        assert ref_idx < obs_idx

    def test_no_sigma_ref_when_absent(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert "user_sigma_ref" not in vars

    def test_includes_n_exponent(self, idata_with_n_exponent):
        vars = get_trace_plot_vars(idata_with_n_exponent, prefix="user_")
        assert "user_n_exponent" in vars

    def test_no_n_exponent_when_absent(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert "user_n_exponent" not in vars

    def test_critic_prefix(self):
        posterior = xr.Dataset(
            {
                "critic_mu_artist": (["chain", "draw"], np.random.randn(2, 50)),
                "critic_sigma_artist": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
                "critic_sigma_rw": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
                "critic_sigma_obs": (["chain", "draw"], np.abs(np.random.randn(2, 50))),
                "critic_rho": (["chain", "draw"], np.random.randn(2, 50) * 0.3),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        vars = get_trace_plot_vars(idata, prefix="critic_")
        assert all(v.startswith("critic_") for v in vars)

    def test_returns_list(self, idata_basic):
        vars = get_trace_plot_vars(idata_basic, prefix="user_")
        assert isinstance(vars, list)
