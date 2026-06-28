"""Opt-in pointwise log-likelihood persistence (issue #34).

The transform x latent bake-off needs each cell's pointwise log-likelihood on a
common test set to run a pairwise ``az.compare``; the evaluate stage writes it
only when ``PANELCAST_SAVE_LOG_LIKELIHOOD`` is set.
"""

from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from panelcast.pipelines.evaluate import (
    _EVAL_OUTPUT_DIR,
    _SAVE_LOG_LIKELIHOOD_ENV,
    _log_likelihood_save_path,
    _save_log_likelihood_idata,
)


def _tiny_log_lik_idata() -> az.InferenceData:
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(-2.0, 0.5, size=(2, 50, 30)),
        dims=["chain", "draw", "obs"],
        coords={"chain": range(2), "draw": range(50), "obs": range(30)},
    )
    return az.InferenceData(log_likelihood=xr.Dataset({"y": da}))


def test_save_path_unset_returns_none(monkeypatch):
    monkeypatch.delenv(_SAVE_LOG_LIKELIHOOD_ENV, raising=False)
    assert _log_likelihood_save_path() is None


def test_save_path_set_returns_eval_dir_netcdf(monkeypatch):
    monkeypatch.setenv(_SAVE_LOG_LIKELIHOOD_ENV, "1")
    assert _log_likelihood_save_path() == Path(_EVAL_OUTPUT_DIR) / "log_likelihood.nc"


def test_save_log_likelihood_idata_roundtrip(tmp_path):
    """The saved netCDF reloads with its log_likelihood group and values intact."""
    idata = _tiny_log_lik_idata()
    path = tmp_path / "nested" / "log_likelihood.nc"  # parent created on demand
    _save_log_likelihood_idata(idata, path)

    assert path.exists()
    reloaded = az.from_netcdf(str(path))
    assert "log_likelihood" in reloaded.groups()
    np.testing.assert_allclose(
        reloaded.log_likelihood["y"].values, idata.log_likelihood["y"].values
    )
