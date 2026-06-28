"""Transform x latent bake-off reporting (issue #34).

Covers the new LOO columns, the metrics reader, and the pairwise ``az.compare``
(elpd_diff +/- dse vs the kept default). The script lives in ``scripts/`` (not an
installed package), so it is loaded by path.
"""

import importlib.util
import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bakeoff_transform_latent.py"
_spec = importlib.util.spec_from_file_location("bakeoff_transform_latent", _SCRIPT)
bakeoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bakeoff)


def _write_cell_metrics(d: Path, *, loo_elpd, se, p, waic_elpd) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "diagnostics.json").write_text(
        json.dumps({"passed": True, "rhat_max": 1.005, "ess_bulk_min": 800, "divergences": 0})
    )
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "point_metrics": {"mae": 5.6, "rmse": 8.2, "r2": 0.42},
                "calibration": {"coverages": {"0.95": {"empirical": 0.95}}, "pit": {}},
                "crps": {"mean_crps": 4.1},
                "ppc": {"extreme_statistics": []},
                "info_criteria": {
                    "loo": {
                        "elpd": loo_elpd,
                        "se": se,
                        "p": p,
                        "pareto_k_max": 0.4,
                        "pareto_k_gt_0_7": 0,
                    },
                    "waic": {"elpd": waic_elpd},
                },
            }
        )
    )


def _log_lik_idata(seed: int, n_obs: int = 40) -> az.InferenceData:
    rng = np.random.default_rng(seed)
    da = xr.DataArray(
        rng.normal(-2.0, 0.5, size=(2, 80, n_obs)),
        dims=["chain", "draw", "obs"],
        coords={"chain": range(2), "draw": range(80), "obs": range(n_obs)},
    )
    post = xr.Dataset(
        {"beta": xr.DataArray(rng.normal(size=(2, 80, 3)), dims=["chain", "draw", "beta_dim"])}
    )
    return az.InferenceData(posterior=post, log_likelihood=xr.Dataset({"y": da}))


def test_read_metrics_pulls_loo_se_p_waic(tmp_path):
    _write_cell_metrics(tmp_path, loo_elpd=-2486.0, se=29.7, p=258.3, waic_elpd=-2485.7)
    out = bakeoff._read_metrics(tmp_path)
    assert out["loo_elpd"] == -2486.0
    assert out["loo_se"] == 29.7
    assert out["p_loo"] == 258.3
    assert out["waic_elpd"] == -2485.7


def test_render_markdown_has_loo_columns_and_pairwise():
    rows = [
        {
            "name": "identity_rw",
            "transform": "identity",
            "latent": "rw",
            "loo_elpd": -2486.0,
            "loo_se": 29.7,
            "p_loo": 258.3,
            "waic_elpd": -2485.7,
        }
    ]
    pairwise = [{"name": "offset_logit_rw", "elpd_diff": 60.5, "dse": 8.0, "z": 7.56}]
    md = bakeoff._render_markdown(rows, pairwise)

    assert "| loo | se | p_loo | waic |" in md
    assert "-2486.0" in md
    assert "Pairwise LOO vs kept default (identity_rw)" in md
    assert "+60.5" in md
    assert "+7.56" in md


def test_render_markdown_notes_cells_absent_from_pairwise():
    """A scored cell with no pairwise row (no pointwise snapshot) is called out."""
    rows = [
        {"name": "identity_rw", "transform": "identity", "latent": "rw", "loo_elpd": -2486.0},
        {
            "name": "offset_logit_ar1",
            "transform": "offset_logit",
            "latent": "ar1",
            "loo_elpd": -2431.7,
        },
    ]
    pairwise = [{"name": "offset_logit_rw", "elpd_diff": 60.1, "dse": 4.6, "z": 13.07}]
    md = bakeoff._render_markdown(rows, pairwise)

    assert "offset_logit_ar1" in md
    assert "pairwise table omits them" in md


def test_pairwise_loo_computes_diff_and_dse(tmp_path):
    """Two persisted log-lik idata -> a signed elpd_diff and a positive paired dse."""
    (tmp_path / "identity_rw").mkdir()
    (tmp_path / "offset_logit_rw").mkdir()
    _log_lik_idata(1).to_netcdf(str(tmp_path / "identity_rw" / "log_likelihood.nc"))
    _log_lik_idata(2).to_netcdf(str(tmp_path / "offset_logit_rw" / "log_likelihood.nc"))
    rows = [
        {"name": "identity_rw", "loo_elpd": -100.0},
        {"name": "offset_logit_rw", "loo_elpd": -90.0},
    ]

    pairwise = bakeoff._pairwise_loo(rows, tmp_path)

    assert len(pairwise) == 1
    entry = pairwise[0]
    assert entry["name"] == "offset_logit_rw"
    assert entry["elpd_diff"] == 10.0  # cell minus default
    assert entry["dse"] > 0
    assert entry["z"] == 10.0 / entry["dse"]


def test_pairwise_loo_empty_without_default_idata(tmp_path):
    rows = [
        {"name": "identity_rw", "loo_elpd": -100.0},
        {"name": "offset_logit_rw", "loo_elpd": -90.0},
    ]
    assert bakeoff._pairwise_loo(rows, tmp_path) == []
