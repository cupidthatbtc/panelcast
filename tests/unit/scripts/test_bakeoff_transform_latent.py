"""Transform x latent bake-off reporting (issues #34, #63).

Covers the held-out elpd columns, the metrics reader, the `.nc`-derived elpd
recompute in the reassemble path, and the paired per-point `elpd_diff +/- dse`
vs the kept default. The script lives in ``scripts/`` (not an installed
package), so it is loaded by path.
"""

import importlib.util
import json
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import xarray as xr

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bakeoff_transform_latent.py"
_spec = importlib.util.spec_from_file_location("bakeoff_transform_latent", _SCRIPT)
bakeoff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bakeoff)


def _write_cell_metrics(d: Path, *, elpd, se, legacy: bool = False) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "diagnostics.json").write_text(
        json.dumps({"passed": True, "rhat_max": 1.005, "ess_bulk_min": 800, "divergences": 0})
    )
    if legacy:
        info = {"loo": {"elpd": elpd, "se": se, "p": 250.0, "pareto_k_max": 0.4}}
    else:
        info = {"heldout_elpd": {"elpd": elpd, "se": se, "n_obs": 40, "elpd_per_obs": elpd / 40}}
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "point_metrics": {"mae": 5.6, "rmse": 8.2, "r2": 0.42},
                "calibration": {"coverages": {"0.95": {"empirical": 0.95}}, "pit": {}},
                "crps": {"mean_crps": 4.1},
                "ppc": {"extreme_statistics": []},
                "info_criteria": info,
            }
        )
    )


def _write_log_lik_nc(path: Path, log_lik: np.ndarray) -> None:
    """Persist a (draws, obs) pointwise log-lik as a 1-chain idata netCDF."""
    da = xr.DataArray(
        log_lik[None, :, :],
        dims=["chain", "draw", "obs"],
        coords={"chain": [0], "draw": range(log_lik.shape[0]), "obs": range(log_lik.shape[1])},
    )
    az.InferenceData(log_likelihood=xr.Dataset({"y": da})).to_netcdf(str(path))


def test_read_metrics_pulls_heldout_elpd(tmp_path):
    _write_cell_metrics(tmp_path, elpd=-2486.0, se=29.7)
    out = bakeoff._read_metrics(tmp_path)
    assert out["elpd"] == -2486.0
    assert out["elpd_se"] == 29.7


def test_read_metrics_ignores_legacy_loo_payload(tmp_path):
    """Pre-#63 PSIS-LOO-on-test numbers must not resurface as elpd."""
    _write_cell_metrics(tmp_path, elpd=-2486.0, se=29.7, legacy=True)
    out = bakeoff._read_metrics(tmp_path)
    assert out["elpd"] is None
    assert out["elpd_se"] is None


def test_render_markdown_has_elpd_columns_and_pairwise():
    rows = [
        {
            "name": "identity_rw",
            "transform": "identity",
            "latent": "rw",
            "elpd": -2486.0,
            "elpd_se": 29.7,
        }
    ]
    pairwise = [{"name": "offset_logit_rw", "elpd_diff": 60.5, "dse": 8.0, "z": 7.56}]
    md = bakeoff._render_markdown(rows, pairwise)

    assert "| elpd | se |" in md
    assert "-2486.0" in md
    assert "Pairwise held-out elpd vs kept default (identity_rw)" in md
    assert "+60.5" in md
    assert "+7.56" in md


def test_render_markdown_notes_cells_absent_from_pairwise():
    """A scored cell with no pairwise row (no pointwise snapshot) is called out."""
    rows = [
        {"name": "identity_rw", "transform": "identity", "latent": "rw", "elpd": -2486.0},
        {
            "name": "offset_logit_ar1",
            "transform": "offset_logit",
            "latent": "ar1",
            "elpd": -2431.7,
        },
    ]
    pairwise = [{"name": "offset_logit_rw", "elpd_diff": 60.1, "dse": 4.6, "z": 13.07}]
    md = bakeoff._render_markdown(rows, pairwise)

    assert "offset_logit_ar1" in md
    assert "pairwise table omits them" in md


def test_pairwise_elpd_hand_computed(tmp_path):
    """Paired diffs: d_i = [log 2, 3 log 2] -> diff = 4 log 2, dse = 2 log 2, z = 2."""
    (tmp_path / "identity_rw").mkdir()
    (tmp_path / "offset_logit_rw").mkdir()
    # Default: density 1 everywhere -> elpd_i = [0, 0]
    _write_log_lik_nc(
        tmp_path / "identity_rw" / "log_likelihood.nc",
        np.zeros((2, 2)),
    )
    # Cell: densities 2 and 8 -> elpd_i = [log 2, 3 log 2]
    _write_log_lik_nc(
        tmp_path / "offset_logit_rw" / "log_likelihood.nc",
        np.log(np.array([[2.0, 8.0], [2.0, 8.0]])),
    )
    rows = [{"name": "identity_rw"}, {"name": "offset_logit_rw"}]

    pairwise = bakeoff._pairwise_elpd(rows, tmp_path)

    assert len(pairwise) == 1
    entry = pairwise[0]
    assert entry["name"] == "offset_logit_rw"
    assert entry["elpd_diff"] == pytest.approx(4 * np.log(2.0))
    assert entry["dse"] == pytest.approx(2 * np.log(2.0))
    assert entry["z"] == pytest.approx(2.0)


def test_pairwise_elpd_skips_mismatched_test_sets(tmp_path, capsys):
    """Different n_obs between cells -> the pair is skipped, not mis-scored."""
    (tmp_path / "identity_rw").mkdir()
    (tmp_path / "offset_logit_rw").mkdir()
    _write_log_lik_nc(tmp_path / "identity_rw" / "log_likelihood.nc", np.zeros((2, 3)))
    _write_log_lik_nc(tmp_path / "offset_logit_rw" / "log_likelihood.nc", np.zeros((2, 2)))
    rows = [{"name": "identity_rw"}, {"name": "offset_logit_rw"}]

    assert bakeoff._pairwise_elpd(rows, tmp_path) == []
    assert "pairwise diff failed" in capsys.readouterr().out


def test_pairwise_elpd_empty_without_default_idata(tmp_path):
    rows = [
        {"name": "identity_rw", "elpd": -100.0},
        {"name": "offset_logit_rw", "elpd": -90.0},
    ]
    assert bakeoff._pairwise_elpd(rows, tmp_path) == []


def test_collect_rows_prefers_nc_derived_elpd(tmp_path):
    """Reassembly recomputes elpd from the .nc snapshot; no-.nc cells stay None."""
    # identity_rw: legacy metrics.json + an .nc -> elpd comes from the .nc
    cell_dir = tmp_path / "identity_rw"
    _write_cell_metrics(cell_dir, elpd=-9999.0, se=1.0, legacy=True)
    _write_log_lik_nc(cell_dir / "log_likelihood.nc", np.log(np.array([[1.0, 1.0], [3.0, 7.0]])))
    # offset_logit_ar1: metrics.json only, no .nc -> elpd stays None
    _write_cell_metrics(tmp_path / "offset_logit_ar1", elpd=-8888.0, se=1.0, legacy=True)

    rows = {r["name"]: r for r in bakeoff._collect_rows_from_snapshots(tmp_path)}

    expected = float(np.log(2.0) + np.log(4.0))
    assert rows["identity_rw"]["elpd"] == pytest.approx(expected)
    assert rows["identity_rw"]["elpd_se"] > 0
    assert rows["offset_logit_ar1"]["elpd"] is None
