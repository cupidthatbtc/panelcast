"""Opt-in predictive snapshot persistence (#154).

``panelcast stack`` scores arm mixtures from thinned score-scale predictive
draws; the evaluate stage writes them only when ``PANELCAST_SAVE_PREDICTIVE``
is set.
"""

from pathlib import Path

import numpy as np

from panelcast.pipelines.evaluate import (
    _EVAL_OUTPUT_DIR,
    _PREDICTIVE_SNAPSHOT_DRAWS,
    _SAVE_PREDICTIVE_ENV,
    _append_predictive_snapshot,
    _predictive_save_path,
)


def test_save_path_unset_returns_none(monkeypatch):
    monkeypatch.delenv(_SAVE_PREDICTIVE_ENV, raising=False)
    assert _predictive_save_path() is None


def test_save_path_set_returns_eval_dir_npz(monkeypatch):
    monkeypatch.setenv(_SAVE_PREDICTIVE_ENV, "1")
    assert _predictive_save_path() == Path(_EVAL_OUTPUT_DIR) / "predictive.npz"


def test_snapshot_thins_to_cap_as_float32(tmp_path):
    path = tmp_path / "nested" / "predictive.npz"  # parent created on demand
    draws = np.arange(2000 * 3, dtype=np.float64).reshape(2000, 3)
    _append_predictive_snapshot(path, "primary", draws, np.array([1.0, 2.0, 3.0]))
    with np.load(path) as npz:
        saved = npz["primary_draws"]
        assert saved.shape == (_PREDICTIVE_SNAPSHOT_DRAWS, 3)
        assert saved.dtype == np.float32
        # Even thinning spans the whole chain, not its head.
        assert saved[0, 0] == 0.0
        assert saved[-1, 0] == draws[-1, 0]
        assert npz["primary_y_true"].tolist() == [1.0, 2.0, 3.0]


def test_small_sample_kept_whole(tmp_path):
    path = tmp_path / "predictive.npz"
    _append_predictive_snapshot(path, "primary", np.ones((40, 2)), np.zeros(2))
    with np.load(path) as npz:
        assert npz["primary_draws"].shape == (40, 2)


def test_second_split_appends_without_clobbering(tmp_path):
    path = tmp_path / "predictive.npz"
    _append_predictive_snapshot(path, "primary", np.ones((10, 2)), np.zeros(2))
    _append_predictive_snapshot(path, "secondary", np.full((10, 4), 2.0), np.zeros(4))
    with np.load(path) as npz:
        assert set(npz.files) == {
            "primary_draws", "primary_y_true", "secondary_draws", "secondary_y_true"
        }
        assert npz["primary_draws"].shape == (10, 2)
        assert npz["secondary_draws"].shape == (10, 4)


def test_fresh_write_drops_stale_splits(tmp_path):
    """A re-run's primary write must not resurrect a prior evaluation's secondary."""
    path = tmp_path / "predictive.npz"
    _append_predictive_snapshot(path, "primary", np.ones((10, 2)), np.zeros(2))
    _append_predictive_snapshot(path, "secondary", np.ones((10, 4)), np.zeros(4))
    _append_predictive_snapshot(path, "primary", np.full((10, 2), 3.0), np.zeros(2), fresh=True)
    with np.load(path) as npz:
        assert set(npz.files) == {"primary_draws", "primary_y_true"}
        assert float(npz["primary_draws"][0, 0]) == 3.0
