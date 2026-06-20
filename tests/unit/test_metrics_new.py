"""New coverage tests for evaluation/metrics.py.

Targets uncovered code paths (95% -> higher):
- compute_point_metrics: NaN in y_true raises, NaN in y_pred raises,
  y_pred_mean not 1D raises, length mismatch raises
- compute_crps: list input conversion
- posterior_mean: 3D input raises
"""

import numpy as np
import pytest

from panelcast.evaluation.metrics import (
    compute_crps,
    compute_point_metrics,
    posterior_mean,
)


class TestComputePointMetricsNaN:
    """Cover NaN validation paths."""

    def test_nan_in_y_true_raises(self):
        """NaN in y_true should raise ValueError."""
        y_true = np.array([50.0, float("nan"), 70.0])
        y_pred = np.array([50.0, 60.0, 70.0])
        with pytest.raises(ValueError, match="NaN"):
            compute_point_metrics(y_true, y_pred)

    def test_nan_in_y_pred_raises(self):
        """NaN in y_pred_mean should raise ValueError."""
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([50.0, float("nan"), 70.0])
        with pytest.raises(ValueError, match="NaN"):
            compute_point_metrics(y_true, y_pred)


class TestComputePointMetricsValidation:
    """Cover validation error paths."""

    def test_y_pred_2d_raises(self):
        """2D y_pred_mean should raise ValueError."""
        y_true = np.array([50.0, 60.0])
        y_pred = np.array([[50.0], [60.0]])
        with pytest.raises(ValueError, match="y_pred_mean must be 1D"):
            compute_point_metrics(y_true, y_pred)

    def test_length_mismatch_raises(self):
        """Length mismatch between y_true and y_pred should raise."""
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([50.0, 60.0])
        with pytest.raises(ValueError, match="observations"):
            compute_point_metrics(y_true, y_pred)


class TestComputeCrpsListInput:
    """Cover automatic conversion from lists."""

    def test_list_inputs(self):
        """Lists should be converted to arrays."""
        y_true = [50.0, 60.0, 70.0]
        y_samples = [[48.0, 58.0, 68.0], [52.0, 62.0, 72.0]]
        result = compute_crps(y_true, y_samples)
        assert result.n_obs == 3
        assert result.crps_values.shape == (3,)


class TestPosteriorMean3D:
    """Cover 3D input validation."""

    def test_3d_input_raises(self):
        """3D input should raise ValueError."""
        y_samples = np.ones((10, 5, 3))
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            posterior_mean(y_samples)

    def test_0d_input_raises(self):
        """0D input should raise ValueError."""
        y_samples = np.array(5.0)
        with pytest.raises(ValueError, match="y_samples must be 2D"):
            posterior_mean(y_samples)
