import numpy as np
import pandas as pd
import pytest

from panelcast.features.base import FeatureContext
from panelcast.features.basis import BasisBlock
from panelcast.reporting.curves import (
    PosteriorCurve,
    basis_matrix,
    extract_curve_draws,
    extract_posterior_curve,
    standardized_basis_matrix,
    summarize_curve_peak,
)


def _state():
    block = BasisBlock(
        {"curves": {"age_curve": {"col": "age", "type": "spline", "df": 5, "center": 27.0}}}
    )
    block.fit(
        pd.DataFrame({"age": [18.0, 20.0, 22.0, 27.0, 33.0, 36.0, 40.0]}),
        FeatureContext(config={}, random_state=42),
    )
    state = block.fitted_state["age_curve"]
    state["standardization"] = {
        "feature_names": state["feature_names"],
        "feature_indices": list(range(len(state["feature_names"]))),
        "mean": [0.0] * len(state["feature_names"]),
        "std": [1.0] * len(state["feature_names"]),
    }
    return state


def test_extract_curve_draws_rebuilds_manifest_basis():
    state = _state()
    coefficients = np.arange(15.0).reshape(3, 5)
    grid = np.array([20.0, 27.0, 35.0])
    curve = extract_curve_draws(coefficients, state, grid=grid)
    np.testing.assert_allclose(curve.draws, coefficients @ basis_matrix(grid, state).T)
    np.testing.assert_array_equal(curve.x, grid)


def test_extract_curve_draws_matches_standardized_design_multiplication():
    state = _state()
    means = np.array([0.1, 0.3, -0.2, 0.7, 0.05])
    stds = np.array([0.5, 1.25, 2.0, 0.8, 3.0])
    state["standardization"]["mean"] = means.tolist()
    state["standardization"]["std"] = stds.tolist()
    coefficients = np.array([[1.0, -2.0, 0.5, 3.0, -0.25]])
    grid = np.array([19.0, 25.0, 34.0, 39.0])

    curve = extract_curve_draws(coefficients, state, grid=grid)

    standardized_design = (basis_matrix(grid, state) - means) / stds
    np.testing.assert_allclose(curve.draws, coefficients @ standardized_design.T)


def test_extract_posterior_curve_selects_named_columns_and_flattens_chains():
    state = _state()
    names = ["other", *state["feature_names"], "last"]
    state["standardization"]["feature_indices"] = list(range(1, 6))
    draws = np.zeros((2, 3, len(names)))
    draws[..., 1:6] = 1.0
    curve = extract_posterior_curve(draws, names, state, grid_size=7)
    assert curve.draws.shape == (6, 7)


def test_peak_summary_reports_vertices_intervals_and_boundaries():
    x = np.array([0.0, 1.0, 2.0])
    draws = np.array([[0.0, 3.0, 1.0], [5.0, 2.0, 0.0], [0.0, 1.0, 4.0]])
    summary = summarize_curve_peak(PosteriorCurve(x=x, draws=draws), credible_mass=0.8)
    np.testing.assert_array_equal(summary.vertex_draws, [1.0, 0.0, 2.0])
    assert summary.vertex_median == 1.0
    assert summary.value_median == 4.0
    assert summary.boundary_fraction == pytest.approx(2 / 3)


def test_extractor_rejects_coefficient_shape_mismatch():
    with pytest.raises(ValueError, match="n_basis=5"):
        extract_curve_draws(np.ones((10, 4)), _state())


def test_extractor_rejects_pretraining_state_without_scaler():
    state = _state()
    del state["standardization"]
    with pytest.raises(ValueError, match="training_summary.json"):
        extract_curve_draws(np.ones((2, 5)), state)


def test_curve_validation_rejects_corrupt_inputs_and_state():
    with pytest.raises(ValueError, match="one-dimensional"):
        basis_matrix([[20.0]], _state())
    with pytest.raises(ValueError, match="finite"):
        basis_matrix([np.nan], _state())

    state = _state()
    state["spec"]["type"] = "natural"
    with pytest.raises(ValueError, match="Unsupported"):
        basis_matrix([20.0], state)

    state = _state()
    state["feature_names"].append("extra")
    with pytest.raises(ValueError, match="declares"):
        basis_matrix([20.0], state)

    state = _state()
    state["standardization"]["feature_names"] = list(reversed(state["feature_names"]))
    with pytest.raises(ValueError, match="ordering"):
        standardized_basis_matrix([20.0], state)

    state = _state()
    state["standardization"]["mean"].pop()
    with pytest.raises(ValueError, match="mean/std"):
        standardized_basis_matrix([20.0], state)

    state = _state()
    state["standardization"]["std"][0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        standardized_basis_matrix([20.0], state)

    with pytest.raises(ValueError, match="finite"):
        extract_curve_draws(np.full((2, 5), np.nan), _state())
    with pytest.raises(ValueError, match="grid_size"):
        extract_curve_draws(np.ones((2, 5)), _state(), grid_size=1)

    state = _state()
    with pytest.raises(ValueError, match="missing basis"):
        extract_posterior_curve(np.ones((2, 1)), ["other"], state)
    names = [*state["feature_names"], "other"]
    with pytest.raises(ValueError, match="last dimension"):
        extract_posterior_curve(np.ones((2, 5)), names, state)

    state["standardization"]["feature_indices"] = [-1, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="valid fitted-model"):
        extract_posterior_curve(np.ones((2, 6)), names, state)
    state["standardization"]["feature_indices"] = [1, 0, 2, 3, 4]
    with pytest.raises(ValueError, match="ordering"):
        extract_posterior_curve(np.ones((2, 6)), names, state)

    curve = PosteriorCurve(x=np.array([0.0, 1.0]), draws=np.ones((2, 2)))
    with pytest.raises(ValueError, match="direction"):
        summarize_curve_peak(curve, direction="left")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="credible_mass"):
        summarize_curve_peak(curve, credible_mass=1.0)
    with pytest.raises(ValueError, match="shape"):
        summarize_curve_peak(PosteriorCurve(x=np.array([0.0]), draws=np.ones((2, 2))))
