from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.features.base import FeatureContext
from panelcast.features.basis import BasisBlock

CTX = FeatureContext(config={}, random_state=42)
SPEC = {"curves": {"age_curve": {"col": "age", "type": "spline", "df": 5, "center": 27.0}}}


def test_deterministic_names_and_state():
    train = pd.DataFrame({"age": [20.0, 22.0, 24.0, 27.0, 31.0, 35.0, 40.0]})
    first = BasisBlock(SPEC).fit(train, CTX)
    second = BasisBlock(SPEC).fit(train, CTX)
    assert first.fitted_state == second.fitted_state
    assert first.fitted_state["age_curve"]["feature_names"] == [
        f"age_curve__basis_{i:02d}" for i in range(5)
    ]
    state = first.fitted_state["age_curve"]
    assert len(state["retained_basis_indices"]) == 5
    assert state["dropped_basis_index"] not in state["retained_basis_indices"]
    matrix = first.transform(train, CTX).data.to_numpy()
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    assert standardized.shape == (7, 5)
    assert np.linalg.matrix_rank(standardized) == 5


def test_transform_reuses_train_knots_without_leakage():
    train = pd.DataFrame({"age": [20.0, 22.0, 25.0, 30.0, 35.0]})
    validation = pd.DataFrame({"age": [21.0, 1000.0]})
    block = BasisBlock(SPEC).fit(train, CTX)
    state_before = block.fitted_state.copy()
    transformed = block.transform(validation, CTX).data
    assert block.fitted_state == state_before
    assert block.fitted_state["age_curve"]["train_max"] == 35.0
    assert np.isfinite(transformed.to_numpy()).all()


def test_same_values_transform_identically_across_splits():
    train = pd.DataFrame({"age": [20.0, 23.0, 27.0, 31.0, 35.0]})
    block = BasisBlock(SPEC).fit(train, CTX)
    left = block.transform(pd.DataFrame({"age": [25.0, 30.0]}), CTX).data
    right = block.transform(pd.DataFrame({"age": [30.0, 25.0]}), CTX).data.iloc[::-1]
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy())


def test_repeated_boundary_quantile_adapts_dimension_to_full_rank():
    values = [0.0] * 8 + [1.0, 2.0, 3.0, 4.0]
    block = BasisBlock(SPEC).fit(pd.DataFrame({"age": values}), CTX)
    state = block.fitted_state["age_curve"]
    matrix = block.transform(pd.DataFrame({"age": values}), CTX).data.to_numpy()

    assert state["requested_df"] == 5
    assert state["fitted_df"] == 4
    assert state["feature_names"] == [f"age_curve__basis_{i:02d}" for i in range(4)]
    assert matrix.shape[1] == len(state["feature_names"])
    centered = matrix - matrix.mean(axis=0)
    assert np.linalg.matrix_rank(centered) == matrix.shape[1]
    assert not np.any(np.all(matrix == 0.0, axis=0))


def test_repeated_interior_quantiles_adapt_dimension_and_order():
    spec = {"curves": {"age_curve": {"col": "age", "type": "spline", "df": 6, "center": 0.0}}}
    values = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]
    block = BasisBlock(spec).fit(pd.DataFrame({"age": values}), CTX)
    state = block.fitted_state["age_curve"]
    output = block.transform(pd.DataFrame({"age": values}), CTX)

    assert state["requested_df"] == 6
    assert state["fitted_df"] == 4
    assert output.feature_names == state["feature_names"]
    assert list(output.data.columns) == state["feature_names"]
    matrix = output.data.to_numpy()
    centered = matrix - matrix.mean(axis=0)
    assert np.linalg.matrix_rank(centered) == state["fitted_df"]


@pytest.mark.parametrize("values", [[1.0, np.nan], [1.0, np.inf], [2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
def test_rejects_unusable_training_values(values):
    with pytest.raises(ValueError, match="finite|constant|identifiable"):
        BasisBlock(SPEC).fit(pd.DataFrame({"age": values}), CTX)
