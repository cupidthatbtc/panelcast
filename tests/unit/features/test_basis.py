from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelcast.features.base import FeatureContext
from panelcast.features.basis import BasisBlock

CTX = FeatureContext(config={}, random_state=42)
SPEC = {"curves": {"age_curve": {"col": "age", "type": "spline", "df": 5, "center": 27.0}}}


def test_deterministic_names_and_state():
    train = pd.DataFrame({"age": [20.0, 24.0, 27.0, 31.0, 40.0]})
    first = BasisBlock(SPEC).fit(train, CTX)
    second = BasisBlock(SPEC).fit(train, CTX)
    assert first.fitted_state == second.fitted_state
    assert first.fitted_state["age_curve"]["feature_names"] == [
        f"age_curve__basis_{i:02d}" for i in range(5)
    ]
    assert first.transform(train, CTX).data.shape == (5, 5)


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


@pytest.mark.parametrize("values", [[1.0, np.nan], [1.0, np.inf], [2.0, 2.0]])
def test_rejects_unusable_training_values(values):
    with pytest.raises(ValueError, match="finite|constant"):
        BasisBlock(SPEC).fit(pd.DataFrame({"age": values}), CTX)
