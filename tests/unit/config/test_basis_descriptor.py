import pytest
from pydantic import ValidationError

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.build_features import get_feature_blocks


def test_basis_capability_is_default_off():
    descriptor = DatasetDescriptor()
    assert descriptor.basis_curves == {}
    assert "basis" not in [block.name for block in get_feature_blocks(descriptor=descriptor)]


def test_declared_curve_adds_basis_block_and_provenance():
    descriptor = DatasetDescriptor(
        basis_curves={
            "age_curve": {"col": "age", "type": "spline", "df": 5, "center": 27}
        }
    )
    block = get_feature_blocks(descriptor=descriptor)[-1]
    assert block.name == "basis"
    assert block.curves["age_curve"]["df"] == 5
    assert descriptor.to_summary_block()["basis_curves"]["age_curve"]["center"] == 27.0


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"col": "age", "type": "natural", "df": 5}, "type"),
        ({"col": "age", "type": "spline", "df": 3}, "greater than or equal"),
        ({"col": "age", "type": "spline", "df": 5, "bogus": 1}, "Extra inputs"),
        ({"col": "", "type": "spline", "df": 5}, "at least 1 character"),
        ({"col": "age", "type": "spline", "df": 5, "center": float("inf")}, "finite"),
    ],
)
def test_malformed_or_unsupported_specs_are_rejected(spec, message):
    with pytest.raises(ValidationError, match=message):
        DatasetDescriptor(basis_curves={"age_curve": spec})


def test_invalid_curve_name_is_rejected():
    with pytest.raises(ValidationError, match="basis_curves names"):
        DatasetDescriptor(
            basis_curves={"age-curve": {"col": "age", "type": "spline", "df": 5}}
        )
