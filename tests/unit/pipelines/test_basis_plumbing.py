import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.pipelines.build_features import build_features
from panelcast.pipelines.train_bayes import _build_basis_model_provenance


def test_basis_columns_and_fitted_state_flow_through_feature_stage(tmp_path, monkeypatch):
    train = pd.DataFrame(
        {"age": [20.0, 22.0, 24.0, 27.0, 31.0, 35.0, 38.0], "count": [10] * 7}
    )
    validation = pd.DataFrame({"age": [19.0, 100.0], "count": [10, 10]})
    test = pd.DataFrame({"age": [25.0, 30.0], "count": [10, 10]})
    splits_root = tmp_path / "data" / "splits"
    for split_name in ["within_entity_temporal", "entity_disjoint"]:
        split_dir = splits_root / split_name
        split_dir.mkdir(parents=True)
        train.to_parquet(split_dir / "train.parquet")
        validation.to_parquet(split_dir / "validation.parquet")
        test.to_parquet(split_dir / "test.parquet")

    descriptor = DatasetDescriptor(
        feature_packs=[],
        feature_blocks=[],
        n_obs_col="count",
        secondary_target_col=None,
        secondary_prefix=None,
        secondary_n_obs_col=None,
        basis_curves={"age_curve": {"col": "age", "type": "spline", "df": 5, "center": 27}},
    )
    ctx = SimpleNamespace(
        seed=42,
        enable_genre=True,
        enable_artist=True,
        enable_temporal=True,
        descriptor=descriptor,
    )
    monkeypatch.setattr("panelcast.pipelines.build_features.Path", lambda path: tmp_path / path)

    manifest = build_features(ctx)

    basis_names = [f"age_curve__basis_{i:02d}" for i in range(5)]
    assert all(name in manifest["feature_names"] for name in basis_names)
    assert manifest["blocks"] == ["basis"]
    fitted = manifest["basis_curves"]["fitted_by_split"]
    for split_name in ["within_entity_temporal", "entity_disjoint"]:
        assert fitted[split_name]["age_curve"]["train_max"] == 38.0
        assert fitted[split_name]["age_curve"]["feature_names"] == basis_names
    validation_features = pd.read_parquet(
        tmp_path / "data" / "features" / "within_entity_temporal" / "validation_features.parquet"
    )
    assert all(name in validation_features for name in basis_names)


def test_training_provenance_binds_basis_names_to_actual_scaler(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    names = ["other", "curve__basis_00", "curve__basis_01", "last"]
    state = {
        "spec": {"type": "spline", "col": "x", "df": 2},
        "feature_names": names[1:3],
    }
    manifest = {
        "legacy_primary_split": "within_entity_temporal",
        "basis_curves": {"fitted_by_split": {"within_entity_temporal": {"curve": state}}},
    }
    (feature_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    means = np.array([9.0, 1.5, -2.0, 7.0])
    stds = np.array([1.0, 0.25, 3.5, 2.0])

    provenance = _build_basis_model_provenance(
        feature_dir / "train_features.parquet", names, means, stds
    )

    assert provenance is not None
    standardization = provenance["curves"]["curve"]["standardization"]
    assert standardization == {
        "feature_names": names[1:3],
        "feature_indices": [1, 2],
        "mean": [1.5, -2.0],
        "std": [0.25, 3.5],
    }
