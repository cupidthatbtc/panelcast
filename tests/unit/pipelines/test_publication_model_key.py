"""The report stage must resolve the model key from the training summary.

Before the fix, generate_publication_artifacts hardcoded the "user_score"
manifest key (and the "user_" site prefix in its table/trace var_names), so
the report stage could never find a model fit under any non-AOTY descriptor.
"""

import json

from panelcast.pipelines.publication import _resolve_model_key


def test_resolves_prefix_from_dataset_block(tmp_path):
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"dataset": {"model_prefix": "perf"}}), encoding="utf-8"
    )
    assert _resolve_model_key(tmp_path) == ("perf_", "perf_score")


def test_missing_summary_falls_back_to_aoty_default(tmp_path):
    assert _resolve_model_key(tmp_path) == ("user_", "user_score")


def test_legacy_summary_without_dataset_block(tmp_path):
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"model_type": "user_score"}), encoding="utf-8"
    )
    assert _resolve_model_key(tmp_path) == ("user_", "user_score")


def test_corrupt_summary_falls_back(tmp_path):
    (tmp_path / "training_summary.json").write_text("{not json", encoding="utf-8")
    assert _resolve_model_key(tmp_path) == ("user_", "user_score")
