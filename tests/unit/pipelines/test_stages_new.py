"""Additional stages tests targeting uncovered code paths.

Covers:
- Stage run_fn wrappers (_run_splits_stage, _run_features_stage, etc.)
- _resolve_raw_dataset_path edge cases
- Stage factory functions (make_stage_*) attributes
- PIPELINE_STAGES module-level list
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panelcast.pipelines.stages import (
    PIPELINE_STAGES,
    PipelineStage,
    StageContext,
    _resolve_raw_dataset_path,
    _run_data_stage,
    _run_evaluate_stage,
    _run_features_stage,
    _run_predict_stage,
    _run_report_stage,
    _run_splits_stage,
    _run_train_stage,
    _topological_sort,
    build_pipeline_stages,
    get_execution_order,
    get_stage,
    make_stage_data,
    make_stage_evaluate,
    make_stage_features,
    make_stage_predict,
    make_stage_report,
    make_stage_splits,
    make_stage_train,
)


def _make_ctx(tmp_path: Path, **overrides) -> StageContext:
    """Create a minimal StageContext for testing stage run functions."""
    defaults = dict(
        run_dir=tmp_path / "run",
        seed=42,
        strict=False,
        verbose=False,
        manifest=MagicMock(),
    )
    defaults.update(overrides)
    return StageContext(**defaults)


# ============================================================================
# _run_splits_stage
# ============================================================================


class TestRunSplitsStage:
    """Tests for _run_splits_stage wrapper."""

    def test_calls_create_splits_with_config(self, tmp_path):
        """_run_splits_stage passes seed and min_ratings to SplitConfig."""
        ctx = _make_ctx(tmp_path, seed=99, min_ratings=20)
        captured = {}

        def fake_create_splits(config):
            captured["config"] = config
            return MagicMock()

        with patch("panelcast.pipelines.create_splits.create_splits", fake_create_splits):
            _run_splits_stage(ctx)

        assert captured["config"].random_state == 99
        assert captured["config"].min_ratings == 20

    def test_returns_create_splits_result(self, tmp_path):
        """_run_splits_stage returns the create_splits result."""
        ctx = _make_ctx(tmp_path)
        expected_result = {"splits": "done"}

        with patch("panelcast.pipelines.create_splits.create_splits", return_value=expected_result):
            result = _run_splits_stage(ctx)

        assert result == expected_result


# ============================================================================
# _run_features_stage
# ============================================================================


class TestRunFeaturesStage:
    """Tests for _run_features_stage wrapper."""

    def test_calls_build_features_with_ctx(self, tmp_path):
        """_run_features_stage passes ctx to build_features."""
        ctx = _make_ctx(tmp_path)
        captured = {}

        def fake_build_features(ctx_arg):
            captured["ctx"] = ctx_arg
            return MagicMock()

        with patch("panelcast.pipelines.build_features.build_features", fake_build_features):
            _run_features_stage(ctx)

        assert captured["ctx"] is ctx


# ============================================================================
# _run_train_stage
# ============================================================================


class TestRunTrainStage:
    """Tests for _run_train_stage wrapper."""

    def test_calls_train_models_with_ctx(self, tmp_path):
        """_run_train_stage passes ctx to train_models."""
        ctx = _make_ctx(tmp_path)
        captured = {}

        def fake_train_models(ctx_arg):
            captured["ctx"] = ctx_arg
            return MagicMock()

        with patch("panelcast.pipelines.train_bayes.train_models", fake_train_models):
            _run_train_stage(ctx)

        assert captured["ctx"] is ctx


# ============================================================================
# _run_evaluate_stage
# ============================================================================


class TestRunEvaluateStage:
    """Tests for _run_evaluate_stage wrapper."""

    def test_calls_evaluate_models_with_ctx(self, tmp_path):
        """_run_evaluate_stage passes ctx to evaluate_models."""
        ctx = _make_ctx(tmp_path)
        captured = {}

        def fake_evaluate_models(ctx_arg):
            captured["ctx"] = ctx_arg
            return MagicMock()

        with patch("panelcast.pipelines.evaluate.evaluate_models", fake_evaluate_models):
            _run_evaluate_stage(ctx)

        assert captured["ctx"] is ctx


# ============================================================================
# _run_predict_stage
# ============================================================================


class TestRunPredictStage:
    """Tests for _run_predict_stage wrapper."""

    def test_calls_predict_next_albums_with_ctx(self, tmp_path):
        """_run_predict_stage passes ctx to predict_next_albums."""
        ctx = _make_ctx(tmp_path)
        captured = {}

        def fake_predict(ctx_arg):
            captured["ctx"] = ctx_arg
            return MagicMock()

        with patch("panelcast.pipelines.predict_next.predict_next_albums", fake_predict):
            _run_predict_stage(ctx)

        assert captured["ctx"] is ctx


# ============================================================================
# _run_report_stage
# ============================================================================


class TestRunReportStage:
    """Tests for _run_report_stage wrapper."""

    def test_calls_generate_publication_artifacts_with_ctx(self, tmp_path):
        """_run_report_stage passes ctx to generate_publication_artifacts."""
        ctx = _make_ctx(tmp_path)
        captured = {}

        def fake_generate(ctx_arg):
            captured["ctx"] = ctx_arg
            return MagicMock()

        with patch("panelcast.pipelines.publication.generate_publication_artifacts", fake_generate):
            _run_report_stage(ctx)

        assert captured["ctx"] is ctx


# ============================================================================
# _run_data_stage: output handling
# ============================================================================


class TestRunDataStageOutputs:
    """Tests for _run_data_stage output handling."""

    def test_returns_cleaned_dataset_path_when_available(self, tmp_path):
        """_run_data_stage includes cleaned dataset in output dict."""
        ctx = _make_ctx(tmp_path)
        (tmp_path / "run").mkdir(parents=True, exist_ok=True)

        mock_result = MagicMock()
        mock_result.datasets_created = {"cleaned_all": "/path/to/cleaned.parquet"}

        with patch(
            "panelcast.pipelines.prepare_dataset.prepare_datasets", return_value=mock_result
        ):
            result = _run_data_stage(ctx)

        assert "cleaned_dataset" in result
        assert result["cleaned_dataset"] == "/path/to/cleaned.parquet"
        assert "dataset_hash" in result

    def test_returns_only_hash_when_no_cleaned(self, tmp_path):
        """_run_data_stage omits cleaned_dataset when not in result."""
        ctx = _make_ctx(tmp_path)
        (tmp_path / "run").mkdir(parents=True, exist_ok=True)

        mock_result = MagicMock()
        mock_result.datasets_created = {}

        with patch(
            "panelcast.pipelines.prepare_dataset.prepare_datasets", return_value=mock_result
        ):
            result = _run_data_stage(ctx)

        assert "cleaned_dataset" not in result
        assert "dataset_hash" in result


# ============================================================================
# Make stage factory functions
# ============================================================================


class TestMakeStageFactories:
    """Tests for individual make_stage_* factory functions."""

    def test_make_stage_data(self):
        """make_stage_data creates correct stage."""
        stage = make_stage_data()
        assert stage.name == "data"
        assert stage.run_fn is not None
        assert stage.depends_on == []
        assert len(stage.output_paths) > 0

    def test_make_stage_splits_default(self):
        """make_stage_splits default min_ratings=10."""
        stage = make_stage_splits()
        assert stage.name == "splits"
        assert "minratings_10" in str(stage.input_paths[0])
        assert stage.depends_on == ["data"]

    def test_make_stage_splits_custom(self):
        """make_stage_splits with custom min_ratings."""
        stage = make_stage_splits(min_ratings=25)
        assert "minratings_25" in str(stage.input_paths[0])

    def test_make_stage_features(self):
        """make_stage_features creates correct stage."""
        stage = make_stage_features()
        assert stage.name == "features"
        assert stage.depends_on == ["splits"]
        assert len(stage.input_paths) >= 6
        assert len(stage.output_paths) >= 9

    def test_make_stage_train(self):
        """make_stage_train creates correct stage."""
        stage = make_stage_train()
        assert stage.name == "train"
        assert stage.depends_on == ["features"]
        assert stage.run_fn is not None

    def test_make_stage_evaluate(self):
        """make_stage_evaluate creates correct stage."""
        stage = make_stage_evaluate()
        assert stage.name == "evaluate"
        assert stage.depends_on == ["train"]

    def test_make_stage_predict(self):
        """make_stage_predict creates correct stage."""
        stage = make_stage_predict()
        assert stage.name == "predict"
        assert stage.depends_on == ["evaluate"]
        assert len(stage.output_paths) == 3

    def test_make_stage_report(self):
        """make_stage_report creates correct stage."""
        stage = make_stage_report()
        assert stage.name == "report"
        assert stage.depends_on == ["predict"]
        assert len(stage.output_paths) >= 7


# ============================================================================
# PIPELINE_STAGES module-level list
# ============================================================================


class TestPipelineStagesConstant:
    """Tests for PIPELINE_STAGES module-level list."""

    def test_pipeline_stages_is_list(self):
        """PIPELINE_STAGES is a list."""
        assert isinstance(PIPELINE_STAGES, list)

    def test_pipeline_stages_has_all_stages(self):
        """PIPELINE_STAGES contains all expected stages."""
        names = {s.name for s in PIPELINE_STAGES}
        assert names == {"data", "splits", "features", "train", "evaluate", "predict", "report"}

    def test_pipeline_stages_uses_default_min_ratings(self):
        """PIPELINE_STAGES uses default min_ratings=10."""
        splits = next(s for s in PIPELINE_STAGES if s.name == "splits")
        assert "minratings_10" in str(splits.input_paths[0])


# ============================================================================
# _resolve_raw_dataset_path
# ============================================================================


class TestResolveRawDatasetPath:
    """Tests for _resolve_raw_dataset_path."""

    def test_default_when_env_not_set(self, monkeypatch):
        """Returns default path when AOTY_DATASET_PATH not set."""
        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        path = _resolve_raw_dataset_path()
        assert path == Path("data/raw/all_albums_full.csv")

    def test_env_override(self, monkeypatch):
        """Environment variable overrides default."""
        monkeypatch.setenv("AOTY_DATASET_PATH", "/data/custom.csv")
        path = _resolve_raw_dataset_path()
        assert path == Path("/data/custom.csv")

    def test_env_with_relative_path(self, monkeypatch):
        """Relative path from env var is accepted."""
        monkeypatch.setenv("AOTY_DATASET_PATH", "my_data/file.csv")
        path = _resolve_raw_dataset_path()
        assert path == Path("my_data/file.csv")


# ============================================================================
# Topological sort with diamond dependency
# ============================================================================


class TestTopologicalSortDiamond:
    """Tests for _topological_sort with diamond dependency patterns."""

    def test_diamond_dependency(self):
        """Diamond dependency pattern is resolved correctly."""
        #   A
        #  / \
        # B   C
        #  \ /
        #   D
        stages = [
            PipelineStage(name="a", description="A", run_fn=None),
            PipelineStage(name="b", description="B", run_fn=None, depends_on=["a"]),
            PipelineStage(name="c", description="C", run_fn=None, depends_on=["a"]),
            PipelineStage(name="d", description="D", run_fn=None, depends_on=["b", "c"]),
        ]
        result = _topological_sort(stages)
        names = [s.name for s in result]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_filtered_diamond_subset(self):
        """Filtered subset of diamond resolves correctly."""
        stages = [
            PipelineStage(name="a", description="A", run_fn=None),
            PipelineStage(name="b", description="B", run_fn=None, depends_on=["a"]),
            PipelineStage(name="c", description="C", run_fn=None, depends_on=["a"]),
            PipelineStage(name="d", description="D", run_fn=None, depends_on=["b", "c"]),
        ]
        result = _topological_sort(stages, stage_names={"b", "c"})
        names = [s.name for s in result]
        assert set(names) == {"b", "c"}


# ============================================================================
# StageContext: all custom values
# ============================================================================


class TestStageContextCustomValues:
    """Tests for StageContext with all custom values."""

    def test_all_custom_params(self):
        """StageContext accepts all custom parameters."""
        ctx = StageContext(
            run_dir=Path("custom_run"),
            seed=99,
            strict=True,
            verbose=True,
            manifest=MagicMock(),
            max_albums=100,
            num_chains=8,
            num_samples=2000,
            num_warmup=500,
            target_accept=0.95,
            max_tree_depth=12,
            chain_method="parallel",
            rhat_threshold=1.05,
            ess_threshold=200,
            allow_divergences=True,
            min_ratings=20,
            min_albums_filter=5,
            enable_genre=False,
            enable_artist=False,
            enable_temporal=False,
            n_exponent=0.5,
            learn_n_exponent=True,
            n_exponent_alpha=3.0,
            n_exponent_beta=5.0,
            n_exponent_prior="beta",
            calibration_intervals=(0.50, 0.80, 0.95),
            coverage_tolerance=0.05,
            prediction_interval=0.90,
            evaluate_secondary_split=False,
        )
        assert ctx.seed == 99
        assert ctx.max_albums == 100
        assert ctx.chain_method == "parallel"
        assert ctx.enable_genre is False
        assert ctx.learn_n_exponent is True
        assert ctx.n_exponent_prior == "beta"
        assert ctx.calibration_intervals == (0.50, 0.80, 0.95)
        assert ctx.evaluate_secondary_split is False


# ============================================================================
# compute_input_hash: order-independent
# ============================================================================


class TestComputeInputHashOrder:
    """Tests for compute_input_hash ordering."""

    def test_hash_is_order_independent(self, tmp_path):
        """Hash is the same regardless of input_paths order."""
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        file_a.write_text("content_a")
        file_b.write_text("content_b")

        stage1 = PipelineStage(
            name="t1",
            description="T",
            run_fn=None,
            input_paths=[file_a, file_b],
        )
        stage2 = PipelineStage(
            name="t2",
            description="T",
            run_fn=None,
            input_paths=[file_b, file_a],
        )
        assert stage1.compute_input_hash() == stage2.compute_input_hash()

    def test_hash_ignores_missing_files(self, tmp_path):
        """Hash only considers existing files."""
        file_a = tmp_path / "a.csv"
        file_a.write_text("data")

        stage_with_missing = PipelineStage(
            name="t",
            description="T",
            run_fn=None,
            input_paths=[file_a, tmp_path / "missing.csv"],
        )
        stage_without_missing = PipelineStage(
            name="t",
            description="T",
            run_fn=None,
            input_paths=[file_a],
        )
        # Both should produce the same hash since missing files are skipped
        assert stage_with_missing.compute_input_hash() == stage_without_missing.compute_input_hash()


# ============================================================================
# get_execution_order with min_ratings
# ============================================================================


class TestGetExecutionOrderMinRatings:
    """Tests for get_execution_order with various min_ratings."""

    def test_min_ratings_5(self):
        """get_execution_order with min_ratings=5 affects splits input."""
        order = get_execution_order(["splits"], min_ratings=5)
        splits = order[0]
        assert "minratings_5" in str(splits.input_paths[0])

    def test_min_ratings_50(self):
        """get_execution_order with min_ratings=50 affects splits input."""
        order = get_execution_order(["splits"], min_ratings=50)
        splits = order[0]
        assert "minratings_50" in str(splits.input_paths[0])

    def test_all_stages_default_min_ratings(self):
        """All stages returned with default min_ratings."""
        order = get_execution_order(min_ratings=10)
        names = [s.name for s in order]
        assert names == ["data", "splits", "features", "train", "evaluate", "predict", "report"]
