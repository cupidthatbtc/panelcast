"""Tests for pipeline stage definitions and execution order."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from panelcast.pipelines.stages import (
    PipelineStage,
    StageContext,
    _run_data_stage,
    build_pipeline_stages,
    get_execution_order,
    get_stage,
)


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_basic_creation(self):
        """PipelineStage can be created with required fields."""
        stage = PipelineStage(
            name="test",
            description="Test stage",
            run_fn=None,
        )
        assert stage.name == "test"
        assert stage.description == "Test stage"
        assert stage.run_fn is None
        assert stage.input_paths == []
        assert stage.output_paths == []
        assert stage.depends_on == []

    def test_with_paths(self, tmp_path: Path):
        """PipelineStage stores input and output paths."""
        stage = PipelineStage(
            name="data",
            description="Process data",
            run_fn=None,
            input_paths=[tmp_path / "input.csv"],
            output_paths=[tmp_path / "output.parquet"],
            depends_on=["prior_stage"],
        )
        assert len(stage.input_paths) == 1
        assert len(stage.output_paths) == 1
        assert stage.depends_on == ["prior_stage"]


class TestComputeInputHash:
    """Tests for PipelineStage.compute_input_hash."""

    def test_empty_when_no_inputs(self):
        """Returns empty string when no input paths defined."""
        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[],
        )
        assert stage.compute_input_hash() == ""

    def test_empty_when_inputs_not_exist(self, tmp_path: Path):
        """Returns empty string when input files don't exist."""
        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[tmp_path / "nonexistent.csv"],
        )
        assert stage.compute_input_hash() == ""

    def test_hash_when_input_exists(self, tmp_path: Path):
        """Returns SHA256 hash when input file exists."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("col1,col2\n1,2\n3,4\n")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )
        hash_value = stage.compute_input_hash()

        # SHA256 is 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_changes_with_content(self, tmp_path: Path):
        """Hash changes when file content changes."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("content_v1")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )
        hash_v1 = stage.compute_input_hash()

        input_file.write_text("content_v2")
        hash_v2 = stage.compute_input_hash()

        assert hash_v1 != hash_v2

    def test_hash_deterministic(self, tmp_path: Path):
        """Same content produces same hash."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("consistent content")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )

        assert stage.compute_input_hash() == stage.compute_input_hash()

    def test_hash_combines_multiple_files(self, tmp_path: Path):
        """Hash combines multiple input files."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text("content1")
        file2.write_text("content2")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[file1, file2],
        )
        hash_both = stage.compute_input_hash()

        # Different from single file hash
        stage_single = PipelineStage(
            name="test2",
            description="Test",
            run_fn=None,
            input_paths=[file1],
        )
        hash_single = stage_single.compute_input_hash()

        assert hash_both != hash_single

    def test_hash_detects_swapped_file_contents(self, tmp_path: Path):
        """Swapping the contents of two inputs must change the combined hash."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text("content_a")
        file2.write_text("content_b")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[file1, file2],
        )
        hash_before = stage.compute_input_hash()

        file1.write_text("content_b")
        file2.write_text("content_a")
        hash_after = stage.compute_input_hash()

        assert hash_before != hash_after


class TestShouldSkip:
    """Tests for PipelineStage.should_skip."""

    def test_no_skip_when_force_true(self, tmp_path: Path):
        """Never skip when force=True."""
        stage = PipelineStage(name="test", description="Test", run_fn=None)
        # Even with a manifest, force=True means no skip
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": "somehash"}

        assert stage.should_skip(mock_manifest, force=True) is False

    def test_no_skip_when_manifest_none(self):
        """No skip when no previous manifest."""
        stage = PipelineStage(name="test", description="Test", run_fn=None)

        assert stage.should_skip(None) is False

    def test_no_skip_when_stage_not_in_manifest(self):
        """No skip when stage not recorded in manifest."""
        stage = PipelineStage(name="new_stage", description="Test", run_fn=None)
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"other_stage": "hash"}

        assert stage.should_skip(mock_manifest) is False

    def test_no_skip_when_hash_changed(self, tmp_path: Path):
        """No skip when input hash has changed."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("new_content")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )

        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": "old_hash_value"}

        assert stage.should_skip(mock_manifest) is False

    def test_no_skip_when_outputs_missing(self, tmp_path: Path):
        """No skip when output files don't exist."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("content")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
            output_paths=[tmp_path / "missing_output.parquet"],
        )

        # Compute actual hash to match
        actual_hash = stage.compute_input_hash()
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": actual_hash}

        # Should not skip because output doesn't exist
        assert stage.should_skip(mock_manifest) is False

    def test_skip_when_all_conditions_met(self, tmp_path: Path):
        """Skip when hash matches and outputs exist."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("content")
        output_file = tmp_path / "output.parquet"
        output_file.write_text("output")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
            output_paths=[output_file],
        )

        # Compute actual hash
        actual_hash = stage.compute_input_hash()
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": actual_hash}

        assert stage.should_skip(mock_manifest) is True


class TestPipelineStages:
    """Tests for build_pipeline_stages registry."""

    def test_all_stages_defined(self):
        """All expected stages are defined."""
        stages = build_pipeline_stages()
        stage_names = {s.name for s in stages}
        expected = {"data", "splits", "features", "train", "evaluate", "predict", "report"}
        assert stage_names == expected

    def test_stages_have_descriptions(self):
        """All stages have non-empty descriptions."""
        for stage in build_pipeline_stages():
            assert stage.description, f"Stage {stage.name} missing description"

    def test_stages_have_valid_dependencies(self):
        """All stage dependencies reference existing stages."""
        stages = build_pipeline_stages()
        valid_names = {s.name for s in stages}
        for stage in stages:
            for dep in stage.depends_on:
                assert dep in valid_names, f"Stage {stage.name} has invalid dependency: {dep}"

    def test_data_stage_has_no_dependencies(self):
        """Data stage is the root with no dependencies."""
        data_stage = get_stage("data")
        assert data_stage.depends_on == []

    def test_data_stage_input_path_uses_env_override(self, monkeypatch):
        """Data stage input hash should track custom dataset path from env."""
        monkeypatch.setenv("AOTY_DATASET_PATH", "custom/raw.csv")
        data_stage = get_stage("data")
        assert data_stage.input_paths == [Path("custom/raw.csv")]

    def test_report_stage_depends_on_predict(self):
        """Report stage depends on predict."""
        report_stage = get_stage("report")
        assert "predict" in report_stage.depends_on

    def test_predict_stage_depends_on_evaluate(self):
        """Predict stage depends on evaluate."""
        predict_stage = get_stage("predict")
        assert "evaluate" in predict_stage.depends_on

    def test_splits_input_path_reflects_min_ratings(self):
        """Splits stage input_paths use the correct min_ratings parquet file."""
        from pathlib import Path

        stages_10 = build_pipeline_stages(min_ratings=10)
        stages_30 = build_pipeline_stages(min_ratings=30)

        splits_10 = next(s for s in stages_10 if s.name == "splits")
        splits_30 = next(s for s in stages_30 if s.name == "splits")

        assert splits_10.input_paths == [Path("data/processed/user_score_minratings_10.parquet")]
        assert splits_30.input_paths == [Path("data/processed/user_score_minratings_30.parquet")]

    def test_predict_stage_has_run_fn(self):
        """Predict stage has a callable run_fn."""
        predict_stage = get_stage("predict")
        assert predict_stage.run_fn is not None
        assert callable(predict_stage.run_fn)

    def test_predict_stage_input_output_paths(self):
        """Predict stage has correct input and output paths."""
        predict_stage = get_stage("predict")
        input_names = [p.name for p in predict_stage.input_paths]
        assert "manifest.json" in input_names
        assert "training_summary.json" in input_names
        output_names = [p.name for p in predict_stage.output_paths]
        assert "next_album_known_artists.csv" in output_names
        assert "next_album_new_artist.csv" in output_names
        assert "prediction_summary.json" in output_names


class TestGetExecutionOrder:
    """Tests for get_execution_order function."""

    def test_returns_all_stages_by_default(self):
        """Returns all stages when no filter provided."""
        order = get_execution_order()
        assert len(order) == len(build_pipeline_stages())

    def test_respects_dependencies(self):
        """Stages come after their dependencies."""
        order = get_execution_order()
        names = [s.name for s in order]

        # data before splits
        assert names.index("data") < names.index("splits")
        # splits before features
        assert names.index("splits") < names.index("features")
        # features before train
        assert names.index("features") < names.index("train")
        # train before evaluate
        assert names.index("train") < names.index("evaluate")
        # evaluate before predict
        assert names.index("evaluate") < names.index("predict")
        # predict before report
        assert names.index("predict") < names.index("report")

    def test_filters_to_specified_stages(self):
        """Returns only specified stages when filter provided."""
        order = get_execution_order(["data", "splits"])
        names = [s.name for s in order]

        assert len(names) == 2
        assert set(names) == {"data", "splits"}

    def test_filtered_stages_in_order(self):
        """Filtered stages still respect dependency order."""
        order = get_execution_order(["splits", "data"])  # Reverse order in input
        names = [s.name for s in order]

        # Still returns in correct order
        assert names.index("data") < names.index("splits")

    def test_unknown_stage_raises_keyerror(self):
        """Raises KeyError for unknown stage name."""
        with pytest.raises(KeyError) as exc_info:
            get_execution_order(["nonexistent"])

        assert "nonexistent" in str(exc_info.value)

    def test_empty_list_returns_empty(self):
        """Empty stage list returns empty result."""
        order = get_execution_order([])
        assert order == []


class TestDataStageStrictSchema:
    """Tests strict-mode schema behavior in data stage."""

    @staticmethod
    def _make_ctx(tmp_path: Path, strict: bool) -> StageContext:
        """Create a minimal StageContext for calling stage run functions directly."""
        return StageContext(
            run_dir=tmp_path / "run",
            seed=42,
            strict=strict,
            verbose=False,
            manifest=MagicMock(),
        )

    def test_strict_mode_enables_raw_schema_validation(self, tmp_path: Path):
        """Data stage should enable raw schema validation when strict=True."""
        ctx = self._make_ctx(tmp_path, strict=True)
        mock_result = MagicMock(datasets_created={})

        with pytest.MonkeyPatch.context() as mp:
            import panelcast.pipelines.prepare_dataset as prepare_dataset_mod

            called = {}

            def _capture_prepare(config):
                called["config"] = config
                return mock_result

            mp.setattr(prepare_dataset_mod, "prepare_datasets", _capture_prepare)
            _run_data_stage(ctx)

        assert called["config"].validate_raw_schema is True

    def test_non_strict_mode_keeps_raw_schema_validation_relaxed(self, tmp_path: Path):
        """Data stage should keep schema validation disabled when strict=False."""
        ctx = self._make_ctx(tmp_path, strict=False)
        mock_result = MagicMock(datasets_created={})

        with pytest.MonkeyPatch.context() as mp:
            import panelcast.pipelines.prepare_dataset as prepare_dataset_mod

            called = {}

            def _capture_prepare(config):
                called["config"] = config
                return mock_result

            mp.setattr(prepare_dataset_mod, "prepare_datasets", _capture_prepare)
            _run_data_stage(ctx)

        assert called["config"].validate_raw_schema is False


class TestGetStage:
    """Tests for get_stage function."""

    def test_finds_existing_stage(self):
        """Returns stage when name exists."""
        stage = get_stage("data")
        assert stage.name == "data"
        assert isinstance(stage, PipelineStage)

    def test_unknown_stage_raises_keyerror(self):
        """Raises KeyError for unknown stage."""
        with pytest.raises(KeyError) as exc_info:
            get_stage("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        # Error message should list valid stages
        assert "data" in str(exc_info.value)

    def test_all_defined_stages_findable(self):
        """All defined stages can be found by name."""
        for stage in build_pipeline_stages():
            found = get_stage(stage.name)
            assert found.name == stage.name

    def test_get_stage_with_custom_min_ratings(self):
        """get_stage respects min_ratings parameter."""
        stage = get_stage("splits", min_ratings=25)
        assert any("minratings_25" in str(p) for p in stage.input_paths)


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestTopologicalSort:
    """Tests for _topological_sort edge cases."""

    def test_cycle_detection(self):
        """Circular dependencies raise ValueError."""
        from panelcast.pipelines.stages import _topological_sort

        stage_a = PipelineStage(name="a", description="A", run_fn=None, depends_on=["b"])
        stage_b = PipelineStage(name="b", description="B", run_fn=None, depends_on=["a"])
        with pytest.raises(ValueError, match="Circular dependency"):
            _topological_sort([stage_a, stage_b])

    def test_unknown_dependency_raises(self):
        """Missing dependency raises ValueError."""
        from panelcast.pipelines.stages import _topological_sort

        stage = PipelineStage(
            name="orphan", description="Test", run_fn=None, depends_on=["nonexistent"]
        )
        with pytest.raises(ValueError, match="unknown stage"):
            _topological_sort([stage])

    def test_single_stage_no_deps(self):
        """Single stage with no dependencies returns as-is."""
        from panelcast.pipelines.stages import _topological_sort

        stage = PipelineStage(name="solo", description="Alone", run_fn=None)
        result = _topological_sort([stage])
        assert len(result) == 1
        assert result[0].name == "solo"

    def test_empty_stages_list(self):
        """Empty stages list returns empty result."""
        from panelcast.pipelines.stages import _topological_sort

        result = _topological_sort([])
        assert result == []

    def test_filtered_stage_names(self):
        """Stage names filter only includes requested stages."""
        from panelcast.pipelines.stages import _topological_sort

        stage_a = PipelineStage(name="a", description="A", run_fn=None)
        stage_b = PipelineStage(name="b", description="B", run_fn=None, depends_on=["a"])
        stage_c = PipelineStage(name="c", description="C", run_fn=None, depends_on=["b"])
        result = _topological_sort([stage_a, stage_b, stage_c], stage_names={"a", "b"})
        names = [s.name for s in result]
        assert "c" not in names
        assert "a" in names
        assert "b" in names

    def test_linear_chain_order(self):
        """Linear dependency chain is sorted correctly."""
        from panelcast.pipelines.stages import _topological_sort

        stages = [
            PipelineStage(name="first", description="1", run_fn=None),
            PipelineStage(name="second", description="2", run_fn=None, depends_on=["first"]),
            PipelineStage(name="third", description="3", run_fn=None, depends_on=["second"]),
        ]
        result = _topological_sort(stages)
        names = [s.name for s in result]
        assert names.index("first") < names.index("second")
        assert names.index("second") < names.index("third")

    def test_three_node_cycle(self):
        """Three-node cycle is detected."""
        from panelcast.pipelines.stages import _topological_sort

        stages = [
            PipelineStage(name="a", description="A", run_fn=None, depends_on=["c"]),
            PipelineStage(name="b", description="B", run_fn=None, depends_on=["a"]),
            PipelineStage(name="c", description="C", run_fn=None, depends_on=["b"]),
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            _topological_sort(stages)


class TestStageContextDefaults:
    """Tests for StageContext default values."""

    def test_default_mcmc_config(self):
        """StageContext has sensible MCMC defaults."""
        ctx = StageContext(
            run_dir=Path("test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.num_chains == 4
        assert ctx.num_samples == 1000
        assert ctx.num_warmup == 1000
        assert ctx.target_accept == 0.90
        assert ctx.max_tree_depth == 10
        assert ctx.chain_method == "sequential"

    def test_default_convergence_thresholds(self):
        """StageContext has expected convergence defaults."""
        ctx = StageContext(
            run_dir=Path("test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.rhat_threshold == 1.01
        assert ctx.ess_threshold == 400
        assert ctx.allow_divergences is False

    def test_default_feature_flags(self):
        """StageContext feature flags default to True."""
        ctx = StageContext(
            run_dir=Path("test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.enable_genre is True
        assert ctx.enable_artist is True
        assert ctx.enable_temporal is True

    def test_default_heteroscedastic_params(self):
        """StageContext heteroscedastic defaults."""
        ctx = StageContext(
            run_dir=Path("test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.n_exponent == 0.0
        assert ctx.learn_n_exponent is False
        assert ctx.n_exponent_prior == "logit-normal"

    def test_default_evaluation_params(self):
        """StageContext evaluation defaults."""
        ctx = StageContext(
            run_dir=Path("test"),
            seed=42,
            strict=False,
            verbose=False,
            manifest=MagicMock(),
        )
        assert ctx.calibration_intervals == (0.80, 0.95)
        assert ctx.coverage_tolerance == 0.03
        assert ctx.prediction_interval == 0.95
        assert ctx.evaluate_secondary_split is True

    def test_custom_values(self):
        """StageContext accepts custom values."""
        ctx = StageContext(
            run_dir=Path("custom"),
            seed=99,
            strict=True,
            verbose=True,
            manifest=MagicMock(),
            max_albums=100,
            num_chains=8,
            n_exponent=0.5,
            learn_n_exponent=True,
        )
        assert ctx.seed == 99
        assert ctx.strict is True
        assert ctx.max_albums == 100
        assert ctx.num_chains == 8
        assert ctx.n_exponent == 0.5
        assert ctx.learn_n_exponent is True


class TestBuildPipelineStages:
    """Additional tests for build_pipeline_stages."""

    def test_min_ratings_5(self):
        """build_pipeline_stages with min_ratings=5."""
        stages = build_pipeline_stages(min_ratings=5)
        splits = next(s for s in stages if s.name == "splits")
        assert any("minratings_5" in str(p) for p in splits.input_paths)

    def test_all_stages_have_run_fn_or_none(self):
        """All stages have run_fn that is callable or None."""
        for stage in build_pipeline_stages():
            assert stage.run_fn is None or callable(stage.run_fn)

    def test_all_stages_have_non_empty_name(self):
        """All stages have non-empty names."""
        for stage in build_pipeline_stages():
            assert stage.name
            assert isinstance(stage.name, str)

    def test_stage_names_are_unique(self):
        """No duplicate stage names."""
        stages = build_pipeline_stages()
        names = [s.name for s in stages]
        assert len(names) == len(set(names))


class TestShouldSkipEdgeCases:
    """Additional edge cases for should_skip."""

    def test_skip_with_no_input_paths_and_no_output_paths(self):
        """Stage with no inputs and no outputs skips when hash matches empty."""
        stage = PipelineStage(name="test", description="Test", run_fn=None)
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": ""}
        # No inputs = hash is "", matches manifest, and no outputs to check
        assert stage.should_skip(mock_manifest) is True

    def test_skip_partial_outputs_missing(self, tmp_path):
        """Stage does not skip if some output files are missing."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("data")
        output1 = tmp_path / "out1.parquet"
        output1.write_text("exists")
        # output2 doesn't exist

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
            output_paths=[output1, tmp_path / "out2_missing.parquet"],
        )
        actual_hash = stage.compute_input_hash()
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": actual_hash}

        assert stage.should_skip(mock_manifest) is False


class TestResolveRawDatasetPath:
    """Tests for _resolve_raw_dataset_path."""

    def test_default_path(self, monkeypatch):
        """Default path when env var not set."""
        from panelcast.pipelines.stages import _resolve_raw_dataset_path

        monkeypatch.delenv("AOTY_DATASET_PATH", raising=False)
        path = _resolve_raw_dataset_path()
        assert path == Path("data/raw/all_albums_full.csv")

    def test_env_override(self, monkeypatch):
        """Environment variable overrides default path."""
        from panelcast.pipelines.stages import _resolve_raw_dataset_path

        monkeypatch.setenv("AOTY_DATASET_PATH", "/custom/data.csv")
        path = _resolve_raw_dataset_path()
        assert path == Path("/custom/data.csv")


class TestGetExecutionOrderEdgeCases:
    """Additional edge cases for get_execution_order."""

    def test_single_stage(self):
        """Single stage request returns single stage."""
        order = get_execution_order(["data"])
        assert len(order) == 1
        assert order[0].name == "data"

    def test_multiple_unknown_stages_raises(self):
        """Multiple unknown stage names raise KeyError."""
        with pytest.raises(KeyError):
            get_execution_order(["fake1", "fake2"])

    def test_min_ratings_propagates(self):
        """min_ratings parameter affects returned stages."""
        order = get_execution_order(["splits"], min_ratings=25)
        splits_stage = next(s for s in order if s.name == "splits")
        assert any("minratings_25" in str(p) for p in splits_stage.input_paths)
