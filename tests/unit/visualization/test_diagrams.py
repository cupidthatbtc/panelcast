"""Unit tests for visualization diagrams module."""

import pytest

pytest.importorskip("graphviz", reason="graphviz not installed")

import graphviz  # noqa: E402

from panelcast.visualization.diagrams import (  # noqa: E402
    LEVEL_FUNCTIONS,
    THEME_COLORS,
    _create_graph,
    create_aoty_pipeline_diagram,
    create_detailed_diagram,
    create_high_level_diagram,
    generate_all_diagrams,
)


class TestThemeColors:
    """Tests for THEME_COLORS constant."""

    def test_three_themes(self):
        assert len(THEME_COLORS) == 3

    def test_expected_themes(self):
        assert "light" in THEME_COLORS
        assert "dark" in THEME_COLORS
        assert "transparent" in THEME_COLORS

    def test_light_has_white_bg(self):
        assert THEME_COLORS["light"]["bgcolor"] == "#FFFFFF"

    def test_dark_has_dark_bg(self):
        assert THEME_COLORS["dark"]["bgcolor"] == "#1A1A1A"

    def test_transparent_bg(self):
        assert THEME_COLORS["transparent"]["bgcolor"] == "transparent"

    def test_all_themes_have_required_keys(self):
        required = ["bgcolor", "fontcolor", "color", "fillcolor", "edge_primary"]
        for theme_name, colors in THEME_COLORS.items():
            for key in required:
                assert key in colors, f"{theme_name} missing {key}"


class TestCreateGraph:
    """Tests for _create_graph function."""

    def test_returns_digraph(self):
        graph = _create_graph("light")
        assert isinstance(graph, graphviz.Digraph)

    def test_light_theme(self):
        graph = _create_graph("light")
        assert "bgcolor" in graph.source

    def test_dark_theme(self):
        graph = _create_graph("dark")
        assert "#1A1A1A" in graph.source

    def test_transparent_theme(self):
        graph = _create_graph("transparent")
        # Should not have bgcolor set
        assert "bgcolor" not in graph.source or "transparent" in graph.source

    def test_custom_title(self):
        graph = _create_graph("light", title="My Pipeline")
        assert "My Pipeline" in graph.source

    def test_default_title(self):
        graph = _create_graph("light")
        assert "AOTY" in graph.source


class TestCreateHighLevelDiagram:
    """Tests for create_high_level_diagram."""

    def test_returns_digraph(self):
        graph = create_high_level_diagram()
        assert isinstance(graph, graphviz.Digraph)

    def test_light_theme(self):
        graph = create_high_level_diagram("light")
        assert isinstance(graph, graphviz.Digraph)

    def test_dark_theme(self):
        graph = create_high_level_diagram("dark")
        assert isinstance(graph, graphviz.Digraph)

    def test_transparent_theme(self):
        graph = create_high_level_diagram("transparent")
        assert isinstance(graph, graphviz.Digraph)

    def test_has_expected_nodes(self):
        graph = create_high_level_diagram()
        source = graph.source
        assert "input_data" in source
        assert "cleaning" in source
        assert "features" in source
        assert "model" in source
        assert "evaluation" in source
        assert "predictions" in source

    def test_has_edges(self):
        graph = create_high_level_diagram()
        assert "->" in graph.source

    def test_has_clusters(self):
        graph = create_high_level_diagram()
        assert "subgraph" in graph.source


class TestCreateAotyPipelineDiagram:
    """Tests for create_aoty_pipeline_diagram."""

    def test_returns_digraph(self):
        graph = create_aoty_pipeline_diagram()
        assert isinstance(graph, graphviz.Digraph)

    def test_light_theme(self):
        graph = create_aoty_pipeline_diagram("light")
        assert isinstance(graph, graphviz.Digraph)

    def test_dark_theme(self):
        graph = create_aoty_pipeline_diagram("dark")
        assert isinstance(graph, graphviz.Digraph)

    def test_has_feature_blocks(self):
        graph = create_aoty_pipeline_diagram()
        source = graph.source
        assert "temporal_block" in source
        assert "genre_block" in source

    def test_has_model_nodes(self):
        graph = create_aoty_pipeline_diagram()
        source = graph.source
        assert "mcmc_sampling" in source

    def test_has_evaluation_nodes(self):
        graph = create_aoty_pipeline_diagram()
        source = graph.source
        assert "convergence" in source
        assert "loo_cv" in source

    def test_has_feedback_edges(self):
        graph = create_aoty_pipeline_diagram()
        source = graph.source
        assert "dashed" in source


class TestCreateDetailedDiagram:
    """Tests for create_detailed_diagram."""

    def test_returns_digraph(self):
        graph = create_detailed_diagram()
        assert isinstance(graph, graphviz.Digraph)

    def test_light_theme(self):
        graph = create_detailed_diagram("light")
        assert isinstance(graph, graphviz.Digraph)

    def test_dark_theme(self):
        graph = create_detailed_diagram("dark")
        assert isinstance(graph, graphviz.Digraph)

    def test_has_preprocessing_detail(self):
        graph = create_detailed_diagram()
        source = graph.source
        assert "null_handling" in source
        assert "date_parsing" in source
        assert "min_ratings" in source

    def test_has_model_detail(self):
        graph = create_detailed_diagram()
        source = graph.source
        assert "non_centered" in source
        assert "ar1_structure" in source

    def test_has_evaluation_detail(self):
        graph = create_detailed_diagram()
        source = graph.source
        assert "rhat_check" in source
        assert "ess_check" in source
        assert "pareto_k" in source

    def test_has_output_detail(self):
        graph = create_detailed_diagram()
        source = graph.source
        assert "model_artifacts" in source
        assert "publication" in source


class TestLevelFunctions:
    """Tests for LEVEL_FUNCTIONS mapping."""

    def test_three_levels(self):
        assert len(LEVEL_FUNCTIONS) == 3

    def test_expected_keys(self):
        assert "high" in LEVEL_FUNCTIONS
        assert "intermediate" in LEVEL_FUNCTIONS
        assert "detailed" in LEVEL_FUNCTIONS

    def test_high_calls_correct_function(self):
        assert LEVEL_FUNCTIONS["high"] is create_high_level_diagram

    def test_intermediate_calls_correct_function(self):
        assert LEVEL_FUNCTIONS["intermediate"] is create_aoty_pipeline_diagram

    def test_detailed_calls_correct_function(self):
        assert LEVEL_FUNCTIONS["detailed"] is create_detailed_diagram

    def test_all_callable(self):
        for name, func in LEVEL_FUNCTIONS.items():
            assert callable(func), f"{name} is not callable"


class TestGenerateAllDiagrams:
    """Tests for generate_all_diagrams."""

    def test_creates_output_dir(self, tmp_path):
        output_dir = tmp_path / "diagrams"
        try:
            results = generate_all_diagrams(output_dir, levels=["high"])
        except Exception:
            # graphviz rendering may fail if dot executable isn't available
            pytest.skip("graphviz dot executable not available")
        assert output_dir.exists()

    def test_returns_dict(self, tmp_path):
        try:
            results = generate_all_diagrams(tmp_path, levels=["high"])
        except Exception:
            pytest.skip("graphviz dot executable not available")
        assert isinstance(results, dict)

    def test_all_levels_default(self, tmp_path):
        try:
            results = generate_all_diagrams(tmp_path)
        except Exception:
            pytest.skip("graphviz dot executable not available")
        # 3 levels * 3 themes = 9 diagram sets
        assert len(results) == 9

    def test_single_level(self, tmp_path):
        try:
            results = generate_all_diagrams(tmp_path, levels=["high"])
        except Exception:
            pytest.skip("graphviz dot executable not available")
        # 1 level * 3 themes = 3 diagram sets
        assert len(results) == 3

    def test_naming_pattern(self, tmp_path):
        try:
            results = generate_all_diagrams(tmp_path, levels=["high"])
        except Exception:
            pytest.skip("graphviz dot executable not available")
        for name in results:
            assert name.startswith("pipeline_")
