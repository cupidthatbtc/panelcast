"""Data flow diagram generation for pipeline visualization using Graphviz DOT.

This module generates publication-quality SVG/PNG/PDF diagrams using Graphviz DOT
format with scientific journal styling: clean typography, professional color
palette, and clear visual hierarchy.

Three theme variants are supported:
- light: White background with black text (print-friendly)
- dark: Dark background with light text (presentations)
- transparent: No background for embedding in documents

Style Reference: Scientific journal aesthetic with:
- rankdir=TB (top-to-bottom flow)
- splines=polyline (clean angled connectors)
- fontname="Arial" (professional sans-serif)
- Clustered subgraphs with minimal borders
- Node shapes: box (rounded), ellipse (data), diamond (decision)
- Clean labels without decorative separators

Usage:
    >>> from panelcast.visualization.diagrams import generate_all_diagrams
    >>> results = generate_all_diagrams(Path("docs/figures"))
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import graphviz

__all__ = [
    "create_aoty_pipeline_diagram",
    "create_detailed_diagram",
    "create_high_level_diagram",
    "DetailLevel",
    "DiagramTheme",
    "generate_all_diagrams",
    "LEVEL_FUNCTIONS",
]

# Type alias for detail level
DetailLevel = Literal["high", "intermediate", "detailed"]

# Type alias for theme
DiagramTheme = Literal["light", "dark", "transparent"]

# Scientific journal color palette - professional, high-contrast, accessible
THEME_COLORS: dict[DiagramTheme, dict[str, str]] = {
    "light": {
        "bgcolor": "#FFFFFF",
        "fontcolor": "#000000",
        "color": "#333333",
        "fillcolor": "#FFFFFF",
        # Cluster fills - very light, subtle differentiation
        "input_fill": "#F7F7F7",
        "preprocess_fill": "#F0F0F0",
        "split_fill": "#FAFAFA",
        "feature_fill": "#F5F9F5",
        "model_fill": "#FFFDF0",
        "eval_fill": "#F8F5FA",
        "output_fill": "#F7F7F7",
        # Node fills - professional, muted
        "data_fill": "#E8E8E8",
        "storage_fill": "#D4E8D4",
        "decision_fill": "#FFE4B5",
        "result_fill": "#B8D4E8",
        "note_fill": "#F5F5DC",
        "merge_fill": "#E8E4D4",
        "train_fill": "#C8E0C8",
        "val_fill": "#C8D8E8",
        "test_fill": "#E0D0E8",
        # Edge colors
        "edge_primary": "#333333",
        "edge_feedback": "#666666",
        "edge_data": "#2E7D32",
    },
    "dark": {
        "bgcolor": "#1A1A1A",
        "fontcolor": "#E8E8E8",
        "color": "#808080",
        "fillcolor": "#2A2A2A",
        # Cluster fills - darker versions
        "input_fill": "#252525",
        "preprocess_fill": "#282828",
        "split_fill": "#232323",
        "feature_fill": "#252A25",
        "model_fill": "#2A2A22",
        "eval_fill": "#28252A",
        "output_fill": "#252525",
        # Node fills - darker, high contrast text
        "data_fill": "#3A3A3A",
        "storage_fill": "#2A3A2A",
        "decision_fill": "#3A3525",
        "result_fill": "#253040",
        "note_fill": "#35352A",
        "merge_fill": "#353530",
        "train_fill": "#2A3A2A",
        "val_fill": "#2A3040",
        "test_fill": "#352A3A",
        # Edge colors
        "edge_primary": "#B0B0B0",
        "edge_feedback": "#808080",
        "edge_data": "#4CAF50",
    },
    "transparent": {
        "bgcolor": "transparent",
        "fontcolor": "#000000",
        "color": "#333333",
        "fillcolor": "#FFFFFF",
        # Same as light for cluster/node fills
        "input_fill": "#F7F7F7",
        "preprocess_fill": "#F0F0F0",
        "split_fill": "#FAFAFA",
        "feature_fill": "#F5F9F5",
        "model_fill": "#FFFDF0",
        "eval_fill": "#F8F5FA",
        "output_fill": "#F7F7F7",
        "data_fill": "#E8E8E8",
        "storage_fill": "#D4E8D4",
        "decision_fill": "#FFE4B5",
        "result_fill": "#B8D4E8",
        "note_fill": "#F5F5DC",
        "merge_fill": "#E8E4D4",
        "train_fill": "#C8E0C8",
        "val_fill": "#C8D8E8",
        "test_fill": "#E0D0E8",
        "edge_primary": "#333333",
        "edge_feedback": "#666666",
        "edge_data": "#2E7D32",
    },
}


def _create_graph(
    theme: DiagramTheme,
    *,
    title: str = "AOTY Prediction Pipeline",
    nodesep: str = "0.5",
    ranksep: str = "0.6",
    node_fontsize: str = "10",
    node_margin: str = "0.15,0.08",
    title_fontsize: str = "14",
) -> graphviz.Digraph:
    """Create base Digraph with theme-specific global settings.

    Parameters
    ----------
    theme : DiagramTheme
        Visual theme: "light", "dark", or "transparent".
    title : str
        Diagram title text.
    nodesep : str
        Horizontal spacing between nodes.
    ranksep : str
        Vertical spacing between ranks.
    node_fontsize : str
        Default font size for nodes.
    node_margin : str
        Default margin for nodes (horizontal, vertical).
    title_fontsize : str
        Font size for the diagram title.

    Returns
    -------
    graphviz.Digraph
        Configured base graph.
    """
    colors = THEME_COLORS[theme]

    graph = graphviz.Digraph(
        name="AOTYPipeline",
        format="svg",
        engine="dot",
    )

    # Global graph settings - scientific journal style
    graph.attr(
        rankdir="TB",
        fontname="Arial",
        fontsize=title_fontsize,
        label=title,
        labelloc="t",
        labeljust="c",
        pad="0.5",
        nodesep=nodesep,
        ranksep=ranksep,
        splines="polyline",
        compound="true",
        overlap="false",
    )

    # Set bgcolor only for non-transparent themes
    if theme != "transparent":
        graph.attr(bgcolor=colors["bgcolor"])

    # Default node style - clean, professional
    graph.attr(
        "node",
        fontname="Arial",
        fontsize=node_fontsize,
        shape="box",
        style="filled,rounded",
        fillcolor=colors["fillcolor"],
        color=colors["color"],
        fontcolor=colors["fontcolor"],
        penwidth="1.2",
        margin=node_margin,
    )

    # Default edge style - clean lines
    graph.attr(
        "edge",
        fontname="Arial",
        fontsize="8",
        color=colors["edge_primary"],
        fontcolor=colors["fontcolor"],
        penwidth="1.2",
        arrowsize="0.8",
    )

    return graph


def create_high_level_diagram(theme: DiagramTheme = "light") -> graphviz.Digraph:
    """Create high-level AOTY pipeline overview diagram.

    Generates a simplified ~7-node diagram showing the main pipeline
    stages, suitable for README files and presentation overviews.

    Features:
    - Single node per stage (no internal details)
    - Clean, professional typography
    - Simplified feedback loop
    - Publication-ready styling

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme: "light" (white bg), "dark" (dark bg), "transparent".

    Returns
    -------
    graphviz.Digraph
        Configured high-level diagram ready for rendering.
    """
    colors = THEME_COLORS[theme]
    graph = _create_graph(
        theme,
        title="AOTY Prediction Pipeline",
        nodesep="0.8",
        ranksep="1.0",
        node_fontsize="14",
        node_margin="0.25,0.15",
        title_fontsize="20",
    )

    # Define cluster style
    cluster_style = {
        "style": "rounded,filled",
        "penwidth": "1.5",
        "fontname": "Arial",
        "fontsize": "14",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. INPUT
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_input") as c:
        c.attr(
            label="Input",
            fillcolor=colors["input_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "input_data",
            "Raw Data\n(130K albums)",
            shape="ellipse",
            fillcolor=colors["data_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_preprocess") as c:
        c.attr(
            label="Preprocessing",
            fillcolor=colors["preprocess_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "cleaning",
            "Data Cleaning\n& Validation",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SPLITTING
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_split") as c:
        c.attr(
            label="Data Splitting",
            fillcolor=colors["split_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "splits",
            "Train / Val / Test\n(temporal split)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_features") as c:
        c.attr(
            label="Feature Engineering",
            fillcolor=colors["feature_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "features",
            "Feature Pipeline\n(6 blocks)",
            shape="box",
            fillcolor=colors["merge_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. MODEL
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_model") as c:
        c.attr(
            label="Bayesian Modeling",
            fillcolor=colors["model_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "model",
            "Hierarchical Model\n(MCMC)",
            shape="box",
            fillcolor=colors["result_fill"],
            penwidth="2",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_eval") as c:
        c.attr(
            label="Evaluation",
            fillcolor=colors["eval_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "evaluation",
            "Diagnostics &\nValidation",
            shape="diamond",
            fillcolor=colors["decision_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_output") as c:
        c.attr(
            label="Output",
            fillcolor=colors["output_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "predictions",
            "Predictions\n(with 95% CI)",
            shape="ellipse",
            fillcolor=colors["storage_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY FLOW
    # ─────────────────────────────────────────────────────────────────────────
    edge_opts = {"penwidth": "1.5", "color": colors["edge_primary"]}
    graph.edge("input_data", "cleaning", **edge_opts)
    graph.edge("cleaning", "splits", **edge_opts)
    graph.edge("splits", "features", **edge_opts)
    graph.edge("features", "model", **edge_opts)
    graph.edge("model", "evaluation", **edge_opts)
    graph.edge("evaluation", "predictions", **edge_opts)

    # ─────────────────────────────────────────────────────────────────────────
    # FEEDBACK LOOP
    # ─────────────────────────────────────────────────────────────────────────
    graph.edge(
        "evaluation",
        "model",
        style="dashed",
        color=colors["edge_feedback"],
        xlabel="tune",
        fontsize="8",
        constraint="false",
    )

    return graph


def create_aoty_pipeline_diagram(theme: DiagramTheme = "light") -> graphviz.Digraph:
    """Create intermediate AOTY prediction pipeline diagram.

    Generates a medium-detail diagram showing pipeline components with:
    - 7 main stages with internal structure
    - Feature engineering blocks
    - Model architecture components
    - Evaluation chain

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme: "light" (white bg), "dark" (dark bg), "transparent".

    Returns
    -------
    graphviz.Digraph
        Configured diagram ready for rendering.
    """
    colors = THEME_COLORS[theme]
    graph = _create_graph(
        theme,
        title="AOTY Prediction Pipeline",
        nodesep="1.2",
        ranksep="1.5",
        node_fontsize="16",
        node_margin="0.3,0.15",
        title_fontsize="24",
    )

    # Cluster styling
    cluster_style = {
        "style": "rounded,filled",
        "penwidth": "2",
        "fontname": "Arial",
        "fontsize": "18",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. INPUT
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_input") as c:
        c.attr(
            label="1. Input",
            fillcolor=colors["input_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "input_file",
            "Raw CSV\n130K albums",
            shape="ellipse",
            fillcolor=colors["data_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_preprocess") as c:
        c.attr(
            label="2. Preprocessing",
            fillcolor=colors["preprocess_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("schema_val", "Schema\nValidation")
        c.node("cleaning", "Cleaning &\nFiltering")
        c.node(
            "cleaned_data",
            "Cleaned Data\n~62K rows",
            shape="ellipse",
            fillcolor=colors["storage_fill"],
        )
        c.edge("schema_val", "cleaning")
        c.edge("cleaning", "cleaned_data")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SPLITTING
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_split") as c:
        c.attr(
            label="3. Data Splitting",
            fillcolor=colors["split_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("split_strategy", "Temporal\nSplit")
        c.node(
            "train_set",
            "Train\n(64%)",
            shape="ellipse",
            fillcolor=colors["train_fill"],
        )
        c.node(
            "val_set",
            "Val\n(12%)",
            shape="ellipse",
            fillcolor=colors["val_fill"],
        )
        c.node(
            "test_set",
            "Test\n(12%)",
            shape="ellipse",
            fillcolor=colors["test_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_features") as c:
        c.attr(
            label="4. Feature Engineering",
            fillcolor=colors["feature_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("temporal_block", "Temporal")
        c.node("album_type_block", "Album Type")
        c.node("artist_history_block", "Artist History")
        c.node("genre_block", "Genre")
        c.node("collab_block", "Collaboration")
        c.node(
            "feature_pipeline",
            "Feature Pipeline",
            fillcolor=colors["merge_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. MODEL
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_model") as c:
        c.attr(
            label="5. Bayesian Model",
            fillcolor=colors["model_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("prior_config", "Prior\nConfiguration")
        c.node("hierarchical", "Hierarchical\nStructure")
        c.node("time_varying", "Time-varying\nEffects")
        c.node(
            "mcmc_sampling",
            "MCMC Sampling\n(NUTS/GPU)",
            fillcolor=colors["result_fill"],
            penwidth="2",
        )
        c.edge("prior_config", "hierarchical")
        c.edge("hierarchical", "time_varying")
        c.edge("time_varying", "mcmc_sampling")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_eval") as c:
        c.attr(
            label="6. Evaluation",
            fillcolor=colors["eval_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "convergence",
            "Convergence\nChecks",
            shape="diamond",
            fillcolor=colors["decision_fill"],
        )
        c.node("loo_cv", "LOO-CV")
        c.node("calibration", "Calibration")
        c.node("sensitivity", "Sensitivity\nAnalysis")
        c.edge("convergence", "loo_cv")
        c.edge("loo_cv", "calibration")
        c.edge("calibration", "sensitivity")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_output") as c:
        c.attr(
            label="7. Output",
            fillcolor=colors["output_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "predictions",
            "Predictions\n(95% CI)",
            shape="ellipse",
            fillcolor=colors["storage_fill"],
        )
        c.node("publication", "Publication\nArtifacts")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN FLOW
    # ─────────────────────────────────────────────────────────────────────────
    edge_opts = {"penwidth": "1.2", "color": colors["edge_primary"]}

    # Input -> Preprocessing
    graph.edge("input_file", "schema_val", **edge_opts)

    # Preprocessing -> Splits
    graph.edge("cleaned_data", "split_strategy", **edge_opts)
    graph.edge("split_strategy", "train_set", **edge_opts)
    graph.edge("split_strategy", "val_set", **edge_opts)
    graph.edge("split_strategy", "test_set", **edge_opts)

    # Train -> Feature Blocks
    train_edge = {"color": colors["edge_data"], "penwidth": "1.2"}
    graph.edge("train_set", "temporal_block", **train_edge)
    graph.edge("train_set", "album_type_block", **train_edge)
    graph.edge("train_set", "artist_history_block", **train_edge)
    graph.edge("train_set", "genre_block", **train_edge)
    graph.edge("train_set", "collab_block", **train_edge)

    # Feature Blocks -> Pipeline
    graph.edge("temporal_block", "feature_pipeline")
    graph.edge("album_type_block", "feature_pipeline")
    graph.edge("artist_history_block", "feature_pipeline")
    graph.edge("genre_block", "feature_pipeline")
    graph.edge("collab_block", "feature_pipeline")

    # Features -> Model
    graph.edge("feature_pipeline", "prior_config", **edge_opts)

    # Model -> Evaluation
    graph.edge("mcmc_sampling", "convergence", **edge_opts)

    # Evaluation -> Output
    graph.edge("convergence", "predictions", **edge_opts)
    graph.edge("sensitivity", "publication", **edge_opts)

    # Feedback loops
    graph.edge(
        "convergence",
        "mcmc_sampling",
        style="dashed",
        color=colors["edge_feedback"],
        xlabel="retry",
        fontsize="7",
        constraint="false",
    )
    graph.edge(
        "sensitivity",
        "prior_config",
        style="dashed",
        color=colors["edge_feedback"],
        xlabel="tune",
        fontsize="7",
        constraint="false",
    )

    # Test evaluation
    graph.edge(
        "test_set",
        "predictions",
        style="dashed",
        color=colors["edge_feedback"],
    )

    return graph


def create_detailed_diagram(theme: DiagramTheme = "light") -> graphviz.Digraph:
    """Create detailed technical reference diagram.

    Generates a comprehensive diagram showing all pipeline components,
    suitable for academic papers and technical documentation.

    Features:
    - Complete preprocessing chain
    - All 6 feature blocks with details
    - Full model architecture
    - Detailed evaluation chain
    - Feedback loops
    - Publication-quality styling

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme: "light" (white bg), "dark" (dark bg), "transparent".

    Returns
    -------
    graphviz.Digraph
        Configured detailed diagram ready for rendering.
    """
    colors = THEME_COLORS[theme]
    graph = _create_graph(
        theme,
        title="AOTY Prediction Pipeline: Technical Reference",
        nodesep="1.0",
        ranksep="1.2",
        node_fontsize="14",
        node_margin="0.25,0.12",
        title_fontsize="22",
    )

    # Cluster styling
    cluster_style = {
        "style": "rounded,filled",
        "penwidth": "2",
        "fontname": "Arial",
        "fontsize": "16",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. INPUT
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_input") as c:
        c.attr(
            label="1. Input",
            fillcolor=colors["input_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "input_file",
            "Raw CSV\n(130K rows)",
            shape="ellipse",
            fillcolor=colors["data_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_preprocess") as c:
        c.attr(
            label="2. Preprocessing",
            fillcolor=colors["preprocess_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("schema_val", "Schema\nValidation")
        c.node("null_handling", "Null\nHandling")
        c.node("date_parsing", "Date\nParsing")
        c.node("min_ratings", "Min Ratings\nFilter")
        c.node(
            "cleaned_data",
            "Cleaned Data\n(62K rows)",
            shape="ellipse",
            fillcolor=colors["storage_fill"],
        )
        c.edge("schema_val", "null_handling")
        c.edge("null_handling", "date_parsing")
        c.edge("date_parsing", "min_ratings")
        c.edge("min_ratings", "cleaned_data")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SPLITTING
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_split") as c:
        c.attr(
            label="3. Data Splitting",
            fillcolor=colors["split_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("within_artist", "Within-Artist\n(temporal)")
        c.node("artist_disjoint", "Artist-Disjoint\n(holdout)")
        c.node(
            "train_set",
            "Train (64%)",
            shape="ellipse",
            fillcolor=colors["train_fill"],
        )
        c.node(
            "val_set",
            "Val (12%)",
            shape="ellipse",
            fillcolor=colors["val_fill"],
        )
        c.node(
            "test_set",
            "Test (12%)",
            shape="ellipse",
            fillcolor=colors["test_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_features") as c:
        c.attr(
            label="4. Feature Engineering (fit on train)",
            fillcolor=colors["feature_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("temporal_block", "Temporal\n(sequence, gap)")
        c.node("album_type_block", "Album Type\n(one-hot)")
        c.node("artist_history_block", "Artist History\n(LOO stats)")
        c.node("artist_rep_block", "Artist Rep\n(smoothed)")
        c.node("genre_block", "Genre\n(PCA)")
        c.node("collab_block", "Collaboration\n(ordinal)")
        c.node(
            "feature_pipeline",
            "Feature Pipeline",
            fillcolor=colors["merge_fill"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. MODEL
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_model") as c:
        c.attr(
            label="5. Bayesian Model",
            fillcolor=colors["model_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node("prior_config", "Prior Config\n(9 params)")
        c.node("hierarchical", "Hierarchical\n(artist effects)")
        c.node("non_centered", "Non-centered\nParam")
        c.node("time_varying", "Time-varying\n(random walk)")
        c.node("ar1_structure", "AR(1)\nStructure")
        c.node(
            "mcmc_sampling",
            "MCMC\n(NUTS/GPU)",
            fillcolor=colors["result_fill"],
            penwidth="2",
        )
        c.node(
            "posterior",
            "Posterior\n(4K samples)",
            shape="ellipse",
            fillcolor=colors["storage_fill"],
        )
        c.edge("prior_config", "hierarchical")
        c.edge("hierarchical", "non_centered")
        c.edge("non_centered", "time_varying")
        c.edge("time_varying", "ar1_structure")
        c.edge("ar1_structure", "mcmc_sampling")
        c.edge("mcmc_sampling", "posterior")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_eval") as c:
        c.attr(
            label="6. Evaluation",
            fillcolor=colors["eval_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "convergence",
            "Convergence",
            shape="diamond",
            fillcolor=colors["decision_fill"],
        )
        c.node("rhat_check", "R-hat\n(<1.01)")
        c.node("ess_check", "ESS\n(>400)")
        c.node("loo_cv", "LOO-CV\n(PSIS)")
        c.node(
            "pareto_k",
            "Pareto-k",
            shape="diamond",
            fillcolor=colors["decision_fill"],
        )
        c.node("calibration", "Calibration")
        c.node("sensitivity", "Sensitivity\nAnalysis")
        c.edge("convergence", "rhat_check")
        c.edge("convergence", "ess_check")
        c.edge("rhat_check", "loo_cv")
        c.edge("ess_check", "loo_cv")
        c.edge("loo_cv", "pareto_k")
        c.edge("pareto_k", "calibration")
        c.edge("calibration", "sensitivity")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    with graph.subgraph(name="cluster_output") as c:
        c.attr(
            label="7. Output",
            fillcolor=colors["output_fill"],
            color=colors["color"],
            fontcolor=colors["fontcolor"],
            **cluster_style,
        )
        c.node(
            "predictions",
            "Predictions\n(95% CI)",
            shape="ellipse",
            fillcolor=colors["storage_fill"],
        )
        c.node("model_artifacts", "Model\nArtifacts")
        c.node("publication", "Publication\nFigures")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN FLOW
    # ─────────────────────────────────────────────────────────────────────────
    edge_opts = {"penwidth": "1.2", "color": colors["edge_primary"]}

    # Input -> Preprocessing
    graph.edge("input_file", "schema_val", **edge_opts)

    # Preprocessing -> Splits
    graph.edge("cleaned_data", "within_artist", **edge_opts)
    graph.edge("cleaned_data", "artist_disjoint", **edge_opts)

    # Splits -> Sets
    graph.edge("within_artist", "train_set", **edge_opts)
    graph.edge("within_artist", "val_set", **edge_opts)
    graph.edge("within_artist", "test_set", **edge_opts)

    # Train -> Feature Blocks
    train_edge = {"color": colors["edge_data"], "penwidth": "1.2"}
    graph.edge("train_set", "temporal_block", **train_edge)
    graph.edge("train_set", "album_type_block", **train_edge)
    graph.edge("train_set", "artist_history_block", **train_edge)
    graph.edge("train_set", "artist_rep_block", **train_edge)
    graph.edge("train_set", "genre_block", **train_edge)
    graph.edge("train_set", "collab_block", **train_edge)

    # Feature Blocks -> Pipeline
    graph.edge("temporal_block", "feature_pipeline")
    graph.edge("album_type_block", "feature_pipeline")
    graph.edge("artist_history_block", "feature_pipeline")
    graph.edge("artist_rep_block", "feature_pipeline")
    graph.edge("genre_block", "feature_pipeline")
    graph.edge("collab_block", "feature_pipeline")

    # Features -> Model
    graph.edge("feature_pipeline", "prior_config", **edge_opts)

    # Model -> Evaluation
    graph.edge("posterior", "convergence", **edge_opts)

    # Evaluation -> Output
    graph.edge("sensitivity", "predictions", **edge_opts)
    graph.edge("posterior", "model_artifacts", style="dashed", color=colors["edge_feedback"])
    graph.edge("sensitivity", "publication", **edge_opts)

    # Feedback loops
    graph.edge(
        "convergence",
        "mcmc_sampling",
        style="dashed",
        color=colors["edge_feedback"],
        xlabel="retry",
        fontsize="7",
        constraint="false",
    )
    graph.edge(
        "sensitivity",
        "prior_config",
        style="dashed",
        color=colors["edge_feedback"],
        xlabel="tune",
        fontsize="7",
        constraint="false",
    )

    # Test evaluation
    graph.edge(
        "test_set",
        "predictions",
        style="dashed",
        color=colors["edge_feedback"],
    )

    return graph


def generate_all_diagrams(
    output_dir: Path,
    levels: list[DetailLevel] | None = None,
) -> dict[str, list[Path]]:
    """Generate all AOTY pipeline diagram variants.

    Creates diagram variants for specified detail levels and all themes.
    Each diagram is saved in SVG, PNG, PDF, and DOT formats.

    Parameters
    ----------
    output_dir : Path
        Directory for output files (created if needed).
    levels : list[DetailLevel] | None, optional
        Detail levels to generate. If None, generates all three levels
        (high, intermediate, detailed). Default is None.

    Returns
    -------
    dict[str, list[Path]]
        Dict mapping diagram name to list of created file paths.
        Names follow pattern: pipeline_{level}_{theme}

    Notes
    -----
    With default parameters (all levels), generates 9 diagram sets
    (3 levels x 3 themes) with 4 files each (svg, png, pdf, dot),
    totaling 36 files.

    Example
    -------
    >>> results = generate_all_diagrams(Path("docs/figures"))
    >>> print(f"Generated {len(results)} diagram sets")
    Generated 9 diagram sets

    >>> # Generate only high-level diagrams (3 themes x 4 formats = 12 files)
    >>> results = generate_all_diagrams(Path("docs/figures"), levels=["high"])
    >>> print(f"Generated {len(results)} diagram sets")
    Generated 3 diagram sets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to all levels if not specified
    if levels is None:
        levels = ["high", "intermediate", "detailed"]

    results: dict[str, list[Path]] = {}
    themes: list[DiagramTheme] = ["light", "dark", "transparent"]
    formats = ["svg", "png", "pdf"]

    for level in levels:
        diagram_func = LEVEL_FUNCTIONS[level]
        for theme in themes:
            name = f"pipeline_{level}_{theme}"
            diagram = diagram_func(theme)

            created_paths: list[Path] = []
            for fmt in formats:
                # graphviz render() writes to directory with auto filename
                # We use render to generate the dot file then convert
                base_path = output_dir / name
                diagram.format = fmt
                output_path = diagram.render(
                    filename=str(base_path),
                    directory=None,
                    cleanup=True,  # Remove intermediate dot file
                )
                created_paths.append(Path(output_path))

            # Also save the .dot source file for reference
            dot_path = output_dir / f"{name}.dot"
            dot_path.write_text(diagram.source, encoding="utf-8")
            created_paths.append(dot_path)

            results[name] = created_paths

    return results


# Level function mapping for programmatic access
LEVEL_FUNCTIONS: dict[DetailLevel, Callable[[DiagramTheme], graphviz.Digraph]] = {
    "high": create_high_level_diagram,
    "intermediate": create_aoty_pipeline_diagram,
    "detailed": create_detailed_diagram,
}
