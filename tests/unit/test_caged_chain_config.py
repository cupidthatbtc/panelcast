import pytest
from typer.main import get_command

from panelcast.cli import app
from panelcast.config.pipeline_yaml import PIPELINE_YAML_MAPPING, apply_yaml_overrides
from panelcast.pipelines.orchestrator import PipelineConfig


def test_caged_chain_defaults_are_detection_only_with_retry_off():
    config = PipelineConfig()

    assert config.caged_chain_retries == 0
    assert config.caged_chain_tree_depth_fraction == 0.95
    assert config.caged_chain_boundary_sigma == 0.005
    assert config.caged_chain_consensus_ratio == 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"caged_chain_retries": -1},
        {"caged_chain_retries": 11},
        {"caged_chain_retries": 1.5},
        {"caged_chain_retries": True},
        {"caged_chain_tree_depth_fraction": 0.0},
        {"caged_chain_tree_depth_fraction": 1.01},
        {"caged_chain_tree_depth_fraction": float("nan")},
        {"caged_chain_tree_depth_fraction": float("inf")},
        {"caged_chain_boundary_sigma": 0.0},
        {"caged_chain_boundary_sigma": float("nan")},
        {"caged_chain_consensus_ratio": 1.0},
        {"caged_chain_consensus_ratio": float("inf")},
        {"caged_chain_consensus_ratio": "large"},
    ],
)
def test_caged_chain_config_validation(kwargs):
    with pytest.raises(ValueError, match="caged_chain"):
        PipelineConfig(**kwargs)


def test_caged_chain_yaml_keys_honor_cli_precedence():
    for key in (
        "caged_chain_retries",
        "caged_chain_tree_depth_fraction",
        "caged_chain_boundary_sigma",
        "caged_chain_consensus_ratio",
    ):
        spec = PIPELINE_YAML_MAPPING[key]
        assert spec.config_field == key
        assert spec.cli_param == key

    out = apply_yaml_overrides(
        {
            "caged_chain_retries": 1,
            "caged_chain_boundary_sigma": 0.005,
        },
        {
            "caged_chain_retries": 2,
            "caged_chain_boundary_sigma": 0.002,
        },
        {"caged_chain_retries"},
    )
    assert out["caged_chain_retries"] == 1
    assert out["caged_chain_boundary_sigma"] == 0.002


def test_caged_chain_cli_surface_is_exposed():
    run_command = get_command(app).commands["run"]
    params = {option: param for param in run_command.params for option in param.opts}

    assert params["--caged-chain-retries"].type.min == 0
    assert params["--caged-chain-tree-depth-fraction"].type.min == 0.0
    assert params["--caged-chain-boundary-sigma"].type.min == 0.0
    assert params["--caged-chain-consensus-ratio"].type.min == 1.0
