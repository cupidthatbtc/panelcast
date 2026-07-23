import pytest
from typer.main import get_command

from panelcast.cli import app
from panelcast.config.pipeline_yaml import PIPELINE_YAML_MAPPING
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
        {"caged_chain_tree_depth_fraction": 0.0},
        {"caged_chain_tree_depth_fraction": 1.01},
        {"caged_chain_boundary_sigma": 0.0},
        {"caged_chain_consensus_ratio": 1.0},
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


def test_caged_chain_cli_surface_is_exposed():
    run_command = get_command(app).commands["run"]
    options = {option for param in run_command.params for option in param.opts}

    assert "--caged-chain-retries" in options
    assert "--caged-chain-tree-depth-fraction" in options
    assert "--caged-chain-boundary-sigma" in options
    assert "--caged-chain-consensus-ratio" in options
