from types import SimpleNamespace
from unittest.mock import Mock

import arviz as az
import numpy as np
import xarray as xr

from panelcast.models.bayes.fit import MCMCConfig
from panelcast.pipelines.train_bayes import _fit_with_caged_chain_retries


def _result(*, caged: bool, bad_survivors: bool = False):
    rng = np.random.default_rng(17)
    chains, draws = 4, 600
    alpha = rng.normal(size=(chains, draws))
    sigma = rng.normal(0.08, 0.002, size=(chains, draws))
    steps = np.full((chains, draws), 63.0)
    if caged:
        sigma[0] = rng.normal(0.001, 0.00005, size=draws)
        steps[0] = 1000.0
    if bad_survivors:
        alpha[1] += 4.0
        alpha[2] -= 4.0
    posterior = xr.Dataset(
        {
            "alpha": xr.DataArray(alpha, dims=["chain", "draw"]),
            "user_sigma_artist": xr.DataArray(sigma, dims=["chain", "draw"]),
        }
    )
    sample_stats = xr.Dataset(
        {
            "num_steps": xr.DataArray(steps, dims=["chain", "draw"]),
            "diverging": xr.DataArray(
                np.zeros((chains, draws), dtype=bool), dims=["chain", "draw"]
            ),
        }
    )
    return SimpleNamespace(idata=az.InferenceData(posterior=posterior, sample_stats=sample_stats))


def _run(initial, fit_once, retries=2):
    return _fit_with_caged_chain_retries(
        initial,
        MCMCConfig(seed=42),
        fit_once=fit_once,
        max_retries=retries,
        scale_parameter="user_sigma_artist",
        tree_depth_fraction=0.95,
        boundary_sigma=0.005,
        consensus_ratio=5.0,
        rhat_threshold=1.01,
        ess_threshold=100,
        allow_divergences=False,
    )


def test_retry_is_off_by_default_and_does_not_call_sampler():
    initial = _result(caged=True)
    fit_once = Mock()

    selected, config, _, caged, attempts = _run(initial, fit_once, retries=0)

    assert selected is initial
    assert config.seed == 42
    assert caged.chain_ids == [0]
    assert attempts[0]["attempt"] == 0
    assert attempts[0]["seed"] == 42
    assert attempts[0]["caged_chain_ids"] == [0]
    assert attempts[0]["survivor_diagnostics"]["passed"] is True
    fit_once.assert_not_called()


def test_survivor_gate_blocks_retry():
    initial = _result(caged=True, bad_survivors=True)
    fit_once = Mock()

    selected, config, _, _, attempts = _run(initial, fit_once)

    assert selected is initial
    assert config.seed == 42
    assert attempts[0]["survivor_diagnostics"]["passed"] is False
    fit_once.assert_not_called()


def test_retry_retains_first_all_consensus_result_with_deterministic_seed():
    initial = _result(caged=True)
    still_caged = _result(caged=True)
    consensus = _result(caged=False)
    fit_once = Mock(side_effect=[still_caged, consensus])

    selected, config, diagnostics, caged, attempts = _run(initial, fit_once)

    assert selected is consensus
    assert config.seed == 44
    assert diagnostics.passed is True
    assert caged.chain_ids == []
    assert [attempt["seed"] for attempt in attempts] == [42, 43, 44]
    assert [call.args[0].seed for call in fit_once.call_args_list] == [43, 44]


def test_retry_exhaustion_retains_original_result():
    initial = _result(caged=True)
    fit_once = Mock(side_effect=[_result(caged=True), _result(caged=True)])

    selected, config, _, caged, attempts = _run(initial, fit_once)

    assert selected is initial
    assert config.seed == 42
    assert caged.chain_ids == [0]
    assert len(attempts) == 3
    assert fit_once.call_count == 2
