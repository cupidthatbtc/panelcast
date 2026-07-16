"""Structural guard: the predictive paths must thread identical observation-
distribution parameters.

The cold-start (``_run_new_artist_predictive`` -> ``predict_new_entity``) and
horizon-rollout (``_evaluate_horizon_rollout`` -> ``predict_horizon``) paths
describe the *same* fitted observation/innovation distribution. Any knob that
shapes that distribution (likelihood family, tail weight, discretization, ...)
must be forwarded by both, or the two evaluation surfaces silently score under
different predictive distributions.

The original defect (#230): ``skew_tailweight`` reached the rollout call but not
the cold-start call, so a non-unit tail weight would have been dropped on the
cold-start path with no error. This test is deliberately structural — it
compares the *set* of distribution parameters each path forwards against a
canonical roster, so it stays valid as new distribution knobs are added: extend
``OBS_DISTRIBUTION_PARAMS`` and wire the knob into both call sites.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable

from panelcast.models.bayes.predict import predict_new_entity
from panelcast.models.bayes.rollout import predict_horizon
from panelcast.pipelines import evaluate

# Parameters that define the predictive observation/innovation distribution.
# A new knob belongs here only if it changes the predictive distribution itself
# (not indexing/plumbing like prefix, seed, target_bounds, group_idx).
OBS_DISTRIBUTION_PARAMS = frozenset(
    {
        "likelihood_family",
        "skew_tailweight",
        "discretize_observation",
        "likelihood_df",
        "target_transform",
        "logit_offset",
        "ar_center",
        "fixed_n_exponent",
    }
)


def _forwarded_param_names(func: Callable) -> set[str]:
    """Every parameter name the function threads into a downstream call.

    Collects dict-literal string keys, keyword-argument names (covering both
    ``predictor(**kwargs)``-style dict literals and ``dict(a=..., b=...)``
    builders), and string-constant subscript targets (``kwargs["x"] = ...``),
    which together enumerate the parameters a path forwards regardless of how
    the kwargs are assembled.
    """
    tree = ast.parse(inspect.getsource(func))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            names.add(node.arg)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                names.add(node.slice.value)
    return names


def test_both_predictors_accept_the_distribution_params():
    """Neither predictor entry point may drop a canonical distribution knob."""
    for predictor in (predict_new_entity, predict_horizon):
        params = set(inspect.signature(predictor).parameters)
        missing = OBS_DISTRIBUTION_PARAMS - params
        assert not missing, f"{predictor.__name__} lacks distribution params: {missing}"


def test_cold_start_path_forwards_all_distribution_params():
    forwarded = _forwarded_param_names(evaluate._run_new_artist_predictive)
    missing = OBS_DISTRIBUTION_PARAMS - forwarded
    assert not missing, f"cold-start predictive path drops: {missing}"


def test_rollout_path_forwards_all_distribution_params():
    forwarded = _forwarded_param_names(evaluate._evaluate_horizon_rollout)
    missing = OBS_DISTRIBUTION_PARAMS - forwarded
    assert not missing, f"rollout predictive path drops: {missing}"


def test_predictive_paths_agree_on_distribution_params():
    """The two paths must forward the same distribution roster as each other."""
    cold = _forwarded_param_names(evaluate._run_new_artist_predictive)
    rollout = _forwarded_param_names(evaluate._evaluate_horizon_rollout)
    cold_dist = cold & OBS_DISTRIBUTION_PARAMS
    rollout_dist = rollout & OBS_DISTRIBUTION_PARAMS
    assert cold_dist == rollout_dist, (
        "cold-start and rollout predictive paths disagree on distribution "
        f"params: only-cold={cold_dist - rollout_dist}, "
        f"only-rollout={rollout_dist - cold_dist}"
    )
