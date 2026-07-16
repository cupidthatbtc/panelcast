"""panelcast — hierarchical Bayesian prediction for bounded scores of events
nested in entities over time, configured by a YAML descriptor.

The names re-exported here are the supported public API and follow semantic
versioning from the next minor release; see ``docs/API.md`` for the guarantee.
Everything reached through ``panelcast.*`` submodules is internal and may change
without notice. Attribute access is lazy (PEP 562), so ``import panelcast`` stays
cheap and does not eagerly import jax.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("panelcast")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

# Each public name mapped to the submodule it is lazily imported from. Keep this
# minimal — every entry is a semver promise (see docs/API.md), and importing a
# submodule here would defeat the point of the lazy seam.
_LAZY_EXPORTS = {
    "DatasetDescriptor": "panelcast.config.descriptor",
    "load_descriptor": "panelcast.config.descriptor",
    "PipelineConfig": "panelcast.pipelines.orchestrator",
    "PipelineOrchestrator": "panelcast.pipelines.orchestrator",
    "run_pipeline": "panelcast.pipelines.orchestrator",
    "FeatureRegistry": "panelcast.features",
    "FeatureBlock": "panelcast.features.base",
    "build_default_registry": "panelcast.features",
    "LikelihoodSpec": "panelcast.models.bayes.likelihoods",
}

if TYPE_CHECKING:  # let type checkers resolve the names without importing jax
    from panelcast.config.descriptor import DatasetDescriptor, load_descriptor
    from panelcast.features import FeatureRegistry, build_default_registry
    from panelcast.features.base import FeatureBlock
    from panelcast.models.bayes.likelihoods import LikelihoodSpec
    from panelcast.pipelines.orchestrator import (
        PipelineConfig,
        PipelineOrchestrator,
        run_pipeline,
    )


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*_LAZY_EXPORTS, "__version__"])


__all__ = [
    "__version__",
    "DatasetDescriptor",
    "load_descriptor",
    "PipelineConfig",
    "PipelineOrchestrator",
    "run_pipeline",
    "FeatureRegistry",
    "FeatureBlock",
    "build_default_registry",
    "LikelihoodSpec",
]
