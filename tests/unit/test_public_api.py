"""The top-level ``panelcast`` package is a semver-guaranteed public surface
(see docs/API.md). These tests pin the three promises: every advertised name
resolves through the PEP 562 lazy ``__getattr__``, unknown names still raise,
and ``import panelcast`` stays light — it must not eagerly import jax.
"""

import os
import pathlib
import subprocess
import sys

import pytest

import panelcast

# Kept in lockstep with panelcast.__all__ / _LAZY_EXPORTS; a name added to the
# API without landing here (or vice versa) should fail this test.
PUBLIC_NAMES = [
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


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_resolves(name):
    obj = getattr(panelcast, name)
    assert obj is not None
    # Resolves to the same object on the second access (import_module is cached).
    assert getattr(panelcast, name) is obj


def test_all_matches_exports():
    assert set(panelcast.__all__) == {*PUBLIC_NAMES, "__version__"}


def test_dir_lists_public_surface():
    listed = set(panelcast.__dir__())
    assert {*PUBLIC_NAMES, "__version__"} <= listed


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        panelcast.does_not_exist


def test_version_is_exposed():
    assert isinstance(panelcast.__version__, str)


def test_import_does_not_pull_jax():
    # Run in a clean interpreter: other tests in this process may have imported
    # jax already, so an in-process ``sys.modules`` check would be unsound.
    code = "import sys, panelcast; assert 'jax' not in sys.modules, sorted(m for m in sys.modules if m.startswith('jax'))"
    # panelcast may be on the path via pytest rather than installed; hand the
    # child the same import root so it resolves the package we are testing.
    src_root = str(pathlib.Path(panelcast.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src_root, os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
