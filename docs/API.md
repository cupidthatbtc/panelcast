# Python API

`panelcast` ships a small, typed public surface so you can run the pipeline and
work with its configuration objects from Python without reading the source. The
package carries a PEP 561 `py.typed` marker, so type checkers use its inline
annotations directly.

## Stability guarantee

The names importable **directly from the top-level `panelcast` package** are the
supported API and follow semantic versioning starting from the next minor
release: they will not be removed or have their signatures broken without a major
version bump.

Everything reached through a submodule path (`panelcast.pipelines.*`,
`panelcast.models.*`, `panelcast.features.*`, …) is **internal**. It may change,
move, or disappear in any release — import it at your own risk.

Attribute access on the package is lazy (PEP 562): `import panelcast` does not
import jax or build the model graph. The cost is paid only when you touch a name.

## Public surface

| Name | Purpose |
|---|---|
| `DatasetDescriptor` | The YAML-backed domain descriptor: columns, bounds, feature pack. |
| `load_descriptor` | Resolve a descriptor from a name, path, or `None` (AOTY default). |
| `PipelineConfig` | Immutable run configuration (seed, MCMC settings, dataset, flags). |
| `PipelineOrchestrator` | Runs the pipeline stages; exposes `run_dir`/`manifest`. |
| `run_pipeline` | Convenience wrapper: build an orchestrator, run it, return the exit code. |
| `FeatureRegistry` | Registry of feature blocks assembled for a run. |
| `FeatureBlock` | Structural type (Protocol) a custom feature block must satisfy. |
| `build_default_registry` | Construct the default feature registry for a descriptor. |
| `LikelihoodSpec` | Descriptor of a likelihood family and its parameters. |

`__version__` is also importable.

## Example

Build (or load) a descriptor, run the pipeline, and read the resulting run
directory:

```python
from pathlib import Path

import panelcast

# Inspect the domain descriptor a run will use. `None` resolves the built-in
# AOTY default; pass a name or a YAML path for another domain.
descriptor = panelcast.load_descriptor("configs/datasets/aero.yaml")
print(descriptor.name, descriptor.target_col, descriptor.target_bounds)

# Configure a small run: 2 chains, short warmup, pointed at that descriptor.
config = panelcast.PipelineConfig(
    dataset="configs/datasets/aero.yaml",
    seed=42,
    num_chains=2,
    num_warmup=200,
    num_samples=200,
)

exit_code = panelcast.run_pipeline(config, output_base="outputs")
assert exit_code == 0

# Each run writes a timestamped directory under output_base; `latest` points at
# the most recent success. Read predictions, diagnostics, and the manifest there.
run_dir = Path("outputs") / "latest"
print("artifacts:", sorted(p.name for p in run_dir.iterdir()))
```

For finer control (resuming a run, inspecting the manifest mid-flight) construct
a `PipelineOrchestrator` directly:

```python
orchestrator = panelcast.PipelineOrchestrator(config, output_base="outputs")
exit_code = orchestrator.run()
print(orchestrator.run_dir)  # concrete run directory for this invocation
```

The CLI (`panelcast run`, `panelcast diagnose`, …) is documented separately in
[`CLI.md`](CLI.md); it is a thin wrapper over the same objects.
