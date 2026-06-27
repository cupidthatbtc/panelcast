# Contributing to panelcast

Thanks for your interest in the project. This guide covers the development
setup, the test and lint workflow, and the conventions the codebase follows.

## Development setup

[pixi](https://pixi.sh) manages the full environment (conda + PyPI, including
the JAX stack). After installing pixi:

```bash
git clone https://github.com/cupidthatbtc/panelcast.git
cd panelcast
pixi install          # resolve the locked environment
pip install -e .      # install the panelcast package + CLI into the env
pre-commit install    # enable the git hooks (once)
```

## Running the tests

The suite uses `pytest` with two markers: `slow` and `e2e`. The fast inner-loop
selection skips both:

```bash
# Fast unit/integration selection (matches CI's default gate)
pixi run test-fast

# With the coverage gate (fail_under = 80, configured in pyproject.toml)
pixi run test-cov

# Everything, including slow and end-to-end tests
pixi run test
```

Tests are designed to run deterministically on CPU. When running pytest
directly (outside the pixi tasks), pin the JAX backend so a missing or busy GPU
never changes behavior:

```bash
JAX_PLATFORMS=cpu python -m pytest -m "not slow"
```

A few useful selections:

```bash
python -m pytest tests/unit                     # unit tests only
python -m pytest tests/e2e -m e2e               # portability + full-pipeline e2e
python -m pytest -k domain_portability          # the zero-source-change proof
```

## Linting and types

Code quality is enforced by `ruff` (format + lint) and `mypy`, wired through
both pre-commit and CI:

```bash
pixi run lint                 # ruff check src/  (what CI runs)
pre-commit run --all-files    # ruff-format, ruff --fix, mypy on everything
```

Please run `pre-commit run --all-files` before opening a PR so the hooks have a
chance to auto-fix formatting and imports.

## Conventions

- **Commit messages** use a feature-area prefix: `feat(preflight):`,
  `fix(gpu-memory):`, `test(cli):`, `docs(porting):`, etc. Keep them concise.
- **Domain portability is a hard contract.** Dataset-specific names (columns,
  bounds, date formats) must flow through the `DatasetDescriptor`
  (`src/panelcast/config/descriptor.py`), never hard-coded. The guard test
  `tests/unit/test_no_domain_literals.py` fails the build if an AOTY column
  literal appears outside the sanctioned whitelist — route new names through the
  descriptor instead. See [`docs/PORTING.md`](docs/PORTING.md) and
  [`docs/EXTENSIBILITY.md`](docs/EXTENSIBILITY.md).
- **Don't commit data or model artifacts.** `data/`, `models/`, and `outputs/`
  are git-ignored by design; the pipeline regenerates them.
- **Diagnostics gate results.** Changes that affect sampling should keep the
  convergence / PPC / calibration checks green; golden-hash tests guard computed
  values against accidental drift.

## Releasing

The Python runtime reads its version from the installed package metadata
(`panelcast.__version__` → `importlib.metadata`), whose source of truth is
`pyproject.toml`. Bump `version` there, then hand-sync the static metadata that
no code can read: `pixi.toml`, `CITATION.cff`, and the `MODEL_CARD.md` header.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For bugs, include the
command you ran, the dataset descriptor (or that you used the AOTY defaults),
and the relevant diagnostics or traceback.
