# Development Setup

## Prerequisites

- Python >= 3.11
- [pixi](https://pixi.sh) package manager

## Installation

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone https://github.com/cupidthatbtc/panelcast.git
cd panelcast
pixi install
```

## Development Workflow

### Pre-commit Hooks

Install hooks (runs ruff + mypy on each commit):

```bash
python -m pip install pre-commit mypy types-PyYAML
pre-commit install
```

Run manually on all files:

```bash
pre-commit run --all-files
```

### Running Tests

```bash
# All tests
pixi run pytest

# Unit tests only (fast)
pixi run pytest tests/unit/

# Skip slow/e2e tests
pixi run pytest -m "not slow and not e2e"

# With coverage
pixi run pytest --cov=src/panelcast
```

### Linting

```bash
# Check only
pixi run lint

# Auto-fix
pixi run ruff check --fix src/
pixi run ruff format src/
```

### Type Checking

```bash
python -m mypy src/panelcast/
```

## GPU Setup

For MCMC model training, see [GPU_SETUP.md](GPU_SETUP.md) for JAX/NumPyro GPU configuration.

Tests and feature building are CPU-only.

## CI

GitHub Actions runs on every push/PR to master:
1. Lint (ruff)
2. Unit tests (excluding slow/e2e markers)
