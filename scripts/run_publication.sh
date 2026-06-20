#!/usr/bin/env bash
set -euo pipefail

# Publication run helper.
# Uses CLI defaults plus --strict for fail-fast diagnostics/artifacts.
# Pass additional CLI arguments through to `panelcast run`.
#
# Examples:
#   ./scripts/run_publication.sh
#   ./scripts/run_publication.sh --num-samples 2000 --num-warmup 2000 --target-accept 0.95

panelcast run --strict "$@"
