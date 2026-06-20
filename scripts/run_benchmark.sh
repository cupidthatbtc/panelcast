#!/bin/bash
set -euo pipefail  # Exit on error, undefined var, or pipe failure

# Run GPU benchmark with pixi environment
# Usage: ./scripts/run_benchmark.sh [config] [output]

CONFIG=${1:-quick}
OUTPUT=${2:-reports/gpu_benchmark_results.json}

# Change to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if ! cd "$PROJECT_ROOT"; then
    echo "Failed to cd to $PROJECT_ROOT" >&2
    exit 1
fi

# Ensure we're using pixi
export PATH="$HOME/.pixi/bin:$PATH"

echo "Running GPU benchmark with config: $CONFIG"
echo "Output: $OUTPUT"

# Ensure output directory exists
OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

pixi run python scripts/benchmark_gpu.py --config "$CONFIG" --output "$OUTPUT"
