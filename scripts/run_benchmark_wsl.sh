#!/bin/bash
# Run GPU benchmark with aoty-gpu conda environment in WSL2
# Usage: ./scripts/run_benchmark_wsl.sh [config] [output]

set -e

CONFIG=${1:-quick}
OUTPUT=${2:-reports/gpu_benchmark_results.json}

PROJECT_DIR=/mnt/c/Users/jcwen/Projects/panelcast
CONDA_BASE=/home/jcwen/miniforge3

cd "$PROJECT_DIR"

# Add project source to PYTHONPATH
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

echo "Running GPU benchmark with config: $CONFIG"
echo "Output: $OUTPUT"

# Run benchmark using aoty-gpu conda environment
$CONDA_BASE/bin/conda run -n aoty-gpu python scripts/benchmark_gpu.py --config "$CONFIG" --output "$OUTPUT"
