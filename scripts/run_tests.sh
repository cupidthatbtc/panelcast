#!/usr/bin/env bash
set -e  # Exit immediately on error

# Forward all arguments to pytest
python -m pytest "$@"
