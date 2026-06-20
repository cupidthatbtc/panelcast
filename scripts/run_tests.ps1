$ErrorActionPreference = 'Stop'

# Run tests in the active environment
# Forward all arguments to pytest
python -m pytest $args

# Propagate pytest's exit code
exit $LASTEXITCODE
