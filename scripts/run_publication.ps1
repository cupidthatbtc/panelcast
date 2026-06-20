# Publication run script
# Runs the full reproducible pipeline with publication-grade defaults.
#
# Example usage:
#   .\scripts\run_publication.ps1
#   .\scripts\run_publication.ps1 --num-samples 2000 --num-warmup 2000

python -m panelcast.cli run --strict $args
exit $LASTEXITCODE
