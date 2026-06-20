## Summary

What this PR does and why.

## Changes

- 

## Testing

- [ ] `pixi run test-fast` passes (or `JAX_PLATFORMS=cpu python -m pytest -m "not slow"`)
- [ ] `pre-commit run --all-files` is clean (ruff-format, ruff, mypy)
- [ ] New behavior is covered by tests

## Domain-portability check

- [ ] No new hard-coded dataset literals — domain-specific names go through the
      `DatasetDescriptor` (the `test_no_domain_literals` guard still passes)
- [ ] If sampling/computation changed, the convergence/PPC and golden-hash
      checks still pass

## Notes

Anything reviewers should know — follow-ups, known limitations, or decisions
made along the way.
