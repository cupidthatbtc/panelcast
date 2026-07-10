# 0.12.0 GPU validation — rung ladder e2e + parallel arms (RTX 5090, 2026-07-09/10)

## Rung ladder e2e (`aoty-012-ladder`, `--effort standard --arm-timeout auto --budget-hours 8`)

Ledger v2, 37 records (`ledger_ladder.json`): 18 rung-0 arms completed and
scored, 2 timeouts, 9 rung-0 + 8 rung-1 budget-skips. All four e2e checks
passed:

- **Promotion arithmetic ran**: `rung_promotion` produced 8 `@r1` records
  (survivors + reference refit) from the scored rung-0 set.
- **`@r1` ledger keys** present and rung-suffixed exactly per the v2 scheme.
- **Screening appendix** rendered with the screening-scale disclaimer.
- **Two-scale honesty under budget exhaustion**: one arm screened at
  z = +4.22, its final-rung fit was budget-skipped, and the verdict correctly
  reported "No candidate cleared the pre-registered bar; defaults hold" —
  screening evidence never fed a promotion claim.

Both timeouts were policy-correct, not spurious: `beta_binomial` (pathological
family) killed at the 1800 s floor after 10x its prediction; `ar1` killed at
exactly 3x its predicted screening runtime (the documented multiplier
trade-off — relevant to the #138 default-flip discussion, not a defect).

## Parallel arms (`aoty-012-parallel`, `--parallel-arms 2 --max-fits 8 --arm-timeout auto`)

Ledger (`ledger_parallel.json`): exactly 8 records — the concurrent
`max_fits` fix (#227) holding in production. 7 completed, 1 ordinary failure
(`beta_binomial` again; no cascade).

- **No OOM**: zero `RESOURCE_EXHAUSTED` / kill-and-serialize events across the
  run; per-child `XLA_PYTHON_CLIENT_MEM_FRACTION` caps applied.
- **Measured throughput gain**: 136.9 min of cumulative fit wall-clock
  completed in 83.1 min of elapsed time — 1.65x overlap, a 39% wall-clock
  saving at `--parallel-arms 2`.
- **Telemetry**: concurrent fits recorded `concurrent: 2` in the calibration
  store; the runtime predictor's serial-only filter excludes them, keeping
  the rate history clean.
