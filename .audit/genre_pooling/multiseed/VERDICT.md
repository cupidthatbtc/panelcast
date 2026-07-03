# Genre pooling multi-seed screening + publication confirmation (#85)

2026-07-03. E1a: paired 2×2000 gate-off/on diagnostic runs at seeds 43 and 44
(same recipe as [`../gate_on_screening.md`](../gate_on_screening.md)); raw
metrics in this directory. Cold-start deltas (gate on − off):

| entity_disjoint | seed 42 | seed 43 (n=832) | seed 44 (n=816) |
|---|---:|---:|---:|
| ΔMAE | −0.135 | −0.076 | −0.110 |
| ΔR² | +0.034 | +0.019 | +0.016 |
| ΔCRPS | −0.108 | −0.058 | −0.053 |

Direction holds at every seed on every metric, coverage nominal, within-entity
at noise scale. Publication-scale confirmation ran as one combined fit with
`gbm_offset` (per the pre-registered budget rule): convergence PASS (R-hat
1.00, ESS 2612, 0 divergences), paired held-out ELPD vs the 0.5.0 baseline
+224.2 (se 12.6, z +17.9), cold-start MAE −0.101 / R² +0.019 at scale.

**Promoted in 0.6.0** as a tri-state auto default (on where the descriptor
defines a usable `entity_group_col`; #95). Full verdicts on the issue thread.
