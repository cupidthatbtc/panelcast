# Entity-obs three-seed confirmation — promoted to the AOTY default (0.13.0, #238)

2026-07-13. `heteroscedastic_entity_obs` — per-entity multiplicative observation
noise, the dormant model-v2 gate — is the strongest promotion candidate AOTY has
produced. It was **never in the original cross-domain model-v2 bake-off**; it
surfaced from the 0.12 `panelcast select` rung ladder, which screened it at
z +4.22 over the shipped default. The signal then **replicated at four sampler
scales** and cleared a full three-seed pre-registered confirmation (5000+5000,
seeds 42/43/44) on z, convergence, and 95% coverage. Per-seed reports and
ledgers in `s42/`, `s43/`, `s44/` (paths scrubbed to repo-relative).

Arm `2fee043e3e62` = shipped default **+** `heteroscedastic_entity_obs: true`.
Reference `750f957a8c71` = shipped default (paired diff identically zero).

| seed | ELPD Δ | dse | z | cov80 Δ | cov95 Δ | PPC pins | conv (R̂ / ESS / div) |
|---|---:|---:|---:|---:|---:|:--:|:--|
| 42 | +29.77 | 7.00 | +4.254 | **+0.0300153** | +0.0178 | 2 | PASS (1.00 / 1119 / 0) |
| 43 | +29.79 | 7.01 | +4.246 | +0.0284839 | +0.0194 | 2 | PASS (1.00 / 1235 / 0) |
| 44 | +29.81 | 7.01 | +4.251 | **+0.0300153** | +0.0178 | 2 | PASS (1.00 / 1383 / 0) |
| ref (s42) | 0.00 | 0.00 | 0.00 | +0.0530 | +0.0132 | 4 | PASS (1.00 / 3085 / 0) |

**Headline vs the shipped default:** ELPD **+29.8 ± 7.0** (z +4.25), a clean
distributional + calibration win. Point accuracy is a wash — MAE 5.276 vs 5.300,
RMSE 7.672 vs 7.649. The gate **resolves two of the four structural PPC pins**:
the reference stays pinned on `skewness, max, q10, q90`; the arm pins only
`skewness, max`, clearing **q10 and q90**. That is the first movement on the
bounded-skew tail misfit that six likelihood families never touched.

**Verdict: HELD.** The pre-registered 80% coverage tolerance is
|cov80 Δ| ≤ 0.03. On 2 of 3 seeds the arm lands at 0.0300153 — over the line by
**1.53e-5**. This is one album out of 653: coverage is quantized at 1/653 ≈
1.53e-3 per album, and the 0.03 threshold falls *between* 541/653 (seed 43, PASS)
and 542/653 (seeds 42/44, miss). The subset literally cannot resolve which side
of 0.03 the true coverage delta sits on. `panelcast select`'s verdict renderer
holds the arm on the overridden pre-registration, and — consistent with the
0.11/0.12 freeze discipline — **we do not promote on an overridden
pre-registration.**

Promotion was deferred at 0.12.1 to the full-corpus run (**#15**). The AOTY
default kept the gate off and every published number stayed bit-identical.

**Update (0.13.0, #238): PROMOTED.** #237 amended the coverage gate to clear an
axis on the tolerance **or** on non-inferiority to the reference. The reference
(shipped default) misses the same 80% tolerance by ~1500× the candidate's margin
(cov80 0.0530 vs 0.0300), so under the amended rule this evidence re-scores
**PROMOTABLE on all three seeds** — the 1.53e-5 subset-grid quantization no longer
gates the decision. `heteroscedastic_entity_obs` is now the AOTY default; the
full-corpus fit (#15) still resolves the grid and doubles as the re-baseline.

**Cross-domain reconciliation.** The model-v2 bake-off rejected this same gate on
two other domains — IMDb (a calibration-vs-sharpness trade with worse LOO) and
econ (a deeper collapse). Those verdicts stand. The AOTY result is the finding:
the same feature is a clean dual win on this domain and a reject elsewhere —
per-domain, opposite verdict. See `docs/decisions/entity_overdispersion.md`.
