# DECISION — entity overdispersion (C1) + lognormal sigma_obs (C2)

Status: **bake-off run on econ + IMDb (IMDb under the fair eval); both verdicts
negative → nothing adopted, so every published number stays bit-identical.** The
two gates are implemented default-off and parity-locked
(`tests/unit/models/bayes/test_fit_entity_obs.py`). Outcome: **econ rejected**
(no variant converges — collapse is deeper than the gates reach); **IMDb
rejected** under the locked LOO gate (C1 is a robust calibration-vs-sharpness
trade — better 95% coverage, worse LOO). Physics control bake-off still pending
(confirm no regression). The gates remain in the codebase, default-off and
reversible, for a future calibration-priority adoption or domain.

## Why this experiment exists

Two real fits failed their diagnostics under the published model:

- **IMDb episodes** *under*-covers its intervals (forward 80/95 coverage
  0.68 / 0.87 vs nominal 0.80 / 0.95; PIT max-dev 0.112, the worst in the
  suite). The episode score series is noisier per entity than the model's
  single homoscedastic `sigma_obs` allows.
- **Science / Economics** catastrophically failed to converge (R-hat 27.8,
  ESS 4). Diagnosis: a **variance-component collapse** — `sci_sigma_obs → 0.004`
  (physics 0.625) with the variance blown up into `sigma_artist` (1.6) and
  `sigma_rw` (1.3). The HalfNormal `sigma_obs` prior piles mass at zero, and on
  econ's heavily zero-inflated log-citations (median ~4) NUTS rides that
  boundary. Same model converged on physics, so it is a model–data mismatch,
  not a code bug.

C1 gives noisy series a per-entity noise home (wider, better-calibrated
intervals); C2 removes the zero-boundary artifact so `sigma_obs` cannot
collapse. They share the variance budget, so they are evaluated as one family.

## Experiment

`scripts/experiment_entity_overdispersion.py`, run per domain from that
domain's working dir at cheap `2x500/500` MCMC:

| Variant | Gate(s) |
|---|---|
| `off` | published baseline (bit-identical) |
| `c1_0.25` | `heteroscedastic_entity_obs`, `tau_entity_scale=0.25` |
| `c1_0.5` | `heteroscedastic_entity_obs`, `tau_entity_scale=0.5` |
| `c2_lognormal` | `sigma_obs_prior_type=lognormal` |
| `c1c2` | both (0.25 + lognormal) |

Domains: **IMDb episodes** (under-covering), **Economics** (collapsing),
**physics** (the over-/well-covered control — must stay put). Metrics per
variant: held-out LOO ELPD (± SE, score scale), per-site ESS / R-hat,
divergences, and held-out 80 / 95 posterior-predictive coverage. Artifact:
`outputs/experiments/entity_overdispersion.json` under each domain.

## Acceptance criteria (per domain)

Adopt the variant for a domain iff **all** hold:

1. **LOO** ≥ baseline + 2·SE where the gate is meant to help (IMDb, econ);
   and **no** variant shows a > 2·SE LOO *regression* on the physics control.
2. **Calibration** (IMDb): 80 and 95 coverage move toward nominal and land
   within `coverage_tolerance = 0.03` of 0.80 / 0.95.
3. **Convergence** (econ): the fit becomes *converged* — R-hat < 1.01,
   ESS > 400, 0 divergences. If econ only converges under `c1c2`, that is the
   adopted variant for econ (the family explicitly tests the interaction).
4. **Control**: physics stays default-off and bit-identical (it is not
   adopted; it exists to catch a silent regression).

If a domain meets none of these, it keeps the published default-off model and
its REPORT row is annotated "upgrade evaluated, not adopted (evidence: …)".

## Results (fill from `entity_overdispersion.json`)

### IMDb episodes — target: coverage → nominal

Bake-off at 2×500 / max_albums 300, **fair eval** (`entity_obs_raw` kept,
n_artists 1,889 ≤ cap, so forward LOO/coverage condition on each series' fitted
overdispersion):

| variant | LOO ELPD ± SE | ΔLOO vs off | div | cov80 | cov95 |
|---|---|---|---|---|---|
| off | −2487 ± 72 | — | 0 | 0.684 | 0.861 |
| c1_0.25 | −2922 ± 75 | **−435 (large)** | 0 | 0.657 | **0.882** |
| c1_0.5 | −2925 ± 75 | −438 (large) | 0 | 0.660 | 0.881 |

The fair eval landed essentially on top of the first (unfair) run (c1_0.25 LOO
−2922 vs −2923; coverage within noise), so the prior-marginalization caveat was
**not** the explanation — the LOO regression is real.

**Verdict: REJECT under the locked LOO gate (robust).** C1 is a clean
calibration-vs-sharpness trade: it widens the tails (95% coverage 0.861 → 0.882,
toward nominal — the intended fix) but lowers the bulk predictive density, so
LOO drops ~430 nats and 80% coverage does not improve. By the acceptance gate
("LOO ≥ baseline + 2 SE where it helps") C1 fails. Worth surfacing for the
record: IMDb is a calibration-priority domain, and C1 *does* improve calibration
— if the project ever prioritizes interval coverage over LOO sharpness for this
one domain, c1_0.25 is the lever, and it stays in the codebase default-off for
exactly that. As the gate is written today, **not adopted**, so IMDb's published
numbers stay bit-identical.

**Methodology fix applied (and it confirmed, not flipped, the verdict).** A
suspected confound — the first run marginalized each *seen* series'
`entity_obs_raw` from its prior (cold-start treatment) because the harness
dropped that plate — was fixed (commit `fix(eval): condition on fitted entity
overdispersion…`): the harness (`--entity-obs-keep-max`) and the train stage
(`_ENTITY_OBS_KEEP_MAX`) now keep `entity_obs_raw` when `n_artists ≤ 20000`, so
forward LOO/coverage condition on the *fitted* per-series factor. The gate-off
path never creates the site, so parity is untouched. Re-running IMDb under the
fair eval landed on the same numbers (table above) — so the LOO regression is
real, and the reject is now robust rather than caveated.

### Economics — target: converged fit (R-hat<1.01, ESS>400, 0-div)

Bake-off at 4×500 / max_albums 200 (`science_impact_econ`, 49,244 rows / 831
economists), watched sites σ_obs / σ_artist / σ_rw:

| variant | R-hat (σ_obs/σ_art/σ_rw) | ESS(min) | div | LOO ELPD ± SE | cov80 |
|---|---|---|---|---|---|
| off | 3.06 / 3.77 / 3.42 | 4 | 0 | −21830 ± 23 | 1.00 |
| c2_lognormal | 2.67 / 3.86 / 4.56 | 4 | 0 | −21764 ± 23 | 1.00 |
| c1c2 | 3.10 / 3.35 / 3.32 | 4 | 172 | −3195 ± 186 | 0.98 |

(Production reference: `off` at 4×2500 was rhat 27.8 / ESS 4 / σ_obs→0.004 —
the cheap 4×500 run reproduces the same collapse at lower amplitude: ESS 4–5,
multi-modal chains, σ_obs still degenerate.)

**Verdict: REJECT for economics.** No variant reaches the acceptance gate
(converged, 0-div). c2_lognormal alone does *not* fix the collapse (ESS still 4);
c1c2's huge ELPD jump is a pathological/unstable mode (172 divergences, an
overflow in the LOO weights), not a real win, and it over-covers (cov 0.98/1.00)
just like the collapsed baseline. The econ failure is therefore **deeper than the
σ_obs zero-boundary artifact** these gates address — it is a likelihood / zero-
inflation mismatch (econ log-citations are far more zero-inflated than physics).
The fix lives in the *deferred* "skew/heavier-tail or zero-inflated likelihood"
track, not in C1/C2. The cross-field Q/random-impact verdict stays **inconclusive**
for economics. (Caveat: cheap 4×500; but ESS 4–5 across every variant is
unambiguous regardless of sample count.)

### Physics — control: must not regress
| variant | LOO ELPD ± SE | ΔLOO vs off | R-hat | div |
|---|---|---|---|---|
| off | _pending_ | — | | |
| c1_0.25 | | | | |
| c2_lognormal | | | | |
| c1c2 | | | | |

## Per-domain verdict

- **IMDb episodes:** **REJECT under the locked LOO gate (robust).** Confirmed on
  the fair eval (conditioning on each series' fitted overdispersion): C1 improves
  95% coverage 0.861 → 0.882 (toward nominal) but worsens LOO ~430 nats and
  doesn't fix 80% coverage — a calibration-vs-sharpness trade the LOO gate
  rejects. The gate stays in the codebase default-off; if IMDb is ever treated
  as calibration-priority, c1_0.25 is the documented lever. Not adopted, so IMDb
  numbers stay bit-identical.
- **Economics:** **REJECT** — no variant converges (ESS 4–5, all rhat ≫ 1.01;
  c1c2 adds 172 divergences). The collapse is a likelihood/zero-inflation
  mismatch, out of scope for these noise-scale gates; deferred to the
  heavier-tail/zero-inflated-likelihood track. Econ stays default-off and its
  cross-field verdict stays inconclusive.
- **Physics:** default-off retained (control); bake-off pending to confirm no
  regression.

## Parity guarantee

Every domain not adopted here (physics, AOTY, games, BGG, Pitchfork,
science-physics, IMDb directors) keeps `heteroscedastic_entity_obs=False` and
`sigma_obs_prior_type=halfnormal`, so its fitted numbers are bit-identical to
the published study — proven by the parity-lock test, which shows the gate-off
forward-draw sequence is unchanged and the gate-on branch only appends sites
after every existing one.
