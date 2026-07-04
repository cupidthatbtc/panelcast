# New-domain playbook: descriptor → select → confirm → publish

[PORTING.md](PORTING.md) gets a new domain *running* — one YAML descriptor, zero
source changes. This playbook gets it *tuned*: which target transform, likelihood
family, and gates actually fit the domain, chosen by the same discipline the AOTY
model was tuned with, run as one command (`panelcast select`).

The core principle: **an option that lost on AOTY is a domain verdict, not a
global one.** No-free-lunch means the *protocol* is what ports, not the winners.
So `select` enumerates its candidate space from the code's own registries — every
likelihood family, transform, and gate — and re-tries all of them. Past AOTY
verdicts travel as metadata in the report (with `.audit` links), never as pruning.
Pruning happens only on *structural* incompatibility a descriptor declares (a
bounded family needs the identity transform; `beta_binomial` needs a real
aggregation count; genre-style pooling needs a group column).

## 1. Descriptor

Write the descriptor as in [PORTING.md](PORTING.md). Two fields specifically shape
the candidate space:

- `n_obs_is_aggregation_count` — `true` only when `n_obs_col` counts independent
  raters whose mean *is* the target (then `beta_binomial` is a candidate). A
  sensor/sample count that doesn't average to the score → `false`.
- `entity_group_col` — a per-event group (AOTY: `primary_genre`) enables the
  `entity_group_pooling` gate. `null` if the domain has no such grouping.

## 2. Dry run — see the space and the cost

```
panelcast select --dataset configs/datasets/<name>.yaml --dry-run
```

Prints the full registry-enumerated candidate space (with any structurally pruned
options called out, never silently dropped), the staged plan (how many fits), and
— once splits+features exist — the predicted GPU cost from this machine's own
calibration history. This is the informed-consent step; nothing runs.

## 3. Select — run the staged sweep

```
# prepare the data once (the sweep rebuilds features per feature-affecting arm)
panelcast run --dataset <name> --stages splits,features

panelcast select --dataset <name> --effort standard
```

Effort tiers (defined in `configs/select.yaml`, override per domain):

| tier | search | sampler | confirmation |
|---|---|---|---|
| `quick` | stage 1 (one-factor-at-a-time) | reduced | none |
| `standard` | stages 1–2 (+ winner interactions) | diagnostic | multi-seed |
| `thorough` | stages 1–3 (+ random backstop) | diagnostic | multi-seed + publication fit |

`--max-fits N` / `--budget-hours H` cap the run; stages are priority-ordered, so a
truncated sweep still covers the highest-value arms first. `--resume` (same
`--sweep-id`) skips completed arms — a long sweep can be split across sessions.

The sweep is **strictly serial**: `data/features` is a flat cross-run cache, so a
feature-affecting arm rebuilds it before fitting and the stamps fail fast if
anything else touches the repo mid-sweep. Keep the checkout to the sweep for its
duration (or run it in a dedicated worktree).

## 4. Read the report

One report lands at `.audit/select_<name>/report.md` (+ `report.json`) — it *is*
the domain's `.audit` entry. Per arm: paired held-out ELPD ± SE vs the reference
(shipped defaults), calibration, PPC pins, convergence, wall-clock. Arms without a
pointwise snapshot show `-` — no other estimator is substituted. A **baseline
floor** section says whether the structured model beats the GBM at all, and the
**promotion verdicts** apply the pre-registered rules (`configs/select.yaml`):
paired-ELPD z ≥ threshold, coverage within tolerance, convergence pass.

Diagnostic-scale fits can mislead on slow-mixing candidates (ESS below the gate at
the diagnostic tier) — the report carries per-arm convergence caveats. That is what
the confirmation stage guards against.

## 5. Confirm — the winner survives multiple seeds

A promotable winner triggers multi-seed confirmation automatically (standard/
thorough): the reference and the winner are re-fit on each `confirmation_seed`, and
the winner confirms only when the direction holds at threshold on **every** seed. A
single-seed z is one draw from the selection lottery; this is the guardrail the
invalid-LOO episode taught. `--effort thorough` adds a publication-scale
confirmation fit. Confirmation lands in `.audit/select_<name>/confirmation.json`.

## 6. Publish — the flip is a manual PR

`select` **recommends**; it never flips a default. Promoting an option to a shipped
default is a manual PR that cites the report + confirmation as its evidence — the
same flip discipline the AOTY defaults were promoted under. This keeps a human in
the loop between "the sweep found an effect" and "the tool ships it."

## Reproducing the AOTY history

`panelcast select --dry-run` on AOTY prints the full space, proving the frozen
options (ar1, the rejected families, EIV, …) are genuinely re-tried. A full AOTY
sweep should reproduce the 0.6.0-era conclusions — offset_logit, gbm_offset, and
pooling promoted; the families and ar1 frozen — from the whole space, not from a
pre-pruned one. That reproduction is the headline validation of the feature.
