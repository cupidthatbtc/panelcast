> This document predates a later issue renumbering: what it calls #14 is now
> [#30](https://github.com/cupidthatbtc/panelcast/issues/30). The "latent-state AR"
> design it describes in §2 is the design that ultimately shipped as the
> errors-in-variables regressor gate (`errors_in_variables` in `PipelineConfig`).
> Kept as a historical record; see current docs for the shipped behavior.

# Response to the model-spec review

Five spec-level points were raised against `models/bayes/model.py` and the
transform/prior seams. Each is grounded against the code and the on-disk run
history below. Two are credited and turned into limitations; one is corrected
with a real experiment; two are confirmed and documented. The substantive new
work is the `offset_logit × ar1` experiment that closes the one cell every prior
verdict had skipped — results in
[`.audit/transform_latent_bakeoff/comparison.md`](transform_latent_bakeoff/comparison.md)
and `docs/LIKELIHOOD_CANDIDATES.md` (§ Transform × latent process).

Promoting a new default is **out of scope**: it would re-baseline every
published number and regenerate the golden fixtures, so it is gated as its own
decision (`docs/DECISIONS_TO_LOCK.md`, Reproducibility).

---

## 1. "`offset_logit` is the un-tried in-tree fix for the pinned skew/max" — corrected

**Largely refuted by the repo, with one genuine gap now closed.** `offset_logit`
was already implemented, evaluated, and HELD (`docs/DECISIONS_TO_LOCK.md:48`,
`docs/LIKELIHOOD_CANDIDATES.md`, `MODEL_CARD.md:208`), and bounded-support
likelihoods (`beta`, `beta_binomial`, `skew_normal`, `split_normal`, `mixture`)
each pinned *more* PPC statistics, not fewer. The real gap: every HELD verdict
used `latent_process = rw`, so the `offset_logit × ar1` combination the review
named had **never actually been fit** (confirmed: zero `offset_logit` runs in any
`outputs/*/manifest.json`; `ar1` never run in a real bake-off).

That cell is now closed with a real 2×2 grid (likelihood fixed `studentt`,
diagnostic 4×1000, same ~5k subset; `scripts/bakeoff_transform_latent.py`):

- `offset_logit × rw` **mixes** at diagnostic scale (R-hat 1.01, 0 divergences) —
  richer than the cheap 2×500 "fails to mix" note — but ~10× slower (maxes tree
  depth) and **does not move the structural pins**: `skewness` worsens to a hard
  1.000, `max` (0.9995) and `q90` (0.9995) stay pinned. It relieves the
  integer-heaping `q50` pin (0.008 → 0.057) only to **newly pin `q10`** (0.060 →
  0.003) — a reshuffle of the lower tail, not a fix for the bounded-skew triplet.
- `offset_logit × ar1` (the named combination) was the most pathological geometry
  of the grid (~2 h/chain, ~8 h total, maxing tree depth) yet still converges
  (R-hat 1.01) to the **exact same four pins as `offset_logit × rw`** at the
  grid's worst bulk ESS (477): `ar1` adds nothing on top of the transform.

**Disposition: the loophole is closed.** No transform/latent cell pulls
`skewness`/`max`/`q90` to the interior; the mismatch is structural, consistent
with the six likelihood families. `target_transform = identity` stays.

## 2. Errors-in-variables on the AR(1) predictor — credited (valid, unacknowledged)

The AR term `ar_term = rho * (prev_score - ar_center)` (`model.py:668`) regresses
on the **observed** lagged score as if noise-free, while the *same quantity* is
modeled as review-count-noisy when it is the response
(`compute_sigma_scaled`, `model.py:70`). Conditioning on a noisy regressor
attenuates `rho` toward zero, worst for sparse-review entities. This is real and
was unacknowledged.

**Disposition:** documented as a limitation (`MODEL_CARD.md`, Limitations) and
the predictor-side asymmetry named in the `TODO(model-v2)` at `model.py:135`; the
principled fix (latent-state AR carrying each entity's true level with its own
uncertainty) is a larger change tracked under issue #14. The response-side
single-review handling was already tracked there.

## 3. RW vs stationary AR(1) default — answered

The grid's `ar1` cells answer "does ar1 help at all?": **no.** On the default
transform, `identity × ar1` drops bulk ESS 802 → 577, tips `skewness` just over
the >0.99 flag (within the borderline noise the existing waves note), and leaves
MAE/RMSE/coverage unchanged. It buys no PPC or predictive gain at a real mixing
cost.

**Disposition:** rationale recorded (`docs/DECISIONS_TO_LOCK.md:55`); `ar1` stays
gated for the LOO-clear-win condition, `latent_process = rw` stays the default.

## 4. `seq_idx` clipping understates long-horizon variance — confirmed

Confirmed in code: `seq_idx = clip(album_seq - 1, 0, max_seq - 1)` (`model.py:654`)
reuses the final latent step beyond the longest training trajectory, and
`predict.py` appends no random-walk innovations past `max_seq`, so a forecast `h`
steps beyond the horizon omits `(h - max_seq)·sigma_rw²` of accumulated RW
variance. Negligible for the one-step flagship use (next album); `--strict`
already blocks horizon extrapolation beyond trained sequence support.

**Disposition:** code comment at `model.py:654` and a `MODEL_CARD.md` Limitations
bullet; the code fix (propagating innovations beyond `max_seq`) is deferred to
issue #14.

## 5. Reproducibility capture — credited (accurate meta-point)

Accurate: the bit-identical-default contract (every new transform/likelihood/
latent option opt-in, parity-tested) keeps the legacy path the default, so the
published numbers and golden fixtures stay valid without re-running. The
trade-off is that adopting a better default is not a one-line flip.

**Disposition:** recorded in `docs/DECISIONS_TO_LOCK.md` (Reproducibility) — a
default change re-baselines every published metric and regenerates the golden
fixtures, so it is gated as its own decision rather than bundled with the
experiment that motivates it.
