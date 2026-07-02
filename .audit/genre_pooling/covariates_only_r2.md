# Covariates-only cold-start diagnostic (#41)

How much cold-start signal do the static covariates carry at all? Ridge
(alpha=1, z-scored features) fit on the entity_disjoint train split
(n=3,614), R² on its test split (n=799) — the same ~5k-album AOTY subset the
bake-offs use.

| feature set | n features | test R² |
| --- | --- | --- |
| all features | 32 | 0.083 |
| static only (history features dropped) | 24 | 0.081 |
| genre PCA only | 10 | 0.034 |

## Reading

The Bayesian model's cold-start R² is ~0.003 (model_v2 bake-off), while a
plain linear read of the covariates reaches ~0.08 — the model leaves nearly
all available cold-start signal on the table because an unseen entity draws
from the population distribution regardless of covariates. Genre alone
carries ~0.034 of that, which is the slice the `entity_group_pooling` gate
targets: a new entity from a seen genre starts at the genre's level instead
of the population mean. The remaining ~0.05 (album type, temporal, collab)
is out of scope for the gate — it flows through beta only within-entity.

Reproduce: sklearn Ridge over the split/feature parquets; script inline in
PR #41's discussion (deterministic, no MCMC).
