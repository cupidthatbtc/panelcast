# Repository lineage

> Not to be confused with [`DATA_LINEAGE.md`](DATA_LINEAGE.md), which documents
> the data pipeline's lineage. This page is about the repository's history.

The public panelcast history begins with scaffolding commits on 2026-06-20.
That is a migration boundary, not the start of the project: panelcast is the
renamed, generalized continuation of a private research repository, and this
page documents the lineage so an automated check (or a JOSS reviewer) does not
misread the root import as a code dump.

## Predecessor: `aoty_pred_pub`

- **What it was:** a private hierarchical Bayesian pipeline for predicting
  Album of the Year user scores — the flagship AOTY domain that still ships as
  panelcast's built-in defaults.
- **History:** development from **2026-01-01 to 2026-06-14**, ~470 commits at
  migration time. The private history contains the original modeling
  iterations (growth-curve models superseded by the hierarchical random walk),
  the leakage-control and evaluation-protocol work, and the domain-descriptor
  refactor that made generalization possible.
- **Why the history was not carried over:** the private repository interleaves
  research artifacts, data paths, and dead ends that predate the public data
  contract. The migration squashed it into reviewed scaffolding commits rather
  than rewriting or backdating history — the public graph is honest about when
  the public project started.

## Migration to panelcast (2026-06-20)

The public repository was created on **2026-06-20** and the initial commits
land the generalized codebase in reviewed slices (packaging, shared IO,
pipeline stages, model, evaluation, reporting). What changed in the
generalization:

- every dataset-specific literal moved into `DatasetDescriptor`
  (default-equals-AOTY, one YAML per new domain — see `docs/PORTING.md`);
- the feature system became a registry of descriptor-declared blocks with
  domain packs;
- split, artifact, and manifest naming moved to entity/event terms with
  compatibility shims.

## Public development since

All development since 2026-06-20 has happened in public on
[github.com/cupidthatbtc/panelcast](https://github.com/cupidthatbtc/panelcast):
600+ commits, 300+ tracked issues, PR-reviewed changes with CI gates, and
releases 0.1.0 through the current version (see `CHANGELOG.md`). Companion
public repository:
[panelcast-replications](https://github.com/cupidthatbtc/panelcast-replications)
(re-analyses of published panel studies through this pipeline).

## JOSS submission timing

JOSS expects a substantial public track record, not a freshly imported
codebase. Policy for this repository:

- **No submission earlier than 2026-12-21** (six months of public
  development).
- **Preferred window: January–March 2027**, after sustained public use and the
  full-corpus validation (#15) has either landed or been explicitly scoped out
  of the software claim (see `MODEL_CARD.md` for the software-vs-domain-model
  claim separation).

`tests/unit/test_lineage_doc.py` keeps this page linked from the README and
CONTRIBUTING and keeps the dates above from silently rotting.
