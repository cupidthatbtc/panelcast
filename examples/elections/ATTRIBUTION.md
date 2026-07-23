# Elections example — data attribution

`results.csv` is a distilled derivative of:

- **MIT Election Data and Science Lab (MEDSL)**, *U.S. Senate statewide
  1976–2020* (with subsequent cycle updates), Harvard Dataverse,
  [doi:10.7910/DVN/PEJ5QU](https://doi.org/10.7910/DVN/PEJ5QU).
  **License: CC0 1.0** public-domain dedication (verified against the
  Dataverse record's license metadata). The distillation keeps two-party
  Democratic vote shares and two-party vote totals per state-cycle.
- **Demographic covariates** (`pct_*`, `median_*`, `log_pop`) are American
  Community Survey figures (U.S. Census Bureau, public domain) retrieved via
  the [Census Reporter](https://censusreporter.org/) API, single vintage.
- Region indicators are the Census Bureau's four-region classification.

Citation courtesy for MEDSL:

> MIT Election Data and Science Lab, 2017, "U.S. Senate statewide 1976–2020",
> https://doi.org/10.7910/DVN/PEJ5QU, Harvard Dataverse.

The file ships here so the worked example runs with no downloads, accounts,
or guestbooks. It is an **apparatus demonstration**, not an electoral
forecast — see the portability-≠-predictive-validity caveat in
`docs/EXTENSIBILITY.md` and the model card.
