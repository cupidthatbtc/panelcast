"""`panelcast preflight`: pre-fit statistical sanity checks on prepared data.

Runs AFTER the features stage and BEFORE the fit. Two checks, warn-only by
default, that would have caught the two failure modes behind an 11-rung baseball
diagnostic ladder:

- Check A (prior/data scale): resolved sigma_rw / sigma_artist prior medians vs
  the data moments they govern, on the model-training scale. AOTY-scale sigma
  priors on a differently-scaled domain put the data 5-6 sigma into the prior
  tail, which made an all-sigmas-zero basin posterior-competitive.
- Check B (collinearity given entity intercepts): within-entity demeaned,
  standardized covariate matrix (plus cohort dummies when group pooling is
  active). Per-entity intercepts absorb per-entity constants, so time-like
  covariates (age, album_sequence, release_year, cohort) collapse into an
  age-period-cohort rank deficiency the raw correlation never reveals.

Distinct from `panelcast run --preflight` (GPU-memory estimation, see
``panelcast.preflight``): this audits the statistics of the fit, not its memory.

Nothing here touches MCMC or a GPU; the heaviest operation is an SVD of the
standardized covariate matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Check A: how far apart (in orders of magnitude, base-10) a prior median and
# its governing data moment may drift before the row warns / fails.
_WARN_ORDERS = 0.75
_FAIL_ORDERS = 1.5

# Check B bands on the RESIDUAL condition number — the condition number after
# machine-exact structural nulls are removed. The raw condition number is
# uninformative here: standard feature blocks carry benign exact redundancies
# (one-hot indicator blocks sum to a per-entity constant; a sequence index and
# its lagged count differ by one) that the N(0, 1) ridge prior on beta absorbs
# without hurting the fit. Stripping those exact nulls leaves the data-driven
# collinearity — a covariate the OTHERS nearly reconstruct — which is what the
# age-period-cohort trap creates. The two regimes are ~2 orders apart (healthy
# AOTY ~80, the baseball APC ~1.3e4), so the bands sit in the gap between them.
_COND_WARN = 2.5e2
_COND_FAIL = 2.0e3
# A singular value at/below this fraction of the largest is a machine-exact
# structural null (removed before the residual condition number is formed).
_EXACT_REL = 1.0e-10
# When flagged, name features loading on any singular vector at/below this
# fraction of the largest (covers the exact nulls and the near-null direction).
_NULL_REL = 1.0e-2
# Minimum standardized loading for a feature to be named in a null direction.
# A genuine member of a k-feature identity carries loading ~1/sqrt(k); unrelated
# columns sit near zero, so this cleanly separates participants from noise.
_LOADING_MIN = 0.15
# A column whose within-entity variation is this small (relative to its overall
# scale) is constant within entity -> fully absorbed by the entity intercept.
_DEGENERATE_REL = 1.0e-6


# Name of the single row returned when the fit inputs can't even be assembled
# (features not built yet). A setup error, not a statistical FAIL — the CLI maps
# it to a distinct exit code under --strict.
SETUP_ERROR_NAME = "preflight"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    suggestion: str | None = None  # ready-to-paste YAML / config note on a flag


def _classify_gap(gap_orders: float) -> str:
    mag = abs(gap_orders)
    if mag > _FAIL_ORDERS:
        return "FAIL"
    if mag > _WARN_ORDERS:
        return "WARN"
    return "PASS"


def within_entity_step_sd(y: np.ndarray, artist_idx: np.ndarray) -> float:
    """Pooled within-entity per-step SD of the target.

    Consecutive-observation diffs within each entity, pooled, divided by
    ``sqrt(2)``: a random walk's one-step innovation SD if the diffs were pure
    innovation. Observation noise and overdispersion inflate it, so it is an
    UPPER bound on the latent ``sigma_rw`` — Check A treats it asymmetrically.
    Entities are grouped by a stable sort, which preserves their in-array order;
    that order is temporal by the same pipeline invariant ``album_seq`` relies on
    (``groupby(entity).cumcount()`` in ``prepare_model_data``).
    """
    y = np.asarray(y, dtype=float)
    idx = np.asarray(artist_idx)
    if y.size < 2:
        return 0.0
    order = np.argsort(idx, kind="stable")
    ys = y[order]
    ids = idx[order]
    diffs = np.diff(ys)
    same_entity = ids[1:] == ids[:-1]
    within = diffs[same_entity]
    if within.size == 0:
        return 0.0
    return float(np.std(within) / math.sqrt(2.0))


def cross_entity_mean_sd(y: np.ndarray, artist_idx: np.ndarray) -> float:
    """SD of per-entity mean targets across entities (the scale sigma_artist governs)."""
    y = np.asarray(y, dtype=float)
    idx = np.asarray(artist_idx).astype(np.int64)
    if y.size == 0:
        return 0.0
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=y)
    present = counts > 0
    means = sums[present] / counts[present]
    if means.size < 2:
        return 0.0
    return float(np.std(means))


def _sigma_rw_prior_median(priors) -> float:
    if priors.sigma_rw_prior_type == "lognormal":
        return math.exp(priors.sigma_rw_lognormal_loc)
    return priors.sigma_rw_scale * 0.6744897501960817  # HalfNormal median


def _sigma_artist_prior_scale(priors) -> float:
    # LogNormal median when lognormal; the HalfNormal scale otherwise (its own
    # spread, the interpretable "typical between-entity SD" the prior expects).
    if priors.sigma_artist_prior_type == "lognormal":
        return math.exp(priors.sigma_artist_lognormal_loc)
    return priors.sigma_artist_scale


def _yaml_lognormal_block(loc_field: str, sigma_field: str, moment: float, sigma: float) -> str:
    loc = round(math.log(moment), 1)
    return f"{loc_field}: {loc}   # ln({moment:.3g}), data-derived\n{sigma_field}: {sigma}"


def check_prior_data_scale(
    *,
    y: np.ndarray,
    artist_idx: np.ndarray,
    priors,
) -> list[CheckResult]:
    """Compare resolved sigma priors against the data moments they govern.

    ``y`` must already be on the model-training scale (the same transform the
    fit applies), because the prior medians live on that scale.
    """
    results: list[CheckResult] = []

    rw_moment = within_entity_step_sd(y, artist_idx)
    rw_median = _sigma_rw_prior_median(priors)
    if rw_moment <= 0.0 or rw_median <= 0.0:
        results.append(
            CheckResult("sigma_rw scale", "WARN", "within-entity step SD is zero — cannot compare")
        )
    else:
        gap = math.log10(rw_median / rw_moment)  # >0: prior median above the moment
        # The moment bounds latent sigma_rw from above, so a prior BELOW it is
        # expected; only a prior far ABOVE, or >~1.5 orders BELOW, is a problem.
        if gap >= 0:
            status = _classify_gap(gap)
        else:
            status = "FAIL" if abs(gap) > _FAIL_ORDERS else "PASS"
        detail = (
            f"prior median {rw_median:.3g} vs within-entity step SD {rw_moment:.3g} "
            f"(log10 gap {gap:+.2f})"
        )
        suggestion = None
        if status != "PASS":
            suggestion = _yaml_lognormal_block(
                "sigma_rw_lognormal_loc", "sigma_rw_lognormal_sigma", rw_moment, 0.8
            )
        results.append(CheckResult("sigma_rw scale", status, detail, suggestion))

    art_moment = cross_entity_mean_sd(y, artist_idx)
    art_scale = _sigma_artist_prior_scale(priors)
    if art_moment <= 0.0 or art_scale <= 0.0:
        results.append(
            CheckResult(
                "sigma_artist scale", "WARN", "cross-entity mean SD is zero — cannot compare"
            )
        )
    else:
        gap = math.log10(art_scale / art_moment)
        status = _classify_gap(gap)
        detail = (
            f"prior scale {art_scale:.3g} vs cross-entity mean SD {art_moment:.3g} "
            f"(log10 gap {gap:+.2f})"
        )
        suggestion = None
        if status != "PASS":
            suggestion = (
                _yaml_lognormal_block(
                    "sigma_artist_lognormal_loc",
                    "sigma_artist_lognormal_sigma",
                    art_moment,
                    0.6,
                )
                + "\nsigma_artist_prior_type: lognormal"
            )
        results.append(CheckResult("sigma_artist scale", status, detail, suggestion))

    return results


def _within_entity_demean(col: np.ndarray, idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sums = np.bincount(idx, weights=col, minlength=counts.size)
    entity_mean = sums / counts
    return col - entity_mean[idx]


def check_collinearity(
    *,
    X: np.ndarray,
    artist_idx: np.ndarray,
    feature_names: list[str],
    group_idx_by_artist: np.ndarray | None = None,
) -> CheckResult:
    """Residual-condition-number audit of the covariates given entity intercepts.

    Within-entity demeaning removes each entity's mean, mimicking what the
    per-entity intercepts absorb; columns constant within entity are then fully
    absorbed and set aside. Standardizing puts the rest on equal footing, and
    the SVD's machine-exact null directions (benign one-hot / sequence-count
    redundancies the ridge prior soaks up) are stripped before the condition
    number is formed — so what remains measures the data-driven collinearity
    the age-period-cohort trap produces. Cohort dummies (when group pooling is
    active) are appended so they enter the absorbed set alongside the entity
    intercepts they duplicate.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return CheckResult("collinearity", "WARN", "no covariate columns to check")
    idx = np.asarray(artist_idx).astype(np.int64)
    counts = np.bincount(idx)
    counts_safe = np.where(counts == 0, 1, counts)

    columns = [X[:, j] for j in range(X.shape[1])]
    names = list(feature_names)

    if group_idx_by_artist is not None:
        cohort = np.asarray(group_idx_by_artist)[idx]
        for g in np.unique(cohort):
            columns.append((cohort == g).astype(float))
            names.append(f"cohort[{int(g)}]")

    n_absorbed = 0  # constant within entity -> absorbed by the intercepts
    M_cols = []
    kept_names: list[str] = []
    for name, col in zip(names, columns, strict=True):
        raw_scale = float(np.std(col)) or 1.0
        dm = _within_entity_demean(col, idx, counts_safe)
        within_scale = float(np.std(dm))
        if within_scale <= _DEGENERATE_REL * raw_scale:
            n_absorbed += 1
            continue
        M_cols.append(dm / within_scale)
        kept_names.append(name)

    if len(M_cols) < 2:
        return CheckResult(
            "collinearity",
            "PASS",
            f"condition number 1 ({len(kept_names)} varying columns, "
            f"{n_absorbed} absorbed by intercepts)",
        )

    M = np.column_stack(M_cols)
    _, svals, Vt = np.linalg.svd(M, full_matrices=False)
    s_max = float(svals[0])
    exact_cut = _EXACT_REL * s_max
    residual = svals[svals > exact_cut]
    n_exact = int(len(svals) - len(residual))
    residual_min = float(residual[-1]) if len(residual) else s_max
    cond = s_max / residual_min if residual_min > 0 else math.inf

    if cond >= _COND_FAIL:
        status = "FAIL"
    elif cond >= _COND_WARN:
        status = "WARN"
    else:
        status = "PASS"

    cond_str = "inf" if cond == math.inf else f"{cond:.1f}"
    detail = (
        f"residual condition number {cond_str} ({len(kept_names)} varying "
        f"columns, {n_absorbed} absorbed, {n_exact} exact-redundant)"
    )
    suggestion = None
    if status != "PASS":
        participating: set[str] = set()
        null_cut = _NULL_REL * s_max
        for i, s in enumerate(svals):
            if s <= null_cut:
                for j in np.where(np.abs(Vt[i]) > _LOADING_MIN)[0]:
                    participating.add(kept_names[j])
        if participating:
            offenders = ", ".join(sorted(participating))
            detail += f"; near-collinear set: {offenders}"
            suggestion = (
                f"{offenders} form an approximate identity given per-entity "
                "intercepts (age-period-cohort). Drop all but one, or remove the "
                "redundant feature block(s) from the descriptor."
            )
    return CheckResult("collinearity", status, detail, suggestion)


def run_model_preflight(
    dataset: str | None = None,
    config_files: list[str] | None = None,
) -> list[CheckResult]:
    """Assemble the exact fit inputs and run both checks. Never raises."""
    results: list[CheckResult] = []
    try:
        from panelcast.model_preflight_data import assemble_preflight_inputs

        inputs = assemble_preflight_inputs(dataset, config_files)
    except Exception as exc:
        return [CheckResult(SETUP_ERROR_NAME, "FAIL", f"could not assemble fit inputs: {exc}")]

    results.extend(
        check_prior_data_scale(
            y=inputs.y,
            artist_idx=inputs.artist_idx,
            priors=inputs.priors,
        )
    )
    results.append(
        check_collinearity(
            X=inputs.X,
            artist_idx=inputs.artist_idx,
            feature_names=inputs.feature_names,
            group_idx_by_artist=inputs.group_idx_by_artist,
        )
    )
    return results
