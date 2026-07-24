"""The candidate space for `panelcast select`, enumerated from the code's own surfaces.

Every searchable option ("knob") is a :class:`PipelineConfig` field whose values
come from the ``config/gates.py`` Literals, the likelihood ``REGISTRY``, or a
gate boolean. The table is declarative so the sweep runner, the report, and the
CI guard (``tests/unit/select/test_space_guard.py``) all read one source of
truth; a gate or family added to the code without a row here fails CI.

Core principle (#78): an option frozen on AOTY is a domain verdict, not a
global one. Past verdicts travel as ``history`` metadata on each knob — they
annotate the report, they never prune. Pruning happens only on structural
incompatibility declared by the :class:`DatasetDescriptor` (or by the
likelihood spec itself).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from panelcast.config import gates
from panelcast.config.descriptor import DatasetDescriptor
from panelcast.models.bayes.likelihoods import all_likelihoods, find_likelihood

KnobKind = Literal["literal", "bool", "tristate"]

# Predicate signature: (value, descriptor, available_columns) -> compatible?
# ``available_columns`` is the prepared training split's columns when the
# caller has data in hand (the sweep runner does), None for a static check.
ValuePredicate = Callable[[Any, DatasetDescriptor, "frozenset[str] | None"], bool]


@dataclass(frozen=True)
class Knob:
    """One searchable pipeline option.

    ``values`` always lists the shipped default first — the sweep's neutral
    base is the current defaults, and OFAT varies each knob to the rest.
    ``active_when`` marks knobs that are inert unless another knob enables
    them (varying them alone would waste a fit); ``value_compatible`` is the
    structural-incompatibility predicate.
    """

    name: str
    kind: KnobKind
    values: tuple[Any, ...]
    default: Any
    affects_features: bool = False
    history: str | None = None
    active_when: Callable[[Mapping[str, Any]], bool] | None = None
    value_compatible: ValuePredicate | None = None


def _literal_values(alias: Any, default: Any) -> tuple[Any, ...]:
    args = get_args(alias)
    if default not in args:
        raise ValueError(f"default {default!r} not among Literal values {args}")
    return (default, *(a for a in args if a != default))


def _family_values() -> tuple[str, ...]:
    # Builtins keep REGISTRY order (arm order is golden-pinned for the
    # no-plugin path); plugin families join as a sorted, deterministic tail.
    from panelcast.models.bayes.likelihoods import REGISTRY

    extras = sorted(f for f in all_likelihoods() if f not in REGISTRY)
    return ("studentt", *(f for f in REGISTRY if f != "studentt"), *extras)


def _family_compatible(
    value: Any, descriptor: DatasetDescriptor, columns: frozenset[str] | None
) -> bool:
    if not all_likelihoods()[value].requires_aggregation_count:
        return True
    return descriptor.n_obs_is_aggregation_count


def _pooling_compatible(
    value: Any, descriptor: DatasetDescriptor, columns: frozenset[str] | None
) -> bool:
    # Only explicit True hard-fails on unsupported domains; None (auto) and
    # False degrade safely everywhere.
    if value is not True:
        return True
    if descriptor.entity_group_col is None:
        return False
    return columns is None or descriptor.entity_group_col in columns


def _ablation_compatible(group: str) -> ValuePredicate:
    # Turning a feature group off is only a real arm when the descriptor
    # defines that ablation group; otherwise False is a no-op fit.
    def check(
        value: Any, descriptor: DatasetDescriptor, columns: frozenset[str] | None
    ) -> bool:
        return bool(value) or group in descriptor.ablation_groups

    return check


KNOBS: tuple[Knob, ...] = (
    Knob(
        "target_transform",
        "literal",
        _literal_values(gates.TargetTransform, "offset_logit"),
        "offset_logit",
        history=(
            "offset_logit promoted 0.5.0 on the corrected #63 ledger "
            "(+22.2±4.5 held-out ELPD, z+4.9, seeds 42/43/44 — "
            ".audit/transform_latent_bakeoff/)"
        ),
    ),
    Knob(
        "latent_process",
        "literal",
        _literal_values(gates.LatentProcess, "rw"),
        "rw",
        history=(
            "ar1 held on AOTY (paired ELPD −2.2±0.93 vs rw, ESS drop — "
            ".audit/transform_latent_bakeoff/comparison.md)"
        ),
    ),
    Knob(
        "likelihood_family",
        "literal",
        _family_values(),
        "studentt",
        value_compatible=_family_compatible,
        history=(
            "studentt kept on AOTY; beta/skew/split/mixture/beta_binomial/"
            "beta_ceiling rejected or nulled there (docs/LIKELIHOOD_CANDIDATES.md, "
            ".audit/beta_ceiling/screening.md)"
        ),
    ),
    Knob(
        "sigma_obs_prior_type",
        "literal",
        _literal_values(gates.SigmaObsPriorType, "halfnormal"),
        "halfnormal",
        history=(
            "lognormal (C2) rejected on the econ study — did not fix the "
            "variance collapse (docs/decisions/entity_overdispersion.md)"
        ),
    ),
    Knob(
        "beta_prior_type",
        "literal",
        _literal_values(gates.BetaPriorType, "normal"),
        "normal",
        history=(
            "horseshoe added #155 as the coefficient-dilution response to the "
            "#76 cold-start covariate anomaly; untested on AOTY as of 0.11.0"
        ),
    ),
    Knob(
        "ar_center",
        "literal",
        _literal_values(gates.ArCenter, "global"),
        "global",
        history=(
            "global adopted on the AR-geometry fix: corr(rho, mu_artist) "
            "−0.997→+0.016, debut AR terms zeroed (docs/decisions/DECISIONS.md)"
        ),
    ),
    Knob(
        "debut_prev_score_source",
        "literal",
        _literal_values(gates.DebutPrevScoreSource, "train_mean"),
        "train_mean",
        history=(
            "train_mean adopted on leakage principle (train-split-only mean); "
            "dataset_stats kept as rollback (docs/decisions/DECISIONS.md)"
        ),
    ),
    Knob(
        "n_exponent_prior",
        "literal",
        _literal_values(gates.NExponentPrior, "logit-normal"),
        "logit-normal",
        active_when=lambda arm: bool(arm.get("learn_n_exponent")),
        history=(
            "prior family for the learned exponent only; the AOTY posterior "
            "concentrates at ≈0 under either (#44, "
            ".audit/n_exponent_posterior_analysis.md)"
        ),
    ),
    Knob(
        "learn_n_exponent",
        "bool",
        (False, True),
        False,
        history=(
            "null on AOTY: posterior 0.0099±0.0048 against a prior near 0.5 "
            "(#44, .audit/n_exponent_posterior_analysis.md)"
        ),
    ),
    Knob(
        "discretize_observation",
        "bool",
        (False, True),
        False,
        history=(
            "default-off on AOTY: relieves the q50 integer-heaping pin "
            "(p 0.009→0.082) but leaves the structural skewness/max/q90 pins "
            "(docs/LIKELIHOOD_CANDIDATES.md)"
        ),
    ),
    Knob(
        "heteroscedastic_entity_obs",
        "bool",
        (True, False),
        True,
        history=(
            "promoted to the AOTY default 0.13.0 (#238): held-out ELPD "
            "+29.8±7.0 (z+4.25), resolves the q10/q90 PPC pins, cleared under "
            "#237's coverage non-inferiority rule; three-seed subset confirmed "
            "(.audit/select_entityobs_confirm/). Rejected on IMDb/econ — a "
            "per-domain verdict (docs/decisions/entity_overdispersion.md)"
        ),
    ),
    Knob(
        "errors_in_variables",
        "bool",
        (False, True),
        False,
        history=(
            "null at publication scale on AOTY (|z|≈0.01 v1 vs v2 — "
            ".audit/model_v2_bakeoff/comparison.md)"
        ),
    ),
    Knob(
        "propagate_rw_horizon",
        "bool",
        (False, True),
        False,
        history=(
            "null on the AOTY subset — gate only acts past max_seq_train, "
            "measurable at full corpus (#15; .audit/model_v2_bakeoff/comparison.md)"
        ),
    ),
    Knob(
        "entity_group_pooling",
        "tristate",
        (None, True, False),
        None,
        value_compatible=_pooling_compatible,
        history=(
            "auto default since 0.6.0 (#85): cold-start MAE −0.135/R² +0.034 "
            "on AOTY, multi-seed confirmed (.audit/genre_pooling/)"
        ),
    ),
    Knob(
        "gbm_offset",
        "bool",
        (True, False),
        True,
        affects_features=True,
        history=(
            "promoted 0.6.0 (#86): MAE 5.65→5.31, R² +0.072, combined "
            "held-out ELPD +224.2±12.6 (z+17.9) with pooling "
            "(.audit/point_accuracy_gap/stacking_screening.md)"
        ),
    ),
    Knob(
        "impute_missing",
        "bool",
        (False, True),
        False,
        affects_features=True,
        history=(
            "added #158: train-median imputation + missingness indicators vs "
            "the legacy fillna(0); untested on AOTY as of 0.11.0 (features "
            "there are mostly complete)"
        ),
    ),
    Knob(
        "enable_genre",
        "bool",
        (True, False),
        True,
        affects_features=True,
        value_compatible=_ablation_compatible("genre"),
        history=(
            "on for AOTY; genre PCA is dead weight for GBM point accuracy but "
            "carries the pooling groups (.audit/point_accuracy_gap/"
            "gbm_feature_ablation.md)"
        ),
    ),
    Knob(
        "enable_artist",
        "bool",
        (True, False),
        True,
        affects_features=True,
        value_compatible=_ablation_compatible("artist"),
        history="on for AOTY; entity-history block carries most covariate signal",
    ),
    Knob(
        "enable_temporal",
        "bool",
        (True, False),
        True,
        affects_features=True,
        value_compatible=_ablation_compatible("temporal"),
        history="on for AOTY; never individually contested",
    ),
)

# PipelineConfig fields that look like gates (bool / Optional[bool] / gates
# Literal) but are deliberately NOT candidates. The guard test forces every
# such field to appear either in KNOBS or here — with a reason.
EXCLUDED_FIELDS: dict[str, str] = {
    "auto_priors": "prior locs become data-derived; the sweep varies priors explicitly",
    "period_effects": "domain-structure gate (#269): needs a descriptor period_col, not sweepable",
    "period_constraint": "identification constraint for the period block, varies with the domain",
    "sigma_period_scale": "prior scale for the period block, tied to the gate",
    "skip_existing": "workflow control, not a model option",
    "dry_run": "workflow control, not a model option",
    "strict": "convergence-warning policy, not a model option",
    "enforce_lockfile": "environment guard, not a model option",
    "verbose": "logging control, not a model option",
    "progress_bar": "execution mechanics; never affects outputs",
    "allow_divergences": "convergence-gate policy, not a model option",
    "chain_method": "execution mechanics; chain scheduling is statistically neutral",
    "exclude_rw_raw_from_collection": "memory-only; posterior parity guarded by tests",
    "evaluate_secondary_split": "evaluation scope, not a model option",
    "seed": "replication knob — multi-seed confirmation varies it, the search does not",
    "sigma_artist_prior_type": (
        "sampler-geometry remedy flipped on diagnostic evidence, not searched"
    ),
    "artist_effect_param": (
        "sampler-geometry reparameterization, not a model-search dimension"
    ),
    "init_strategy": "execution mechanics; initialization is statistically neutral",
    "sigma_rw_lognormal_loc": (
        "prior-scale calibration per domain, not a search dimension"
    ),
    "sigma_rw_lognormal_sigma": (
        "prior-scale calibration per domain, not a search dimension"
    ),
    "sigma_artist_lognormal_loc": (
        "prior-scale calibration per domain, not a search dimension"
    ),
    "sigma_artist_lognormal_sigma": (
        "prior-scale calibration per domain, not a search dimension"
    ),
    "rho_loc": (
        "prior-scale calibration per domain, not a search dimension"
    ),
    "rho_scale": (
        "prior-scale calibration per domain, not a search dimension"
    ),
    "conformal_calibration": (
        "post-hoc evaluation layer over the predictive; no refit, not a model option"
    ),
}


# Descriptor-deferred knobs (#268) are None on a bare PipelineConfig; the
# sweep's neutral base uses the pipeline fallbacks the AOTY orchestrator
# resolves to. A descriptor-declared value is a per-domain default the sweep
# does not model.
_SENTINEL_KNOB_FALLBACKS = {
    "likelihood_family": "studentt",
    "target_transform": "offset_logit",
}


def default_arm() -> dict[str, Any]:
    """The neutral base: every knob at its shipped default.

    Read from a live ``PipelineConfig`` so a default flip in a future cycle
    propagates here mechanically (the guard test pins table defaults to it).
    """
    from panelcast.pipelines.orchestrator import PipelineConfig

    cfg = PipelineConfig()
    arm = {knob.name: getattr(cfg, knob.name) for knob in KNOBS}
    for name, fallback in _SENTINEL_KNOB_FALLBACKS.items():
        if arm.get(name) is None:
            arm[name] = fallback
    return arm


def knob_is_active(knob: Knob, arm: Mapping[str, Any]) -> bool:
    """Whether varying ``knob`` is meaningful in the context of ``arm``."""
    return knob.active_when is None or knob.active_when(arm)


def enumerate_space(
    descriptor: DatasetDescriptor,
    available_columns: frozenset[str] | set[str] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Per-knob candidate values after structural pruning for one domain.

    A knob whose alternatives are all pruned still appears with just its
    surviving values — the report shows what was structurally excluded and
    why, rather than silently shrinking the table.
    """
    cols = frozenset(available_columns) if available_columns is not None else None
    return {
        knob.name: tuple(
            v
            for v in knob.values
            if knob.value_compatible is None or knob.value_compatible(v, descriptor, cols)
        )
        for knob in KNOBS
    }


def arm_conflicts(
    arm: Mapping[str, Any],
    descriptor: DatasetDescriptor,
    available_columns: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Structural violations in a composed arm (empty list = valid).

    Cross-knob constraints are derived from the likelihood specs, not
    restated: a family that declares ``requires_identity_transform`` or
    rejects discretization carries that constraint into every arm.
    """
    cols = frozenset(available_columns) if available_columns is not None else None
    by_name = {knob.name: knob for knob in KNOBS}

    conflicts = [f"unknown knob: {name}" for name in arm if name not in by_name]
    base = default_arm()
    merged = {**base, **{k: v for k, v in arm.items() if k in by_name}}

    for name, value in merged.items():
        knob = by_name[name]
        if value not in knob.values:
            conflicts.append(f"{name}: {value!r} is not a candidate value")
        elif knob.value_compatible is not None and not knob.value_compatible(
            value, descriptor, cols
        ):
            conflicts.append(
                f"{name}={value!r} is structurally incompatible with "
                f"dataset '{descriptor.name}'"
            )

    family = merged["likelihood_family"]
    spec = find_likelihood(family)
    transform = merged["target_transform"]
    if spec is not None:
        if spec.requires_identity_transform and transform != "identity":
            conflicts.append(
                f"likelihood_family='{family}' requires target_transform='identity' "
                f"(got '{transform}')"
            )
        if spec.samples_bare_phi and merged["latent_process"] == "ar1":
            conflicts.append(
                f"likelihood_family='{family}' conflicts with latent_process='ar1' "
                "(both sample a 'phi' site; site names must be unique)"
            )
        if merged["discretize_observation"] and not spec.supports_discretization:
            conflicts.append(
                f"discretize_observation is unsupported for likelihood_family='{family}'"
            )
        if not spec.uses_sigma:
            # Sigma-side knobs are inert for families that never read sigma;
            # such arms would score identically to the base arm while labeled
            # as different models. Only an arm that MOVES one off its default
            # is mislabeled that way — if a knob is inert at its default value,
            # every arm for this family inherits it and none of them conflict.
            for knob_name in ("learn_n_exponent", "heteroscedastic_entity_obs"):
                if merged[knob_name] != base[knob_name]:
                    conflicts.append(
                        f"{knob_name} has no effect with likelihood_family='{family}' "
                        "(the family ignores sigma)"
                    )
    if merged["discretize_observation"] and transform != "identity":
        conflicts.append(
            f"discretize_observation requires target_transform='identity' (got '{transform}')"
        )
    return conflicts
