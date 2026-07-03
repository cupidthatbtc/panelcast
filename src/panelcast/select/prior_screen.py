"""Pre-sweep prior-predictive screen for `panelcast select` (#99, A5).

Before any arm is fit, draw from the prior alone under each candidate target
transform and check that the implied predictions land plausibly inside the
descriptor's ``target_bounds``. Priors were tuned on AOTY; a domain with
different units or spread deserves to hear about it before burning GPU hours,
not after. Flags are informational — they reorder nothing and prune nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from panelcast.config.descriptor import DatasetDescriptor
from panelcast.evaluation.prior_predictive import PriorPredictiveResult, run_prior_predictive
from panelcast.models.bayes.priors import PriorConfig
from panelcast.models.bayes.transforms import get_transform
from panelcast.select.space import enumerate_space

# The subset of prepare_model_data output the model callable accepts.
_MODEL_ARG_KEYS = (
    "artist_idx",
    "album_seq",
    "prev_score",
    "X",
    "y",
    "n_reviews",
    "n_artists",
    "ar_center",
)


@dataclass
class TransformScreen:
    """Prior-predictive verdict for one candidate transform."""

    transform: str
    fraction_in_bounds: float
    reasonable: bool
    summary: dict[str, float]
    checks: dict[str, dict[str, Any]]
    flags: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def _suggestions_from_result(
    result: PriorPredictiveResult,
    train_mean: float,
    train_sd: float,
) -> list[str]:
    """Scale-adjustment hints from train stats when a check fails.

    Suggestions speak in score-scale quantities the user can sanity-check;
    translating them into exact PriorConfig edits stays a human decision.
    """
    suggestions: list[str] = []
    if not result.reasonable:
        low, high = result.bounds
        suggestions.append(
            f"only {result.fraction_in_bounds:.0%} of prior-predictive mass lands in "
            f"[{low:g}, {high:g}] — the AOTY-tuned priors likely mis-scale this domain; "
            "review mu_artist/sigma_obs scales before sweeping"
        )
    checks = result.checks or {}
    mean_check = checks.get("mean", {})
    if mean_check and not mean_check.get("passed", True):
        pp_mean = float(mean_check["value"])
        suggestions.append(
            f"prior-predictive mean {pp_mean:.1f} vs train mean {train_mean:.1f} "
            f"(plausible range [{mean_check['low']:.1f}, {mean_check['high']:.1f}]); "
            f"a location shift of {train_mean - pp_mean:+.1f} on the score scale "
            "would recenter it"
        )
    sd_check = checks.get("sd", {})
    if sd_check and not sd_check.get("passed", True):
        pp_sd = float(sd_check["value"])
        if pp_sd > 0 and train_sd > 0:
            suggestions.append(
                f"prior-predictive sd {pp_sd:.1f} vs train sd {train_sd:.1f} "
                f"(plausible range [{sd_check['low']:.1f}, {sd_check['high']:.1f}]); "
                f"consider scaling the observation-noise priors by ~{train_sd / pp_sd:.2f}"
            )
    return suggestions


def screen_transforms(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    descriptor: DatasetDescriptor,
    transforms: tuple[str, ...] | None = None,
    n_samples: int = 200,
    max_obs: int = 2000,
    seed: int = 42,
    logit_offset: float = 0.5,
    max_albums: int = 50,
) -> list[TransformScreen]:
    """Run the prior-predictive check under each candidate transform.

    ``transforms=None`` takes the structurally surviving values from the
    candidate space. Data prep mirrors the production path per transform
    (prepare_model_data → max-albums cap → X standardization) so the screen
    exercises the same model_args a sweep arm would fit on.
    """
    from panelcast.models.bayes.model import make_score_model
    from panelcast.pipelines.train_bayes import _apply_max_albums_cap, prepare_model_data

    if transforms is None:
        transforms = enumerate_space(descriptor)["target_transform"]

    target_col = descriptor.target_col
    train_mean = float(train_df[target_col].mean())
    train_sd = float(train_df[target_col].std())
    model = make_score_model(descriptor.model_prefix)

    screens: list[TransformScreen] = []
    for name in transforms:
        model_args, _ = prepare_model_data(
            train_df,
            feature_cols,
            descriptor=descriptor,
            target_transform=name,
            logit_offset=logit_offset,
        )
        counts = model_args.pop("artist_album_counts")
        model_args = _apply_max_albums_cap(model_args, max_albums, counts)

        X = model_args["X"]
        std = X.std(axis=0)
        std_safe = np.where(std == 0.0, 1.0, std)
        model_args["X"] = ((X - X.mean(axis=0)) / std_safe).astype(np.float32)

        args = {k: model_args[k] for k in _MODEL_ARG_KEYS if k in model_args}
        args["max_seq"] = int(np.max(model_args["album_seq"]))
        args["priors"] = PriorConfig(target_transform=name)
        args["target_bounds"] = tuple(descriptor.target_bounds)

        result = run_prior_predictive(
            model,
            args,
            n_samples=n_samples,
            max_obs=max_obs,
            seed=seed,
            score_bounds=tuple(descriptor.target_bounds),
            transform=get_transform(name, tuple(descriptor.target_bounds), logit_offset),
        )
        screens.append(
            TransformScreen(
                transform=name,
                fraction_in_bounds=result.fraction_in_bounds,
                reasonable=result.reasonable,
                summary=result.summary,
                checks=result.checks or {},
                flags=list(result.informational_flags or []),
                suggestions=_suggestions_from_result(result, train_mean, train_sd),
            )
        )
    return screens


def render_prior_block(screens: list[TransformScreen]) -> tuple[str, dict]:
    """Markdown block + JSON payload for the sweep report (A6)."""
    lines = [
        "## Prior-predictive screen",
        "",
        "| transform | in-bounds | mean | sd | flags |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in screens:
        flag_text = "; ".join(s.flags) if s.flags else "none"
        lines.append(
            f"| {s.transform} | {s.fraction_in_bounds:.1%} | "
            f"{s.summary['mean']:.1f} | {s.summary['sd']:.1f} | {flag_text} |"
        )
    suggestions = [(s.transform, text) for s in screens for text in s.suggestions]
    if suggestions:
        lines.append("")
        lines.append("Suggestions (informational — nothing is pruned):")
        lines.extend(f"- `{t}`: {text}" for t, text in suggestions)

    payload = {
        "screens": [
            {
                "transform": s.transform,
                "fraction_in_bounds": s.fraction_in_bounds,
                "reasonable": s.reasonable,
                "summary": s.summary,
                "checks": s.checks,
                "flags": s.flags,
                "suggestions": s.suggestions,
            }
            for s in screens
        ]
    }
    return "\n".join(lines) + "\n", payload
