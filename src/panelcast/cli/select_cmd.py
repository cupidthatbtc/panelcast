"""The `panelcast select` command (#103, A7+A8).

Runs the portable model-selection protocol: prints a pre-run plan with predicted
cost for informed consent, then (unless --dry-run) drives the staged sweep,
scores every arm, applies the pre-registered rules, and writes one ranked report
that is the domain's `.audit/` entry. `select` RECOMMENDS; a default flip stays
a manual PR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from panelcast.cli import app

_DEFAULT_CONFIG = "configs/select.yaml"


@app.command("select")
def select(
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Dataset descriptor (bare name or YAML path; omit for AOTY defaults).",
    ),
    effort: str = typer.Option(
        "standard",
        "--effort",
        help="Effort tier: quick (screen), standard (default), thorough (+random +publication).",
    ),
    max_fits: int | None = typer.Option(
        None, "--max-fits", min=1, help="Hard cap on diagnostic fits (overrides the tier)."
    ),
    budget_hours: float | None = typer.Option(
        None, "--budget-hours", min=0.1, help="GPU-hour budget; stages truncate in priority order."
    ),
    arm_timeout: str = typer.Option(
        "1800",
        "--arm-timeout",
        help=(
            "Per-arm wall-clock timeout in seconds, or 'auto' to size each "
            "arm's timeout from its predicted runtime; a fit exceeding it is "
            "killed and marked failed."
        ),
    ),
    sweep_id: str = typer.Option(
        "sweep", "--sweep-id", help="Sweep directory name under outputs/select/ (enables --resume)."
    ),
    config_path: str = typer.Option(
        _DEFAULT_CONFIG, "--config", help="YAML with the rules and effort tiers."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the enumerated space, staged plan, and predicted cost only."
    ),
) -> None:
    """Select transform / likelihood / gates for a domain from the full candidate space.

    The candidate space is enumerated from the code's own registries, so a frozen
    option is genuinely re-tried, not pre-pruned; pruning is structural only.

    Examples:
        panelcast select --dry-run
        panelcast select --dataset examples/aerospace/descriptor.yaml --effort quick
    """
    from panelcast.config.descriptor import load_descriptor
    from panelcast.select.orchestrate import build_plan, render_plan, resolve_dims, run_select
    from panelcast.select.rules import DecisionRules
    from panelcast.select.tiers import resolve_tier, tier_to_sweep_config

    cfg_path = Path(config_path)
    # A typo'd explicit --config must not silently run under shipped defaults;
    # only the shipped default path may be absent (fresh domain repo).
    if not cfg_path.exists() and config_path != _DEFAULT_CONFIG:
        raise typer.BadParameter(f"config file not found: {config_path}", param_hint="--config")
    try:
        tier = resolve_tier(effort, cfg_path)
        rules = DecisionRules.load(cfg_path)
    except ValueError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    descriptor = load_descriptor(dataset)
    label = dataset or descriptor.name

    cfg = tier_to_sweep_config(
        tier,
        sweep_id=sweep_id,
        dataset=dataset,
        max_fits=max_fits,
        budget_hours=budget_hours,
        promote_z=rules.promote_z,
        arm_timeout_seconds=_parse_arm_timeout(arm_timeout),
    )

    dims = resolve_dims(_prepared_paths(descriptor))
    plan = build_plan(
        descriptor,
        tier,
        cfg,
        dataset_label=label,
        n_confirmation_seeds=len(rules.confirmation_seeds),
        dims=dims,
    )
    plan.notes.append(
        f"pre-registered rules in effect: promote_z={rules.promote_z:g}, "
        f"coverage tolerance ±{rules.coverage_tolerance:g}, convergence "
        f"{'required' if rules.require_convergence else 'not required'}, "
        f"confirmation seeds {list(rules.confirmation_seeds)}"
    )
    typer.echo(render_plan(plan))

    if dry_run:
        raise typer.Exit(code=0)

    train_df, feature_cols = _load_prepared_frame(descriptor)
    if train_df is None:
        typer.echo(
            "No prepared splits/features found. The sweep rebuilds them per arm, "
            "but the prior-predictive screen and data diagnostics are skipped.\n"
        )

    result = run_select(
        dataset,
        tier,
        rules,
        cfg,
        train_df=train_df,
        feature_cols=feature_cols,
        available_columns=frozenset(train_df.columns) if train_df is not None else None,
        dims=dims,
    )
    typer.echo(f"\nSweep complete. Report: {result['report_dir']}/report.md")
    if result["winner_arm"]:
        typer.echo(
            f"Recommended (pre-registered rules + multi-seed confirmation): arm "
            f"{result['winner_arm']}. A default flip is a manual PR with the report as evidence."
        )
    elif result.get("promotable"):
        typer.echo(
            f"Candidate(s) {', '.join(result['promotable'])} cleared the rules but did not "
            "survive multi-seed confirmation; defaults hold."
        )
    else:
        typer.echo("No candidate cleared the pre-registered bar; defaults hold.")


def _parse_arm_timeout(value: str) -> float | str:
    """Seconds as a number, or the literal 'auto' (predicted-runtime timeouts)."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        seconds = float(value)
    except ValueError:
        raise typer.BadParameter(
            f"expected a number of seconds or 'auto', got {value!r}",
            param_hint="--arm-timeout",
        ) from None
    if seconds < 1.0:
        raise typer.BadParameter("must be at least 1 second", param_hint="--arm-timeout")
    return seconds


_FEATURES_PATH = Path("data/features/train_features.parquet")
_SPLIT_PATH = Path("data/splits/within_entity_temporal/train.parquet")


def _prepared_paths(descriptor) -> dict | None:
    """Feature path plus an entity-count hint (from the split, via the descriptor)."""
    if not _FEATURES_PATH.exists():
        return None
    hint: dict = {"features": _FEATURES_PATH}
    if _SPLIT_PATH.exists():
        try:
            import pandas as pd

            split_df = pd.read_parquet(_SPLIT_PATH, columns=[descriptor.entity_col])
            hint["n_artists"] = int(split_df[descriptor.entity_col].nunique())
        except (OSError, ValueError, KeyError):
            pass
    return hint


def _load_prepared_frame(descriptor) -> tuple[Any, list[str] | None]:
    """(joined train frame, feature_cols) from prepared artifacts, or (None, None).

    The flat ``data/`` artifacts are shared across domains: a frame left behind
    by another domain's run would crash (or silently mis-screen) the prior
    screen, so a frame missing the descriptor's own columns is treated as not
    prepared rather than passed through.
    """
    if not (_SPLIT_PATH.exists() and _FEATURES_PATH.exists()):
        return None, None
    try:
        import pandas as pd

        from panelcast.data.alignment import join_splits_with_features

        split_df = pd.read_parquet(_SPLIT_PATH)
        features_df = pd.read_parquet(_FEATURES_PATH)
        joined = join_splits_with_features(split_df, features_df, name="select_train")
        missing = [
            c for c in (descriptor.entity_col, descriptor.target_col) if c not in joined.columns
        ]
        if missing:
            typer.echo(
                f"Prepared splits/features do not match dataset '{descriptor.name}' "
                f"(missing column(s): {', '.join(missing)}); ignoring them."
            )
            return None, None
        feature_cols = [c for c in features_df.columns if c != "original_row_id"]
        joined[feature_cols] = joined[feature_cols].fillna(0)
        return joined, feature_cols
    except (OSError, ValueError, KeyError):
        return None, None
