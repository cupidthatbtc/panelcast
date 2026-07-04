"""The `panelcast select` command (#103, A7+A8).

Runs the portable model-selection protocol: prints a pre-run plan with predicted
cost for informed consent, then (unless --dry-run) drives the staged sweep,
scores every arm, applies the pre-registered rules, and writes one ranked report
that is the domain's `.audit/` entry. `select` RECOMMENDS; a default flip stays
a manual PR.
"""

from __future__ import annotations

from pathlib import Path

import typer

from panelcast.cli import app


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
    sweep_id: str = typer.Option(
        "sweep", "--sweep-id", help="Sweep directory name under outputs/select/ (enables --resume)."
    ),
    config_path: str = typer.Option(
        "configs/select.yaml", "--config", help="YAML with the rules and effort tiers."
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
    from panelcast.select.orchestrate import build_plan, render_plan, run_select
    from panelcast.select.rules import DecisionRules
    from panelcast.select.tiers import resolve_tier, tier_to_sweep_config

    cfg_path = Path(config_path)
    try:
        tier = resolve_tier(effort, cfg_path)
    except ValueError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    rules = DecisionRules.load(cfg_path)
    descriptor = load_descriptor(dataset)
    label = dataset or descriptor.name

    cfg = tier_to_sweep_config(
        tier,
        sweep_id=sweep_id,
        dataset=dataset,
        max_fits=max_fits,
        budget_hours=budget_hours,
        promote_z=rules.promote_z,
    )

    from panelcast.select.orchestrate import _resolve_dims

    plan = build_plan(
        descriptor,
        tier,
        cfg,
        dataset_label=label,
        n_confirmation_seeds=len(rules.confirmation_seeds),
        dims=_resolve_dims(_prepared_paths(descriptor)),
    )
    typer.echo(render_plan(plan))

    if dry_run:
        raise typer.Exit(code=0)

    train_df, feature_cols = _load_prepared_frame()
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
    )
    typer.echo(f"\nSweep complete. Report: {result['report_dir']}/report.md")
    if result["winner_arm"]:
        typer.echo(
            f"Recommended (pre-registered rules): arm {result['winner_arm']}. "
            "A default flip is a manual PR with the report as evidence."
        )
    else:
        typer.echo("No candidate cleared the pre-registered bar; defaults hold.")


def _prepared_paths(descriptor) -> dict | None:
    """Feature path plus an entity-count hint (from the split, via the descriptor)."""
    features = Path("data/features/train_features.parquet")
    if not features.exists():
        return None
    hint: dict = {"features": features}
    splits = Path("data/splits/within_entity_temporal/train.parquet")
    if splits.exists():
        try:
            import pandas as pd

            split_df = pd.read_parquet(splits, columns=[descriptor.entity_col])
            hint["n_artists"] = int(split_df[descriptor.entity_col].nunique())
        except (OSError, ValueError, KeyError):
            pass
    return hint


def _load_prepared_frame():
    """(joined train frame, feature_cols) from prepared artifacts, or (None, None)."""
    splits = Path("data/splits/within_entity_temporal/train.parquet")
    features = Path("data/features/train_features.parquet")
    if not (splits.exists() and features.exists()):
        return None, None
    try:
        import pandas as pd

        from panelcast.data.alignment import join_splits_with_features

        split_df = pd.read_parquet(splits)
        features_df = pd.read_parquet(features)
        joined = join_splits_with_features(split_df, features_df, name="select_train")
        feature_cols = [c for c in features_df.columns if c != "original_row_id"]
        joined[feature_cols] = joined[feature_cols].fillna(0)
        return joined, feature_cols
    except (OSError, ValueError, KeyError):
        return None, None
