"""Rich Console output rendering for preflight results.

Provides TTY-aware colored output for preflight check results,
with verbose mode for detailed memory breakdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from panelcast.data.ingest import DataDimensions
    from panelcast.preflight import (
        ExtrapolationResult,
        FullPreflightResult,
        PreflightResult,
        PreflightStatus,
    )


def render_preflight_result(
    result: PreflightResult,
    *,
    verbose: bool = False,
    dimensions: DataDimensions | None = None,
) -> None:
    """Render preflight result with TTY-aware colored output.

    Uses Rich Console which automatically handles TTY detection:
    - TTY: Displays colored markup
    - Non-TTY (pipe, file): Strips markup for plain text

    Args:
        result: PreflightResult from run_preflight_check().
        verbose: If True, show detailed memory breakdown.

    Example:
        >>> from panelcast.preflight import run_preflight_check
        >>> result = run_preflight_check(1000, 20, 50, 10, 4, 1000, 1000)
        >>> render_preflight_result(result, verbose=True)
    """
    console = Console()

    # Status line with color
    status_line = _format_status_line(result.status, result.message)
    console.print(status_line)

    # Verbose: memory breakdown
    if verbose and result.estimate is not None:
        console.print()
        console.print("[bold]Memory breakdown:[/bold]")
        console.print(f"  Base model:    {result.estimate.base_model_gb:.2f} GB")
        console.print(
            f"  Chains ({result.estimate.num_chains}):   {result.estimate.chain_memory_gb:.2f} GB"
        )
        console.print(f"  JIT buffer:    {result.estimate.jit_buffer_gb:.2f} GB")
        console.print("  " + "\u2500" * 25)
        console.print(f"  [bold]Total:         {result.estimate.total_gb:.2f} GB[/bold]")

    # GPU info (if available)
    if result.device_name is not None:
        console.print()
        console.print(f"[bold]GPU:[/bold] {result.device_name}")
        console.print(
            f"  Available: {result.available_gb:.1f} GB / {result.total_gpu_gb:.1f} GB total"
        )

    # Suggestions
    if result.suggestions:
        console.print()
        console.print("[bold]Suggestions:[/bold]")
        for suggestion in result.suggestions:
            console.print(f"  \u2022 {suggestion}")

    # Note about data source
    console.print()
    if dimensions is not None:
        console.print(
            f"[dim]Data: {dimensions.n_observations:,} obs, "
            f"{dimensions.n_artists:,} artists ({dimensions.source})[/dim]"
        )
    else:
        console.print(
            "[dim]Note: Estimates based on fixed defaults "
            "(1000 obs, 20 features, 100 artists).[/dim]"
        )
    console.print("[dim]Use --preflight-full for accurate data-specific checking.[/dim]")


def _format_status_line(status: PreflightStatus, message: str) -> str:
    """Format status with appropriate color markup."""
    from panelcast.preflight import PreflightStatus

    match status:
        case PreflightStatus.PASS:
            return f"[green bold]PASS[/green bold] {message}"
        case PreflightStatus.WARNING:
            return f"[yellow bold]WARNING[/yellow bold] {message}"
        case PreflightStatus.FAIL:
            return f"[red bold]FAIL[/red bold] {message}"
        case PreflightStatus.CANNOT_CHECK:
            return f"[yellow bold]CANNOT CHECK[/yellow bold] {message}"
        case _:
            return f"[dim]UNKNOWN[/dim] {message}"


def render_full_preflight_result(result: FullPreflightResult, *, verbose: bool = False) -> None:
    """Render full preflight result with TTY-aware colored output.

    Uses Rich Console which automatically handles TTY detection:
    - TTY: Displays colored markup
    - Non-TTY (pipe, file): Strips markup for plain text

    This renderer displays MEASURED peak memory from a mini-MCMC run,
    as opposed to render_preflight_result which shows ESTIMATED memory.

    Args:
        result: FullPreflightResult from run_full_preflight_check().
        verbose: If True, show additional mini-run details.

    Example:
        >>> from panelcast.preflight import run_full_preflight_check
        >>> result = run_full_preflight_check(model_args)
        >>> render_full_preflight_result(result, verbose=True)
    """
    console = Console()

    # Status line with color
    status_line = _format_status_line(result.status, result.message)
    console.print(status_line)

    # Always show measured peak and mini-run time
    console.print()
    console.print("[bold]Measured Peak:[/bold]")
    console.print(f"  {result.measured_peak_gb:.2f} GB [dim](actual measurement)[/dim]")
    console.print(f"  Mini-run time: {result.mini_run_seconds:.1f} seconds")

    # Verbose: show additional measurement details
    if verbose:
        console.print()
        console.print(
            "[dim]This is an actual measurement from a 1-chain, 10-warmup, 1-sample mini-run.[/dim]"
        )

    # GPU info (if available)
    if result.device_name is not None:
        console.print()
        console.print(f"[bold]GPU:[/bold] {result.device_name}")
        console.print(
            f"  Available: {result.available_gb:.1f} GB / {result.total_gpu_gb:.1f} GB total"
        )

    # Suggestions
    if result.suggestions:
        console.print()
        console.print("[bold]Suggestions:[/bold]")
        for suggestion in result.suggestions:
            console.print(f"  \u2022 {suggestion}")


def format_uncertainty(projected_gb: float, uncertainty_percent: float) -> str:
    """Format projected memory with uncertainty range.

    Args:
        projected_gb: Projected memory in GB.
        uncertainty_percent: Uncertainty as percentage (e.g., 10.0 for 10%).

    Returns:
        Formatted string like "15.8 GB +/-10%"
    """
    return f"{projected_gb:.1f} GB +/-{uncertainty_percent:.0f}%"


def render_extrapolation_result(result: ExtrapolationResult, *, verbose: bool = False) -> None:
    """Render extrapolation preflight result with TTY-aware colored output.

    Uses Rich Console which automatically handles TTY detection:
    - TTY: Displays colored markup
    - Non-TTY (pipe, file): Strips markup for plain text

    This renderer displays BOTH measured peak memory from calibration runs
    AND projected estimate extrapolated to target samples.

    Args:
        result: ExtrapolationResult from run_extrapolated_preflight_check().
        verbose: If True, show calibration coefficients.

    Example:
        >>> from panelcast.preflight import run_extrapolated_preflight_check
        >>> result = run_extrapolated_preflight_check(model_args, target_samples=2000, ...)
        >>> render_extrapolation_result(result, verbose=True)
    """
    console = Console()

    # Status line with color
    status_line = _format_status_line(result.status, result.message)
    console.print(status_line)

    # Memory Projection section - CRITICAL: show BOTH measured and projected
    console.print()
    console.print("[bold]Memory Projection:[/bold]")
    console.print(
        f"  Measured peak (calibration at 10+50 samples): {result.measured_peak_gb:.2f} GB"
    )
    console.print(
        f"  Projected ({result.target_samples:,} samples): "
        f"{format_uncertainty(result.projected_gb, result.uncertainty_percent)}"
    )

    # Verbose: show calibration coefficients
    if verbose:
        console.print()
        console.print(
            f"[dim]Calibration: fixed={result.calibration.fixed_overhead_gb:.2f} GB "
            f"+ {result.calibration.per_sample_gb:.4f} GB/sample[/dim]"
        )

    # Cache status
    console.print()
    if result.from_cache:
        console.print("[dim]Calibration: from cache[/dim]")
    else:
        console.print("[dim]Calibration: fresh (cached for future)[/dim]")

    # GPU info (if available)
    if result.device_name is not None:
        console.print()
        console.print(f"[bold]GPU:[/bold] {result.device_name}")
        console.print(
            f"  Available: {result.available_gb:.1f} GB / {result.total_gpu_gb:.1f} GB total"
        )

    # Suggestions
    if result.suggestions:
        console.print()
        console.print("[bold]Suggestions:[/bold]")
        for suggestion in result.suggestions:
            console.print(f"  \u2022 {suggestion}")
