"""What a failed run leaves behind: failure.json, the console epilogue, and
releasing the log-file handles that would otherwise block quarantine on Windows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import structlog

from panelcast.pipelines.manifest import RunManifest

log = structlog.get_logger()


def write_failure_payload(
    run_dir: Path | None,
    manifest: RunManifest | None,
    error: Exception,
    stage: str,
) -> None:
    """Structured forensics for `runs why`; must never raise."""
    if run_dir is None or not run_dir.exists():
        return
    import traceback

    from panelcast.pipelines.errors import failure_hint
    from panelcast.utils.logging import recent_events

    try:
        payload = {
            "run_id": manifest.run_id if manifest else None,
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback_tail": traceback.format_exception(error)[-8:],
            "stages_completed": (list(manifest.stages_completed) if manifest else []),
            "hint": failure_hint(error),
            "resume_command": f"panelcast run --resume {run_dir.name}",
            "recent_events": recent_events(),
        }
        (run_dir / "failure.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    except Exception as e:  # forensics must never mask the real failure
        log.debug("failure_payload_write_failed", error=str(e))


def print_failure_epilogue(error: Exception, stage: str, final_path) -> None:
    """The 10-second answer to 'what happened and what do I type next'."""
    from rich.console import Console

    from panelcast.pipelines.errors import failure_hint

    console = Console(stderr=True)
    run_name = final_path.name if final_path is not None else "unknown"
    console.print(f"\n[red bold]{stage} failed:[/] {type(error).__name__}: {error}")
    if final_path is not None:
        console.print(f"run moved to: {final_path}")
    console.print(f"resume with:  panelcast run --resume {run_name}")
    hint = failure_hint(error)
    if hint:
        console.print(f"hint:         {hint}")
    console.print(f"details:      panelcast runs why {run_name}")


def close_log_handlers() -> None:
    """Close file handlers to release locks (needed for Windows).

    On Windows, file handlers keep files locked which prevents moving
    directories containing log files. This closes all handlers on the
    root logger to release those locks.
    """
    root_logger = logging.getLogger()
    handlers_to_remove = []

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)
