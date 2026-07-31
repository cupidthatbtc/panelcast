"""Pipeline orchestrator for end-to-end execution with progress tracking.

This module provides the PipelineOrchestrator class that executes pipeline
stages in dependency order, with features for:
- Progress display using Rich
- Hash-based skip logic for incremental runs
- Environment verification via pixi.lock
- Error handling with fail-fast semantics
- Manifest tracking for reproducibility

The pieces it drives live next door: ``pipeline_config`` (the resolved
experiment), ``run_command`` (manifest command provenance), ``stage_context``
(config to StageContext), ``artifact_routing`` (run-scoped product roots), and
``failure_report`` (what a failed run leaves behind).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import structlog
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from panelcast import __version__ as panelcast_version
from panelcast.config.descriptor import load_descriptor, resolve_descriptor_path
from panelcast.model_preflight import beta_binomial_trial_scale
from panelcast.paths import (
    ArtifactPaths,
    RunPathError,
    resolve_latest,
    safe_run_dir,
)
from panelcast.pipelines.artifact_routing import find_run_with_product, resolve_artifact_paths
from panelcast.pipelines.errors import (
    ConvergenceError,
    EnvironmentError,
    PipelineError,
    StageSkipped,
)
from panelcast.pipelines.failure_report import (
    close_log_handlers,
    print_failure_epilogue,
    write_failure_payload,
)
from panelcast.pipelines.manifest import (
    GitStateModel,
    RunManifest,
    capture_environment,
    flag_differences,
    generate_run_id,
    load_run_manifest,
    save_run_manifest,
)

# Re-exported: these have always been importable (and monkeypatchable) from
# this module, and the CLI, select runner and tests still reach for them here.
from panelcast.pipelines.pipeline_config import (  # noqa: F401
    _RESUME_CONFIG_KEYS,
    _RESUME_EXCLUDED_KEYS,
    PipelineConfig,
    _field_defaults,
    _get_default_config,
    _reset_default_config,
    resolve_model_facts,
)
from panelcast.pipelines.run_command import build_command_string
from panelcast.pipelines.stage_context import build_stage_context
from panelcast.pipelines.stages import PipelineStage, StageContext, get_execution_order
from panelcast.pipelines.stamps import (
    CONSUMER_STAGES,
    DATA_STAGE_ROOTS,
    read_stamp,
    verify_stamps,
    write_stamp,
)
from panelcast.utils.environment import ensure_environment_locked, verify_environment
from panelcast.utils.git_state import capture_git_state
from panelcast.utils.hashing import sha256_path
from panelcast.utils.logging import is_interactive, setup_pipeline_logging
from panelcast.utils.random import set_seeds

log = structlog.get_logger()


def _installed_plugins() -> dict[str, dict[str, str]]:
    """Entry-point plugin provenance for the run manifest (#172)."""
    from panelcast.features.registry import discovered_plugins

    return discovered_plugins()


class PipelineOrchestrator:
    """Orchestrates pipeline execution with progress tracking and error handling.

    The orchestrator manages the full pipeline lifecycle:
    1. Verify environment (pixi.lock check)
    2. Create run directory and manifest
    3. Execute stages in dependency order
    4. Track progress with Rich display
    5. Handle errors with fail-fast semantics
    6. Create outputs/latest symlink on success

    Attributes:
        config: Pipeline configuration options.
        output_base: Base directory for output runs (default "outputs").
        run_dir: Path to current run directory (set during run).
        manifest: Current run manifest (set during run).

    Example:
        >>> config = PipelineConfig(seed=42, dry_run=True)
        >>> orchestrator = PipelineOrchestrator(config)
        >>> exit_code = orchestrator.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_base: Path | str = Path("outputs"),
    ) -> None:
        """Initialize orchestrator with configuration.

        Args:
            config: Pipeline configuration.
            output_base: Base directory for outputs (default "outputs").
        """
        self.config = config
        self.output_base = Path(output_base)
        self.run_dir: Path | None = None
        self.manifest: RunManifest | None = None
        self._start_time: float = 0.0
        self._resolved_paths: ArtifactPaths | None = None
        # Resolve the dataset descriptor once; every stage reads it from the
        # StageContext rather than re-deriving domain names from literals.
        self.descriptor = load_descriptor(config.dataset)
        self.descriptor_path = resolve_descriptor_path(config.dataset)
        resolve_model_facts(config, self.descriptor)
        if config.likelihood_family == "beta_binomial":
            span, is_unit = beta_binomial_trial_scale(self.descriptor.target_bounds)
            if not is_unit:
                log.warning(
                    "beta_binomial_trial_count_scaled",
                    target_span=span,
                    count_multiplier=span,
                    message=(
                        "The Beta-Binomial expands each aggregation count by the target span. "
                        "For a genuine proportion, rescale the target and target_bounds to [0, 1]."
                    ),
                )
        # Resolve the observation threshold: an explicit CLI/YAML value wins;
        # otherwise fall back to the descriptor's primary_min_obs so retargeted
        # domains don't need --min-ratings on the command line. The data stage
        # only materializes parquets at the descriptor's thresholds, so any
        # other value would die hours later at the splits stage.
        if config.min_ratings is None:
            config.min_ratings = self.descriptor.primary_min_obs
        elif config.min_ratings not in self.descriptor.min_obs_thresholds:
            raise ValueError(
                f"min_ratings={config.min_ratings} has no materialized dataset for "
                f"descriptor '{self.descriptor.name}': the data stage only writes "
                f"thresholds {sorted(self.descriptor.min_obs_thresholds)}. Pick one "
                "of those, or add the value to the descriptor's min_obs_thresholds."
            )

    def _require_run_dir(self) -> Path:
        """The run directory, or a hard error if a stage runs before setup."""
        if self.run_dir is None:
            raise PipelineError("run directory not initialized before use", stage="setup")
        return self.run_dir

    def _require_manifest(self) -> RunManifest:
        """The run manifest, or a hard error if a stage runs before setup."""
        if self.manifest is None:
            raise PipelineError("manifest not initialized before use", stage="setup")
        return self.manifest

    def _resolved_min_ratings(self) -> int:
        """min_ratings after __init__/resume resolution (None here is a bug)."""
        if self.config.min_ratings is None:
            raise PipelineError("min_ratings unresolved before use", stage="setup")
        return self.config.min_ratings

    def _resolved_event_cap(self) -> int:
        """The per-entity event cap after descriptor resolution (#268)."""
        if self.config.max_events is None:
            raise PipelineError("event cap unresolved before use", stage="setup")
        return self.config.max_events

    def _resolved_likelihood_family(self) -> str:
        """likelihood_family after descriptor resolution (#268)."""
        if self.config.likelihood_family is None:
            raise PipelineError("likelihood_family unresolved before use", stage="setup")
        return self.config.likelihood_family

    def _resolved_target_transform(self) -> str:
        """target_transform after descriptor resolution (#268)."""
        if self.config.target_transform is None:
            raise PipelineError("target_transform unresolved before use", stage="setup")
        return self.config.target_transform

    def run(self) -> int:
        """Execute the pipeline and return exit code.

        Runs all configured stages in dependency order with progress tracking.
        Creates run manifest, handles errors, and maintains output structure.

        Returns:
            Exit code: 0 on success, error's exit_code on failure.

        Raises:
            EnvironmentError: If strict=True and pixi.lock missing.
        """
        self._start_time = time()

        # 1. Verify environment
        try:
            self._verify_environment()
        except EnvironmentError as e:
            log.error("environment_verification_failed", error=str(e))
            return e.exit_code

        # Resolve the learn_n_exponent/n_exponent conflict before _setup_run
        # persists the manifest, so manifest.json and resolved_config.yaml record
        # the value the run actually uses rather than the stale fixed exponent.
        if self.config.learn_n_exponent and self.config.n_exponent != 0.0:
            log.warning(
                "config_conflict",
                message="Both --n-exponent and --learn-n-exponent set; using learned mode",
            )
            self.config.n_exponent = 0.0

        # 2. Set up run directory and manifest
        self._setup_run()

        # 3. Set up logging
        log_file = self.run_dir / "pipeline.log.json" if self.run_dir else None
        setup_pipeline_logging(verbose=self.config.verbose, log_file=log_file)

        # 4. Set random seeds
        set_seeds(self.config.seed)

        log.info(
            "pipeline_started",
            run_id=self.manifest.run_id if self.manifest else "unknown",
            seed=self.config.seed,
            dry_run=self.config.dry_run,
            stages=self.config.stages,
            n_exponent=self.config.n_exponent,
            learn_n_exponent=self.config.learn_n_exponent,
        )

        # 5. Get execution order (pass min_ratings for correct input_paths)
        try:
            stages = get_execution_order(
                self.config.stages,
                min_ratings=self._resolved_min_ratings(),
                descriptor=self.descriptor,
                descriptor_path=self.descriptor_path,
                paths=self._artifact_paths(),
            )
        except KeyError as e:
            log.error("invalid_stage", error=str(e))
            return 1
        except PipelineError as e:
            # Consumer-only invocations fail here when no prior run supplies
            # their inputs; route through the normal failure path.
            self._handle_failure(e, e.stage)
            return e.exit_code

        if not stages:
            log.warning("no_stages_to_execute")
            self._finalize_success()
            return 0

        # 6. Execute stages
        try:
            self._execute_stages(stages)
            self._finalize_success()
            return 0
        except PipelineError as e:
            self._handle_failure(e, e.stage)
            return e.exit_code
        except Exception as e:
            self._handle_failure(e, "unknown")
            return 1

    def _verify_environment(self) -> None:
        """Verify environment is locked for reproducibility.

        Raises EnvironmentError if pixi.lock is not found when
        config.enforce_lockfile=True.
        """
        log.debug("verifying_environment", enforce_lockfile=self.config.enforce_lockfile)

        try:
            ensure_environment_locked(strict=self.config.enforce_lockfile)
        except Exception as e:
            # Re-raise as our EnvironmentError for consistent exit code
            raise EnvironmentError(str(e)) from e

        # Log environment status
        status = verify_environment()
        if status.is_reproducible:
            log.info(
                "environment_verified",
                pixi_lock_hash=status.pixi_lock_hash[:12] if status.pixi_lock_hash else None,
            )
        else:
            log.warning("environment_not_locked", warnings=status.warnings)

    def _setup_run(self) -> None:
        """Create run directory and initialize manifest."""
        # Handle resume vs fresh run
        if self.config.resume:
            self._setup_resume()
            return

        # Generate new run ID and create its directory EXCLUSIVELY: a second
        # run minting the same id must retry with a fresh one instead of
        # silently sharing (and, on failure, rmtree'ing) this run's dir.
        if self.config.run_id is not None:
            # Caller-supplied id (#167): the select runner names each arm's run
            # up front so it never has to race the mutable `latest` pointer.
            # A collision is a hard error — the caller promised uniqueness.
            run_id = self.config.run_id
            try:
                self.run_dir = safe_run_dir(self.output_base, run_id)
            except RunPathError as e:
                raise PipelineError(str(e), stage="setup") from e
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                raise PipelineError(
                    f"run_id '{run_id}' already exists under {self.output_base}; "
                    "caller-supplied run ids must be unique.",
                    stage="setup",
                ) from None
        else:
            run_id = ""
            for _ in range(10):
                run_id = generate_run_id()
                self.run_dir = safe_run_dir(self.output_base, run_id)
                try:
                    self.run_dir.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    continue
            else:
                raise PipelineError(
                    f"Could not create a unique run directory under {self.output_base} "
                    f"after 10 attempts (last id: {run_id}).",
                    stage="setup",
                )

        # Capture git state and environment
        git_state = capture_git_state()
        environment = capture_environment()

        # Build command string for manifest
        command = self._build_command_string()

        # Create manifest. Flags are derived from the dataclass fields so a new
        # config knob cannot be forgotten (#296); the descriptor hash rides
        # along as provenance.
        flags = {}
        for f in dataclass_fields(PipelineConfig):
            value = getattr(self.config, f.name)
            flags[f.name] = list(value) if isinstance(value, tuple) else value
        flags["dataset_descriptor_hash"] = self.descriptor.descriptor_hash()

        from panelcast.config.pipeline_yaml import (
            experiment_config_hash,
            experiment_config_payload,
        )

        self.manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            version=panelcast_version,
            tag=self.config.tag,
            command=command,
            flags=flags,
            experiment_identity={
                "config_hash": experiment_config_hash(self.config),
                # The hashed payload itself, so an identity mismatch on resume
                # can name the differing keys instead of two opaque hashes.
                "config_payload": experiment_config_payload(self.config),
                "descriptor_hash": self.descriptor.descriptor_hash(),
                "source": {"commit": git_state.commit, "dirty": git_state.dirty},
                "environment_fingerprint": environment.fingerprint,
                "pixi_lock_hash": environment.pixi_lock_hash,
                "package_version": panelcast_version,
            },
            plugins=_installed_plugins(),
            seed=self.config.seed,
            git=GitStateModel.from_git_state(git_state),
            environment=environment,
            input_hashes={},
            stage_hashes={},
            stages_completed=[],
            stages_skipped=[],
            outputs={},
            success=False,
            error=None,
            duration_seconds=0.0,
        )

        # Save initial manifest
        save_run_manifest(self.manifest, self.run_dir)

        # The post-layering truth (preset + YAML overlays + CLI wins +
        # descriptor-resolved values) — the manifest command string cannot
        # express YAML-only gates, so this is what `runs reproduce` re-executes.
        from panelcast.config.pipeline_yaml import dump_resolved_config

        (self.run_dir / "resolved_config.yaml").write_text(
            dump_resolved_config(self.config), encoding="utf-8"
        )

    # Config fields NOT restored from the manifest on resume: execution
    # mechanics and per-invocation provenance that never affect outputs. Every
    # other field — the complete resolved experiment — is restored (#296), so a
    # hand-maintained key list can no longer silently omit a control.
    RESUME_EXCLUDED_KEYS = _RESUME_EXCLUDED_KEYS
    RESUME_CONFIG_KEYS = _RESUME_CONFIG_KEYS
    # Flags that should not invalidate input-hash skip detection.
    # These only affect execution mechanics, not stage outputs.
    SKIP_FLAG_IGNORE = frozenset(
        {"skip_existing", "dry_run", "verbose", "resume", "progress_bar",
         # Provenance-only flags — never applied to outputs, must not force reruns.
         "unknown_config_keys", "tag", "run_id"}
    )

    def _skip_flag_differences(self, previous_manifest: RunManifest) -> list[str]:
        """Return output-affecting flag keys that changed since previous run."""
        if self.manifest is None:
            return []
        return [
            key
            for key, _, _ in flag_differences(
                self.manifest.flags,
                previous_manifest.flags,
                _get_default_config(),
                ignore=self.SKIP_FLAG_IGNORE,
            )
        ]

    def _setup_resume(self) -> None:
        """Set up for resuming a previous run."""
        resume_id = self.config.resume
        if resume_id is None:
            raise PipelineError("resume requested without a run id", stage="setup")

        try:
            run_dir = safe_run_dir(self.output_base, resume_id, field="resume")
        except RunPathError as e:
            raise PipelineError(str(e), stage="setup") from e

        if run_dir.exists():
            self.run_dir = run_dir
        else:
            try:
                failed_dir = safe_run_dir(
                    self.output_base, resume_id, subdir="failed", field="resume"
                )
            except RunPathError as e:
                raise PipelineError(str(e), stage="setup") from e
            if not failed_dir.exists():
                raise PipelineError(
                    f"Cannot find run to resume: {resume_id}",
                    stage="setup",
                )
            # Move back from failed for retry
            self.run_dir = run_dir
            shutil.move(str(failed_dir), str(run_dir))

        # Load existing manifest
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            raise PipelineError(
                f"No manifest.json in run directory: {resume_id}",
                stage="setup",
            )

        self.manifest = load_run_manifest(manifest_path)

        # Restore MCMC config from manifest to prevent config drift
        self._restore_config_from_manifest()

        log.info(
            "resuming_run",
            run_id=resume_id,
            completed_stages=self.manifest.stages_completed,
        )

    def _restore_config_from_manifest(self) -> None:
        """Restore the complete resolved experiment on resume (#296).

        ``resolved_config.yaml`` is the post-layering truth and covers every
        output-affecting knob, so it is preferred; manifest flags are the
        fallback for runs that predate it. Execution mechanics
        (RESUME_EXCLUDED_KEYS) keep their current CLI values.
        """
        if self.manifest is None:
            return

        resolved_path = self.run_dir / "resolved_config.yaml" if self.run_dir else None
        restored_from_resolved: set[str] = set()
        if resolved_path is not None and resolved_path.exists():
            from panelcast.config.pipeline_yaml import load_resolved_config

            for field_name, value in load_resolved_config(resolved_path).items():
                if field_name in self.RESUME_EXCLUDED_KEYS:
                    continue
                setattr(self.config, field_name, value)
                restored_from_resolved.add(field_name)
            log.debug(
                "resume_config_restored_from_resolved",
                keys=sorted(restored_from_resolved),
            )

        for key in self.RESUME_CONFIG_KEYS:
            if key in restored_from_resolved:
                continue
            if key in self.manifest.flags:
                manifest_value = self.manifest.flags[key]
                if isinstance(getattr(self.config, key), tuple) and isinstance(
                    manifest_value, list
                ):
                    manifest_value = tuple(manifest_value)
                setattr(self.config, key, manifest_value)
                log.debug("resume_config_restored", key=key, value=manifest_value)
            elif resolved_path is None or not resolved_path.exists():
                # With a resolved config on disk, absence just means "unset"
                # (the dump omits None-valued knobs); without one, an absent
                # flag is genuine provenance loss worth flagging.
                current_default = getattr(self.config, key)
                log.warning(
                    "resume_config_missing",
                    key=key,
                    current_default=current_default,
                    message=(
                        f"manifest missing '{key}', using current default {current_default} "
                        "- verify this matches original run"
                    ),
                )

        # Re-resolve the descriptor for the restored dataset reference and
        # guard against descriptor drift: resuming a run whose descriptor
        # YAML has changed since the original run would silently mix domains.
        self.descriptor = load_descriptor(self.config.dataset)
        self.descriptor_path = resolve_descriptor_path(self.config.dataset)
        # __init__ resolved min_ratings against the pre-resume (CLI/default)
        # descriptor; the manifest restore above re-pointed the dataset, so
        # re-derive the threshold from the restored descriptor when it wasn't
        # pinned in the manifest. Without this a resumed cross-domain run keeps
        # the wrong threshold and reads the wrong processed parquet.
        if self.config.min_ratings is None:
            self.config.min_ratings = self.descriptor.primary_min_obs
        recorded_hash = self.manifest.flags.get("dataset_descriptor_hash")
        if recorded_hash is None:
            log.warning(
                "resume_descriptor_hash_missing",
                message=(
                    "manifest predates descriptor tracking; assuming AOTY "
                    "defaults match the original run"
                ),
            )
        elif recorded_hash != self.descriptor.descriptor_hash():
            raise PipelineError(
                "Dataset descriptor changed since the original run "
                f"(recorded hash {recorded_hash[:12]}…, current "
                f"{self.descriptor.descriptor_hash()[:12]}…). Resuming would mix "
                "artifacts from different dataset definitions. Start a fresh "
                "run instead.",
                stage="setup",
            )

        # Re-validate after restoration (catches corrupted/invalid manifest values)
        self.config._validate()

        self._verify_experiment_identity()

    def _verify_experiment_identity(self) -> None:
        """Prove the resumed config is the recorded experiment, or refuse (#296)."""
        from panelcast.config.pipeline_yaml import (
            experiment_config_hash,
            experiment_config_payload,
            experiment_payload_hash,
            normalize_experiment_payload,
        )

        recorded = getattr(self.manifest, "experiment_identity", None)
        if not isinstance(recorded, dict) or not recorded:
            log.warning(
                "resume_experiment_identity_missing",
                message=(
                    "manifest predates experiment-identity tracking; resuming "
                    "on the restored flags without a config-hash proof"
                ),
            )
            return

        current_hash = experiment_config_hash(self.config)
        recorded_hash = recorded.get("config_hash")
        if recorded_hash and current_hash != recorded_hash:
            current_payload = experiment_config_payload(self.config)
            recorded_payload: dict[str, Any] = recorded.get("config_payload") or {}
            # A pre-#303 run recorded its identity under the deprecated key
            # spellings; the same experiment hashes differently only because
            # of the rename. Translate and re-compare before refusing.
            normalized_recorded = normalize_experiment_payload(recorded_payload)
            if (
                normalized_recorded != recorded_payload
                and experiment_payload_hash(normalized_recorded) == current_hash
            ):
                log.info(
                    "resume_experiment_identity_translated",
                    message=(
                        "recorded identity used pre-rename key spellings; "
                        "hashes match after #303 alias translation"
                    ),
                )
                return
            recorded_payload = normalized_recorded
            diff = [
                f"  {key}: recorded {recorded_payload.get(key)!r} != "
                f"requested {current_payload.get(key)!r}"
                for key in sorted(set(current_payload) | set(recorded_payload))
                if recorded_payload.get(key) != current_payload.get(key)
            ]
            raise PipelineError(
                "Resume identity mismatch: the resumed configuration does not "
                f"hash to the recorded experiment ({recorded_hash[:12]}… vs "
                f"{current_hash[:12]}…). Differing keys:\n"
                + ("\n".join(diff) or "  (difference not attributable to a resolved-config key)")
                + "\nStart a fresh run, or restore the original configuration.",
                stage="setup",
            )

        # Source/environment drift is legitimate (bugfix commit, new machine)
        # but breaks bit-identity — surface it loudly instead of refusing.
        git_state = capture_git_state()
        environment = capture_environment()
        drift: dict[str, Any] = {}
        recorded_source = recorded.get("source") or {}
        if recorded_source.get("commit") and recorded_source["commit"] != git_state.commit:
            drift["source_commit"] = {
                "recorded": recorded_source["commit"],
                "current": git_state.commit,
            }
        if (
            recorded.get("environment_fingerprint")
            and recorded["environment_fingerprint"] != environment.fingerprint
        ):
            drift["environment_fingerprint"] = {
                "recorded": recorded["environment_fingerprint"],
                "current": environment.fingerprint,
            }
        if recorded.get("package_version") and recorded["package_version"] != panelcast_version:
            drift["package_version"] = {
                "recorded": recorded["package_version"],
                "current": panelcast_version,
            }
        if drift:
            log.warning(
                "resume_experiment_environment_drift",
                message=(
                    "resuming under a different source/environment than the "
                    "recorded run; results may not be bit-identical"
                ),
                **drift,
            )

    def _build_command_string(self) -> str:
        """Build command string representation for manifest."""
        return build_command_string(self.config, self.descriptor)

    def _output_verification_roots(self, previous_run: Path) -> tuple[Path, ...]:
        """Roots an output recorded by the run at ``previous_run`` may live under (#367).

        The cross-run artifact roots, plus that one run's own directory. The
        whole working tree is not a root, so a rewritten manifest cannot
        substitute an unrelated file.

        Both halves are named by what they are, not by what contains them
        (#385). The output base would admit every sibling run, and *this* run's
        run-scoped products would admit the directories the current run is
        writing into as it goes — either lets a rewritten previous manifest
        point a key with no declared binding at something this pipeline
        produced and have it believed.

        The run root is the directory the manifest was *read from*, never a
        field of the manifest itself: deriving it from ``run_id`` would let the
        document under verification choose which run it is checked against.
        Required rather than defaulted, since every default here is the
        permissive one.

        The shared roots come from the flat layout because that is where
        cross-run artifacts are recorded, and the layout of the run being
        verified is not something its manifest states. A run-scoped product of
        that run is covered by its run directory instead, so the residue is
        only that a run-scoped run also admits the workspace's own flat
        artifact roots — directories it does not write to, not arbitrary paths.
        """
        return (*ArtifactPaths.flat().roots(), previous_run)

    def _carry_skipped_stage_provenance(
        self,
        stage: PipelineStage,
        previous_manifest: RunManifest,
        previous_run: Path | None,
    ) -> None:
        """Carry reusable evidence, never paths owned by the previous run."""
        manifest = self._require_manifest()
        previous_hash = previous_manifest.stage_hashes.get(stage.name)
        if previous_hash is not None:
            manifest.stage_hashes[stage.name] = previous_hash
        prefix = f"{stage.name}:"
        carried: set[str] = set()
        previous_root = previous_run.resolve() if previous_run else None
        for key, value in (previous_manifest.outputs or {}).items():
            if not key.startswith(prefix):
                continue
            try:
                resolved = Path(value).resolve()
            except (OSError, ValueError):
                continue
            if previous_root is not None and (
                resolved == previous_root or previous_root in resolved.parents
            ):
                continue
            manifest.outputs[key] = value
            carried.add(key)
        for key, value in (previous_manifest.output_hashes or {}).items():
            if key in carried:
                manifest.output_hashes[key] = value

    def _execute_stages(self, stages: list[PipelineStage]) -> None:
        """Execute stages with progress display.

        Args:
            stages: List of stages in execution order.
        """
        # Load previous manifest for skip detection
        previous_manifest: RunManifest | None = None
        previous_run: Path | None = None
        if self.config.skip_existing and self.run_dir:
            previous_run = resolve_latest(self.output_base)
            if previous_run is not None:
                try:
                    prev_manifest_path = previous_run / "manifest.json"
                    if prev_manifest_path.exists():
                        previous_manifest = load_run_manifest(prev_manifest_path)
                except OSError as e:
                    log.debug("could_not_load_previous_manifest", error=str(e), exc_info=True)
                except Exception as e:
                    log.debug("could_not_load_previous_manifest", error=str(e), exc_info=True)

        # Defensive: a latest pointer written by an older checkout may still
        # target a dry run, whose recorded hashes cover nothing on disk.
        if previous_manifest is not None and previous_manifest.flags.get("dry_run"):
            log.debug("skip_existing_ignores_dry_run_manifest", run_id=previous_manifest.run_id)
            previous_manifest = None

        # Progress weighting reads durations before the flag-change reset below:
        # a config change invalidates skip detection, not how long stages take.
        previous_durations: dict[str, float] = (
            dict(previous_manifest.stage_durations or {}) if previous_manifest else {}
        )

        if previous_manifest is not None:
            changed_flags = self._skip_flag_differences(previous_manifest)
            if changed_flags:
                log.info(
                    "skip_existing_disabled_due_flag_change",
                    previous_run_id=previous_manifest.run_id,
                    n_changed=len(changed_flags),
                    changed_flags=changed_flags[:10],
                )
                previous_manifest = None

        # Set up progress display; stages advance by predicted duration, not by
        # count, so a 6-hour train doesn't move the bar like a 4-second stage.
        weights = self._stage_weights(stages, previous_durations)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=not is_interactive(),
        ) as progress:
            task_id = progress.add_task("Pipeline", total=sum(weights.values()))

            for stage in stages:
                progress.update(task_id, description=f"[cyan]{stage.name}")

                # Check if this stage was already completed (for resume)
                if self.manifest and stage.name in self.manifest.stages_completed:
                    log.info(
                        "stage_already_completed",
                        stage=stage.name,
                    )
                    progress.advance(task_id, weights[stage.name])
                    continue

                # Check if stage should be skipped
                if self.config.skip_existing and not self.config.dry_run:
                    # `skip_decision` returns before reading roots when there is
                    # no previous manifest, and no previous run means no
                    # previous manifest — so this argument is unread in that
                    # case rather than meaning anything. `None` is the shape
                    # that says so; an empty list would instead say every
                    # recorded output escapes, a tampering verdict this is not.
                    decision = stage.skip_decision(
                        previous_manifest,
                        force=False,
                        allowed_roots=(
                            self._output_verification_roots(previous_run)
                            if previous_run is not None
                            else None
                        ),
                    )
                    if decision.skip:
                        log.info(
                            "stage_skipped",
                            stage=stage.name,
                            reason="inputs unchanged, recorded outputs re-hashed",
                        )
                        if self.manifest:
                            self.manifest.stages_skipped.append(stage.name)
                            if previous_manifest is not None:
                                self._carry_skipped_stage_provenance(
                                    stage, previous_manifest, previous_run
                                )
                            save_run_manifest(self._require_manifest(), self._require_run_dir())
                        progress.advance(task_id, weights[stage.name])
                        continue
                    if decision.outputs_untrusted:
                        log.warning(
                            "stage_outputs_unverified",
                            stage=stage.name,
                            reason=decision.reason,
                            output=decision.key or None,
                            message="recorded outputs failed verification; rerunning the stage",
                        )
                    elif decision.outputs_unverifiable:
                        log.info(
                            "stage_outputs_unverifiable",
                            stage=stage.name,
                            reason=decision.reason,
                            output=decision.key or None,
                            message="no trusted output hashes available; rerunning the stage",
                        )

                # Execute stage
                self._execute_stage(stage)
                progress.advance(task_id, weights[stage.name])

    _FALLBACK_STAGE_SECONDS = 30.0

    def _stage_weights(
        self, stages: list[PipelineStage], previous_durations: dict[str, float]
    ) -> dict[str, float]:
        """Predicted seconds per stage for progress weighting (presentation only).

        Train comes from the runtime predictor when prepared features exist
        (config-aware: it prices the sampler settings this run will use), other
        stages from the previous manifest's durations. With no history at all,
        degrade to equal weights — a bar with made-up proportions is worse than
        a stage-counted one.
        """
        train_predicted = self._predicted_train_seconds() if "train" in (
            s.name for s in stages
        ) else None
        if not previous_durations and train_predicted is None:
            return {s.name: 1.0 for s in stages}
        weights: dict[str, float] = {}
        for s in stages:
            w: float | None = previous_durations.get(s.name)
            if s.name == "train" and train_predicted is not None:
                w = train_predicted
            weights[s.name] = float(w) if w and w > 0 else self._FALLBACK_STAGE_SECONDS
        return weights

    def _predicted_train_seconds(self) -> float | None:
        """Runtime-predictor train weight; None without prepared features to size from."""
        features = Path("data/features/train_features.parquet")
        if not features.exists():
            return None
        try:
            import pandas as pd

            from panelcast.gpu_memory.runtime_predictor import predict_fit_seconds

            n_obs = int(len(pd.read_parquet(features, columns=[])))
            return predict_fit_seconds(
                self.config.num_chains,
                self.config.num_samples,
                self.config.num_warmup,
                n_obs,
                transform=self.config.target_transform,
            ).seconds
        except Exception:
            return None

    # Run-scoped product roots: the stages that write each root and the stages
    # that read it. A root stays in the current run dir when one of its writers
    # is part of this invocation; otherwise consumers read it from the most
    # recent successful run that produced it.
    PRODUCT_WRITERS: dict[str, tuple[str, ...]] = {
        "models": ("train",),
        "evaluation": ("evaluate",),
        "predictions": ("predict",),
        "reports": ("report", "sensitivity"),
    }
    PRODUCT_READERS: dict[str, tuple[str, ...]] = {
        "models": ("evaluate", "predict", "sensitivity"),
        "evaluation": ("report",),
        "predictions": ("report",),
    }

    def _artifact_paths(self) -> ArtifactPaths:
        """Artifact roots for this invocation; flat layout before a run dir exists."""
        if self.run_dir is None:
            return ArtifactPaths.flat()
        if self._resolved_paths is None:
            self._resolved_paths = self._resolve_artifact_paths()
        return self._resolved_paths

    def _resolve_artifact_paths(self) -> ArtifactPaths:
        """Run-scoped roots, with read roots redirected for consumer-only runs."""
        return resolve_artifact_paths(
            run_dir=self._require_run_dir(),
            output_base=self.output_base,
            stages=self.config.stages,
            dry_run=self.config.dry_run,
            product_writers=self.PRODUCT_WRITERS,
            product_readers=self.PRODUCT_READERS,
            find_source=self._find_run_with_product,
        )

    def _find_run_with_product(self, product: str, writers: tuple[str, ...]) -> Path | None:
        """Most recent successful non-dry run whose dir contains ``product``."""
        return find_run_with_product(self.output_base, self.run_dir, product, writers)

    def _create_stage_context(self) -> StageContext:
        """Create StageContext for stage execution.

        Returns:
            StageContext with current configuration.
        """
        return build_stage_context(
            self.config,
            self.descriptor,
            run_dir=self.run_dir or Path("outputs"),
            paths=self._artifact_paths(),
            manifest=self.manifest,
            max_events=self._resolved_event_cap(),
            min_ratings=self._resolved_min_ratings(),
            likelihood_family=self._resolved_likelihood_family(),
            target_transform=self._resolved_target_transform(),
        )

    def _observe_data_stamps(self) -> None:
        """Record on-disk data-root stamps this run hasn't produced or seen yet.

        Covers consumer-only runs (``--stages train,evaluate``) and skipped
        data stages: the first consumer pins the world it starts from, so a
        later consumer in the same run detects a foreign regeneration.
        """
        if self.manifest is None:
            return
        for stage_name, root in DATA_STAGE_ROOTS.items():
            if stage_name in self.manifest.data_stamps:
                continue
            current = read_stamp(root)
            if current is not None:
                self.manifest.data_stamps[stage_name] = current

    def _capture_stage_input_hashes(self, stage: PipelineStage) -> dict[str, str]:
        """Capture per-path input hashes for manifest provenance."""
        hashes: dict[str, str] = {}
        for path in stage.input_paths:
            if not path.exists():
                continue
            try:
                hashes[str(path)] = sha256_path(path)
            except Exception as e:
                log.warning(
                    "input_hash_failed",
                    stage=stage.name,
                    path=str(path),
                    error=str(e),
                )
        return hashes

    def _record_stage_outputs(
        self,
        stage: PipelineStage,
        run_result: Any | None,
    ) -> None:
        """Record stage outputs (and their content hashes) in the manifest."""
        if self.manifest is None:
            return

        recorded: dict[str, str] = {}
        # Static stage output declarations
        for output_path in stage.output_paths:
            if output_path.exists():
                recorded[f"{stage.name}:{output_path.as_posix()}"] = str(output_path)

        # Dynamic run_fn result paths
        if isinstance(run_result, dict):
            for key, value in run_result.items():
                if isinstance(value, (str, Path)):
                    candidate = Path(value)
                    if candidate.exists():
                        recorded[f"{stage.name}:{key}"] = str(candidate)

        self.manifest.outputs.update(recorded)
        hash_started = time()
        for manifest_key, path_str in recorded.items():
            try:
                self.manifest.output_hashes[manifest_key] = sha256_path(path_str)
            except OSError as e:
                log.debug("output_hash_failed", key=manifest_key, error=str(e))
        if recorded:
            log.debug(
                "outputs_hashed",
                stage=stage.name,
                n=len(recorded),
                seconds=round(time() - hash_started, 3),
            )

    def _execute_stage(self, stage: PipelineStage) -> None:
        """Execute a single pipeline stage.

        Args:
            stage: Stage to execute.
        """
        log.info("stage_started", stage=stage.name, description=stage.description)

        if self.manifest:
            self.manifest.input_hashes.update(self._capture_stage_input_hashes(stage))

        if self.config.dry_run:
            # Record nothing beyond the plan: completed stages, stage hashes,
            # or outputs from a run that executed nothing would poison
            # --skip-existing and latest-run resolution with stale state.
            log.info("stage_dry_run", stage=stage.name, would_run=stage.description)
            return

        if stage.name in CONSUMER_STAGES and self.manifest is not None:
            self._observe_data_stamps()
            verify_stamps(self.manifest.data_stamps, stage.name)

        # Create stage context
        ctx = self._create_stage_context()

        stage_started = time()

        # Execute the stage's run function
        if stage.run_fn is None:
            log.warning(
                "stage_no_run_fn",
                stage=stage.name,
                message="Stage has no run function defined",
            )
            run_result = None
        else:
            try:
                run_result = stage.run_fn(ctx)
            except StageSkipped as e:
                log.info("stage_skipped", stage=stage.name, reason=e.message)
                if self.manifest:
                    self.manifest.stages_skipped.append(stage.name)
                    save_run_manifest(self._require_manifest(), self._require_run_dir())
                return
            except ConvergenceError as e:
                # Handle convergence errors: fail in strict mode, warn otherwise
                if self.config.strict:
                    raise
                log.warning(
                    "convergence_warning",
                    stage=stage.name,
                    error=str(e),
                    message="Continuing despite convergence issues (strict=False)",
                )
                # The fit raised before binding run_result; leave it unset so the
                # manifest update below treats the stage like a no-result run
                # instead of raising UnboundLocalError (defeating the "continue").
                run_result = None
            except PipelineError:
                raise
            except Exception as e:
                # Wrap unexpected errors
                raise PipelineError(str(e), stage=stage.name) from e

        # Update manifest
        if self.manifest:
            self.manifest.stages_completed.append(stage.name)
            self.manifest.stage_hashes[stage.name] = stage.compute_input_hash()
            self.manifest.stage_durations[stage.name] = round(time() - stage_started, 3)
            if isinstance(run_result, dict) and isinstance(
                run_result.get("resource_usage"), dict
            ):
                self.manifest.resources[stage.name] = run_result["resource_usage"]
            if stage.name in DATA_STAGE_ROOTS:
                self.manifest.data_stamps[stage.name] = write_stamp(
                    DATA_STAGE_ROOTS[stage.name],
                    stage.name,
                    self.manifest.stage_hashes[stage.name],
                    self.manifest.run_id,
                )
            self._record_stage_outputs(stage, run_result=run_result)
            save_run_manifest(self._require_manifest(), self._require_run_dir())

        log.info(
            "stage_completed",
            stage=stage.name,
            duration_seconds=round(time() - stage_started, 3),
        )

    def _handle_failure(self, error: Exception, stage: str) -> None:
        """Handle pipeline failure with cleanup.

        Args:
            error: The exception that caused failure.
            stage: Name of the stage that failed.
        """
        log.error(
            "pipeline_failed",
            stage=stage,
            error=str(error),
            exc_info=True,
        )

        # Update manifest
        if self.manifest:
            self.manifest.success = False
            self.manifest.error = str(error)
            self.manifest.duration_seconds = time() - self._start_time
            if self.run_dir:
                save_run_manifest(self.manifest, self.run_dir)

        # failure.json is written BEFORE the move so it survives it.
        self._write_failure_payload(error, stage)

        # Close logging handlers before moving directory (Windows file lock issue)
        self._close_log_handlers()

        # Move to failed directory
        final_path = self.run_dir
        if self.run_dir and self.run_dir.exists():
            # Quarantine deletes an existing target; refuse outright rather
            # than rmtree whatever a symlinked quarantine slot points at.
            try:
                failed_path = safe_run_dir(
                    self.output_base,
                    self.run_dir.name,
                    subdir="failed",
                    field="failed run dir",
                )
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                if failed_path.exists():
                    shutil.rmtree(failed_path)
                shutil.move(str(self.run_dir), str(failed_path))
                final_path = failed_path
                log.info("run_moved_to_failed", path=str(failed_path))
            except RunPathError as e:
                log.warning("failed_quarantine_path_rejected", error=str(e))
            except OSError as e:
                # Quarantine is secondary recovery and must not mask the pipeline failure.
                log.warning(
                    "failed_to_move_to_failed",
                    error=str(e),
                    run_dir=str(self.run_dir),
                )

        self._print_failure_epilogue(error, stage, final_path)

    def _write_failure_payload(self, error: Exception, stage: str) -> None:
        """Structured forensics for `runs why`; must never raise."""
        write_failure_payload(self.run_dir, self.manifest, error, stage)

    def _print_failure_epilogue(self, error: Exception, stage: str, final_path) -> None:
        """The 10-second answer to 'what happened and what do I type next'."""
        print_failure_epilogue(error, stage, final_path)

    def _close_log_handlers(self) -> None:
        """Close file handlers to release locks (needed for Windows)."""
        close_log_handlers()

    def _finalize_success(self) -> None:
        """Finalize successful run with manifest update and latest pointer."""
        # Update manifest
        if self.manifest:
            self.manifest.success = True
            self.manifest.duration_seconds = time() - self._start_time
            if self.run_dir:
                save_run_manifest(self.manifest, self.run_dir)

        # latest.json is the authoritative pointer; the link is opportunistic
        # convenience (symlink/junction creation can fail on Windows/NTFS).
        # Dry runs never take the pointer: their run dir holds no artifacts,
        # so latest-run consumers (diagnose, compare, dashboards) would
        # resolve to an empty run.
        if self.run_dir and not self.config.dry_run:
            self._write_latest_pointer()
            self._create_latest_link()

        log.info(
            "pipeline_completed",
            run_id=self.manifest.run_id if self.manifest else "unknown",
            duration=f"{self.manifest.duration_seconds:.2f}s" if self.manifest else "unknown",
            stages_completed=len(self.manifest.stages_completed) if self.manifest else 0,
            stages_skipped=len(self.manifest.stages_skipped) if self.manifest else 0,
        )

    def _write_latest_pointer(self) -> None:
        """Atomically write outputs/latest.json pointing at the current run."""
        if not self.run_dir:
            return
        try:
            run_dir_rel = self.run_dir.relative_to(self.output_base)
        except ValueError:
            run_dir_rel = Path(self.run_dir.name)
        payload = {
            "run_id": self.manifest.run_id if self.manifest else self.run_dir.name,
            "run_dir": run_dir_rel.as_posix(),
        }
        pointer = self.output_base / "latest.json"
        tmp = self.output_base / "latest.json.tmp"
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, pointer)
        except OSError as e:
            log.warning("failed_to_write_latest_pointer", error=str(e))

    def _create_latest_link(self) -> None:
        """Create outputs/latest symlink/junction to current run."""
        if not self.run_dir:
            return

        latest_link = self.output_base / "latest"

        # Remove existing link/junction
        # Wrap exists()/is_symlink() in try/except for WSL where broken
        # symlinks on NTFS can raise OSError (WinError 1920).
        try:
            should_remove = latest_link.exists() or latest_link.is_symlink()
        except OSError:
            should_remove = True
        if should_remove:
            try:
                if sys.platform == "win32" and latest_link.is_dir():
                    # On Windows, junctions appear as directories
                    os.rmdir(latest_link)
                else:
                    latest_link.unlink()
            except Exception as e:
                log.warning("failed_to_remove_latest_link", error=str(e))
                return

        # Create new link
        try:
            if sys.platform == "win32":
                # Try symlink first (requires Developer Mode or admin)
                try:
                    os.symlink(self.run_dir, latest_link, target_is_directory=True)
                    log.debug("created_symlink", target=str(self.run_dir))
                except OSError:
                    # Fall back to directory junction (no special permissions)
                    # Validate paths don't contain shell metacharacters
                    link_str = str(latest_link)
                    target_str = str(self.run_dir)
                    if any(c in link_str + target_str for c in "&|;<>`$^%\r\n"):
                        log.warning("unsafe_path_characters", link=link_str, target=target_str)
                        return
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", link_str, target_str],
                        capture_output=True,
                        check=True,
                    )
                    log.debug("created_junction", target=target_str)
            else:
                os.symlink(self.run_dir, latest_link, target_is_directory=True)
                log.debug("created_symlink", target=str(self.run_dir))
        except Exception as e:
            log.warning("failed_to_create_latest_link", error=str(e))


def run_pipeline(config: PipelineConfig, output_base: Path | str = Path("outputs")) -> int:
    """Convenience function to run pipeline with given configuration.

    Creates an orchestrator and runs the pipeline, returning the exit code.

    Args:
        config: Pipeline configuration.
        output_base: Base directory for outputs (default "outputs").

    Returns:
        Exit code: 0 on success, error's exit_code on failure.

    Example:
        >>> exit_code = run_pipeline(PipelineConfig(seed=42))
    """
    orchestrator = PipelineOrchestrator(config, output_base=output_base)
    return orchestrator.run()
