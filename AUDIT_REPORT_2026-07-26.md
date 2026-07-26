# panelcast Deep Audit Report

- **Repository:** `cupidthatbtc/panelcast`
- **Report date:** 2026-07-26
- **Audited release branch:** `release/0.20.0`
- **Audited release head:** `603e3fc4c797207dcd4a3f3e25037859ccdb2877`
- **Audit type:** Read-only repository, implementation, statistical, security, packaging, CI/CD, documentation, and operational review

## Executive assessment

### Overall condition: good, above-average, but not fully release-hardened

panelcast is **not in bad shape**. It is a serious, unusually well-documented research-software repository with broad tests, explicit reproducibility machinery, typed public boundaries, detailed scientific protocols, pinned GitHub Actions, automated release workflows, model cards, data-lineage records, and substantial defensive validation.

My subjective overall engineering grade is:

| Dimension | Assessment | Approximate grade |
|---|---:|---:|
| Repository organization and documentation | Strong | A- |
| Test depth and scientific verification culture | Strong | A- |
| Core implementation quality | Good | B+ |
| Reproducibility design | Ambitious but has critical edge-case gaps | B |
| Statistical implementation correctness | Good overall, with two material calibration defects | B |
| Packaging and installation consistency | Mixed | C+ |
| Security and supply-chain posture | Mixed | C+ |
| Operational durability and concurrency | Needs hardening | C+ |
| Maintainability and typing consistency | Good foundation, meaningful debt | B- |
| Overall repository health | Good | **B / roughly 7.5 out of 10** |

If the high-priority findings were fixed, the repository could plausibly move into the **8.5–9/10 range**. The fundamentals are already present; the principal problems are concentrated around difficult failure modes and trust boundaries rather than pervasive low-quality code.

### How extreme are the findings?

The findings are **serious in consequence but not evidence that the whole repository is broken**.

There are no confirmed P0 emergencies:

- No evidence of an active compromise.
- No evidence that secrets have already been exfiltrated.
- No public unauthenticated remote-code-execution path was established.
- No evidence that ordinary successful pipeline runs are universally invalid.
- Current CI is green.
- All focused suites used to validate the findings passed.
- All Python files parse successfully.
- No apparent committed credentials or private keys were found.

The P1 findings are nevertheless legitimate release blockers because they can affect:

- Filesystem containment and destructive failure handling.
- MCMC checkpoint continuity.
- Whether corrupted outputs are silently reused.
- Statistical interval and calibration claims.
- GPU admission guarantees.
- Credential-bearing CI execution.
- Dependency vulnerability exposure.

Many require a particular edge condition—such as a crash between checkpoint writes, an untrusted same-repository branch, an adversarial resume identifier, discrete predictive distributions, or concurrent calibration writers. That lowers their day-to-day likelihood, but their impact is high enough that they should not be dismissed as theoretical.

### Practical release judgment

The repository is suitable for continued development and controlled research use. I would not describe it as unsafe or unusable.

I would, however, avoid making the strongest possible claims about:

- Crash-safe checkpoint parity.
- Tamper-proof incremental execution.
- Uniform PIT calibration for discrete likelihoods.
- Strict filesystem containment of all run-selection paths.
- Secure execution of collaborator-authored pull requests in the reviewer workflow.
- Fully audited dependency security.

For a high-confidence v0.20 release, the most important release-gating work is:

1. Resume path containment.
2. Transactional checkpointing.
3. Output-hash verification before incremental skips.
4. HDI and discrete PIT corrections.
5. Claude review workflow isolation.
6. Dependency refresh and automated vulnerability scanning.
7. GPU calibration invariant enforcement.

## Audit scope

- Audited all **590 tracked files**.
- Read all **583 UTF-8 text files**, totaling approximately **181,290 lines**.
- Parsed all **404 Python files** with the Python AST.
- Visually inspected all five tracked PNG images.
- Loaded both NPZ fixtures with `allow_pickle=False` and inspected every array, shape, dtype, and finiteness.
- Checked all 24 Markdown documents for broken relative links.
- Inspected repository metadata, Git history, workflows, release automation, packaging, dependencies, scripts, configuration, statistical routines, serialization, concurrency, GPU scheduling, tests, documentation, generated audit evidence, and GitHub repository settings.
- The full-tree pass began at v0.19.0 commit `613f699`.
- The terminology/API delta through merged PR #343 was reviewed separately.
- The final v0.20.0 release metadata delta through `603e3fc` was also reviewed.

No GitHub issue, pull request, setting, commit, tracked source file, or branch was modified during the audit itself.

## What the repository does well

The strongest parts of the repository are important context for the findings:

- Extensive unit, integration, smoke, nightly, statistical, and domain-portability testing.
- Explicit coverage gates in CI.
- Detailed model cards, evaluation protocols, configuration specifications, decisions, data lineage, and external-domain playbooks.
- Strong attention to leakage prevention and temporal evaluation.
- Run manifests, data stamps, hashes, environment fingerprints, and reproduction commands.
- Structured handling of model gates and descriptor-driven domain portability.
- Pinned GitHub Action commit SHAs.
- PyPI trusted publishing through OIDC.
- GitHub secret scanning and push protection.
- Typed and documented top-level public API.
- Strong targeted tests around checkpoints, warm starts, GPU memory, calibration, prediction, model gates, and resume identity.
- No Python syntax failures.
- No broken local documentation links.
- No apparent committed secrets or private keys.

The repository’s problems mainly arise because its reproducibility and scientific-control systems are ambitious. Those systems create additional correctness requirements around atomicity, identity, trust, and failure recovery.

## Severity definitions

| Priority | Meaning |
|---|---|
| P0 | Active compromise, unrecoverable corruption, or universally broken core behavior. None confirmed. |
| P1 | Release-blocking security, scientific-integrity, destructive-path, or reproducibility defect. |
| P2 | Material correctness, reliability, packaging, or operational defect that should be scheduled promptly. |
| P3 | Maintainability, portability, documentation, governance, or lower-exposure hardening. |

## P1 findings

### P1-01: Resume identifiers permit path traversal and potentially destructive moves

**Evidence**

- `PipelineConfig._validate_run_id()` validates only `run_id`.
- `resume` is joined directly to `output_base`.
- Direct configuration construction accepts values such as `../outside`, `..\outside`, `failed/other`, and `.`.
- The failure handler can later move the selected directory into `outputs/failed` and remove an existing failed destination.
- `runs` commands and `latest.json` resolution use related paths without one centralized containment policy.

**Primary locations**

- `src/panelcast/pipelines/orchestrator.py:515`
- `src/panelcast/pipelines/orchestrator.py:1085`
- `src/panelcast/pipelines/orchestrator.py:1991`
- `src/panelcast/cli/runs_cmd.py:19`
- `src/panelcast/paths.py:73`

**Impact**

An adversarial or mistaken resume identifier can select a directory outside the intended output root. Subsequent failure handling can move that directory, creating a data-loss and filesystem-containment risk.

**Recommended acceptance criteria**

- Route every run lookup through one safe resolver.
- Require a bare run identifier.
- Resolve the path and prove it remains beneath `output_base`.
- Reject separators, traversal, absolute paths, symlink escapes, reserved names, and out-of-root `latest.json` targets.
- Test CLI, YAML, direct API, resume, show, reproduce, failure handling, Windows paths, and symlinks.

### P1-02: MCMC checkpoint updates are not transactional

**Evidence**

Each checkpoint block performs:

1. Write `block_NNNN.npz`.
2. Overwrite `state.pkl`.
3. Overwrite `cursor.json`.

This occurs in `src/panelcast/models/bayes/fit.py:430`.

**Impact**

A crash after the state update but before the cursor update leaves an advanced Markov-chain state paired with an old block number. Resume then reruns the previous block from the advanced state, overwrites that block, and silently creates a chain gap. Partial state or cursor writes can also make the checkpoint unreadable.

**Recommended acceptance criteria**

- Store immutable state per block.
- Write each artifact through a temporary file.
- Flush/fsync and atomically replace.
- Record checksums and exact block/state identities.
- Commit the cursor only after every referenced artifact is durable.
- Validate keys, shapes, draw counts, block counts, hashes, and identities on load.
- Add fault-injection tests after every write boundary.
- Include every output-affecting fit argument in checkpoint identity.

### P1-03: `--skip-existing` ignores recorded output hashes

**Evidence**

`PipelineStage.should_skip()` in `src/panelcast/pipelines/stages.py:284` checks:

- Previous input hash.
- Current input hash.
- Output existence.

It does not compare the exact output hashes already stored in the prior run manifest.

**Impact**

A truncated, manually edited, corrupted, or substituted artifact can be treated as unchanged and consumed downstream.

**Recommended acceptance criteria**

- Rehash every previous output before allowing a skip.
- Resolve recorded paths safely and verify containment.
- Treat missing legacy hashes as unable to skip safely.
- Rerun or fail loudly on mismatch.
- Test mutation and truncation of Parquet, JSON, model, and directory outputs.

### P1-04: HDI calculation contains an off-by-one statistical error

**Evidence**

`_hdi_per_observation()` in `src/panelcast/evaluation/calibration.py:163` calculates:

```text
window = ceil(probability * number_of_samples)
upper = sorted_samples[start + window]
```

The resulting inclusive interval contains `window + 1` samples.

Reproduced examples:

- `n=10`, `probability=.5`: intended 5 samples, returns 6.
- `n=10`, `probability=.8`: intended 8 samples, returns 9.
- `n=4`, `probability=.5`: intended 2 samples, returns 3.

**Impact**

Intervals are wider than requested. This can affect coverage, sharpness, WIS-related reporting, and model-comparison conclusions.

**Recommended acceptance criteria**

- Correct the inclusive upper endpoint and candidate-window count.
- Add exact sample-mass properties across probabilities, ties, and small arrays.
- Compare against an independent ArviZ/reference implementation.
- Regenerate any published evidence that depends materially on this HDI implementation.

### P1-05: Discrete PIT is described as randomized but implements deterministic mid-P

**Evidence**

`src/panelcast/evaluation/calibration.py:562` computes:

```text
(number_below + 0.5 * number_equal) / number_of_draws
```

This is deterministic mid-P, not randomized PIT.

**Impact**

For discrete predictive distributions, the result is not uniform even under correct calibration. A balanced Bernoulli model, for example, produces point masses near `.25` and `.75`. This can distort PIT histograms and quantile-recalibration behavior.

**Recommended acceptance criteria**

- Implement seeded randomized PIT across the equality mass, or rename the metric and weaken its claims.
- Define conventions for censored and beta-binomial observations.
- Add simulation-based uniformity tests.
- Coordinate with the remaining scientific-validation criteria in issue #234.

### P1-06: GPU calibration can violate its stated “never under” contract

**Evidence**

`refit_constants()` in `src/panelcast/gpu_memory/calibration_store.py:165` promises that every local calibration point remains at least 1.05x over-covered.

The implementation performs no more than five scaling iterations and returns without verifying the final invariant. Property probes produced final ratios below the promised envelope.

Non-finite `actual_peak_gb` values can also enter the fitting path and produce non-finite coefficients that are later selected.

**Impact**

The admission layer may use a local calibration that underestimates required GPU memory despite explicitly claiming it never does so.

**Recommended acceptance criteria**

- Reject non-finite records and coefficients.
- Solve the envelope constraints directly or iterate until verified with a defined tolerance.
- Fall back to shipped constants when verification fails.
- Add adversarial property tests across broad scales and leverage points.
- Report per-machine calibration only after the invariant passes.

### P1-07: Credential-bearing Claude review can execute pull-request-controlled code

**Evidence**

`.github/workflows/claude-review.yml`:

- Runs for same-repository pull requests.
- Checks out PR code.
- Passes `CLAUDE_CODE_OAUTH_TOKEN`.
- Grants pull-request write and OIDC permissions.
- Allows `Bash(pixi run *)`.

The pinned action places Claude OAuth, GitHub, and OIDC information in its process environment. Its subprocess-environment scrubbing mode is not enabled by this workflow.

Fork pull requests are excluded, which is good, but a malicious or compromised collaborator account can still change Pixi tasks or prompt-inject the reviewer into running branch-controlled code.

**Impact**

Potential credential exposure, unauthorized PR actions, or supply-chain compromise within the collaborator trust boundary.

**Recommended acceptance criteria**

- Run tests in a separate secretless job.
- Remove Bash execution from the credential-bearing reviewer.
- Prefer short-lived workload identity to a static OAuth secret.
- Enable subprocess-environment scrubbing where supported.
- Minimize PR and OIDC permissions.
- Add a regression test proving PR code cannot read secret-bearing environment variables.

### P1-08: Locked dependencies have current vulnerability matches and no automated security signal

**Definite PyPI matches**

- GitPython 3.1.46: 13 distinct advisories involving command injection, file overwrite, path traversal, config injection, and environment-variable exfiltration. The cumulative fixed target is 3.1.55.
- orjson 3.11.5: deeply nested JSON recursion denial of service, fixed in 3.11.6.

**Additional conda-version heuristic**

A second PyPI-version crosscheck flagged 35 advisories against identically versioned conda packages:

- Click
- Pillow
- pip
- PyArrow
- Pygments
- pytest
- setuptools
- Tornado

Those conda matches require a conda-aware scanner to determine whether package builds contain backported fixes.

**Repository-security gap**

- Dependabot security updates are disabled.
- No code-scanning analysis exists.
- No working SBOM/dependency-graph signal was available.

**Recommended acceptance criteria**

- Update GitPython to at least 3.1.55.
- Update orjson to at least 3.11.6.
- Run a conda-aware vulnerability audit and update affected artifacts.
- Generate CycloneDX or SPDX SBOMs for releases.
- Add automated OSV/dependency scanning for both Pixi/conda and PyPI artifacts.
- Block high/critical vulnerabilities unless a reviewed exception documents non-exposure.

## P2 findings

### P2-01: Plain pip and wheel installs silently lose Git provenance

GitPython is declared in `pixi.toml` but not in the runtime dependency list in `pyproject.toml`.

A documented pip installation records:

- `commit=gitpython-not-installed`
- `branch=unknown`
- `dirty=False`
- `untracked=0`

`panelcast doctor` can treat this placeholder as a pass.

Add GitPython with a secure minimum version, or implement a subprocess-based Git fallback. Missing provenance should be WARN, FAIL, or explicitly not-applicable—not PASS.

### P2-02: Static figure export is advertised but unavailable in plain installs

Kaleido is available through Pixi but omitted from pip runtime dependencies. `export-figures` defaults to static formats that require Kaleido.

Some format combinations bypass the early availability check and fail later. Kaleido may also download Chrome at runtime without a prominent confirmation boundary.

Recommended options:

- Add a documented `panelcast[export]` extra.
- Or include Kaleido in runtime dependencies.
- Validate every requested static format before starting.
- Disclose and confirm runtime browser downloads.

### P2-03: `PipelineConfig` validation is incomplete and bypassable

Direct construction and YAML accepted values including:

- `logit_offset=0`
- `likelihood_df=-1`
- `n_exponent_alpha=-2`
- `n_exponent=2`
- `val_albums=-1`
- `predictive_batch_size=0`
- `chain_method="made-up"`
- `coverage_tolerance=NaN`

An offset of zero produces infinite transformed values at score bounds.

CLI constraints do not protect YAML or library callers. Validation should live in the configuration objects themselves and cover finite values, ranges, enum choices, batch sizes, split counts, diagnostics, exponents, offsets, and prior scales.

### P2-04: Feature pipelines permit duplicate output columns

`FeaturePipeline.transform()` concatenates block outputs without checking column uniqueness.

Descriptor validation does not prevent:

- Duplicate block names.
- Blocks that emit the same column.
- Raw-column mappings with duplicate canonical targets.
- Basis/output-name collisions.

A custom two-block probe produced a DataFrame with non-unique columns.

Fail before transformation or persistence with a complete collision report.

### P2-05: Genre PCA fails for small datasets and genre parsing is brittle

The PCA validator compares `n_components` with the number of genre features but not with the number of samples. PCA requires:

```text
n_components <= min(n_samples, n_features)
```

A two-row, four-genre frame with three components crashes.

The genre parser also splits only the literal comma-plus-space sequence. `"Rock,Jazz"` becomes one token.

Validate both PCA dimensions and make separators part of the descriptor/schema contract.

### P2-06: Authoritative manifests use non-atomic writes

`save_run_manifest()` directly overwrites `manifest.json`. A crash can truncate the artifact required for:

- Resume.
- Verification.
- History.
- Reproduction.
- Output attribution.

Split manifests contain similar direct writes.

Use temporary files, flush/fsync, atomic replace, and directory durability where supported.

### P2-07: `runs reproduce` can collide or compare the wrong run

The saved resolved configuration includes `run_id`. Reproduction clears resume and skip behavior but not the saved run ID, so named runs can collide with their source.

Automatic-ID reproduction compares against the mutable `latest` pointer. A concurrent run can finish first and be compared instead.

Generate a fresh known reproduction ID and compare that exact directory.

### P2-08: Warm-start and checkpoint artifacts use executable pickle

User-selectable warmup imports and checkpoint resumes call `pickle.load`.

Loading a downloaded, shared, or otherwise untrusted artifact can execute arbitrary Python.

Prefer schema-validated JSON/NPZ serialization of array leaves. Until replacement is practical:

- Label artifacts as trusted-code inputs.
- Warn prominently in CLI and documentation.
- Reject obviously unsafe ownership or permissions where possible.

### P2-09: Target-bound semantics are inconsistent

The offset-logit inverse can return values in:

```text
[low - offset, high + offset]
```

Documentation often claims predictions stay inside dataset bounds.

Current behavior is inconsistent:

- Known-entity prediction clips.
- Cold-start prediction does not.
- Evaluation does not consistently clip.
- Monitoring hard-codes 0/100.
- Interval-width warnings hard-code 80.
- Prediction summary hard-codes batch size 500.

Define one descriptor-aware support policy and apply it consistently to training, evaluation, prediction, reporting, monitoring, and model-card language.

### P2-10: Concurrent GPU calibration writes lose records

Calibration append is read-modify-write with a per-process temporary file. Two processes can read the same old store; the final replace loses the other process’s record.

Use:

- A cross-process lock.
- Append-only per-process shards with merge.
- Or a compare-and-swap retry loop.

Also use UTC timestamps and quarantine unreadable stores for diagnosis.

### P2-11: `panelcast doctor` is not strictly read-only

The module and CLI documentation claim strict read-only behavior. The cache probe creates directories, writes a file, deletes the file, and can leave newly created directories behind.

Either:

- Stop claiming strict read-only behavior and document the probe.
- Or avoid persistent filesystem creation and guarantee cleanup of everything created.

### P2-12: Repository and release governance lacks enforcement

Current repository state:

- `main` is unprotected.
- No repository rulesets.
- No tag-protection rules.
- Any `v*` tag can trigger a release.
- Dependabot security updates are disabled.
- No code-scanning results.
- No release SBOM.
- No `SECURITY.md`.
- No `CODEOWNERS`.
- No code of conduct.
- Release build tooling is not fully pinned.

Existing positive controls:

- GitHub Action SHAs are pinned.
- PyPI publishing uses trusted OIDC.
- Secret scanning is enabled.
- Push protection is enabled.

Recommended improvements:

- Protect `main`.
- Require CI and reviews.
- Protect release tags or use protected environments.
- Generate provenance and SBOM attestations.
- Add security policy and ownership metadata.

### P2-13: Per-group cold-start variance is trained but not used

Cold-start prediction deliberately uses pooled `sigma_artist` instead of the trained `sigma_artist_group[group]`.

This is documented and therefore is a scientific limitation rather than a hidden implementation bug. It should receive a follow-up only if group-specific cold-start uncertainty is part of the intended claims.

## P3 findings

### P3-01: First GPU job is admitted even when estimated to exceed headroom

The admission layer always admits the first job. This can knowingly launch a fit that is predicted not to fit.

Fail early or require an explicit force-alone policy.

### P3-02: Model-library writes can collide

Model filenames use second-level timestamp resolution, and the library manifest is updated through an unlocked read-modify-write cycle.

Two saves of the same model type in the same second, or concurrent saves, can overwrite records.

### P3-03: Developer-specific paths remain in scripts and historical evidence

Examples include:

- `/mnt/c/Users/jcwen/Projects/panelcast`
- `/home/jcwen/miniforge3`
- Named local Pixi/Conda environments.
- Developer-specific historical run directories.

One CSV builder broadly catches failures and can silently produce incomplete output.

Parameterize active scripts and clearly label historical evidence paths as archival and non-portable.

### P3-04: Dashboard HTML title is not escaped

`src/panelcast/visualization/export.py` inserts arbitrary title text directly into `<title>` and heading markup.

This can create local dashboard script injection if the title is untrusted. The separate reporting HTML implementation already escapes correctly and should be used as the model.

### P3-05: Environment expansion can leak values into manifests

The configuration loader expands environment variables, while documentation suggests this keeps secrets out of version control.

Expanded values can be persisted in resolved configuration or manifest data. Known fields are mostly non-secret paths, so current exposure is limited, but the contract should be explicit.

Either add redaction or state clearly that configuration interpolation is not a secret-management system.

### P3-06: Timestamps are inconsistent and sometimes timezone-naive

Run IDs, manifests, calibration records, and other artifacts use a mixture of:

- Local timestamps.
- Naive ISO strings.
- UTC dates.

Standardize on timezone-aware UTC and preserve timezone information in serialized artifacts.

### P3-07: Ranking ties are resolved by row order

Top-K selection uses stable row-order tie breaking. For rounded or discrete predictions, this can arbitrarily select among tied entities and distort ranking metrics.

Add:

- A documented tie policy.
- Tie-aware metrics where appropriate.
- Shape validation.
- Entity-length validation.
- Draw-count validation.
- Positive-`k` validation.

### P3-08: Baseline methodology language is inconsistent

Some core descriptions call residual resampling distribution-free or exact, while the dedicated baseline documentation correctly describes it as a Monte Carlo approximation.

Use the cautious description consistently.

### P3-09: A remaining asymmetry TODO points at a closed issue

`src/panelcast/models/bayes/model.py` contains a response-side asymmetry TODO linked to a closed issue.

Resolve it or give it a current tracking issue so it does not appear intentionally completed.

### P3-10: Documentation-figure polish

The tracked pairplot has overlapping bottom-axis labels. The other four tracked images were visually sound.

Generated audit JSON files frequently omit a final newline, and the pull-request template has trailing whitespace.

These are low impact but easy to normalize.

### P3-11: Internal split-helper API changed without a compatibility bridge

The v0.20 terminology tranche changes:

- `test_albums` to `test_events`
- `val_albums` to `val_events`
- `min_train_albums` to `min_train_events`
- `assert_no_artist_overlap` to `assert_no_entity_overlap`

The top-level documented public API remains separate, so this is not necessarily a semantic-versioning violation. Nevertheless, downstream users importing internal helpers by keyword will break.

Document the boundary prominently or provide one-cycle deprecation aliases.

## Code-quality inventory

### Static structure

- 404 Python files parsed successfully.
- Approximately 7,338 function definitions and 1,520 class definitions were observed by recursive AST traversal.
- No invalid UTF-8 text files.
- No CRLF inconsistency.
- No obvious committed private keys or credentials.

Largest production modules included:

- `pipelines/orchestrator.py`: approximately 2,195 lines.
- `pipelines/evaluate.py`: approximately 2,149 lines.
- `pipelines/train_bayes.py`: approximately 2,010 lines.
- `pipelines/sensitivity.py`: approximately 1,340 lines.
- `pipelines/publication.py`: approximately 1,296 lines.
- `models/bayes/model.py`: approximately 1,214 lines.
- `reporting/figures.py`: approximately 1,181 lines.
- `reporting/model_card.py`: approximately 1,144 lines.
- `select/runner.py`: approximately 1,130 lines.

Large functions included:

- `train_models`: approximately 599 lines.
- Main CLI run command: approximately 584 lines.
- `make_score_model`: approximately 521 lines.
- `predict_new_entity`: approximately 346 lines.
- `fit_model`: approximately 316 lines.
- `evaluate_models`: approximately 297 lines.

`_build_command_string` has roughly 83 branches, and `run_sweep` roughly 62.

This is not proof of incorrectness, but it raises review cost and makes output-affecting changes harder to reason about.

### Ruff

Current repository-wide `ruff check` findings:

1. `scripts/generate_trajectories.py`: `main` complexity 25 exceeds threshold 22.
2. `tests/unit/select/test_parallel_prereqs.py`: unsorted import block.

`ruff format --check` reports approximately 132 files needing formatting.

CI currently runs `ruff check src/`, which misses:

- Scripts.
- Tests.
- Formatter drift.
- Repository-level generated helpers.

An extended lint pass found approximately 881 maintainability diagnostics, dominated by:

- Import-inside-function.
- Excessive arguments.
- Excessive statements.
- Excessive branches.
- Magic comparisons.
- Unused suppressions.
- Naive datetime usage.
- Mutable class defaults.

### Typing

Boundary mypy targets pass.

The whole-package typing ratchet tracks approximately 90 existing errors. The global mypy configuration remains intentionally lenient:

- Some error codes disabled.
- Missing imports ignored.
- Untyped definitions allowed.
- `warn_return_any` disabled.
- `warn_unused_ignores` disabled.

This is a reasonable staged migration, but new strictness should continue to be ratcheted rather than leaving the baseline static.

## Test assessment

### Positive results

Focused suites passed for:

- GPU memory.
- Feature configuration.
- Replication.
- Checkpoints.
- Warm starts.
- Resume behavior.
- Next-event prediction.
- Orchestrator behavior.
- Calibration.
- Conformal prediction.
- Metrics.
- Ranking.
- Terminology/split refactor.

The current GitHub fast-test job passed in approximately 7 minutes 39 seconds.

### Runtime concern

A complete local fast-unit invocation exceeded five minutes without producing a failure before the audit timeout. This is consistent with the GitHub runtime rather than evidence of a failed suite.

The fast selection contains approximately:

- 5,703 collected tests.
- 63 deselected.
- 5,640 selected.

Consider sharding the fast suite and profiling its slowest fixtures/tests. A seven-to-eight-minute pull-request feedback loop is acceptable but no longer particularly fast.

### Coverage evidence

CI declares a 95% coverage gate.

The local `.coverage` artifact was stale and referred to a different WSL filesystem path, so it was not used as evidence for current local coverage.

## Packaging and distribution assessment

### Strengths

- Clear `pyproject.toml`.
- Typed package surface.
- Semantic-versioning documentation.
- Cross-platform Pixi lock.
- Wheel workflow.
- Trusted PyPI publishing.
- Release metadata synchronized across project files.

### Weaknesses

- Pip and Pixi installations do not have equivalent functionality.
- Git provenance silently degrades in pip installations.
- Static export silently depends on Pixi-only packages.
- Build-tool version ranges remain partially unpinned.
- The release trigger trusts any matching version tag.
- No release SBOM or vulnerability gate.

## Documentation assessment

Documentation quality is one of the repository’s strongest areas.

Reviewed documentation covers:

- Public API.
- Configuration.
- Model card.
- Data lineage.
- Evaluation protocol.
- Decisions.
- Baselines.
- External-domain onboarding.
- Replication.
- Release results.
- Contribution process.

No broken relative Markdown links were found.

Remaining documentation concerns:

- Bound semantics are stronger in prose than in implementation.
- “Randomized PIT” does not match the actual calculation.
- `doctor` overstates read-only behavior.
- Environment expansion language can be read as secret-management guidance.
- Baseline exactness wording is inconsistent.
- Some legacy artist/album terminology remains.
- The pairplot needs layout cleanup.

## GitHub and governance assessment

### Positive controls

- Secret scanning enabled.
- Push protection enabled.
- GitHub Actions pinned by commit.
- PyPI OIDC publishing.
- Active issue tracking.
- Current CI passing.

### Missing controls

- No main-branch protection.
- No rulesets.
- No protected release tags.
- Dependabot security updates disabled.
- No code-scanning analysis.
- No SBOM/dependency-graph result.
- No `SECURITY.md`.
- No `CODEOWNERS`.
- No code of conduct.

These omissions do not make the source code bad, but they lower release and collaborator-compromise resilience.

## Existing issue reconciliation

Five GitHub issues remained open at the end of the audit:

1. **#303 — terminology debt**
2. **#234 — censored observations at bounds**
3. **#233 — asymmetric random-walk innovations**
4. **#232 — skew-normal latent population prior**
5. **#15 — full-corpus scale validation**

The v0.19 code for #232–#234 is implemented and documented. Their issue acceptance criteria also require scientific screening or confirmation evidence, so they should not simply be closed as stale.

Recommended tracker action:

- Mark implementation as shipped.
- Preserve the remaining scientific-validation checklist.
- Link exact evidence artifacts when the screening ladder completes.

Issue #15 remains a real publication-scale limitation.

Issue #303 remains active after the v0.20 tranche because serialized/YAML terminology and additional generic surfaces still use older domain language.

## Recommended remediation sequence

### Phase 1: release and trust boundary

1. Contain resume and run-resolution paths.
2. Isolate the Claude reviewer from PR-controlled code and secret-bearing environments.
3. Refresh vulnerable dependencies.
4. Add SBOM and vulnerability scanning.
5. Protect `main` and release tags.

### Phase 2: scientific and artifact integrity

1. Make checkpoint writes transactional.
2. Verify output hashes before skipping work.
3. Correct HDI indexing.
4. Implement or correctly label discrete PIT.
5. Enforce the GPU calibration envelope.
6. Regenerate materially affected calibration evidence.

### Phase 3: packaging and configuration

1. Align pip and Pixi provenance behavior.
2. Define static-export installation extras.
3. Centralize `PipelineConfig`, MCMC, and prior validation.
4. Reject feature-schema collisions.
5. Harden genre parsing and PCA dimensions.

### Phase 4: operational durability

1. Make run and split manifests atomic.
2. Fix reproduction run identity and latest-pointer races.
3. Replace or explicitly constrain pickle artifacts.
4. Lock GPU telemetry updates.
5. Lock model-library manifests and strengthen names.

### Phase 5: maintainability and polish

1. Expand CI linting to tests, scripts, and format checks.
2. Continue the typing ratchet.
3. Split the largest orchestration and evaluation functions.
4. Shard/profile the test suite.
5. Remove hard-coded paths.
6. Normalize UTC timestamps.
7. Finish terminology cleanup.
8. Correct documentation claims and figure layout.

## Final conclusion

panelcast is a **good repository with several serious edge-case defects**, not a poor repository full of routine mistakes.

Its strongest qualities are:

- Scientific discipline.
- Test breadth.
- Documentation.
- Reproducibility ambition.
- Domain-portability architecture.
- Explicit evaluation and model-selection controls.

Its weakest qualities are:

- Crash consistency.
- Filesystem containment.
- Trust isolation in automated review.
- Dependency-security automation.
- Pip/Pixi parity.
- A small number of statistically important calibration details.

The most accurate concise summary is:

> **Strong research software, good engineering foundations, and unusually thorough documentation—currently held back from top-tier release confidence by a concentrated set of high-impact integrity and security edge cases.**

No evidence suggests that the entire model or repository must be rewritten. Most findings have localized, testable fixes. Resolving the P1 group would materially change the release-confidence assessment from approximately **5.5–6/10** to roughly **8/10 or better**, without requiring a fundamental redesign.
