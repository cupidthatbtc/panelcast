# panelcast preflight audit — whole-codebase bug hunt + feature/scope scouting

2026-07-09, ahead of v0.12.0 (last major release before the official release). 9 subsystem finders + 3 scouts + 3 adversarial verifier lenses, all findings majority-confirmed; several reproduced by execution. **64 findings: 1 red / 16 orange / 28 yellow / 19 notes. 33 ideas (25 proposal seeds).**

## Findings

### RED (1)

#### src/panelcast/pipelines/orchestrator.py:1319 — --checkpoint-every / YAML checkpoint_every is silently inert through the pipeline: StageContext never receives checkpoint_every_draws

PipelineConfig.checkpoint_every_draws is validated, printed in the manifest command string, and listed in RESUME_CONFIG_KEYS, but StageContext (stages.py) has no checkpoint_every_draws field and _create_stage_context never passes it. train_bayes.py:1174 reads getattr(ctx, "checkpoint_every_draws", None), so it is always None: `panelcast run --checkpoint-every 200` on a 6-hour GPU fit creates no checkpoint dir and runs a single-shot fit (verified: field absent from dataclasses.fields(StageContext), name absent from _create_stage_context source). When the process dies at draw 1900/2000 there is no block to resume from — the exact loss the flag promises to prevent (help text: "an interrupted run resumes from the last block"). Compounding: the flags dict in _setup_run also omits checkpoint_every_draws, so even after fixing the threading, every `--resume` reverts it to None and logs a spurious resume_config_missing warning.

**Fix:** Add `checkpoint_every_draws: int | None = None` to StageContext (stages.py, next to chain_method), pass `checkpoint_every_draws=self.config.checkpoint_every_draws` in _create_stage_context, and add `"checkpoint_every_draws": self.config.checkpoint_every_draws` to the manifest flags dict in _setup_run so RESUME_CONFIG_KEYS can actually restore it.

### ORANGE (16)

#### src/panelcast/models/bayes/fit.py:333 — Checkpoint identity hashes only y/X + MCMCConfig, so a resume silently mixes draws from two different models

CONFIRMED by repro. _checkpoint_identity covers MCMCConfig, a hash of run_args['y']/['X'], and numpyro/jax versions — nothing else. Everything else that defines the posterior rides run_args and is unchecked: priors (all gates and scales), likelihood_df, n_exponent/learn_n_exponent, ar_center, prev_score, artist_idx, album_seq, n_reviews, n_ref, target_bounds, and collection settings. Production uses a fixed dir (models/checkpoint, train_bayes.py:1255). Repro: blocked fit interrupted after 2/3 blocks, rerun with prev_score shifted +20 and likelihood_df 4→30 — resume was ACCEPTED (resumed_from_checkpoint=True, no error), and the published posterior concatenates blocks drawn from two different models. The refusal message ('config, data, or numpyro/jax version changed') promises protection it doesn't provide; a user tweaking a CLI flag (--likelihood-df, --ar-centering, prior overrides, --n-exponent) after a crash is the natural workflow this feature exists for. Toggling --exclude-rw-raw-from-collection mid-checkpoint is a milder variant (blocks disagree on saved sites).

**Fix:** Extend _checkpoint_identity to cover the whole model input: hash every array in run_args (np.ascontiguousarray(...).tobytes() in sorted key order), and JSON-serialize the scalar/config args including asdict(priors), likelihood_df, ar_center, n_exponent, learn_n_exponent, n_ref, target_bounds; optionally also include _model_latent_signature(model, run_args) to catch model-function changes.

#### src/panelcast/models/bayes/diagnostics.py:173 — check_convergence passes fits whose parameters have NaN or stuck-constant chains (NaN r-hat/ESS silently skipped)

CONFIRMED by repro. summary['r_hat'].max() and summary['ess_bulk'].min() skip NaN (pandas skipna default), and NaN compares False in the failing-params filters. A posterior containing an all-NaN parameter (numerical blow-up — precedent exists: the econ log-citation sigma_obs collapse) plus a zero-variance stuck parameter returns passed=True, rhat_max=1.0, failing_params=[]. This is the publication gate: train_bayes strict mode (ConvergenceError) never fires for such a fit, and it gets saved and published. Additionally, if ALL ess_bulk values are NaN, int(nan) raises ValueError instead of reporting failure.

**Fix:** Treat NaN diagnostics as failures for non-constant sites: flag params where summary['r_hat'].isna() or summary['ess_bulk'].isna() (excluding sites whose draws are exactly constant across all chains/draws, e.g. the beta_ceiling effective_ceiling deterministic, which legitimately yields NaN), and use max/min with a NaN-propagating reduction or an explicit isnan check before the pass/fail decision.

#### src/panelcast/pipelines/orchestrator.py:1278 — _find_run_with_product picks 'most recent' run by lexicographic dir name, so select-arm runs (sel_*) permanently shadow newer full runs

Candidates are `sorted(self.output_base.iterdir(), reverse=True)` — name order, not creation order. Caller-supplied run ids from the select runner are `sel_<sweep>_...` and 's' > '9' in ASCII, so every sel_* dir sorts before every timestamp id (verified: sorted(['sel_x','2026-07-09_...'], reverse=True) puts sel_x first). Arm runs execute train+evaluate with success=True into the same outputs/ base, so after any sweep, a consumer-only invocation (`panelcast run --stages evaluate`, `--stages report`, `panelcast stage predict`) resolves its models/evaluation/predictions read roots to a stale sweep arm — a run fitted with deliberately non-default knobs — instead of yesterday's full publication run, and generates reports/predictions from the wrong model with only a log.info line as evidence.

**Fix:** Order candidates by recency rather than name: load each candidate's manifest and pick max by manifest.created_at (or sort run dirs by st_mtime descending) in _find_run_with_product.

#### src/panelcast/pipelines/orchestrator.py:759 — Resume does not restore enable_genre/enable_artist/enable_temporal: a resumed ablation run silently rebuilds features with the wrong flags

The feature-ablation flags are recorded in manifest flags and consumed by the features stage (build_features.py:221-286 reads ctx.enable_*), but they are absent from RESUME_CONFIG_KEYS — and since the restore loop only warns for keys IN the tuple, there is no warning either. Run `panelcast run --no-genre`; it fails at the data or splits stage (bad raw path, disk full); fix and run `panelcast run --resume <id>`. config.enable_genre reverts to the CLI default True, the features stage builds the genre block, train fits with genre covariates — while the manifest (and runs show/diff) still records enable_genre=false. The delivered model contradicts its own provenance record.

**Fix:** Add "enable_genre", "enable_artist", "enable_temporal" to RESUME_CONFIG_KEYS (and audit the tuple against the flags dict for other output-affecting escapes — these three are the clear ones).

#### src/panelcast/cli/runs_cmd.py:405 — runs reproduce of any run recorded with a caller-supplied run_id (every select arm) always crashes on run-dir collision

PIPELINE_YAML_MAPPING includes run_id, so dump_resolved_config writes `run_id: sel_...` into the arm run's resolved_config.yaml (verified: dump of PipelineConfig(run_id='sel_test_arm') contains run_id). runs_reproduce clears config.resume and config.skip_existing but not config.run_id, so _setup_run does mkdir(outputs/sel_..., exist_ok=False) against the very directory the manifest was just loaded from → FileExistsError → PipelineError('run_id already exists') — which additionally escapes run() as a raw traceback (see the setup-exception finding). `panelcast runs reproduce <any sweep arm>` can never succeed.

**Fix:** In runs_reproduce, add `config.run_id = None` beside the resume/skip_existing resets (a reproduction must mint its own run dir); alternatively exclude run_id from dump_resolved_config since it is an identity, not a knob.

#### src/panelcast/pipelines/orchestrator.py:812 — Resume does not restore the 'stages' selection: resuming a --stages run silently expands to the full pipeline

flags records "stages" but RESUME_CONFIG_KEYS omits it, so `panelcast run --resume <id>` runs with stages=None. Original run `panelcast run --stages evaluate,report` (consumer-only, models redirected from a previous run) fails at report; plain resume rebuilds the full stage list — data/splits/features re-run (never in stages_completed) and train REFITS THE MODEL from scratch for hours in a run that was never supposed to train, then overwrites the shared data roots and stamps. The resume command printed by the failure epilogue itself (`panelcast run --resume <id>`) triggers this.

**Fix:** Restore "stages" from manifest flags in _restore_config_from_manifest like the other output-affecting keys (add to RESUME_CONFIG_KEYS), letting an explicitly passed --stages on the resume command line win if that override is desired.

#### src/panelcast/features/history.py:121 — History/temporal feature blocks sort NaT dates as latest, contradicting the split and AR-model contract that undated events are earliest-in-history

within_entity_temporal_split sorts with na_position="first" and logs the lineage assumption "NaT events sorted earliest-in-history"; prepare_model_data's prev_score/album_seq chain inherits that order from the split parquet. But EntityHistoryBlock.transform (history.py:121) and TemporalBlock.transform (temporal.py:119) sort with the default na_position="last". Verified repro (artist with albums undated/50, d1/60, d2/70 in train, d3 in test): (a) the undated TRAIN row gets user_prior_mean=65 and prior_count=2 computed from d1 and d2 — scores of chronologically LATER albums, violating the block's own leave-one-out "each event sees only prior events" invariant; (b) is_debut lands on d1 instead of the undated row; (c) the held-out test row's features silently exclude the undated train event (prior_count=2, album_sequence=3 instead of 3/4); (d) the SAME undated row is simultaneously treated as the artist's FIRST album by the model's AR chain (prev_score=global-mean fill, album_seq=1, confirmed via prepare_model_data) and as the LAST album by the feature blocks — two contradictory chronologies feed one training row. Reachable whenever tier-3 rows (no parseable date AND no year — parse_release_dates keeps NaT, and filter_for_target_model does not filter on dates) survive into the modeling dataset; no held-out label leakage (targets are masked in _transform_with_train_history).

**Fix:** In EntityHistoryBlock.transform and TemporalBlock.transform, sort with na_position="first" to match the split/model contract (df.sort_values([entity, date, event], na_position="first")); update the feature golden-hash pins if any tier-3 rows exist in the reference data.

#### src/panelcast/pipelines/sensitivity.py:1273 — sensitivity_results.json loses variant names: records-orient drops the 'name' index

Run the sensitivity stage with the default axes. `aggregate_sensitivity_results()` builds rows with a 'name' key, then does `df.set_index("name")` and sorts by ELPD descending. `run_sensitivity_suite` serializes it via `.to_dict(orient="records")` (lines 1273 and 1288), which drops the index entirely — verified in the repo's pixi env: `pd.DataFrame([...]).set_index('name').to_dict(orient='records')` returns `[{'elpd': -10.0}, {'elpd': -12.0}]` with no names. The published reports/sensitivity/sensitivity_results.json 'priors' and 'ablation' blocks are anonymous rows re-sorted by ELPD, so default/diffuse/informative and full/no_genre/no_temporal results cannot be attributed to any variant. Tests only assert the keys exist, not the content.

**Fix:** Serialize with the index preserved: `aggregate_sensitivity_results(prior_results).reset_index().to_dict(orient="records")` at both call sites (lines 1273 and 1288-1290).

#### src/panelcast/select/runner.py:899 — Resume has no protocol-identity guard: same sweep_id silently reuses records across datasets and sampler scales

Arm identity is the knob-dict hash only — neither cfg.dataset nor num_chains/num_samples/num_warmup/rungs enter arm_id(), and run_sweep loads whatever ledger.json exists under sweep_dir with no comparison against the stored sweep_config.json (which it unconditionally overwrites, destroying the evidence). The CLI defaults --sweep-id to "sweep", so `panelcast select --dataset A` followed by `panelcast select --dataset B` (or the same dataset at a different --effort) reuses the collision by default. Reproduced in the repo env: a second run_sweep with dataset="examples/aerospace/descriptor.yaml" and 4x5000 sampler against a ledger produced under AOTY defaults at 2x500 launched 0 fits and reported 29 "completed" arms — dataset B's report.md would be built entirely from dataset-A runs at the wrong scale, silently. Note the asymmetry: confirmation.py::_reusable_prior_seeds already implements exactly this identity check (archive-on-mismatch) for confirmation.json; the ledger has none.

**Fix:** In run_sweep, before constructing SweepLedger, read the existing sweep_config.json and compare a protocol identity (dataset, num_chains/num_samples/num_warmup, rungs, extra_config) against cfg; on mismatch raise (or archive the old ledger and start fresh, the _reusable_prior_seeds pattern) instead of resuming. Write sweep_config.json only after the check.

#### src/panelcast/select/stacking.py:212 — load_stack_arms ignores the ledger's rung field: ladder sweeps stack screening-scale duplicates of every promoted arm

A rung-ladder sweep (#164) persists a completed record per rung for each promoted arm and for the reference (keys aid and aid@rN, same arm_id, distinct run_dirs). load_stack_arms admits every completed entry with a snapshot and never reads entry["rung"]; the obs-dimension check cannot exclude them because all rungs share identical test rows. Reproduced in the repo env: a 2-rung ledger with reference+1 arm yields 4 admitted StackArms — two rows labeled "reference" in the weights table, weight mass split arbitrarily between the 2x500 screening fit and the final-rung fit of the same model, screening-scale draws mixed into the honest-split mixture, and the total_elpd "champion" row possibly a screening fit. The whole select subsystem elsewhere treats screening-rung evidence as "not comparable" to the final rung (orchestrate.py appendix), but `panelcast stack` on a ladder sweep_dir mixes them into the shipped forecast product. Both features ship together in 0.12.

**Fix:** In load_stack_arms, filter entries to the final rung: read rungs from the sweep dir's sweep_config.json (final = len(rungs)-1, default 0), or minimally keep only entries with rung == max(entry.get("rung", 0) for completed entries), and record dropped screening records in `excluded` with a "screening rung" reason.

#### src/panelcast/select/runner.py:983 — Admission queue-wait is billed as GPU hours against the sweep budget

`started = time.monotonic()` (line 983) is captured before `admission.admit(admitted_gb)` (line 988), which spin-waits for VRAM, and `record.wall_clock_seconds = time.monotonic() - started` (line 998) feeds `SweepLedger.hours_spent()` (runner.py:469-471), which is exactly what `_budget_skip_reason` charges against `budget_hours`. With `--parallel-arms 4` and one large arm holding VRAM for 2h, three blocked arms each accumulate ~2h of pure queue time; when they finish, hours_spent has jumped ~8h for ~2h of real GPU work, and the remaining arms are recorded `skipped_budget` long before the consented GPU-hour budget is actually spent.

**Fix:** Move `started = time.monotonic()` to after the `admission.admit()` call (immediately before `launch`), or subtract the measured admit-wait from `record.wall_clock_seconds`.

#### src/panelcast/select/runner.py:990 — Auto arm timeout sized from serial history is enforced on concurrent launches, and timeout is terminal

`resolve_arm_timeout` sizes auto timeouts as `max(floor, 3.0 * predicted)` where the prediction deliberately excludes concurrent records (runtime_predictor.py:110-112, serial rates only), yet `_execute` passes that same timeout to concurrently launched arms (line 990) that contend for SMs. With parallel_arms=4 and arms that individually saturate the GPU, each arm can run >3x its serial prediction, so all four are killed as `timeout` — a status that is terminal: skipped on resume (runner.py:955), never retried by the OOM kill-and-serialize loop (line 1097 only retries status=='failed'), and unscored so it can never promote. Healthy candidate arms are silently and permanently lost from the sweep.

**Fix:** When launching concurrently, scale the auto timeout by the contention bound (e.g. `timeout_seconds * cfg.parallel_arms`), or make concurrency-killed timeouts retryable-serial like OOM failures instead of terminal.

#### src/panelcast/utils/jax_cache.py:53 — doctor --json and runs history --json stdout is polluted by a structlog debug line from the CLI callback

main_callback (cli/main.py:49) calls enable_jax_compilation_cache() before every subcommand; with structlog unconfigured (any non-pipeline command) its log.debug prints to stdout. Verified end-to-end: `panelcast doctor --json 2>/dev/null` emits `2026-07-09 ... [debug] jax_compilation_cache_enabled cache_dir=...` before the JSON array, so `panelcast doctor --json | jq .` (the documented machine-readable CI mode) fails to parse. Same for `runs history --json`. The doctor command's own contextlib.redirect_stdout guard runs too late to catch it.

**Fix:** In enable_jax_compilation_cache, log via stdlib logging (logging.getLogger(__name__).debug — silent when logging is unconfigured, still captured once setup_pipeline_logging runs) instead of structlog's default stdout PrintLogger; alternatively bind structlog's default output to stderr in main_callback before enabling the cache.

#### src/panelcast/cli/commands.py:122 — export-figures crashes with KeyError 'mean' when the coefficients fallback CSV lacks estimate/HDI columns

load_dashboard_data falls back to *summary*.csv (e.g. metrics_summary.csv with columns Metric/Value) when no *coefficient*.csv exists — a state produced whenever the coefficient-table build failed (publication.py catches that failure and continues) but metrics_summary succeeded. _coefficient_columns correctly returns None for that frame, but the else branch then calls create_forest_plot(data.coefficients) with default columns mean/hdi_3%/hdi_97%. Verified: create_forest_plot on a Metric/Value frame raises KeyError 'mean' (charts.py:399), which is uncaught, so `panelcast export-figures` dies with a traceback and exports none of the other available figures.

**Fix:** Drop the else branch: when _coefficient_columns returns None, skip the coefficients figure (optionally typer.echo a warning) instead of calling create_forest_plot with defaults that the resolver just proved absent.

#### src/panelcast/reporting/model_card.py:460 — MODEL_CARD.tex hyperparameter values are not LaTeX-escaped, producing an uncompilable .tex on the default configuration

_generate_latex escapes the parameter name but interpolates the value raw: f"{_latex_escape(param)} & {value}". Every default run records target_transform='offset_logit' in the training summary (train_bayes.py:1451), which publication.py copies into hyperparameters, so the generated tabular contains `target\_transform & offset_logit \\` (verified by generating the card). The bare underscore in text mode makes MODEL_CARD.tex fail to compile (Missing $ inserted) for every run since 0.5.0's offset_logit default; any other value with _ / % / & (entity_group_pooling modes, likelihood names) does the same.

**Fix:** Escape values too: f"{_latex_escape(param)} & {_latex_escape(str(value))} \\\\" in the hyperparameters loop.

#### src/panelcast/select/stacking.py:206 — panelcast stack ignores the ledger's rung field: on ladder sweeps it stacks screening-scale fits and duplicates promoted arms

The shipped configs/select.yaml gives the default `standard` (and `thorough`) tier a rung ladder, so a v2 ledger holds one completed record per (arm, rung): the reference and every promoted arm appear at both rung 0 (2x500 screening) and the final rung (4x1000), each with its own run_dir and log_likelihood.nc (PANELCAST_SAVE_LOG_LIKELIHOOD=1 is set for every arm). load_stack_arms reads only arm_id/status/run_dir and never filters by rung, so after `panelcast select --effort standard` + `panelcast stack outputs/select/sweep`: stacking weights are fit over duplicate copies of each promoted arm plus low-fidelity screening fits, the weights table lists "reference" twice, `champion = max(total_elpd)` can pick a screening-scale run, and the honest secondary-split headline mixes 2x500 predictive draws — directly violating the #164 contract that screening-rung evidence never feeds report-grade output (run_select filters to final_rung; stack does not).

**Fix:** In load_stack_arms, restrict entries to the final rung (e.g. `final = max(int(e.get("rung", 0)) for e in entries)` or read rungs from sweep_config.json) and exclude records with `entry.get("rung", 0) != final`, mirroring orchestrate.run_select.

### YELLOW (27)

#### src/panelcast/models/bayes/likelihoods.py:698 — Trained betabinom_max_n_reviews cap never reaches prediction or rollout

_beta_binomial_predict_draws accepts max_n_reviews=DEFAULT_BETABINOM_MAX_N (100), but neither caller (predict.py:591 predict_new_entity nor rollout.py:192) ever forwards the trained priors.betabinom_max_n_reviews. Train a beta_binomial model with betabinom_max_n_reviews=1000 (a documented PriorConfig knob whose comment says to raise it when span*n stays smooth) on events with ~800 raters: inference uses n_eff=800 while every cold-start predictive and multi-step rollout draw caps n_eff at 100, so the predictive Beta-Binomial has inflated binomial spread (sd ratio sqrt(((phi+100)/100)/((phi+800)/800)), ~8% wider at phi=20) and a 8x coarser score grid than the fitted likelihood. The priors.py comment claims the constant is 'shared so inference and the predict path agree' — they only agree at the default.

**Fix:** Thread the trained cap through: include betabinom_max_n_reviews in the summary-derived kwargs and pass max_n_reviews=... from predict.py/rollout.py (or carry it in the sites dict like the other family parameters).

#### src/panelcast/models/bayes/model.py:840 — sigma_obs_prior_type='lognormal' silently ignored whenever sigma-ref mode is active

train_bayes.py:1119 always passes n_ref=median(n_reviews) when learn_n_exponent or n_exponent != 0, so use_sigma_ref is True for every pipeline heteroscedastic fit; the model then samples sigma_ref ~ HalfNormal(priors.sigma_ref_scale) unconditionally and derives sigma_obs deterministically, never consulting priors.sigma_obs_prior_type. A user running `--sigma-obs-prior-type lognormal --n-exponent 0.5` (both public flags; lognormal is the documented fix for sigma_obs zero-boundary collapse on zero-inflated targets) silently keeps the HalfNormal zero pile-up they configured away — no error, no warning. Verified: no validation in orchestrator.py (only value-string check at line 324) and no test covers the combination.

**Fix:** Route sigma_ref through the same prior-type seam (add a lognormal branch mirroring _sample_sigma_obs, reusing sigma_obs_prior_type or a new sigma_ref_prior_type), or raise/warn in config validation when sigma_obs_prior_type != 'halfnormal' combines with heteroscedastic mode.

#### src/panelcast/models/bayes/model.py:266 — Regularized-horseshoe lambda_tilde overflows to inf/NaN in float32 at large local scales

lambda_tilde = sqrt(c2*lambda^2 / (c2 + tau^2*lambda^2)) squares the HalfCauchy local scales. The repo never enables jax x64 (grep confirms), so with beta_prior_type='horseshoe' the numerator c2*lambda^2 overflows float32: verified numerically lambda_tilde(1e19)=inf and lambda_tilde(2e19)=nan (then beta and mu become inf/NaN). NUTS samples unconstrained log-lambda, and the HalfCauchy log-density at log-lambda≈44 is only ~-43 nats below typical — reachable during warmup step-size exploration, producing NaN-gradient divergent transitions and degraded adaptation rather than a clean geometry.

**Fix:** Use the algebraically equivalent, uniformly stable form lambda_tilde = jnp.sqrt(beta_c2 / (beta_c2 / beta_lambda**2 + beta_tau**2)) — correct limits at both lambda->0 (underflow gives inf denominator -> 0) and lambda->inf (-> c/tau).

#### src/panelcast/models/bayes/priors.py:112 — PriorConfig.likelihood_df is dead — the model only reads the likelihood_df argument

The field is documented ('Likelihood degrees of freedom (Student-t; inf => Normal)') but grep over src/ shows nothing ever reads priors.likelihood_df: _sample_likelihood and every family sample_obs use the separate likelihood_df model argument (default 4.0), and the pipeline plumbs df from PipelineConfig, not PriorConfig. A direct-API user who builds PriorConfig(likelihood_df=30) and runs make_score_model without also passing the model kwarg silently fits Student-t(4) — wrong tails with no error.

**Fix:** Remove the field (and its docstring promise), or make the model fall back to priors.likelihood_df when the likelihood_df argument is left at its sentinel/default.

#### src/panelcast/models/bayes/fit.py:425 — Checkpoint state.pkl and cursor.json are written non-atomically, destroying the only recovery copy on a crash mid-write

_run_blocked writes state via state_path.open('wb') and the cursor via cursor_path.write_text — both truncate the previous good file before writing. The feature exists to survive arbitrary crashes (power loss, OOM-kill during GPU fits); a kill inside either write window leaves a truncated state.pkl or cursor.json. On the next resume, pickle.load raises EOFError/UnpicklingError (or json.loads raises JSONDecodeError) unhandled, and since the last-known-good state was overwritten, all completed blocks become uncontinuable — the user must delete the whole checkpoint and restart, which is exactly the loss checkpointing was built to prevent. io.py's save_manifest already does tmp+replace for the same reason.

**Fix:** Write both files to a temp name and os.replace() into place (mirroring save_manifest's tmp_path.replace pattern), and wrap the resume-side pickle/json loads in a clear error suggesting which artifact is corrupt.

#### src/panelcast/pipelines/orchestrator.py:539 — learn_n_exponent conflict-clear runs after _setup_run already wrote the manifest and resolved_config.yaml, defeating its own stated purpose

run() calls _setup_run() (which snapshots flags{n_exponent: 0.5} into manifest.json and dumps resolved_config.yaml) BEFORE the conflict check that sets self.config.n_exponent = 0.0 — despite the comment 'Clear the fixed exponent to prevent manifest recording stale value'. `panelcast run --n-exponent 0.5 --learn-n-exponent` records n_exponent=0.5 in both provenance files while the fit actually used 0.0; runs show/diff misreport the run and a later --skip-existing run with plain --learn-n-exponent sees a flag change and needlessly disables skipping.

**Fix:** Move the learn_n_exponent/n_exponent conflict check (and clear) to before self._setup_run() in run(), or into PipelineConfig.__post_init__ so every entry path normalizes it.

#### src/panelcast/pipelines/orchestrator.py:1146 — Skipped stages don't carry their stage_hash into the new manifest, so --skip-existing only works against a run that actually executed the stage

When should_skip fires, only stages_skipped is appended — stage_hashes[stage.name] stays absent. Run 1 executes data; run 2 --skip-existing skips it (hash from run 1); run 3 --skip-existing loads run 2's manifest via latest.json, finds no stage_hash for data, and re-executes the data/splits/features stages with unchanged inputs — skips alternate on/off forever, and each needless re-execution rewrites the shared parquets and stamps (which will StaleArtifactError any concurrently running consumer of those roots).

**Fix:** On the skip path in _execute_stages, also copy previous_manifest.stage_hashes[stage.name] into self.manifest.stage_hashes so skip decisions chain across consecutive runs.

#### src/panelcast/pipelines/orchestrator.py:523 — Setup-phase failures escape run()'s exit-code contract as raw tracebacks (no epilogue, no failure.json)

run() wraps _verify_environment and _execute_stages but calls self._setup_run() bare, and PipelineOrchestrator.__init__ can raise ValueError (min_ratings not in descriptor thresholds, beta_binomial mismatch) with no CLI guard around run_pipeline. `panelcast run --resume 2026-01-19_typo` (PipelineError: cannot find run), a resume with descriptor drift, a caller-supplied run_id collision, or `panelcast run --min-ratings 7` all surface as an unhandled typer traceback instead of the failure epilogue / exit-code path, and programmatic callers of run_pipeline get an exception where the API documents an int return.

**Fix:** Wrap _setup_run (and steps 3-4) in try/except PipelineError returning e.exit_code with a log line, and in cli/run.py catch ValueError from run_pipeline's orchestrator construction like the existing PipelineConfig ValueError→BadParameter handling.

#### src/panelcast/pipelines/orchestrator.py:371 — _validate_run_id accepts the reserved names 'latest' and 'failed'

YAML `run_id: latest` creates outputs/latest as a real run dir: _create_latest_link then tries to remove the run's own directory (os.rmdir on a non-empty dir fails, link creation aborts) and outputs/latest permanently shadows the pointer fallback while _find_run_with_product/_setup skip any dir named 'latest', making the run's products invisible. `run_id: failed` is worse: on a pipeline failure _handle_failure computes failed_path = outputs/failed/failed and shutil.move(outputs/failed, outputs/failed/failed) attempts to move the directory into itself.

**Fix:** In _validate_run_id, additionally reject run_id in ("latest", "failed") (and names starting with '.') with the same ValueError.

#### src/panelcast/pipelines/orchestrator.py:346 — _validate_sampling never checks num_warmup or target_accept, which the YAML path can set to invalid values the CLI bounds would reject

CLI enforces num_warmup>=50 and 0.5<=target_accept<=0.999, but YAML values pass through _passthrough with no bounds and _validate_sampling checks num_chains/num_samples/ess only. A run_config.yaml with `num_warmup: 0` (typo for 1000) silently fits with zero adaptation — garbage step size, unconverged posterior that may still pass loose thresholds — and `target_accept: 1.2` or a negative num_warmup only detonates inside NumPyro at the train stage after data/splits/features have already run.

**Fix:** Add to _validate_sampling: `num_warmup >= 1` (or >=0 if warm-started runs legitimately skip warmup) and `0.0 < target_accept < 1.0`, mirroring the other YAML-reachable knobs already validated there.

#### src/panelcast/data/cleaning.py:140 — Tier-2 date fill crashes the whole data stage with OutOfBoundsDatetime on a single out-of-range Year value

parse_release_dates builds the jan1 fill with pd.to_datetime(year.astype(int).astype(str) + "-01-01") and no errors="coerce". One row with an unparseable date and Year outside pandas' ns-datetime bounds (e.g. a 1500 typo, or year > 2262) raises OutOfBoundsDatetime and aborts prepare_datasets with a raw traceback — no exclusion, no audit record. Verified: parse_release_dates(pd.DataFrame({'Release_Date':[None],'Year':[1500.0]})) raises OutOfBoundsDatetime. The raw-schema YEAR_RANGE(1900,2030) check that would catch it is off on the default path: PrepareConfig.validate_raw_schema defaults to False and the orchestrator enables it only under --strict; the cleaned-schema check runs after parse_release_dates, too late. This is the module whose whole purpose is classifying messy dates, and the tool is marketed as domain-portable to arbitrary raw CSVs.

**Fix:** Use pd.to_datetime(..., errors="coerce") for the tier-2 fill and demote rows whose fill fails to tier 3 (date_risk='high', date_imputation_type='unimputed', date_missing=True), so wild year values become audited unimputed rows instead of a stage crash.

#### src/panelcast/pipelines/sensitivity.py:1079 — run_split_seed_sensitivity scores coverage under the wrong observation model for non-default fits

The split-seed axis calls `predict_new_entity` passing only transform/df/ar_center kwargs. It omits `likelihood_family` (config-exposed via pipeline_yaml), `discretize_observation`, and `fixed_n_exponent`. For a model trained with e.g. `likelihood_family: skewt`, `discretize_observation: true`, or a fixed non-zero `n_exponent`, predict_new_entity falls back to plain Student-t and (per predict.py:441-446, `has_fixed_exponent` False) homoscedastic sigma — so the published seed-sensitivity coverage row is computed under a different predictive distribution than the metrics.json cold-start evaluation it is meant to stress-test, silently (no error, just a shifted coverage number). The production path `_run_new_artist_predictive` in evaluate.py passes all three.

**Fix:** Mirror `_run_new_artist_predictive`: pass `likelihood_family=summary.get("likelihood_family") or "studentt"`, `discretize_observation=bool(summary.get("discretize_observation", False))`, and `fixed_n_exponent` when `not summary.get("learn_n_exponent")` and the recorded `n_exponent` is non-zero, into the `predict_new_entity` call at sensitivity.py:1079-1091.

#### src/panelcast/select/confirmation.py:74 — ConfirmationResult.confirmed compares against seeds-so-far, so the per-seed checkpoint persists confirmed:true after 1 of 3 seeds

The property gates on `len(measured) < len(self.seeds)` where self.seeds is the list appended during the loop, not seeds_planned. run_confirmation checkpoints result.to_dict() to confirmation.json after every seed, so after the first passing seed the on-disk file says "confirmed": true with seeds_planned [42,43,44] and only one seed present. Reproduced in the repo env (to_dict()["confirmed"] is True with 1/3 seeds measured). Confirmation runs are hours of publication-scale fits; a crash/Ctrl-C after seed 1 leaves a false CONFIRMED verdict as the persisted .audit-grade evidence until someone resumes. The live return path is unaffected (the loop always appends all seeds before returning).

**Fix:** Gate on the plan: `planned = self.seeds_planned or [s.seed for s in self.seeds]` and return False when fewer measured seeds than len(planned), i.e. replace `len(measured) < len(self.seeds)` with `len(measured) < len(planned)`.

#### src/panelcast/select/orchestrate.py:317 — run_select's final scoring loop is unguarded: one bad snapshot crashes the report after the whole sweep has been paid for

score_arm -> paired_elpd raises ValueError when an arm's log_likelihood.nc has a different obs count than the reference (and pointwise_elpd raises OSError on a corrupt netCDF). The runner's in-sweep scorer wraps scoring in try/except ("scoring must never kill the sweep", runner.py:1034) but run_select's post-sweep loop (lines 313-323) appends score_arm() results with no guard. Concrete path: source data grows between resume sessions (or the cross-dataset resume collision above) so an old completed record's snapshot has fewer test rows than the fresh reference — run_select then raises after all GPU hours are spent, before report.md, verdicts, or confirmation are written, and every retry crashes the same way until the stale ledger entry is hand-deleted.

**Fix:** Wrap the score_arm call in try/except (OSError, ValueError), appending an unscored ArmScore with the failure in notes (the same no-substitute discipline the scorer already uses for missing snapshots).

#### src/panelcast/select/runner.py:874 — Ladder break on empty survivors still runs stage 3, burning publication-scale fits that can never be scored

If a screening rung completes its reference but every arm fails to score (e.g. all OFAT arms hit the auto-timeout at rung 0, or the reference's snapshot is missing so every z is None), rung_survivors returns [] and _run_stage1_ladder breaks — returning True. run_sweep then proceeds: stage 2 is a harmless no-op, but stage 3 (thorough tier: 8 arms) executes at final_rung sampler scale while no final-rung reference exists (reference_runs lacks final_rung, and no final-rung reference record is ever fit), so every stage-3 fit completes unscoreable — exactly the waste the dead-reference guard at line 855-858 exists to prevent ("stage 3 would burn unscoreable fits — stop instead").

**Fix:** Treat the empty-survivors break like the dead-reference case: `return False` (or set a flag that suppresses stage 3) instead of `break` when rung_idx < final_rung and survivors is empty.

#### src/panelcast/select/scoring.py:295 — Report verdict can name the reference's own self-pair as "the closest" challenger

run_select scores every completed final-rung record including the reference, whose self-pair yields z=0.0 (scored, not None). Whenever no challenger has positive z — the most common sweep outcome ("no arm beats shipped" was 0.7.0's own result) — rank_arms puts the reference first among scored arms and _verdict renders: "No arm beats shipped defaults (reference arm); the closest is <reference's hash id> at +0.0 +/- 0.0 held-out ELPD (z +0.00)", presenting the reference to the report reader as its own nearest challenger under an anonymous arm id.

**Fix:** In _verdict, exclude the reference from the challenger pick: `scored = [s for s in ranked if s.elpd_z is not None and s.knobs]` (ArmScore.knobs is {} exactly for the reference), falling back to the current text when nothing remains.

#### src/panelcast/gpu_memory/calibration_store.py:208 — Never-under envelope loop gives up after 5 iterations and resolve_calibration adopts an under-covering calibration

The envelope inflation loop (`for _ in range(5)`) scales only the two constants, so when a record's peak is dominated by the raw base term (big X, tiny collection unit — e.g. a wide-feature fit at 1 sample) the multiplicative passes converge too slowly to reach _MIN_LOCAL_ENVELOPE. Reproduced against the real module: 9 slope-consistent records plus one record (n_obs=800k, n_features=6700, 1 sample, actual 40 GB) yield PerMachineCalibration(min_ratio=0.980) — the calibrated estimator UNDER-prices a fit shape already measured on this machine — and `resolve_calibration` (line 229) adopts it unconditionally, breaking the never-under contract the module documents.

**Fix:** After the loop, if `min_ratio < _MIN_LOCAL_ENVELOPE` return None (fall back to shipped constants), or compute the exact required scale in closed form: s = max_i((target*actual_i/(1+jbp) - base_i)/(fixed + factor*unit_i)).

#### src/panelcast/pipelines/train_bayes.py:1147 — chain_method 'auto' resolution prices the fit without the model-structure gates

The estimate_inputs dict passed to `_resolve_chain_method` (lines 1147-1158) omits errors_in_variables, heteroscedastic_entity_obs, and entity_group_pooling/n_groups, although estimate_memory_gb models all of them. Measured with the estimator itself at representative dims (24k obs, 12k artists, max_seq 30, 4x1000, exclusion on): auto prices 0.76 GB while the entity-obs-gated fit is 1.43 GB (~2x under); with exclusion off + EIV the gap is 1.34 GB. On a device where vectorized is borderline under the 0.8 headroom, `--chain-method auto --heteroscedastic-entity-obs` (or errors_in_variables) flips to vectorized when the true footprint does not fit, and the fit OOMs hours in.

**Fix:** Include the three gate flags (and n_groups from model_args when pooling is on) in the estimate_inputs dict built at line 1147, mirroring what _build_resource_usage records.

#### src/panelcast/gpu_memory/calibration_store.py:180 — Memory refit mixes vectorized and sequential records under a sequential-only linear model

`refit_constants` consumes every record with an actual peak, with no filter on context.chain_method (the runtime predictor filters exactly this at runtime_predictor.py:109), and `_linear_terms` prices the base term with live_chains=1. Records from vectorized fits (reachable via auto resolution or --chain-method vectorized; context records the method but estimate_inputs do not) carry peaks the sequential model cannot explain, and auto selects vectorized precisely for SMALL fits — so elevated-y points cluster at small units, rotating the least-squares line: intercept up, slope down. The 1.05 envelope only covers observed points, so a larger sequential production fit extrapolated with the depressed slope is under-priced — a never-under break the store cannot detect.

**Fix:** Skip records whose context.chain_method != 'sequential' in the refit loop (mirroring the runtime predictor), and record chain_method inside estimate_inputs so future models can key on it.

#### src/panelcast/select/runner.py:797 — _admission_env prices every arm gates-off, under-reserving and under-capping the EIV/entity-obs OFAT arms

`estimate_with_calibration` at line 797 receives only the 7 dimension/sampler args — never the arm's knobs — yet stage 1 always contains arms with errors_in_variables=True and heteroscedastic_entity_obs=True (they are KNOBS in select/space.py). For those arms the reservation and the XLA_PYTHON_CLIENT_MEM_FRACTION pool cap (line 813) are computed for the smaller gates-off model (0.3-1.3+ GB short at tier scale), so a concurrently-launched gated arm can exhaust its capped pool and OOM even though the estimator would have priced it correctly — costing a wasted fit plus a kill-and-serialize retry. Additionally, when NVML is absent (line 811), the fallback pool cap is headroom/parallel_arms while admission simultaneously degrades to one-at-a-time, so a solo arm runs with a 1/N-sized pool.

**Fix:** Pass the merged arm's gate flags (errors_in_variables, heteroscedastic_entity_obs, entity_group_pooling) into estimate_with_calibration in _admission_env (it needs the `merged` dict as a parameter), and use the full headroom rather than equal_share when admission is serializing anyway.

#### src/panelcast/reporting/tables.py:340 — Diagnostics table marks every parameter 'Fail (R-hat)' on single-chain runs, contradicting the pipeline's convergence gate

With one chain az.summary yields r_hat=NaN; get_status computes `nan <= rhat_threshold` -> False, so Status is 'Fail (R-hat)' for every row (verified empirically). check_convergence and the model card treat single-chain R-hat as 'unavailable (requires >=2 chains)' and do not fail on it, so the published diagnostics.csv/.tex contradicts the run's own readiness verdict. Reachable on the default `panelcast demo` (num_chains=1) and any --num-chains 1 exploratory run.

**Fix:** In get_status, treat non-finite r_hat as not-failing (report 'n/a (single chain)' or gate only on ESS), mirroring check_convergence's single-chain semantics.

#### src/panelcast/reporting/figures.py:1114 — Artist fan charts hardcode a 0-100 y-axis and 'User Score' label, mis-scaling every non-AOTY domain

save_artist_prediction_plot always does ax.set_ylim(0, 100) and ax.set_ylabel('User Score'). _save_artist_fan_charts (publication.py:911) calls it for any descriptor; the bundled aerospace demo has target_bounds [0, 10], so `panelcast demo` publishes fan charts with all data squashed into the bottom 10% of an axis labeled 'User Score' — a wrong-looking publication figure on the advertised first-touch path. Same for the x-label 'Album'.

**Fix:** Add target_bounds and axis-label parameters (the caller already holds fan_descriptor with target_bounds/target_col/event_col) and use them for set_ylim/labels, defaulting to the AOTY values.

#### src/panelcast/cli/commands.py:508 — runs list resolves the latest marker from the opportunistic 'latest' link instead of the authoritative latest.json

The orchestrator writes latest.json as the authoritative pointer and creates the outputs/latest link only opportunistically (orchestrator.py:1658 comment; creation can fail on Windows/NTFS, and if removal of a stale link fails it returns leaving the OLD link in place). runs_list marks the latest run solely via (base/'latest').resolve(), so the * marker can be missing (link never created) or point at a stale run while every other consumer (resolve_latest) follows latest.json to a newer one — the listing then flags the wrong run as latest.

**Fix:** Use panelcast.paths.resolve_latest(base) to determine latest_target in runs_list (fall back to the link only if that returns None).

#### src/panelcast/utils/jax_cache.py:44 — Unguarded cache_dir.mkdir in the CLI callback crashes every command — including doctor — when the cache dir is unwritable

main_callback unconditionally calls enable_jax_compilation_cache(), whose cache_dir.mkdir(parents=True, exist_ok=True) raises PermissionError/OSError when ~/.cache (or PANELCAST_JAX_CACHE_DIR) is unwritable or a file. Every subcommand then dies with a traceback before doing anything — including `panelcast doctor`, whose compile-cache check (doctor.py:_compile_cache) exists precisely to diagnose this state but can never run. Workaround (PANELCAST_JAX_CACHE=0) is only discoverable from the docs.

**Fix:** Wrap the mkdir/config in try/except OSError inside enable_jax_compilation_cache, log a warning, and return None so the CLI proceeds without the cache.

#### src/panelcast/cli/select_cmd.py:188 — select hardcodes the canonical split path, silently skipping the prior screen for legacy-named split artifacts

_SPLIT_PATH is fixed to data/splits/within_entity_temporal/train.parquet, while the rest of the code reads splits through resolve_split_dir which falls back to the pre-rename 'within_artist_temporal' directory. On a workspace with pre-rename artifacts, _load_prepared_frame and _prepared_paths report 'not prepared': the prior-predictive screen and data diagnostics are silently skipped and the n_artists dim hint (cost projection, auto timeouts) degrades to the coarse n_obs//5 estimate, even though the splits exist and load fine elsewhere.

**Fix:** Derive the split path via resolve_split_dir(Path('data/splits'), SplitType.WITHIN_ENTITY_TEMPORAL) / 'train.parquet' (compute at call time, not import time).

#### src/panelcast/pipelines/orchestrator.py:833 — Resuming a --stages-scoped run silently expands it to the full pipeline

"stages" is recorded in manifest flags but is not in RESUME_CONFIG_KEYS, and the failure epilogue tells users to `panelcast run --resume <id>` after any failure — including failures of scoped invocations like `panelcast stage train` or a select arm (stages=[train,evaluate]). The resume restores the output-affecting knobs but leaves config.stages=None, so get_execution_order returns ALL stages: the resume re-runs data/splits/features (rewriting the shared flat caches and their stamps — poison for a select sweep still in flight) plus predict/report the user never requested, and the run dir ends with stages_completed disagreeing with both its flags.stages and its resolved_config.yaml.

**Fix:** Add "stages" to RESUME_CONFIG_KEYS (restore it from the manifest like the other flags; an explicit --stages on the resume command line could still win if desired, matching reproduce's use of the recorded scope).

#### src/panelcast/pipelines/backtest.py:163 — Backtest run attribution never verifies origin_offset (or any flag), unlike the hardened select handshake

_attributed_run_dir resolves the mutable outputs/latest.json and checks only created_at >= launched_at and prior claims; it never opens manifest flags. select/runner.py got the #167 hardening (named run_id + knob-agreement check via _attribution_error) but backtest kept the weak pointer handshake. If any other `panelcast run` completes between an origin subprocess finishing and resolve_latest() (a user kicking off a manual run while the backtest grinds), that foreign run — with the wrong origin_offset, possibly a different dataset — is claimed and _harvest_origin folds its metrics.json into backtest_metrics.json aggregates with no error, corrupting the cross-origin mean/SE table.

**Fix:** After loading the manifest in _attributed_run_dir, reject when `manifest.get("flags", {}).get("origin_offset") != origin` (pass origin in) — or adopt the select pattern and name each origin's run_id up front in _write_origin_config.

### NOTE (20)

#### src/panelcast/models/bayes/likelihoods.py:508 — beta_ceiling: mu soft-clipped to full target bounds leaves a zero-gradient plateau above the ceiling

Under the required identity transform, mu is soft-clipped into the full target bounds (e.g. (0,100)) before _beta_emit_obs rescales by the ceiling; for mu in (effective_ceiling, high) the mu01 clip at 1-eps makes the Beta log-likelihood flat in mu, so entities whose linear predictor drifts above the ceiling get no likelihood gradient pulling them back — only priors do. Sampling nuisance, not incorrectness (posterior mass just piles at the clip).

**Fix:** For the beta_ceiling family, soft-clip mu to (low, effective_ceiling) instead of the full target bounds (pass the ceiling into the transform_mu step).

#### src/panelcast/models/bayes/likelihoods.py:658 — Cold-start beta draws pin boundary eps to the default, not the trained beta_boundary_eps

_beta_core_predict_draws hardcodes eps = DEFAULT_BETA_BOUNDARY_EPS (comment acknowledges the predict path has no priors object). A model trained with a custom priors.beta_boundary_eps clips mu differently at inference vs cold-start prediction; near-boundary mu values then map to slightly different Beta means. Zero impact at the default, small drift only when the knob is overridden.

**Fix:** Carry beta_boundary_eps alongside phi in the summary/sites passed to predict_draws, mirroring how betabinom's cap should also travel.

#### src/panelcast/models/bayes/model.py:75 — compute_sigma_scaled's min_sigma=0.01 floor is absolute, not transform/scale-aware

The 0.01 floor was sized for the raw 0-100 score scale, but under offset_logit (the pipeline default since 0.5.0) typical sigma is ~0.3-0.5, so with e.g. sigma_obs=0.5 and exponent=0.5 the floor binds at n_reviews >= ~2500, silently capping the precision heteroscedastic scaling grants mega-reviewed events. Harmless at the fitted AOTY exponent (~0.002) but a hidden scale dependence for domains that actually use large exponents.

**Fix:** Expose min_sigma in PriorConfig and shrink it alongside the other sigma scales in priors_for_transform (e.g. 1e-3 on the logit scale), or express it as a fraction of sigma_obs.

#### src/panelcast/models/bayes/fit.py:453 — Requested warmup export is silently skipped when a resume finds all blocks already done

When _run_blocked resumes a fully-checkpointed fit, mcmc is None and _maybe_export_warmup returns without writing the file or logging anything at warning level. A user who passed warmup_export_path gets no export; a later --warmup-import of the missing path degrades to a cold fit via the 'unreadable warmup export' miss (safe but surprising, and the reduced num_warmup the user paired with it stays reduced).

**Fix:** Log a warning (or persist the adapt state from the pickled checkpoint state instead of mcmc.last_state) when warmup_export_path is set but mcmc is None.

#### src/panelcast/models/bayes/predict.py:484 — Dead `seed = seed + 1` with a misleading comment in predict_new_entity's subsample branch

After posterior subsampling, `seed = seed + 1  # Use different key for subsequent sampling` is never read again — all subsequent randomness correctly flows from the already-split rng_key chain. The assignment is dead and the comment implies a reseed that doesn't (and shouldn't) happen, inviting a future 'fix' that would change published draws.

**Fix:** Delete the `seed = seed + 1` line and its comment.

#### src/panelcast/models/bayes/io.py:253 — ModelManifest does not record warm_started / resumed_from_checkpoint provenance

FitResult documents that warm-started fits are 'screening-grade evidence, never confirmation' and that resumed fits' wall clocks are not calibration-grade, but save_model writes runtime_seconds and no flags — a warm-started or checkpoint-resumed model is indistinguishable from a cold single-shot fit in manifest.json, the artifact meant to make fits auditable/reproducible.

**Fix:** Add warm_started and resumed_from_checkpoint fields to ModelManifest (defaulting False in from_dict for old manifests) and populate them from fit_result in save_model.

#### src/panelcast/features/gbm_offset.py:86 — gbm_offset out-of-fold predictions use ungrouped random KFold, so train-row offsets condition on the same entity's chronologically later targets while test-row offsets cannot

GbmOffsetBlock.fit uses KFold(shuffle=True) over rows: a fold model predicting artist A's early album was trained on A's later albums' targets, so train rows' gbm_offset values embed within-entity future information; held-out rows get the full-train (past-only) model. The offset covariate therefore has systematically different information content between train and test, which can inflate the learned beta on gbm_offset and cost test calibration. The row's own target is never seen (docstring claim holds), and the feature shipped with held-out ELPD validation, so this is a recorded design smell rather than a defect. Also random_state=0 is hardcoded rather than ctx.seed (deterministic, but the block ignores the pipeline seed).

**Fix:** Use GroupKFold keyed on the entity column (or a temporal/blocked split) for the OOF pass, and plumb ctx.random_state into the block; re-validate ELPD before changing the default.

#### src/panelcast/pipelines/train_bayes.py:629 — _apply_max_albums_cap mixes post-filter artist counts with pre-filter sequence ranks, so max_seq can exceed max_albums_cap when invalid-n_reviews rows were dropped

album_seq is computed in prepare_model_data before the invalid-n_reviews valid_mask filtering and keeps its pre-drop ranks (with gaps), while artist_album_counts is recomputed from the filtered rows. An artist with 55 albums of which 5 have invalid n_reviews gets count=50, offset=0, but retains ranks up to 55 — max_seq exceeds the cap and the random-walk trajectory allocates more positions than --max-albums promises. Unreachable in the standard pipeline (prepare's min-obs filter guarantees n_reviews >= threshold > 0, so nothing is dropped), only degenerate custom data that triggers the n_reviews_invalid_rows warning path.

**Fix:** Recompute album_seq (groupby.cumcount within the filtered frame) after applying valid_mask, or derive both counts and ranks from the same filtered frame in _apply_max_albums_cap.

#### src/panelcast/pipelines/build_features.py:322 — Features manifest records feature_names from the last split iterated (entity_disjoint), not per split

feature_names is overwritten each loop iteration, so manifest.json's feature_names reflects the entity_disjoint pipeline's columns. Vocabulary-dependent blocks fit per split (AlbumTypeBlock one-hots, GenreBlock without PCA) can produce different column sets for the two splits' training data, making the manifest wrong for the primary within_entity_temporal split. No in-repo consumer reads manifest["feature_names"] (train derives columns from the parquet itself), so impact is provenance/debugging only.

**Fix:** Record feature_names per split inside split_manifests (e.g. split_manifests[split_name]["feature_names"]) or capture the within_entity_temporal list explicitly since that is the legacy primary split.

#### src/panelcast/pipelines/evaluate.py:944 — _run_new_artist_predictive omits skew_tailweight while the rollout path passes it

`_evaluate_horizon_rollout` forwards `priors_obj.skew_tailweight` to predict_horizon, but the secondary-split cold-start path `_run_new_artist_predictive` never passes `skew_tailweight` to `predict_new_entity` (defaults to 1.0). Today nothing sets skew_tailweight != 1.0 (it is not config-exposed), so this is unreachable — but the asymmetry is a trap: the moment the knob is exposed, sinh-arcsinh-family cold-start draws would get the wrong tail shape with no error.

**Fix:** Add `"skew_tailweight": PriorConfig(**summary["priors"]).skew_tailweight` to the kwargs dict in `_run_new_artist_predictive` (evaluate.py:944-957) to match the rollout call.

#### src/panelcast/select/orchestrate.py:358 — Screening-appendix sort treats a legitimate z=0.0 as unscored

The sort key `-(float((r.score or {}).get("z") or -1e9))` uses `or`, so an arm whose screening z is exactly 0.0 sorts to the bottom with the unscored arms even though its row renders "+0.00". Ordering-only cosmetic defect in the appendix table.

**Fix:** Use an explicit None check: `z = (r.score or {}).get("z"); key = -1e9 if z is None else float(z)`.

#### src/panelcast/select/orchestrate.py:47 — resolve_dims trusts the flat data/features cache with no domain check, skewing auto-timeouts

The CLI's _load_prepared_frame explicitly guards against a stale cross-domain frame in the flat data/ cache ("would crash or silently mis-screen"), but resolve_dims reads data/features/train_features.parquet with no such check. A leftover features file from a much smaller domain yields wrong n_observations, which feeds not just the consent cost estimate but every "auto" per-arm timeout (resolve_arm_timeout) — undersized dims can push predicted runtimes below reality so that legitimate arms of the new domain are killed at the 1800s floor. Soft degradation (floor + 3x multiplier absorb some error) but it quietly undermines the #138 auto-timeout fix.

**Fix:** Have the caller pass dims only when _load_prepared_frame accepted the prepared frame (they already share the existence checks), or add the same descriptor-column sanity check before trusting the parquet's row count.

#### src/panelcast/select/tiers.py:107 — Malformed tiers: block raises raw AttributeError instead of the promised ValueError

load_tiers documents "a present-but-malformed file raises" and the CLI catches ValueError to print a friendly error, but a tier entry that is not a mapping (e.g. `tiers: {quick: [2, 500]}` or `quick: null`) hits `spec.get(...)` and raises AttributeError, giving users an unhandled traceback instead of the "malformed select config" message; likewise `tiers: [..]` fails at block.items().

**Fix:** Validate shape before merging: `if not isinstance(block, dict): raise ValueError(...)` and per-entry `if not isinstance(spec, dict): raise ValueError(f"tier '{name}' must be a mapping")`.

#### src/panelcast/pipelines/train_bayes.py:902 — expected_gb telemetry priced before gate flags are attached; comment claims the estimator doesn't consume them

`expected_gb = estimate_memory_gb(**estimate_inputs)` runs before errors_in_variables / heteroscedastic_entity_obs / entity_group_pooling are added to the dict, and the comment at line 905 ('the estimator doesn't consume them') is stale — estimate_memory_gb accepts all three. A gated fit's logged resource_usage.expected_gb and ratio therefore under-read (ratio inflated), which misleads anyone auditing per-run prediction error; the calibration refit itself is unaffected because it recomputes terms from the recorded inputs including the gates. chain_method is likewise not priced into expected_gb for vectorized fits.

**Fix:** Build the full estimate_inputs (gates included) first and compute expected_gb from it, passing chain_method=mcmc_config.chain_method; delete the stale comment.

#### src/panelcast/gpu_memory/calibration_store.py:69 — append_record read-modify-write loses records under concurrent appenders

Two concurrent select-arm children finishing near-simultaneously both read the store's N records, each appends its own record, and both os.replace the file — the last writer wins and one calibration/runtime datapoint is silently dropped. The per-pid tmp name (line 80) prevents tmp-file collisions but not the lost update. Bounded impact (best-effort telemetry, next fits re-append), but under --parallel-arms this happens routinely at bucket boundaries.

**Fix:** Guard the read-append-replace with a sidecar lock file (e.g. `path.with_suffix('.lock')` + O_CREAT|O_EXCL retry), or accept and document the loss.

#### src/panelcast/reporting/model_card.py:1119 — update_model_card_with_results silently drops flagged_slices, expected_false_flags, and ranking_summary

The returned ModelCardData is constructed without the three #181/#182 fields, resetting them to defaults. Currently masked only because publication.py happens to set them after the call; any caller that sets them before (the natural order for a data-then-results API) loses the sliced-calibration and ranking sections without any error.

**Fix:** Pass flagged_slices=data.flagged_slices, expected_false_flags=data.expected_false_flags, ranking_summary=data.ranking_summary through the constructor (or build the new instance with dataclasses.replace).

#### src/panelcast/cli/runs_cmd.py:290 — runs diff lists non-output-affecting flags under the 'output-affecting' config delta

flag_differences is called with no ignore set, and manifest flags include verbose, progress_bar, resume, skip_existing, and dry_run; diffing a --verbose run against a quiet one prints 'verbose: False -> True' under the header 'config delta (output-affecting, defaults-aware)', misleading like-for-like judgments.

**Fix:** Pass ignore=frozenset({'verbose', 'progress_bar', 'resume', 'skip_existing', 'dry_run'}) (dataset_descriptor_hash is already surfaced separately) to flag_differences in runs_diff.

#### src/panelcast/reporting/tables.py:650 — export_table writes .tex with the platform-default encoding

tex_path.write_text(latex_str) omits encoding=; on a native-Windows Python (cp1252) a table containing non-cp1252 characters — e.g. the ΔELPD column create_sensitivity_summary_table produces — raises UnicodeEncodeError and kills the export. Every other text write in the repo pins encoding='utf-8'.

**Fix:** tex_path.write_text(latex_str, encoding='utf-8').

#### src/panelcast/reporting/model_card.py:444 — LaTeX model card passes Markdown markup through as literal text and omits the md-only sections

architecture_summary/limitations contain Markdown (**bold**, numbered/bulleted lists) that _generate_latex emits verbatim, so the compiled card (once the underscore bug is fixed) shows literal ** and dash bullets; the LaTeX variant also silently lacks the flagged-calibration-slices and ranking sections the Markdown card renders, so the two formats disagree.

**Fix:** Either strip/convert the Markdown markers when rendering LaTeX prose and add the missing sections, or document the .tex card as a reduced summary.

#### src/panelcast/select/scoring.py:145 — Select scoring and backtest aggregation hardcode coverage keys "0.80"/"0.95" while calibration_intervals is configurable

_apply_metrics (and backtest's _AGGREGATED_METRICS) read metrics.json coverages only at keys "0.80"/"0.95". A sweep whose extra_config (or a run whose YAML) sets non-default calibration_intervals writes different keys, so cov80_delta/cov95_delta stay None, screenable() then rejects every arm ("no coverage evidence") and the sweep can never nominate a confirmation candidate — silently. Not reachable from the current select CLI (extra_config is never populated there), hence a smell rather than a bug today.

**Fix:** Derive the coverage keys from the levels present in metrics.json (or pin select to always pass calibration_intervals 0.80,0.95 into arm configs), and document the contract next to _evaluate_predictions' f"{prob:.2f}" key format.

## Feature / scope / research ideas

### Missing features (7)

#### [M] What-if next-event scoring: `panelcast predict` verb + a Predictor facade

High. The predict stage emits only three canned scenarios (same/population_mean/entity_mean — SCENARIOS_KNOWN in pipelines/predict_next.py) batch-dumped for all entities. A user asking the most natural forecasting question — 'what does the model say about entity X's next event with these covariates?' — has no supported path; MODEL_CARD.md's How-to-Use section is a ~50-line manual snippet (load manifest, extract samples, rebuild transform, fetch ar_center) that is itself evidence the surface is missing. Distinct from filed #173, which exports pipeline seams (run_pipeline, DatasetDescriptor), not a prediction facade.

*Sketch:* Add `panelcast predict --run <id> --entity NAME [--covariates file.yaml] [--new-entity]` in cli/commands.py. Implement a `Predictor.from_run(run_dir)` facade (new src/panelcast/predictor.py) that wraps exactly the boilerplate in MODEL_CARD.md lines 260-311: load_manifest/load_model, extract_posterior_samples, get_transform from training_summary.json, ar_center_on_model_scale. Reuse _build_batch_scenario_args and predict_new_entity for the compute; standardize user covariates against the run's feature scaler. Output draws + quantiles as JSON/CSV. Feeds #173's docs/API.md as the headline example.

#### [L] Fast approximate-inference tier (SVI/ADVI/Pathfinder) alongside NUTS — **proposal seed**

Medium-high. Every fit is NUTS-only (no SVI/variational code anywhere in src/); iteration and sweep screening burn GPU-hours — the whole #164/#166 rung-ladder machinery exists because fits are expensive. An ADVI tier gives seconds-scale exploratory fits on CPU and a cheap pre-screen for `select` arms. Risk to manage: calibrated uncertainty is the project's brand, so VI output must be visibly labeled approximate and never feed the publication path.

*Sketch:* `--inference {nuts,svi}` in models/bayes/fit.py using numpyro.infer.SVI with AutoNormal / AutoLowRankMultivariateNormal over the unchanged model; sample the fitted guide into an InferenceData so evaluate/diagnose/report work as-is; stamp the manifest and every report with an approximate-inference banner and skip the convergence gate (R-hat is meaningless for VI). Research angle: measure VI coverage/PIT degradation vs NUTS on the real AOTY subset across the nine likelihood families — a clean, publishable calibration study an undergraduate could own.

#### [L] Incremental refresh: `panelcast refresh` for new-data arrival

Medium-high. A forecasting tool over accumulating entity histories implies periodic new events, but the only supported reaction to new rows is a full from-scratch rerun. All the pieces already exist unchained: data-root stamps detect changed inputs, the select warmup-transfer machinery exports adapted mass matrices, and `runs diff` compares runs — no verb connects them. Not covered by #138 (mid-fit checkpointing) or --resume (failed-run recovery).

*Sketch:* `panelcast refresh` = (1) hash raw input vs the last run's recorded raw-input hash, exit 0 'no new data' when unchanged; (2) rerun data→splits→features; (3) refit warm-started from the previous run's adapted state when the model signature matches (reuse select/warm-start export), cold otherwise; (4) auto-emit a runs-diff report flagging per-entity forecast movement beyond a threshold. Label refreshed runs in the manifest; keep cold refit as the reference path. Secondary research question (posterior-as-prior sequential updating vs full refit fidelity) exists but the core is engineering.

#### [L] Wire up the secondary target — then model targets jointly — **proposal seed**

Medium, with a real research payoff. The descriptor carries secondary_target_col/secondary_prefix end-to-end — validation.py schemas it, cleaning.py cleans it, prepare_dataset.py writes critic_score.parquet, features/registry.py appends its specs, and model.py even exports a ready critic_score_model — yet train_bayes.py never fits it (verified: zero references). That's a dangling half-contract a serious user will trip over. The joint version (shared entity random walk, correlated residuals) is the interesting one: borrow critic signal to fix the model's weakest split, cold-start (R² 0.113).

*Sketch:* Phase 1 (S-M): loop train/evaluate/predict over the descriptor's target specs — everything downstream is already parameterized by prefix, so this is mostly orchestration in train_bayes.py/evaluate.py plus per-target artifacts. Phase 2 (research): a bivariate likelihood with a shared latent artist trajectory and target-specific loadings/noise, gated default-off through the select bake-off; headline experiment is whether joint modeling moves cold-start R² on real AOTY data. Undergraduate-proposal-ready as 'multivariate hierarchical outcomes for entity panels'.

#### [M] New-domain onboarding tooling: `panelcast dataset init` + standalone `dataset validate`

High for adoption relative to cost. The README's core pitch is 'point at a new domain with a single descriptor and zero source changes', but authoring that descriptor is hand-written YAML whose first real validation is a pipeline run; `doctor` checks dataset *resolution*, not content, and there is no validate command in the CLI surface (verified against cli/commands.py + main.py). This is the top of the adoption funnel for 1.0.

*Sketch:* `panelcast dataset init raw.csv`: sniff columns/dtypes, interactively map entity/event/score/date/observation-count, detect score bounds from the data, emit a descriptor YAML plus a warnings report (nulls, duplicate events, non-monotone dates). `panelcast dataset validate <descriptor>`: run the existing pandera schema (data/validation.py) and a cleaning dry-run against the raw file, printing row-level exclusion counts and reasons without touching data roots. Both are thin CLI wrappers over code that already exists inside the data stage.

#### [S] Raw-input formats beyond CSV (Parquet at minimum)

Low-medium but very cheap. io/readers.py exposes exactly one reader (read_csv) and AOTY_DATASET_PATH must be a local CSV — for a tool that itself writes Parquet everywhere and targets ~62k-row-plus corpora, Parquet input is table stakes; large real-world panels rarely live as CSVs.

*Sketch:* Dispatch on file suffix in io/readers.py (read_parquet, optionally read_json); thread the column-subset fast path in data/ingest.py::extract_data_dimensions through pyarrow's columns argument (it currently assumes usecols on CSV); keep sha256_file hashing and the data contract unchanged; document in DATA_CONTRACT.md and CLI.md's environment-variable section.

#### [S] Decision-oriented prediction outputs: exceedance probabilities, custom quantiles, documented draw export

Medium. next_event CSVs carry a fixed q05/q25/q50/q75/q95 grid (verified in predict_next.py); a user asking 'P(next score ≥ 80)?' or wanting the full predictive draws must load the NetCDF and reimplement the transform chain by hand. The per-row draw snapshot already exists (evaluation/predictive.npz) but only behind the sweep-internal PANELCAST_SAVE_PREDICTIVE env var — undocumented for end users.

*Sketch:* Add `--prediction-quantiles 0.05,0.5,0.95` and `--exceedance-thresholds 70,80,90` config knobs; in predict_next.py's summary-row builders the draws are already in hand, so each p_ge_{t} column is a one-line np.mean(samples >= t). Promote PANELCAST_SAVE_PREDICTIVE to a documented `--save-predictive` flag and describe predictive.npz in ARTIFACTS.md so downstream users get the full distribution without touching ArviZ.

### Scope widening (12)

#### [M] Lightweight local scoring service over a completed run

Low-medium, flagged honestly: MODEL_CARD.md explicitly lists 'real-time prediction systems in production environments' as out-of-scope, so a full serving stack contradicts stated intent. But a localhost read-only HTTP wrapper (load one run, expose /predict and /health) lets dashboards and notebooks consume forecasts without shelling out to the CLI, and costs little once the Predictor facade (idea 1) exists. Do not prioritize before 1.0; strictly sequenced after idea 1.

*Sketch:* `panelcast serve --run <id> --port 8799`: FastAPI (optional extra, not a core dep) with POST /predict accepting {entity, covariates?} and returning mean/quantiles/exceedance from the in-memory Predictor; GET /run returning the manifest summary. Read-only over an existing run directory, no refit path, explicit 'research tool, not production' banner in the docs.

#### [M] Unbounded and open-interval targets (target_bounds: null) — **proposal seed**

Opens finance returns, log-citations, ELO-style ratings, and lab values — any panel target without natural bounds. The codebase already has a scar from forcing one in: model.py's _sample_sigma_obs docstring records 'the econ log-citation failure, sci_sigma_obs -> 0.004' from squeezing a zero-inflated unbounded target into the bounded machinery.

*Sketch:* Blocked today in three places: DatasetDescriptor requires a finite lo<hi target_bounds tuple (src/panelcast/config/descriptor.py:140 + validator ~line 177), data/cleaning.py drops rows outside bounds (line 441) and data/validation.py enforces them, and the identity transform hard-wires soft_clip into bounds (models/bayes/transforms.py:86). Minimal change: allow target_bounds: null in the descriptor; register a 'none' TargetTransform (identity forward/inverse, zero log-jacobian, transform_mu = identity, no clip); skip bound filtering/validation and bound-dependent PPC stats when bounds are null; offset_logit/beta/beta_binomial families reject null bounds at config time. The transform registry makes this one new entry plus guards.

#### [M] Count-target likelihood family (NegBin/Poisson with log link) — **proposal seed**

Raw event counts — weekly streams, citations per paper, goals per match, hospital admissions — are the most common panel target the tool can't express: beta_binomial only covers rater-mean aggregation counts (gated by n_obs_is_aggregation_count), and every other family is continuous location-scale.

*Sketch:* The likelihood seam was built for exactly this: REGISTRY in models/bayes/likelihoods.py (lines 748-825) resolves families by name, and docs/LIKELIHOOD_CANDIDATES.md documents 'adding a family is a single new entry'. Add a 'negbinom' LikelihoodSpec: mu passes through an exp (or softplus) link, phi-style overdispersion site {prefix}nb_conc, predict_draws for cold-start, cdf=None; uses_sigma=False rides the existing sigma-skip path the beta family already exercises (model.py:783-798). Depends on relaxing target_bounds to [0, inf) or null (previous idea). y stays on the count scale so evaluate/predict are untouched — the same contract every existing family honors.

#### [M] Censored observations via the existing CDF plumbing — **proposal seed**

Clinical panels (assay detection limits, dropout), sports (DNF/withdrawn), and platform data (scores clipped at display floors) all carry partially observed targets; today a censored row must be dropped or treated as exact, biasing entity trajectories downward/upward.

*Sketch:* Every location-scale LikelihoodSpec already carries a cdf field, and RoundedDistribution (likelihoods.py:240) proves the interval-censored log-mass pattern log(F(hi)-F(lo)) works in this codebase — it's just currently wired only to integer dequantization. Minimal change: descriptor gains censor_col (values none/left/right per row, default absent = all exact); the likelihood wrapper contributes log F(y) for left-censored and log(1-F(y)) for right-censored rows using the family's cdf, exact log_prob otherwise; families with cdf=None reject the column. Cleaning keeps censored rows instead of bound-filtering them. Prediction paths are unchanged (censoring is a training-likelihood concern).

#### [L] Gap-aware continuous-time latent process (rw_ct / OU) — **proposal seed**

The latent RW/AR(1) is indexed by event sequence, so a 2-week gap and an 8-year hiatus contribute identical innovation variance — release_gap_days exists only as a fixed covariate (features/temporal.py:134). Fixing this is the single biggest correctness upgrade for irregular-cadence domains: poll timing in elections, off-seasons in sports, clinical visit windows.

*Sketch:* Blocked in _build_latent_effects (models/bayes/model.py:357): innovations are sigma_rw * unit-normal per sequence step, gap-blind. Minimal change: a third latent_process option 'rw_ct' behind the existing seam — the pipeline passes a per-(entity, seq) gap matrix dt (already derivable from the parsed date column the temporal block sorts on), innovations become sigma_rw * sqrt(dt) (Wiener), or the OU discretization phi^dt for the ar1 variant. The seam's contract (return (max_seq, n_artists)) is unchanged, gate-off stays bit-identical, and rollout.py's terminal-state resampling picks up the same dt scaling. The main cost is threading the dt matrix through train/evaluate/predict alongside album_seq.

#### [M] Shared global time effect (exogenous shocks / common shocks) — **proposal seed**

Nothing in the model is common across entities at the same calendar time: release_year enters only as a linear covariate, so score inflation, COVID-style shocks, a rule change in a sports league, or a national swing in elections gets smeared into per-entity RW innovations. A shared period effect is what makes panel models useful for shock-heavy domains — and it's the standard 'national swing' term an elections model needs.

*Sketch:* Add a gated global_time_effect: a zero-sum (identified against mu_artist) random walk over calendar periods — year bins derived from the descriptor's parsed date column — indexed per observation and added to mu_raw in model.py (~line 768). Implementation follows the two established gate patterns: sites appended after all existing ones for bit-identical gate-off RNG (the _apply_entity_overdispersion contract, model.py:296-304), and a descriptor-driven period index like group_idx_by_artist. Evaluation/predict hold the last period's effect (or propagate the RW, mirroring propagate_rw_horizon).

#### [M] Crossed observation-level random effects (pollster/critic/venue house effects) — **proposal seed**

The model has exactly one grouping plate (the entity); domains where each observation also belongs to a measurement source — pollster house effects in elections, publication effects in criticism, venue/referee effects in sports — can't be expressed. This is the concrete gap between the elections retarget being structural (issue #175's distilled example) and being credible: poll aggregation without house effects is not competitive.

*Sketch:* Blocked because entity_group_col is per-entity ('modal value over training rows becomes its group', descriptor.py:130-132) — there is no per-observation grouping concept. Minimal change: descriptor obs_group_col; the pipeline builds an obs-level group index; the model adds a gated ZeroSumNormal intercept per obs-group to mu_raw, copying the entity_group_pooling implementation nearly line-for-line (model.py:711-731, sigma_group HalfNormal + ZeroSumNormal offsets). Cold-start prediction marginalizes over an unknown source (group_idx_new = -1 already models this pattern in predict.py).

#### [M] Multi-level hierarchical nesting (entity ⊂ group ⊂ supergroup) — **proposal seed**

One pooling level caps the tool at flat panels. Player⊂team⊂league, county⊂state⊂region (elections), patient⊂site⊂trial (clinical) all need two-plus levels, and partial pooling across levels is precisely where hierarchical Bayes beats the GBM baselines on sparse groups.

*Sketch:* Blocked: descriptor.entity_group_col is a single Optional[str] (descriptor.py:132) and the model accepts one (group_idx_by_artist, n_groups) pair feeding one ZeroSumNormal offset (model.py:711-731). Minimal change: generalize to entity_group_cols: list[str] ordered leaf-to-root; the model loops levels, each level's ZeroSumNormal offset added to mu_entity with its own sigma_group_l, child offsets centered within parents. Site names {prefix}group_offset_{level} keep single-level configs byte-compatible (level 0 keeps the current names). The train stage's group-index derivation generalizes from one modal-value pass to one per column.

#### [L] Covariate-informed cold-start (entity-level intercept regression) — **proposal seed**

Cold-start is the model's admitted weakest split (entity-disjoint R² 0.113 per MODEL_CARD.md) because predict_new_entity draws the effect from the bare population Normal(mu_artist, sigma_artist) (predict.py:291-301) — a debut artist's label/genre metadata, a new candidate's district fundamentals, a rookie's draft position can't shift the prior. Fixing this converts panelcast from 'good on knowns' to usable for the debut-heavy questions most domains actually ask.

*Sketch:* Blocked: init-effect location is mu_entity only (model.py:729-734); no entity-level design matrix exists anywhere in the pipeline (X is observation-level). Minimal change: gated entity_covariates — descriptor names static entity columns; the features stage aggregates them to a per-entity Z matrix (first-event values); the model adds mu_entity += Z @ gamma with gamma ~ Normal(0, scale) sampled before the init-effect site (mid-sequence insertion, the entity_group_pooling RNG precedent); predict_new_entity gains a Z_new argument that conditions the cold-start draw. Bake-off metric already exists: the entity-disjoint split.

#### [L] Interrupted-time-series intervention effects (causal-adjacent uplift) — **proposal seed**

The model already estimates per-entity counterfactual trajectories (hierarchical RW + AR(1)); adding a pre/post intervention term turns it into a principled interrupted-time-series tool — did the label signing, coaching change, or policy rollout shift this entity's trajectory, with partial pooling of the effect across treated entities? That is a genuinely differentiated research artifact, not just another forecaster.

*Sketch:* Blocked: no intervention concept in the descriptor, no step-change term in mu_raw, and reporting is purely predictive. Minimal change: descriptor intervention_col (per-row 0/1 flag, domains mark rows at/after the entity's event); model adds a gated hierarchical effect delta_e ~ Normal(delta_pop, sigma_delta) entering mu_raw as delta_e * flag, sites appended last for gate-off parity; a small reporting path summarizing the delta_pop / per-entity delta posteriors (the run-scoped reports directory already exists). Honest scoping note in docs: identification rests on the RW counterfactual assumption, same as ITS generally — document it the way LEAKAGE_CONTROLS.md documents split honesty.

#### [M] Ordinal-target family (ordered probit with learned cutpoints) — **proposal seed**

Letter-grade panels (credit ratings agencies, Pitchfork-era letter grades, wine ratings, Likert survey panels) are ordinal, not interval — currently the only options are pretending categories are equal-width integers (DequantizedDistribution assumes a uniform grid) or dropping the data.

*Sketch:* One new LikelihoodSpec in models/bayes/likelihoods.py: 'ordered_probit' with K-1 learned ordered cutpoints ({prefix}cutpoints via an ordered transform), mu from the existing linear predictor, y as category indices; predict_draws samples categories for cold-start. uses_sigma=False rides the beta-family sigma-skip path. Needs target_bounds interpreted as category range [0, K-1] and evaluation to skip continuous-only metrics (CRPS→ranked probability score) — the metric dispatch is the larger half of the work. Registry design keeps model.py untouched.

#### [L] Joint multivariate outcomes (correlated user+critic style targets) — **proposal seed**

The secondary-target path (descriptor secondary_* triple) fits two fully independent models, so the tool can't answer conditional questions — 'critics scored it 82, update the user-score forecast' — nor borrow strength across sparse targets. Multi-outcome panels (multiple assays per patient visit, approve/favorable pairs in polling, box-office+score) are common and currently only expressible as disconnected runs.

*Sketch:* Blocked by architecture, not names: make_score_model builds one prefix, one likelihood, one entity-effect stack (model.py:541+), and train_bayes.py runs the secondary model as a separate MCMC. A minimal first step short of a full joint model: correlated entity effects — fit both targets in one model sharing artist_idx/album_seq, with a 2-d init effect and LKJ(2) correlation between the two targets' entity effects and observation noises, site names keeping both existing prefixes so evaluation code reads each margin unchanged. Gate behind joint_secondary (default off = two independent fits, parity preserved).

### Research angles (for the proposals) (14)

#### [L] Time-gap-aware latent process (gap-scaled random walk / continuous-time OU) — **proposal seed**

High as statistics, medium as engineering. Verified in models/bayes/model.py (~lines 396-424): the artist trajectory is a cumsum of unit innovations indexed by event sequence, so a 10-year hiatus carries exactly the same innovation variance as a 6-month follow-up — a real misspecification for a tool whose pitch is 'entity histories over time'. Release dates already flow through the temporal feature block, so Δt is available in-pipeline. Also compounds the shipped propagate_rw_horizon gate: horizon widening would become calendar-time-aware.

*Sketch:* Compute per-entity inter-event Δt in build_features; add `--latent-process rw_time` (innovation sd scaled by sqrt(Δt)) and/or `ou_time` (phi = exp(-Δt/tau), tau learned) next to the existing rw/ar1 options in make_score_model. Run it through the same pre-registered select/bake-off discipline used for #155/#158, AOTY defaults byte-identical until promoted. Proposal-ready question: does discretizing a continuous-time latent process beat event-index random walks for irregularly-sampled entity panels? Testable on AOTY (hiatus artists) and the aero example.

#### [M] Simulation-based calibration (SBC) harness for the inference machinery — **proposal seed**

Medium-high for a project whose differentiator is 'diagnostics as gating checks': the sampler+model machinery itself is validated only by synthetic recovery spot-tests, not by the field-standard SBC rank-uniformity check (Talts et al. 2018). For the public release, 'the inference passes SBC at pipeline scale' is a credibility line no current artifact can back.

*Sketch:* `panelcast sbc --sims 200 --preset quick`: draw theta from the prior, simulate observations through the model's own generative path (numpyro Predictive over make_score_model), refit, accumulate posterior rank statistics per parameter block; report ECDF-difference / chi-square uniformity with pass/fail thresholds and a figure in the run report. Size like the aero example so it runs nightly on CPU in CI. Proposal-ready: SBC behavior across the nine likelihood families and both transforms on bounded, skewed targets — a self-contained methods project.

#### [S] Cross-domain prior transfer (priors-from-run) — **proposal seed**

The hyperparameters the model learns on AOTY (sigma_artist, sigma_rw, rho — all on the standardized/logit scale) are plausibly portable to any new sparse domain, and a data-poor domain is exactly where informative priors matter most. Today docs/PORTING.md warns portability is 'structural, not predictive'; transferring learned hyperpriors is the cheapest experiment that could change that sentence, and it's publishable methodology (Bayesian transfer across panel domains).

*Sketch:* Nothing blocks it structurally — priors are already fully configurable via PriorConfig (models/bayes/priors.py) and the config layer; what's missing is the bridge. Add a small CLI command (pattern: existing runs subcommands) that reads a completed run's posterior summary and emits a prior-config YAML: e.g. sigma_artist_lognormal_loc/sigma from the fitted sigma_artist posterior, rho_loc/scale from rho, sigma_rw likewise. Validation experiment: fit the aerospace example (and the #175 elections example) with default vs AOTY-transferred priors at small n and compare held-out ELPD — a self-contained study using only existing evaluation machinery.

#### [M] The bounded-skew likelihood problem: why six observation families fail on aggregated bounded scores — **proposal seed**

This is the strongest proposal seed in the repo: a documented, reproducible, open modeling problem. The proposal would claim to characterize (and ideally resolve) the structural PPC misfit — skewness/max/q90 p-values pinned at the extremes on real AOTY data after beta, skew_studentt, skew_normal, split_normal, beta_binomial, mixture, beta_ceiling, dequantization, and the offset_logit x ar1 grid ALL failed (docs/LIKELIHOOD_CANDIDATES.md). Candidate next moves are genuinely publishable: ordinal/graded-response models treating scores as aggregated ordinal ratings, covariate-dependent (distributional/GAMLSS-style) skewness, Dirichlet-process or normalizing-flow residuals, or a formal argument that the pins are irreducible for any location-scale family on this data-generating process. Even a well-argued negative extends a six-family negative-result dossier into a paper.

*Sketch:* panelcast already provides: the plug-and-play LikelihoodSpec REGISTRY in src/panelcast/models/bayes/likelihoods.py (one entry adds a family, no scattered edits), the real ~5k-album left-skewed subset (scripts/make_aoty_subset.py, skewness -2.08 matching the full corpus), the PPC pin harness, bake-off drivers (scripts/bakeoff_likelihoods.py), and seven archived negative results with convergence/PPC/point-metric tables. Needs building: 1-2 new candidate families (ordinal-aggregation or distributional-skew), GPU diagnostic fits (4x1000) per candidate, and a short theory section on why bounded-skew aggregate means defeat symmetric-and-simple-skew families. The pre-registered screen-then-confirm discipline is already the methodology section.

#### [M] Simulation-based calibration (SBC) audit of the full hierarchical pipeline — **proposal seed**

The proposal would claim the first end-to-end SBC validation (Talts et al. rank statistics) of a production hierarchical RW+AR(1) panel model — does the inference machinery recover its own priors, and where does it break (sigma_rw, rho, the heteroscedastic seam)? SBC is the recognized gold standard the project conspicuously lacks: grep confirms no SBC harness exists anywhere in the repo, while every other diagnostic (R-hat/ESS gates, PPC, coverage, prior predictive) is already first-class. A secondary claim: SBC as a cheap CI-gating check for Bayesian pipeline software, demonstrated on a real codebase.

*Sketch:* panelcast provides: the exact generative model as code, prior-predictive machinery (src/panelcast/evaluation/prior_predictive.py, select/prior_screen.py), synthetic panel generators (scripts/generate_aero_example.py and the synthetic panels in experiment scripts), and reduced-scale presets so hundreds of fits are tractable on CPU. Needs building: an SBC driver (draw theta~prior, simulate a panel, fit at validation scale, compute posterior ranks, chi-square/ECDF uniformity tests per parameter), which is a well-scoped few-hundred-line harness — ideal undergraduate scope with a clear pass/fail deliverable and a natural CI integration story.

#### [M] What does full Bayes buy over conformal-wrapped ML? A calibrated-uncertainty benchmark — **proposal seed**

The proposal would formalize the question docs/BASELINES.md already stages: conformal GBM reaches near-nominal coverage cheaply (0.954 at 95%) but loses on CRPS, width, and cold-start, and its exchangeability guarantee is false by construction for never-seen entities (0.920 coverage). Claim: on entity-nested bounded-score panels, quantify exactly when a generative hierarchical model beats conformal-wrapped ML (and Stan/brms/prophet/NGBoost implementations of comparable models) on proper scores, sliced calibration, and cold-start — a decision-relevant answer to a live methods debate (Bayes vs conformal).

*Sketch:* panelcast provides: the leak-safe split harness, `panelcast compare --baselines` with ridge/GBM/conformal-GBM/persistence on identical splits, CRPS/coverage/sliced-calibration/Wilson-CI metrics, the conformal_calibration wrapper on the Bayesian model itself (both directions of the comparison in one codebase), and rolling-origin backtest for multiple evaluation slates. Needs building: 2-4 external competitors (a brms/Stan refit of the same model as a cross-engine check, NGBoost or deep-ensemble intervals, a prophet-style per-entity baseline) plugged into the existing baseline panel format, plus a second domain to show the ordering is not AOTY-specific.

#### [M] Decision-theoretic evaluation: from calibrated posteriors to top-K selection utility — **proposal seed**

The proposal would claim that calibration differences which look small in coverage tables become large in decision utility: simulate an A&R-style budget-constrained shortlisting problem (pick K artists/albums expected to score highest) and measure expected-utility regret of the Bayesian posterior vs the over-confident GBM (0.763/0.888 coverage) vs conformal intervals. This converts panelcast's core thesis — 'calibrated uncertainty is the deliverable, not point accuracy' — into a measurable, defensible claim rather than a slogan.

*Sketch:* panelcast provides: ranking metrics already shipped in evaluate (Spearman/Kendall, expected vs realized rank, P(top-K) for K in {5,10,25} each with audited reliability curves, per docs/EVALUATION_PROTOCOL.md), entity-identified predictions with error decomposition, and `panelcast backtest --origins K` to generate multiple held-out slates so top-K variance is tamed. Needs building: a decision layer (utility function, budget constraint, regret computation over posterior draws vs competitors' intervals) — a read-only consumer of existing prediction artifacts, so no model changes and no GPU dependency beyond fits that already exist.

#### [S] Meta-analysis of the model-selection ledger: does diagnostic-scale screening predict publication-scale truth? — **proposal seed**

The proposal would mine an unusually complete, versioned selection history: 26 recorded verdicts with evidence classifications (.audit/verdict_ledger/LEDGER.md), sweep arm ledgers (.audit/select_aoty_011/, the 26-arm 0.7.0 ledger), and paired held-out ELPD discipline. Claims: (1) empirically, which cheap signals (diagnostic-scale ELPD z, PPC pin count, ESS) predicted publication-scale confirmations across the project's history; (2) a cautionary methods case study of the #63 estimator bug — PSIS-LOO misapplied to held-out data — including which verdicts survived re-verification (24/26) and which were contaminated. Meta-research on Bayesian workflow with the raw materials already public.

*Sketch:* panelcast provides: the verdict ledger with per-verdict evidence classes, byte-identical recompute receipts from persisted log_likelihood.nc snapshots, select sweep reports landing in .audit/ by construction since 0.7.0, and the documented gaps list of unrecoverable pre-#63 comparisons. Needs building: aggregation/analysis notebooks over the JSON ledgers, a screening-vs-confirmation concordance table, and optionally 1-2 cheap refits to close named gaps (e.g. the entity_overdispersion C1 row, the one genuinely suspect verdict). Almost no new infrastructure — the lowest-effort viable proposal here.

#### [L] From structural to predictive portability: a multi-domain transferability study — **proposal seed**

MODEL_CARD.md states the honest limitation verbatim: 'Domain portability is structural, not predictive... Accuracy on any non-AOTY domain is untested by construction.' The proposal closes that gap as its thesis: fit the identical descriptor-driven model on 3-5 real bounded-score panel domains (elections via the existing sibling retarget, movie/IMDb scores and an econ panel — both already touched by the entity-overdispersion experiments, sports ratings, course evaluations) and characterize when partial pooling + random-walk trajectories + AR(1) transfers. Deliverable: an empirical map of the model shape's domain of validity, which is exactly the claim the 'general tool' framing needs but cannot yet make.

*Sketch:* panelcast provides: the one-YAML DatasetDescriptor retarget with zero source changes (proven by tests/e2e/test_domain_portability.py), the aerospace worked example, the elections_pred sibling-repo precedent, per-domain baseline comparison on identical splits, and experiment_entity_overdispersion.py's existing IMDb/econ data hooks. Needs building: dataset curation and descriptors for each new domain, GPU fits per domain, and a cross-domain results synthesis (which variance component carries the signal per domain — extending docs/decisions/what_carries_the_signal.md). Large but cleanly parallelizable across domains, so it also scopes down gracefully to 2 domains.

#### [L] Scaling the fit: exact NUTS vs approximate inference at the full 62k-album corpus (#15) — **proposal seed**

Full-corpus validation is the project's own declared open item (#15: needs >24 GB GPU even with the rw_raw collection exclusion). The proposal turns that blocker into the research question: at what corpus scale do cheap approximate methods (SVI/ADVI, pathfinder, Laplace) match full NUTS on the metrics that matter (coverage, CRPS, PPC pins, variance decomposition) for hierarchical panel models? The subset-vs-full design is pre-built — the ~5k subset with matched skewness is the validated anchor, so approximation error can be measured against a known-good exact posterior at small scale before extrapolating.

*Sketch:* panelcast provides: the GPU memory estimator and preflight gates, the rw_raw exclusion machinery (94% peak-memory cut, parity-tested), subset construction with matched skewness (scripts/make_aoty_subset.py), vectorized-chains and checkpoint/resume work from the 0.8-0.12 trains, and NumPyro — where SVI is available in the same framework as the existing NUTS model, so the model function is reused verbatim. Needs building: an SVI/pathfinder fitting path plus the exact-vs-approximate comparison harness, and cloud A100/H100 hours for the full-corpus endpoints. The compute ask itself is a natural line item for a proposal budget.

#### [S] How reproducible is 'reproducible'? An empirical study of bit-exactness domains in MCMC — **proposal seed**

The proposal would test panelcast's own two-tier reproducibility claim (docs/EVALUATION_PROTOCOL.md): draws reproduce bit-exactly within a matching environment fingerprint (python/jax/jaxlib/numpyro/platform/device hash) and only statistically across fingerprints. Claims: empirically map which fingerprint components actually break bit-exactness (jaxlib patch versions? GPU vs CPU? driver versions?), quantify the statistical divergence when they do, and evaluate the fingerprint design (it deliberately excludes pixi.lock and OS release — is that exclusion justified?). Doubles as a ReScience-style teaching artifact on trustworthy Bayesian workflow, with runs verify/reproduce and output hashing as the demonstration vehicle.

*Sketch:* panelcast provides: environment fingerprinting in every run manifest, output hashing + `runs verify`, `resolved_config.yaml` + `runs reproduce` (the 0.9.0 provenance arc), and fast validation-scale presets so a full matrix of environments is cheap. Needs building: a test matrix (2-3 jax/jaxlib versions x CPU/GPU x 2 machines), a divergence-measurement script (first-divergent-draw index, posterior-summary deltas), and a short write-up. Small, self-contained, and the infrastructure being evaluated is the finished part of the project.

#### [M] Dynamic honesty at horizon: calibration decay of multi-step ancestral rollouts — **proposal seed**

Every flagship number is one-step-ahead teacher-forced; the model card documents a known defect at depth — predictions past the training horizon omit accumulated random-walk variance, so deep-extrapolation intervals are provably too narrow, with a shipped default-off fix (propagate_rw_horizon) whose value is unmeasurable on the within-horizon holdout. The proposal claims: measure how forecast calibration and CRPS decay with horizon h for hierarchical panel models, verify the sqrt(h - max_seq)*sigma_rw widening correction empirically, and compare against persistence/ML baselines whose uncertainty does not compound. A crisp pre-registered hypothesis (the gate widens intervals toward nominal coverage at depth) with the falsification machinery already shipped.

*Sketch:* panelcast provides: eval_horizon ancestral rollout (per-draw feedback of sampled scores into the AR lag plus fresh latent innovations, landing CRPS/coverage/RMSE per horizon in horizon_rollout.json with an h=1 reconciliation anchor), the propagate_rw_horizon gate, and rolling-origin backtest to build genuinely deep holdouts. Needs building: an evaluation design that reserves 3-5 trailing events per prolific entity (the current splits hold out only the last event), runs with the gate on/off, and a horizon-decay analysis. Moderate compute; no model changes.

#### [S] The predictability ceiling of debut reception: headroom analysis for cold-start forecasting — **proposal seed**

The proposal would formalize a finding the repo states in passing: cold-start R-squared is 0.113 against a covariates-only headroom estimate of ~0.083 (.audit/genre_pooling/covariates_only_r2.md), i.e. the model already captures essentially all extractable debut signal and debut reception is close to unpredictable from available features. Claims: estimate the information-theoretic ceiling on new-entity prediction rigorously (oracle/flexible-model headroom, feature-set ablations, richer external features like label/promotion metadata), and either raise the ceiling or establish it as a robust negative — 'how predictable is a debut?' is an accessible, headline-friendly research question for an undergraduate project.

*Sketch:* panelcast provides: the artist-disjoint cold-start split with the population-level predictive path, the genre-pooling tier and gbm_offset covariate machinery (the two shipped attacks on this ceiling, both audited), the existing headroom analysis to extend, and feature-ablation support in the sensitivity stage. Needs building: a headroom-estimation protocol (flexible oracle regressors on covariates only, with CIs), optional new feature curation, and the comparison write-up. Small-to-medium; most runs are CPU-cheap because cold-start baselines dominate the compute.

#### [M] When does errors-in-variables correction matter? Mapping the attenuation regime for AR panel models — **proposal seed**

The gated EIV option corrects a real specification error (regressing on the observed noisy previous score attenuates rho), synthetic-recovery tests confirm de-attenuation works, yet the real-data bake-off was null (LOO +0.4 vs SE ~29.6). The proposal claims: map the data-regime boundary where EIV correction transitions from irrelevant to essential — as a function of review-count distribution, true rho magnitude, entity history length, and noise scale — via a designed simulation study, then place real domains (AOTY, elections, IMDb) on that map. A clean 'why was the fix null here, and where would it not be?' methods paper anchored by an already-shipped implementation.

*Sketch:* panelcast provides: the gated errors_in_variables implementation with its fixed data-derived measurement-error latent (MODEL_CARD.md documents the design and its transform-nonlinearity caveat), synthetic-recovery tests, the v1-vs-v2 bake-off protocol under .audit/model_v2_bakeoff/, and synthetic panel generators to vary the regime knobs. Needs building: a factorial simulation grid over (n_reviews distribution, rho, panel depth), the attenuation-vs-correction measurement harness, and the regime-map synthesis. CPU-tractable at validation scale, so no GPU dependency.