# Artifacts

Data roots are a deterministic cross-run cache and stay flat; mutable products
(models, evaluation, predictions, reports) are scoped under a timestamped run
directory `outputs/<run_id>/`, with `outputs/latest.json` pointing at the most
recent successful run (failed runs move to `outputs/failed/<run_id>/`).

Data (shared cross-run cache)
- data/raw/all_albums_full.csv (raw input)
- data/processed/cleaned_all.parquet
- data/processed/user_score_minratings_*.parquet
- data/features/within_entity_temporal/*.parquet
- data/features/entity_disjoint/*.parquet
- data/features/manifest.json
- data/splits/within_entity_temporal/manifest.json
- data/splits/entity_disjoint/manifest.json
- data/splits/within_entity_temporal/split_*.json (immutable archive)
- data/splits/entity_disjoint/split_*.json (immutable archive)
- data/audit/* (cleaning/exclusion provenance)

Runs
- outputs/<run_id>/manifest.json
- outputs/<run_id>/resolved_config.yaml (the fully resolved PipelineConfig for this run; the provenance source `runs reproduce` re-executes from)
- outputs/<run_id>/dataset_hash.txt
- outputs/<run_id>/pipeline.log.json
- outputs/latest.json (pointer to the latest successful run)
- outputs/failed/<run_id>/failure.json (failed runs only: stage, exception, hint, resume command, and recent events — surfaced by `runs why`)

Models
- outputs/<run_id>/models/user_score_*.nc (ArviZ InferenceData, NetCDF)
- outputs/<run_id>/models/manifest.json (current models + history)
- outputs/<run_id>/models/training_summary.json

Evaluation
- outputs/<run_id>/evaluation/metrics.json
- outputs/<run_id>/evaluation/diagnostics.json
- outputs/<run_id>/evaluation/within_entity_temporal/predictions.json
- outputs/<run_id>/evaluation/within_entity_temporal/calibration.json
- outputs/<run_id>/evaluation/entity_disjoint/predictions.json
- outputs/<run_id>/evaluation/entity_disjoint/calibration.json
- outputs/<run_id>/evaluation/<split>/ranked_slate.csv (per-row ranked slate: entity, y_true, pred_mean, predicted/expected/realized rank, and one p_top{K} column per usable K)

predictions.json carries parallel per-row arrays. Alongside the legacy keys
(y_true, y_pred_mean, y_pred_lower, y_pred_upper, residuals) it is identified:
entity, event, n_reviews, train_history (the entity's training-event count),
group when the descriptor names an entity_group_col, plus per-row y_pred_sd,
pit, and covered flags per calibration interval. `panelcast diagnose --errors`
decomposes these into error_decomposition_<split>.csv, per-entity / group /
review-count-decile rollup CSVs, and a worst-25 Markdown table under the
run-scoped reports dir — read-only, so it works on any past run whose payload
is identified (pre-0.10.0 payloads get a clear re-run message instead).

Predictions
- outputs/<run_id>/predictions/next_event_known_entities.csv
- outputs/<run_id>/predictions/next_event_new_entity.csv
- outputs/<run_id>/predictions/prediction_summary.json

Reports
- outputs/<run_id>/reports/artifact_status.json
- outputs/<run_id>/reports/tables/*.csv
- outputs/<run_id>/reports/figures/*.png
- outputs/<run_id>/reports/index.html (self-contained HTML run dashboard; the
  report stage writes it best-effort and `panelcast report` regenerates it)
- outputs/<run_id>/reports/MODEL_CARD.md (run-generated; the repo-root
  MODEL_CARD.md is the curated card)

Backtest (`panelcast backtest`)
- outputs/backtest/<id>/ledger.json (per-origin checkpoint; identity by origin index enables resume)
- outputs/backtest/<id>/backtest_metrics.json (aggregate: mean/SE/min/max per metric plus per-origin populations)
- outputs/backtest/<id>/backtest_report.md (rendered aggregate table)

With `conformal_calibration` enabled (default off), `metrics.json` gains a
`calibration.conformal` block and the next-event CSVs gain `conformal_q05` /
`conformal_q95` columns; with it off, these artifacts are byte-identical to
before.
