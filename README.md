# Drug Shortage Prediction

This repository builds a monthly drug shortage panel from public administrative sources plus licensed prescription data, then trains the current canonical onset and offset survival models.

This public source snapshot includes the analysis code, aggregate manuscript tables and figures, and reproducibility metadata. It does not include raw source files, record-level datasets, licensed ASHP/UUDIS or Symphony Health data, fitted prediction files, or submission DOCX files. See `DATA_AVAILABILITY.md`, `REPRODUCIBILITY.md`, and `REPOSITORY_MANIFEST.md`.

## Current active design

The analysis is two separate discrete-time survival models, each compared across five rows that use the same eligible observations and validation protocol: logistic regression, a time-only logistic baseline, LightGBM, LightGBM with focal loss, and a time-aware transformer encoder.

- **Onset.** Among drug groups not currently in shortage, the model estimates the probability of a first observed shortage onset in the next month. The clock starts at dataset start, a group stops contributing onset observations at its first onset, groups that never enter shortage are censored at data end, and recurrent onsets after recovery are not modeled in the primary onset task.
- **Offset.** Among drug groups currently in shortage, the model estimates the probability that the shortage episode will end in the next month. The clock starts when that shortage begins, and unresolved episodes are censored at data end.

Each eligible drug-month is a person-period row with a one-step-ahead transition label. There is no fixed horizon. Validation is temporal walk-forward forecasting. For each prediction month, a model is trained only on person-months whose label is already observed, then it predicts the next-month probability for eligible drugs in that month, and predictions are pooled out of time. Reported metrics are C-statistic, AUPRC, Brier score, expected calibration error, and precision and recall at fixed monthly review sizes.

The earlier status-risk design (one any-shortage score read four ways) is superseded and its scripts are archived under `Programs/old/survival_refactor_20260623/`.

The default group unit is `ingredient_route_form`, normalized nonproprietary name plus route plus dosage form. Sensitivity runs can set `GROUP_PANEL_UNIT=ingredient_form` or `GROUP_PANEL_UNIT=ingredient`.

## Canonical commands

Run the full data and model pipeline:

```powershell
python Programs\28_pipeline.py
```

Run only the model from the assembled panel:

```powershell
python Programs\28_pipeline.py --from-models
```

After a model rerun, refresh the manuscript uncertainty and supporting exhibits:

```powershell
python Programs\26_bootstrap_ci.py
python Programs\27_supporting_analyses.py
python Manuscripts\build_exhibits.py
```

Run the all-architecture no-prescription onset sensitivity:

```powershell
python Programs\25_survival_benchmarks.py --task onset --architectures logistic,logistic_time_only,lgbm,lgbm_focal,transformer --tune --val-start 2022-01 --val-end 2022-12 --rolling-start 2023-01 --rolling-end 2025-08 --retrain-every 12 --output-root Data\analysis\onset_without_licensed_prescription_full --skip-shap --feature-scenario without_licensed_prescription --transformer-epochs 12 --cpu-transformer
```

Run the primary logistic shortage-history sensitivity:

```powershell
python Programs\25_survival_benchmarks.py --task onset --architectures logistic --tune --val-start 2022-01 --val-end 2022-12 --rolling-start 2023-01 --rolling-end 2025-08 --retrain-every 12 --output-root Data\analysis\feature_scenario_runs\without_shortage_history_logistic --skip-shap --feature-scenario without_shortage_history --transformer-epochs 12
```

These maintained sensitivity commands retune within the reduced feature set on the same 2022 temporal validation carve-out used by the canonical benchmark.

The PowerShell wrapper remains only for compatibility and calls the same Python orchestrator:

```powershell
Programs\run_full_rebuild.ps1
```

## Active program map

Data build:

- `Programs/00_utilities.py`
- `Programs/01_build_drug_universe.py`
- `Programs/02_build_shortage_outcome.py`
- `Programs/03_drug_characteristics.py`
- `Programs/04_market_structure.py`
- `Programs/04b_mergers.py`
- `Programs/04c_pricing.py`
- `Programs/04d_utilization.py`
- `Programs/04e_recalls.py`
- `Programs/04f_inspections.py`
- `Programs/04g_adverse_events.py`
- `Programs/04h_api_sourcing.py`
- `Programs/04i_asp_pricing.py`
- `Programs/05_warning_letters.py`
- `Programs/06_geographic_disasters.py`
- `Programs/07_patents_exclusivity.py`
- `Programs/08_assemble_panel.py`
- `Programs/09_validate.py`

Model and orchestration:

- `Programs/25_survival_benchmarks.py`, the canonical onset and offset survival benchmark analysis (five model rows, temporal walk-forward, optional `--tune` carve-out hyperparameter selection, and maintained `--feature-scenario` sensitivity runs)
- `Programs/26_bootstrap_ci.py`, cluster-bootstrap confidence intervals over the saved predictions
- `Programs/27_supporting_analyses.py`, heterogeneity tables, top-risk rankings, Harrell concordance, and permutation importance
- `Programs/23_onset_group_benchmark_enhanced.py`, retained as the shared group-panel and feature builder (imported, not run directly)
- `Programs/28_pipeline.py`, the canonical orchestrator
- `Programs/make_variable_dictionary.py`, variable dictionary generator

Downloaders remain separate because they depend on external source refreshes and credentials.

## Canonical outputs

The assembled NDC-month panel is `Data/analysis/drug_shortage_panel.parquet`, and the cached grouped survival panel is `Data/analysis/survival_grouped_panel.parquet` (rebuilt automatically when the NDC panel is newer).

Model artifacts are written per task to `Data/analysis/survival_onset/` and `Data/analysis/survival_offset/`:

- `pooled_metrics.csv`, out-of-time metrics by architecture
- `origin_metrics.csv`, per-origin metrics by architecture
- `predictions_<architecture>.parquet`, pooled out-of-time forecasts with drug identifiers, raw `pred`, and blockwise calibrated `pred_calibrated`
- `calibration.csv`, decile calibration by architecture
- `shap_features.csv` and `shap_domains.csv`, full-data LightGBM TreeSHAP by feature and domain
- `best_hparams.json`, selected 2022 validation-window hyperparameters by architecture
- `feature_list_<scenario>.json`, the final screened feature list and feature-list checksum for the run
- `bootstrap_ci.csv`, cluster-bootstrap confidence intervals
- `harrell_c.csv`, subject-level concordance at the 2023 landmark
- `permutation_importance.csv`, shuffled-feature importance summaries
- `summary.json`

Panel-level descriptive tables are in `Data/analysis/survival_descriptives/`:

- `variables_by_domain.csv`, all model features grouped by domain
- `feature_missingness.csv`, post-fill prevalence and residual missingness for every feature
- `event_frequency_by_month.csv`, onset and offset counts by calendar month

The current-shortage indicator and imputed shortage-end flag are excluded from model features. The offset task censors spells whose ASHP record was resolved but had no parseable resolution date, and records the censored spell and row counts in `summary.json`.

The current no-prescription sensitivity writes to `Data/analysis/onset_without_licensed_prescription_full/survival_onset/`. It removes licensed Symphony Health prescription features and reruns all five onset model rows with its own 2022 validation-window hyperparameter selection.

The current shortage-history sensitivity used in the manuscript writes to `Data/analysis/feature_scenario_runs/without_shortage_history_logistic/survival_onset/`. It removes same-group and related-drug shortage-record features, including generated lag, change, rolling, and shortage-recency variants, and reruns the logistic onset model with its own 2022 validation-window hyperparameter selection.

The supporting-analysis and exhibit builders also write monthly operational-yield, post-hoc temporal calibration, and 6-month onset horizon tables from the saved prediction files. These outputs are in `Manuscripts/generated/tables/tableA12_operational_yield.csv`, `Manuscripts/generated/tables/tableA13_posthoc_calibration.csv`, `Manuscripts/generated/tables/tableA14_horizon6_onset.csv`, `Data/analysis/posthoc_calibration_sensitivity/metrics.csv`, and `Data/analysis/survival_onset/horizon6_metrics.csv`. The 6-month table evaluates the existing one-month onset hazard against first onset in the next six observed months; it is not a horizon-retuned model.

## Current run status

The canonical analysis is the onset and offset survival benchmarks in `Programs/25_survival_benchmarks.py`, run through `python Programs\28_pipeline.py --from-models` or directly. Metrics live in `Data/analysis/survival_onset/pooled_metrics.csv` and `Data/analysis/survival_offset/pooled_metrics.csv` (concordance, AUROC, AUPRC, Brier, ECE, and precision/recall at fixed budgets, by architecture). Do not copy metrics into this file, they go stale, read them from those outputs.

The prior status-risk run and its metrics are superseded. The archived modeling scripts and archived `shortage_risk_results*` directories reflect that old design and are not part of the active pipeline.

## Archived code

Superseded and stale analyses are retained under `Programs/old/`, `Data/analysis/old/`, `Manuscripts/old/`, and `Manuscripts/generated/old/`. Current archive locations include:

- `Programs/old/refactor_20260618/`
- `Programs/old/survival_refactor_20260623/`
- `Programs/old/sensitivity_helpers_archived_20260630/`
- `Data/analysis/old/status_risk_outputs_archived_20260630/`
- `Manuscripts/generated/old/status_study_archived_20260630/`
- `Manuscripts/old/survival_docx_scaffold_archived_20260630/`

Archived scripts are kept for provenance. They are not part of the active pipeline unless explicitly restored.

## Manuscript artifacts

The manuscript and supplement are maintained by direct, surgical edits to the Word files. The earlier generator scripts are archived under `Manuscripts/old/builder_archived_20260622/` and `Manuscripts/old/survival_docx_scaffold_archived_20260630/`. Do not regenerate the Word documents from archived builders.

Edit the Word files in place:

- `Manuscripts/generated/NEJM_AI_manuscript 7.2.2026.docx`
- `Manuscripts/generated/NEJM_AI_supplement 7.2.2026.docx`

To make precise text changes, unpack the document, edit `word/document.xml`, and repack, preserving the rendered numbers and the superscript citation runs.

After any edit, check the documents with:

```powershell
python Manuscripts\check_docs.py
python Manuscripts\verify_submission_artifacts.py
```

The preflight report checks the active Word files, current survival outputs, current manuscript tables and figures, structured abstract headings, punctuation rules, superscript citation runs, the local five-exhibit cap, local abstract and main-text word limits, and non-failing advisories for unresolved author-input placeholders. It also writes `Manuscripts/generated/tables/submission_placeholder_audit.csv` with document, paragraph or table-cell location, and context for each unresolved placeholder.

Current NEJM AI initial-submission basis is Original Article, 300-word structured abstract, 3000-word main text, 1-2 sentence description, and up to 5 tables and figures.

Outputs:

- `Manuscripts/generated/NEJM_AI_manuscript 7.2.2026.docx` (local submission artifact, ignored by Git)
- `Manuscripts/generated/NEJM_AI_supplement 7.2.2026.docx` (local submission artifact, ignored by Git)
- `Manuscripts/generated/NEJM AI Cover Letter 7.2.2026.docx` (local submission artifact, ignored by Git)
- `Manuscripts/generated/submission_preflight_report.json` (local validation output, ignored by Git)
- `Manuscripts/generated/tables/`
- `Manuscripts/generated/figures/`

Key active generated tables now include:

- `Manuscripts/generated/tables/table1_variables_by_domain.csv`
- `Manuscripts/generated/tables/table2_onset_performance.csv`
- `Manuscripts/generated/tables/table3_offset_performance.csv`
- `Manuscripts/generated/tables/tableA1_generic_vs_brand.csv`
- `Manuscripts/generated/tables/tableA2_top_therapeutic_classes.csv`
- `Manuscripts/generated/tables/tableA3_logistic_feature_sensitivity.csv`
- `Manuscripts/generated/tables/tableA3_top20_onset_risk.csv`
- `Manuscripts/generated/tables/tableA4_top20_offset_resolve.csv`
- `Manuscripts/generated/tables/tableA5_feature_missingness.csv`
- `Manuscripts/generated/tables/tableA5_feature_missingness.xlsx`
- `Manuscripts/generated/tables/tableA8_feature_missingness_summary.csv`
- `Manuscripts/generated/tables/tableA11_no_prescription_benchmark.csv`
- `Manuscripts/generated/tables/tableA12_operational_yield.csv`
- `Manuscripts/generated/tables/tableA13_posthoc_calibration.csv`
- `Manuscripts/generated/tables/tableA14_horizon6_onset.csv`

Key active generated figure files include PNG, TIFF, PDF, and SVG versions of
`fig1_onset_shap`, `fig2_offset_shap`, their full top-25 supplemental versions,
`figA1_event_frequency`, and `figA2_calibration`. The main manuscript embeds
the top-15 SHAP figures; the full top-25 files are separate supplemental figure
assets. TIFF files are local submission exports and are ignored by Git because
of size.

## Methodological work still open

- Missingness audit status: the current grouped survival feature table reports residual post-fill missingness and prevalence after structural indicators and prespecified fills. It is not a raw pre-fill missingness table. The current audit outputs are `Data/analysis/survival_descriptives/feature_missingness.csv`, `Manuscripts/generated/tables/tableA5_feature_missingness.csv`, `Manuscripts/generated/tables/tableA5_feature_missingness.xlsx`, and the compact Word-table summary `Manuscripts/generated/tables/tableA8_feature_missingness_summary.csv`.
- Highest remaining source-coverage issues are ASP pricing fields and selected payer-coverage indicators. These are documented as structural source-coverage limits rather than silently imputed values.
- Continue improving covariate missingness handling where source coverage can be distinguished more cleanly from true zero values.
- Repackagers are retained by default and flagged with `is_repackager`. The old exclusion was run as a sensitivity analysis and did not materially improve new-onset prediction.
- Integrate Symphony Health only through an approved export or API covered by the data-use terms. Do not scrape restricted systems unless the license explicitly permits it.
- NEJM AI initial-submission instructions supplied by the author on 2026-06-19 have been incorporated into the manuscript verifier and administrative tables. Manually recheck the official author instructions before final upload in case journal policies change. The local administrative checklist is `Manuscripts/generated/tables/submission_readiness_checklist.csv`, with author-input fields in `Manuscripts/generated/tables/submission_admin_fields.csv`.
- Keep code-availability wording aligned with this repository and the licensed-data redistribution limits in `DATA_AVAILABILITY.md`.

## Publication plan

The manuscript reports two discrete-time survival analyses, onset and offset, each comparing logistic regression, a time-only logistic baseline, LightGBM, LightGBM with focal loss, and a time-aware transformer under temporal walk-forward forecasting. The contribution is a fair, like-for-like comparison of what public administrative sources plus licensed prescription data can forecast for the two clinically meaningful transitions, entering shortage and leaving shortage.

### Main manuscript tables and figures

1. Table 1. Predictor domains and named feature families used by the models.
2. Table 2. Onset survival performance, all metrics by architecture.
3. Table 3. Offset survival performance, all metrics by architecture.
4. Figure 1. Onset SHAP, panel A top 15 features, panel B top variable domains.
5. Figure 2. Offset SHAP, panel A top 15 features, panel B top variable domains.

### Appendix

1. Appendix Table 1. Data timing and leakage controls by source.
2. Appendix Table 2. Model configuration and software.
3. Appendix Table 3. Logistic feature-exclusion sensitivity in the current onset model.
4. Appendix Table 4. Performance on generic versus brand drug groups.
5. Appendix Table 5. Performance in the top five named therapeutic classes, excluding placeholder and unclassified class labels before class selection.
6. Appendix Table 6. Top 20 drugs not currently in shortage at greatest retrospective onset risk.
7. Appendix Table 7. Top 20 drugs in shortage most likely to resolve.
8. Appendix Table 8. Compact summary of post-fill feature prevalence and source-coverage indicators. Full feature-level table is provided as machine-readable CSV and Excel.
9. Appendix Table 9. Subject-level survival concordance.
10. Appendix Table 10. Permutation importance for the LightGBM onset model.
11. Appendix Table 11. No-prescription onset benchmark after removing licensed prescription features, rerun for all five onset model rows.
12. Appendix Table 12. Monthly operational yield at fixed review budgets.
13. Appendix Table 13. Post-hoc temporal calibration sensitivity.
14. Appendix Table 14. Existing onset hazard scores evaluated as a six-month watchlist.
15. Appendix Figure 1. Frequency of shortage onsets and offsets by calendar time.
16. Appendix Figure 2. Calibration of the forecasts.

Subgroup tables (generic versus brand, therapeutic class), the top-risk rankings, and the feature-exclusion sensitivity tables are built from the per-architecture prediction files and maintained sensitivity outputs. The variables-by-domain, post-fill missingness, and event-frequency tables come from `Data/analysis/survival_descriptives/`.

### Code architecture target

Keep the active path small:

- data-source builders stay as separate scripts because they correspond to separate public data sources
- `08_assemble_panel.py` is the only NDC-month assembly script
- `09_validate.py` is the only panel validation script
- `25_survival_benchmarks.py` is the only manuscript modeling script
- `28_pipeline.py` is the only full-pipeline orchestrator
- the manuscript and supplement are maintained by direct edits to the Word files under `Manuscripts/generated/`, not by a generator script

Do not add new exploratory analysis scripts unless the result will become a maintained manuscript artifact.
