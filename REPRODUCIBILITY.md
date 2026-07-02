# Reproducibility Notes

## Environment

The maintained run path is Python on Windows PowerShell. Install Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The transformer path runs on CPU when `--cpu-transformer` is supplied. Data-loader worker count is kept at zero for Windows compatibility.

## Main Pipeline

From the repository root:

```powershell
python Programs\28_pipeline.py
python Programs\26_bootstrap_ci.py
python Programs\27_supporting_analyses.py
python Manuscripts\build_exhibits.py
```

The two maintained onset sensitivity analyses are:

```powershell
python Programs\25_survival_benchmarks.py --task onset --architectures logistic,logistic_time_only,lgbm,lgbm_focal,transformer --tune --val-start 2022-01 --val-end 2022-12 --rolling-start 2023-01 --rolling-end 2025-08 --retrain-every 12 --output-root Data\analysis\onset_without_licensed_prescription_full --skip-shap --feature-scenario without_licensed_prescription --transformer-epochs 12 --cpu-transformer

python Programs\25_survival_benchmarks.py --task onset --architectures logistic --tune --val-start 2022-01 --val-end 2022-12 --rolling-start 2023-01 --rolling-end 2025-08 --retrain-every 12 --output-root Data\analysis\feature_scenario_runs\without_shortage_history_logistic --skip-shap --feature-scenario without_shortage_history --transformer-epochs 12
```

## Validation

After rerunning the exhibits, validate the manuscript artifact set with:

```powershell
python Manuscripts\check_docs.py
python Manuscripts\verify_submission_artifacts.py
```

The verifier checks expected aggregate files, main-table row counts, active architecture names, structured abstract headings, word limits, punctuation rules used for this submission package, exhibit count, placeholders, and citation order.

## Current Analysis Design

The active benchmark is a pair of discrete-time survival tasks:

- Onset: first observed shortage onset among drug groups not currently in shortage.
- Offset: resolution among drug groups currently in shortage.

Each task compares logistic regression, a time-only logistic baseline, LightGBM, focal-loss LightGBM, and a transformer encoder under the same temporal walk-forward validation protocol.
