"""
28_pipeline.py - Automated end-to-end retraining pipeline.

Runs the active data, validation, and modeling path:
  1. Data build.
  2. Panel validation.
  3. Canonical onset and offset survival benchmarks.

The active analysis fits two separate discrete-time survival tasks:
shortage onset among drugs currently available, and shortage resolution among
drugs currently in shortage. Superseded six-month status-risk, onset-only,
neural, and standalone readiness scripts are retained for provenance under
old/ folders but are not part of the active rerun path.

Usage:
  python Programs/28_pipeline.py
  python Programs/28_pipeline.py --from-models
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = PROJECT_ROOT / "Programs"
ANALYSIS = PROJECT_ROOT / "Data" / "analysis"
LOG_DIR = ANALYSIS / "pipeline_logs"


DATA_STAGES = [
    ("01_build_drug_universe", "01_build_drug_universe.py"),
    ("02_build_shortage_outcome", "02_build_shortage_outcome.py"),
    ("03_drug_characteristics", "03_drug_characteristics.py"),
    ("04_market_structure", "04_market_structure.py"),
    ("04b_mergers", "04b_mergers.py"),
    ("04c_pricing", "04c_pricing.py"),
    ("04d_utilization", "04d_utilization.py"),
    ("04j_symphony_prescriptions", "04j_symphony_prescriptions.py"),
    ("04e_recalls", "04e_recalls.py"),
    ("04f_inspections", "04f_inspections.py"),
    ("04g_adverse_events", "04g_adverse_events.py"),
    ("04h_api_sourcing", "04h_api_sourcing.py"),
    ("04i_asp_pricing", "04i_asp_pricing.py"),
    ("05_warning_letters", "05_warning_letters.py"),
    ("06_geographic_disasters", "06_geographic_disasters.py"),
    ("07_patents_exclusivity", "07_patents_exclusivity.py"),
    ("08_assemble_panel", "08_assemble_panel.py"),
    ("09_validate", "09_validate.py"),
]

MODEL_STAGES = [
    (
        "25_survival_benchmarks",
        "25_survival_benchmarks.py",
        [
            "--task", "both",
            "--architectures", "logistic,logistic_time_only,lgbm,lgbm_focal,transformer",
            "--tune",
            "--val-start", "2022-01",
            "--val-end", "2022-12",
            "--rolling-start", "2023-01",
            "--rolling-end", "2025-08",
            "--retrain-every", "12",
            "--transformer-epochs", "12",
        ],
    ),
]

KEY_OUTPUTS = [
    ANALYSIS / "drug_shortage_panel.parquet",
    ANALYSIS / "survival_grouped_panel.parquet",
    ANALYSIS / "survival_onset" / "pooled_metrics.csv",
    ANALYSIS / "survival_onset" / "summary.json",
    ANALYSIS / "survival_offset" / "pooled_metrics.csv",
    ANALYSIS / "survival_offset" / "summary.json",
    ANALYSIS / "survival_descriptives" / "variables_by_domain.csv",
]
FROM_MODEL_INPUTS = {
    ANALYSIS / "drug_shortage_panel.parquet",
    ANALYSIS / "survival_grouped_panel.parquet",
}


def run_stage(name, script, log_fh, extra_args=None):
    """Run a single pipeline stage, return (success, elapsed_seconds)."""
    extra_args = extra_args or []
    script_path = PROGRAMS / script
    if not script_path.exists():
        msg = f"FAIL {name}: {script} not found"
        print(f"  {msg}")
        log_fh.write(f"{msg}\n")
        return False, 0.0

    print(f"  Running {name}...", end="", flush=True)
    log_fh.write(f"\n{'=' * 60}\n{name} ({script})\n{'=' * 60}\n")
    log_fh.flush()

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script_path), *extra_args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=21600,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f" TIMEOUT ({elapsed:.0f}s)")
        log_fh.write(f"\n--- TIMEOUT after {elapsed:.0f}s ---\n")
        return False, elapsed
    except Exception as exc:
        elapsed = time.time() - start
        print(f" ERROR: {exc}")
        log_fh.write(f"\n--- ERROR: {exc} ---\n")
        return False, elapsed

    elapsed = time.time() - start
    log_fh.write(result.stdout)
    if result.stderr:
        log_fh.write(f"\n--- stderr ---\n{result.stderr}")
    log_fh.flush()

    if result.returncode == 0:
        print(f" OK ({elapsed:.0f}s)")
        return True, elapsed

    print(f" FAILED (exit {result.returncode}, {elapsed:.0f}s)")
    log_fh.write(f"\n--- FAILED with exit code {result.returncode} ---\n")
    return False, elapsed


def build_stage_list(args):
    stages = []
    if not args.from_models and not args.predictions_only:
        stages.extend(DATA_STAGES)
    stages.extend(MODEL_STAGES)
    return stages


def main():
    parser = argparse.ArgumentParser(description="Run the shortage-risk pipeline")
    parser.add_argument("--from-models", action="store_true", help="Skip data build and start from modeling")
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Backward-compatible alias for --from-models",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pipeline_{timestamp}.log"
    stages = build_stage_list(args)

    print(f"Pipeline started: {datetime.now().isoformat()}")
    print(f"Log: {log_path}")
    print(f"Stages: {len(stages)}")
    print()

    results = []
    pipeline_start = time.time()
    failed = False

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"Pipeline started: {datetime.now().isoformat()}\n")
        mode = "full" if not args.from_models and not args.predictions_only else "from-models"
        log_fh.write(f"Mode: {mode}\n")
        log_fh.write(f"Stages: {len(stages)}\n\n")

        for stage in stages:
            if len(stage) == 2:
                name, script = stage
                extra_args = []
            else:
                name, script, extra_args = stage
            if failed:
                results.append({
                    "stage": name,
                    "script": script,
                    "status": "skipped",
                    "elapsed_seconds": 0.0,
                })
                continue

            success, elapsed = run_stage(name, script, log_fh, extra_args=extra_args)
            results.append({
                "stage": name,
                "script": script,
                "status": "ok" if success else "failed",
                "elapsed_seconds": round(elapsed, 1),
            })
            if not success:
                failed = True
                print("  Pipeline will skip downstream stages to avoid stale dependent outputs")

    pipeline_elapsed = time.time() - pipeline_start
    report = {
        "timestamp": timestamp,
        "mode": "full" if not args.from_models and not args.predictions_only else "from-models",
        "total_elapsed_seconds": round(pipeline_elapsed, 1),
        "total_elapsed_minutes": round(pipeline_elapsed / 60, 1),
        "stages": results,
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "n_skipped": sum(1 for r in results if r["status"] == "skipped"),
    }

    report_path = LOG_DIR / f"pipeline_{timestamp}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (LOG_DIR / "latest_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 60}")
    print(
        f"Pipeline complete: {report['n_ok']} ok, {report['n_failed']} failed, "
        f"{report['n_skipped']} skipped"
    )
    print(f"Total time: {report['total_elapsed_minutes']:.1f} minutes")
    print(f"Report: {report_path}")

    for path in KEY_OUTPUTS:
        if path.exists():
            if path.stat().st_mtime >= pipeline_start:
                status = "FRESH"
            elif (args.from_models or args.predictions_only) and path in FROM_MODEL_INPUTS:
                status = "INPUT"
            else:
                status = "STALE"
        else:
            status = "MISSING"
        print(f"  {status}: {path}")

    if report["n_failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
