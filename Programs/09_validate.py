"""
09_validate.py - Data quality checks and summary statistics.

Validates the assembled panel for completeness, consistency,
and expected patterns.

Output: Prints validation report to console.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("utilities", Path(__file__).parent / "00_utilities.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})

PANEL_PATH = Path(os.environ.get("PANEL_PATH", str(ANALYSIS / "drug_shortage_panel.parquet")))


def main():
    print("=" * 70)
    print("09_validate.py - Panel Validation Report")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load panel
    # ------------------------------------------------------------------
    print("\n[1] Loading panel...")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  Source: {PANEL_PATH}")
    print(f"  Shape: {panel.shape}")
    print(f"  Rows: {len(panel):,}")
    print(f"  Columns: {panel.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Basic checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[2] BASIC CHECKS")
    print("=" * 70)

    # Unique counts
    n_ndcs = panel['ndc_11'].nunique()
    n_months = panel['year_month'].nunique()
    print(f"  Unique NDCs: {n_ndcs:,}")
    print(f"  Unique months: {n_months}")
    print(f"  Expected max rows (NDCs x months): {n_ndcs * n_months:,}")
    print(f"  Actual rows: {len(panel):,}")
    print(f"  Fill rate: {len(panel) / (n_ndcs * n_months):.1%}")

    # Date range
    print(f"\n  Date range: {panel['year_month'].min()} to {panel['year_month'].max()}")

    # Duplicates
    dupes = panel.duplicated(subset=['ndc_11', 'year_month']).sum()
    print(f"  Duplicate NDC-month rows: {dupes:,}")
    if dupes > 0:
        print("  *** WARNING: Duplicates found! ***")

    # ------------------------------------------------------------------
    # 3. Row counts per month
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[3] ROW COUNTS PER MONTH")
    print("=" * 70)
    monthly_counts = panel.groupby('year_month').size()
    print(f"  Min: {monthly_counts.min():,} ({monthly_counts.idxmin()})")
    print(f"  Max: {monthly_counts.max():,} ({monthly_counts.idxmax()})")
    print(f"  Mean: {monthly_counts.mean():,.0f}")
    print(f"  Std: {monthly_counts.std():,.0f}")
    print(f"\n  Monthly counts:")
    for ym, count in monthly_counts.items():
        print(f"    {ym}: {count:>8,}")

    # ------------------------------------------------------------------
    # 4. Shortage prevalence
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[4] SHORTAGE PREVALENCE")
    print("=" * 70)
    print(f"  Overall prevalence: {panel['shortage'].mean():.4%}")
    print(f"  Total shortage NDC-months: {panel['shortage'].sum():,}")
    print(f"  Total shortage onsets: {panel['shortage_start'].sum():,}")
    print(f"  Unique NDCs ever in shortage: {panel[panel['shortage']==1]['ndc_11'].nunique():,}")

    print(f"\n  Shortage by year:")
    if 'year' not in panel.columns:
        panel['year'] = panel['year_month'].str[:4]
    yearly = panel.groupby('year').agg(
        n_rows=('shortage', 'count'),
        n_shortage=('shortage', 'sum'),
        prevalence=('shortage', 'mean'),
        n_onsets=('shortage_start', 'sum'),
    )
    print(yearly.to_string())

    print(f"\n  Shortage by month:")
    monthly = panel.groupby('year_month').agg(
        n_shortage=('shortage', 'sum'),
        prevalence=('shortage', 'mean'),
    )
    print(monthly.to_string())

    # ------------------------------------------------------------------
    # 5. Missing value rates
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[5] MISSING VALUE RATES")
    print("=" * 70)
    null_rates = panel.isnull().mean().sort_values(ascending=False)
    for col, rate in null_rates.items():
        flag = " ***" if rate > 0.5 else ""
        print(f"  {col:45s} {rate:7.1%}{flag}")

    # ------------------------------------------------------------------
    # 6. Cross-tabs: shortage rate by key features
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[6] SHORTAGE RATE BY KEY FEATURES")
    print("=" * 70)

    # By dosage form
    if 'dosage_form' in panel.columns:
        print("\n  Shortage rate by dosage form (top 15):")
        df_cross = (
            panel.groupby('dosage_form')
            .agg(n=('shortage', 'count'), shortage_rate=('shortage', 'mean'))
            .sort_values('n', ascending=False)
            .head(15)
        )
        print(df_cross.to_string())

    # By injectable
    if 'is_injectable' in panel.columns:
        print("\n  Shortage rate: Injectable vs Non-injectable:")
        print(panel.groupby('is_injectable')['shortage'].agg(['count', 'mean']).to_string())

    # By generic vs brand
    if 'is_generic' in panel.columns:
        print("\n  Shortage rate: Generic vs Brand:")
        print(panel.groupby('is_generic')['shortage'].agg(['count', 'mean']).to_string())

    # By sole source
    if 'sole_source' in panel.columns:
        print("\n  Shortage rate: Sole source vs Multi-source:")
        ss = panel.dropna(subset=['sole_source'])
        print(ss.groupby('sole_source')['shortage'].agg(['count', 'mean']).to_string())

    # By number of manufacturers
    if 'n_manufacturers' in panel.columns:
        print("\n  Shortage rate by manufacturer count:")
        panel['mfr_bin'] = pd.cut(panel['n_manufacturers'].fillna(0),
                                   bins=[0, 1, 2, 3, 5, 10, 100],
                                   labels=['1', '2', '3', '4-5', '6-10', '11+'])
        print(panel.groupby('mfr_bin')['shortage'].agg(['count', 'mean']).to_string())

    # By warning letter
    if 'warning_letter_12m' in panel.columns:
        print("\n  Shortage rate: Warning letter in 12m:")
        wl_flag = (panel['warning_letter_12m'] > 0).astype(int)
        print(panel.groupby(wl_flag)['shortage'].agg(['count', 'mean']).to_string())

    # By domestic vs foreign
    if 'is_domestic' in panel.columns:
        print("\n  Shortage rate: Domestic vs Foreign manufacturer:")
        dom = panel.dropna(subset=['is_domestic'])
        print(dom.groupby('is_domestic')['shortage'].agg(['count', 'mean']).to_string())

    # ------------------------------------------------------------------
    # 7. Look-ahead bias check
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[7] LOOK-AHEAD BIAS CHECK")
    print("=" * 70)

    # Check that lagged features only use past information
    if 'shortage_lag1m' in panel.columns:
        # The first month for each NDC should have lag = 0 (no prior info)
        first_months = panel.groupby('ndc_11')['year_month'].min().reset_index()
        first_months.columns = ['ndc_11', 'first_ym']
        panel_check = panel.merge(first_months, on='ndc_11')
        first_obs = panel_check[panel_check['year_month'] == panel_check['first_ym']]
        lag_nonzero = (first_obs['shortage_lag1m'] != 0).sum()
        print(f"  First-month observations with non-zero lag: {lag_nonzero:,}")
        if lag_nonzero > 0:
            print("  *** WARNING: Possible look-ahead bias in lagged shortage ***")
        else:
            print("  OK: Lagged shortage is zero for first observations")

    # Empirical lag checks. The recall/inspection/adverse-event features are
    # built on a monthly grid that starts at the panel start and shift by at
    # least one month before rolling, so they MUST be zero in the first panel
    # month. A nonzero value there means a window includes month t or future
    # months. (Warning letters are different: they are computed from raw
    # letter dates that predate the panel, so first-month values may
    # legitimately be positive and are reported as informational only.)
    first_panel_month = panel['year_month'].min()
    first_rows = panel[panel['year_month'] == first_panel_month]
    strict_zero_cols = [
        'recall_count_12m', 'recall_count_24m', 'class1_recall_12m',
        'oai_inspection_12m', 'vai_inspection_12m', 'inspection_count_24m',
        'ae_reports_3m', 'ae_reports_12m',
    ]
    any_lag_failure = False
    for col in strict_zero_cols:
        if col not in panel.columns:
            continue
        n_bad = int((first_rows[col].fillna(0) > 0).sum())
        status = "OK" if n_bad == 0 else "*** FAIL: window includes month t ***"
        if n_bad > 0:
            any_lag_failure = True
        print(f"  {col:30s} nonzero in {first_panel_month}: {n_bad:>6,}  {status}")
    for col in ['warning_letter_6m', 'warning_letter_12m', 'warning_letter_24m']:
        if col in panel.columns:
            n_nonzero = int((first_rows[col].fillna(0) > 0).sum())
            print(f"  {col:30s} nonzero in {first_panel_month}: {n_nonzero:>6,}  "
                  f"(informational; raw letters predate the panel)")
    if any_lag_failure:
        print("  *** WARNING: grid-based trailing windows show history at panel start.")
        print("      Check the shift(1)/shift(3) in 04e/04f/04g. ***")
    else:
        print("  OK: grid-based trailing event features are empty at panel start")

    # ------------------------------------------------------------------
    # 8. Summary statistics table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[8] SUMMARY STATISTICS")
    print("=" * 70)

    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    stats = panel[numeric_cols].describe().T
    stats['missing'] = panel[numeric_cols].isnull().mean()
    print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing']].to_string())

    # ------------------------------------------------------------------
    # 9. Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[9] FINAL VERDICT")
    print("=" * 70)

    issues = []
    if dupes > 0:
        issues.append(f"Found {dupes:,} duplicate NDC-month rows")
    if panel['shortage'].mean() > 0.50:
        issues.append(f"Shortage prevalence ({panel['shortage'].mean():.1%}) seems too high")
    if panel['shortage'].mean() == 0:
        issues.append("No shortages found!")
    if n_ndcs < 10000:
        issues.append(f"Only {n_ndcs:,} unique NDCs (expected ~50K+)")
    if n_months < 60:
        issues.append(f"Only {n_months} months (expected ~69)")
    if any_lag_failure:
        issues.append("One or more lagged event windows contain information in the first panel month")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All checks passed!")

    print(f"\n  Panel ready for modeling: {PANEL_PATH}")
    print(f"  {len(panel):,} rows x {panel.shape[1]} columns")

    print("\nValidation complete!")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
