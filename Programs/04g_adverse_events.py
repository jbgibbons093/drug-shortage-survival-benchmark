"""
04g_adverse_events.py - Build FDA adverse event features for the drug shortage panel.

Reads adverse event count data (by manufacturer and by NDC) from openFDA downloads,
matches to panel labelers, and creates rolling AE count features.

Output: Data/intermediate/adverse_events.parquet
"""

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


def main():
    print("=" * 70)
    print("04g_adverse_events.py - Building FDA adverse event features")
    print("=" * 70)

    ae_dir = RAW_DATA / "FDA Adverse Events"

    # ------------------------------------------------------------------
    # 1. Load AE count data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading adverse event data...")

    mfr_file = ae_dir / "ae_counts_by_manufacturer.csv"
    ndc_file = ae_dir / "ae_counts_by_ndc.csv"

    if not mfr_file.exists() and not ndc_file.exists():
        print(f"  WARNING: No AE data found. Run download_fda_adverse_events.py first.")
        _create_empty_output()
        return

    # --- Manufacturer-level AE counts ---
    has_mfr = False
    if mfr_file.exists():
        mfr_ae = pd.read_csv(mfr_file, dtype=str)
        mfr_ae.columns = mfr_ae.columns.str.strip()
        mfr_ae['ae_count'] = pd.to_numeric(mfr_ae['ae_count'], errors='coerce').fillna(0).astype(int)
        print(f"  Manufacturer AE rows: {len(mfr_ae):,}")
        has_mfr = True

    # --- NDC-level AE counts ---
    has_ndc = False
    if ndc_file.exists():
        ndc_ae = pd.read_csv(ndc_file, dtype=str)
        ndc_ae.columns = ndc_ae.columns.str.strip()
        ndc_ae['ae_count'] = pd.to_numeric(ndc_ae['ae_count'], errors='coerce').fillna(0).astype(int)
        print(f"  NDC AE rows: {len(ndc_ae):,}")
        has_ndc = True

    # ------------------------------------------------------------------
    # 2. Match manufacturer names to panel labelers
    # ------------------------------------------------------------------
    print("\n[2/4] Matching to panel labelers...")

    skeleton = pd.read_parquet(INTERMEDIATE / "panel_skeleton.parquet",
                               columns=['ndc_11', 'LABELERNAME'])
    skeleton['labeler_code'] = skeleton['ndc_11'].apply(labeler_code_from_ndc)
    labelers = skeleton[['labeler_code', 'LABELERNAME']].drop_duplicates(subset=['labeler_code'])
    labelers['labeler_normalized'] = labelers['LABELERNAME'].apply(normalize_company_name)
    labelers = labelers[labelers['labeler_normalized'] != '']

    labeler_ae_events = pd.DataFrame()

    if has_mfr:
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            print("  ERROR: rapidfuzz not installed.")
            has_mfr = False

    if has_mfr:
        mfr_ae['mfr_normalized'] = mfr_ae['manufacturer_name'].apply(normalize_company_name)
        unique_mfrs = mfr_ae['mfr_normalized'].unique()
        labeler_names = labelers['labeler_normalized'].tolist()
        labeler_codes = labelers['labeler_code'].tolist()

        print(f"  Matching {len(unique_mfrs)} AE manufacturers to {len(labeler_names)} labelers...")
        mfr_to_labeler = {}
        for mfr in unique_mfrs:
            if not mfr:
                continue
            result = process.extractOne(mfr, labeler_names,
                                        scorer=fuzz.token_sort_ratio, score_cutoff=85)
            if result:
                _, _, idx = result
                mfr_to_labeler[mfr] = labeler_codes[idx]

        print(f"  Matched {len(mfr_to_labeler)} of {len(unique_mfrs)} manufacturers")

        mfr_ae['labeler_code'] = mfr_ae['mfr_normalized'].map(mfr_to_labeler)
        matched_mfr = mfr_ae.dropna(subset=['labeler_code'])

        # Aggregate AE counts per labelerxmonth
        labeler_ae_events = matched_mfr.groupby(['labeler_code', 'year_month']).agg(
            ae_reports=('ae_count', 'sum')
        ).reset_index()

    # ------------------------------------------------------------------
    # 3. Compute rolling features
    # ------------------------------------------------------------------
    print("\n[3/4] Computing rolling AE features...")

    all_months = generate_year_months(STUDY_START, STUDY_END)
    all_labelers = labelers['labeler_code'].unique()

    grid = pd.MultiIndex.from_product([all_labelers, all_months],
                                       names=['labeler_code', 'year_month'])
    result = pd.DataFrame(index=grid).reset_index()

    if len(labeler_ae_events) > 0:
        result = result.merge(labeler_ae_events, on=['labeler_code', 'year_month'], how='left')
    result['ae_reports'] = result.get('ae_reports', pd.Series(0, index=result.index)).fillna(0).astype(int)

    result = result.sort_values(['labeler_code', 'year_month'])

    # Rolling AE counts. FAERS is published quarterly with roughly a
    # one-quarter lag, so shift the monthly series by 3 months before
    # rolling: the window for month t ends at t-3, the most recent month
    # whose reports would actually have been public at issuance time.
    FAERS_PUBLICATION_LAG_MONTHS = 3
    result['ae_reports_12m'] = result.groupby('labeler_code')['ae_reports'].transform(
        lambda x: x.shift(FAERS_PUBLICATION_LAG_MONTHS).rolling(12, min_periods=1).sum()
    ).fillna(0).astype(int)

    result['ae_reports_3m'] = result.groupby('labeler_code')['ae_reports'].transform(
        lambda x: x.shift(FAERS_PUBLICATION_LAG_MONTHS).rolling(3, min_periods=1).sum()
    ).fillna(0).astype(int)

    # AE trend: pct change in 12m rolling sum vs prior 12m
    result['ae_trend_12m'] = result.groupby('labeler_code')['ae_reports_12m'].transform(
        lambda x: x.pct_change(periods=12)
    ).clip(-10, 10).fillna(0)

    # Keep only labelers with any AE activity
    active_labelers = result[result['ae_reports_12m'] > 0]['labeler_code'].unique()
    result = result[result['labeler_code'].isin(active_labelers)]

    result = result[['labeler_code', 'year_month', 'ae_reports_3m',
                      'ae_reports_12m', 'ae_trend_12m']]

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print("\n[4/4] Saving...")
    output_path = INTERMEDIATE / "adverse_events.parquet"
    result.to_parquet(output_path, index=False)

    print(f"\n  Output: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  Unique labelers: {result['labeler_code'].nunique():,}")
    print(f"  ae_reports_12m > 0: {(result['ae_reports_12m'] > 0).mean():.1%}")

    print("\nDone!")
    return result


def _create_empty_output():
    empty = pd.DataFrame({
        'labeler_code': pd.Series(dtype=str),
        'year_month': pd.Series(dtype=str),
        'ae_reports_3m': pd.Series(dtype=int),
        'ae_reports_12m': pd.Series(dtype=int),
        'ae_trend_12m': pd.Series(dtype=float),
    })
    output_path = INTERMEDIATE / "adverse_events.parquet"
    empty.to_parquet(output_path, index=False)
    print(f"  Empty output saved to: {output_path}")


if __name__ == "__main__":
    main()
