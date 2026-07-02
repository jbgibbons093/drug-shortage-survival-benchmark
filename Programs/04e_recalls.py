"""
04e_recalls.py - Build FDA recall features for the drug shortage panel.

Reads downloaded FDA recall enforcement data, matches to panel labelers
via fuzzy name matching, and creates rolling recall count features.

Output: Data/intermediate/recalls.parquet
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
    print("04e_recalls.py - Building FDA recall features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load recall data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading FDA recall data...")
    recall_file = RAW_DATA / "FDA Recalls" / "fda_drug_recalls_2018_2025.csv"

    if not recall_file.exists():
        print(f"  WARNING: {recall_file} not found.")
        print("  Run download_fda_recalls.py first.")
        print("  Creating empty recall features...")
        _create_empty_output()
        return

    recalls = pd.read_csv(recall_file, dtype=str)
    recalls.columns = recalls.columns.str.strip()
    print(f"  Loaded {len(recalls):,} recall records")

    # ------------------------------------------------------------------
    # 2. Parse dates, normalize firm names
    # ------------------------------------------------------------------
    print("\n[2/5] Parsing dates and normalizing firm names...")

    # Parse report_date (YYYYMMDD format from openFDA)
    recalls['report_dt'] = pd.to_datetime(recalls['report_date'], format='%Y%m%d', errors='coerce')
    recalls['year_month'] = recalls['report_dt'].dt.to_period('M').astype(str)
    recalls = recalls.dropna(subset=['year_month'])
    recalls = recalls[recalls['year_month'] >= '2018-01']

    # Get the firm name for matching
    recalls['firm'] = recalls['recalling_firm'].fillna(recalls.get('initial_firm_recalling', ''))
    recalls['firm_normalized'] = recalls['firm'].apply(normalize_company_name)

    # Classification
    recalls['is_class1'] = (recalls['classification'] == 'Class I').astype(int)

    print(f"  Records with valid dates: {len(recalls):,}")
    print(f"  Date range: {recalls['year_month'].min()} to {recalls['year_month'].max()}")
    print(f"  Classification distribution:")
    print(recalls['classification'].value_counts().to_string(header=False))

    # ------------------------------------------------------------------
    # 3. Match to panel labelers
    # ------------------------------------------------------------------
    print("\n[3/5] Matching to panel labelers...")

    # Load panel skeleton for labeler codes and names
    skeleton = pd.read_parquet(INTERMEDIATE / "panel_skeleton.parquet",
                               columns=['ndc_11', 'LABELERNAME'])
    skeleton['labeler_code'] = skeleton['ndc_11'].apply(labeler_code_from_ndc)
    labelers = skeleton[['labeler_code', 'LABELERNAME']].drop_duplicates(subset=['labeler_code'])
    labelers['labeler_normalized'] = labelers['LABELERNAME'].apply(normalize_company_name)
    labelers = labelers[labelers['labeler_normalized'] != '']

    # Fuzzy match recall firms to labelers
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        print("  ERROR: rapidfuzz not installed. pip install rapidfuzz")
        _create_empty_output()
        return

    unique_firms = recalls['firm_normalized'].unique()
    labeler_names = labelers['labeler_normalized'].tolist()
    labeler_codes = labelers['labeler_code'].tolist()

    print(f"  Matching {len(unique_firms)} unique recall firms to {len(labeler_names)} labelers...")
    firm_to_labeler = {}
    for firm in unique_firms:
        if not firm:
            continue
        result = process.extractOne(firm, labeler_names, scorer=fuzz.token_sort_ratio, score_cutoff=85)
        if result:
            matched_name, score, idx = result
            firm_to_labeler[firm] = labeler_codes[idx]

    print(f"  Matched {len(firm_to_labeler)} of {len(unique_firms)} firms ({100*len(firm_to_labeler)/max(len(unique_firms),1):.1f}%)")

    recalls['labeler_code'] = recalls['firm_normalized'].map(firm_to_labeler)
    matched_recalls = recalls.dropna(subset=['labeler_code'])
    print(f"  Recall records with labeler match: {len(matched_recalls):,}")

    # ------------------------------------------------------------------
    # 4. Compute rolling features per labelerxmonth
    # ------------------------------------------------------------------
    print("\n[4/5] Computing rolling recall features...")

    # Generate all year_months for the study period
    all_months = generate_year_months(STUDY_START, STUDY_END)
    all_labelers = labelers['labeler_code'].unique()

    # Create recall events by labelerxmonth
    recall_events = matched_recalls.groupby(['labeler_code', 'year_month']).agg(
        n_recalls=('recall_number', 'nunique'),
        any_class1=('is_class1', 'max'),
    ).reset_index()

    # Build full labelerxmonth grid
    grid = pd.MultiIndex.from_product([all_labelers, all_months],
                                       names=['labeler_code', 'year_month'])
    result = pd.DataFrame(index=grid).reset_index()

    result = result.merge(recall_events, on=['labeler_code', 'year_month'], how='left')
    result['n_recalls'] = result['n_recalls'].fillna(0).astype(int)
    result['any_class1'] = result['any_class1'].fillna(0).astype(int)

    # Sort for rolling computation
    result = result.sort_values(['labeler_code', 'year_month'])

    # Rolling features (12-month and 24-month lookback). Shift by one month
    # before rolling so the window ends at month t-1: same-month recall
    # events are excluded, matching the strictly-trailing windows used by
    # the merger and warning-letter builders.
    result['recall_count_12m'] = result.groupby('labeler_code')['n_recalls'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum()
    ).fillna(0).astype(int)

    result['class1_recall_12m'] = result.groupby('labeler_code')['any_class1'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).max()
    ).fillna(0).astype(int)

    result['recall_count_24m'] = result.groupby('labeler_code')['n_recalls'].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).sum()
    ).fillna(0).astype(int)

    # Keep only rows with any recall activity (to keep file small)
    # Actually, keep all rows so panel merge works cleanly
    result = result[['labeler_code', 'year_month', 'recall_count_12m',
                      'class1_recall_12m', 'recall_count_24m']]

    # Only keep labelers that have at least one recall
    active_labelers = result[result['recall_count_24m'] > 0]['labeler_code'].unique()
    result = result[result['labeler_code'].isin(active_labelers)]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print("\n[5/5] Saving...")
    output_path = INTERMEDIATE / "recalls.parquet"
    result.to_parquet(output_path, index=False)

    print(f"\n  Output: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  Unique labelers with recalls: {result['labeler_code'].nunique():,}")
    print(f"  recall_count_12m > 0: {(result['recall_count_12m'] > 0).mean():.1%}")
    print(f"  class1_recall_12m > 0: {(result['class1_recall_12m'] > 0).mean():.1%}")

    print("\nDone!")
    return result


def _create_empty_output():
    """Create an empty recalls parquet with expected schema."""
    empty = pd.DataFrame({
        'labeler_code': pd.Series(dtype=str),
        'year_month': pd.Series(dtype=str),
        'recall_count_12m': pd.Series(dtype=int),
        'class1_recall_12m': pd.Series(dtype=int),
        'recall_count_24m': pd.Series(dtype=int),
    })
    output_path = INTERMEDIATE / "recalls.parquet"
    empty.to_parquet(output_path, index=False)
    print(f"  Empty output saved to: {output_path}")


if __name__ == "__main__":
    main()
