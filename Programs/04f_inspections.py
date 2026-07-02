"""
04f_inspections.py - Build FDA inspection features for the drug shortage panel.

Reads FDA inspection classification data, matches to panel labelers
via fuzzy name matching, and creates rolling inspection features.

Output: Data/intermediate/inspections.parquet
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


def _find_inspection_input():
    """Locate a likely FDA inspections export in Raw Data/FDA Inspections."""
    insp_dir = RAW_DATA / "FDA Inspections"
    candidates = [
        insp_dir / "fda_inspections_dashboard_export.xlsx",
        insp_dir / "fda_inspections_2018_2025.csv",
        insp_dir / "fda_inspections_2018_2026.csv",
        insp_dir / "inspections.csv",
        insp_dir / "inspection_classifications.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    patterns = ("*inspection*.csv", "*inspect*.csv", "*inspection*.xlsx", "*inspect*.xlsx", "*inspection*.xls", "*inspect*.xls")
    matches = []
    for pattern in patterns:
        matches.extend(insp_dir.glob(pattern))
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def main():
    print("=" * 70)
    print("04f_inspections.py - Building FDA inspection features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load inspection data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading FDA inspection data...")
    insp_file = _find_inspection_input()

    if insp_file is None:
        print(f"  WARNING: No inspection file found in {RAW_DATA / 'FDA Inspections'}.")
        print("  Run download_fda_inspections.py first, or manually download the data.")
        print("  Creating empty inspection features...")
        _create_empty_output()
        return

    print(f"  Using input file: {insp_file.name}")
    if insp_file.suffix.lower() in ('.xlsx', '.xls'):
        inspections = pd.read_excel(insp_file, dtype=str)
    else:
        inspections = pd.read_csv(insp_file, dtype=str)
    inspections.columns = inspections.columns.str.strip()
    print(f"  Loaded {len(inspections):,} inspection records")

    if len(inspections) == 0:
        print("  No records in file. Creating empty output.")
        _create_empty_output()
        return

    def find_column(candidates):
        lookup = {str(col).strip().lower(): col for col in inspections.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in lookup:
                return lookup[key]
        return None

    # ------------------------------------------------------------------
    # 2. Parse dates, normalize firm names
    # ------------------------------------------------------------------
    print("\n[2/5] Parsing dates and normalizing firm names...")

    # Try multiple date column names
    date_col = find_column([
        'inspection_end_date', 'InspectionEndDate', 'INSPECTION_END_DATE',
        'Inspection End Date',
    ])

    if date_col is None:
        print("  ERROR: No inspection end date column found.")
        print(f"  Available columns: {list(inspections.columns)}")
        _create_empty_output()
        return

    product_type_col = find_column(['product_type', 'ProductType', 'PRODUCT_TYPE', 'Product Type'])
    if product_type_col is not None:
        pre_filter_n = len(inspections)
        inspections = inspections[
            inspections[product_type_col].astype(str).str.contains('drug', case=False, na=False)
        ].copy()
        print(f"  Drug inspections retained: {len(inspections):,} of {pre_filter_n:,}")

    if len(inspections) == 0:
        print("  No drug inspection records found after filtering.")
        _create_empty_output()
        return

    inspections['insp_date'] = pd.to_datetime(inspections[date_col], format='mixed', errors='coerce')
    inspections['year_month'] = inspections['insp_date'].dt.to_period('M').astype(str)
    inspections = inspections.dropna(subset=['year_month'])
    inspections = inspections[inspections['year_month'] >= '2018-01']

    # Firm name
    firm_col = find_column([
        'firm_name', 'FirmName', 'FIRM_NAME',
        'Legal Name',
    ])

    if firm_col is None:
        print("  ERROR: No firm name column found.")
        _create_empty_output()
        return

    inspections['firm_normalized'] = inspections[firm_col].apply(normalize_company_name)

    # Classification
    class_col = find_column(['classification', 'Classification', 'CLASSIFICATION'])

    if class_col:
        inspections['is_oai'] = inspections[class_col].str.upper().str.contains('OAI', na=False).astype(int)
        inspections['is_vai'] = inspections[class_col].str.upper().str.contains('VAI', na=False).astype(int)
    else:
        inspections['is_oai'] = 0
        inspections['is_vai'] = 0

    print(f"  Records with valid dates: {len(inspections):,}")
    if class_col:
        print(f"  Classification distribution:")
        print(inspections[class_col].value_counts().to_string(header=False))

    # ------------------------------------------------------------------
    # 3. Match to panel labelers
    # ------------------------------------------------------------------
    print("\n[3/5] Matching to panel labelers...")

    skeleton = pd.read_parquet(INTERMEDIATE / "panel_skeleton.parquet",
                               columns=['ndc_11', 'LABELERNAME'])
    skeleton['labeler_code'] = skeleton['ndc_11'].apply(labeler_code_from_ndc)
    labelers = skeleton[['labeler_code', 'LABELERNAME']].drop_duplicates(subset=['labeler_code'])
    labelers['labeler_normalized'] = labelers['LABELERNAME'].apply(normalize_company_name)
    labelers = labelers[labelers['labeler_normalized'] != '']

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        print("  ERROR: rapidfuzz not installed. pip install rapidfuzz")
        _create_empty_output()
        return

    unique_firms = inspections['firm_normalized'].unique()
    labeler_names = labelers['labeler_normalized'].tolist()
    labeler_codes = labelers['labeler_code'].tolist()

    print(f"  Matching {len(unique_firms)} inspection firms to {len(labeler_names)} labelers...")
    firm_to_labeler = {}
    for firm in unique_firms:
        if not firm:
            continue
        result = process.extractOne(firm, labeler_names, scorer=fuzz.token_sort_ratio, score_cutoff=85)
        if result:
            matched_name, score, idx = result
            firm_to_labeler[firm] = labeler_codes[idx]

    print(f"  Matched {len(firm_to_labeler)} of {len(unique_firms)} firms")

    inspections['labeler_code'] = inspections['firm_normalized'].map(firm_to_labeler)
    matched = inspections.dropna(subset=['labeler_code'])
    print(f"  Inspection records with labeler match: {len(matched):,}")

    # ------------------------------------------------------------------
    # 4. Compute rolling features per labelerxmonth
    # ------------------------------------------------------------------
    print("\n[4/5] Computing rolling inspection features...")

    all_months = generate_year_months(STUDY_START, STUDY_END)
    all_labelers = labelers['labeler_code'].unique()

    # Inspection events by labelerxmonth
    insp_events = matched.groupby(['labeler_code', 'year_month']).agg(
        n_inspections=('firm_normalized', 'count'),
        any_oai=('is_oai', 'max'),
        any_vai=('is_vai', 'max'),
    ).reset_index()

    # Full grid
    grid = pd.MultiIndex.from_product([all_labelers, all_months],
                                       names=['labeler_code', 'year_month'])
    result = pd.DataFrame(index=grid).reset_index()
    result = result.merge(insp_events, on=['labeler_code', 'year_month'], how='left')
    result['n_inspections'] = result['n_inspections'].fillna(0).astype(int)
    result['any_oai'] = result['any_oai'].fillna(0).astype(int)
    result['any_vai'] = result['any_vai'].fillna(0).astype(int)

    result = result.sort_values(['labeler_code', 'year_month'])

    # Rolling features. Shift by one month before rolling so the window
    # ends at month t-1: inspection outcomes are often classified and
    # published weeks after the inspection, so same-month outcomes would
    # not be reliably visible at issuance time.
    result['oai_inspection_12m'] = result.groupby('labeler_code')['any_oai'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).max()
    ).fillna(0).astype(int)

    result['vai_inspection_12m'] = result.groupby('labeler_code')['any_vai'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).max()
    ).fillna(0).astype(int)

    result['inspection_count_24m'] = result.groupby('labeler_code')['n_inspections'].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).sum()
    ).fillna(0).astype(int)

    result = result[['labeler_code', 'year_month', 'oai_inspection_12m',
                      'vai_inspection_12m', 'inspection_count_24m']]

    # Keep only labelers with inspection activity
    active_labelers = result[result['inspection_count_24m'] > 0]['labeler_code'].unique()
    result = result[result['labeler_code'].isin(active_labelers)]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print("\n[5/5] Saving...")
    output_path = INTERMEDIATE / "inspections.parquet"
    result.to_parquet(output_path, index=False)

    print(f"\n  Output: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  Unique labelers with inspections: {result['labeler_code'].nunique():,}")
    print(f"  OAI (12m) > 0: {(result['oai_inspection_12m'] > 0).mean():.1%}")
    print(f"  VAI (12m) > 0: {(result['vai_inspection_12m'] > 0).mean():.1%}")

    print("\nDone!")
    return result


def _create_empty_output():
    """Create an empty inspections parquet with expected schema."""
    empty = pd.DataFrame({
        'labeler_code': pd.Series(dtype=str),
        'year_month': pd.Series(dtype=str),
        'oai_inspection_12m': pd.Series(dtype=int),
        'vai_inspection_12m': pd.Series(dtype=int),
        'inspection_count_24m': pd.Series(dtype=int),
    })
    output_path = INTERMEDIATE / "inspections.parquet"
    empty.to_parquet(output_path, index=False)
    print(f"  Empty output saved to: {output_path}")


if __name__ == "__main__":
    main()
