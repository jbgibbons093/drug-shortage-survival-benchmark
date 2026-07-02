"""
07_patents_exclusivity.py - Patent protection and exclusivity features.

Computes time-varying patent counts, expiry timing, and exclusivity status
per NDC-month by linking NDCs to Orange Book patent/exclusivity records.

Output: Data/intermediate/patents_exclusivity.parquet
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
    print("07_patents_exclusivity.py - Building patent & exclusivity features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load Orange Book patents and exclusivity
    # ------------------------------------------------------------------
    print("\n[1/5] Loading Orange Book patent and exclusivity data...")
    patents = read_orange_book_patents(RAW_DATA / "Orange Book Data" / "patent.txt")
    exclusivity = read_orange_book_exclusivity(RAW_DATA / "Orange Book Data" / "exclusivity.txt")

    print(f"  Patents: {len(patents):,}")
    print(f"  Exclusivity: {len(exclusivity):,}")

    # Parse patent expiration dates
    patents['patent_expiry_dt'] = patents['Patent_Expire_Date_Text'].apply(parse_date_flexible)
    n_parsed = patents['patent_expiry_dt'].notna().sum()
    print(f"  Patents with parsed expiry date: {n_parsed:,}")

    # Parse exclusivity dates
    exclusivity['exclusivity_dt'] = exclusivity['Exclusivity_Date'].apply(parse_date_flexible)
    n_parsed_excl = exclusivity['exclusivity_dt'].notna().sum()
    print(f"  Exclusivity with parsed date: {n_parsed_excl:,}")

    # ------------------------------------------------------------------
    # 2. Build NDC -> Appl_No mapping
    # ------------------------------------------------------------------
    print("\n[2/5] Building NDC to Application Number mapping...")
    ndc_prod = read_ndc_product(RAW_DATA / "NDC Directory" / "product.txt")
    ndc_pkg = read_ndc_package(RAW_DATA / "NDC Directory" / "package.txt")

    ndc_prod['product_ndc'] = ndc_prod['PRODUCTNDC'].apply(format_productndc)
    ndc_pkg['product_ndc'] = ndc_pkg['PRODUCTNDC'].apply(format_productndc)
    ndc_pkg['ndc_11'] = ndc_pkg['NDCPACKAGECODE'].apply(format_ndcpackagecode)

    ndc_appl = ndc_pkg[['ndc_11', 'product_ndc']].merge(
        ndc_prod[['product_ndc', 'APPLICATIONNUMBER']],
        on='product_ndc', how='left'
    )
    ndc_appl = ndc_appl.dropna(subset=['ndc_11']).drop_duplicates(subset=['ndc_11'])
    ndc_appl['appl_no'] = ndc_appl['APPLICATIONNUMBER'].apply(extract_appl_no_from_ndc)
    ndc_appl['appl_type'] = ndc_appl['APPLICATIONNUMBER'].apply(extract_appl_type_from_ndc)

    ndc_appl = ndc_appl.dropna(subset=['appl_no'])
    print(f"  NDCs with application number: {len(ndc_appl):,}")

    # ------------------------------------------------------------------
    # 3. Compute patent features per application
    # ------------------------------------------------------------------
    print("\n[3/5] Computing patent features per application...")

    # Group patents by application
    patent_by_appl = (
        patents.groupby('Appl_No')
        .agg(
            total_patents=('Patent_No', 'nunique'),
            patent_expiry_dates=('patent_expiry_dt', list),
            has_substance_patent=('Drug_Substance_Flag', lambda x: (x.fillna('') == 'Y').any()),
            has_product_patent=('Drug_Product_Flag', lambda x: (x.fillna('') == 'Y').any()),
        )
        .reset_index()
        .rename(columns={'Appl_No': 'appl_no'})
    )

    print(f"  Applications with patents: {len(patent_by_appl):,}")

    # ------------------------------------------------------------------
    # 4. Compute exclusivity features per application
    # ------------------------------------------------------------------
    print("\n[4/5] Computing exclusivity features per application...")

    excl_by_appl = (
        exclusivity.groupby('Appl_No')
        .agg(
            exclusivity_dates=('exclusivity_dt', list),
            exclusivity_codes=('Exclusivity_Code', list),
            n_exclusivities=('Exclusivity_Code', 'nunique'),
        )
        .reset_index()
        .rename(columns={'Appl_No': 'appl_no'})
    )

    print(f"  Applications with exclusivity: {len(excl_by_appl):,}")

    # ------------------------------------------------------------------
    # 5. Expand to NDC-month level with time-varying features
    # ------------------------------------------------------------------
    print("\n[5/5] Expanding to NDC x month level...")
    year_months = generate_year_months(STUDY_START, STUDY_END)

    # Merge NDC -> patent/exclusivity info (inner join: only NDCs with actual data)
    ndc_patent = ndc_appl[['ndc_11', 'appl_no']].merge(patent_by_appl, on='appl_no', how='inner')
    ndc_excl = ndc_appl[['ndc_11', 'appl_no']].merge(excl_by_appl, on='appl_no', how='inner')

    print(f"  NDCs with patent data: {ndc_patent['ndc_11'].nunique():,}")
    print(f"  NDCs with exclusivity data: {ndc_excl['ndc_11'].nunique():,}")

    # Build time-varying features
    records = []
    ndcs_with_data = set(ndc_patent['ndc_11'].unique()) | set(ndc_excl['ndc_11'].unique())

    # Create lookup dicts for efficiency
    patent_lookup = ndc_patent.set_index('ndc_11').to_dict('index')
    excl_lookup = ndc_excl.set_index('ndc_11').to_dict('index')

    print(f"  Processing {len(ndcs_with_data):,} NDCs across {len(year_months)} months...")

    for ndc in ndcs_with_data:
        # Patent info
        pat_info = patent_lookup.get(ndc, {})
        total_patents = pat_info.get('total_patents', 0)
        if pd.isna(total_patents):
            total_patents = 0
        patent_dates = pat_info.get('patent_expiry_dates', [])
        if not isinstance(patent_dates, list):
            patent_dates = []
        # Filter to valid dates
        patent_dates = [d for d in patent_dates if pd.notna(d)]
        _sub = pat_info.get('has_substance_patent', False)
        has_substance = int(bool(_sub)) if not (isinstance(_sub, float) and np.isnan(_sub)) else 0
        _prod = pat_info.get('has_product_patent', False)
        has_product = int(bool(_prod)) if not (isinstance(_prod, float) and np.isnan(_prod)) else 0

        # Exclusivity info
        excl_info = excl_lookup.get(ndc, {})
        excl_dates = excl_info.get('exclusivity_dates', [])
        if not isinstance(excl_dates, list):
            excl_dates = []
        excl_dates = [d for d in excl_dates if pd.notna(d)]

        for ym in year_months:
            ym_dt = pd.Timestamp(f"{ym}-01")

            # Patent features
            active_patents = [d for d in patent_dates if d > ym_dt]
            expired_recent = [d for d in patent_dates
                             if d <= ym_dt and d >= ym_dt - pd.DateOffset(months=12)]

            patent_count = len(active_patents)
            if active_patents:
                nearest_expiry = min(active_patents)
                months_to_expiry = max(0, (nearest_expiry.year - ym_dt.year) * 12 +
                                        (nearest_expiry.month - ym_dt.month))
            else:
                months_to_expiry = -1  # No active patents

            recent_patent_expiry = 1 if expired_recent else 0

            # Exclusivity features
            active_excl = [d for d in excl_dates if d > ym_dt]
            has_active_exclusivity = 1 if active_excl else 0
            if active_excl:
                nearest_excl = min(active_excl)
                months_to_excl_end = max(0, (nearest_excl.year - ym_dt.year) * 12 +
                                          (nearest_excl.month - ym_dt.month))
            else:
                months_to_excl_end = -1

            records.append({
                'ndc_11': ndc,
                'year_month': ym,
                'patent_count': int(patent_count),
                'total_patents_ever': int(total_patents),
                'months_to_nearest_expiry': int(months_to_expiry),
                'recent_patent_expiry': recent_patent_expiry,
                'has_substance_patent': has_substance,
                'has_product_patent': has_product,
                'has_active_exclusivity': has_active_exclusivity,
                'months_to_exclusivity_end': int(months_to_excl_end),
            })

    pe_features = pd.DataFrame(records)
    print(f"  Patent/exclusivity feature rows: {len(pe_features):,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = INTERMEDIATE / "patents_exclusivity.parquet"
    pe_features.to_parquet(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    print(f"  Shape: {pe_features.shape}")
    print(f"  Unique NDCs: {pe_features['ndc_11'].nunique():,}")

    if len(pe_features) > 0:
        ndc_snap = pe_features.drop_duplicates('ndc_11')
        print(f"\n  Patent count distribution:")
        print(ndc_snap['patent_count'].describe().to_string())
        print(f"\n  Has active exclusivity: {ndc_snap['has_active_exclusivity'].mean():.1%}")
        print(f"  Recent patent expiry: {ndc_snap['recent_patent_expiry'].mean():.1%}")

    print("\nDone!")
    return pe_features


if __name__ == "__main__":
    main()
