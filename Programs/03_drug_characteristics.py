"""
03_drug_characteristics.py - Extract static/slow-changing drug features.

Builds one row per NDC with drug-level attributes from the NDC Directory
and Orange Book.

Uses both the current NDC Directory AND the panel skeleton (which contains
attributes from historical NDC snapshots) to maximize coverage.

Output: Data/intermediate/drug_characteristics.parquet
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


INJECTABLE_FORMS = ['INJECTABLE', 'INJECTION', 'INJECTION, SOLUTION',
                    'INJECTION, POWDER, LYOPHILIZED, FOR SOLUTION',
                    'INJECTION, SUSPENSION', 'INJECTION, EMULSION',
                    'INJECTION, POWDER, FOR SOLUTION',
                    'INJECTION, LIPID COMPLEX',
                    'INJECTION, SOLUTION, CONCENTRATE']


def extract_primary_pharm_class(pharm_classes):
    """Extract primary pharmacologic class from PHARM_CLASSES field."""
    if pd.isna(pharm_classes) or str(pharm_classes).strip() == '':
        return None
    classes = str(pharm_classes).split(',')
    epc = [c.strip() for c in classes if '[EPC]' in c]
    if epc:
        return epc[0].replace('[EPC]', '').strip()
    moa = [c.strip() for c in classes if '[MoA]' in c]
    if moa:
        return moa[0].replace('[MoA]', '').strip()
    return classes[0].strip()


def build_features_from_ndc_row(ndc_df):
    """Build feature columns from a DataFrame with NDC Directory columns."""
    features = pd.DataFrame()
    features['ndc_11'] = ndc_df['ndc_11'].values
    features['product_ndc'] = ndc_df.get('product_ndc', pd.Series(dtype=str)).values
    features['labeler_code'] = ndc_df['ndc_11'].apply(labeler_code_from_ndc).values

    features['dosage_form'] = ndc_df['DOSAGEFORMNAME'].fillna('UNKNOWN').values
    features['is_injectable'] = ndc_df['DOSAGEFORMNAME'].str.upper().fillna('').apply(
        lambda x: 1 if any(inj in x for inj in INJECTABLE_FORMS) else 0
    ).values
    features['route'] = ndc_df['ROUTENAME'].fillna('UNKNOWN').values
    features['is_intravenous'] = ndc_df['ROUTENAME'].fillna('').str.contains(
        'INTRAVENOUS', case=False, na=False
    ).astype(int).values

    features['marketing_category'] = ndc_df['MARKETINGCATEGORYNAME'].fillna('UNKNOWN').values
    features['is_generic'] = ndc_df['MARKETINGCATEGORYNAME'].isin(
        ['ANDA', 'NDA AUTHORIZED GENERIC']
    ).astype(int).values

    features['substance_name'] = ndc_df['SUBSTANCENAME'].fillna('').values
    features['active_ingredient_count'] = ndc_df['SUBSTANCENAME'].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0
    ).values

    features['nonproprietary_name'] = ndc_df['NONPROPRIETARYNAME'].fillna('').values
    features['proprietary_name'] = ndc_df.get('PROPRIETARYNAME', pd.Series(dtype=str)).fillna('').values
    features['labeler_name'] = ndc_df['LABELERNAME'].fillna('').values

    features['pharm_class_raw'] = ndc_df.get('PHARM_CLASSES', pd.Series(dtype=str)).fillna('').values
    features['therapeutic_class'] = ndc_df.get('PHARM_CLASSES', pd.Series(dtype=str)).apply(
        extract_primary_pharm_class
    ).values

    features['dea_schedule'] = ndc_df.get('DEASCHEDULE', pd.Series(dtype=str)).fillna('').values
    features['is_controlled'] = (features['dea_schedule'] != '').astype(int)

    # Start marketing date
    if 'STARTMARKETINGDATE' in ndc_df.columns:
        features['start_marketing_date'] = ndc_df['STARTMARKETINGDATE'].apply(parse_date_flexible).values
    else:
        features['start_marketing_date'] = pd.NaT

    # Application number
    app_col = ndc_df.get('APPLICATIONNUMBER', pd.Series(dtype=str)).fillna('')
    features['application_number'] = app_col.values
    features['appl_no'] = app_col.apply(extract_appl_no_from_ndc).values
    features['appl_type'] = app_col.apply(extract_appl_type_from_ndc).values

    features['product_type'] = ndc_df.get('PRODUCTTYPENAME', pd.Series(dtype=str)).fillna('').values
    features['strength'] = ndc_df.get('ACTIVE_NUMERATOR_STRENGTH', pd.Series(dtype=str)).fillna('').values

    return features


def main():
    print("=" * 70)
    print("03_drug_characteristics.py - Building drug characteristic features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load NDC Directory (current)
    # ------------------------------------------------------------------
    print("\n[1/5] Loading current NDC Directory...")
    prod = read_ndc_product(RAW_DATA / "NDC Directory" / "product.txt")
    pkg = read_ndc_package(RAW_DATA / "NDC Directory" / "package.txt")

    prod['product_ndc'] = prod['PRODUCTNDC'].apply(format_productndc)
    pkg['product_ndc'] = pkg['PRODUCTNDC'].apply(format_productndc)
    pkg['ndc_11'] = pkg['NDCPACKAGECODE'].apply(format_ndcpackagecode)

    ndc = pkg[['ndc_11', 'product_ndc']].merge(
        prod, on='product_ndc', how='left', suffixes=('', '_prod')
    )
    ndc = ndc.dropna(subset=['ndc_11'])
    ndc = ndc.drop_duplicates(subset=['ndc_11'], keep='first')
    print(f"  Current NDC Directory: {len(ndc):,} unique NDCs")

    # Build features from current NDC Directory
    features_current = build_features_from_ndc_row(ndc)
    current_ndcs = set(features_current['ndc_11'])
    print(f"  Features from current directory: {len(features_current):,}")

    # ------------------------------------------------------------------
    # 2. Load panel skeleton for historical NDCs
    # ------------------------------------------------------------------
    print("\n[2/5] Loading panel skeleton for historical NDCs...")
    skeleton_path = INTERMEDIATE / "panel_skeleton.parquet"
    if skeleton_path.exists():
        skel = pd.read_parquet(skeleton_path)
        # Get one row per NDC (first occurrence carries the attributes)
        skel_ndcs = skel.drop_duplicates(subset=['ndc_11'], keep='first')
        # Find NDCs not in current directory
        skel_ndcs = skel_ndcs[~skel_ndcs['ndc_11'].isin(current_ndcs)].copy()
        print(f"  Historical NDCs not in current directory: {len(skel_ndcs):,}")

        if len(skel_ndcs) > 0:
            # The skeleton has the same column names as NDC Directory
            features_historical = build_features_from_ndc_row(skel_ndcs)
            features = pd.concat([features_current, features_historical], ignore_index=True)
            print(f"  Combined features: {len(features):,} NDCs")
        else:
            features = features_current
    else:
        print("  Panel skeleton not found, using current directory only")
        features = features_current

    features = features.drop_duplicates(subset=['ndc_11'], keep='first')
    print(f"  After dedup: {len(features):,} NDCs")

    # ------------------------------------------------------------------
    # 3. Enrich with Orange Book data
    # ------------------------------------------------------------------
    print("\n[3/5] Enriching with Orange Book data...")
    ob_products = read_orange_book_products(RAW_DATA / "Orange Book Data" / "products.txt")

    # Create a lookup of Appl_No -> Approval_Date, TE_Code
    ob_lookup = ob_products.drop_duplicates(subset=['Appl_No'], keep='first')[
        ['Appl_No', 'Approval_Date', 'TE_Code', 'Applicant_Full_Name']
    ].copy()
    ob_lookup['approval_date_dt'] = ob_lookup['Approval_Date'].apply(parse_date_flexible)

    # Merge on application number
    features = features.merge(
        ob_lookup.rename(columns={'Appl_No': 'appl_no'}),
        on='appl_no', how='left'
    )
    features['ob_approval_date'] = features['approval_date_dt']
    features['te_code'] = features.get('TE_Code', pd.Series(dtype=str)).fillna('')
    features['ob_applicant'] = features.get('Applicant_Full_Name', pd.Series(dtype=str)).fillna('')

    drop_cols = ['Approval_Date', 'TE_Code', 'Applicant_Full_Name', 'approval_date_dt']
    features.drop(columns=[c for c in drop_cols if c in features.columns], inplace=True)

    # ------------------------------------------------------------------
    # 4. Also try historical Orange Book snapshots for NDCs still missing OB match
    # ------------------------------------------------------------------
    print("\n[4/5] Checking historical Orange Book snapshots for unmatched NDCs...")
    missing_ob = features['appl_no'].notna() & features['ob_approval_date'].isna()
    n_missing = missing_ob.sum()
    print(f"  NDCs with appl_no but no OB match: {n_missing:,}")

    if n_missing > 0:
        # Try historical OB snapshots
        hist_matched = 0
        for year in [2024, 2023, 2022, 2021, 2020, 2019]:
            hist_path = HIST_OB / f"ob_{year}" / "products.txt"
            alt_path = HIST_OB / f"ob_{year}_v2" / "products.txt"
            path = hist_path if hist_path.exists() else alt_path
            if not path.exists():
                continue

            try:
                hist_ob = read_orange_book_products(path)
                hist_lookup = hist_ob.drop_duplicates(subset=['Appl_No'], keep='first')[
                    ['Appl_No', 'Approval_Date', 'TE_Code', 'Applicant_Full_Name']
                ].copy()
                hist_lookup['approval_date_dt'] = hist_lookup['Approval_Date'].apply(parse_date_flexible)

                still_missing = features['appl_no'].notna() & features['ob_approval_date'].isna()
                missing_appls = features.loc[still_missing, 'appl_no'].unique()
                matches = hist_lookup[hist_lookup['Appl_No'].isin(missing_appls)]

                if len(matches) > 0:
                    match_dict = matches.set_index('Appl_No').to_dict('index')
                    for idx in features.index[still_missing]:
                        appl = features.loc[idx, 'appl_no']
                        if appl in match_dict:
                            info = match_dict[appl]
                            features.loc[idx, 'ob_approval_date'] = info['approval_date_dt']
                            if pd.notna(info.get('TE_Code')):
                                features.loc[idx, 'te_code'] = info['TE_Code']
                            if pd.notna(info.get('Applicant_Full_Name')):
                                features.loc[idx, 'ob_applicant'] = info['Applicant_Full_Name']
                            hist_matched += 1

                    print(f"    {year}: matched {len(matches):,} additional Appl_Nos")
            except Exception as e:
                print(f"    {year}: error reading - {e}")

        print(f"  Total historically matched: {hist_matched:,}")

    # ------------------------------------------------------------------
    # 5. Summary and save
    # ------------------------------------------------------------------
    print("\n[5/5] Saving...")
    output_path = INTERMEDIATE / "drug_characteristics.parquet"
    features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Shape: {features.shape}")

    print(f"\n  Dosage form distribution (top 10):")
    print(features['dosage_form'].value_counts().head(10).to_string())

    has_appl = features['appl_no'].notna().sum()
    has_ob = features['ob_approval_date'].notna().sum()
    print(f"\n  Injectable: {features['is_injectable'].sum():,} / {len(features):,} ({features['is_injectable'].mean():.1%})")
    print(f"  Generic: {features['is_generic'].sum():,} / {len(features):,} ({features['is_generic'].mean():.1%})")
    print(f"  Controlled: {features['is_controlled'].sum():,}")
    print(f"  Has appl_no: {has_appl:,} ({has_appl/len(features):.1%})")
    print(f"  Has OB match (approval date): {has_ob:,} ({has_ob/len(features):.1%})")

    print("\nDone!")
    return features


if __name__ == "__main__":
    main()
