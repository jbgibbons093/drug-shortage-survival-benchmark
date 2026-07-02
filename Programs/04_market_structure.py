"""
04_market_structure.py - Time-varying market competition measures.

Uses historical Orange Book snapshots to count manufacturers per product
(Ingredient + DF;Route) per year, then interpolates monthly.

Uses both the current NDC Directory AND the panel skeleton (which contains
attributes from historical NDC snapshots) to maximize NDC→product_key mapping.

For BLA products (biologics not in Orange Book), computes manufacturer counts
directly from the NDC Directory using SUBSTANCENAME+DOSAGEFORMNAME+ROUTENAME.

Output: Data/intermediate/market_structure.parquet
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


# Map of year -> Orange Book snapshot path
OB_SNAPSHOTS = {
    2019: HIST_OB / "ob_2019" / "products.txt",
    2020: HIST_OB / "ob_2020_v2" / "products.txt",
    2021: HIST_OB / "ob_2021_v2" / "products.txt",
    2022: HIST_OB / "ob_2022" / "products.txt",
    2023: HIST_OB / "ob_2023_v2" / "products.txt",
    2024: HIST_OB / "ob_2024_v2" / "products.txt",
    2025: RAW_DATA / "Orange Book Data" / "products.txt",
}


def compute_market_structure_snapshot(ob_df, year):
    """
    For one Orange Book snapshot, compute:
    - Number of unique applicants per Ingredient + DF;Route
    - Number of unique Appl_No per Ingredient + DF;Route
    """
    ob_df = ob_df.copy()
    ob_df['Ingredient'] = ob_df['Ingredient'].str.upper().str.strip()
    ob_df['DF;Route'] = ob_df['DF;Route'].str.upper().str.strip()
    ob_df['product_key'] = ob_df['Ingredient'] + '|' + ob_df['DF;Route']

    mfr_counts = (
        ob_df.groupby('product_key')['Applicant']
        .nunique()
        .reset_index()
        .rename(columns={'Applicant': 'n_manufacturers'})
    )

    app_counts = (
        ob_df.groupby('product_key')['Appl_No']
        .nunique()
        .reset_index()
        .rename(columns={'Appl_No': 'n_applications'})
    )

    applicant_sets = (
        ob_df.groupby('product_key')['Applicant']
        .apply(set)
        .reset_index()
        .rename(columns={'Applicant': 'applicant_set'})
    )

    result = mfr_counts.merge(app_counts, on='product_key', how='outer')
    result = result.merge(applicant_sets, on='product_key', how='left')
    result['year'] = year

    return result


def build_ndc_product_key_map(ndc_prod, ndc_pkg, ob_current, skeleton_path):
    """
    Build a comprehensive NDC→product_key mapping using:
    1. Current NDC Directory + Orange Book Appl_No lookup
    2. Panel skeleton for historical NDCs not in current directory
    3. Fallback to SUBSTANCENAME+DOSAGEFORMNAME;ROUTENAME for unmatched

    Returns DataFrame with ['ndc_11', 'product_key']
    """
    # --- Step 1: Current NDC Directory ---
    ndc_prod = ndc_prod.copy()
    ndc_pkg = ndc_pkg.copy()
    ndc_prod['product_ndc'] = ndc_prod['PRODUCTNDC'].apply(format_productndc)
    ndc_pkg['product_ndc'] = ndc_pkg['PRODUCTNDC'].apply(format_productndc)
    ndc_pkg['ndc_11'] = ndc_pkg['NDCPACKAGECODE'].apply(format_ndcpackagecode)

    ndc_merged = ndc_pkg[['ndc_11', 'product_ndc']].merge(
        ndc_prod[['product_ndc', 'APPLICATIONNUMBER', 'SUBSTANCENAME',
                   'DOSAGEFORMNAME', 'ROUTENAME']],
        on='product_ndc', how='left'
    )
    ndc_merged = ndc_merged.dropna(subset=['ndc_11']).drop_duplicates(subset=['ndc_11'])
    ndc_merged['appl_no'] = ndc_merged['APPLICATIONNUMBER'].apply(extract_appl_no_from_ndc)

    current_ndcs = set(ndc_merged['ndc_11'])
    print(f"  Current NDC Directory: {len(ndc_merged):,} NDCs")

    # --- Step 2: Add historical NDCs from skeleton ---
    if skeleton_path.exists():
        skel = pd.read_parquet(skeleton_path)
        skel_ndcs = skel.drop_duplicates(subset=['ndc_11'], keep='first')
        skel_ndcs = skel_ndcs[~skel_ndcs['ndc_11'].isin(current_ndcs)].copy()
        print(f"  Historical NDCs from skeleton: {len(skel_ndcs):,}")

        if len(skel_ndcs) > 0:
            skel_subset = skel_ndcs[['ndc_11', 'product_ndc', 'APPLICATIONNUMBER',
                                      'SUBSTANCENAME', 'DOSAGEFORMNAME', 'ROUTENAME']].copy()
            skel_subset['appl_no'] = skel_subset['APPLICATIONNUMBER'].apply(extract_appl_no_from_ndc)
            ndc_merged = pd.concat([ndc_merged, skel_subset], ignore_index=True)
            ndc_merged = ndc_merged.drop_duplicates(subset=['ndc_11'], keep='first')
            print(f"  Combined: {len(ndc_merged):,} NDCs")
    else:
        print("  Panel skeleton not found, using current directory only")

    # --- Step 3: Map via Orange Book Appl_No ---
    ob = ob_current.copy()
    ob['Ingredient'] = ob['Ingredient'].str.upper().str.strip()
    ob['DF;Route'] = ob['DF;Route'].str.upper().str.strip()
    ob['product_key'] = ob['Ingredient'] + '|' + ob['DF;Route']

    ob_appl_to_key = (
        ob.drop_duplicates(subset=['Appl_No'])
        [['Appl_No', 'product_key']]
        .rename(columns={'Appl_No': 'appl_no'})
    )

    ndc_merged = ndc_merged.merge(ob_appl_to_key, on='appl_no', how='left')
    ob_matched = ndc_merged['product_key'].notna().sum()
    print(f"  Matched via OB Appl_No: {ob_matched:,}")

    # --- Step 4: Also try historical OB for unmatched ---
    no_key = ndc_merged['product_key'].isna() & ndc_merged['appl_no'].notna()
    if no_key.sum() > 0:
        hist_matched = 0
        for year in [2024, 2023, 2022, 2021, 2020, 2019]:
            hist_path = HIST_OB / f"ob_{year}" / "products.txt"
            alt_path = HIST_OB / f"ob_{year}_v2" / "products.txt"
            path = hist_path if hist_path.exists() else alt_path
            if not path.exists():
                continue
            try:
                hist_ob = read_orange_book_products(path)
                hist_ob['Ingredient'] = hist_ob['Ingredient'].str.upper().str.strip()
                hist_ob['DF;Route'] = hist_ob['DF;Route'].str.upper().str.strip()
                hist_ob['product_key'] = hist_ob['Ingredient'] + '|' + hist_ob['DF;Route']
                hist_appl_to_key = (
                    hist_ob.drop_duplicates(subset=['Appl_No'])
                    [['Appl_No', 'product_key']]
                    .rename(columns={'Appl_No': 'appl_no'})
                )

                still_missing = ndc_merged['product_key'].isna() & ndc_merged['appl_no'].notna()
                missing_appls = set(ndc_merged.loc[still_missing, 'appl_no'].dropna())
                hist_matches = hist_appl_to_key[hist_appl_to_key['appl_no'].isin(missing_appls)]
                if len(hist_matches) > 0:
                    match_dict = hist_matches.set_index('appl_no')['product_key'].to_dict()
                    matched_mask = still_missing & ndc_merged['appl_no'].isin(match_dict.keys())
                    ndc_merged.loc[matched_mask, 'product_key'] = (
                        ndc_merged.loc[matched_mask, 'appl_no'].map(match_dict)
                    )
                    n = matched_mask.sum()
                    hist_matched += n
                    print(f"    OB {year}: matched {n:,} NDCs via historical Appl_No")
            except Exception as e:
                print(f"    OB {year}: error - {e}")
        print(f"  Total matched via historical OB: {hist_matched:,}")

    # --- Step 5: Fallback for remaining - use NDC fields ---
    no_key = ndc_merged['product_key'].isna()
    if no_key.any():
        ndc_merged.loc[no_key, 'product_key'] = (
            ndc_merged.loc[no_key, 'SUBSTANCENAME'].fillna('').str.upper().str.strip() + '|' +
            ndc_merged.loc[no_key, 'DOSAGEFORMNAME'].fillna('').str.upper().str.strip() + ';' +
            ndc_merged.loc[no_key, 'ROUTENAME'].fillna('').str.upper().str.strip()
        )
        n_fallback = no_key.sum()
        print(f"  Fallback to NDC fields: {n_fallback:,} NDCs")

    ndc_to_key = ndc_merged[['ndc_11', 'product_key']].dropna(subset=['product_key'])
    # Remove rows with empty product keys
    ndc_to_key = ndc_to_key[ndc_to_key['product_key'].str.strip() != '|;']
    ndc_to_key = ndc_to_key.drop_duplicates(subset=['ndc_11'])
    print(f"  Final NDC->product_key map: {len(ndc_to_key):,} NDCs")

    return ndc_to_key


def main():
    print("=" * 70)
    print("04_market_structure.py - Building market structure features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load all Orange Book snapshots
    # ------------------------------------------------------------------
    print("\n[1/6] Loading Orange Book snapshots...")
    snapshots = {}
    for year, path in OB_SNAPSHOTS.items():
        if path.exists():
            print(f"  Loading {year}: {path}")
            snapshots[year] = read_orange_book_products(path)
        else:
            alt_path = path.parent.parent / f"ob_{year}" / "products.txt"
            if alt_path.exists():
                print(f"  Loading {year} (alt): {alt_path}")
                snapshots[year] = read_orange_book_products(alt_path)
            else:
                print(f"  WARNING: No snapshot for {year}")

    print(f"  Loaded {len(snapshots)} snapshots: {sorted(snapshots.keys())}")

    # ------------------------------------------------------------------
    # 2. Compute market structure per snapshot
    # ------------------------------------------------------------------
    print("\n[2/6] Computing market structure per snapshot...")
    ms_by_year = {}
    for year, ob_df in sorted(snapshots.items()):
        ms = compute_market_structure_snapshot(ob_df, year)
        ms_by_year[year] = ms
        print(f"  {year}: {len(ms):,} products, avg {ms['n_manufacturers'].mean():.1f} manufacturers")

    # ------------------------------------------------------------------
    # 3. Build NDC -> product_key mapping (comprehensive)
    # ------------------------------------------------------------------
    print("\n[3/6] Building comprehensive NDC to product_key mapping...")
    ndc_prod = read_ndc_product(RAW_DATA / "NDC Directory" / "product.txt")
    ndc_pkg = read_ndc_package(RAW_DATA / "NDC Directory" / "package.txt")
    ob_current = snapshots.get(2025, snapshots.get(max(snapshots.keys())))
    skeleton_path = INTERMEDIATE / "panel_skeleton.parquet"

    ndc_to_key = build_ndc_product_key_map(ndc_prod, ndc_pkg, ob_current, skeleton_path)

    # ------------------------------------------------------------------
    # 4. For NDCs whose product_key isn't in OB market structure,
    #    compute manufacturer counts from the NDC directory itself
    # ------------------------------------------------------------------
    print("\n[4/6] Computing NDC-based market structure for non-OB products...")

    # Identify product_keys that appear in OB snapshots
    all_ob_keys = set()
    for ms in ms_by_year.values():
        all_ob_keys.update(ms['product_key'].unique())

    # NDCs with product_keys NOT in Orange Book (BLAs, unapproved, etc.)
    non_ob_ndcs = ndc_to_key[~ndc_to_key['product_key'].isin(all_ob_keys)]
    n_non_ob = len(non_ob_ndcs)
    print(f"  NDCs with product_key not in OB: {n_non_ob:,}")

    if n_non_ob > 0:
        # Derive labeler_code directly from NDC (first 5 digits)
        non_ob_with_labeler = non_ob_ndcs.copy()
        non_ob_with_labeler['labeler_code'] = non_ob_with_labeler['ndc_11'].apply(labeler_code_from_ndc)

        # Count unique labelers per product_key as a proxy for manufacturers
        ndc_mfr_counts = (
            non_ob_with_labeler.groupby('product_key')['labeler_code']
            .nunique()
            .reset_index()
            .rename(columns={'labeler_code': 'n_manufacturers'})
        )
        ndc_mfr_counts['n_applications'] = ndc_mfr_counts['n_manufacturers']  # approximate
        print(f"  Non-OB product_keys with manufacturer counts: {len(ndc_mfr_counts):,}")
        print(f"  Avg manufacturers for non-OB products: {ndc_mfr_counts['n_manufacturers'].mean():.1f}")

        # Add these to each year's market structure
        for year in ms_by_year:
            existing_keys = set(ms_by_year[year]['product_key'])
            new_keys = ndc_mfr_counts[~ndc_mfr_counts['product_key'].isin(existing_keys)].copy()
            new_keys['year'] = year
            new_keys['applicant_set'] = new_keys['product_key'].apply(lambda x: set())
            ms_by_year[year] = pd.concat([ms_by_year[year], new_keys], ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Interpolate monthly and merge to NDC level
    # ------------------------------------------------------------------
    print("\n[5/6] Interpolating to monthly frequency...")

    year_months = generate_year_months(STUDY_START, STUDY_END)
    available_years = sorted(ms_by_year.keys())

    monthly_records = []
    for ym in year_months:
        yr = int(ym[:4])
        # Use the latest snapshot from a PRIOR year. The year-Y snapshot can
        # reflect approvals/withdrawals from any point in year Y, so applying
        # it to months within year Y would leak within-year future events.
        # Months in the earliest available year fall back to that year's
        # snapshot (no earlier data exists).
        valid_years = [y for y in available_years if y < yr]
        if not valid_years:
            valid_years = [available_years[0]]
        snap_year = max(valid_years)

        ms_snap = ms_by_year[snap_year][['product_key', 'n_manufacturers', 'n_applications']].copy()
        ms_snap['year_month'] = ym
        monthly_records.append(ms_snap)

    ms_monthly = pd.concat(monthly_records, ignore_index=True)

    # ------------------------------------------------------------------
    # 6. Compute derived features and merge to NDC-month level
    # ------------------------------------------------------------------
    print("\n[6/6] Computing derived features...")

    ms_monthly['sole_source'] = (ms_monthly['n_manufacturers'] == 1).astype(int)

    # Detect recent entry/exit
    yearly_mfr = pd.DataFrame({
        'product_key': [],
        'year': [],
        'n_manufacturers': [],
    })
    for year, ms in sorted(ms_by_year.items()):
        chunk = ms[['product_key', 'n_manufacturers']].copy()
        chunk['year'] = year
        yearly_mfr = pd.concat([yearly_mfr, chunk], ignore_index=True)

    yearly_mfr = yearly_mfr.sort_values(['product_key', 'year'])
    yearly_mfr['prev_n_mfr'] = yearly_mfr.groupby('product_key')['n_manufacturers'].shift(1)
    yearly_mfr['mfr_change'] = yearly_mfr['n_manufacturers'] - yearly_mfr['prev_n_mfr']

    entry_exit = yearly_mfr[yearly_mfr['mfr_change'].notna()][
        ['product_key', 'year', 'mfr_change']
    ].copy()

    def get_entry_exit_flags(ym, entry_exit_df):
        # Only changes visible in PRIOR-year snapshots count as "recent".
        # Including the current year's snapshot would let a change from
        # late in year Y set the flag for early months of year Y.
        yr = int(ym[:4])
        recent = entry_exit_df[(entry_exit_df['year'] >= yr - 2) & (entry_exit_df['year'] <= yr - 1)]
        flags = recent.groupby('product_key').agg(
            recent_generic_entry=('mfr_change', lambda x: int((x > 0).any())),
            recent_manufacturer_exit=('mfr_change', lambda x: int((x < 0).any())),
        ).reset_index()
        flags['year_month'] = ym
        return flags

    entry_exit_records = []
    for ym in year_months:
        flags = get_entry_exit_flags(ym, entry_exit)
        entry_exit_records.append(flags)

    if entry_exit_records:
        entry_exit_monthly = pd.concat(entry_exit_records, ignore_index=True)
        ms_monthly = ms_monthly.merge(entry_exit_monthly, on=['product_key', 'year_month'], how='left')
    ms_monthly['recent_generic_entry'] = ms_monthly.get('recent_generic_entry', 0).fillna(0).astype(int)
    ms_monthly['recent_manufacturer_exit'] = ms_monthly.get('recent_manufacturer_exit', 0).fillna(0).astype(int)

    # Now merge to NDC level
    ndc_ms = ndc_to_key.merge(ms_monthly, on='product_key', how='inner')
    keep_cols = ['ndc_11', 'year_month', 'n_manufacturers', 'n_applications',
                 'sole_source', 'recent_generic_entry', 'recent_manufacturer_exit']
    ndc_ms = ndc_ms[keep_cols].drop_duplicates(subset=['ndc_11', 'year_month'])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = INTERMEDIATE / "market_structure.parquet"
    ndc_ms.to_parquet(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    print(f"  Shape: {ndc_ms.shape}")
    print(f"  Unique NDCs: {ndc_ms['ndc_11'].nunique():,}")

    print(f"\n  Manufacturer count distribution:")
    print(ndc_ms.drop_duplicates('ndc_11')['n_manufacturers'].describe().to_string())

    print(f"\n  Sole source: {ndc_ms.drop_duplicates('ndc_11')['sole_source'].mean():.1%}")
    print(f"  Recent entry: {ndc_ms.drop_duplicates('ndc_11')['recent_generic_entry'].mean():.1%}")

    print("\nDone!")
    return ndc_ms


if __name__ == "__main__":
    main()
