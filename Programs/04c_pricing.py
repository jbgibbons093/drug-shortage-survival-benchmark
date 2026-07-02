"""
04c_pricing.py - Build NADAC pricing features for the drug shortage panel.

Reads NADAC CSV files (~730 MB), handles inconsistent column naming
across years, converts NDCs to panel format, and computes per-NDCxmonth
pricing features including price-to-market-median ratio.

Three strategies maximize NADAC coverage:
  1. Forward-fill within-NDC temporal gaps (NADAC prices are stable between
     weekly survey updates, so carrying last known price forward is valid)
  2. Product-level imputation for unmatched NDCs: use median price from same
     ingredient + dosage form + route in same month
  3. Include dedicated 2021 download (nadac_2021.csv) to fill the gap between
     the historical file (through mid-2021) and the 2022 file

Output: Data/intermediate/pricing.parquet
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

# ---------------------------------------------------------------------------
# Column name normalization - NADAC files have 3 naming conventions:
#   Historical (<=2021): lowercase_with_underscores
#   2022:                Title_Case_With_Underscores
#   2021/2023+:          Title Case With Spaces
# ---------------------------------------------------------------------------
NADAC_COL_MAP = {
    # NDC
    'ndc': 'NDC', 'NDC': 'NDC',
    # Price
    'nadac_per_unit': 'NADAC_Per_Unit',
    'NADAC_Per_Unit': 'NADAC_Per_Unit',
    'NADAC Per Unit': 'NADAC_Per_Unit',
    # Effective date
    'effective_date': 'Effective_Date',
    'Effective_Date': 'Effective_Date',
    'Effective Date': 'Effective_Date',
    # OTC
    'otc': 'OTC', 'OTC': 'OTC',
    # Pharmacy type
    'pharmacy_type_indicator': 'Pharmacy_Type_Indicator',
    'Pharmacy_Type_Indicator': 'Pharmacy_Type_Indicator',
    'Pharmacy Type Indicator': 'Pharmacy_Type_Indicator',
    # Classification
    'classification_for_rate_setting': 'Classification',
    'Classification_for_Rate_Setting': 'Classification',
    'Classification for Rate Setting': 'Classification',
    # Corresponding generic NADAC
    'corresponding_generic_drug_nadac_per_unit': 'Generic_NADAC',
    'Corresponding_Generic_Drug_NADAC_Per_Unit': 'Generic_NADAC',
    'Corresponding Generic Drug NADAC Per Unit': 'Generic_NADAC',
    # NDC description
    'ndc_description': 'NDC_Description',
    'NDC_Description': 'NDC_Description',
    'NDC Description': 'NDC_Description',
}


def normalize_nadac_columns(df):
    """Normalize NADAC column names to a consistent convention."""
    rename = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in NADAC_COL_MAP:
            rename[col] = NADAC_COL_MAP[col_stripped]
        elif col_stripped.lower() in {k.lower(): v for k, v in NADAC_COL_MAP.items()}:
            # Case-insensitive fallback
            for k, v in NADAC_COL_MAP.items():
                if col_stripped.lower() == k.lower():
                    rename[col] = v
                    break
    return df.rename(columns=rename)


def load_nadac_files():
    """Load and concatenate all NADAC CSV files with column normalization."""
    nadac_dir = RAW_DATA / "NADAC"
    files = sorted(nadac_dir.glob("nadac_*.csv"))
    print(f"  Found {len(files)} NADAC files")

    dfs = []
    for f in files:
        print(f"    Loading {f.name}...", end=" ")
        df = pd.read_csv(f, dtype=str)
        df.columns = df.columns.str.strip()
        df = normalize_nadac_columns(df)
        print(f"{len(df):,} rows, cols: {list(df.columns[:3])}...")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows")
    return combined


def main():
    print("=" * 70)
    print("04c_pricing.py - Building NADAC pricing features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load NADAC data
    # ------------------------------------------------------------------
    print("\n[1/9] Loading NADAC data...")
    nadac = load_nadac_files()

    # ------------------------------------------------------------------
    # 2. Filter and convert
    # ------------------------------------------------------------------
    print("\n[2/9] Filtering and converting...")

    # Filter: keep Rx only (exclude OTC='Y'), community/independent pharmacies
    if 'OTC' in nadac.columns:
        before = len(nadac)
        nadac = nadac[nadac['OTC'] != 'Y']
        print(f"  Removed {before - len(nadac):,} OTC rows")

    if 'Pharmacy_Type_Indicator' in nadac.columns:
        nadac = nadac[nadac['Pharmacy_Type_Indicator'].isin(['C/I'])]
        print(f"  After pharmacy type filter: {len(nadac):,} rows")

    # Convert NDC (11-digit no-dash)
    nadac['ndc_11'] = nadac['NDC'].apply(ndc_nodash_to_panel)
    nadac = nadac.dropna(subset=['ndc_11'])
    print(f"  Valid NDCs: {len(nadac):,} rows")

    # Parse price
    nadac['price'] = pd.to_numeric(nadac['NADAC_Per_Unit'], errors='coerce')

    # Parse corresponding generic NADAC
    if 'Generic_NADAC' in nadac.columns:
        nadac['generic_price'] = pd.to_numeric(nadac['Generic_NADAC'], errors='coerce')
    else:
        nadac['generic_price'] = np.nan

    # Parse classification for brand/generic indicator
    if 'Classification' in nadac.columns:
        nadac['is_brand_nadac'] = nadac['Classification'].str.upper().str.startswith('B').fillna(False)
    else:
        nadac['is_brand_nadac'] = False

    # Parse effective date and create year_month
    nadac['eff_date'] = pd.to_datetime(nadac['Effective_Date'], format='mixed', errors='coerce')
    nadac['year_month'] = nadac['eff_date'].dt.to_period('M').astype(str)
    nadac = nadac.dropna(subset=['year_month', 'price'])

    # Deduplicate: multiple files may overlap (e.g. historical + 2021 + 2022
    # all contain some 2021 data). Keep latest effective_date per NDCxdate.
    before_dedup = len(nadac)
    nadac = nadac.sort_values(['ndc_11', 'year_month', 'eff_date'])
    nadac = nadac.drop_duplicates(subset=['ndc_11', 'eff_date'], keep='last')
    print(f"  Deduplicated: removed {before_dedup - len(nadac):,} duplicate NDCxdate rows")

    # Filter to study period (with lookback for trends)
    nadac = nadac[nadac['year_month'] >= '2019-01']
    print(f"  After date filter (>=2019-01): {len(nadac):,} rows")

    # Report coverage by year
    print("\n  NADAC rows by year:")
    for year, grp in nadac.groupby(nadac['eff_date'].dt.year):
        print(f"    {year}: {len(grp):>9,} rows, {grp['ndc_11'].nunique():>6,} NDCs")

    # ------------------------------------------------------------------
    # 3. Take latest price per NDCxmonth (observed prices)
    # ------------------------------------------------------------------
    print("\n[3/9] Computing monthly prices (observed)...")

    nadac = nadac.sort_values(['ndc_11', 'year_month', 'eff_date'])
    monthly = nadac.groupby(['ndc_11', 'year_month']).agg(
        nadac_per_unit=('price', 'last'),
        generic_price=('generic_price', 'last'),
    ).reset_index()
    observed_keys = set(zip(monthly['ndc_11'], monthly['year_month']))

    print(f"  Observed NDCxmonth prices: {len(monthly):,}")
    print(f"  Unique NDCs with observed prices: {monthly['ndc_11'].nunique():,}")
    print(f"  Date range: {monthly['year_month'].min()} to {monthly['year_month'].max()}")

    # ------------------------------------------------------------------
    # 4. Forward-fill within-NDC temporal gaps
    # ------------------------------------------------------------------
    print("\n[4/9] Forward-filling within-NDC temporal gaps...")

    # NADAC prices are set periodically (weekly survey). Between updates,
    # the last known price remains valid. We forward-fill up to 6 months
    # to avoid carrying stale prices too far.
    all_months = generate_year_months('2019-01', STUDY_END)
    nadac_ndcs = monthly['ndc_11'].unique()

    # Find first and last observed month per NDC
    ndc_bounds = monthly.groupby('ndc_11')['year_month'].agg(['min', 'max']).reset_index()
    ndc_bounds.columns = ['ndc_11', 'first_obs', 'last_obs']

    # Create full month spine for each NDC (only within their observed window)
    spine_parts = []
    for _, row in ndc_bounds.iterrows():
        ndc_months = [m for m in all_months if row['first_obs'] <= m <= row['last_obs']]
        spine_parts.append(pd.DataFrame({
            'ndc_11': row['ndc_11'],
            'year_month': ndc_months
        }))
    spine = pd.concat(spine_parts, ignore_index=True)

    before_fill = len(monthly)
    monthly = spine.merge(monthly, on=['ndc_11', 'year_month'], how='left')
    monthly = monthly.sort_values(['ndc_11', 'year_month'])

    # Track direct NADAC observations before any carry-forward logic.
    monthly['nadac_is_observed'] = [
        int((ndc, ym) in observed_keys)
        for ndc, ym in zip(monthly['ndc_11'], monthly['year_month'])
    ]

    # Forward-fill price within each NDC, limit to 6 months
    monthly['nadac_per_unit'] = monthly.groupby('ndc_11')['nadac_per_unit'].transform(
        lambda x: x.ffill(limit=6)
    )
    monthly['generic_price'] = monthly.groupby('ndc_11')['generic_price'].transform(
        lambda x: x.ffill(limit=6)
    )

    # Mark which rows were carried forward within-NDC.
    monthly['nadac_is_ffill'] = (
        monthly['nadac_per_unit'].notna() &
        (monthly['nadac_is_observed'] == 0)
    ).astype(int)
    monthly['nadac_is_imputed'] = 0

    # Drop rows still missing after forward-fill
    monthly = monthly.dropna(subset=['nadac_per_unit'])

    filled_count = len(monthly) - before_fill
    print(f"  After forward-fill: {len(monthly):,} NDCxmonth rows")
    print(f"  Rows added by forward-fill: {filled_count:,}")

    # ------------------------------------------------------------------
    # 5. Compute NDC-level pricing features
    # ------------------------------------------------------------------
    print("\n[5/9] Computing NDC-level pricing features...")

    monthly = monthly.sort_values(['ndc_11', 'year_month'])

    # Price change features (within each NDC)
    monthly['nadac_pct_change_3m'] = monthly.groupby('ndc_11')['nadac_per_unit'].transform(
        lambda x: x.pct_change(periods=3)
    )
    monthly['nadac_pct_change_12m'] = monthly.groupby('ndc_11')['nadac_per_unit'].transform(
        lambda x: x.pct_change(periods=12)
    )

    # Cap extreme pct changes at +/- 10x
    for col in ['nadac_pct_change_3m', 'nadac_pct_change_12m']:
        monthly[col] = monthly[col].clip(-10, 10)

    # Generic ratio (brand price / generic price)
    monthly['nadac_generic_ratio'] = np.where(
        monthly['generic_price'] > 0,
        monthly['nadac_per_unit'] / monthly['generic_price'],
        np.nan
    )

    # Low price indicator - below 25th percentile within same year_month
    p25 = monthly.groupby('year_month')['nadac_per_unit'].transform('quantile', 0.25)
    monthly['nadac_is_low_price'] = (monthly['nadac_per_unit'] <= p25).astype(int)

    # ------------------------------------------------------------------
    # 6. Price-to-market-median ratio
    # ------------------------------------------------------------------
    print("\n[6/9] Computing price-to-market-median ratio...")

    # Need ingredient+dosage+route grouping from panel skeleton
    skeleton = pd.read_parquet(
        INTERMEDIATE / "panel_skeleton.parquet",
        columns=['ndc_11', 'SUBSTANCENAME', 'DOSAGEFORMNAME', 'ROUTENAME']
    )
    skeleton = skeleton.drop_duplicates(subset=['ndc_11'])

    monthly = monthly.merge(skeleton, on='ndc_11', how='left')

    # Create product group key (ingredient + dosage + route)
    monthly['product_group'] = (
        monthly['SUBSTANCENAME'].fillna('').str.upper().str.strip() + '|' +
        monthly['DOSAGEFORMNAME'].fillna('').str.upper().str.strip() + '|' +
        monthly['ROUTENAME'].fillna('').str.upper().str.strip()
    )

    # Compute median price per product_groupxmonth
    group_median = monthly.groupby(['product_group', 'year_month'])['nadac_per_unit'].transform('median')
    monthly['nadac_vs_market_median'] = np.where(
        group_median > 0,
        monthly['nadac_per_unit'] / group_median,
        np.nan
    )
    # Cap at 10x (extreme outliers)
    monthly['nadac_vs_market_median'] = monthly['nadac_vs_market_median'].clip(0, 10)

    # ------------------------------------------------------------------
    # 7. Product-level imputation for unmatched NDCs
    # ------------------------------------------------------------------
    print("\n[7/9] Product-level imputation for unmatched NDCs...")

    # Build a lookup of product_group x year_month → median price and features
    # from the observed NADAC data, to assign to NDCs with no NADAC match
    product_month_stats = monthly.groupby(['product_group', 'year_month']).agg(
        product_median_price=('nadac_per_unit', 'median'),
        product_p25=('nadac_per_unit', lambda x: x.quantile(0.25)),
        product_generic_ratio=('nadac_generic_ratio', 'median'),
    ).reset_index()

    # Compute product-group-level price trends for imputed rows
    product_month_stats = product_month_stats.sort_values(['product_group', 'year_month'])
    product_month_stats['product_pct_change_3m'] = product_month_stats.groupby(
        'product_group')['product_median_price'].transform(lambda x: x.pct_change(periods=3))
    product_month_stats['product_pct_change_12m'] = product_month_stats.groupby(
        'product_group')['product_median_price'].transform(lambda x: x.pct_change(periods=12))
    for col in ['product_pct_change_3m', 'product_pct_change_12m']:
        product_month_stats[col] = product_month_stats[col].clip(-10, 10)

    # Get all panel NDCs and their product groups
    panel_skeleton_full = pd.read_parquet(
        INTERMEDIATE / "panel_skeleton.parquet",
        columns=['ndc_11', 'year_month', 'SUBSTANCENAME', 'DOSAGEFORMNAME', 'ROUTENAME']
    )
    # Filter to study period
    panel_skeleton_full = panel_skeleton_full[panel_skeleton_full['year_month'] >= STUDY_START]

    panel_skeleton_full['product_group'] = (
        panel_skeleton_full['SUBSTANCENAME'].fillna('').str.upper().str.strip() + '|' +
        panel_skeleton_full['DOSAGEFORMNAME'].fillna('').str.upper().str.strip() + '|' +
        panel_skeleton_full['ROUTENAME'].fillna('').str.upper().str.strip()
    )

    # Find NDCxmonths NOT in the observed/forward-filled data
    observed_keys = set(zip(monthly['ndc_11'], monthly['year_month']))
    panel_keys = panel_skeleton_full[['ndc_11', 'year_month', 'product_group']].drop_duplicates()
    panel_keys['_key'] = list(zip(panel_keys['ndc_11'], panel_keys['year_month']))
    unmatched = panel_keys[~panel_keys['_key'].isin(observed_keys)].drop(columns=['_key'])
    print(f"  Unmatched NDCxmonths: {len(unmatched):,}")
    print(f"  Unmatched unique NDCs: {unmatched['ndc_11'].nunique():,}")

    # Merge product-level stats onto unmatched
    imputed = unmatched.merge(product_month_stats, on=['product_group', 'year_month'], how='inner')
    print(f"  Imputable (same product has NADAC): {len(imputed):,} ({len(imputed)/max(1,len(unmatched)):.1%})")

    if len(imputed) > 0:
        # Build imputed rows, using product-group-level trends
        imputed_rows = pd.DataFrame({
            'ndc_11': imputed['ndc_11'],
            'year_month': imputed['year_month'],
            'nadac_per_unit': imputed['product_median_price'],
            'nadac_pct_change_3m': imputed['product_pct_change_3m'],
            'nadac_pct_change_12m': imputed['product_pct_change_12m'],
            'nadac_generic_ratio': imputed['product_generic_ratio'],
            'nadac_is_low_price': (imputed['product_median_price'] <= imputed['product_p25']).astype(int),
            'nadac_vs_market_median': 1.0,  # By definition, imputed at median
            'nadac_is_observed': 0,
            'nadac_is_ffill': 0,
            'nadac_is_imputed': 1,
        })

        # Combine observed + imputed
        # First prepare observed monthly for concat
        monthly_out = monthly[['ndc_11', 'year_month', 'nadac_per_unit',
                               'nadac_pct_change_3m', 'nadac_pct_change_12m',
                               'nadac_generic_ratio', 'nadac_is_low_price',
                               'nadac_vs_market_median', 'nadac_is_observed',
                               'nadac_is_ffill', 'nadac_is_imputed']].copy()

        monthly_final = pd.concat([monthly_out, imputed_rows], ignore_index=True)
        print(f"  Combined: {len(monthly_final):,} NDCxmonth rows")
    else:
        monthly_final = monthly[['ndc_11', 'year_month', 'nadac_per_unit',
                                  'nadac_pct_change_3m', 'nadac_pct_change_12m',
                                  'nadac_generic_ratio', 'nadac_is_low_price',
                                  'nadac_vs_market_median', 'nadac_is_observed',
                                  'nadac_is_ffill', 'nadac_is_imputed']].copy()

    # Drop helper columns from monthly (already selected above)
    del monthly

    # ------------------------------------------------------------------
    # 8. Filter to study period
    # ------------------------------------------------------------------
    print("\n[8/9] Filtering to study period...")
    monthly_final = monthly_final[monthly_final['year_month'] >= STUDY_START]
    monthly_final = monthly_final.drop_duplicates(subset=['ndc_11', 'year_month'], keep='first')
    print(f"  Final rows: {len(monthly_final):,}")
    print(f"  Final unique NDCs: {monthly_final['ndc_11'].nunique():,}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    print("\n[9/9] Saving...")
    output_path = INTERMEDIATE / "pricing.parquet"
    monthly_final.to_parquet(output_path, index=False)

    print(f"\n  Output: {output_path}")
    print(f"  Shape: {monthly_final.shape}")
    print(f"  NDCxmonth rows: {len(monthly_final):,}")
    print(f"  Unique NDCs: {monthly_final['ndc_11'].nunique():,}")
    print(f"  Date range: {monthly_final['year_month'].min()} to {monthly_final['year_month'].max()}")

    # Coverage stats
    n_observed = (monthly_final['nadac_is_observed'] == 1).sum()
    n_ffill = (monthly_final['nadac_is_ffill'] == 1).sum()
    n_imputed = (monthly_final['nadac_is_imputed'] == 1).sum()
    print(f"\n  Observed (direct NADAC match): {n_observed:,} ({n_observed/len(monthly_final):.1%})")
    print(f"  Forward-filled within NDC: {n_ffill:,} ({n_ffill/len(monthly_final):.1%})")
    print(f"  Imputed (product-level median): {n_imputed:,} ({n_imputed/len(monthly_final):.1%})")

    print(f"\n  Column stats:")
    for col in ['nadac_per_unit', 'nadac_pct_change_3m', 'nadac_pct_change_12m',
                'nadac_is_low_price', 'nadac_generic_ratio', 'nadac_vs_market_median']:
        if col in monthly_final.columns:
            nulls = monthly_final[col].isna().mean()
            print(f"    {col:30s} {nulls:.1%} null, mean={monthly_final[col].mean():.4f}")

    print("\nDone!")
    return monthly_final


if __name__ == "__main__":
    main()
