"""
01_build_drug_universe.py - Build the NDC x Month panel skeleton.

For each month in 2020-01 through 2025-09, identifies which NDCs were actively
marketed based on current and historical NDC Directory snapshots.

Output: Data/intermediate/panel_skeleton.parquet
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


def load_ndc_snapshot(product_path, package_path, snapshot_label):
    """Load and merge one NDC product + package snapshot."""
    print(f"  Loading NDC snapshot: {snapshot_label}")
    prod = read_ndc_product(product_path)
    pkg = read_ndc_package(package_path)

    # Standardize PRODUCTNDC in both
    prod['product_ndc'] = prod['PRODUCTNDC'].apply(format_productndc)
    pkg['product_ndc'] = pkg['PRODUCTNDC'].apply(format_productndc)

    # Standardize package NDC
    pkg['ndc_11'] = pkg['NDCPACKAGECODE'].apply(format_ndcpackagecode)

    # Merge product attributes onto packages
    merged = pkg.merge(prod, on='product_ndc', how='left', suffixes=('_pkg', '_prod'))

    # Use product-level fields from prod (they have _prod suffix after merge if duplicated)
    # Key columns
    cols = {
        'ndc_11': 'ndc_11',
        'product_ndc': 'product_ndc',
    }

    # Resolve suffixed columns
    for base in ['PRODUCTTYPENAME', 'DOSAGEFORMNAME', 'ROUTENAME', 'MARKETINGCATEGORYNAME',
                 'APPLICATIONNUMBER', 'LABELERNAME', 'NONPROPRIETARYNAME', 'SUBSTANCENAME',
                 'PHARM_CLASSES', 'DEASCHEDULE', 'STARTMARKETINGDATE', 'ENDMARKETINGDATE',
                 'NDC_EXCLUDE_FLAG', 'PROPRIETARYNAME', 'ACTIVE_NUMERATOR_STRENGTH',
                 'ACTIVE_INGRED_UNIT']:
        if f'{base}_prod' in merged.columns:
            cols[f'{base}_prod'] = base
        elif base in merged.columns:
            cols[base] = base

    # Also grab package-level dates if different
    for base in ['STARTMARKETINGDATE', 'ENDMARKETINGDATE']:
        if f'{base}_pkg' in merged.columns:
            cols[f'{base}_pkg'] = f'{base}_PKG'

    result = merged.rename(columns=cols)
    # Only keep renamed columns that exist
    keep_cols = list(set(cols.values()))
    keep_cols = [c for c in keep_cols if c in result.columns]
    result = result[keep_cols].copy()
    result['snapshot'] = snapshot_label

    return result


def main():
    print("=" * 70)
    print("01_build_drug_universe.py - Building NDC x Month panel skeleton")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load current NDC Directory
    # ------------------------------------------------------------------
    print("\n[1/5] Loading current NDC Directory...")
    current_prod_path = RAW_DATA / "NDC Directory" / "product.txt"
    current_pkg_path = RAW_DATA / "NDC Directory" / "package.txt"
    current = load_ndc_snapshot(current_prod_path, current_pkg_path, "current_2025")

    # ------------------------------------------------------------------
    # 2. Load historical NDC snapshots (2019-2024)
    # ------------------------------------------------------------------
    print("\n[2/5] Loading historical NDC snapshots...")
    historical_snapshots = []

    hist_years = {
        '2019': HIST_NDC / "ndctext_2019",
        '2020': HIST_NDC / "ndctext_2020",
        '2021': HIST_NDC / "ndctext_2021",
        '2022': HIST_NDC / "ndctext_2022",
        '2024': HIST_NDC / "ndctext_2024",
    }

    for year, folder in hist_years.items():
        prod_path = folder / "product.txt"
        pkg_path = folder / "package.txt"
        if prod_path.exists() and pkg_path.exists():
            snap = load_ndc_snapshot(prod_path, pkg_path, f"ndc_{year}")
            historical_snapshots.append(snap)
        else:
            print(f"  WARNING: Missing NDC snapshot for {year}")

    # ------------------------------------------------------------------
    # 3. Combine all snapshots and deduplicate
    # ------------------------------------------------------------------
    print("\n[3/5] Combining snapshots and building drug universe...")
    all_ndcs = pd.concat([current] + historical_snapshots, ignore_index=True)
    print(f"  Total rows across all snapshots: {len(all_ndcs):,}")

    # Drop rows with no valid NDC
    all_ndcs = all_ndcs.dropna(subset=['ndc_11'])
    print(f"  After dropping null NDCs: {len(all_ndcs):,}")

    # Filter to human drugs only
    valid_product_types = ['HUMAN PRESCRIPTION DRUG']
    if 'PRODUCTTYPENAME' in all_ndcs.columns:
        all_ndcs = all_ndcs[all_ndcs['PRODUCTTYPENAME'].isin(valid_product_types)]
        print(f"  After filtering to human Rx only: {len(all_ndcs):,}")

    # Filter to FDA-approved marketing categories
    valid_categories = ['ANDA', 'NDA', 'BLA', 'NDA AUTHORIZED GENERIC']
    if 'MARKETINGCATEGORYNAME' in all_ndcs.columns:
        all_ndcs = all_ndcs[all_ndcs['MARKETINGCATEGORYNAME'].isin(valid_categories)]
        print(f"  After filtering to ANDA/NDA/BLA: {len(all_ndcs):,}")

    # Exclude NDCs flagged for exclusion
    if 'NDC_EXCLUDE_FLAG' in all_ndcs.columns:
        all_ndcs = all_ndcs[all_ndcs['NDC_EXCLUDE_FLAG'].fillna('N') != 'Y']
        print(f"  After excluding NDC_EXCLUDE_FLAG=Y: {len(all_ndcs):,}")

    # ------------------------------------------------------------------
    # 4. Determine active window for each NDC
    # ------------------------------------------------------------------
    print("\n[4/5] Determining active windows per NDC...")

    # Parse marketing dates
    all_ndcs['start_date'] = all_ndcs['STARTMARKETINGDATE'].apply(parse_date_flexible)
    all_ndcs['end_date'] = all_ndcs['ENDMARKETINGDATE'].apply(parse_date_flexible)

    # For each unique NDC, take the earliest start date and latest end date
    # across all snapshots (this gives the broadest known active window)
    ndc_windows = (
        all_ndcs.groupby('ndc_11')
        .agg(
            start_date=('start_date', 'min'),
            end_date=('end_date', 'max'),
        )
        .reset_index()
    )

    # For NDCs without an end date in any snapshot, assume still active
    ndc_windows['end_date'] = ndc_windows['end_date'].fillna(pd.Timestamp('2025-09-30'))

    # For NDCs without a start date, use the earliest snapshot where they appeared
    earliest_snapshot_dates = {
        'ndc_2019': pd.Timestamp('2019-06-01'),
        'ndc_2020': pd.Timestamp('2020-10-01'),
        'ndc_2021': pd.Timestamp('2021-08-01'),
        'ndc_2022': pd.Timestamp('2022-04-01'),
        'ndc_2024': pd.Timestamp('2024-07-01'),
        'current_2025': pd.Timestamp('2025-01-01'),
    }
    # Get earliest snapshot each NDC appeared in. Map every snapshot label
    # to its date and take the MINIMUM per NDC. (The previous .first() took
    # whichever row happened to come first in the concat, which was the
    # current-2025 snapshot, so NDCs with a null marketing start date that
    # also appeared in historical snapshots were wrongly assigned a 2025
    # fallback start and dropped from the 2020-2024 panel.)
    first_seen = (
        all_ndcs.assign(_snap_date=all_ndcs['snapshot'].map(earliest_snapshot_dates))
        .groupby('ndc_11')['_snap_date']
        .min()
    )
    ndc_windows.loc[ndc_windows['start_date'].isna(), 'start_date'] = (
        ndc_windows.loc[ndc_windows['start_date'].isna(), 'ndc_11'].map(first_seen)
    )
    # Final fallback
    ndc_windows['start_date'] = ndc_windows['start_date'].fillna(pd.Timestamp('2019-01-01'))

    # Clip to study period
    study_start_dt = pd.Timestamp('2020-01-01')
    study_end_dt = pd.Timestamp('2025-09-30')
    ndc_windows['start_date'] = ndc_windows['start_date'].clip(upper=study_end_dt)
    ndc_windows['end_date'] = ndc_windows['end_date'].clip(lower=study_start_dt)

    # Keep only NDCs active during at least part of the study period
    ndc_windows = ndc_windows[
        (ndc_windows['start_date'] <= study_end_dt) &
        (ndc_windows['end_date'] >= study_start_dt)
    ]
    print(f"  NDCs active during study period: {len(ndc_windows):,}")

    # ------------------------------------------------------------------
    # 5. Build panel skeleton: NDC x Month
    # ------------------------------------------------------------------
    print("\n[5/5] Building NDC x Month panel skeleton...")
    year_months = generate_year_months(STUDY_START, STUDY_END)
    ym_dt = pd.to_datetime([f"{ym}-01" for ym in year_months])

    print("  Expanding NDC x month (this may take a minute)...")

    # Vectorized approach: for each NDC, determine start/end month indices
    start_months = ndc_windows['start_date'].dt.to_period('M').astype(str)
    end_months = ndc_windows['end_date'].dt.to_period('M').astype(str)

    # Create a mapping from year_month string to index
    ym_to_idx = {ym: i for i, ym in enumerate(year_months)}

    ndc_windows['start_ym'] = start_months.map(ym_to_idx).fillna(0).astype(int).clip(lower=0)
    ndc_windows['end_ym'] = end_months.map(ym_to_idx).fillna(len(year_months) - 1).astype(int).clip(upper=len(year_months) - 1)

    # Expand using numpy repeat
    ndc_arr = ndc_windows['ndc_11'].values
    start_arr = ndc_windows['start_ym'].values
    end_arr = ndc_windows['end_ym'].values

    # Number of months for each NDC
    n_months = end_arr - start_arr + 1
    n_months = np.maximum(n_months, 0)

    total_rows = n_months.sum()
    print(f"  Expected panel size: {total_rows:,} rows")

    # Build arrays
    ndc_repeated = np.repeat(ndc_arr, n_months)
    ym_indices = np.concatenate([np.arange(s, e + 1) for s, e in zip(start_arr, end_arr) if e >= s])
    ym_repeated = np.array(year_months)[ym_indices]

    panel = pd.DataFrame({
        'ndc_11': ndc_repeated,
        'year_month': ym_repeated,
    })
    print(f"  Panel skeleton: {len(panel):,} rows")

    # ------------------------------------------------------------------
    # Attach key attributes (from most recent snapshot for each NDC)
    # ------------------------------------------------------------------
    # Get the best attribute record for each NDC (prefer current snapshot)
    attr_cols = ['ndc_11', 'product_ndc', 'LABELERNAME', 'NONPROPRIETARYNAME',
                 'DOSAGEFORMNAME', 'ROUTENAME', 'MARKETINGCATEGORYNAME',
                 'APPLICATIONNUMBER', 'SUBSTANCENAME', 'PHARM_CLASSES',
                 'DEASCHEDULE', 'PRODUCTTYPENAME', 'ACTIVE_NUMERATOR_STRENGTH',
                 'ACTIVE_INGRED_UNIT', 'PROPRIETARYNAME', 'STARTMARKETINGDATE']
    attr_cols = [c for c in attr_cols if c in all_ndcs.columns]

    # Prefer the most RECENT snapshot's attributes for each NDC. Sort by the
    # snapshot's actual date (not the label string, whose lexicographic order
    # puts 'current_2025' FIRST, not last) and keep the last row per NDC.
    all_ndcs_sorted = all_ndcs.assign(
        _snap_date=all_ndcs['snapshot'].map(earliest_snapshot_dates)
    ).sort_values(['ndc_11', '_snap_date'], kind='stable')
    ndc_attrs = all_ndcs_sorted.drop_duplicates(subset=['ndc_11'], keep='last')[attr_cols]

    # Derive labeler_code
    ndc_attrs['labeler_code'] = ndc_attrs['ndc_11'].apply(labeler_code_from_ndc)

    panel = panel.merge(ndc_attrs, on='ndc_11', how='left')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = INTERMEDIATE / "panel_skeleton.parquet"
    panel.to_parquet(output_path, index=False)
    print(f"\n  Saved panel skeleton to {output_path}")
    print(f"  Shape: {panel.shape}")
    print(f"  Unique NDCs: {panel['ndc_11'].nunique():,}")
    print(f"  Unique months: {panel['year_month'].nunique()}")
    print(f"  Date range: {panel['year_month'].min()} to {panel['year_month'].max()}")

    # Summary stats
    print("\n  NDCs per month:")
    print(panel.groupby('year_month').size().describe().to_string())

    print("\n  Top marketing categories:")
    if 'MARKETINGCATEGORYNAME' in panel.columns:
        print(panel.drop_duplicates('ndc_11')['MARKETINGCATEGORYNAME'].value_counts().to_string())

    print("\nDone!")
    return panel


if __name__ == "__main__":
    main()
