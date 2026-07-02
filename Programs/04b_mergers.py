"""
04b_mergers.py - Merger / ownership change features.

Uses ownership change data from the Generic mergers project to flag
drugs involved in mergers, consolidation events, and divestitures.

Output: Data/intermediate/mergers.parquet
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

MERGERS_DIR = GENERIC_MERGERS / "data" / "intermediate"


def main():
    print("=" * 70)
    print("04b_mergers.py - Building merger / ownership change features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load ownership changes
    # ------------------------------------------------------------------
    print("\n[1/5] Loading ownership changes...")
    oc = pd.read_csv(MERGERS_DIR / "ownership_changes_all.csv", dtype=str)
    print(f"  Ownership change records: {len(oc):,}")
    print(f"  Columns: {list(oc.columns)}")

    # Parse dates
    oc['date_from'] = pd.to_datetime(oc['date_from'], errors='coerce')
    oc['date_to'] = pd.to_datetime(oc['date_to'], errors='coerce')
    oc['year_from'] = pd.to_numeric(oc['year_from'], errors='coerce')
    oc['year_to'] = pd.to_numeric(oc['year_to'], errors='coerce')

    # Pad appl_no to 6 digits
    oc['appl_no'] = oc['appl_no'].astype(str).str.zfill(6)

    # Use midpoint of date_from and date_to as the event date
    oc['event_date'] = oc['date_from'] + (oc['date_to'] - oc['date_from']) / 2
    oc.loc[oc['event_date'].isna(), 'event_date'] = oc.loc[oc['event_date'].isna(), 'date_to']

    # Filter to events that overlap with or provide lookback for our study period
    oc = oc[oc['event_date'].notna()]
    oc = oc[(oc['event_date'] >= '2018-01-01') & (oc['event_date'] <= '2025-12-31')]
    print(f"  After date filter (2018-2025): {len(oc):,}")

    # Filter out internal reorganizations (same parent company, different subsidiary)
    # Heuristic: if normalized names share the first word, likely internal
    def is_likely_internal(row):
        from_name = str(row.get('applicant_from_normalized', '')).split()
        to_name = str(row.get('applicant_to_normalized', '')).split()
        if from_name and to_name and from_name[0] == to_name[0]:
            return True
        return False

    oc['is_internal'] = oc.apply(is_likely_internal, axis=1)
    print(f"  Internal reorganizations: {oc['is_internal'].sum():,}")
    print(f"  External ownership changes: {(~oc['is_internal']).sum():,}")

    # ------------------------------------------------------------------
    # 2. Load merger-affected products with NDC links
    # ------------------------------------------------------------------
    print("\n[2/5] Loading merger-affected products with NDC links...")
    overlap_path = MERGERS_DIR / "product_overlap_with_ndcs_final.csv"
    if overlap_path.exists():
        overlap = pd.read_csv(overlap_path, dtype=str)
        print(f"  Merger-affected products: {len(overlap):,}")
        print(f"  With NDC matches: {(overlap['has_ndc'] == 'True').sum():,}")

        # Explode NDC matches into individual rows
        overlap_with_ndc = overlap[overlap['has_ndc'] == 'True'].copy()
        overlap_with_ndc['ndc_list'] = overlap_with_ndc['ndc_matches'].str.split(';')
        overlap_exploded = overlap_with_ndc.explode('ndc_list')
        overlap_exploded['product_ndc'] = overlap_exploded['ndc_list'].apply(format_productndc)
        overlap_exploded = overlap_exploded.dropna(subset=['product_ndc'])

        # Parse transfer periods to get event dates
        # transfer_period looks like "2019->2020" or "2019->2020; 2020->2021"
        def parse_transfer_years(period_str):
            if pd.isna(period_str):
                return []
            years = []
            for part in str(period_str).split(';'):
                part = part.strip()
                if '->' in part:
                    try:
                        y_from, y_to = part.split('->')
                        years.append(int(y_to.strip()))
                    except (ValueError, IndexError):
                        continue
            return years

        overlap_exploded['transfer_years'] = overlap_exploded['transfer_period'].apply(parse_transfer_years)
        print(f"  Exploded NDC rows: {len(overlap_exploded):,}")
    else:
        print("  product_overlap_with_ndcs_final.csv not found, using ownership changes only")
        overlap_exploded = pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Build NDC -> appl_no mapping for ownership changes
    # ------------------------------------------------------------------
    print("\n[3/5] Building NDC to application number mapping...")
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

    # Also build product_ndc -> ndc_11 mapping for the overlap file
    pndc_to_ndc11 = ndc_pkg[['product_ndc', 'ndc_11']].dropna().drop_duplicates()

    print(f"  NDCs with appl_no: {ndc_appl['appl_no'].notna().sum():,}")

    # ------------------------------------------------------------------
    # 4. Compute merger features per NDC-month
    # ------------------------------------------------------------------
    print("\n[4/5] Computing merger features per NDC x month...")
    year_months = generate_year_months(STUDY_START, STUDY_END)

    # Approach A: From ownership_changes via appl_no
    # For each appl_no that had an ownership change, flag NDCs linked to it
    oc_events = oc[['appl_no', 'event_date', 'is_internal']].copy()

    # Map appl_no to NDCs
    appl_to_ndcs = ndc_appl[ndc_appl['appl_no'].notna()].groupby('appl_no')['ndc_11'].apply(list).to_dict()

    # Approach B: From product_overlap via product_ndc
    # Map product_ndc to NDCs
    if len(overlap_exploded) > 0:
        overlap_ndc_map = overlap_exploded.merge(pndc_to_ndc11, on='product_ndc', how='inner')
        print(f"  Overlap products matched to NDC-11: {overlap_ndc_map['ndc_11'].nunique():,}")

    # Build event list: (ndc_11, event_date, is_internal)
    events = []

    # From ownership changes
    for _, row in oc_events.iterrows():
        ndcs = appl_to_ndcs.get(row['appl_no'], [])
        for ndc in ndcs:
            events.append({
                'ndc_11': ndc,
                'event_date': row['event_date'],
                'is_external': not row['is_internal'],
            })

    # From product overlap (merger-specific)
    if len(overlap_exploded) > 0:
        for _, row in overlap_ndc_map.iterrows():
            for yr in row.get('transfer_years', []):
                events.append({
                    'ndc_11': row['ndc_11'],
                    'event_date': pd.Timestamp(f"{yr}-07-01"),  # Midpoint of year
                    'is_external': True,
                })

    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.drop_duplicates(subset=['ndc_11', 'event_date'])
    print(f"  Total merger/ownership events: {len(events_df):,}")
    print(f"  Unique NDCs affected: {events_df['ndc_11'].nunique():,}")

    # For each NDC x month, compute lagged merger features
    records = []
    if len(events_df) > 0:
        ndc_events = events_df.groupby('ndc_11').apply(
            lambda g: g[['event_date', 'is_external']].to_dict('records')
        ).to_dict()

        for ndc, evt_list in ndc_events.items():
            event_dates = [e['event_date'] for e in evt_list]
            external_dates = [e['event_date'] for e in evt_list if e['is_external']]

            for ym in year_months:
                ym_dt = pd.Timestamp(f"{ym}-01")

                # Any ownership change in prior 12 months
                any_change_12m = any(
                    ym_dt - pd.DateOffset(months=12) <= d < ym_dt
                    for d in event_dates
                )
                # Any ownership change in prior 24 months
                any_change_24m = any(
                    ym_dt - pd.DateOffset(months=24) <= d < ym_dt
                    for d in event_dates
                )
                # External merger in prior 12 months
                external_merger_12m = any(
                    ym_dt - pd.DateOffset(months=12) <= d < ym_dt
                    for d in external_dates
                )
                # External merger in prior 24 months
                external_merger_24m = any(
                    ym_dt - pd.DateOffset(months=24) <= d < ym_dt
                    for d in external_dates
                )
                # Count of changes in prior 24 months
                n_changes_24m = sum(
                    1 for d in event_dates
                    if ym_dt - pd.DateOffset(months=24) <= d < ym_dt
                )

                if any_change_24m:  # Only store rows with at least one event
                    records.append({
                        'ndc_11': ndc,
                        'year_month': ym,
                        'ownership_change_12m': int(any_change_12m),
                        'ownership_change_24m': int(any_change_24m),
                        'external_merger_12m': int(external_merger_12m),
                        'external_merger_24m': int(external_merger_24m),
                        'n_ownership_changes_24m': n_changes_24m,
                    })

    merger_features = pd.DataFrame(records)
    print(f"  Merger feature rows: {len(merger_features):,}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print("\n[5/5] Saving...")
    output_path = INTERMEDIATE / "mergers.parquet"
    merger_features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Shape: {merger_features.shape}")
    if len(merger_features) > 0:
        print(f"  Unique NDCs: {merger_features['ndc_11'].nunique():,}")
        ndc_snap = merger_features.drop_duplicates('ndc_11')
        print(f"\n  Ownership change (12m): {ndc_snap['ownership_change_12m'].mean():.1%}")
        print(f"  External merger (12m): {ndc_snap['external_merger_12m'].mean():.1%}")
        print(f"  Ownership change (24m): {ndc_snap['ownership_change_24m'].mean():.1%}")

    print("\nDone!")
    return merger_features


if __name__ == "__main__":
    main()
