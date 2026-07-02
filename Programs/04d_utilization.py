"""
04d_utilization.py - Build SDUD + CMS Part D utilization features.

Reads SDUD CSV files (7 files including 2019 for lookback, ~3 GB) and CMS
Part D spending data, aggregates across states, and creates per-NDCxmonth
utilization features.

Improvements over prior version:
  1. Includes 2019 SDUD data for lookback → enables 2020 YoY trends
  2. Drug-name normalization for Part D matching (HYDROCHLORIDE→HCL, etc.)
  3. Ingredient-level utilization features for NDCs not in SDUD

Output: Data/intermediate/utilization.parquet
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


# Quarter-end month mapping
QUARTER_TO_MONTHS = {1: ['01', '02', '03'], 2: ['04', '05', '06'],
                     3: ['07', '08', '09'], 4: ['10', '11', '12']}

# Publication-lag alignment. Features must reflect what was publicly
# available in the panel month, not the period the data describes.
# SDUD quarterly files are released roughly two quarters after the
# quarter closes. CMS Part D / Part B spending files for year Y are
# released during year Y+1.
SDUD_PUBLICATION_LAG_QUARTERS = 2
CMS_SPENDING_PUBLICATION_LAG_YEARS = 1


def load_sdud_files():
    """Load and concatenate all SDUD CSV files."""
    sdud_dir = RAW_DATA / "SDUD"
    files = sorted(sdud_dir.glob("sdud_*.csv"))
    print(f"  Found {len(files)} SDUD files")

    dfs = []
    for f in files:
        print(f"    Loading {f.name}...", end=" ")
        # Read with low_memory=False due to mixed types in numeric columns
        df = pd.read_csv(f, dtype=str, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"{len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows")
    return combined


def process_sdud(sdud):
    """Process SDUD data into NDCxmonth features."""
    print("\n  Converting NDCs...")
    # SDUD NDC column is already 11-digit with leading zeros
    sdud['ndc_11'] = sdud['NDC'].apply(ndc_nodash_to_panel)
    sdud = sdud.dropna(subset=['ndc_11'])

    # Parse numeric columns
    for col in ['Units Reimbursed', 'Number of Prescriptions', 'Total Amount Reimbursed']:
        if col in sdud.columns:
            sdud[col] = pd.to_numeric(sdud[col], errors='coerce').fillna(0)

    # Remove suppressed rows
    if 'Suppression Used' in sdud.columns:
        sdud = sdud[sdud['Suppression Used'].str.lower() != 'true']

    # Create year and quarter
    sdud['year'] = pd.to_numeric(sdud['Year'], errors='coerce').astype('Int64')
    sdud['quarter'] = pd.to_numeric(sdud['Quarter'], errors='coerce').astype('Int64')
    sdud = sdud.dropna(subset=['year', 'quarter'])

    # Filter: include 2018+ for lookback (need 4 quarters before 2020-Q1)
    sdud = sdud[sdud['year'] >= 2018]
    print(f"  After date filter (>=2018): {len(sdud):,} rows")

    # Aggregate across states per NDCxquarter
    print("  Aggregating across states...")
    quarterly = sdud.groupby(['ndc_11', 'year', 'quarter']).agg(
        medicaid_units=('Units Reimbursed', 'sum'),
        medicaid_rx_count=('Number of Prescriptions', 'sum'),
        medicaid_spending=('Total Amount Reimbursed', 'sum'),
    ).reset_index()

    print(f"  NDCxquarter rows: {len(quarterly):,}")

    # Compute YoY trends at quarterly level (before expanding to monthly)
    quarterly = quarterly.sort_values(['ndc_11', 'year', 'quarter'])
    quarterly['medicaid_rx_trend_4q'] = quarterly.groupby('ndc_11')['medicaid_rx_count'].transform(
        lambda x: x.pct_change(periods=4)
    ).clip(-10, 10)
    quarterly['medicaid_units_trend_4q'] = quarterly.groupby('ndc_11')['medicaid_units'].transform(
        lambda x: x.pct_change(periods=4)
    ).clip(-10, 10)

    # Utilization volatility - CV of Rx count over trailing 4 quarters (#15)
    quarterly['medicaid_rx_cv_4q'] = quarterly.groupby('ndc_11')['medicaid_rx_count'].transform(
        lambda x: x.rolling(4, min_periods=2).std() / x.rolling(4, min_periods=2).mean()
    ).clip(0, 10).fillna(0)

    # Expand to monthly with a publication-lag shift. Quarter Q values are
    # assigned to the months of quarter Q + SDUD_PUBLICATION_LAG_QUARTERS,
    # the earliest months in which the quarter's file was actually public.
    print(f"  Expanding to monthly ({SDUD_PUBLICATION_LAG_QUARTERS}-quarter publication lag)...")
    monthly_rows = []
    for _, row in quarterly.iterrows():
        avail_q = (pd.Period(f"{int(row['year'])}Q{int(row['quarter'])}", freq='Q')
                   + SDUD_PUBLICATION_LAG_QUARTERS)
        yr = avail_q.year
        qtr = avail_q.quarter
        for m in QUARTER_TO_MONTHS[qtr]:
            ym = f"{yr}-{m}"
            monthly_rows.append({
                'ndc_11': row['ndc_11'],
                'year_month': ym,
                'medicaid_rx_count': row['medicaid_rx_count'],
                'medicaid_units': row['medicaid_units'],
                'medicaid_spending': row['medicaid_spending'],
                'medicaid_rx_trend_4q': row['medicaid_rx_trend_4q'],
                'medicaid_units_trend_4q': row['medicaid_units_trend_4q'],
                'medicaid_rx_cv_4q': row['medicaid_rx_cv_4q'],
            })

    monthly = pd.DataFrame(monthly_rows)
    monthly['has_medicaid_data'] = 1
    monthly['utilization_product_imputed'] = 0
    # Filter to study period
    monthly = monthly[monthly['year_month'] >= STUDY_START]
    print(f"  Monthly observations: {len(monthly):,}")

    return monthly


def normalize_drug_name(name):
    """Normalize generic drug names for matching between NDC and Part D.

    Part D uses abbreviations (HCL, HBR, SOD) while the NDC directory
    uses full names (HYDROCHLORIDE, HYDROBROMIDE, SODIUM).
    """
    import re
    if pd.isna(name):
        return ''
    s = str(name).upper().strip()

    # Standardize salt forms to abbreviations used by Part D
    replacements = [
        (r'\bHYDROCHLORIDE\b', 'HCL'),
        (r'\bHYDROBROMIDE\b', 'HBR'),
        (r'\bMESYLATE\b', 'MESYLATE'),
        (r'\bBESYLATE\b', 'BESYLATE'),
        (r'\bSODIUM\b', 'SOD'),
        (r'\bPOTASSIUM\b', 'POT'),
        (r'\bSUCCINATE\b', 'SUCC'),
        (r'\bFUMARATE\b', 'FUMARATE'),
        (r'\bMEDOXOMIL\b', 'MEDOXOMIL'),
        (r'\bTARTRATE\b', 'TARTRATE'),
        (r'\bMALEATE\b', 'MALEATE'),
        (r'\bPHOSPHATE\b', 'PHOSPHATE'),
        (r'\bSULFATE\b', 'SULFATE'),
        (r'\bCALCIUM\b', 'CALCIUM'),
        (r'\bACETATE\b', 'ACETATE'),
        (r'\bCITRATE\b', 'CITRATE'),
        (r'\b AND \b', '/'),
    ]
    for pattern, replacement in replacements:
        s = re.sub(pattern, replacement, s)

    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def process_partd():
    """Process CMS Part D spending data (drug-name level)."""
    print("\n  Processing CMS Part D data...")
    partd_dir = RAW_DATA / "CMS Drug Spending"

    # --- Annual data (2019-2023) ---
    annual_file = partd_dir / "medicare_partd_spending_by_drug_2019_2023.csv"
    partd_rows = []

    if annual_file.exists():
        print(f"    Loading {annual_file.name}...")
        df = pd.read_csv(annual_file, dtype=str)
        df.columns = df.columns.str.strip()

        # Keep only "Overall" manufacturer rows (aggregated)
        df = df[df['Mftr_Name'] == 'Overall']

        # Melt wide to long: columns like Tot_Spndng_2019, Tot_Clms_2019, etc.
        for year in range(2019, 2024):
            yr_cols = {
                f'Tot_Spndng_{year}': 'partd_total_spending',
                f'Tot_Clms_{year}': 'partd_total_claims',
                f'Tot_Benes_{year}': 'partd_total_beneficiaries',
                f'Avg_Spnd_Per_Clm_{year}': 'partd_avg_cost_per_claim',
            }
            # Check which columns exist
            available = {k: v for k, v in yr_cols.items() if k in df.columns}
            if not available:
                continue

            year_df = df[['Gnrc_Name'] + list(available.keys())].copy()
            year_df = year_df.rename(columns=available)
            for col in available.values():
                year_df[col] = pd.to_numeric(year_df[col], errors='coerce')
            year_df['year'] = year

            partd_rows.append(year_df)

    # --- Quarterly data (2024-2025) ---
    quarterly_file = partd_dir / "medicare_partd_spending_quarterly_through_2025q2.csv"
    if quarterly_file.exists():
        print(f"    Loading {quarterly_file.name}...")
        df = pd.read_csv(quarterly_file, dtype=str)
        df.columns = df.columns.str.strip()

        # Keep "Overall" rows
        df = df[df['Mftr_Name'] == 'Overall']

        # Parse year from 'Year' column (e.g. "2024 (Q1-Q4)")
        df['year_parsed'] = df['Year'].str.extract(r'(\d{4})').astype(float)

        for col in ['Tot_Spndng', 'Tot_Clms', 'Tot_Benes', 'Avg_Spnd_Per_Clm']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        year_df = df[['Gnrc_Name', 'year_parsed']].copy()
        year_df = year_df.rename(columns={'year_parsed': 'year'})
        if 'Tot_Spndng' in df.columns:
            year_df['partd_total_spending'] = df['Tot_Spndng']
        if 'Tot_Clms' in df.columns:
            year_df['partd_total_claims'] = df['Tot_Clms']
        if 'Tot_Benes' in df.columns:
            year_df['partd_total_beneficiaries'] = df['Tot_Benes']
        if 'Avg_Spnd_Per_Clm' in df.columns:
            year_df['partd_avg_cost_per_claim'] = df['Avg_Spnd_Per_Clm']

        partd_rows.append(year_df)

    if not partd_rows:
        print("    No Part D data found.")
        return pd.DataFrame(columns=['generic_name_upper', 'year_month',
                                      'partd_total_claims', 'partd_avg_cost_per_claim'])

    partd = pd.concat(partd_rows, ignore_index=True)
    partd['generic_name_upper'] = partd['Gnrc_Name'].str.upper().str.strip()
    # Also create normalized name for fuzzy matching
    partd['generic_name_norm'] = partd['Gnrc_Name'].apply(normalize_drug_name)
    partd = partd.dropna(subset=['generic_name_upper', 'year'])
    partd['year'] = partd['year'].astype(int)

    # Expand annual to monthly with a publication-lag shift. CMS publishes
    # the year-Y spending file during year Y+1, so year-Y values are
    # assigned to the months of year Y+1 (the first full year in which
    # they were available).
    monthly_rows = []
    for _, row in partd.iterrows():
        avail_yr = int(row['year']) + CMS_SPENDING_PUBLICATION_LAG_YEARS
        for m in range(1, 13):
            ym = f"{avail_yr}-{m:02d}"
            if ym < STUDY_START or ym > STUDY_END:
                continue
            monthly_rows.append({
                'generic_name_upper': row['generic_name_upper'],
                'generic_name_norm': row.get('generic_name_norm', ''),
                'year_month': ym,
                'partd_total_claims': row.get('partd_total_claims', np.nan),
                'partd_avg_cost_per_claim': row.get('partd_avg_cost_per_claim', np.nan),
            })

    partd_monthly = pd.DataFrame(monthly_rows)
    print(f"    Part D monthly rows: {len(partd_monthly):,}")
    print(f"    Unique generic names (raw): {partd_monthly['generic_name_upper'].nunique():,}")
    print(f"    Unique generic names (normalized): {partd_monthly['generic_name_norm'].nunique():,}")

    return partd_monthly


def process_partb():
    """Process CMS Part B spending files (HCPCS -> Gnrc_Name level).

    Mirrors process_partd() but reads the Part B files. Produces monthly
    rows keyed by generic_name_upper for downstream merging via the same
    crosswalk pattern. Part B captures spending on physician-administered
    drugs (IV/infused, oncology, etc.) that Part D doesn't see.

    Two sources are combined:
      - annual wide file (2019-2023): fills the training window; with the
        Y+1 publication-lag shift this supplies panel months 2020-2024
      - quarterly long file (2024+): supplies panel months 2025+
    """
    print("\n  Processing CMS Part B data...")
    partb_dir = RAW_DATA / "CMS Drug Spending"
    frames = []

    # --- Annual wide file (2019-2023), same layout as the Part D annual ---
    annual_file = partb_dir / "medicare_partb_spending_by_drug_2019_2023.csv"
    if annual_file.exists():
        print(f"    Loading {annual_file.name}...")
        adf = pd.read_csv(annual_file, dtype=str)
        adf.columns = adf.columns.str.strip()
        for year in range(2019, 2024):
            cols = {
                f'Tot_Spndng_{year}': 'Tot_Spndng',
                f'Tot_Clms_{year}': 'Tot_Clms',
                f'Tot_Benes_{year}': 'Tot_Benes',
            }
            available = {k: v for k, v in cols.items() if k in adf.columns}
            if not available:
                continue
            ydf = adf[['Gnrc_Name'] + list(available.keys())].rename(columns=available)
            ydf['year_parsed'] = float(year)
            frames.append(ydf)

    # --- Quarterly long file: prefer the most recent vintage on disk ---
    quarterly_loaded = False
    for qname in ['medicare_partb_spending_quarterly_through_2025q3.csv',
                  'medicare_partb_spending_quarterly_through_2025q2.csv']:
        quarterly_file = partb_dir / qname
        if quarterly_file.exists():
            print(f"    Loading {quarterly_file.name}...")
            qdf = pd.read_csv(quarterly_file, dtype=str)
            qdf.columns = qdf.columns.str.strip()
            qdf['year_parsed'] = qdf['Year'].str.extract(r'(\d{4})').astype(float)
            frames.append(qdf[['Gnrc_Name', 'year_parsed', 'Tot_Spndng', 'Tot_Clms', 'Tot_Benes']])
            quarterly_loaded = True
            break

    if not frames:
        print(f"    No Part B files found under {partb_dir}; skipping.")
        return pd.DataFrame(columns=[
            'generic_name_upper', 'year_month',
            'partb_total_claims', 'partb_avg_cost_per_claim',
            'partb_total_spending', 'partb_total_beneficiaries',
        ])

    # If a year appears in both sources, the annual file wins: drop that
    # year from the quarterly frame (always the last frame when loaded)
    # before concatenating.
    if annual_file.exists() and quarterly_loaded:
        annual_years = {float(y) for y in range(2019, 2024)}
        qframe = frames[-1]
        frames[-1] = qframe[~qframe['year_parsed'].isin(annual_years)]

    df = pd.concat(frames, ignore_index=True)
    for col in ['Tot_Spndng', 'Tot_Clms', 'Tot_Benes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Gnrc_Name', 'year_parsed'])

    # Sum-aggregate spending/claims/beneficiaries across HCPCS codes within
    # the same Gnrc_Name x year; recompute avg cost from totals.
    agg = df.groupby(['Gnrc_Name', 'year_parsed'], as_index=False).agg(
        partb_total_spending=('Tot_Spndng', 'sum'),
        partb_total_claims=('Tot_Clms', 'sum'),
        partb_total_beneficiaries=('Tot_Benes', 'sum'),
    )
    agg['partb_avg_cost_per_claim'] = agg['partb_total_spending'] / agg['partb_total_claims'].replace(0, np.nan)
    agg = agg.rename(columns={'year_parsed': 'year'})
    agg['year'] = agg['year'].astype(int)
    agg['generic_name_upper'] = agg['Gnrc_Name'].str.upper().str.strip()
    agg['generic_name_norm'] = agg['Gnrc_Name'].apply(normalize_drug_name)

    # Expand annual rows to monthly with the same publication-lag shift as
    # Part D: year-Y values are assigned to the months of year Y+1.
    monthly_rows = []
    for _, row in agg.iterrows():
        avail_yr = int(row['year']) + CMS_SPENDING_PUBLICATION_LAG_YEARS
        for m in range(1, 13):
            ym = f"{avail_yr}-{m:02d}"
            if ym < STUDY_START or ym > STUDY_END:
                continue
            monthly_rows.append({
                'generic_name_upper': row['generic_name_upper'],
                'generic_name_norm': row['generic_name_norm'],
                'year_month': ym,
                'partb_total_spending': row['partb_total_spending'],
                'partb_total_claims': row['partb_total_claims'],
                'partb_total_beneficiaries': row['partb_total_beneficiaries'],
                'partb_avg_cost_per_claim': row['partb_avg_cost_per_claim'],
            })

    partb_monthly = pd.DataFrame(monthly_rows)
    print(f"    Part B monthly rows: {len(partb_monthly):,}")
    print(f"    Unique generic names (raw): {partb_monthly['generic_name_upper'].nunique():,}")
    return partb_monthly


def main():
    print("=" * 70)
    print("04d_utilization.py - Building utilization features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Process SDUD
    # ------------------------------------------------------------------
    print("\n[1/4] Loading SDUD data...")
    sdud = load_sdud_files()

    print("\n[2/4] Processing SDUD into monthly features...")
    sdud_monthly = process_sdud(sdud)
    del sdud  # Free memory

    # ------------------------------------------------------------------
    # 2. Process Part D and Part B
    # ------------------------------------------------------------------
    print("\n[3/4] Processing CMS Part D data...")
    partd_monthly = process_partd()
    partb_monthly = process_partb()

    # ------------------------------------------------------------------
    # 3. Merge Part D onto SDUD (by generic name, with normalization)
    # ------------------------------------------------------------------
    print("\n[4/6] Merging Part D via generic name crosswalk...")

    # Load panel skeleton for NDC → generic name mapping
    skeleton = pd.read_parquet(INTERMEDIATE / "panel_skeleton.parquet",
                               columns=['ndc_11', 'NONPROPRIETARYNAME'])
    skeleton = skeleton.drop_duplicates(subset=['ndc_11'])
    skeleton['generic_name_upper'] = skeleton['NONPROPRIETARYNAME'].str.upper().str.strip()
    skeleton['generic_name_norm'] = skeleton['NONPROPRIETARYNAME'].apply(normalize_drug_name)

    if len(partd_monthly) > 0:
        # Get NDC→name mapping
        ndc_names = skeleton[['ndc_11', 'generic_name_upper', 'generic_name_norm']].drop_duplicates()

        # Merge name onto SDUD monthly
        sdud_monthly = sdud_monthly.merge(ndc_names, on='ndc_11', how='left')

        # Try exact match first (raw name)
        sdud_monthly = sdud_monthly.merge(
            partd_monthly[['generic_name_upper', 'year_month',
                            'partd_total_claims', 'partd_avg_cost_per_claim']],
            on=['generic_name_upper', 'year_month'],
            how='left'
        )

        # For unmatched, try normalized name match
        unmatched_mask = sdud_monthly['partd_total_claims'].isna()
        n_unmatched_before = unmatched_mask.sum()

        if n_unmatched_before > 0:
            partd_norm = partd_monthly[['generic_name_norm', 'year_month',
                                         'partd_total_claims', 'partd_avg_cost_per_claim']].copy()
            partd_norm = partd_norm.rename(columns={
                'partd_total_claims': '_partd_claims_norm',
                'partd_avg_cost_per_claim': '_partd_cost_norm',
            })
            sdud_monthly = sdud_monthly.merge(
                partd_norm.drop_duplicates(subset=['generic_name_norm', 'year_month']),
                on=['generic_name_norm', 'year_month'],
                how='left'
            )
            # Fill in from normalized match where exact match failed
            fill_mask = sdud_monthly['partd_total_claims'].isna() & sdud_monthly['_partd_claims_norm'].notna()
            sdud_monthly.loc[fill_mask, 'partd_total_claims'] = sdud_monthly.loc[fill_mask, '_partd_claims_norm']
            sdud_monthly.loc[fill_mask, 'partd_avg_cost_per_claim'] = sdud_monthly.loc[fill_mask, '_partd_cost_norm']
            n_filled = fill_mask.sum()
            print(f"  Part D exact match: {n_unmatched_before - sdud_monthly['partd_total_claims'].isna().sum() - n_filled:,} unmatched -> {n_filled:,} filled by normalized name")
            sdud_monthly.drop(columns=['_partd_claims_norm', '_partd_cost_norm'], inplace=True)

        sdud_monthly.drop(columns=['generic_name_upper', 'generic_name_norm'], inplace=True)
    else:
        sdud_monthly['partd_total_claims'] = np.nan
        sdud_monthly['partd_avg_cost_per_claim'] = np.nan

    sdud_monthly['has_partd_data'] = sdud_monthly['partd_total_claims'].notna().astype(int)

    # ------------------------------------------------------------------
    # 3b. Merge Part B onto SDUD (same exact + normalized fallback pattern)
    # ------------------------------------------------------------------
    print("\n[4b/6] Merging Part B via generic name crosswalk...")
    partb_cols = ['partb_total_spending', 'partb_total_claims',
                  'partb_total_beneficiaries', 'partb_avg_cost_per_claim']
    if len(partb_monthly) > 0:
        # The generic_name_upper/_norm columns from the Part D merge were
        # dropped; re-attach them from the panel skeleton.
        ndc_names = skeleton[['ndc_11', 'generic_name_upper', 'generic_name_norm']].drop_duplicates()
        sdud_monthly = sdud_monthly.merge(ndc_names, on='ndc_11', how='left')

        sdud_monthly = sdud_monthly.merge(
            partb_monthly[['generic_name_upper', 'year_month'] + partb_cols],
            on=['generic_name_upper', 'year_month'], how='left',
        )

        unmatched_mask = sdud_monthly['partb_total_claims'].isna()
        n_unmatched_before = unmatched_mask.sum()

        if n_unmatched_before > 0:
            partb_norm = partb_monthly[
                ['generic_name_norm', 'year_month'] + partb_cols
            ].copy()
            renames = {c: f'_{c}_norm' for c in partb_cols}
            partb_norm = partb_norm.rename(columns=renames)
            sdud_monthly = sdud_monthly.merge(
                partb_norm.drop_duplicates(subset=['generic_name_norm', 'year_month']),
                on=['generic_name_norm', 'year_month'], how='left',
            )
            fill_mask = sdud_monthly['partb_total_claims'].isna() & sdud_monthly['_partb_total_claims_norm'].notna()
            for c in partb_cols:
                sdud_monthly.loc[fill_mask, c] = sdud_monthly.loc[fill_mask, f'_{c}_norm']
            n_filled = int(fill_mask.sum())
            print(f"  Part B: {n_filled:,} additional rows filled by normalized-name match")
            sdud_monthly.drop(columns=[f'_{c}_norm' for c in partb_cols], inplace=True)

        sdud_monthly.drop(columns=['generic_name_upper', 'generic_name_norm'], inplace=True)
    else:
        for c in partb_cols:
            sdud_monthly[c] = np.nan

    sdud_monthly['has_partb_data'] = sdud_monthly['partb_total_claims'].notna().astype(int)

    # ------------------------------------------------------------------
    # 4. Add ingredient-level utilization for unmatched panel NDCs
    # ------------------------------------------------------------------
    print("\n[5/6] Computing ingredient-level utilization for unmatched NDCs...")

    # Load full panel skeleton to find NDCs not in SDUD
    panel_skel = pd.read_parquet(
        INTERMEDIATE / "panel_skeleton.parquet",
        columns=['ndc_11', 'year_month', 'SUBSTANCENAME', 'DOSAGEFORMNAME', 'ROUTENAME']
    )
    panel_skel = panel_skel[panel_skel['year_month'] >= STUDY_START]

    # Create product group key
    panel_skel['product_group'] = (
        panel_skel['SUBSTANCENAME'].fillna('').str.upper().str.strip() + '|' +
        panel_skel['DOSAGEFORMNAME'].fillna('').str.upper().str.strip() + '|' +
        panel_skel['ROUTENAME'].fillna('').str.upper().str.strip()
    )

    # Get product groups for SDUD-matched NDCs
    sdud_ndcs_with_group = skeleton[['ndc_11']].merge(
        panel_skel[['ndc_11', 'product_group']].drop_duplicates(subset=['ndc_11']),
        on='ndc_11', how='inner'
    )

    # Compute product-group-level stats from SDUD data
    sdud_with_group = sdud_monthly.merge(
        sdud_ndcs_with_group[['ndc_11', 'product_group']].drop_duplicates(),
        on='ndc_11', how='inner'
    )

    if len(sdud_with_group) > 0:
        product_util_stats = sdud_with_group.groupby(['product_group', 'year_month']).agg(
            product_medicaid_rx=('medicaid_rx_count', 'sum'),
            product_medicaid_units=('medicaid_units', 'sum'),
            product_medicaid_spending=('medicaid_spending', 'sum'),
        ).reset_index()

        # Also get Part D stats at product level. Part D values are
        # generic-name-level totals replicated across the group's NDCs, so
        # use max (deterministic) rather than first (sort-order dependent).
        if 'partd_total_claims' in sdud_with_group.columns:
            partd_stats = sdud_with_group.groupby(['product_group', 'year_month']).agg(
                product_partd_claims=('partd_total_claims', 'max'),
                product_partd_cost=('partd_avg_cost_per_claim', 'max'),
            ).reset_index()
            product_util_stats = product_util_stats.merge(
                partd_stats, on=['product_group', 'year_month'], how='left')

        # Find unmatched NDCxmonths
        sdud_keys = set(zip(sdud_monthly['ndc_11'], sdud_monthly['year_month']))
        panel_keys = panel_skel[['ndc_11', 'year_month', 'product_group']].drop_duplicates()
        panel_keys['_key'] = list(zip(panel_keys['ndc_11'], panel_keys['year_month']))
        unmatched = panel_keys[~panel_keys['_key'].isin(sdud_keys)].drop(columns=['_key'])

        # Merge product stats onto unmatched
        imputed = unmatched.merge(product_util_stats, on=['product_group', 'year_month'], how='inner')
        print(f"  Unmatched NDCxmonths: {len(unmatched):,}")
        print(f"  Imputable via product-group: {len(imputed):,} ({len(imputed)/max(1,len(unmatched)):.1%})")

        if len(imputed) > 0:
            # For imputed rows, we assign product-level totals (not per-NDC)
            # The model uses these as "total ingredient market volume" features
            imputed_rows = pd.DataFrame({
                'ndc_11': imputed['ndc_11'],
                'year_month': imputed['year_month'],
                'medicaid_rx_count': 0,  # This NDC specifically has no Medicaid Rx
                'medicaid_units': 0,
                'medicaid_spending': 0,
                'medicaid_rx_trend_4q': np.nan,
                'medicaid_units_trend_4q': np.nan,
                'medicaid_rx_cv_4q': np.nan,
                'partd_total_claims': imputed.get('product_partd_claims', np.nan),
                'partd_avg_cost_per_claim': imputed.get('product_partd_cost', np.nan),
                # Part B is not imputed at the ingredient level (we don't carry
                # product-group Part B aggregates), so leave its measurements
                # as NaN but set the indicator to 0 so downstream merges don't
                # see NaN flags.
                'partb_total_claims': np.nan,
                'partb_avg_cost_per_claim': np.nan,
                'partb_total_spending': np.nan,
                'partb_total_beneficiaries': np.nan,
                'has_partb_data': 0,
                'has_medicaid_data': 0,
                'utilization_product_imputed': 1,
            })
            imputed_rows['has_partd_data'] = imputed_rows['partd_total_claims'].notna().astype(int)

            sdud_monthly = pd.concat([sdud_monthly, imputed_rows], ignore_index=True)
            sdud_monthly = sdud_monthly.drop_duplicates(subset=['ndc_11', 'year_month'], keep='first')
            print(f"  Combined rows: {len(sdud_monthly):,}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print("\n[6/6] Saving...")

    # Select final columns
    output_cols = [
        'ndc_11', 'year_month',
        'medicaid_rx_count', 'medicaid_units', 'medicaid_spending',
        'medicaid_rx_trend_4q', 'medicaid_units_trend_4q',
        'medicaid_rx_cv_4q',
        'partd_total_claims', 'partd_avg_cost_per_claim',
        'partb_total_claims', 'partb_avg_cost_per_claim',
        'partb_total_spending', 'partb_total_beneficiaries',
        'has_medicaid_data', 'has_partd_data', 'has_partb_data',
        'utilization_product_imputed',
    ]
    result = sdud_monthly[[c for c in output_cols if c in sdud_monthly.columns]]

    # Save
    output_path = INTERMEDIATE / "utilization.parquet"
    result.to_parquet(output_path, index=False)

    print(f"\n  Output: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  NDCxmonth rows: {len(result):,}")
    print(f"  Unique NDCs: {result['ndc_11'].nunique():,}")
    print(f"  Date range: {result['year_month'].min()} to {result['year_month'].max()}")
    print(f"\n  Column stats:")
    for col in result.columns:
        if col not in ('ndc_11', 'year_month'):
            nulls = result[col].isna().mean()
            print(f"    {col:35s} {nulls:.1%} null")

    print("\nDone!")
    return result


if __name__ == "__main__":
    main()
