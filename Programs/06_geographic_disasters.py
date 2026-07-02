"""
06_geographic_disasters.py - Manufacturer location + natural disaster exposure.

Links manufacturer facilities to countries, then computes disaster exposure
features per labeler per month.

Output: Data/intermediate/geographic_disasters.parquet
"""

import sys
import re
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("utilities", Path(__file__).parent / "00_utilities.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})

try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("WARNING: rapidfuzz not installed.")
    raise

# Regex to extract ISO country code from ADDRESS field: "(XXX)" at end
COUNTRY_RE = re.compile(r'\(([A-Z]{2,3})\)\s*$')

# Disaster types relevant to pharmaceutical supply chain
RELEVANT_DISASTER_TYPES = [
    'Flood', 'Storm', 'Earthquake', 'Volcanic activity', 'Wildfire',
    'Epidemic', 'Extreme temperature', 'Drought', 'Mass movement (wet)',
    'Industrial accident',
]


def extract_country_from_address(address):
    """Extract ISO country code from facility address field."""
    if pd.isna(address):
        return None
    m = COUNTRY_RE.search(str(address))
    if m:
        code = m.group(1)
        return code
    return None


def main():
    print("=" * 70)
    print("06_geographic_disasters.py - Building geographic & disaster features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load manufacturer locations
    # ------------------------------------------------------------------
    print("\n[1/6] Loading manufacturer locations...")
    zip_path = RAW_DATA / "Drug Manufacturer locations.zip"

    # Read from zip - the txt file has an extra trailing tab in data rows,
    # so we manually parse to avoid column misalignment
    import io
    with zipfile.ZipFile(zip_path, 'r') as z:
        txt_files = [f for f in z.namelist() if f.endswith('.txt')]
        xls_files = [f for f in z.namelist() if f.endswith('.xls') or f.endswith('.xlsx')]

        loaded = False
        if txt_files:
            with z.open(txt_files[0]) as f:
                raw = f.read()
            text = raw.decode('cp1252')
            lines = text.strip().split('\n')
            # Parse header (strip whitespace from each field)
            header = [h.strip() for h in lines[0].split('\t')]
            n_cols = len(header)
            # Parse data rows (take only first n_cols fields to handle trailing tab)
            data_rows = []
            for line in lines[1:]:
                fields = [f.strip() for f in line.split('\t')][:n_cols]
                if len(fields) == n_cols:
                    data_rows.append(fields)
            facilities = pd.DataFrame(data_rows, columns=header)
            loaded = True
            print(f"  Read {txt_files[0]} ({len(facilities):,} rows)")

        if not loaded and xls_files:
            with z.open(xls_files[0]) as f:
                facilities = pd.read_excel(io.BytesIO(f.read()), dtype=str)
            loaded = True

        if not loaded:
            raise FileNotFoundError(f"No suitable file found in {zip_path}")

    facilities.columns = facilities.columns.str.strip()
    print(f"  Total facilities: {len(facilities):,}")
    print(f"  Columns: {list(facilities.columns)}")

    # ------------------------------------------------------------------
    # 2. Extract country and filter to manufacturing operations
    # ------------------------------------------------------------------
    print("\n[2/6] Extracting country codes and filtering to manufacturers...")
    facilities['country_iso'] = facilities['ADDRESS'].apply(extract_country_from_address)

    n_with_country = facilities['country_iso'].notna().sum()
    print(f"  Facilities with country code: {n_with_country:,}")

    # Filter to manufacturing operations
    if 'OPERATIONS' in facilities.columns:
        mfr_ops = facilities['OPERATIONS'].fillna('').str.upper()
        is_manufacturer = mfr_ops.str.contains('MANUFACTURE', na=False)
        mfr_facilities = facilities[is_manufacturer].copy()
        print(f"  Manufacturing facilities: {len(mfr_facilities):,}")
    else:
        mfr_facilities = facilities.copy()
        print("  WARNING: No OPERATIONS column, using all facilities")

    # Country distribution
    print(f"\n  Top manufacturing countries:")
    print(mfr_facilities['country_iso'].value_counts().head(15).to_string())

    # ------------------------------------------------------------------
    # 3. Match facilities to NDC labelers
    # ------------------------------------------------------------------
    print("\n[3/6] Matching facilities to NDC labelers...")

    # Normalize facility names
    mfr_facilities['firm_norm'] = mfr_facilities['FIRM_NAME'].apply(normalize_company_name)
    if 'REGISTRANT_NAME' in mfr_facilities.columns:
        mfr_facilities['registrant_norm'] = mfr_facilities['REGISTRANT_NAME'].apply(normalize_company_name)

    # --- Build labeler list from BOTH current NDC Directory AND skeleton ---
    # Current NDC Directory
    ndc_prod = read_ndc_product(RAW_DATA / "NDC Directory" / "product.txt")
    ndc_prod['product_ndc'] = ndc_prod['PRODUCTNDC'].apply(format_productndc)
    ndc_prod['labeler_code'] = ndc_prod['product_ndc'].apply(
        lambda x: x.split('-')[0] if pd.notna(x) and '-' in str(x) else None
    )
    labelers_current = ndc_prod[['labeler_code', 'LABELERNAME']].dropna(subset=['labeler_code'])
    labelers_current = labelers_current.drop_duplicates(subset=['labeler_code'])

    # Historical labelers from panel skeleton
    skeleton_path = INTERMEDIATE / "panel_skeleton.parquet"
    if skeleton_path.exists():
        skel = pd.read_parquet(skeleton_path, columns=['labeler_code', 'LABELERNAME'])
        skel_labelers = skel.drop_duplicates(subset=['labeler_code'])
        # Only add labelers not in current directory
        current_codes = set(labelers_current['labeler_code'])
        skel_new = skel_labelers[~skel_labelers['labeler_code'].isin(current_codes)]
        labelers = pd.concat([labelers_current, skel_new[['labeler_code', 'LABELERNAME']]], ignore_index=True)
        labelers = labelers.drop_duplicates(subset=['labeler_code'])
        print(f"  Labelers from current NDC: {len(labelers_current):,}")
        print(f"  Additional from skeleton: {len(skel_new):,}")
    else:
        labelers = labelers_current

    labelers['labeler_norm'] = labelers['LABELERNAME'].apply(normalize_company_name)
    labelers = labelers[labelers['labeler_norm'].str.len() > 2]
    print(f"  Total unique labelers: {len(labelers):,}")

    # Build facility name lookup sets
    firm_norm_set = set(mfr_facilities['firm_norm'].unique())
    registrant_norm_set = set()
    if 'registrant_norm' in mfr_facilities.columns:
        registrant_norm_set = set(mfr_facilities['registrant_norm'].unique())
    all_facility_norms = firm_norm_set | registrant_norm_set
    all_facility_names = [n for n in all_facility_norms if len(n) > 3]
    print(f"  Unique facility/registrant names: {len(all_facility_names):,}")

    # --- Step A: Exact normalized name match (fast) ---
    print("  Step A: Exact normalized name matching...")
    name_to_facility = {n: n for n in all_facility_norms}
    labelers['facility_match'] = labelers['labeler_norm'].map(name_to_facility)
    labelers['match_score'] = labelers['facility_match'].apply(lambda x: 100 if pd.notna(x) else 0)
    n_exact = labelers['facility_match'].notna().sum()
    print(f"    Exact matches: {n_exact:,}")

    # --- Step B: Fuzzy match for remaining (lower threshold) ---
    unmatched_mask = labelers['facility_match'].isna()
    n_to_fuzzy = unmatched_mask.sum()
    print(f"  Step B: Fuzzy matching {n_to_fuzzy:,} remaining labelers...")

    def best_match(name_norm, choices, score_cutoff=80):
        if not name_norm or len(name_norm) < 3:
            return (None, 0)
        res = process.extractOne(
            name_norm, choices, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff
        )
        if res is None:
            return (None, 0)
        return (res[0], res[1])

    if n_to_fuzzy > 0:
        fuzzy_results = labelers.loc[unmatched_mask, 'labeler_norm'].apply(
            lambda x: best_match(x, all_facility_names, score_cutoff=80)
        )
        labelers.loc[unmatched_mask, 'facility_match'] = fuzzy_results.apply(lambda x: x[0])
        labelers.loc[unmatched_mask, 'match_score'] = fuzzy_results.apply(lambda x: x[1])
        n_fuzzy = labelers.loc[unmatched_mask, 'facility_match'].notna().sum()
        print(f"    Fuzzy matches: {n_fuzzy:,}")

    matched = labelers[labelers['facility_match'].notna()]
    print(f"  Total labelers matched: {len(matched):,} / {len(labelers):,} ({len(matched)/len(labelers):.1%})")

    # For matched labelers, get their countries and facility counts
    # Build facility_norm -> country mapping
    facility_country = (
        mfr_facilities.groupby('firm_norm')
        .agg(
            countries=('country_iso', lambda x: list(x.dropna().unique())),
            n_facilities=('FEI_NUMBER', 'nunique'),
        )
        .reset_index()
    )

    # Also add registrant matches
    if 'registrant_norm' in mfr_facilities.columns:
        reg_country = (
            mfr_facilities.groupby('registrant_norm')
            .agg(
                countries=('country_iso', lambda x: list(x.dropna().unique())),
                n_facilities=('FEI_NUMBER', 'nunique'),
            )
            .reset_index()
            .rename(columns={'registrant_norm': 'firm_norm'})
        )
        facility_country = pd.concat([facility_country, reg_country]).drop_duplicates(subset=['firm_norm'])

    labeler_geo = matched.merge(
        facility_country.rename(columns={'firm_norm': 'facility_match'}),
        on='facility_match', how='left'
    )

    # Extract primary country (most common)
    labeler_geo['primary_country'] = labeler_geo['countries'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    labeler_geo['is_domestic'] = labeler_geo['primary_country'].apply(
        lambda x: 1 if x in ('USA', 'US') else 0
    )
    labeler_geo['n_countries'] = labeler_geo['countries'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Build labeler_code -> geographic features mapping
    geo_static = labeler_geo[['labeler_code', 'primary_country', 'is_domestic',
                               'n_facilities', 'n_countries']].copy()
    geo_static['n_facilities'] = geo_static['n_facilities'].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 4. Load natural disasters
    # ------------------------------------------------------------------
    print("\n[4/6] Loading natural disaster data...")
    disasters = pd.read_excel(RAW_DATA / "Naturaldisasterdata.xlsx", dtype=str)
    disasters.columns = disasters.columns.str.strip()
    print(f"  Total disaster events: {len(disasters):,}")

    # Parse dates
    disasters['start_year'] = pd.to_numeric(disasters.get('Start Year', pd.Series(dtype=str)), errors='coerce')
    disasters['start_month'] = pd.to_numeric(disasters.get('Start Month', pd.Series(dtype=str)), errors='coerce')
    disasters['start_month'] = disasters['start_month'].fillna(6).astype(int)  # Default to mid-year

    # Build approximate start date
    disasters['disaster_date'] = pd.to_datetime(
        disasters['start_year'].astype(int).astype(str) + '-' +
        disasters['start_month'].astype(int).astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )

    # Filter to relevant period (2018-2025 for 24-month lookback)
    disasters = disasters[disasters['disaster_date'].notna()]
    disasters = disasters[(disasters['disaster_date'] >= '2018-01-01') &
                          (disasters['disaster_date'] <= '2025-12-31')]
    print(f"  Disasters in 2018-2025: {len(disasters):,}")

    # Filter to relevant disaster types
    if 'Disaster Type' in disasters.columns:
        disasters_relevant = disasters[disasters['Disaster Type'].isin(RELEVANT_DISASTER_TYPES)].copy()
        print(f"  Relevant disaster types: {len(disasters_relevant):,}")
    else:
        disasters_relevant = disasters.copy()

    # Parse damage for major disaster flag
    disasters_relevant['total_damage'] = pd.to_numeric(
        disasters_relevant.get("Total Damage ('000 US$)", pd.Series(dtype=str)),
        errors='coerce'
    )

    # Get ISO codes
    disasters_relevant['iso'] = disasters_relevant.get('ISO', pd.Series(dtype=str)).fillna('')

    print(f"\n  Top disaster countries:")
    print(disasters_relevant['iso'].value_counts().head(10).to_string())

    # ------------------------------------------------------------------
    # 5. Compute disaster exposure per labeler per month
    # ------------------------------------------------------------------
    print("\n[5/6] Computing disaster exposure features...")

    # Get all countries per labeler
    labeler_countries = {}
    for _, row in labeler_geo.iterrows():
        lc = row['labeler_code']
        countries = row['countries'] if isinstance(row['countries'], list) else []
        labeler_countries[lc] = set(countries)

    year_months = generate_year_months(STUDY_START, STUDY_END)
    disaster_records = []

    for labeler_code, countries in labeler_countries.items():
        if not countries:
            continue

        # Filter disasters to this labeler's countries
        country_disasters = disasters_relevant[disasters_relevant['iso'].isin(countries)]
        if len(country_disasters) == 0:
            continue

        disaster_dates = country_disasters['disaster_date']
        disaster_damages = country_disasters['total_damage']

        for ym in year_months:
            ym_dt = pd.Timestamp(f"{ym}-01")

            # Count disasters in prior 3, 12 months
            d_3m = ((disaster_dates >= ym_dt - pd.DateOffset(months=3)) &
                    (disaster_dates < ym_dt)).sum()
            d_12m = ((disaster_dates >= ym_dt - pd.DateOffset(months=12)) &
                     (disaster_dates < ym_dt)).sum()

            # Major disasters (damage > $1B = 1,000,000 in '000 US$)
            recent_12m = ((disaster_dates >= ym_dt - pd.DateOffset(months=12)) &
                          (disaster_dates < ym_dt))
            major_12m = (recent_12m & (disaster_damages > 1_000_000)).sum()

            disaster_records.append({
                'labeler_code': labeler_code,
                'year_month': ym,
                'disaster_count_3m': int(d_3m),
                'disaster_count_12m': int(d_12m),
                'major_disaster_12m': int(major_12m),
            })

    disaster_features = pd.DataFrame(disaster_records)
    print(f"  Disaster feature rows: {len(disaster_features):,}")

    # ------------------------------------------------------------------
    # 6. Combine static and time-varying features
    # ------------------------------------------------------------------
    print("\n[6/6] Combining and saving...")

    # Merge static geo features with disaster features
    if len(disaster_features) > 0:
        geo_combined = disaster_features.merge(geo_static, on='labeler_code', how='outer')
    else:
        geo_combined = geo_static.copy()
        for col in ['year_month', 'disaster_count_3m', 'disaster_count_12m', 'major_disaster_12m']:
            geo_combined[col] = np.nan

    # For labelers with geo info but no disaster exposure, add zero-disaster rows
    labelers_with_geo = set(geo_static['labeler_code'].unique())
    labelers_with_disasters = set(disaster_features['labeler_code'].unique()) if len(disaster_features) > 0 else set()
    labelers_no_disasters = labelers_with_geo - labelers_with_disasters

    if labelers_no_disasters:
        zero_records = []
        for lc in labelers_no_disasters:
            for ym in year_months:
                zero_records.append({
                    'labeler_code': lc,
                    'year_month': ym,
                    'disaster_count_3m': 0,
                    'disaster_count_12m': 0,
                    'major_disaster_12m': 0,
                })
        zero_df = pd.DataFrame(zero_records)
        zero_df = zero_df.merge(geo_static, on='labeler_code', how='left')
        geo_combined = pd.concat([
            geo_combined.dropna(subset=['year_month']),
            zero_df
        ], ignore_index=True)

    # Fill NaN disaster counts with 0
    for col in ['disaster_count_3m', 'disaster_count_12m', 'major_disaster_12m']:
        if col in geo_combined.columns:
            geo_combined[col] = geo_combined[col].fillna(0).astype(int)

    geo_combined = geo_combined.drop_duplicates(subset=['labeler_code', 'year_month'])

    output_path = INTERMEDIATE / "geographic_disasters.parquet"
    geo_combined.to_parquet(output_path, index=False)
    print(f"\n  Saved to {output_path}")
    print(f"  Shape: {geo_combined.shape}")
    print(f"  Unique labelers: {geo_combined['labeler_code'].nunique():,}")
    if 'is_domestic' in geo_combined.columns:
        domestic_pct = geo_combined.drop_duplicates('labeler_code')['is_domestic'].mean()
        print(f"  Domestic manufacturers: {domestic_pct:.1%}")
    if 'disaster_count_12m' in geo_combined.columns:
        any_disaster = (geo_combined['disaster_count_12m'] > 0).mean()
        print(f"  Labeler-months with disaster exposure: {any_disaster:.1%}")

    print("\nDone!")
    return geo_combined


if __name__ == "__main__":
    main()
