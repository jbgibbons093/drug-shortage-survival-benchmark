"""
04h_api_sourcing.py - Build API (Active Pharmaceutical Ingredient) sourcing
features from the FDA Drug Master File (DMF) database.

Two outputs:
1. Static features per NDC: number of API suppliers, geographic concentration,
   India/China share -> Data/intermediate/api_sourcing.parquet
2. Time-varying features per NDC x month: disaster exposure weighted by API
   source country shares -> Data/intermediate/api_disasters.parquet
"""

import sys
import re
from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("utilities", Path(__file__).parent / "00_utilities.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})


# ── Country inference from DMF holder name ──────────────────────────────

KNOWN_INDIAN = [
    'ALKEM', 'CIPLA', 'DR REDDY', 'SUN PHARMA', 'LUPIN', 'WOCKHARDT',
    'AUROBINDO', 'HETERO', 'LAURUS', 'BIOCON', 'NEULAND', 'IPCA',
    'MANKIND', 'ZYDUS', 'UNICHEM', 'ORCHID', 'SHILPA', 'HARMAN FINO',
    'NECTAR LIFE', 'ALEMBIC', 'HONOUR LAB', 'ALIVUS', 'COHANCE',
    'DIVIS LAB', 'GLENMARK', 'JUBILANT', 'NATCO', 'STRIDES', 'TORRENT',
    'CADILA', 'MSN LAB', 'LAXMI ORGANIC', 'SUVEN', 'SOLARA', 'GRANULES',
    'AARTI', 'HIKAL', 'DIVI', 'MYLAN LAB', 'PIRAMAL', 'OPTIMUS',
    'VASUDHA', 'SYNGENE', 'MEGAFINE', 'VIVIMED', 'SEQUENT',
    'SRI KRISHNA', 'WAVELENGTH', 'PERRIGO API', 'CELLUPRO',
    'CENTAUR PHARMA', 'CHEM TECH', 'SUPRIYA LIFE', 'CTX LIFE',
    'SIGACHI', 'CHIRAL BIOSCIENCES',
]

KNOWN_CHINESE = [
    'WUHAN', 'YANGZHOU', 'WUXI', 'ALLSINO', 'CHENGDA', 'HISOUND',
    'YABAO', 'REGEN-AGEEK', 'SUZHOU NANOMICRO', 'WISDOM PHARMA',
    'ZHUHAI', 'NOVOPROTEIN', 'EL-PEPTIDO', 'EI-PEPTIDO',
    'CHANGZHOU', 'NANJING', 'GUANGZHOU', 'XINHUA', 'LUYE',
    'APELOA', 'PORTON', 'ASYMCHEM',
]

# Map inferred country names to ISO-3 codes (for joining with EM-DAT disaster data)
COUNTRY_TO_ISO3 = {
    'INDIA': 'IND',
    'CHINA': 'CHN',
    'USA': 'USA',
    'ITALY': 'ITA',
    'GERMANY': 'DEU',
    'SWITZERLAND': 'CHE',
    'ISRAEL': 'ISR',
    'JAPAN': 'JPN',
    'SOUTH KOREA': 'KOR',
    'NETHERLANDS': 'NLD',
    'FRANCE': 'FRA',
    'SPAIN': 'ESP',
    'UK': 'GBR',
    'OTHER': None,
    'UNKNOWN': None,
}


def infer_country(holder):
    """Infer country of a DMF holder from their name."""
    if pd.isna(holder):
        return 'UNKNOWN'
    h = holder.upper().strip()

    # Known Indian companies
    if any(x in h for x in KNOWN_INDIAN):
        return 'INDIA'
    if any(x in h for x in [
        'PRIVATE LTD', 'PVT LTD', 'INDIA', 'HYDERABAD', 'MUMBAI',
        'CHENNAI', 'BANGALORE', 'GUJARAT', 'AURANGABAD', 'VIZAG',
        'VISAKHAPATNAM', 'AHMEDABAD', 'PUNE', 'NAGPUR', 'KOLKATA',
        'THANE', 'VADODARA', 'GOA', 'SIKKIM', 'BADDI',
    ]):
        return 'INDIA'

    # Known Chinese companies + location signals
    if any(x in h for x in KNOWN_CHINESE):
        return 'CHINA'
    if any(x in h for x in [
        'SHANGHAI', 'BEIJING', 'ZHEJIANG', 'SHANDONG', 'JIANGSU',
        'HEBEI', 'HUBEI', 'SICHUAN', 'GUANGDONG', 'ANHUI',
        'CHONGQING', 'HANGZHOU', 'SHENZHEN', 'TIANJIN', 'CHINA',
        'HEILONGJIANG', 'HENAN', 'HUNAN', 'FUJIAN', 'LIAONING',
        'NANJING', 'WUHAN', 'SUZHOU', 'DALIAN', 'KUNMING',
        'HAINAN', 'INNER MONGOL', 'JIANGXI', 'QINGDAO',
        'CHANGZHOU', 'YANTAI', 'TAIZHOU', 'NINGBO', 'XIAMEN',
        'CHENGDU', 'LANZHOU', 'GANSU', 'GUIZHOU', 'YUNNAN',
        'SHAANXI', 'SHANXI',
    ]):
        return 'CHINA'

    # Italy
    if any(x in h for x in ['ITALY', 'ITALIA', 'S.P.A', 'S.R.L', 'MILANO']):
        return 'ITALY'
    # "SPA" only counts as the Italian corporate suffix when it ends the
    # name (e.g. "ANGELINI PHARMA SPA"); a mid-name "SPA" is more likely an
    # English trade name and previously produced false Italy assignments.
    if re.search(r'\bS\.?P\.?A\.?\s*$', h) and 'USA' not in h:
        return 'ITALY'

    # Germany
    if any(x in h for x in ['GERMANY', 'GMBH', 'DEUTSCHLAND']):
        return 'GERMANY'

    # Switzerland
    if any(x in h for x in ['SWITZERLAND', 'SIEGFRIED', 'CERBIOS']):
        return 'SWITZERLAND'

    # USA
    if any(x in h for x in [' INC', ' LLC', ' CORP', 'USA', 'U.S.', 'AMERICA']):
        return 'USA'

    # Israel
    if any(x in h for x in ['ISRAEL', 'TEVA ']):
        return 'ISRAEL'

    # Japan / Korea
    if any(x in h for x in ['JAPAN', 'TOKYO', 'OSAKA']):
        return 'JAPAN'
    if any(x in h for x in ['KOREA', 'SEOUL']):
        return 'SOUTH KOREA'

    # Netherlands
    if any(x in h for x in [' BV', 'NETHERLANDS']):
        return 'NETHERLANDS'

    # France
    if any(x in h for x in ['FRANCE', 'PARIS', ' SAS']):
        return 'FRANCE'

    # Spain
    if any(x in h for x in ['SPAIN', 'ESPANA']):
        return 'SPAIN'

    # UK
    if any(x in h for x in ['UNITED KINGDOM', ' PLC']):
        return 'UK'

    # Generic 'LTD' without other signals - likely Indian
    if h.endswith(' LTD') and not any(x in h for x in ['CO LTD', 'BV', 'SA', 'AG']):
        return 'INDIA'

    return 'OTHER'


def clean_api_name(name):
    """Standardize API/ingredient name for matching."""
    if pd.isna(name):
        return ''
    n = name.upper().strip()
    n = re.sub(r'\s+(USP|NF|BP|EP|JP|DRUG SUBSTANCE|MICRONIZED|'
               r'NON-STERILE|STERILE|BULK DRUG|ANHYDROUS)\b', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def load_disaster_data():
    """Load and prepare EM-DAT natural disaster data, aggregated by country x month."""
    disaster_file = RAW_DATA / "Naturaldisasterdata.xlsx"
    if not disaster_file.exists():
        print(f"  WARNING: {disaster_file} not found, skipping disaster features")
        return None

    dis = pd.read_excel(disaster_file, dtype=str)
    print(f"  Raw disasters: {len(dis):,} events")

    # Parse dates
    dis['year'] = pd.to_numeric(dis['Start Year'], errors='coerce')
    dis['month'] = pd.to_numeric(dis['Start Month'], errors='coerce').fillna(6).astype(int)
    dis = dis[dis['year'].notna()].copy()
    dis['year'] = dis['year'].astype(int)

    # Filter to relevant period (2018+ for 24-month lookback from 2020)
    dis = dis[(dis['year'] >= 2018) & (dis['year'] <= 2025)].copy()

    # Filter to relevant disaster types
    relevant_types = [
        'Flood', 'Storm', 'Earthquake', 'Volcanic activity', 'Wildfire',
        'Epidemic', 'Extreme temperature', 'Drought', 'Mass movement (wet)',
        'Industrial accident',
    ]
    dis = dis[dis['Disaster Type'].isin(relevant_types)].copy()

    # Parse damage for major disaster flag
    dis['damage_k'] = pd.to_numeric(dis.get("Total Damage ('000 US$)", pd.Series(dtype=str)),
                                     errors='coerce').fillna(0)
    dis['is_major'] = (dis['damage_k'] > 1_000_000).astype(int)  # > $1B

    # Create year_month
    dis['year_month'] = dis['year'].astype(str) + '-' + dis['month'].apply(lambda m: f'{m:02d}')

    # Aggregate by ISO country code x month
    country_month = dis.groupby(['ISO', 'year_month']).agg(
        disaster_count=('ISO', 'size'),
        major_disaster=('is_major', 'max'),
    ).reset_index()

    print(f"  Disaster country-months: {len(country_month):,}")
    print(f"  Unique countries with disasters: {country_month['ISO'].nunique()}")
    return country_month


def compute_rolling_disasters(country_month, year_months):
    """Compute rolling 3m and 12m disaster counts per country, for each study month."""
    # Build a full country x month grid with disaster counts
    all_countries = country_month['ISO'].unique()

    # Create lookup: (iso, year_month) -> disaster_count, major_disaster
    disaster_lookup = {}
    for _, row in country_month.iterrows():
        disaster_lookup[(row['ISO'], row['year_month'])] = (
            row['disaster_count'], row['major_disaster']
        )

    # For each study month, compute trailing 3m and 12m counts per country
    ym_sorted = sorted(year_months)
    # Build a sequential list of all months from 2018-01 to 2025-12
    all_months = []
    for y in range(2018, 2026):
        for m in range(1, 13):
            all_months.append(f'{y}-{m:02d}')
    ym_to_idx = {ym: i for i, ym in enumerate(all_months)}

    records = []
    for ym in ym_sorted:
        if ym not in ym_to_idx:
            continue
        idx = ym_to_idx[ym]
        for iso in all_countries:
            # Trailing 3 months (strictly before current)
            count_3m = 0
            major_3m = 0
            for offset in range(1, 4):
                if idx - offset >= 0:
                    prev_ym = all_months[idx - offset]
                    val = disaster_lookup.get((iso, prev_ym), (0, 0))
                    count_3m += val[0]
                    major_3m = max(major_3m, val[1])

            # Trailing 12 months
            count_12m = 0
            major_12m = 0
            for offset in range(1, 13):
                if idx - offset >= 0:
                    prev_ym = all_months[idx - offset]
                    val = disaster_lookup.get((iso, prev_ym), (0, 0))
                    count_12m += val[0]
                    major_12m = max(major_12m, val[1])

            if count_3m > 0 or count_12m > 0:
                records.append({
                    'iso3': iso,
                    'year_month': ym,
                    'api_country_disaster_3m': count_3m,
                    'api_country_disaster_12m': count_12m,
                    'api_country_major_disaster_12m': major_12m,
                })

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("04h_api_sourcing.py -- Building API sourcing features from DMF data")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load DMF data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading DMF data...")
    dmf_file = RAW_DATA / "FDA DMF" / "dmf_list_4q2025.xls"
    if not dmf_file.exists():
        print(f"  ERROR: {dmf_file} not found")
        _create_empty_output()
        return

    dmf = pd.read_excel(dmf_file, header=1)
    dmf.columns = ['DMF_NUM', 'STATUS', 'TYPE', 'SUBMIT_DATE', 'HOLDER', 'SUBJECT']
    print(f"  Total DMF records: {len(dmf):,}")

    # Filter to active Type II (drug substance / API).
    #
    # TEMPORAL CAVEATS. The file is a single 4Q2025 snapshot.
    #   1. SUBMIT_DATE is fully populated, so a DMF is only credited to panel
    #      months on or after its submission (enforced below). Without this,
    #      5,254 active Type II DMFs submitted after 2020-01 would be
    #      projected back to the start of the panel.
    #   2. Status is current-as-of-4Q2025 only. A DMF that was active in 2021
    #      but withdrawn by 2025 is invisible (survivor bias), because the
    #      snapshot records no inactivation date. Fixing this would require
    #      historical quarterly DMF lists (FDA publishes the list quarterly;
    #      old snapshots exist via the Internet Archive).
    # The features are therefore best read as "suppliers registered by month
    # t and still active today," a lower bound on point-in-time suppliers.
    at2 = dmf[(dmf['TYPE'] == 'II') & (dmf['STATUS'] == 'A')].copy()
    at2['SUBMIT_DATE'] = pd.to_datetime(at2['SUBMIT_DATE'], errors='coerce')
    at2 = at2.dropna(subset=['SUBMIT_DATE'])
    print(f"  Active Type II (API): {len(at2):,}")
    print(f"  Submitted 2020+ (now time-gated): {(at2['SUBMIT_DATE'] >= '2020-01-01').sum():,}")

    # Infer country
    at2['api_country'] = at2['HOLDER'].apply(infer_country)
    at2['iso3'] = at2['api_country'].map(COUNTRY_TO_ISO3)
    print(f"  Country distribution:")
    for country, count in at2['api_country'].value_counts().head(10).items():
        print(f"    {country:20s}: {count:5,} ({100*count/len(at2):.1f}%)")

    # Clean API names
    at2['api_name_clean'] = at2['SUBJECT'].apply(clean_api_name)
    at2 = at2[at2['api_name_clean'] != ''].copy()
    print(f"  DMFs with valid subject: {len(at2):,}")

    # ------------------------------------------------------------------
    # 2. Load panel drug names
    # ------------------------------------------------------------------
    print("\n[2/7] Loading panel drug names...")
    skel = pd.read_parquet(
        INTERMEDIATE / "panel_skeleton.parquet",
        columns=['ndc_11', 'NONPROPRIETARYNAME', 'year_month']
    )
    skel['drug_name_clean'] = skel['NONPROPRIETARYNAME'].apply(clean_api_name)
    ndc_to_drug = skel[['ndc_11', 'drug_name_clean']].drop_duplicates()
    unique_drugs = ndc_to_drug['drug_name_clean'].unique()
    year_months = sorted(skel['year_month'].unique())
    print(f"  Unique panel drug names: {len(unique_drugs):,}")
    print(f"  Study period: {year_months[0]} to {year_months[-1]} ({len(year_months)} months)")

    # ------------------------------------------------------------------
    # 3. Match DMF subjects to panel drugs
    # ------------------------------------------------------------------
    print("\n[3/7] Matching DMF subjects to panel drugs...")

    dmf_by_api = at2.groupby('api_name_clean').apply(
        lambda g: g[['HOLDER', 'api_country', 'iso3', 'SUBMIT_DATE']].to_dict('records'),
        include_groups=False
    ).to_dict()

    panel_drug_set = set(unique_drugs)
    dmf_api_set = set(dmf_by_api.keys())
    matched_names = panel_drug_set & dmf_api_set
    print(f"  Direct matches: {len(matched_names):,} / {len(panel_drug_set):,} "
          f"({100*len(matched_names)/len(panel_drug_set):.1f}%)")

    # Multi-ingredient matching
    multi_ingredient = [d for d in unique_drugs if ';' in d]
    extra_matches = {}
    for drug in multi_ingredient:
        components = [clean_api_name(c) for c in drug.split(';')]
        all_holders = []
        for comp in components:
            if comp in dmf_by_api:
                all_holders.extend(dmf_by_api[comp])
        if all_holders:
            extra_matches[drug] = all_holders

    print(f"  Multi-ingredient matches: {len(extra_matches):,} / {len(multi_ingredient):,}")

    all_matches = {}
    for name in matched_names:
        all_matches[name] = dmf_by_api[name]
    all_matches.update(extra_matches)
    print(f"  Total matched drug names: {len(all_matches):,}")

    # ------------------------------------------------------------------
    # 4. Compute TIME-VARYING features per drug name x month
    # ------------------------------------------------------------------
    # A DMF counts toward month t only if it was submitted by the end of t.
    # This replaces the prior static computation, which credited every
    # currently-active DMF to all panel months regardless of when it was
    # registered. has_api_data stays a name-match flag (source coverage),
    # while n_api_suppliers can legitimately be 0 in early months for a
    # matched drug whose suppliers registered later.
    print("\n[4/7] Computing time-varying API sourcing features (gated on SUBMIT_DATE)...")
    month_ends = {ym: pd.Period(ym, freq='M').to_timestamp(how='end')
                  for ym in year_months}

    records = []
    for drug_name, holders in all_matches.items():
        holders_sorted = sorted(holders, key=lambda h: h['SUBMIT_DATE'])
        submit_dates = [h['SUBMIT_DATE'] for h in holders_sorted]
        countries = [h['api_country'] for h in holders_sorted]

        prev_k = -1
        prev_row = None
        for ym in year_months:
            cutoff = month_ends[ym]
            k = 0
            for d in submit_dates:
                if d <= cutoff:
                    k += 1
                else:
                    break
            if k == prev_k:
                row = dict(prev_row)
                row['year_month'] = ym
            elif k == 0:
                row = {
                    'drug_name_clean': drug_name, 'year_month': ym,
                    'n_api_suppliers': 0, 'n_api_countries': 0,
                    'api_india_share': 0.0, 'api_china_share': 0.0,
                    'api_us_share': 0.0, 'api_india_china_share': 0.0,
                    'api_country_hhi': 0.0,
                }
            else:
                active_countries = countries[:k]
                country_counts = Counter(active_countries)
                n_india = country_counts.get('INDIA', 0)
                n_china = country_counts.get('CHINA', 0)
                n_usa = country_counts.get('USA', 0)
                hhi = sum((cnt / k) ** 2 for cnt in country_counts.values())
                row = {
                    'drug_name_clean': drug_name, 'year_month': ym,
                    'n_api_suppliers': k,
                    'n_api_countries': len(country_counts),
                    'api_india_share': n_india / k,
                    'api_china_share': n_china / k,
                    'api_us_share': n_usa / k,
                    'api_india_china_share': (n_india + n_china) / k,
                    'api_country_hhi': hhi,
                }
            records.append(row)
            prev_k = k
            prev_row = row

    api_features = pd.DataFrame(records)
    print(f"  Drug-month feature rows: {len(api_features):,} "
          f"({api_features['drug_name_clean'].nunique():,} drugs x {len(year_months)} months)")
    first_m, last_m = year_months[0], year_months[-1]
    for m in (first_m, last_m):
        sub = api_features[api_features['year_month'] == m]
        print(f"  Mean suppliers in {m}: {sub['n_api_suppliers'].mean():.1f}")

    # Map to NDC x month and save
    skel_keys = skel[['ndc_11', 'year_month', 'drug_name_clean']].drop_duplicates()
    ndc_features = skel_keys.merge(api_features, on=['drug_name_clean', 'year_month'], how='inner')
    ndc_features['has_api_data'] = 1
    for col in ['n_api_suppliers', 'n_api_countries', 'has_api_data']:
        ndc_features[col] = ndc_features[col].astype(np.int16)
    for col in ['api_india_share', 'api_china_share', 'api_us_share',
                'api_india_china_share', 'api_country_hhi']:
        ndc_features[col] = ndc_features[col].astype(np.float32)

    static_output = ndc_features[['ndc_11', 'year_month', 'n_api_suppliers', 'n_api_countries',
                                   'api_india_share', 'api_china_share', 'api_us_share',
                                   'api_india_china_share', 'api_country_hhi',
                                   'has_api_data']]

    static_path = INTERMEDIATE / "api_sourcing.parquet"
    static_output.to_parquet(static_path, index=False)
    print(f"  Saved time-varying features to {static_path}")
    print(f"  NDC-month rows with API data: {len(static_output):,} "
          f"({static_output['ndc_11'].nunique():,} NDCs)")

    # ------------------------------------------------------------------
    # 5. Load disaster data
    # ------------------------------------------------------------------
    print("\n[5/7] Loading disaster data...")
    country_month_disasters = load_disaster_data()
    if country_month_disasters is None:
        print("  Skipping API disaster features (no disaster data)")
        _create_empty_disaster_output()
        return

    # ------------------------------------------------------------------
    # 6. Compute rolling disaster counts per country
    # ------------------------------------------------------------------
    print("\n[6/7] Computing rolling disaster counts per country...")
    rolling_disasters = compute_rolling_disasters(country_month_disasters, year_months)
    print(f"  Country-month disaster records: {len(rolling_disasters):,}")

    # ------------------------------------------------------------------
    # 7. Compute weighted API disaster exposure per drug x month
    # ------------------------------------------------------------------
    print("\n[7/7] Computing weighted API disaster exposure per NDC x month...")

    # Build per-drug country share vectors
    # drug_name_clean -> {iso3: share}
    # Disaster weighting uses the snapshot's full holder set rather
    # than the time-gated set. Country composition shifts slowly, and the
    # disaster counts themselves are strictly backward-looking, so the
    # static shares are an acceptable approximation here.
    drug_country_shares = {}
    for drug_name, holders in all_matches.items():
        iso_codes = [h['iso3'] for h in holders if h['iso3'] is not None]
        if not iso_codes:
            continue
        total = len(iso_codes)
        shares = Counter(iso_codes)
        drug_country_shares[drug_name] = {iso: cnt / total for iso, cnt in shares.items()}

    print(f"  Drugs with ISO-mapped API countries: {len(drug_country_shares):,}")

    # Create disaster lookup: (iso3, year_month) -> (3m, 12m, major_12m)
    dis_lookup = {}
    for _, row in rolling_disasters.iterrows():
        dis_lookup[(row['iso3'], row['year_month'])] = (
            row['api_country_disaster_3m'],
            row['api_country_disaster_12m'],
            row['api_country_major_disaster_12m'],
        )

    # For each drug x month, compute weighted disaster exposure
    # Weighted = sum over API source countries of (country_share * country_disaster_count)
    drug_name_to_ndcs = ndc_to_drug.groupby('drug_name_clean')['ndc_11'].apply(list).to_dict()

    disaster_records = []
    n_drugs_processed = 0
    for drug_name, country_shares in drug_country_shares.items():
        ndcs = drug_name_to_ndcs.get(drug_name, [])
        if not ndcs:
            continue

        for ym in year_months:
            weighted_3m = 0.0
            weighted_12m = 0.0
            any_major_12m = 0

            for iso, share in country_shares.items():
                vals = dis_lookup.get((iso, ym), (0, 0, 0))
                weighted_3m += share * vals[0]
                weighted_12m += share * vals[1]
                if vals[2] > 0:
                    any_major_12m = 1

            for ndc in ndcs:
                disaster_records.append({
                    'ndc_11': ndc,
                    'year_month': ym,
                    'api_disaster_exposure_3m': round(weighted_3m, 4),
                    'api_disaster_exposure_12m': round(weighted_12m, 4),
                    'api_major_disaster_12m': any_major_12m,
                })

        n_drugs_processed += 1
        if n_drugs_processed % 200 == 0:
            print(f"    Processed {n_drugs_processed:,} / {len(drug_country_shares):,} drugs...")

    api_disasters = pd.DataFrame(disaster_records)
    print(f"  Total API disaster records: {len(api_disasters):,}")

    # Downcast for memory
    api_disasters['api_disaster_exposure_3m'] = api_disasters['api_disaster_exposure_3m'].astype(np.float32)
    api_disasters['api_disaster_exposure_12m'] = api_disasters['api_disaster_exposure_12m'].astype(np.float32)
    api_disasters['api_major_disaster_12m'] = api_disasters['api_major_disaster_12m'].astype(np.int8)

    disaster_path = INTERMEDIATE / "api_disasters.parquet"
    api_disasters.to_parquet(disaster_path, index=False)
    print(f"\n  Saved to {disaster_path}")
    print(f"  Shape: {api_disasters.shape}")
    print(f"  NDCs with disaster exposure: {api_disasters['ndc_11'].nunique():,}")

    # Summary stats
    has_exposure = api_disasters['api_disaster_exposure_12m'] > 0
    print(f"  NDC-months with any API disaster exposure (12m): {has_exposure.mean():.1%}")
    print(f"  Mean exposure (when >0): {api_disasters.loc[has_exposure, 'api_disaster_exposure_12m'].mean():.2f}")
    print(f"  Mean exposure (all): {api_disasters['api_disaster_exposure_12m'].mean():.2f}")
    print(f"  Any major disaster (12m): {api_disasters['api_major_disaster_12m'].mean():.1%}")

    print("\nDone!")


def _create_empty_output():
    """Create empty static output."""
    output = pd.DataFrame(columns=[
        'ndc_11', 'n_api_suppliers', 'n_api_countries',
        'api_india_share', 'api_china_share', 'api_us_share',
        'api_india_china_share', 'api_country_hhi', 'has_api_data'
    ])
    (INTERMEDIATE / "api_sourcing.parquet").write_bytes(b'')
    output.to_parquet(INTERMEDIATE / "api_sourcing.parquet", index=False)
    _create_empty_disaster_output()


def _create_empty_disaster_output():
    """Create empty disaster output."""
    output = pd.DataFrame(columns=[
        'ndc_11', 'year_month',
        'api_disaster_exposure_3m', 'api_disaster_exposure_12m',
        'api_major_disaster_12m',
    ])
    output.to_parquet(INTERMEDIATE / "api_disasters.parquet", index=False)
    print(f"  Empty disaster output saved")


if __name__ == "__main__":
    main()
