"""
05_warning_letters.py - Manufacturer warning letter features (lagged).

Fuzzy-matches CDER warning letter recipients to NDC labeler names,
then computes lagged warning letter counts per labeler per month.

Output: Data/intermediate/warning_letters.parquet
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

try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("WARNING: rapidfuzz not installed. Install with: pip install rapidfuzz")
    raise


# CDER-relevant issuing offices for drug-related warning letters
CDER_OFFICES = [
    'Center for Drug Evaluation and Research',
    'CDER',
]

# Broader set that includes offices relevant to drug manufacturing
DRUG_RELEVANT_OFFICES = [
    'Center for Drug Evaluation and Research',
    'CDER',
    'Office of Pharmaceutical Quality',
    'Office of Manufacturing Quality',
    'Office of Regulatory Affairs',  # ORA does facility inspections
]


def main():
    print("=" * 70)
    print("05_warning_letters.py - Building warning letter features")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and filter warning letters
    # ------------------------------------------------------------------
    print("\n[1/5] Loading warning letters...")
    wl = pd.read_excel(RAW_DATA / "warning-letters.xlsx", dtype=str)
    wl.columns = wl.columns.str.strip()
    print(f"  Total warning letters: {len(wl):,}")

    # Parse dates
    wl['letter_date'] = pd.to_datetime(wl['Letter Issue Date'], errors='coerce')
    wl['posted_date'] = pd.to_datetime(wl['Posted Date'], errors='coerce')
    # Use letter date if available, else posted date
    wl['date'] = wl['letter_date'].fillna(wl['posted_date'])

    # Filter to drug-relevant issuing offices
    # The Issuing Office field may contain partial matches
    print("\n  Issuing Office distribution:")
    print(wl['Issuing Office'].value_counts().head(15).to_string())

    # Filter out clearly non-drug offices
    exclude_offices = [
        'Center for Tobacco Products',
        'Center for Food Safety',
        'Center for Veterinary Medicine',
    ]
    mask = ~wl['Issuing Office'].fillna('').str.contains('|'.join(exclude_offices), case=False, na=False)
    # Also include only offices that are plausibly drug-related
    drug_office_pattern = '|'.join([
        'Drug Evaluation',
        'CDER',
        'Pharmaceutical',
        'Regulatory Affairs',
        'Manufacturing Quality',
        'Biologics',  # CBER - biological drugs
    ])
    mask = wl['Issuing Office'].fillna('').str.contains(drug_office_pattern, case=False, na=False)
    wl_cder = wl[mask].copy()
    print(f"\n  Drug-relevant warning letters: {len(wl_cder):,}")

    # Filter to study period with lookback (2018-2025 to allow 24-month lag)
    wl_cder = wl_cder[wl_cder['date'].notna()]
    wl_cder = wl_cder[(wl_cder['date'] >= '2018-01-01') & (wl_cder['date'] <= '2025-12-31')]
    print(f"  After date filter (2018-2025): {len(wl_cder):,}")

    if len(wl_cder) == 0:
        print("  WARNING: No CDER warning letters found. Saving empty file.")
        empty = pd.DataFrame(columns=['labeler_code', 'year_month',
                                       'warning_letter_6m', 'warning_letter_12m',
                                       'warning_letter_24m'])
        empty.to_parquet(INTERMEDIATE / "warning_letters.parquet", index=False)
        return empty

    # ------------------------------------------------------------------
    # 2. Normalize warning letter company names
    # ------------------------------------------------------------------
    print("\n[2/5] Normalizing company names...")
    wl_cder['company_norm'] = wl_cder['Company Name'].apply(normalize_company_name)
    wl_companies = wl_cder['company_norm'].unique().tolist()
    print(f"  Unique warning letter companies: {len(wl_companies):,}")

    # ------------------------------------------------------------------
    # 3. Load NDC labeler names and fuzzy match
    # ------------------------------------------------------------------
    print("\n[3/5] Fuzzy matching labeler names to warning letter companies...")
    ndc_prod = read_ndc_product(RAW_DATA / "NDC Directory" / "product.txt")
    ndc_pkg = read_ndc_package(RAW_DATA / "NDC Directory" / "package.txt")

    ndc_prod['product_ndc'] = ndc_prod['PRODUCTNDC'].apply(format_productndc)
    ndc_pkg['product_ndc'] = ndc_pkg['PRODUCTNDC'].apply(format_productndc)
    ndc_pkg['ndc_11'] = ndc_pkg['NDCPACKAGECODE'].apply(format_ndcpackagecode)
    ndc_pkg['labeler_code'] = ndc_pkg['ndc_11'].apply(labeler_code_from_ndc)

    # Get unique labelers with their codes
    labeler_map = ndc_prod[['PRODUCTNDC', 'LABELERNAME']].copy()
    labeler_map['product_ndc'] = labeler_map['PRODUCTNDC'].apply(format_productndc)
    labeler_map['labeler_code'] = labeler_map['product_ndc'].apply(
        lambda x: x.split('-')[0] if pd.notna(x) and '-' in str(x) else None
    )
    labelers = labeler_map.dropna(subset=['labeler_code']).drop_duplicates(subset=['labeler_code'])
    labelers['labeler_norm'] = labelers['LABELERNAME'].apply(normalize_company_name)

    # Filter out empty normalized names
    labelers = labelers[labelers['labeler_norm'].str.len() > 2]
    print(f"  Unique labelers: {len(labelers):,}")

    # Fuzzy match each labeler to warning letter companies
    def best_match(name_norm, choices, score_cutoff=85):
        if not name_norm or len(name_norm) < 3:
            return (None, 0)
        res = process.extractOne(
            name_norm, choices, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff
        )
        if res is None:
            return (None, 0)
        return (res[0], res[1])

    print("  Running fuzzy matching (this may take a moment)...")
    matches = labelers['labeler_norm'].apply(
        lambda x: best_match(x, wl_companies, score_cutoff=85)
    )
    labelers['wl_match'] = matches.apply(lambda x: x[0])
    labelers['wl_score'] = matches.apply(lambda x: x[1])

    matched = labelers[labelers['wl_match'].notna()]
    print(f"  Labelers matched to warning letter companies: {len(matched):,}")

    if len(matched) > 0:
        print(f"\n  Sample matches (top 10 by score):")
        sample = matched.nlargest(10, 'wl_score')[['LABELERNAME', 'wl_match', 'wl_score']]
        print(sample.to_string(index=False))

    # Build labeler_code -> wl_company_norm mapping
    labeler_to_wl = matched.set_index('labeler_code')['wl_match'].to_dict()

    # ------------------------------------------------------------------
    # 4. Compute lagged warning letter features per labeler per month
    # ------------------------------------------------------------------
    print("\n[4/5] Computing lagged warning letter features...")
    year_months = generate_year_months(STUDY_START, STUDY_END)

    # Build a time series of warning letters per normalized company name
    wl_cder['year_month_wl'] = wl_cder['date'].dt.to_period('M').astype(str)

    wl_by_company_month = (
        wl_cder.groupby(['company_norm', 'year_month_wl'])
        .size()
        .reset_index(name='n_letters')
    )

    # For each labeler_code x month, compute lagged counts
    records = []
    matched_labelers = list(labeler_to_wl.keys())

    for labeler_code in matched_labelers:
        wl_company = labeler_to_wl[labeler_code]
        company_letters = wl_cder[wl_cder['company_norm'] == wl_company]['date'].sort_values()

        if len(company_letters) == 0:
            continue

        for ym in year_months:
            ym_dt = pd.Timestamp(f"{ym}-01")
            # Count letters in prior 6, 12, 24 months
            n_6m = ((company_letters >= ym_dt - pd.DateOffset(months=6)) &
                    (company_letters < ym_dt)).sum()
            n_12m = ((company_letters >= ym_dt - pd.DateOffset(months=12)) &
                     (company_letters < ym_dt)).sum()
            n_24m = ((company_letters >= ym_dt - pd.DateOffset(months=24)) &
                     (company_letters < ym_dt)).sum()

            if n_24m > 0:  # Only store rows with at least one letter
                records.append({
                    'labeler_code': labeler_code,
                    'year_month': ym,
                    # All three horizons are counts (the 6m horizon was
                    # previously a binary flag, inconsistent with 12m/24m).
                    'warning_letter_6m': int(n_6m),
                    'warning_letter_12m': int(n_12m),
                    'warning_letter_24m': int(n_24m),
                })

    wl_features = pd.DataFrame(records)
    print(f"  Warning letter feature rows: {len(wl_features):,}")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print("\n[5/5] Saving...")
    output_path = INTERMEDIATE / "warning_letters.parquet"
    wl_features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Shape: {wl_features.shape}")
    print(f"  Unique labelers with warning letters: {wl_features['labeler_code'].nunique():,}")

    if len(wl_features) > 0:
        print(f"\n  Warning letter coverage:")
        print(f"  Any letter in 6m:  {wl_features['warning_letter_6m'].mean():.1%} of labeler-months")
        print(f"  Any letter in 12m: {(wl_features['warning_letter_12m'] > 0).mean():.1%}")
        print(f"  Any letter in 24m: {(wl_features['warning_letter_24m'] > 0).mean():.1%}")

    print("\nDone!")
    return wl_features


if __name__ == "__main__":
    main()
