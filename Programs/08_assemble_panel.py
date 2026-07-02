"""
08_assemble_panel.py - Merge all features onto the NDC-month panel skeleton.

Performs left joins of all feature files onto the panel skeleton,
adds temporal features and lagged shortage history.

Output: Data/analysis/drug_shortage_panel.parquet
"""

import os
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


DROP_REPACKAGERS = os.environ.get("DROP_REPACKAGERS", "0") == "1"
PANEL_OUTPUT_PATH = Path(
    os.environ.get("PANEL_OUTPUT_PATH", str(ANALYSIS / "drug_shortage_panel.parquet"))
)


def add_missing_flag_and_fill(panel, col, fill_value=0):
    """Preserve missingness as a feature, then fill a numeric covariate."""
    if col not in panel.columns:
        return panel
    panel[f'{col}_missing'] = panel[col].isna().astype(int)
    panel[col] = panel[col].fillna(fill_value)
    return panel


def normalize_structural_missingness(panel):
    """Make source-family missingness explicit before model training.

    Several covariate families are structurally unavailable for many NDC-months
    because the source covers only selected drugs, products, or manufacturers.
    The model should see both the filled value and the fact that the source was
    unavailable, instead of relying on downstream median imputation.
    """
    exact_fills = {
        # Market structure fallback. A zero value means unavailable once paired
        # with the *_missing flag, not a literal zero-market product.
        'n_manufacturers': 0,
        'n_applications': 0,
        'sole_source': 0,
        'recent_generic_entry': 0,
        'recent_manufacturer_exit': 0,

        # Geographic source coverage. is_domestic uses -1 so unknown is not
        # collapsed into known foreign manufacture.
        'is_domestic': -1,
        'n_facilities': 0,
        'n_countries': 0,

        # NADAC pricing. Missing means no usable acquisition-cost observation
        # for that NDC-month or no computable trend.
        'nadac_per_unit': 0.0,
        'nadac_pct_change_3m': 0.0,
        'nadac_pct_change_12m': 0.0,
        'nadac_vs_market_median': 0.0,

        # Product age can be missing when historical marketing dates are absent.
        'years_on_market': 0.0,
    }

    for col, fill_value in exact_fills.items():
        panel = add_missing_flag_and_fill(panel, col, fill_value)

    prefix_fills = (
        'asp_',
        'medicaid_',
        'partd_',
        'partb_',
        'symphony_',
    )
    skip_prefix_cols = {
        'asp_quarter',
    }
    for col in list(panel.columns):
        if col in skip_prefix_cols:
            continue
        if not any(col.startswith(prefix) for prefix in prefix_fills):
            continue
        if col.startswith('has_') or col.endswith('_missing') or col.endswith('_is_observed'):
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            panel = add_missing_flag_and_fill(panel, col, 0.0)

    missing_flag_cols = [c for c in panel.columns if c.endswith('_missing')]
    for col in missing_flag_cols:
        panel[col] = panel[col].fillna(0).astype(np.int8)

    print(f"  Added structural missingness flags: {len(missing_flag_cols)}")
    return panel


def main():
    print("=" * 70)
    print("08_assemble_panel.py - Assembling final drug shortage panel")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load panel skeleton
    # ------------------------------------------------------------------
    print("\n[1/8] Loading panel skeleton...")
    panel = pd.read_parquet(INTERMEDIATE / "panel_skeleton.parquet")
    print(f"  Panel skeleton: {panel.shape[0]:,} rows, {panel['ndc_11'].nunique():,} NDCs")

    # ------------------------------------------------------------------
    # 1b. Identify repackagers and compute repackager count feature
    # ------------------------------------------------------------------
    print("\n[1b] Identifying repackagers...")
    # Known repackager/relabeler keywords
    REPACKAGER_KEYWORDS = [
        'prepack', 'repack', 'remedyrepack', 'apotheca', 'pd-rx',
        'a-s medication', 'nelco lab', 'nucare', 'proficient rx',
        'aphena pharma', 'asclemed', 'rpk pharma', 'aidarex',
        'contract pharmacy', 'northwind', 'denton pharma',
        'preferred pharmaceut', 'rebel distribut',
        'unit dose', 'readymeds', 'direct rx',
        'golden state medical', 'hj harkins', 'h.j. harkins',
        'southwood', 'dispensing solutions',
        'stat rx', 'physicians total care', 'ncs healthcare',
        'liberty pharma', 'rxpak', 'avpak',
        'mckesson packaging', 'cardinal health repack',
        'redpharm', 'lake erie medical', 'med-health pharma',
        'quality care', 'clinical solutions', 'bi-coastal pharma',
    ]

    labeler_name_col = 'LABELERNAME' if 'LABELERNAME' in panel.columns else 'labeler_name'
    name_upper = panel[labeler_name_col].fillna('').str.upper()
    panel['is_repackager'] = name_upper.apply(
        lambda x: any(kw.upper() in x for kw in REPACKAGER_KEYWORDS)
    )
    n_repack = panel['is_repackager'].sum()
    n_repack_ndcs = panel.loc[panel['is_repackager'], 'ndc_11'].nunique()
    print(f"  Repackager rows: {n_repack:,} ({n_repack/len(panel)*100:.1f}%)")
    print(f"  Repackager NDCs: {n_repack_ndcs:,}")

    # Compute n_repackagers per drug (ingredient + dosage form + route) per month
    # Use NONPROPRIETARYNAME, DOSAGEFORMNAME, ROUTENAME from the skeleton
    ingr_col = 'NONPROPRIETARYNAME' if 'NONPROPRIETARYNAME' in panel.columns else 'nonproprietary_name'
    form_col = 'DOSAGEFORMNAME' if 'DOSAGEFORMNAME' in panel.columns else 'dosage_form'
    route_col = 'ROUTENAME' if 'ROUTENAME' in panel.columns else 'route'

    repack_rows = panel[panel['is_repackager']].copy()
    repack_rows['_ingr'] = repack_rows[ingr_col].fillna('').str.upper().str.strip()
    repack_rows['_form'] = repack_rows[form_col].fillna('').str.upper().str.strip()
    repack_rows['_route'] = repack_rows[route_col].fillna('').str.upper().str.strip()

    repack_counts = repack_rows.groupby(
        ['_ingr', '_form', '_route', 'year_month']
    )['labeler_code'].nunique().reset_index()
    repack_counts.columns = ['_ingr', '_form', '_route', 'year_month', 'n_repackagers']

    if DROP_REPACKAGERS:
        panel = panel[~panel['is_repackager']].copy()
        panel['repackager_rows_retained'] = 0
        print(f"  DROP_REPACKAGERS=1, panel after removal: {len(panel):,} rows, {panel['ndc_11'].nunique():,} NDCs")
    else:
        panel['repackager_rows_retained'] = 1
        print("  Repackager rows retained with is_repackager flag")
        print(f"  Panel after flagging: {len(panel):,} rows, {panel['ndc_11'].nunique():,} NDCs")

    # Merge repackager count onto manufacturer NDCs
    panel['_ingr'] = panel[ingr_col].fillna('').str.upper().str.strip()
    panel['_form'] = panel[form_col].fillna('').str.upper().str.strip()
    panel['_route'] = panel[route_col].fillna('').str.upper().str.strip()
    panel = panel.merge(repack_counts, on=['_ingr', '_form', '_route', 'year_month'], how='left')
    panel['n_repackagers'] = panel['n_repackagers'].fillna(0).astype(int)
    panel['is_repackager'] = panel['is_repackager'].fillna(False).astype(int)
    panel.drop(columns=['_ingr', '_form', '_route'], inplace=True)
    has_repack = (panel['n_repackagers'] > 0).mean()
    print(f"  NDC-months with repackagers: {has_repack:.1%}")
    print(f"  Mean repackagers (when >0): {panel.loc[panel['n_repackagers']>0, 'n_repackagers'].mean():.1f}")

    # ------------------------------------------------------------------
    # 2. Merge shortage outcome
    # ------------------------------------------------------------------
    print("\n[2/8] Merging shortage outcome...")
    shortage = pd.read_parquet(INTERMEDIATE / "shortage_outcome.parquet")
    panel = panel.merge(shortage, on=['ndc_11', 'year_month'], how='left')
    panel['shortage'] = panel['shortage'].fillna(0).astype(int)
    panel['shortage_start'] = panel['shortage_start'].fillna(0).astype(int)
    panel['shortage_end'] = panel['shortage_end'].fillna(0).astype(int) if 'shortage_end' in panel.columns else 0
    panel['shortage_end_imputed'] = (
        panel['shortage_end_imputed'].fillna(0).astype(int)
        if 'shortage_end_imputed' in panel.columns else 0
    )
    if 'months_remaining' in panel.columns:
        panel['months_remaining'] = panel['months_remaining'].astype(np.float32)
    else:
        panel['months_remaining'] = np.nan
    panel['episode_duration'] = panel['episode_duration'].fillna(0).astype(int) if 'episode_duration' in panel.columns else 0
    if 'episode_censored' in panel.columns:
        panel['episode_censored'] = panel['episode_censored'].fillna(0).astype(int)
    else:
        panel['episode_censored'] = 0
    print(f"  Shortage prevalence: {panel['shortage'].mean():.4%}")
    print(f"  Shortage onsets: {panel['shortage_start'].sum():,}")
    print(f"  Shortage ends: {panel['shortage_end'].sum():,}")
    print(f"  Imputed-end shortage rows: {panel['shortage_end_imputed'].sum():,}")
    print(f"  Mean months remaining (resolved episodes only): {panel.loc[panel['months_remaining'].notna(), 'months_remaining'].mean():.1f}")
    print(f"  Censored shortage rows: {panel['episode_censored'].sum():,}")

    # ------------------------------------------------------------------
    # 3. Merge drug characteristics
    # ------------------------------------------------------------------
    print("\n[3/8] Merging drug characteristics...")
    drug_chars = pd.read_parquet(INTERMEDIATE / "drug_characteristics.parquet")
    # Avoid duplicating columns already in panel skeleton
    panel_cols = set(panel.columns)
    char_cols = ['ndc_11'] + [c for c in drug_chars.columns
                              if c != 'ndc_11' and c not in panel_cols]
    drug_chars_slim = drug_chars[char_cols].drop_duplicates(subset=['ndc_11'])
    panel = panel.merge(drug_chars_slim, on='ndc_11', how='left')
    print(f"  Matched from drug_characteristics file: {panel['dosage_form'].notna().mean():.1%}")

    # Fill in derived features from skeleton attributes for unmatched NDCs
    # The panel skeleton already carries DOSAGEFORMNAME, ROUTENAME, etc. from
    # the historical NDC snapshots that built it.
    injectable_forms = ['INJECTABLE', 'INJECTION', 'INJECTION, SOLUTION',
                        'INJECTION, POWDER, LYOPHILIZED, FOR SOLUTION',
                        'INJECTION, SUSPENSION', 'INJECTION, EMULSION',
                        'INJECTION, POWDER, FOR SOLUTION',
                        'INJECTION, LIPID COMPLEX',
                        'INJECTION, SOLUTION, CONCENTRATE']
    fill_mask = panel['dosage_form'].isna()
    if fill_mask.any():
        panel.loc[fill_mask, 'dosage_form'] = panel.loc[fill_mask, 'DOSAGEFORMNAME'].fillna('UNKNOWN')
        panel.loc[fill_mask, 'route'] = panel.loc[fill_mask, 'ROUTENAME'].fillna('UNKNOWN')
        panel.loc[fill_mask, 'is_injectable'] = panel.loc[fill_mask, 'DOSAGEFORMNAME'].fillna('').str.upper().apply(
            lambda x: 1 if any(inj in x for inj in injectable_forms) else 0
        )
        panel.loc[fill_mask, 'is_intravenous'] = panel.loc[fill_mask, 'ROUTENAME'].fillna('').str.contains(
            'INTRAVENOUS', case=False, na=False
        ).astype(int)
        panel.loc[fill_mask, 'is_generic'] = panel.loc[fill_mask, 'MARKETINGCATEGORYNAME'].isin(
            ['ANDA', 'NDA AUTHORIZED GENERIC']
        ).astype(int)
        panel.loc[fill_mask, 'marketing_category'] = panel.loc[fill_mask, 'MARKETINGCATEGORYNAME'].fillna('UNKNOWN')
        panel.loc[fill_mask, 'substance_name'] = panel.loc[fill_mask, 'SUBSTANCENAME'].fillna('')
        panel.loc[fill_mask, 'active_ingredient_count'] = panel.loc[fill_mask, 'SUBSTANCENAME'].fillna('').apply(
            lambda x: len(x.split(';')) if x else 0
        )
        panel.loc[fill_mask, 'nonproprietary_name'] = panel.loc[fill_mask, 'NONPROPRIETARYNAME'].fillna('')
        panel.loc[fill_mask, 'labeler_name'] = panel.loc[fill_mask, 'LABELERNAME'].fillna('')
        panel.loc[fill_mask, 'is_controlled'] = 0  # Conservative default
        print(f"  Filled from skeleton attributes: {fill_mask.sum():,} rows")
    print(f"  Final coverage: {panel['dosage_form'].notna().mean():.1%}")

    # ------------------------------------------------------------------
    # 4. Merge market structure
    # ------------------------------------------------------------------
    print("\n[4/8] Merging market structure...")
    try:
        market = pd.read_parquet(INTERMEDIATE / "market_structure.parquet")
        panel = panel.merge(market, on=['ndc_11', 'year_month'], how='left')
        panel['has_market_structure_data'] = panel['n_manufacturers'].notna().astype(int)
        print(f"  Matched: {panel['n_manufacturers'].notna().mean():.1%}")
    except FileNotFoundError:
        print("  WARNING: market_structure.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 5. Merge warning letters
    # ------------------------------------------------------------------
    print("\n[5/8] Merging warning letters...")
    try:
        wl = pd.read_parquet(INTERMEDIATE / "warning_letters.parquet")
        # Ensure labeler_code exists in panel
        if 'labeler_code' not in panel.columns:
            panel['labeler_code'] = panel['ndc_11'].apply(labeler_code_from_ndc)
        panel = panel.merge(wl, on=['labeler_code', 'year_month'], how='left')
        panel['has_warning_letter_data'] = panel[
            [c for c in ['warning_letter_6m', 'warning_letter_12m', 'warning_letter_24m'] if c in panel.columns]
        ].notna().any(axis=1).astype(int)
        for col in ['warning_letter_6m', 'warning_letter_12m', 'warning_letter_24m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        print(f"  Any warning letter (12m): {panel.get('warning_letter_12m', pd.Series([0])).gt(0).mean():.4%}")
    except FileNotFoundError:
        print("  WARNING: warning_letters.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 6. Merge geographic/disaster features
    # ------------------------------------------------------------------
    print("\n[6/8] Merging geographic & disaster features...")
    try:
        geo = pd.read_parquet(INTERMEDIATE / "geographic_disasters.parquet")
        if 'labeler_code' not in panel.columns:
            panel['labeler_code'] = panel['ndc_11'].apply(labeler_code_from_ndc)
        panel = panel.merge(geo, on=['labeler_code', 'year_month'], how='left')
        panel['has_geo_data'] = panel['is_domestic'].notna().astype(int)
        for col in ['disaster_count_3m', 'disaster_count_12m', 'major_disaster_12m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        if 'is_domestic' in panel.columns:
            print(f"  Domestic: {panel['is_domestic'].mean():.1%}")
    except FileNotFoundError:
        print("  WARNING: geographic_disasters.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7. Merge patent/exclusivity features
    # ------------------------------------------------------------------
    print("\n[7/8] Merging patent & exclusivity features...")
    try:
        pe = pd.read_parquet(INTERMEDIATE / "patents_exclusivity.parquet")
        panel = panel.merge(pe, on=['ndc_11', 'year_month'], how='left')
        panel['has_patent_data'] = (
            panel.get('patent_count', pd.Series(index=panel.index, dtype=float)).notna() |
            panel.get('months_to_nearest_expiry', pd.Series(index=panel.index, dtype=float)).notna()
        ).astype(int)
        for col in ['patent_count', 'total_patents_ever', 'has_active_exclusivity',
                     'recent_patent_expiry', 'has_substance_patent', 'has_product_patent']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        for col in ['months_to_nearest_expiry', 'months_to_exclusivity_end']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(-1).astype(int)
        print(f"  Has patents: {panel.get('patent_count', pd.Series([0])).gt(0).mean():.1%}")
    except FileNotFoundError:
        print("  WARNING: patents_exclusivity.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7b. Merge merger / ownership change features
    # ------------------------------------------------------------------
    print("\n[7b/9] Merging merger / ownership change features...")
    try:
        mergers = pd.read_parquet(INTERMEDIATE / "mergers.parquet")
        panel = panel.merge(mergers, on=['ndc_11', 'year_month'], how='left')
        panel['has_merger_data'] = panel.get(
            'ownership_change_12m', pd.Series(index=panel.index, dtype=float)
        ).notna().astype(int)
        for col in ['ownership_change_12m', 'ownership_change_24m',
                     'external_merger_12m', 'external_merger_24m',
                     'n_ownership_changes_24m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        print(f"  External merger (12m): {panel.get('external_merger_12m', pd.Series([0])).gt(0).mean():.4%}")
    except FileNotFoundError:
        print("  WARNING: mergers.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7c. Merge pricing features
    # ------------------------------------------------------------------
    print("\n[7c/12] Merging pricing features...")
    try:
        pricing = pd.read_parquet(INTERMEDIATE / "pricing.parquet")
        panel = panel.merge(pricing, on=['ndc_11', 'year_month'], how='left')
        panel['has_nadac_data'] = panel['nadac_per_unit'].notna().astype(int)
        panel['has_nadac_trend_3m'] = panel.get(
            'nadac_pct_change_3m', pd.Series(index=panel.index, dtype=float)
        ).notna().astype(int)
        panel['has_nadac_trend_12m'] = panel.get(
            'nadac_pct_change_12m', pd.Series(index=panel.index, dtype=float)
        ).notna().astype(int)
        panel['has_nadac_market_median'] = panel.get(
            'nadac_vs_market_median', pd.Series(index=panel.index, dtype=float)
        ).notna().astype(int)
        if 'nadac_is_observed' not in panel.columns:
            # Legacy pricing outputs distinguish product-level imputation but not carry-forward rows.
            panel['nadac_is_observed'] = (
                panel['nadac_per_unit'].notna() &
                panel.get('nadac_is_imputed', pd.Series(0, index=panel.index)).fillna(0).eq(0)
            ).astype(int)
        if 'nadac_is_ffill' not in panel.columns:
            panel['nadac_is_ffill'] = 0
        for col in ['nadac_is_low_price', 'nadac_is_imputed']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        for col in ['nadac_is_observed', 'nadac_is_ffill']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        match_rate = panel['nadac_per_unit'].notna().mean()
        n_imputed = panel.get('nadac_is_imputed', pd.Series([0])).sum()
        print(f"  NADAC price match rate: {match_rate:.1%}")
        print(f"  NADAC imputed (product-level): {n_imputed:,}")
    except FileNotFoundError:
        print("  WARNING: pricing.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7c2. Merge CMS ASP pricing features (Medicare Part B drugs)
    # ------------------------------------------------------------------
    print("\n[7c2/12] Merging CMS ASP pricing features...")
    try:
        asp = pd.read_parquet(INTERMEDIATE / "asp_pricing.parquet")
        # Downcast numeric columns before merge to keep memory in check.
        for col in asp.columns:
            if col in ('ndc_11', 'year_month', 'asp_quarter', 'asp_pkg_size'):
                continue
            if asp[col].dtype == 'int64':
                asp[col] = pd.to_numeric(asp[col], downcast='integer')
            elif asp[col].dtype == 'float64':
                asp[col] = pd.to_numeric(asp[col], downcast='float')
        panel = panel.merge(asp, on=['ndc_11', 'year_month'], how='left')
        panel['has_asp_data'] = panel.get(
            'has_asp_data',
            panel.get('asp_payment_limit', pd.Series(index=panel.index, dtype=float)).notna().astype(int),
        ).fillna(0).astype(int)
        panel['asp_is_observed'] = panel.get(
            'asp_is_observed', pd.Series(0, index=panel.index, dtype=int),
        ).fillna(0).astype(int)
        match_rate = panel['has_asp_data'].mean()
        print(f"  ASP price match rate: {match_rate:.1%}")
        print(f"  Median ASP payment limit (per HCPCS billing unit): "
              f"${panel.loc[panel['has_asp_data'] == 1, 'asp_payment_limit'].median():.2f}")
    except FileNotFoundError:
        print("  WARNING: asp_pricing.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7d. Merge utilization features
    # ------------------------------------------------------------------
    print("\n[7d/12] Merging utilization features...")
    try:
        util = pd.read_parquet(INTERMEDIATE / "utilization.parquet")
        # Downcast numeric columns to reduce memory before merge
        for col in util.columns:
            if col in ('ndc_11', 'year_month'):
                continue
            if util[col].dtype == 'int64':
                util[col] = pd.to_numeric(util[col], downcast='integer')
            elif util[col].dtype == 'float64':
                util[col] = pd.to_numeric(util[col], downcast='float')
        # Also downcast the main panel before merge to free memory
        for col in panel.select_dtypes(include=['int64']).columns:
            if col not in ('ndc_11',):
                panel[col] = pd.to_numeric(panel[col], downcast='integer')
        for col in panel.select_dtypes(include=['float64']).columns:
            panel[col] = pd.to_numeric(panel[col], downcast='float')
        import gc; gc.collect()
        panel = panel.merge(util, on=['ndc_11', 'year_month'], how='left')
        panel['has_medicaid_data'] = panel.get(
            'has_medicaid_data',
            panel.get('medicaid_rx_count', pd.Series(index=panel.index, dtype=float)).notna().astype(int)
        )
        panel['has_partd_data'] = panel.get(
            'has_partd_data',
            panel.get('partd_total_claims', pd.Series(index=panel.index, dtype=float)).notna().astype(int)
        )
        panel['has_partb_data'] = panel.get(
            'has_partb_data',
            panel.get('partb_total_claims', pd.Series(index=panel.index, dtype=float)).notna().astype(int)
        ).fillna(0).astype(int)
        panel['utilization_product_imputed'] = panel.get(
            'utilization_product_imputed', pd.Series(0, index=panel.index, dtype=int)
        )
        panel['has_utilization_data'] = (
            panel['has_medicaid_data'].fillna(0).astype(int) |
            panel['has_partd_data'].fillna(0).astype(int) |
            panel['has_partb_data'].fillna(0).astype(int)
        ).astype(int)
        panel['has_medicaid_trend_data'] = panel.get(
            'medicaid_rx_trend_4q', pd.Series(index=panel.index, dtype=float)
        ).notna().astype(int)
        for col in ['medicaid_rx_count', 'medicaid_units', 'medicaid_spending']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0)
        match_rate = panel['medicaid_rx_count'].gt(0).mean() if 'medicaid_rx_count' in panel.columns else 0
        print(f"  Medicaid utilization match rate (>0): {match_rate:.1%}")
        if 'partb_total_claims' in panel.columns:
            partb_match_rate = panel['partb_total_claims'].notna().mean()
            print(f"  Part B utilization match rate: {partb_match_rate:.1%}")
    except FileNotFoundError:
        print("  WARNING: utilization.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7d2. Merge Symphony prescription utilization features
    # ------------------------------------------------------------------
    print("\n[7d2/12] Merging Symphony prescription features...")
    try:
        symphony = pd.read_parquet(INTERMEDIATE / "symphony_prescriptions_ndc_month_study.parquet")
        symphony = symphony.drop(
            columns=[c for c in ["symphony_ndc_11", "source_zip", "source_member"] if c in symphony.columns],
            errors="ignore",
        )
        metric_cols = [c for c in symphony.columns if c not in ("ndc_11", "year_month")]
        symphony = symphony.rename(columns={c: f"symphony_{c}" for c in metric_cols})
        symphony["has_symphony_data"] = 1

        # Downcast numeric columns before the wide panel merge.
        for col in symphony.columns:
            if col in ("ndc_11", "year_month"):
                continue
            if symphony[col].dtype == "int64":
                symphony[col] = pd.to_numeric(symphony[col], downcast="integer")
            elif symphony[col].dtype == "float64":
                symphony[col] = pd.to_numeric(symphony[col], downcast="float")

        panel = panel.merge(symphony, on=["ndc_11", "year_month"], how="left")
        panel["has_symphony_data"] = panel["has_symphony_data"].fillna(0).astype(int)
        row_match_rate = panel["has_symphony_data"].mean()
        matched_ndcs = panel.loc[panel["has_symphony_data"].eq(1), "ndc_11"].nunique()
        ndc_match_rate = matched_ndcs / max(panel["ndc_11"].nunique(), 1)
        print(f"  Symphony NDC-month match rate: {row_match_rate:.1%}")
        print(f"  Symphony NDC match rate: {ndc_match_rate:.1%} ({matched_ndcs:,} NDCs)")
        if "symphony_trx_count" in panel.columns:
            print(
                "  Median monthly TRx among matched rows: "
                f"{panel.loc[panel['has_symphony_data'].eq(1), 'symphony_trx_count'].median():,.0f}"
            )
    except FileNotFoundError:
        print("  WARNING: symphony_prescriptions_ndc_month_study.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7e. Merge recall features
    # ------------------------------------------------------------------
    print("\n[7e/12] Merging recall features...")
    try:
        recalls = pd.read_parquet(INTERMEDIATE / "recalls.parquet")
        if 'labeler_code' not in panel.columns:
            panel['labeler_code'] = panel['ndc_11'].apply(labeler_code_from_ndc)
        panel = panel.merge(recalls, on=['labeler_code', 'year_month'], how='left')
        panel['has_recall_data'] = panel[
            [c for c in ['recall_count_12m', 'class1_recall_12m', 'recall_count_24m'] if c in panel.columns]
        ].notna().any(axis=1).astype(int)
        for col in ['recall_count_12m', 'class1_recall_12m', 'recall_count_24m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        print(f"  Any recall (12m): {panel.get('recall_count_12m', pd.Series([0])).gt(0).mean():.4%}")
    except FileNotFoundError:
        print("  WARNING: recalls.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7f. Merge inspection features
    # ------------------------------------------------------------------
    print("\n[7f/12] Merging inspection features...")
    try:
        insp = pd.read_parquet(INTERMEDIATE / "inspections.parquet")
        if 'labeler_code' not in panel.columns:
            panel['labeler_code'] = panel['ndc_11'].apply(labeler_code_from_ndc)
        panel = panel.merge(insp, on=['labeler_code', 'year_month'], how='left')
        panel['has_inspection_data'] = panel[
            [c for c in ['oai_inspection_12m', 'vai_inspection_12m', 'inspection_count_24m'] if c in panel.columns]
        ].notna().any(axis=1).astype(int)
        for col in ['oai_inspection_12m', 'vai_inspection_12m', 'inspection_count_24m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        print(f"  Any OAI (12m): {panel.get('oai_inspection_12m', pd.Series([0])).gt(0).mean():.4%}")
    except FileNotFoundError:
        print("  WARNING: inspections.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7g. Merge adverse event features
    # ------------------------------------------------------------------
    print("\n[7g/14] Merging adverse event features...")
    try:
        ae = pd.read_parquet(INTERMEDIATE / "adverse_events.parquet")
        if 'labeler_code' not in panel.columns:
            panel['labeler_code'] = panel['ndc_11'].apply(labeler_code_from_ndc)
        panel = panel.merge(ae, on=['labeler_code', 'year_month'], how='left')
        panel['has_adverse_event_data'] = panel[
            [c for c in ['ae_reports_3m', 'ae_reports_12m', 'ae_trend_12m'] if c in panel.columns]
        ].notna().any(axis=1).astype(int)
        for col in ['ae_reports_3m', 'ae_reports_12m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        if 'ae_trend_12m' in panel.columns:
            panel['ae_trend_12m'] = panel['ae_trend_12m'].fillna(0)
        print(f"  AE reports (12m) > 0: {panel.get('ae_reports_12m', pd.Series([0])).gt(0).mean():.4%}")
    except FileNotFoundError:
        print("  WARNING: adverse_events.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7h. Merge API sourcing features
    # ------------------------------------------------------------------
    # API sourcing features are derived from FDA Drug Master File (DMF) data.
    # Coverage is limited to NDCs with registered API sources; unmatched NDCs
    # get has_api_data=0.  An ablation study in 23_onset_group_benchmark_enhanced.py
    # quantifies the marginal predictive contribution of these features.
    print("\n[7h/15] Merging API sourcing features...")
    try:
        api = pd.read_parquet(INTERMEDIATE / "api_sourcing.parquet")
        # 04h now emits time-varying features keyed on (ndc_11, year_month),
        # gated on DMF SUBMIT_DATE. Fall back to the static per-NDC merge if
        # an old-format file is on disk.
        api_keys = ['ndc_11', 'year_month'] if 'year_month' in api.columns else ['ndc_11']
        panel = panel.merge(api, on=api_keys, how='left')
        for col in ['n_api_suppliers', 'n_api_countries', 'has_api_data']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)
        for col in ['api_india_share', 'api_china_share', 'api_us_share',
                     'api_india_china_share', 'api_country_hhi']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(np.float32)
        match_rate = panel.get('has_api_data', pd.Series([0])).mean()
        print(f"  API sourcing match rate: {match_rate:.1%}")
        print(f"  Mean API suppliers (matched): {panel.loc[panel.get('has_api_data', 0)==1, 'n_api_suppliers'].mean():.1f}")
    except FileNotFoundError:
        print("  WARNING: api_sourcing.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7h2. Merge API disaster exposure features (time-varying)
    # ------------------------------------------------------------------
    print("\n[7h2/15] Merging API disaster exposure features...")
    try:
        api_dis = pd.read_parquet(INTERMEDIATE / "api_disasters.parquet")
        panel = panel.merge(api_dis, on=['ndc_11', 'year_month'], how='left')
        for col in ['api_disaster_exposure_3m', 'api_disaster_exposure_12m']:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(np.float32)
        if 'api_major_disaster_12m' in panel.columns:
            panel['api_major_disaster_12m'] = panel['api_major_disaster_12m'].fillna(0).astype(int)
        has_exposure = panel.get('api_disaster_exposure_12m', pd.Series([0])).gt(0).mean()
        print(f"  NDC-months with API disaster exposure: {has_exposure:.1%}")
    except FileNotFoundError:
        print("  WARNING: api_disasters.parquet not found, skipping")

    # ------------------------------------------------------------------
    # 7i. Orange Book match flag (#9)
    # ------------------------------------------------------------------
    print("\n[7h/14] Adding data quality flags...")
    panel['ob_match_flag'] = panel.get('appl_no', pd.Series(dtype=str)).notna().astype(int)
    print(f"  OB match flag: {panel['ob_match_flag'].mean():.1%} matched")

    for flag_col in [
        'has_market_structure_data',
        'has_geo_data',
        'has_patent_data',
        'has_merger_data',
        'has_nadac_data',
        'has_nadac_trend_3m',
        'has_nadac_trend_12m',
        'has_nadac_market_median',
        'has_medicaid_data',
        'has_partd_data',
        'has_utilization_data',
        'has_medicaid_trend_data',
        'has_symphony_data',
        'utilization_product_imputed',
        'has_inspection_data',
        'has_warning_letter_data',
        'has_recall_data',
        'has_adverse_event_data',
        'nadac_is_observed',
        'nadac_is_ffill',
        'is_repackager',
        'repackager_rows_retained',
    ]:
        if flag_col not in panel.columns:
            panel[flag_col] = 0
        panel[flag_col] = panel[flag_col].fillna(0).astype(int)

    panel = normalize_structural_missingness(panel)

    # ------------------------------------------------------------------
    # 8. Add temporal features and lagged shortage history
    # ------------------------------------------------------------------
    print("\n[8/14] Adding temporal features and lagged shortage history...")

    # Temporal features
    panel['year'] = panel['year_month'].str[:4].astype(int)
    panel['month'] = panel['year_month'].str[5:7].astype(int)
    panel['quarter'] = ((panel['month'] - 1) // 3) + 1

    # COVID period indicator (March 2020 - June 2021)
    panel['covid_period'] = ((panel['year_month'] >= '2020-03') &
                              (panel['year_month'] <= '2021-06')).astype(int)

    # Hurricane season (June-November)
    panel['hurricane_season'] = panel['month'].isin([6, 7, 8, 9, 10, 11]).astype(int)

    # Years on market (time-varying)
    if 'start_marketing_date' in panel.columns:
        ym_dt = pd.to_datetime(panel['year_month'] + '-01')
        panel['years_on_market'] = (
            (ym_dt - panel['start_marketing_date']).dt.days / 365.25
        ).clip(lower=0)
    elif 'STARTMARKETINGDATE' in panel.columns:
        start_dt = panel['STARTMARKETINGDATE'].apply(parse_date_flexible)
        ym_dt = pd.to_datetime(panel['year_month'] + '-01')
        panel['years_on_market'] = ((ym_dt - start_dt).dt.days / 365.25).clip(lower=0)

    # Lagged shortage history
    print("  Computing lagged shortage history...")
    panel = panel.sort_values(['ndc_11', 'year_month'])

    # Create year_month index for efficient lag computation
    ym_list = sorted(panel['year_month'].unique())
    ym_to_idx = {ym: i for i, ym in enumerate(ym_list)}
    panel['ym_idx'] = panel['year_month'].map(ym_to_idx)

    # Group by NDC and compute lags
    panel['shortage_lag1m'] = panel.groupby('ndc_11')['shortage'].shift(1).fillna(0).astype(int)
    panel['shortage_lag3m'] = (
        panel.groupby('ndc_11')['shortage']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).max())
        .fillna(0).astype(int)
    )
    panel['shortage_lag12m'] = (
        panel.groupby('ndc_11')['shortage']
        .transform(lambda x: x.shift(1).rolling(12, min_periods=1).max())
        .fillna(0).astype(int)
    )

    # Drop helper column
    panel.drop(columns=['ym_idx'], inplace=True)

    # ------------------------------------------------------------------
    # 9. Derived features from existing panel data (#5-7)
    # ------------------------------------------------------------------
    print("\n[9/14] Computing derived features...")

    # --- #5: Ingredient-level shortage contagion ---
    # For each NDCxmonth, check if any OTHER NDC with the same active ingredient
    # is currently in shortage. This captures substitution demand shocks.
    print("  Computing ingredient-level shortage contagion...")
    substance_col = 'substance_name' if 'substance_name' in panel.columns else 'SUBSTANCENAME'
    if substance_col in panel.columns:
        # Total shortages per ingredientxmonth
        panel['_substance'] = panel[substance_col].fillna('').str.upper().str.strip()
        ingredient_shortage = panel.groupby(['_substance', 'year_month'])['shortage'].transform('sum')
        # Subtract own shortage to get "other NDCs in shortage"
        panel['same_ingredient_in_shortage'] = (
            (ingredient_shortage - panel['shortage']).clip(lower=0) > 0
        ).astype(int)
        panel.drop(columns=['_substance'], inplace=True)
        print(f"    Same ingredient in shortage: {panel['same_ingredient_in_shortage'].mean():.4%}")
    else:
        panel['same_ingredient_in_shortage'] = 0

    # --- #6: Labeler-level shortage burden ---
    # Count of active shortages across ALL products for same labeler
    print("  Computing labeler-level shortage burden...")
    if 'labeler_code' not in panel.columns:
        panel['labeler_code'] = panel['ndc_11'].apply(labeler_code_from_ndc)
    labeler_shortage = panel.groupby(['labeler_code', 'year_month'])['shortage'].transform('sum')
    panel['labeler_shortage_burden'] = (labeler_shortage - panel['shortage']).clip(lower=0).astype(int)
    print(f"    Labeler shortage burden > 0: {(panel['labeler_shortage_burden'] > 0).mean():.4%}")

    # --- #7: Time since last shortage ---
    # For each NDC, months since most recent shortage resolution
    print("  Computing time since last shortage...")
    panel = panel.sort_values(['ndc_11', 'year_month'])
    # Mark the last month of each shortage episode (shortage=1 followed by shortage=0)
    panel['_shortage_end'] = (
        (panel.groupby('ndc_11')['shortage'].shift(1) == 1) &
        (panel['shortage'] == 0)
    ).astype(int)

    # Compute months since last shortage end using cumulative group counting
    def months_since_last_event(group):
        result = pd.Series(np.nan, index=group.index)
        last_event_idx = None
        for i, (idx, val) in enumerate(group.items()):
            if val == 1:
                last_event_idx = i
            if last_event_idx is not None:
                result.iloc[i] = i - last_event_idx
        return result

    panel['time_since_last_shortage'] = (
        panel.groupby('ndc_11')['_shortage_end']
        .transform(months_since_last_event)
        .fillna(-1)  # -1 means no prior shortage
        .astype(int)
    )
    panel.drop(columns=['_shortage_end'], inplace=True)
    has_prior = (panel['time_since_last_shortage'] >= 0).mean()
    print(f"    NDC-months with prior shortage: {has_prior:.4%}")

    # --- Labeler product count ---
    print("  Computing labeler product counts...")
    labeler_product_count = panel.groupby(['labeler_code', 'year_month'])['ndc_11'].transform('nunique')
    panel['labeler_product_count'] = labeler_product_count

    # --- Net manufacturer change ---
    print("  Computing net manufacturer change...")
    if 'n_manufacturers' in panel.columns:
        panel['net_manufacturer_change_12m'] = panel.groupby('ndc_11')['n_manufacturers'].transform(
            lambda x: x - x.shift(12)
        ).fillna(0).astype(int)
    else:
        panel['net_manufacturer_change_12m'] = 0

    print("  Derived features complete.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = PANEL_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"  FINAL PANEL saved to {output_path}")
    print(f"  Shape: {panel.shape}")
    print(f"  Rows: {len(panel):,}")
    print(f"  Columns: {panel.shape[1]}")
    print(f"  Unique NDCs: {panel['ndc_11'].nunique():,}")
    print(f"  Unique months: {panel['year_month'].nunique()}")
    print(f"  Date range: {panel['year_month'].min()} to {panel['year_month'].max()}")
    print(f"  Shortage prevalence: {panel['shortage'].mean():.4%}")
    print(f"  Shortage onsets: {panel['shortage_start'].sum():,}")

    print(f"\n  Column list:")
    for col in sorted(panel.columns):
        dtype = panel[col].dtype
        nulls = panel[col].isna().mean()
        print(f"    {col:40s} {str(dtype):12s} {nulls:.1%} null")

    print("\nDone!")
    return panel


if __name__ == "__main__":
    main()
