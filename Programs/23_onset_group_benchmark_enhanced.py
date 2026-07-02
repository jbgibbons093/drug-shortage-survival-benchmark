"""
23_onset_group_benchmark_enhanced.py - grouped feature builder.

The current manuscript analysis is the discrete-time survival benchmark in
25_survival_benchmarks.py. This module supplies the grouped-panel feature
builder used by that benchmark.

Do not run this file directly for manuscript results.

Feature families:
- historical shortage memory features
- event recency features
- within-group trend and rolling-risk features
- lag-safe utilization recovery features
- class / route / dosage-form trailing onset pressure

Target timing:
  `onset_any6` is 1 when any `shortage_start` occurs in the next 6 months for
  a drug group. In the active survival pipeline, `shortage_start` comes from
  ASHP `date_notified`; prediction lead times are relative to that verified
  notification month.
"""

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from importlib.util import module_from_spec, spec_from_file_location

_util_spec = spec_from_file_location("utilities", Path(__file__).parent / "00_utilities.py")
_util_mod = module_from_spec(_util_spec)
_util_spec.loader.exec_module(_util_mod)
ANALYSIS = _util_mod.ANALYSIS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

PANEL_PATH = Path(os.environ.get("PANEL_PATH", str(ANALYSIS / "drug_shortage_panel.parquet")))
OUTPUT_DIR = ANALYSIS / "onset_group_results_enhanced"

# Supply-chain leading-indicator feature families (built by scripts 04j/k/l/m,
# 06b). Group-level files join on (drug_group_key, year_month); DEA quotas are
# ingredient-level and join on a normalized nonproprietary name. All are built
# lag-safe (window ends at t-1), so they cannot leak future information.
INTERMEDIATE = ANALYSIS.parent / "intermediate"
LEADING_INDICATOR_GROUP_FILES = (
    "leading_indicator_group_features.parquet",
    "leading_indicator_crosswalk_features.parquet",
    "sec_disruption_group_features.parquet",
    "disaster_facility_features.parquet",
)
DEA_QUOTA_FILE = "dea_quota_features.parquet"
INCLUDE_EXPERIMENTAL_LEADING_INDICATORS = (
    os.environ.get("INCLUDE_EXPERIMENTAL_LEADING_INDICATORS", "0") == "1"
)
GROUP_PANEL_UNIT = os.environ.get("GROUP_PANEL_UNIT", "ingredient_route_form")
VALID_GROUP_PANEL_UNITS = {"ingredient", "ingredient_form", "ingredient_route_form"}
if GROUP_PANEL_UNIT not in VALID_GROUP_PANEL_UNITS:
    raise ValueError(
        f"Unsupported GROUP_PANEL_UNIT={GROUP_PANEL_UNIT!r}. "
        f"Use one of {sorted(VALID_GROUP_PANEL_UNITS)}."
    )

STUDY_START = "2020-01"
ROLLING_START = "2024-07"
PREDICTION_HORIZON = 6
SEED = 42
TRAIN_ROWS_LGBM = 350_000
TRAIN_ROWS_LOGIT = 220_000
RANK_KS = [50, 100, 250, 500]
TRAIN_HISTORY_MONTHS = 36
FINAL_MODEL_HISTORY_MONTHS = 36
VALIDATION_RECENT_MONTHS = 6
VALIDATION_MAX_MONTHS = 12
VALIDATION_MIN_POSITIVES = 25
RECENCY_HALF_LIFE_MONTHS = 12
RECENCY_WEIGHT_FLOOR = 0.35

CATEGORICAL = [
    "dosage_form",
    "route",
    "therapeutic_class",
    "therapeutic_category",
    "marketing_category",
    "primary_country",
    "dea_schedule",
]

BASE_NUMERIC = [
    "is_injectable",
    "is_intravenous",
    "is_generic",
    "is_controlled",
    "is_domestic",
    "active_ingredient_count",
    "n_api_suppliers",
    "n_api_countries",
    "api_india_share",
    "api_china_share",
    "api_us_share",
    "api_india_china_share",
    "api_country_hhi",
    "has_api_data",
    "year",
    "covid_period",
    "hurricane_season",
    "years_on_market",
    "patent_count",
    "months_to_nearest_expiry",
    "has_active_exclusivity",
    "months_to_exclusivity_end",
    "n_manufacturers",
    "n_applications",
    "n_facilities",
    "n_countries",
    "n_repackagers",
    "sole_source",
    "has_market_structure_data",
    "has_patent_data",
    "has_merger_data",
    "has_geo_data",
    "total_patents_ever",
    "recent_patent_expiry",
    "has_substance_patent",
    "has_product_patent",
    "warning_letter_6m",
    "warning_letter_12m",
    "warning_letter_24m",
    "has_warning_letter_data",
    "disaster_count_3m",
    "disaster_count_12m",
    "major_disaster_12m",
    "ownership_change_12m",
    "ownership_change_24m",
    "external_merger_12m",
    "external_merger_24m",
    "n_ownership_changes_24m",
    "recent_generic_entry",
    "recent_manufacturer_exit",
    "net_manufacturer_change_12m",
    "nadac_per_unit",
    "nadac_pct_change_3m",
    "nadac_pct_change_12m",
    "nadac_generic_ratio",
    "nadac_is_low_price",
    "nadac_vs_market_median",
    "has_nadac_data",
    "has_nadac_trend_3m",
    "has_nadac_trend_12m",
    "has_nadac_market_median",
    "nadac_is_observed",
    "nadac_is_ffill",
    "nadac_is_imputed",
    # Medicare Part B ASP pricing (added 2026-04-27 - fills NADAC gap for
    # injectables/IV agents). Known at decision time (CMS publishes quarterly
    # ASP files in advance), so safe to include in BASE_NUMERIC.
    "asp_payment_limit",
    "asp_payment_per_pkg",
    "asp_billunits_per_pkg",
    "has_asp_data",
    "asp_is_observed",
    "recall_count_12m",
    "class1_recall_12m",
    "recall_count_24m",
    "has_recall_data",
    "oai_inspection_12m",
    "vai_inspection_12m",
    "inspection_count_24m",
    "has_inspection_data",
    "ae_reports_3m",
    "ae_reports_12m",
    "ae_trend_12m",
    "has_adverse_event_data",
    "api_disaster_exposure_3m",
    "api_disaster_exposure_12m",
    "api_major_disaster_12m",
    "same_ingredient_in_shortage",
    "labeler_shortage_burden",
    "time_since_last_shortage",
    "labeler_product_count",
    "ob_match_flag",
]

RAW_GROUP_EXTRA = [
    "labeler_code",
    "application_number",
]

SYMPHONY_NUMERIC = [
    "symphony_trx_count",
    "symphony_nrx_count",
    "symphony_rrx_count",
    "symphony_trx_quantity",
    "symphony_nrx_quantity",
    "symphony_rrx_quantity",
    "symphony_trx_mbs_dollars",
    "symphony_nrx_mbs_dollars",
    "symphony_rrx_mbs_dollars",
    "symphony_trx_writer_count",
    "symphony_nrx_writer_count",
    "symphony_rrx_writer_count",
    "symphony_trx_pack_units",
    "symphony_nrx_pack_units",
    "symphony_rrx_pack_units",
    "symphony_avg_trx_size",
    "symphony_avg_nrx_size",
    "symphony_avg_rrx_size",
    "symphony_avg_trx_price",
    "symphony_trx_dollars",
    "symphony_nrx_dollars",
    "symphony_rrx_dollars",
    "symphony_trx_price",
    "symphony_nrx_price",
    "symphony_rrx_price",
    "symphony_total_cost_per_unit",
    "symphony_dacon",
    "has_symphony_data",
]

SYMPHONY_SUM_NUMERIC = {
    "symphony_trx_count",
    "symphony_nrx_count",
    "symphony_rrx_count",
    "symphony_trx_quantity",
    "symphony_nrx_quantity",
    "symphony_rrx_quantity",
    "symphony_trx_mbs_dollars",
    "symphony_nrx_mbs_dollars",
    "symphony_rrx_mbs_dollars",
    "symphony_trx_writer_count",
    "symphony_nrx_writer_count",
    "symphony_rrx_writer_count",
    "symphony_trx_pack_units",
    "symphony_nrx_pack_units",
    "symphony_rrx_pack_units",
    "symphony_trx_dollars",
    "symphony_nrx_dollars",
    "symphony_rrx_dollars",
}

SYMPHONY_MEDIAN_NUMERIC = set(SYMPHONY_NUMERIC) - SYMPHONY_SUM_NUMERIC - {"has_symphony_data"}
SYMPHONY_LAG_NUMERIC = [c for c in SYMPHONY_NUMERIC if c != "has_symphony_data"]
SYMPHONY_ROLLING_NUMERIC = [
    "symphony_trx_count",
    "symphony_nrx_count",
    "symphony_rrx_count",
    "symphony_trx_quantity",
    "symphony_trx_dollars",
    "symphony_trx_writer_count",
    "symphony_trx_pack_units",
]

UTILIZATION_NUMERIC = [
    "medicaid_rx_count",
    "medicaid_units",
    "medicaid_spending",
    "medicaid_rx_trend_4q",
    "medicaid_units_trend_4q",
    "medicaid_rx_cv_4q",
    "partd_total_claims",
    "partd_avg_cost_per_claim",
    # Medicare Part B utilization (added 2026-04-27). Captures
    # physician-administered drugs that Part D doesn't see.
    "partb_total_claims",
    "partb_avg_cost_per_claim",
    "partb_total_spending",
    "partb_total_beneficiaries",
    "has_medicaid_data",
    "has_partd_data",
    "has_partb_data",
    "utilization_product_imputed",
    "has_utilization_data",
    "has_medicaid_trend_data",
] + SYMPHONY_NUMERIC

UNSAFE_CURRENT_UTILIZATION = set(UTILIZATION_NUMERIC)

ENGINEERED = [
    "few_manufacturers",
    "few_api_suppliers",
    "market_vulnerability",
    "quality_signal_any",
    "quality_signal_count",
    "disruption_signal",
    "price_shock",
    "supply_chain_risk",
    "ingredient_pressure",
    "manufacturer_exit_pressure",
    "patent_cliff_12m",
    "exclusivity_loss_12m",
    "injectable_sole_source",
    "quality_stack",
    "capacity_stress",
    "contagion_stress",
    "commercial_risk",
]


def group_key(nonproprietary_name, dosage_form, route=None):
    name = str(nonproprietary_name).strip().lower() if pd.notna(nonproprietary_name) else "unknown"
    form = str(dosage_form).strip().lower() if pd.notna(dosage_form) else "unknown"
    route_value = str(route).strip().lower() if pd.notna(route) else "unknown"
    # Tidy: drop stray parentheses (combination-product / data-entry artifacts such as
    # "(chloroprocaine hci") and collapse whitespace so the same molecule does not
    # fragment into separate groups on a stray "(". Keeps the parenthesized CONTENT.
    name = " ".join(name.replace("(", " ").replace(")", " ").split()).strip(" .,-") or "unknown"
    form = " ".join(form.split())
    route_value = " ".join(route_value.split())
    if GROUP_PANEL_UNIT == "ingredient":
        return name
    if GROUP_PANEL_UNIT == "ingredient_form":
        return f"{name} | {form}"
    return f"{name} | {route_value} | {form}"


def add_engineered(df):
    df = df.copy()
    n_mfr = df["n_manufacturers"].fillna(0.0)
    n_api = df["n_api_suppliers"].fillna(0.0)
    sole = df["sole_source"].fillna(0.0)
    inj = df["is_injectable"].fillna(0.0)
    warning = df["warning_letter_12m"].fillna(0.0).clip(0, 1)
    oai = df["oai_inspection_12m"].fillna(0.0).clip(0, 1)
    vai = df["vai_inspection_12m"].fillna(0.0).clip(0, 1)
    recall = df["recall_count_12m"].fillna(0.0).clip(0, 1)
    class1 = df["class1_recall_12m"].fillna(0.0).clip(0, 1)
    months_patent = df["months_to_nearest_expiry"].fillna(999.0)
    months_excl = df["months_to_exclusivity_end"].fillna(999.0)
    df["few_manufacturers"] = (n_mfr <= 2).astype(np.float32)
    df["few_api_suppliers"] = (n_api <= 2).astype(np.float32)
    df["market_vulnerability"] = (sole + df["few_manufacturers"] + inj).astype(np.float32)
    df["quality_signal_any"] = ((warning + oai + vai + recall + class1) > 0).astype(np.float32)
    df["quality_signal_count"] = (warning + oai + vai + recall + class1).astype(np.float32)
    df["disruption_signal"] = (
        np.log1p(df["disaster_count_12m"].fillna(0.0))
        + np.log1p(df["api_disaster_exposure_12m"].fillna(0.0))
        + df["api_major_disaster_12m"].fillna(0.0)
    ).astype(np.float32)
    df["price_shock"] = (
        np.abs(df["nadac_pct_change_3m"].fillna(0.0))
        + 0.5 * np.abs(df["nadac_pct_change_12m"].fillna(0.0))
    ).astype(np.float32)
    df["supply_chain_risk"] = (
        df["api_country_hhi"].fillna(0.0)
        + df["api_india_china_share"].fillna(0.0)
        + df["api_major_disaster_12m"].fillna(0.0)
    ).astype(np.float32)
    df["ingredient_pressure"] = (
        df["same_ingredient_in_shortage"].fillna(0.0)
        * (1.0 + df["few_manufacturers"] + df["few_api_suppliers"])
    ).astype(np.float32)
    df["manufacturer_exit_pressure"] = (
        df["recent_manufacturer_exit"].fillna(0.0).clip(0, 1)
        + np.clip(-df["net_manufacturer_change_12m"].fillna(0.0), 0, None)
    ).astype(np.float32)
    df["patent_cliff_12m"] = ((months_patent >= 0) & (months_patent <= 12)).astype(np.float32)
    df["exclusivity_loss_12m"] = ((months_excl >= 0) & (months_excl <= 12)).astype(np.float32)
    df["injectable_sole_source"] = (inj * sole).astype(np.float32)
    df["quality_stack"] = (
        1.3 * df["quality_signal_count"].fillna(0.0)
        + 0.8 * df["has_inspection_data"].fillna(0.0)
        + 0.6 * df["ae_trend_12m"].fillna(0.0).clip(lower=0)
    ).astype(np.float32)
    df["capacity_stress"] = (
        (3.0 - np.minimum(n_mfr, 3.0))
        + (3.0 - np.minimum(n_api, 3.0))
        + np.log1p(df["n_repackagers"].fillna(0.0))
    ).astype(np.float32)
    df["contagion_stress"] = (
        1.5 * df["same_ingredient_in_shortage"].fillna(0.0)
        + df["labeler_shortage_burden"].fillna(0.0)
    ).astype(np.float32)
    df["commercial_risk"] = (
        np.abs(df["nadac_pct_change_12m"].fillna(0.0))
        + np.clip(-df["medicaid_rx_trend_4q"].fillna(0.0), 0, None)
        + np.clip(df["medicaid_rx_cv_4q"].fillna(0.0), 0, None)
    ).astype(np.float32)
    return df


def add_group_composition_features(panel):
    comp = panel[
        [
            "drug_group_key",
            "year_month",
            "ndc_11",
            "labeler_code",
            "application_number",
            "route",
            "therapeutic_class",
            "warning_letter_12m",
            "recall_count_12m",
            "class1_recall_12m",
            "oai_inspection_12m",
            "vai_inspection_12m",
            "n_api_suppliers",
            "api_country_hhi",
            "api_india_china_share",
            "n_manufacturers",
            "sole_source",
            "nadac_pct_change_12m",
            "ae_trend_12m",
            "major_disaster_12m",
            "api_major_disaster_12m",
        ]
    ].copy()
    comp["application_number"] = comp["application_number"].fillna("UNKNOWN").astype(str)
    comp["labeler_code"] = comp["labeler_code"].fillna("UNKNOWN").astype(str)
    comp["route"] = comp["route"].fillna("UNKNOWN").astype(str)
    comp["therapeutic_class"] = comp["therapeutic_class"].fillna("UNKNOWN").astype(str)

    comp["ndc_quality_signal"] = (
        (comp["warning_letter_12m"].fillna(0.0) > 0)
        | (comp["recall_count_12m"].fillna(0.0) > 0)
        | (comp["class1_recall_12m"].fillna(0.0) > 0)
        | (comp["oai_inspection_12m"].fillna(0.0) > 0)
        | (comp["vai_inspection_12m"].fillna(0.0) > 0)
    ).astype(np.float32)
    comp["ndc_low_api"] = (comp["n_api_suppliers"].fillna(0.0) <= 2).astype(np.float32)
    comp["ndc_api_concentrated"] = (comp["api_country_hhi"].fillna(0.0) >= 0.6).astype(np.float32)
    comp["ndc_india_china_exposed"] = (comp["api_india_china_share"].fillna(0.0) >= 0.8).astype(np.float32)
    comp["ndc_low_competition"] = (
        (comp["sole_source"].fillna(0.0) > 0)
        | (comp["n_manufacturers"].fillna(0.0) <= 2)
    ).astype(np.float32)
    comp["ndc_price_shock"] = (np.abs(comp["nadac_pct_change_12m"].fillna(0.0)) >= 0.25).astype(np.float32)
    comp["ndc_ae_spike"] = (comp["ae_trend_12m"].fillna(0.0) > 0.25).astype(np.float32)
    comp["ndc_disaster_risk"] = (
        (comp["major_disaster_12m"].fillna(0.0) > 0)
        | (comp["api_major_disaster_12m"].fillna(0.0) > 0)
    ).astype(np.float32)
    comp["abs_nadac_pct_change_12m"] = np.abs(comp["nadac_pct_change_12m"].fillna(0.0)).astype(np.float32)

    grouped = comp.groupby(["drug_group_key", "year_month"], observed=False)
    summary = grouped.agg(
        n_labelers_group=("labeler_code", "nunique"),
        n_application_numbers_group=("application_number", "nunique"),
        n_routes_group=("route", "nunique"),
        n_therapeutic_classes_group=("therapeutic_class", "nunique"),
        share_quality_signal_ndcs=("ndc_quality_signal", "mean"),
        share_low_api_ndcs=("ndc_low_api", "mean"),
        share_api_concentrated_ndcs=("ndc_api_concentrated", "mean"),
        share_india_china_exposed_ndcs=("ndc_india_china_exposed", "mean"),
        share_low_competition_ndcs=("ndc_low_competition", "mean"),
        share_price_shock_ndcs=("ndc_price_shock", "mean"),
        share_ae_spike_ndcs=("ndc_ae_spike", "mean"),
        share_disaster_risk_ndcs=("ndc_disaster_risk", "mean"),
        mean_api_country_hhi_group=("api_country_hhi", "mean"),
        std_api_country_hhi_group=("api_country_hhi", "std"),
        mean_api_suppliers_group=("n_api_suppliers", "mean"),
        std_api_suppliers_group=("n_api_suppliers", "std"),
        mean_manufacturers_group=("n_manufacturers", "mean"),
        std_manufacturers_group=("n_manufacturers", "std"),
        mean_abs_nadac_pct_change_12m_group=("abs_nadac_pct_change_12m", "mean"),
        mean_ae_trend_12m_group=("ae_trend_12m", "mean"),
    ).reset_index()

    labeler_counts = (
        comp.groupby(["drug_group_key", "year_month", "labeler_code"], observed=False)["ndc_11"]
        .nunique()
        .rename("labeler_ndcs")
        .reset_index()
    )
    labeler_dom = (
        labeler_counts.groupby(["drug_group_key", "year_month"], observed=False)["labeler_ndcs"]
        .max()
        .rename("dominant_labeler_ndc_count")
        .reset_index()
    )
    app_counts = (
        comp.groupby(["drug_group_key", "year_month", "application_number"], observed=False)["ndc_11"]
        .nunique()
        .rename("application_ndcs")
        .reset_index()
    )
    app_dom = (
        app_counts.groupby(["drug_group_key", "year_month"], observed=False)["application_ndcs"]
        .max()
        .rename("dominant_application_ndc_count")
        .reset_index()
    )

    summary = summary.merge(labeler_dom, on=["drug_group_key", "year_month"], how="left")
    summary = summary.merge(app_dom, on=["drug_group_key", "year_month"], how="left")
    total_ndcs = grouped["ndc_11"].nunique().rename("n_ndcs_tmp").reset_index()
    summary = summary.merge(total_ndcs, on=["drug_group_key", "year_month"], how="left")
    summary["dominant_labeler_share_ndcs"] = (
        summary["dominant_labeler_ndc_count"].fillna(0.0) / summary["n_ndcs_tmp"].clip(lower=1.0)
    ).astype(np.float32)
    summary["dominant_application_share_ndcs"] = (
        summary["dominant_application_ndc_count"].fillna(0.0) / summary["n_ndcs_tmp"].clip(lower=1.0)
    ).astype(np.float32)
    summary["ndc_fragmentation_inverse"] = (
        summary["n_ndcs_tmp"].clip(lower=1.0) / summary["n_labelers_group"].clip(lower=1.0)
    ).astype(np.float32)
    summary = summary.drop(columns=["dominant_labeler_ndc_count", "dominant_application_ndc_count", "n_ndcs_tmp"])

    fill_zero = [c for c in summary.columns if c not in {"drug_group_key", "year_month"}]
    for col in fill_zero:
        summary[col] = summary[col].fillna(0.0).astype(np.float32)
    return summary


def add_recency_feature(df, source_col, feature_name, positive_threshold=0.0):
    values = df[source_col].fillna(0.0).to_numpy(dtype=np.float32)
    out = np.zeros(len(df), dtype=np.float32)
    last_idx = -1
    for i, val in enumerate(values):
        if last_idx < 0:
            out[i] = 999.0
        else:
            out[i] = float(i - last_idx)
        if val > positive_threshold:
            last_idx = i
    df[feature_name] = out


def add_group_history_features(grouped):
    grouped = grouped.copy()
    grouped["period"] = pd.PeriodIndex(grouped["year_month"], freq="M")
    grouped = grouped.sort_values(["drug_group_key", "period"]).reset_index(drop=True)

    dynamic_cols = [
        "n_manufacturers",
        "n_api_suppliers",
        "n_facilities",
        "nadac_per_unit",
        "nadac_pct_change_3m",
        "nadac_pct_change_12m",
        "ae_reports_3m",
        "ae_trend_12m",
        "labeler_shortage_burden",
        "same_ingredient_in_shortage",
        "disaster_count_3m",
        "api_disaster_exposure_3m",
        "quality_signal_count",
        "price_shock",
        "commercial_risk",
        "contagion_stress",
    ]

    frames = []
    for _, g in grouped.groupby("drug_group_key", sort=False):
        g = g.copy()
        for col in dynamic_cols:
            s = g[col].fillna(0.0)
            g[f"{col}_lag3"] = s.shift(3).fillna(0.0).astype(np.float32)
            g[f"{col}_lag6"] = s.shift(6).fillna(0.0).astype(np.float32)
            g[f"{col}_delta3"] = (s - s.shift(3).fillna(0.0)).astype(np.float32)
            g[f"{col}_delta6"] = (s - s.shift(6).fillna(0.0)).astype(np.float32)
            g[f"{col}_rollmax6"] = s.shift(1).rolling(6, min_periods=1).max().fillna(0.0).astype(np.float32)
            g[f"{col}_rollmean6"] = s.shift(1).rolling(6, min_periods=1).mean().fillna(0.0).astype(np.float32)

        shortage_shift = g["shortage"].fillna(0.0).shift(1).fillna(0.0)
        start_shift = g["shortage_start"].fillna(0.0).shift(1).fillna(0.0)
        end_shift = g["shortage_end"].fillna(0.0).shift(1).fillna(0.0)
        g["shortage_months_past_12m"] = shortage_shift.rolling(12, min_periods=1).sum().astype(np.float32)
        g["shortage_months_past_24m"] = shortage_shift.rolling(24, min_periods=1).sum().astype(np.float32)
        g["shortage_starts_past_24m"] = start_shift.rolling(24, min_periods=1).sum().astype(np.float32)
        g["shortage_ends_past_24m"] = end_shift.rolling(24, min_periods=1).sum().astype(np.float32)
        g["ever_shortage_before"] = (start_shift.cumsum() > 0).astype(np.float32)
        g["shortage_burden_past_24m"] = (g["shortage_months_past_24m"] / 24.0).astype(np.float32)

        shortage_state = g["shortage"].fillna(0.0).to_numpy(dtype=np.int8)
        last_episode_duration = np.zeros(len(g), dtype=np.float32)
        mean_episode_duration = np.zeros(len(g), dtype=np.float32)
        max_episode_duration = np.zeros(len(g), dtype=np.float32)
        months_since_group_resolution = np.full(len(g), 999.0, dtype=np.float32)
        months_since_group_onset = np.full(len(g), 999.0, dtype=np.float32)
        recent_resolution_rebound = np.zeros(len(g), dtype=np.float32)

        past_durations = []
        run_len = 0
        last_resolution_idx = None
        last_onset_idx = None
        prev = 0
        for i, val in enumerate(shortage_state):
            if past_durations:
                last_episode_duration[i] = float(past_durations[-1])
                mean_episode_duration[i] = float(np.mean(past_durations))
                max_episode_duration[i] = float(np.max(past_durations))
            if last_resolution_idx is not None:
                months_since_group_resolution[i] = float(i - last_resolution_idx)
                recent_resolution_rebound[i] = float(max(0, 6 - (i - last_resolution_idx)))
            if last_onset_idx is not None:
                months_since_group_onset[i] = float(i - last_onset_idx)

            if val == 1 and prev == 0:
                last_onset_idx = i
                run_len = 1
            elif val == 1:
                run_len += 1
            elif val == 0 and prev == 1:
                past_durations.append(float(run_len))
                last_resolution_idx = i
                run_len = 0
            prev = int(val)

        g["last_group_episode_duration"] = last_episode_duration
        g["mean_group_episode_duration"] = mean_episode_duration
        g["max_group_episode_duration"] = max_episode_duration
        g["months_since_group_resolution"] = months_since_group_resolution
        g["months_since_group_onset"] = months_since_group_onset
        g["recent_resolution_rebound"] = recent_resolution_rebound
        g["repeat_shortage_flag"] = (g["shortage_starts_past_24m"] >= 2).astype(np.float32)
        g["episode_duration_memory"] = (
            0.6 * g["last_group_episode_duration"] + 0.4 * g["mean_group_episode_duration"]
        ).astype(np.float32)
        g["recurrence_pressure"] = (
            g["repeat_shortage_flag"] * (1.0 + g["shortage_burden_past_24m"])
        ).astype(np.float32)

        add_recency_feature(g, "warning_letter_12m", "months_since_warning", 0.0)
        add_recency_feature(g, "recall_count_12m", "months_since_recall", 0.0)
        add_recency_feature(g, "oai_inspection_12m", "months_since_oai", 0.0)
        add_recency_feature(g, "major_disaster_12m", "months_since_major_disaster", 0.0)
        add_recency_feature(g, "recent_manufacturer_exit", "months_since_manufacturer_exit", 0.0)
        add_recency_feature(g, "same_ingredient_in_shortage", "months_since_ingredient_pressure", 0.0)
        add_recency_feature(g, "has_utilization_data", "months_since_utilization_observed", 0.0)
        if "has_symphony_data" not in g.columns:
            g["has_symphony_data"] = 0.0
        add_recency_feature(g, "has_symphony_data", "months_since_symphony_observed", 0.0)

        medicaid = g["medicaid_rx_count"].replace(0, np.nan)
        partd = g["partd_total_claims"].replace(0, np.nan)
        g["medicaid_rx_count_last6"] = medicaid.shift(6).fillna(0.0).astype(np.float32)
        g["partd_total_claims_last6"] = partd.shift(6).fillna(0.0).astype(np.float32)
        g["medicaid_spending_last6"] = g["medicaid_spending"].replace(0, np.nan).shift(6).fillna(0.0).astype(np.float32)
        g["partd_avg_cost_per_claim_last6"] = g["partd_avg_cost_per_claim"].replace(0, np.nan).shift(6).fillna(0.0).astype(np.float32)
        g["medicaid_trailing_mean_last12"] = medicaid.shift(3).rolling(12, min_periods=1).mean().fillna(0.0).astype(np.float32)
        g["partd_trailing_mean_last12"] = partd.shift(3).rolling(12, min_periods=1).mean().fillna(0.0).astype(np.float32)
        g["utilization_claims_ratio_last6"] = (
            g["medicaid_rx_count_last6"] / np.maximum(g["partd_total_claims_last6"], 1.0)
        ).astype(np.float32)

        symphony_observed = g["has_symphony_data"].fillna(0.0).astype(np.float32)
        symphony_observed_lag = symphony_observed.shift(1)
        g["has_symphony_data_lag1"] = symphony_observed_lag.fillna(0.0).astype(np.float32)
        g["symphony_observed_share_last3"] = (
            symphony_observed_lag.rolling(3, min_periods=1).mean().fillna(0.0).astype(np.float32)
        )
        g["symphony_observed_share_last6"] = (
            symphony_observed_lag.rolling(6, min_periods=1).mean().fillna(0.0).astype(np.float32)
        )

        for col in SYMPHONY_LAG_NUMERIC:
            if col not in g.columns:
                continue
            s = pd.to_numeric(g[col], errors="coerce").fillna(0.0).astype(np.float32)
            lag1 = s.shift(1)
            g[f"{col}_lag1"] = lag1.fillna(0.0).astype(np.float32)
            g[f"{col}_lag3"] = s.shift(3).fillna(0.0).astype(np.float32)
            g[f"{col}_lag6"] = s.shift(6).fillna(0.0).astype(np.float32)
            if col in SYMPHONY_ROLLING_NUMERIC:
                g[f"{col}_rollmean3"] = (
                    lag1.rolling(3, min_periods=1).mean().fillna(0.0).astype(np.float32)
                )
                g[f"{col}_rollmean6"] = (
                    lag1.rolling(6, min_periods=1).mean().fillna(0.0).astype(np.float32)
                )
                g[f"{col}_rollmean12"] = (
                    lag1.rolling(12, min_periods=1).mean().fillna(0.0).astype(np.float32)
                )
                g[f"{col}_rollsum6"] = (
                    lag1.rolling(6, min_periods=1).sum().fillna(0.0).astype(np.float32)
                )
                prev6 = s.shift(7)
                pct_change6 = (lag1 - prev6) / np.maximum(np.abs(prev6), 1.0)
                g[f"{col}_pct_change6"] = (
                    pct_change6.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
                )

        frames.append(g)

    grouped = pd.concat(frames, ignore_index=True)
    return grouped


def add_peer_pressure_features(grouped):
    grouped = grouped.copy()
    grouped = grouped.sort_values("period").reset_index(drop=True)
    for cat_col in ["therapeutic_class", "therapeutic_category", "route", "dosage_form"]:
        monthly = grouped.groupby([cat_col, "period"], observed=False).agg(
            cat_onsets=("shortage_start", "sum"),
            cat_groups=("drug_group_key", "nunique"),
            cat_shortages=("shortage", "sum"),
        ).reset_index()
        monthly = monthly.sort_values([cat_col, "period"]).reset_index(drop=True)
        onset_shift = monthly.groupby(cat_col, observed=False)["cat_onsets"].shift(1)
        groups_shift = monthly.groupby(cat_col, observed=False)["cat_groups"].shift(1)
        shortage_shift = monthly.groupby(cat_col, observed=False)["cat_shortages"].shift(1)
        monthly["peer_onset_rate_6m"] = (
            onset_shift.groupby(monthly[cat_col], observed=False).transform(
                lambda s: s.rolling(6, min_periods=1).sum()
            )
            / np.maximum(
                groups_shift.groupby(monthly[cat_col], observed=False).transform(
                    lambda s: s.rolling(6, min_periods=1).mean()
                ),
                1.0,
            )
        ).fillna(0.0).astype(np.float32)
        monthly["peer_shortage_burden_3m"] = (
            shortage_shift.groupby(monthly[cat_col], observed=False).transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
            / np.maximum(
                groups_shift.groupby(monthly[cat_col], observed=False).transform(
                    lambda s: s.rolling(3, min_periods=1).mean()
                ),
                1.0,
            )
        ).fillna(0.0).astype(np.float32)
        monthly = monthly[[cat_col, "period", "peer_onset_rate_6m", "peer_shortage_burden_3m"]]
        grouped = grouped.merge(monthly, on=[cat_col, "period"], how="left", suffixes=("", f"_{cat_col}"))
        grouped.rename(
            columns={
                "peer_onset_rate_6m": f"{cat_col}_peer_onset_rate_6m",
                "peer_shortage_burden_3m": f"{cat_col}_peer_shortage_burden_3m",
            },
            inplace=True,
        )
    return grouped


def add_interaction_features(grouped):
    grouped = grouped.copy()
    grouped["peer_vulnerability_interaction"] = (
        grouped["therapeutic_class_peer_onset_rate_6m"].fillna(0.0)
        * (1.0 + grouped["market_vulnerability"].fillna(0.0) + grouped["share_low_competition_ndcs"].fillna(0.0))
    ).astype(np.float32)
    grouped["quality_peer_interaction"] = (
        grouped["route_peer_shortage_burden_3m"].fillna(0.0)
        * (1.0 + grouped["quality_signal_count"].fillna(0.0) + grouped["share_quality_signal_ndcs"].fillna(0.0))
    ).astype(np.float32)
    grouped["recurrence_x_peer_pressure"] = (
        grouped["recurrence_pressure"].fillna(0.0)
        * (1.0 + grouped["dosage_form_peer_onset_rate_6m"].fillna(0.0))
    ).astype(np.float32)
    grouped["supplier_concentration_risk"] = (
        grouped["dominant_labeler_share_ndcs"].fillna(0.0)
        + grouped["dominant_application_share_ndcs"].fillna(0.0)
        + grouped["share_api_concentrated_ndcs"].fillna(0.0)
    ).astype(np.float32)
    grouped["rebound_risk"] = (
        grouped["recent_resolution_rebound"].fillna(0.0)
        * (1.0 + grouped["episode_duration_memory"].fillna(0.0) / 12.0)
    ).astype(np.float32)
    return grouped


def _norm_ingredient_for_dea(name):
    """Mirror 04j_dea_quota_features._norm_ingredient so ingredient-level DEA
    production-quota features join to a group's nonproprietary name."""
    s = str(name).lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\b[dl],?[dl]?-", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def merge_leading_indicators(grouped):
    """Left-join the supply-chain leading-indicator feature families onto the
    grouped panel. Group-level families (FDA compliance citations, SEC supply
    disruptions, facility/API disaster exposure) join on (drug_group_key,
    year_month); DEA production quotas are ingredient-level and join on a
    normalized nonproprietary name. Every family is built lag-safe (rolling
    windows end at t-1), so none can leak future information. A missing file is
    skipped with a warning. These features are experimental and are excluded
    unless INCLUDE_EXPERIMENTAL_LEADING_INDICATORS=1 is set.
    """
    if not INCLUDE_EXPERIMENTAL_LEADING_INDICATORS:
        return grouped
    for fname in LEADING_INDICATOR_GROUP_FILES:
        fpath = INTERMEDIATE / fname
        if not fpath.exists():
            print(f"  [leading-indicator] missing {fname}; skipping")
            continue
        li = pd.read_parquet(fpath)
        feat_cols = [c for c in li.columns if c not in {"drug_group_key", "year_month"}]
        grouped = grouped.merge(li, on=["drug_group_key", "year_month"], how="left")
        for c in feat_cols:
            grouped[c] = grouped[c].fillna(0.0)
    dea_path = INTERMEDIATE / DEA_QUOTA_FILE
    if dea_path.exists():
        dea = pd.read_parquet(dea_path).rename(columns={"ingredient_norm": "_ing"})
        dea_cols = [c for c in dea.columns if c not in {"_ing", "year_month"}]
        grouped["_ing"] = grouped["drug_group_name"].map(_norm_ingredient_for_dea)
        grouped = grouped.merge(dea, on=["_ing", "year_month"], how="left")
        for c in dea_cols:
            grouped[c] = grouped[c].fillna(0.0)
        grouped = grouped.drop(columns=["_ing"])
    else:
        print(f"  [leading-indicator] missing {DEA_QUOTA_FILE}; skipping")
    return grouped


def load_group_panel(include_unevaluable=False, exclude_current_shortage=True):
    dynamic_numeric = []
    available = None
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(PANEL_PATH).names)
        dynamic_numeric = sorted(
            c for c in available
            if c.endswith("_missing") or c in {"is_repackager", "repackager_rows_retained"}
        )
    except Exception:
        dynamic_numeric = []

    cols = [
        "year_month",
        "ndc_11",
        "nonproprietary_name",
        "dosage_form",
        "shortage",
        "shortage_start",
        "shortage_end",
        "shortage_end_imputed",
    ] + CATEGORICAL + BASE_NUMERIC + UTILIZATION_NUMERIC + RAW_GROUP_EXTRA + dynamic_numeric
    cols = list(dict.fromkeys(cols))
    if available is not None:
        cols = [c for c in cols if c in available]
    panel = pd.read_parquet(PANEL_PATH, columns=cols)
    panel["drug_group_key"] = [
        group_key(n, d, r)
        for n, d, r in zip(panel["nonproprietary_name"], panel["dosage_form"], panel["route"])
    ]
    panel["drug_group_name"] = panel["nonproprietary_name"].fillna("UNKNOWN")
    panel["dosage_form_name"] = panel["dosage_form"].fillna("UNKNOWN")
    composition = add_group_composition_features(panel)

    agg = {
        "ndc_11": "nunique",
        "shortage": "max",
        "shortage_start": "max",
        "shortage_end": "max",
        "drug_group_name": "first",
        "dosage_form_name": "first",
    }
    if "shortage_end_imputed" in panel.columns:
        agg["shortage_end_imputed"] = "max"
    for col in CATEGORICAL:
        if col in panel.columns:
            agg[col] = "first"
    for col in BASE_NUMERIC + UTILIZATION_NUMERIC:
        if col not in panel.columns:
            continue
        if col in SYMPHONY_SUM_NUMERIC:
            agg[col] = "sum"
        elif col in SYMPHONY_MEDIAN_NUMERIC:
            agg[col] = "median"
        elif col in {
            "nadac_per_unit",
            "nadac_pct_change_3m",
            "nadac_pct_change_12m",
            "nadac_generic_ratio",
            "nadac_vs_market_median",
            "partd_avg_cost_per_claim",
        }:
            agg[col] = "median"
        else:
            agg[col] = "max"
    for col in dynamic_numeric:
        if col in panel.columns and col not in agg:
            agg[col] = "max"

    grouped = panel.groupby(["drug_group_key", "year_month"], as_index=False).agg(agg)
    grouped.rename(columns={"ndc_11": "n_ndcs"}, inplace=True)
    grouped = grouped.merge(composition, on=["drug_group_key", "year_month"], how="left")
    grouped = grouped.sort_values(["drug_group_key", "year_month"]).reset_index(drop=True)
    for col in CATEGORICAL:
        grouped[col] = grouped[col].fillna("UNKNOWN").astype("category")

    grouped = add_engineered(grouped)
    grouped = add_group_history_features(grouped)
    grouped = add_peer_pressure_features(grouped)
    grouped = add_interaction_features(grouped)
    grouped = grouped.sort_values(["drug_group_key", "year_month"]).reset_index(drop=True)
    grouped = merge_leading_indicators(grouped)

    for h in range(1, PREDICTION_HORIZON + 1):
        grouped[f"onset_t{h}"] = grouped.groupby("drug_group_key")["shortage_start"].shift(-h)
    horizon_cols = [f"onset_t{h}" for h in range(1, PREDICTION_HORIZON + 1)]
    # A 6-month target is evaluable only when all six future months are present.
    # pandas.DataFrame.max(skipna=True) would otherwise convert incomplete tail
    # windows into false negatives whenever no observed future onset had appeared.
    horizon_complete = grouped[f"onset_t{PREDICTION_HORIZON}"].notna()
    grouped["onset_any6"] = grouped[horizon_cols].max(axis=1)
    grouped.loc[~horizon_complete, "onset_any6"] = np.nan

    latest_prediction_month = grouped["year_month"].max()
    grouped = grouped[(grouped["year_month"] >= STUDY_START) & (grouped["year_month"] <= latest_prediction_month)].copy()
    if exclude_current_shortage:
        grouped = grouped[grouped["shortage"] == 0].copy().reset_index(drop=True)
    evaluable = grouped[grouped["onset_t6"].notna()].copy()
    latest_evaluable_month = evaluable["year_month"].max()
    if include_unevaluable:
        return grouped, latest_prediction_month, latest_evaluable_month
    return evaluable.reset_index(drop=True), latest_prediction_month, latest_evaluable_month


def build_feature_columns(grouped):
    excluded = {
        "drug_group_key",
        "drug_group_name",
        "dosage_form_name",
        "year_month",
        "period",
        "shortage",
        "shortage_start",
        "shortage_end",
        "onset_any6",
    }
    excluded.update({f"onset_t{h}" for h in range(1, PREDICTION_HORIZON + 1)})
    feature_cols = [c for c in grouped.columns if c not in excluded]
    feature_cols = [c for c in feature_cols if c not in UNSAFE_CURRENT_UTILIZATION]
    feature_cols = [c for c in feature_cols if not (c.startswith("symphony_") and c.endswith("_missing"))]
    feature_cols = [c for c in feature_cols if grouped[c].notna().mean() > 0.1]
    return feature_cols


def rolling_months(grouped):
    months = sorted(grouped["year_month"].unique())
    return [m for m in months if ROLLING_START <= m <= grouped["year_month"].max()]


def month_index_map(months):
    return {month: idx for idx, month in enumerate(months)}


def label_complete_months_before(month_order, reference_month, horizon_months=PREDICTION_HORIZON):
    """Months whose forward labels are observable at `reference_month`."""
    reference_period = pd.Period(reference_month, freq="M")
    cutoff = reference_period - int(horizon_months)
    return [m for m in month_order if pd.Period(m, freq="M") <= cutoff]


def restrict_train_history(
    df,
    reference_month,
    month_order,
    history_months=TRAIN_HISTORY_MONTHS,
    label_horizon_months=PREDICTION_HORIZON,
):
    """Keep a recent window whose labels are observable at `reference_month`.

    For the 6-month onset target, a row at month t is usable for a forecast
    issued in month m only when t + 6 <= m. The default window therefore ends
    at m - 6, not m - 1.
    """
    eligible_months = label_complete_months_before(
        month_order, reference_month, label_horizon_months
    )
    if not eligible_months:
        return df.iloc[0:0].copy()
    if history_months is not None and history_months > 0:
        eligible_months = eligible_months[-history_months:]
    allowed_months = set(eligible_months)
    return df[df["year_month"].isin(allowed_months)].copy()


def build_recent_validation_split(train_df):
    """Use a recent contiguous validation block with enough positives."""
    train_months = sorted(train_df["year_month"].unique())
    if len(train_months) <= VALIDATION_RECENT_MONTHS:
        return train_df.copy(), None
    max_months = min(VALIDATION_MAX_MONTHS, len(train_months) - 1)
    for val_month_count in range(VALIDATION_RECENT_MONTHS, max_months + 1):
        val_months = train_months[-val_month_count:]
        val_df = train_df[train_df["year_month"].isin(val_months)].copy()
        train_only = train_df[~train_df["year_month"].isin(val_months)].copy()
        if int(val_df["onset_any6"].sum()) >= VALIDATION_MIN_POSITIVES and int(train_only["onset_any6"].sum()) > 0:
            return train_only, val_df
    return train_df.copy(), None


def recency_weights(month_series, reference_month, month_order):
    month_to_idx = month_index_map(month_order)
    ref_idx = month_to_idx[reference_month]
    month_idx = month_series.map(month_to_idx).fillna(ref_idx).to_numpy(dtype=np.int64)
    months_ago = np.clip(ref_idx - month_idx, 0, None).astype(np.float32)
    decay = np.exp(-np.log(2.0) * months_ago / max(float(RECENCY_HALF_LIFE_MONTHS), 1.0))
    return np.clip(decay, RECENCY_WEIGHT_FLOOR, None)


def sample_train(df, max_rows, reference_month=None, month_order=None):
    if len(df) <= max_rows:
        return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    if reference_month is None:
        reference_month = max(sorted(df["year_month"].unique()))
    if month_order is None:
        month_order = sorted(df["year_month"].unique())
    pos = df[df["onset_any6"] > 0].copy()
    neg = df[df["onset_any6"] == 0].copy()
    pos_weights = recency_weights(pos["year_month"], reference_month, month_order) if len(pos) else np.array([], dtype=np.float32)
    if len(pos) >= max_rows:
        pos_weights = np.clip(pos_weights, 1e-6, None)
        pos_weights = pos_weights / pos_weights.sum()
        rng = np.random.RandomState(SEED)
        idx = rng.choice(pos.index.to_numpy(), size=max_rows, replace=False, p=pos_weights)
        return pos.loc[idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    need = max_rows - len(pos)
    weights = (
        1.0
        + 4.0 * neg["same_ingredient_in_shortage"].fillna(0.0).to_numpy(dtype=np.float32)
        + 3.5 * neg["sole_source"].fillna(0.0).to_numpy(dtype=np.float32)
        + 2.5 * neg["is_injectable"].fillna(0.0).to_numpy(dtype=np.float32)
        + 2.0 * neg["few_manufacturers"].fillna(0.0).to_numpy(dtype=np.float32)
        + 1.5 * neg["quality_signal_any"].fillna(0.0).to_numpy(dtype=np.float32)
        + 1.5 * neg["ingredient_pressure"].fillna(0.0).to_numpy(dtype=np.float32)
        + 1.2 * neg["therapeutic_class_peer_onset_rate_6m"].fillna(0.0).to_numpy(dtype=np.float32)
        + 1.2 * neg["months_since_warning"].fillna(999.0).to_numpy(dtype=np.float32).clip(0, 24) / 24.0
    )
    weights = weights * recency_weights(neg["year_month"], reference_month, month_order)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()
    rng = np.random.RandomState(SEED)
    idx = rng.choice(neg.index.to_numpy(), size=min(need, len(neg)), replace=False, p=weights)
    return pd.concat([pos, neg.loc[idx]], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def compute_metrics(y, p):
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auroc"] = 0.0
    try:
        out["auprc"] = float(average_precision_score(y, p))
    except ValueError:
        out["auprc"] = 0.0
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.5, 0.01):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    out["f1"] = float(best_f1)
    out["threshold"] = float(best_t)
    for k in RANK_KS:
        order = np.argsort(-p)
        top = np.asarray(y)[order][: min(k, len(y))]
        total_pos = max(int(np.sum(y)), 1)
        out[f"precision_at_{k}"] = float(np.mean(top)) if len(top) else 0.0
        out[f"recall_at_{k}"] = float(np.sum(top) / total_pos) if len(top) else 0.0
    return out


def fit_logistic(train_df, feature_cols):
    cat_cols = [c for c in CATEGORICAL if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    prep = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0)), ("scale", MaxAbsScaler())]), num_cols),
        ]
    )
    y = train_df["onset_any6"].values.astype(np.int8)
    pos_weight = max(float((y == 0).sum()) / max((y == 1).sum(), 1), 1.0)
    model = Pipeline(
        steps=[
            ("prep", prep),
            ("clf", LogisticRegression(solver="saga", penalty="l2", C=0.7, max_iter=500, class_weight={0: 1.0, 1: pos_weight}, n_jobs=1, random_state=SEED)),
        ]
    )
    model.fit(train_df[feature_cols], y)
    return model


LGBM_SCALE_POS_WEIGHT = 3  # Pre-specified; not computed from resampled data.


def _lgb_auprc_eval(preds, dataset):
    """Custom LightGBM evaluation function: AUPRC on validation set.

    AUPRC is used for early stopping instead of logloss because logloss on
    low-prevalence validation data is dominated by the negative class and
    stops the model before it learns to rank positives well.
    """
    labels = dataset.get_label()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return "auprc", 0.0, True
    score = float(average_precision_score(labels, preds))
    return "auprc", score, True  # True = higher is better


def fit_lgbm(train_df, feature_cols, val_df=None):
    """Fit LightGBM binary classifier for onset prediction.

    Class balance uses two complementary mechanisms:
    - `sample_train()` provides *exposure*: oversamples positives and
      informative negatives so the model sees enough rare events.
    - `scale_pos_weight` provides *gradient emphasis*: a fixed, pre-specified
      constant (LGBM_SCALE_POS_WEIGHT=5), NOT computed from the post-sampling
      distribution.  This avoids the double-correction that would arise from
      computing scale_pos_weight adaptively on already-oversampled data.

    When `val_df` is provided, the model first early-stops on validation AUPRC
    to choose the boosting-round count, then refits on the combined
    train+validation data using that selected iteration count. This preserves a
    recent contiguous validation block for model selection without discarding
    the freshest months from the final fitted model.

    When `val_df=None` (final model or backtest without a validation block), a
    fixed 350 rounds is used.
    """
    import lightgbm as lgb

    cat_cols = [c for c in CATEGORICAL if c in feature_cols]
    y = train_df["onset_any6"].values
    ds = lgb.Dataset(train_df[feature_cols], label=y, categorical_feature=cat_cols, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "verbose": -1,
        "seed": SEED,
        "feature_pre_filter": False,
        "scale_pos_weight": LGBM_SCALE_POS_WEIGHT,
        "num_leaves": 127,
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_samples": 35,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 0.2,
        "min_gain_to_split": 0.01,
    }

    if val_df is not None and len(val_df) > 0:
        y_val = val_df["onset_any6"].values
        ds_val = lgb.Dataset(val_df[feature_cols], label=y_val, categorical_feature=cat_cols, free_raw_data=False)
        selection_model = lgb.train(
            params,
            ds,
            num_boost_round=1000,
            valid_sets=[ds_val],
            valid_names=["val"],
            feval=_lgb_auprc_eval,
            callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(50)],
        )
        best_iter = max(int(getattr(selection_model, "best_iteration", selection_model.num_trees())), 1)
        full_train = pd.concat([train_df, val_df], ignore_index=True)
        ds_full = lgb.Dataset(
            full_train[feature_cols],
            label=full_train["onset_any6"].values,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            ds_full,
            num_boost_round=best_iter,
            valid_sets=[ds_full],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        model.selected_iteration = best_iter
    else:
        model = lgb.train(
            params,
            ds,
            num_boost_round=350,
            valid_sets=[ds],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        model.selected_iteration = model.num_trees()
    return model


def rolling_backtest(grouped, feature_cols):
    month_order = sorted(grouped["year_month"].unique())
    months = rolling_months(grouped)
    model_defs = [
        ("logistic_enhanced", TRAIN_ROWS_LOGIT),
        ("lgbm_enhanced", TRAIN_ROWS_LGBM),
    ]
    monthly_rows = []
    pooled_rows = []
    importance_records = []  # fold-level feature-importance snapshots
    for month in months:
        test = grouped[grouped["year_month"] == month].copy()
        if test["onset_any6"].sum() == 0:
            continue

        for model_name, max_rows in model_defs:
            train = restrict_train_history(grouped, month, month_order, TRAIN_HISTORY_MONTHS)
            if train["onset_any6"].sum() == 0:
                continue
            train_fit_base, val_df = build_recent_validation_split(train)
            fit_df = sample_train(train_fit_base, max_rows, reference_month=month, month_order=month_order)
            if model_name == "logistic_enhanced":
                model = fit_logistic(fit_df, feature_cols)
                preds = model.predict_proba(test[feature_cols])[:, 1]
            else:
                model = fit_lgbm(fit_df, feature_cols, val_df=val_df)
                preds = model.predict(test[feature_cols])

                # Monitor early stopping behavior
                if val_df is not None:
                    best_iter = getattr(model, "selected_iteration", model.num_trees())
                    print(f"    {month} lgbm: selected_iteration={best_iter}/{model.num_trees()}"
                          f" (recent val months={len(val_df['year_month'].unique())}, "
                          f"val_pos={int(val_df['onset_any6'].sum())})")

                # Collect fold-level feature importances for stability checks.
                importance_records.append({
                    "year_month": month,
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split"),
                })

            metrics = compute_metrics(test["onset_any6"].values, preds)

            # Decompose the 6-month composite endpoint by event month. These
            # diagnostics use the same composite model predictions, so they are
            # not horizon-specific model evaluations.
            horizon_metrics = {}
            for h in range(1, PREDICTION_HORIZON + 1):
                col = f"onset_t{h}"
                if col in test.columns:
                    y_h = test[col].values
                    valid = ~np.isnan(y_h)
                    if valid.sum() > 0 and y_h[valid].sum() > 0:
                        y_h_clean = y_h[valid].astype(np.int8)
                        p_h_clean = preds[valid]
                        try:
                            horizon_metrics[f"auprc_t{h}"] = float(average_precision_score(y_h_clean, p_h_clean))
                        except ValueError:
                            horizon_metrics[f"auprc_t{h}"] = 0.0
                        try:
                            horizon_metrics[f"auroc_t{h}"] = float(roc_auc_score(y_h_clean, p_h_clean))
                        except ValueError:
                            horizon_metrics[f"auroc_t{h}"] = 0.0
                    else:
                        horizon_metrics[f"auprc_t{h}"] = np.nan
                        horizon_metrics[f"auroc_t{h}"] = np.nan

            monthly_rows.append({
                "model": model_name, "year_month": month,
                "n_rows": int(len(test)), "n_pos": int(test["onset_any6"].sum()),
                **metrics, **horizon_metrics,
            })
            pooled_rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "year_month": month,
                        "drug_group_key": test["drug_group_key"].values,
                        "drug_group_name": test["drug_group_name"].values,
                        "dosage_form_name": test["dosage_form_name"].values,
                        "y": test["onset_any6"].values,
                        "pred": preds,
                    }
                )
            )
    monthly_df = pd.DataFrame(monthly_rows)
    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    summary_rows = []
    for model_name in monthly_df["model"].unique():
        mdf = monthly_df[monthly_df["model"] == model_name]
        pdf = pooled_df[pooled_df["model"] == model_name]
        pooled_metrics = compute_metrics(pdf["y"].values, pdf["pred"].values)
        summary_rows.append(
            {
                "model": model_name,
                "mean_monthly_auprc": float(mdf["auprc"].mean()),
                "mean_monthly_precision_at_100": float(mdf["precision_at_100"].mean()),
                "mean_monthly_recall_at_100": float(mdf["recall_at_100"].mean()),
                "pooled_auprc": pooled_metrics["auprc"],
                "pooled_auroc": pooled_metrics["auroc"],
                "pooled_precision_at_100": pooled_metrics["precision_at_100"],
                "pooled_recall_at_100": pooled_metrics["recall_at_100"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return monthly_df, pooled_df, summary_df, importance_records


def fit_final_and_predict(grouped, feature_cols, latest_prediction_month, selected_model):
    grouped_full, _, _ = load_group_panel(include_unevaluable=True)
    month_order = sorted(grouped_full["year_month"].unique())
    latest_raw = pd.read_parquet(PANEL_PATH, columns=["year_month", "ndc_11", "nonproprietary_name", "dosage_form", "route"])
    latest_raw["drug_group_key"] = [
        group_key(n, d, r)
        for n, d, r in zip(latest_raw["nonproprietary_name"], latest_raw["dosage_form"], latest_raw["route"])
    ]
    latest_raw = latest_raw[latest_raw["year_month"] == latest_prediction_month][["ndc_11", "drug_group_key"]].drop_duplicates()

    final_train = restrict_train_history(grouped, latest_prediction_month, month_order, FINAL_MODEL_HISTORY_MONTHS)
    fit_df = sample_train(
        final_train,
        TRAIN_ROWS_LGBM if selected_model == "lgbm_enhanced" else TRAIN_ROWS_LOGIT,
        reference_month=latest_prediction_month,
        month_order=month_order,
    )
    latest_group = grouped_full[grouped_full["year_month"] == latest_prediction_month].copy()
    if selected_model == "logistic_enhanced":
        model = fit_logistic(fit_df, feature_cols)
        preds = model.predict_proba(latest_group[feature_cols])[:, 1]
    else:
        model = fit_lgbm(fit_df, feature_cols)
        preds = model.predict(latest_group[feature_cols])

    latest_group_pred = latest_group[["drug_group_key", "drug_group_name", "dosage_form_name"]].copy()
    latest_group_pred["onset_any6_pred"] = preds.astype(np.float32)
    # The model outputs one 6-month cumulative probability. The per-horizon
    # columns below decompose it under a UNIFORM monthly-hazard assumption,
    # so onset_prob_t1 through onset_prob_t6 are all the SAME number (the
    # equivalent constant monthly hazard), not month-specific estimates.
    monthly_hazard = 1.0 - np.power(1.0 - np.clip(latest_group_pred["onset_any6_pred"].to_numpy(dtype=np.float64), 0.0, 1.0), 1.0 / PREDICTION_HORIZON)
    for h in range(1, PREDICTION_HORIZON + 1):
        latest_group_pred[f"onset_prob_t{h}"] = monthly_hazard.astype(np.float32)
    latest_group_pred["prediction_month"] = latest_prediction_month

    latest_ndc_pred = latest_raw.merge(
        latest_group_pred.drop(columns=["drug_group_name", "dosage_form_name"]),
        on="drug_group_key",
        how="left",
    ).drop(columns=["drug_group_key"])
    return model, latest_group_pred, latest_ndc_pred


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped, latest_prediction_month, latest_evaluable_month = load_group_panel()
    feature_cols = build_feature_columns(grouped)
    monthly_df, pooled_df, summary_df, importance_records = rolling_backtest(grouped, feature_cols)

    # This legacy benchmark keeps its original primary-model choice fixed so
    # the same test data are not used for model selection.
    selected_model = "lgbm_enhanced"
    final_model, latest_group_pred, latest_ndc_pred = fit_final_and_predict(grouped, feature_cols, latest_prediction_month, selected_model)

    monthly_df.to_csv(OUTPUT_DIR / "rolling_monthly_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pooled_df.to_csv(OUTPUT_DIR / "rolling_predictions.csv", index=False)
    latest_group_pred.to_parquet(OUTPUT_DIR / "latest_group_predictions.parquet", index=False)
    latest_ndc_pred.to_parquet(OUTPUT_DIR / "latest_predictions.parquet", index=False)

    # Per-horizon diagnostic summary for the legacy 6-month endpoint.
    lgbm_monthly = monthly_df[monthly_df["model"] == "lgbm_enhanced"]
    horizon_cols_auprc = [f"auprc_t{h}" for h in range(1, PREDICTION_HORIZON + 1)]
    horizon_cols_auroc = [f"auroc_t{h}" for h in range(1, PREDICTION_HORIZON + 1)]
    horizon_summary_rows = []
    for h in range(1, PREDICTION_HORIZON + 1):
        ac = f"auprc_t{h}"
        rc = f"auroc_t{h}"
        if ac in lgbm_monthly.columns:
            vals_a = lgbm_monthly[ac].dropna()
            vals_r = lgbm_monthly[rc].dropna()
            horizon_summary_rows.append({
                "horizon": h,
                "mean_auprc": float(vals_a.mean()) if len(vals_a) else np.nan,
                "std_auprc": float(vals_a.std()) if len(vals_a) > 1 else np.nan,
                "mean_auroc": float(vals_r.mean()) if len(vals_r) else np.nan,
                "std_auroc": float(vals_r.std()) if len(vals_r) > 1 else np.nan,
                "n_months": int(len(vals_a)),
            })
    if horizon_summary_rows:
        pd.DataFrame(horizon_summary_rows).to_csv(OUTPUT_DIR / "per_horizon_metrics.csv", index=False)
        print("  Saved per_horizon_metrics.csv")

    # Final-model feature importance (from last fit)
    if selected_model == "lgbm_enhanced":
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "gain": final_model.feature_importance(importance_type="gain"),
                "split": final_model.feature_importance(importance_type="split"),
            }
        ).sort_values("gain", ascending=False)
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Feature-importance stability across rolling folds.
    if importance_records:
        stability_rows = []
        for rec in importance_records:
            for i, feat in enumerate(feature_cols):
                stability_rows.append({
                    "year_month": rec["year_month"],
                    "feature": feat,
                    "gain": float(rec["gain"][i]),
                    "split": float(rec["split"][i]),
                })
        stab_df = pd.DataFrame(stability_rows)
        stab_summary = stab_df.groupby("feature").agg(
            mean_gain=("gain", "mean"),
            std_gain=("gain", "std"),
            mean_split=("split", "mean"),
            std_split=("split", "std"),
            n_folds=("gain", "count"),
        ).reset_index()
        stab_summary["cv_gain"] = stab_summary["std_gain"] / stab_summary["mean_gain"].clip(lower=1e-9)
        stab_summary["cv_split"] = stab_summary["std_split"] / stab_summary["mean_split"].clip(lower=1e-9)
        stab_summary = stab_summary.sort_values("mean_gain", ascending=False)
        stab_summary.to_csv(OUTPUT_DIR / "feature_importance_stability.csv", index=False)
        print("  Saved feature_importance_stability.csv")

    # Build summary with the fixed primary model and logistic baseline.
    lgbm_summary = summary_df[summary_df["model"] == "lgbm_enhanced"].iloc[0].to_dict() if "lgbm_enhanced" in summary_df["model"].values else {}
    logistic_summary = summary_df[summary_df["model"] == "logistic_enhanced"].iloc[0].to_dict() if "logistic_enhanced" in summary_df["model"].values else {}

    # Fold-level standard deviations for the primary model.
    summary = dict(lgbm_summary)
    summary.update(
        {
            "selected_model": selected_model,
            "latest_prediction_month": latest_prediction_month,
            "latest_evaluable_month": latest_evaluable_month,
            "n_groups": int(grouped["drug_group_key"].nunique()),
            "n_group_rows": int(len(grouped)),
            "feature_count": int(len(feature_cols)),
            "std_monthly_auprc": float(lgbm_monthly["auprc"].std()) if len(lgbm_monthly) > 1 else 0.0,
            "std_monthly_precision_at_100": float(lgbm_monthly["precision_at_100"].std()) if len(lgbm_monthly) > 1 else 0.0,
            "n_test_months": int(len(lgbm_monthly)),
            "baseline_model_metrics": logistic_summary,
        }
    )

    # API feature ablation.
    print("\n  Running API ablation backtest...")
    api_prefixes = ("n_api_", "api_india", "api_china", "api_us_", "api_india_china",
                    "api_country_hhi", "has_api_data", "api_disaster", "api_major_disaster",
                    "share_api_concentrated", "share_india_china_exposed",
                    "mean_api_country_hhi", "std_api_country_hhi",
                    "mean_api_suppliers", "std_api_suppliers",
                    "supply_chain_risk")
    api_feature_cols = [c for c in feature_cols if any(c.startswith(p) or c == p for p in api_prefixes)]
    noapi_feature_cols = [c for c in feature_cols if c not in api_feature_cols]
    print(f"    API features removed: {len(api_feature_cols)} (of {len(feature_cols)})")
    print(f"    Ablated features: {api_feature_cols}")

    # Lightweight LightGBM-only ablation backtest
    ablation_months = rolling_months(grouped)
    ablation_rows = []
    abl_month_order = sorted(grouped["year_month"].unique())
    for month in ablation_months:
        abl_test = grouped[grouped["year_month"] == month].copy()
        abl_train = restrict_train_history(grouped, month, abl_month_order, TRAIN_HISTORY_MONTHS)
        if abl_train["onset_any6"].sum() == 0 or abl_test["onset_any6"].sum() == 0:
            continue
        # Mirror the main backtest's training recipe exactly (36-month
        # history window, recent-months validation block, recency-weighted
        # sampling) so the ablation delta isolates the API features rather
        # than a difference in split construction.
        abl_train_fit, abl_val_df = build_recent_validation_split(abl_train)
        abl_fit_df = sample_train(abl_train_fit, TRAIN_ROWS_LGBM,
                                  reference_month=month, month_order=abl_month_order)
        abl_model = fit_lgbm(abl_fit_df, noapi_feature_cols, val_df=abl_val_df)
        abl_preds = abl_model.predict(abl_test[noapi_feature_cols])
        abl_metrics = compute_metrics(abl_test["onset_any6"].values, abl_preds)
        ablation_rows.append({"year_month": month, **abl_metrics})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUTPUT_DIR / "api_ablation_monthly.csv", index=False)
    print("    Saved api_ablation_monthly.csv")

    # Add ablation summary to test_metrics
    summary["api_ablation"] = {
        "n_api_features_removed": len(api_feature_cols),
        "n_remaining_features": len(noapi_feature_cols),
        "mean_monthly_auprc_no_api": float(ablation_df["auprc"].mean()) if len(ablation_df) else 0.0,
        "mean_monthly_p100_no_api": float(ablation_df["precision_at_100"].mean()) if len(ablation_df) else 0.0,
        "auprc_drop": float(lgbm_monthly["auprc"].mean() - ablation_df["auprc"].mean()) if len(ablation_df) else 0.0,
        "p100_drop": float(lgbm_monthly["precision_at_100"].mean() - ablation_df["precision_at_100"].mean()) if len(ablation_df) else 0.0,
    }

    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(summary, indent=2))

    print(summary_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(
        "23_onset_group_benchmark_enhanced.py is a legacy helper. "
        "Run Programs/25_survival_benchmarks.py or Programs/28_pipeline.py instead."
    )
