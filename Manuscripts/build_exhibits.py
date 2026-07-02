"""build_exhibits.py - Build manuscript figures and tables from the survival benchmark artifacts.

Reads Data/analysis/survival_onset, survival_offset, and survival_descriptives (written by
25_survival_benchmarks.py, with CIs from 26_bootstrap_ci.py) and writes:

Main:
  tables/table1_variables_by_domain.csv     all model features grouped by domain
  tables/table2_onset_performance.csv        onset metrics by architecture, with bootstrap CIs
  tables/table3_offset_performance.csv       offset metrics by architecture, with bootstrap CIs
  figures/fig1_onset_shap.png                onset SHAP: top-15 features (A) + top domains (B)
  figures/fig2_offset_shap.png               offset SHAP
  figures/fig1_onset_shap_full25.png         onset SHAP full top-25 feature panel
  figures/fig2_offset_shap_full25.png        offset SHAP full top-25 feature panel

Appendix:
  tables/tableA1_generic_vs_brand.csv        performance by generic vs brand
  tables/tableA2_top_therapeutic_classes.csv performance on the top-5 shortage classes
  tables/tableA3_top20_onset_risk.csv        top-20 not-in-shortage drugs at highest onset hazard
  tables/tableA4_top20_offset_resolve.csv    top-20 in-shortage drugs most likely to resolve
  tables/tableA5_feature_missingness.csv     prevalence and missingness of every feature
  tables/tableA5_feature_missingness.xlsx    machine-readable Excel copy of every feature
  tables/tableA8_feature_missingness_summary.csv compact summary for Appendix Table 8
  tables/tableA10_permutation_importance.csv permutation importance with readable feature labels
  tables/tableA11_no_prescription_benchmark.csv  canonical no-prescription onset benchmark
  tables/tableA12_operational_yield.csv      monthly fixed-budget precision and recall
  tables/tableA13_posthoc_calibration.csv    temporal Platt calibration sensitivity
  tables/tableA14_horizon6_onset.csv       existing onset scores evaluated at 6 months
  figures/figA1_event_frequency.png          onset and offset counts by calendar month
  figures/figA2_calibration.png              observed event rates by predicted-risk decile

Run: python Manuscripts/build_exhibits.py
"""
import json
import textwrap
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("utilities", HERE.parent / "Programs" / "00_utilities.py")
_util = module_from_spec(_spec)
_spec.loader.exec_module(_util)
ANALYSIS = _util.ANALYSIS
DESC = ANALYSIS / "survival_descriptives"
FIG = HERE / "generated" / "figures"
TAB = HERE / "generated" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ARCHS = ["logistic", "logistic_time_only", "lgbm", "lgbm_focal", "transformer"]
ARCH_LABEL = {
    "logistic": "Logistic",
    "logistic_time_only": "Time-only baseline",
    "lgbm": "LightGBM",
    "lgbm_focal": "LightGBM + focal",
    "transformer": "Transformer",
}
REPORT_METRICS = ["concordance", "auprc", "brier", "ece"]
METRIC_LABEL = {"concordance": "C-statistic", "auprc": "AUPRC", "brier": "Brier", "ece": "ECE",
                "precision_at_100": "Precision at 100"}
PLACEHOLDER_LABELS = {"", "UNKNOWN", "NA", "NAN", "NONE"}

TABLE1_DOMAIN_LABELS = {
    "drug_characteristics": "Drug characteristics",
    "geographic_disaster": "Natural-disaster exposure",
    "market_structure": "Market structure",
    "missingness_indicator": "Source coverage",
    "other": "Same-ingredient shortage pressure",
    "patents_exclusivity": "Patents and exclusivity",
    "pricing": "Pricing",
    "recalls": "Recalls",
    "regulatory_quality": "Regulatory quality",
    "repackagers": "Repackagers",
    "shortage_history": "Shortage history",
    "supply_chain_geography": "Supply-chain geography",
    "time_history": "Time history",
    "utilization": "Utilization",
}

TABLE1_DOMAIN_DESCRIPTIONS = {
    "drug_characteristics": (
        "Route, dosage form, therapeutic class, generic or brand status, injectable or intravenous formulation, "
        "controlled-substance status, domestic status, repackager status, marketing category, ingredient count, "
        "and calendar indicators."
    ),
    "geographic_disaster": (
        "Recent and one-year natural-disaster counts, major-disaster indicators, lagged and rolling disaster "
        "measures, time since major disaster, and the share of related NDCs with disaster exposure."
    ),
    "market_structure": (
        "Number and concentration of manufacturers, labelers, applications, active products, routes, and "
        "therapeutic classes. Measures also identify sole-source or low-competition markets, manufacturer exits, "
        "ownership changes, and market-fragility composites."
    ),
    "missingness_indicator": (
        "Indicators that a data source did or did not cover a drug-month before fill rules were applied, including "
        "pricing, utilization, market-structure, supply-chain, recall, and regulatory-quality coverage."
    ),
    "other": (
        "Generated lagged, rolling, and change measures of same-ingredient shortage pressure, plus residual "
        "engineered predictors not assigned to a source-specific domain."
    ),
    "patents_exclusivity": (
        "Active exclusivity, product and substance patents, total patent counts, recent patent expiry, patent-cliff "
        "indicators, exclusivity loss, and months to nearest expiry."
    ),
    "pricing": (
        "Acquisition and reimbursement price levels from NADAC and ASP, per-package and per-unit payment measures, "
        "prices relative to market medians, low-price flags, price shocks, and recent price changes."
    ),
    "recalls": (
        "Class I recall indicators, recall counts over 12 and 24 months, time since last recall, and recall-source "
        "coverage."
    ),
    "regulatory_quality": (
        "FDA inspection counts and outcomes, warning letters, OAI and VAI findings, adverse-event report levels "
        "and trends, composite quality-signal measures, and time since recent quality events."
    ),
    "repackagers": (
        "Number of repackagers associated with a group and indicators that repackager records were retained in the "
        "marketed-product panel."
    ),
    "shortage_history": (
        "Prior shortage starts, ends, and months in shortage for the same drug group. Related-drug pressure from "
        "the same ingredient, route, dosage form, therapeutic class, and labeler. Episode-duration and recurrence "
        "measures. Current shortage status was used only to define which drug-months were eligible for onset or "
        "resolution modeling."
    ),
    "supply_chain_geography": (
        "API supplier and registered facility counts, supplier-country concentration, domestic share, China and "
        "India exposure, low-supplier indicators, and disaster exposure near API suppliers or manufacturing "
        "facilities."
    ),
    "time_history": (
        "Elapsed time since prior group onset, resolution, warning letter, recall, nearest patent or exclusivity "
        "expiry, and last shortage, plus years since market entry."
    ),
    "utilization": (
        "Prescription volume, spending, units dispensed, prescriber counts, average prescription size, and "
        "observed-use trends from Medicaid, Medicare Part B and Part D, and Symphony Health. Measures include "
        "recent levels, prior-month values, changes, and rolling summaries."
    ),
}

DOMAIN_LABELS = {
    "drug_characteristics": "Drug characteristics",
    "geographic_disaster": "Natural-disaster exposure",
    "market_structure": "Market structure",
    "missingness_indicator": "Source coverage",
    "other": "Related-product signals",
    "patents_exclusivity": "Patents and exclusivity",
    "pricing": "Pricing",
    "recalls": "Recalls",
    "regulatory_quality": "Regulatory quality",
    "repackagers": "Repackagers",
    "shortage_history": "Shortage history",
    "supply_chain_geography": "Supply-chain geography",
    "time_history": "Time history",
    "utilization": "Utilization",
}

FEATURE_LABELS = {
    "active_ingredient_count": "Number of active ingredients",
    "ae_trend_12m_delta3": "Change in adverse-event trend",
    "ae_trend_12m_rollmax6": "Peak adverse-event trend",
    "ae_trend_12m_rollmean6": "Average adverse-event trend",
    "api_disaster_exposure_3m": "API supplier disaster exposure, 3 months",
    "api_disaster_exposure_3m_delta3": "Change in API disaster exposure, 3 months",
    "api_disaster_exposure_3m_delta6": "Change in API disaster exposure, 6 months",
    "api_disaster_exposure_3m_lag3": "API supplier disaster exposure, 3-month lag",
    "api_disaster_exposure_3m_lag6": "API supplier disaster exposure, 6-month lag",
    "api_disaster_exposure_3m_rollmax6": "Peak API disaster exposure",
    "api_disaster_exposure_3m_rollmean6": "Average API disaster exposure",
    "api_major_disaster_12m": "Major disaster near API suppliers, past year",
    "asp_billunits_per_pkg": "ASP billing units per package",
    "contagion_stress_delta3": "Change in related-shortage pressure",
    "contagion_stress_lag3": "Related-shortage pressure, 3-month lag",
    "contagion_stress_rollmean6": "Average related-shortage pressure",
    "contagion_stress_lag6": "Related-shortage pressure, 6-month lag",
    "disaster_count_3m_delta3": "Change in natural-disaster count, 3 months",
    "disaster_count_3m_delta6": "Change in natural-disaster count, 6 months",
    "disaster_count_3m_lag6": "Natural-disaster count, 6-month lag",
    "disaster_count_3m_rollmax6": "Peak recent natural-disaster count",
    "disruption_signal": "Composite natural-disaster exposure",
    "dosage_form": "Dosage form",
    "dosage_form_peer_onset_rate_6m": "Recent onset rate in same dosage form",
    "dosage_form_peer_shortage_burden_3m": "Recent shortages among same-dosage-form drugs",
    "labeler_shortage_burden_rollmean6": "Average labeler shortage burden",
    "symphony_trx_pack_units_rollmean6": "Average prescription pack units",
    "recall_count_24m": "Recall count, past 24 months",
    "api_india_china_share": "API supply share from India and China",
    "contagion_stress_rollmax6": "Peak related-shortage pressure",
    "n_api_suppliers_lag6": "Number of API suppliers, 6-month lag",
    "ae_trend_12m_lag3": "Adverse-event trend, 3-month lag",
    "ever_shortage_before": "Prior shortage history",
    "ingredient_pressure": "Same-ingredient shortage pressure",
    "is_injectable": "Injectable formulation",
    "labeler_shortage_burden_lag6": "Labeler shortage burden, 6-month lag",
    "medicaid_rx_count_last6": "Medicaid prescription volume, past 6 months",
    "medicaid_spending_last6": "Medicaid spending, past 6 months",
    "mean_api_country_hhi_group": "API supplier-country concentration",
    "months_since_group_onset": "Months since shortage began",
    "months_since_group_resolution": "Months since prior resolution",
    "months_since_ingredient_pressure": "Months since ingredient-level pressure",
    "n_api_suppliers_delta3": "Change in API supplier count, 3 months",
    "n_applications": "Number of FDA applications",
    "n_manufacturers_lag3": "Number of manufacturers, 3-month lag",
    "n_countries": "Number of API supplier countries",
    "nadac_pct_change_12m": "12-month NADAC price change",
    "nadac_pct_change_12m_lag6": "12-month NADAC price change, 6-month lag",
    "nadac_pct_change_3m_rollmean6": "Average 3-month NADAC price change",
    "nadac_per_unit": "NADAC price per unit",
    "patent_count": "Patent count",
    "price_shock_lag6": "Price shock, 6-month lag",
    "primary_country": "Primary supplier country",
    "quality_stack": "Composite quality-signal burden",
    "route": "Route of administration",
    "route_peer_onset_rate_6m": "Recent onset rate in same-route drugs",
    "route_peer_shortage_burden_3m": "Recent shortages among same-route drugs",
    "same_ingredient_in_shortage_delta3": "Change in same-ingredient shortage pressure",
    "same_ingredient_in_shortage_delta6": "Change in same-ingredient shortage pressure, 6 months",
    "shortage_months_past_12m": "Shortage months in past year",
    "sole_source": "Sole-source market",
    "supply_chain_risk": "Supply-chain risk composite",
    "symphony_avg_trx_price_lag1": "Average prescription price, 1-month lag",
    "symphony_avg_trx_price_lag3": "Average prescription price, 3-month lag",
    "symphony_nrx_pack_units_lag1": "New-prescription pack units, 1-month lag",
    "symphony_nrx_pack_units_lag3": "New-prescription pack units, 3-month lag",
    "symphony_rrx_mbs_dollars_lag3": "Refill prescription spending, 3-month lag",
    "therapeutic_class": "Therapeutic class",
    "therapeutic_class_peer_onset_rate_6m": "Recent onset rate in therapeutic class",
    "therapeutic_class_peer_shortage_burden_3m": "Recent shortages in therapeutic class",
    "time_index": "Calendar time in study",
    "year": "Calendar year",
    "years_on_market": "Years on market",
}


def save_figure_variants(fig, stem, dpi=400):
    fig.savefig(FIG / f"{stem}.png", dpi=dpi)
    fig.savefig(FIG / f"{stem}.tiff", dpi=dpi)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.svg")


def _metrics(y, p):
    y = np.asarray(y, np.int8); p = np.asarray(p, np.float64)
    out = {"n": int(len(y)), "events": int(y.sum())}
    if len(np.unique(y)) == 2:
        out["concordance"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["concordance"] = np.nan; out["auprc"] = np.nan
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else np.nan
    return out


def monthly_budget_metrics(preds, score_col="pred", k=100):
    reviewed = 0
    flagged_events = 0
    total_events = int(preds["event"].sum())
    for _, sub in preds.groupby("origin"):
        ranked = sub.sort_values(score_col, ascending=False).head(min(k, len(sub)))
        reviewed += len(ranked)
        flagged_events += int(ranked["event"].sum())
    return {
        "reviewed": int(reviewed),
        "events_flagged": int(flagged_events),
        "total_events": int(total_events),
        "precision": float(flagged_events / reviewed) if reviewed else 0.0,
        "recall": float(flagged_events / total_events) if total_events else 0.0,
    }


# ---- main tables -----------------------------------------------------------------------------
def table1_variables_by_domain():
    v = pd.read_csv(DESC / "variables_by_domain.csv")
    counts = v.groupby("domain").size().reset_index(name="n_variables").sort_values("n_variables", ascending=False)
    counts["variables"] = counts["domain"].map(TABLE1_DOMAIN_DESCRIPTIONS)
    counts["domain"] = counts["domain"].map(TABLE1_DOMAIN_LABELS).fillna(counts["domain"].str.replace("_", " "))
    counts.to_csv(TAB / "table1_variables_by_domain.csv", index=False)
    print(f"  table1: {len(v)} variables across {len(counts)} domains")


def performance_table(task):
    pooled = pd.read_csv(ANALYSIS / f"survival_{task}" / "pooled_metrics.csv")
    ci_path = ANALYSIS / f"survival_{task}" / "bootstrap_ci.csv"
    ci = pd.read_csv(ci_path) if ci_path.exists() else None
    rows = []
    for arch in ARCHS:
        pr = pooled[pooled["architecture"] == arch]
        if pr.empty:
            continue
        pr = pr.iloc[0]
        row = {"Architecture": ARCH_LABEL[arch], "N": int(pr["n"]), "Events": int(pr["positives"])}
        for m in REPORT_METRICS:
            val = pr[m]
            if ci is not None:
                c = ci[(ci["architecture"] == arch) & (ci["metric"] == m)]
                if not c.empty and not pd.isna(c.iloc[0].get("ci_lo", np.nan)):
                    row[METRIC_LABEL[m]] = f"{val:.3f} ({c.iloc[0]['ci_lo']:.3f}-{c.iloc[0]['ci_hi']:.3f})"
                    continue
            row[METRIC_LABEL[m]] = f"{val:.3f}"
        pred_file = ANALYSIS / f"survival_{task}" / f"predictions_{arch}.parquet"
        if pred_file.exists():
            monthly = monthly_budget_metrics(pd.read_parquet(pred_file), k=100)
            precision_value = monthly["precision"]
            if ci is not None:
                c = ci[(ci["architecture"] == arch) & (ci["metric"] == "precision_at_100_monthly")]
                if not c.empty and not pd.isna(c.iloc[0].get("ci_lo", np.nan)):
                    row[METRIC_LABEL["precision_at_100"]] = (
                        f"{precision_value:.3f} ({c.iloc[0]['ci_lo']:.3f}-{c.iloc[0]['ci_hi']:.3f})"
                    )
                else:
                    row[METRIC_LABEL["precision_at_100"]] = f"{precision_value:.3f}"
            else:
                row[METRIC_LABEL["precision_at_100"]] = f"{precision_value:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    n = {"onset": "2", "offset": "3"}[task]
    out.to_csv(TAB / f"table{n}_{task}_performance.csv", index=False)
    print(f"  table{n}: {task} performance for {len(out)} architectures")


def _wrap_label(text, width):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _prettify_label(value, max_len=46):
    text = str(value).replace("_", " ")
    return _wrap_label(text, max_len)


def _feature_label(value, max_len=42):
    return _wrap_label(FEATURE_LABELS.get(str(value), str(value).replace("_", " ")), max_len)


def _domain_label(value, max_len=34):
    return _wrap_label(DOMAIN_LABELS.get(str(value), str(value).replace("_", " ")), max_len)


def shap_figure(task, fignum):
    all_feat = pd.read_csv(ANALYSIS / f"survival_{task}" / "shap_features.csv")
    feat = all_feat.head(15).iloc[::-1]
    dom = pd.read_csv(ANALYSIS / f"survival_{task}" / "shap_domains.csv")
    dom = dom[dom["family"] != "outcome_target"].head(10).iloc[::-1]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 8.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axA.barh(range(len(feat)), feat["mean_abs_shap"], color="#2f6f9f")
    axA.set_yticks(range(len(feat)))
    axA.set_yticklabels([_feature_label(v, 38) for v in feat["feature"]], fontsize=9)
    axA.set_xlabel("Mean absolute SHAP", fontsize=11)
    axA.set_title(f"A. Top 15 {task} features", fontsize=12, loc="left")
    axA.tick_params(axis="x", labelsize=10)
    axB.barh(range(len(dom)), dom["mean_abs_shap"], color="#9a5b35")
    axB.set_yticks(range(len(dom)))
    axB.set_yticklabels([_domain_label(v, 30) for v in dom["family"]], fontsize=10)
    axB.set_xlabel("Summed mean absolute SHAP", fontsize=11)
    axB.set_title(f"B. {task.capitalize()} variable domains", fontsize=12, loc="left")
    axB.tick_params(axis="x", labelsize=10)
    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.0)
    save_figure_variants(fig, f"fig{fignum}_{task}_shap")
    plt.close(fig)

    full = all_feat.head(25).iloc[::-1]
    fig_full, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.barh(range(len(full)), full["mean_abs_shap"], color="#2f6f9f")
    ax.set_yticks(range(len(full)))
    ax.set_yticklabels([_feature_label(v, 44) for v in full["feature"]], fontsize=8)
    ax.set_xlabel("Mean absolute SHAP", fontsize=11)
    ax.set_title(f"Top 25 {task} features", fontsize=12, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_full.tight_layout(pad=1.0)
    save_figure_variants(fig_full, f"fig{fignum}_{task}_shap_full25")
    plt.close(fig_full)
    print(f"  fig{fignum}: {task} SHAP")


# ---- appendix --------------------------------------------------------------------------------
def _load_preds(task):
    out = {}
    for a in ARCHS:
        f = ANALYSIS / f"survival_{task}" / f"predictions_{a}.parquet"
        if f.exists():
            out[a] = pd.read_parquet(f)
    return out


def subgroup_generic_brand():
    preds = _load_preds("onset")
    rows = []
    for a, pr in preds.items():
        for label, mask in [("Generic", pr["is_generic"] == 1), ("Brand", pr["is_generic"] == 0)]:
            sub = pr[mask]
            m = _metrics(sub["event"], sub["pred"])
            rows.append({"Architecture": ARCH_LABEL[a], "Subgroup": label, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
    out = pd.DataFrame(rows).rename(columns={
        "n": "N",
        "events": "Events",
        "concordance": "C-statistic",
        "auprc": "AUPRC",
        "brier": "Brier",
    })
    out.to_csv(TAB / "tableA1_generic_vs_brand.csv", index=False)
    print("  tableA1: generic vs brand")


def subgroup_therapeutic_class(top=5):
    preds = _load_preds("onset")
    any_pr = next(iter(preds.values()))
    tc = any_pr["therapeutic_class"].astype("object")
    valid_class = tc.notna() & ~tc.astype(str).str.strip().str.upper().isin(PLACEHOLDER_LABELS)
    top_classes = (any_pr[(any_pr["event"] == 1) & valid_class]["therapeutic_class"]
                   .value_counts().head(top).index.tolist())
    rows = []
    for a, pr in preds.items():
        for tc in top_classes:
            sub = pr[pr["therapeutic_class"] == tc]
            m = _metrics(sub["event"], sub["pred"])
            rows.append({"Architecture": ARCH_LABEL[a], "Therapeutic class": tc, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
    out = pd.DataFrame(rows).rename(columns={
        "n": "N",
        "events": "Events",
        "concordance": "C-statistic",
        "auprc": "AUPRC",
        "brier": "Brier",
    })
    out.to_csv(TAB / "tableA2_top_therapeutic_classes.csv", index=False)
    print(f"  tableA2: top {top} therapeutic classes")


def top20_risk(task, fname, ref_arch="logistic"):
    preds = _load_preds(task)
    pr = preds.get(ref_arch, next(iter(preds.values())))
    latest = pr["origin"].max()
    cur = pr[pr["origin"] == latest].sort_values("pred", ascending=False).head(20)
    cols = ["drug_group_name", "route", "dosage_form_name", "therapeutic_class", "is_generic", "pred"]
    out = cur[[c for c in cols if c in cur.columns]].copy()
    out = out.rename(columns={"pred": "predicted_hazard"})
    for col in ["route", "dosage_form_name", "therapeutic_class"]:
        if col in out.columns:
            out[col] = out[col].astype("object").where(
                ~out[col].astype(str).str.strip().str.upper().isin(PLACEHOLDER_LABELS),
                "Unclassified",
            )
            out[col] = out[col].astype(str).str.replace(";", ",", regex=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    out["prediction_month"] = latest
    if "is_generic" in out.columns:
        out["is_generic"] = out["is_generic"].map({1: "Yes", 0: "No"}).fillna(out["is_generic"])
    for col in ["route", "dosage_form_name"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.title()
    if "predicted_hazard" in out.columns:
        out["predicted_hazard"] = out["predicted_hazard"].map(lambda x: f"{float(x):.4f}")
    out = out.rename(columns={
        "rank": "Rank",
        "drug_group_name": "Drug group",
        "route": "Route",
        "dosage_form_name": "Dosage form",
        "therapeutic_class": "Therapeutic class",
        "is_generic": "Generic",
        "predicted_hazard": "Predicted probability",
        "prediction_month": "Prediction month",
    })
    out.to_csv(TAB / fname, index=False)
    print(f"  {fname}: top 20 at {latest}")


def tableA5_missingness():
    m = pd.read_csv(DESC / "feature_missingness.csv").sort_values(["domain", "feature"])
    m.to_csv(TAB / "tableA5_feature_missingness.csv", index=False)
    try:
        m.to_excel(TAB / "tableA5_feature_missingness.xlsx", index=False, sheet_name="features")
    except Exception as exc:
        print(f"  tableA5 xlsx skipped: {exc}")
    # Residual post-fill missingness is zero by construction (zero fills plus indicators),
    # so the summary reports source-coverage shares instead. Coverage-share indicators are
    # has_*, *_is_observed, *_observed_share_*, and inverted *_missing; recency-type
    # indicators (months_since_*_observed) do not encode a share and are excluded, as are
    # substantive attribute flags that describe the drug rather than data availability.
    SUBSTANTIVE_HAS_FLAGS = {"has_active_exclusivity", "has_substance_patent", "has_product_patent"}

    def _coverage_share(row):
        f, v = str(row["feature"]), row["mean_or_prevalence"]
        if pd.isna(v) or f in SUBSTANTIVE_HAS_FLAGS:
            return np.nan
        if f.endswith("_missing"):
            return 1.0 - float(v)
        if f.startswith("has_") or f.endswith("_is_observed") or "_observed_share_" in f:
            return float(v)
        return np.nan

    summary = (
        m.assign(
            is_coverage_indicator=(
                m["feature"].astype(str).str.contains("missing|observed|coverage|has_", case=False, regex=True)
                & ~m["feature"].astype(str).isin(SUBSTANTIVE_HAS_FLAGS)
            ),
            coverage_share=m.apply(_coverage_share, axis=1),
        )
        .groupby("domain", as_index=False)
        .agg(
            Features=("feature", "count"),
            Coverage_or_missingness_indicators=("is_coverage_indicator", "sum"),
            Median_source_coverage=("coverage_share", "median"),
            Lowest_source_coverage=("coverage_share", "min"),
            Median_mean_or_prevalence=("mean_or_prevalence", "median"),
        )
        .sort_values("Features", ascending=False)
    )
    for col in ["Median_source_coverage", "Lowest_source_coverage"]:
        summary[col] = summary[col].map(lambda x: "NA" if pd.isna(x) else f"{float(x):.3f}")
    summary["Median_mean_or_prevalence"] = summary["Median_mean_or_prevalence"].fillna(0).map(lambda x: f"{float(x):.3f}")
    summary["domain"] = summary["domain"].map(DOMAIN_LABELS).fillna(summary["domain"].str.replace("_", " "))
    summary = summary.rename(columns={
        "domain": "Domain",
        "Coverage_or_missingness_indicators": "Coverage or missingness indicators",
        "Median_source_coverage": "Median source coverage",
        "Lowest_source_coverage": "Lowest source coverage",
        "Median_mean_or_prevalence": "Median value or prevalence",
    })
    summary.to_csv(TAB / "tableA8_feature_missingness_summary.csv", index=False)
    print(f"  tableA5: {len(m)} feature missingness rows and {len(summary)} summary rows")


def permutation_importance_table(top=20):
    p = ANALYSIS / "survival_onset" / "permutation_importance.csv"
    if not p.exists():
        print("  tableA10: skipped, permutation importance not found")
        return
    src = pd.read_csv(p).head(top).copy()
    src["Feature"] = src["feature"].map(lambda x: FEATURE_LABELS.get(str(x), str(x).replace("_", " ")))
    src["Domain"] = src["family"].map(DOMAIN_LABELS).fillna(src["family"].str.replace("_", " "))
    src["C-statistic drop"] = src["auroc_drop"].map(lambda x: f"{float(x):.4f}")
    src[["Feature", "Domain", "C-statistic drop"]].to_csv(TAB / "tableA10_permutation_importance.csv", index=False)
    print(f"  tableA10: permutation importance for {len(src)} features")


def no_prescription_table():
    out_dir = ANALYSIS / "onset_without_licensed_prescription_full" / "survival_onset"
    p = out_dir / "pooled_metrics.csv"
    if not p.exists():
        print("  tableA11: skipped, no-prescription benchmark not found")
        return
    m = pd.read_csv(p)
    summary_path = out_dir / "summary.json"
    n_features = ""
    if summary_path.exists():
        n_features = json.loads(summary_path.read_text()).get("n_features", "")
    rows = []
    for _, r in m.iterrows():
        arch = r["architecture"]
        preds_path = out_dir / f"predictions_{arch}.parquet"
        precision_at_100 = r.get("precision_at_100", np.nan)
        if preds_path.exists():
            precision_at_100 = monthly_budget_metrics(pd.read_parquet(preds_path), k=100)["precision"]
        rows.append({
            "Task": "Onset",
            "Model": ARCH_LABEL.get(arch, arch),
            "Excluded features": "Licensed Symphony prescription features",
            "N": int(r["n"]),
            "Events": int(r["positives"]),
            "C-statistic": f"{r['concordance']:.3f}",
            "AUPRC": f"{r['auprc']:.3f}",
            "Brier": f"{r['brier']:.3f}",
            "ECE": f"{r['ece']:.3f}",
            "Precision at 100": f"{precision_at_100:.3f}",
            "Features": n_features,
        })
    pd.DataFrame(rows).to_csv(TAB / "tableA11_no_prescription_benchmark.csv", index=False)
    print(f"  tableA11: no-prescription benchmark for {len(rows)} models")


def logistic_feature_sensitivity_table():
    specs = [
        (
            "Full model",
            ANALYSIS / "survival_onset",
        ),
        (
            "No licensed prescription features",
            ANALYSIS / "onset_without_licensed_prescription_full" / "survival_onset",
        ),
        (
            "No historical shortage features",
            ANALYSIS / "feature_scenario_runs" / "without_shortage_history_logistic" / "survival_onset",
        ),
    ]
    rows = []
    for scenario, out_dir in specs:
        metrics_path = out_dir / "pooled_metrics.csv"
        summary_path = out_dir / "summary.json"
        preds_path = out_dir / "predictions_logistic.parquet"
        if not metrics_path.exists() or not summary_path.exists() or not preds_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        row = metrics[metrics["architecture"] == "logistic"]
        if row.empty:
            continue
        row = row.iloc[0]
        summary = json.loads(summary_path.read_text())
        monthly = monthly_budget_metrics(pd.read_parquet(preds_path), k=100)
        rows.append({
            "Scenario": scenario,
            "Features": int(summary.get("n_features", "")),
            "Hyperparameter selection": "2022 temporal validation, AUPRC",
            "C-statistic": f"{row['concordance']:.3f}",
            "AUPRC": f"{row['auprc']:.3f}",
            "Brier": f"{row['brier']:.3f}",
            "ECE": f"{row['ece']:.3f}",
            "Monthly precision at 100": f"{monthly['precision']:.3f}",
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "tableA3_logistic_feature_sensitivity.csv", index=False)
    print(f"  tableA3: logistic feature sensitivity for {len(out)} scenarios")


def operational_yield_table():
    rows = []
    for task in ["onset", "offset"]:
        for arch, preds in _load_preds(task).items():
            for k in [25, 50, 100, 250, 500]:
                m = monthly_budget_metrics(preds, k=k)
                rows.append({
                    "Task": "Onset" if task == "onset" else "Resolution",
                    "Model": ARCH_LABEL[arch],
                    "Monthly review size": k,
                    "Reviewed": m["reviewed"],
                    "Events flagged": m["events_flagged"],
                    "Total events": m["total_events"],
                    "Precision": f"{m['precision']:.3f}",
                    "Recall": f"{m['recall']:.3f}",
                })
    pd.DataFrame(rows).to_csv(TAB / "tableA12_operational_yield.csv", index=False)
    print(f"  tableA12: monthly operational yield for {len(rows)} rows")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def posthoc_calibration_table():
    out_dir = ANALYSIS / "posthoc_calibration_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ["onset", "offset"]:
        for arch, preds in _load_preds(task).items():
            cal = preds[(preds["origin"] >= "2023-01") & (preds["origin"] <= "2023-12")].copy()
            eval_df = preds[(preds["origin"] >= "2024-01") & (preds["origin"] <= "2025-08")].copy()
            if cal.empty or eval_df.empty or cal["event"].nunique() < 2:
                continue
            x_cal = _logit(cal["pred"]).reshape(-1, 1)
            y_cal = cal["event"].to_numpy(np.int8)
            calibrator = LogisticRegression(solver="lbfgs", C=1_000_000, max_iter=1000)
            calibrator.fit(x_cal, y_cal)
            eval_df["platt_pred"] = calibrator.predict_proba(_logit(eval_df["pred"]).reshape(-1, 1))[:, 1]
            for label, score_col in [("Original", "pred"), ("Platt scaled", "platt_pred")]:
                y = eval_df["event"].to_numpy(np.int8)
                score = eval_df[score_col].to_numpy(np.float64)
                m = _metrics(y, score)
                ece = _compute_ece(y, score)
                op = monthly_budget_metrics(eval_df, score_col=score_col, k=100)
                rows.append({
                    "Task": "Onset" if task == "onset" else "Resolution",
                    "Model": ARCH_LABEL[arch],
                    "Calibration": label,
                    "Evaluation window": "2024-01 to 2025-08",
                    "N": m["n"],
                    "Events": m["events"],
                    "C-statistic": f"{m['concordance']:.3f}",
                    "AUPRC": f"{m['auprc']:.3f}",
                    "Brier": f"{m['brier']:.3f}",
                    "ECE": f"{ece:.3f}",
                    "Precision at 100": f"{op['precision']:.3f}",
                    "Platt slope": f"{float(calibrator.coef_[0][0]):.3f}",
                })
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "tableA13_posthoc_calibration.csv", index=False)
    tab.to_csv(out_dir / "metrics.csv", index=False)
    print(f"  tableA13: post-hoc calibration sensitivity for {len(tab)} rows")


def _onset_horizon_labels(preds, horizon=6):
    panel_path = ANALYSIS / "survival_grouped_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Grouped survival panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path, columns=["drug_group_key", "year_month", "shortage_start"])
    panel = panel.copy()
    panel["period_ord"] = pd.PeriodIndex(panel["year_month"].astype(str), freq="M").astype("int64")

    eval_df = preds.copy()
    eval_df["origin_ord"] = pd.PeriodIndex(eval_df["origin"].astype(str), freq="M").astype("int64")
    max_panel_ord = int(panel["period_ord"].max())
    eval_df = eval_df[eval_df["origin_ord"] + horizon <= max_panel_ord].copy()
    if eval_df.empty:
        return eval_df.assign(event_6m=pd.Series(dtype=np.int8))

    panel_keys = panel[["drug_group_key", "period_ord"]].drop_duplicates()
    coverage_keys = eval_df[["drug_group_key", "origin_ord"]].drop_duplicates().copy()
    coverage_keys["complete_horizon"] = True
    for step in range(1, horizon + 1):
        check = coverage_keys[["drug_group_key", "origin_ord"]].copy()
        check["period_ord"] = check["origin_ord"] + step
        check = check.merge(panel_keys.assign(_present=1), on=["drug_group_key", "period_ord"], how="left")
        coverage_keys["complete_horizon"] &= check["_present"].fillna(0).to_numpy(dtype=np.int8).astype(bool)
    eval_df = eval_df.merge(coverage_keys, on=["drug_group_key", "origin_ord"], how="left")
    eval_df = eval_df[eval_df["complete_horizon"].fillna(False)].copy()
    if eval_df.empty:
        return eval_df.assign(event_6m=pd.Series(dtype=np.int8))

    starts = panel[panel["shortage_start"].fillna(0).astype(float) > 0][["drug_group_key", "period_ord"]]
    origin_hits = []
    for step in range(1, horizon + 1):
        hit = starts.copy()
        hit["origin_ord"] = hit["period_ord"] - step
        origin_hits.append(hit[["drug_group_key", "origin_ord"]])
    horizon_events = pd.concat(origin_hits, ignore_index=True).drop_duplicates()
    eval_df = eval_df.merge(horizon_events.assign(event_6m=1), on=["drug_group_key", "origin_ord"], how="left")
    eval_df["event_6m"] = eval_df["event_6m"].fillna(0).astype(np.int8)
    return eval_df


def horizon6_onset_table():
    src_path = ANALYSIS / "survival_onset" / "horizon6_metrics.csv"
    if not src_path.exists():
        print("  tableA14: skipped, horizon6_metrics.csv not found")
        return
    src = pd.read_csv(src_path)
    rows = []
    for _, r in src.iterrows():
        prevalence = float(r["prevalence"])
        auprc = float(r["auprc"])
        rows.append({
            "Task": "Onset",
            "Model": ARCH_LABEL.get(r["architecture"], r["architecture"]),
            "Score used": "Existing one-month hazard",
            "Outcome": r["outcome"],
            "Evaluation window": f"{r['evaluation_start']} to {r['evaluation_end']}",
            "Origins": int(r["origins"]),
            "Dropped incomplete origins": int(r["dropped_incomplete_origins"]),
            "N": int(r["n"]),
            "Events": int(r["events"]),
            "Prevalence": f"{prevalence:.4f}",
            "C-statistic": f"{float(r['auroc']):.3f}",
            "AUPRC": f"{auprc:.3f}",
            "AUPRC lift vs prevalence": f"{(auprc / prevalence):.1f}" if prevalence > 0 else "",
            "Precision at 25": f"{float(r['precision_at_25_monthly']):.3f}",
            "Recall at 25": f"{float(r['recall_at_25_monthly']):.3f}",
            "Precision at 50": f"{float(r['precision_at_50_monthly']):.3f}",
            "Recall at 50": f"{float(r['recall_at_50_monthly']):.3f}",
            "Precision at 100": f"{float(r['precision_at_100_monthly']):.3f}",
            "Recall at 100": f"{float(r['recall_at_100_monthly']):.3f}",
            "Note": r["note"],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "tableA14_horizon6_onset.csv", index=False)
    print(f"  tableA14: 6-month onset horizon evaluation for {len(tab)} models")


def _compute_ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        return float(abs(y.mean() - p.mean()))
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += abs(float(y[mask].mean()) - float(p[mask].mean())) * (n / len(y))
    return float(ece)


def figA1_event_frequency():
    f = pd.read_csv(DESC / "event_frequency_by_month.csv")
    f["date"] = pd.PeriodIndex(f["year_month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(f["date"], f["onsets"], label="Shortage onsets", color="#a55a3b", marker="o", ms=3)
    ax.plot(f["date"], f["offsets"], label="Shortage offsets", color="#3b6ea5", marker="s", ms=3)
    ax.set_xlabel("Month"); ax.set_ylabel("Count"); ax.legend()
    ax.set_title("Shortage onsets and offsets by calendar month")
    fig.tight_layout()
    save_figure_variants(fig, "figA1_event_frequency", dpi=400)
    plt.close(fig)
    print("  figA1: event frequency")


def figA2_calibration_deciles():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    colors = {
        "logistic": "#0072B2",
        "lgbm": "#D55E00",
        "lgbm_focal": "#009E73",
        "transformer": "#CC79A7",
    }
    for ax, task, title in zip(axes, ["onset", "offset"], ["Onset", "Resolution"]):
        cpath = ANALYSIS / f"survival_{task}" / "calibration.csv"
        calib = pd.read_csv(cpath)
        for arch in ARCHS:
            sub = calib[calib["architecture"] == arch].sort_values("bin")
            if sub.empty:
                continue
            x = sub["bin"].astype(int).to_numpy() + 1
            y = sub["observed_rate"].astype(float).to_numpy() * 100
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors.get(arch, None),
                label=ARCH_LABEL[arch],
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted-risk decile")
        ax.set_xticks(range(1, 11))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Observed next-month event rate (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure_variants(fig, "figA2_calibration", dpi=400)
    plt.close(fig)
    print("  figA2: observed event rates by predicted-risk decile")


def main():
    print("Building main exhibits ...")
    table1_variables_by_domain()
    performance_table("onset")
    performance_table("offset")
    shap_figure("onset", 1)
    shap_figure("offset", 2)
    print("Building appendix exhibits ...")
    subgroup_generic_brand()
    subgroup_therapeutic_class()
    top20_risk("onset", "tableA3_top20_onset_risk.csv")
    top20_risk("offset", "tableA4_top20_offset_resolve.csv")
    tableA5_missingness()
    permutation_importance_table()
    logistic_feature_sensitivity_table()
    no_prescription_table()
    operational_yield_table()
    posthoc_calibration_table()
    horizon6_onset_table()
    figA1_event_frequency()
    figA2_calibration_deciles()
    print(f"\nExhibits written to {FIG} and {TAB}")


if __name__ == "__main__":
    main()
