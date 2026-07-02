"""25_survival_benchmarks.py - Discrete-time survival benchmarks for shortage onset and offset.

Two separate survival tasks, five model rows each, identical in temporal validation and
eligible observations, so the comparison isolates the model.

  onset   among shortage-free group-months, the hazard of entering shortage.
          Clock starts at dataset start. A group stops contributing onset rows at its first onset.
  offset  among in-shortage group-months, one subject per shortage spell, the hazard of
          the shortage ending. Clock starts when that shortage begins.

Both are discrete-time survival hazards. Each eligible subject-month is a person-period row
with a one-step-ahead event label (the transition between this month and the next). A subject
with no event before data end is right-censored. There is no fixed horizon.

Architectures (identical features, validation protocol, and metrics; only the estimator differs):
  logistic     person-period logistic regression (classic discrete-time hazard)
  logistic_time_only  person-period logistic regression using only the survival clock
  lgbm         person-period gradient-boosted hazard
  lgbm_focal   same, with a focal-loss objective
  transformer  time-aware transformer encoder over each subject's lookback window

Validation is temporal walk-forward (forecasting): at each prediction month the model is trained
only on person-months whose label is already observed, then it forecasts the hazard for the
eligible drugs at that month. Predictions are pooled out-of-time.

Metrics: concordance (AUROC of the per-period hazard), AUPRC, Brier, ECE, precision/recall at
fixed budgets, and a decile calibration table.

Outputs: Data/analysis/survival_onset/ and Data/analysis/survival_offset/.
Run: python Programs/25_survival_benchmarks.py --task both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

HERE = Path(__file__).resolve().parent

# ---- shared paths and grouped-panel feature builder ----------------------------------------
_util_spec = spec_from_file_location("utilities", HERE / "00_utilities.py")
_util_mod = module_from_spec(_util_spec)
_util_spec.loader.exec_module(_util_mod)
ANALYSIS = _util_mod.ANALYSIS

_group_spec = spec_from_file_location("group_features", HERE / "23_onset_group_benchmark_enhanced.py")
_group_mod = module_from_spec(_group_spec)
_group_spec.loader.exec_module(_group_mod)

CATEGORICAL = [c for c in _group_mod.CATEGORICAL if c != "therapeutic_category"]
UNSAFE_CURRENT_UTILIZATION = set(getattr(_group_mod, "UNSAFE_CURRENT_UTILIZATION", set()))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---- constants -----------------------------------------------------------------------------
SEED = 42
FOCAL_EPS = 1e-7
RANK_KS = [25, 50, 100, 250, 500]
LOOKBACK_MONTHS = 36
ROLLING_START = "2022-01"
TRAIN_HISTORY_MONTHS = 36
VALIDATION_RECENT_MONTHS = 6
VALIDATION_MAX_MONTHS = 12
VALIDATION_MIN_POSITIVES = 20

# Identifiers, outcome metadata, and event markers never enter the feature set.
MODEL_FEATURE_EXCLUSIONS = {
    "drug_group_key", "drug_group_name", "dosage_form_name", "year_month", "period",
    "shortage", "shortage_start", "shortage_end", "shortage_end_imputed",
    "onset_any6", "therapeutic_category",
    "reason_for_shortage", "shortage_generic_name", "shortage_company",
    "months_remaining", "episode_duration", "episode_censored", "resolved_by_6m",
    # survival bookkeeping columns added by this script
    "subject_id", "time_index", "event", "period_ord", "month", "row_id",
}
OUTCOME_PREFIXES = ("onset_t", "future_", "therapeutic_category")

ARCHITECTURES = ["logistic", "logistic_time_only", "lgbm", "lgbm_focal", "transformer"]
FEATURE_SCENARIOS = {
    "all": "All eligible model features",
    "without_licensed_prescription": "Exclude licensed Symphony prescription features",
    "without_shortage_history": "Exclude historical shortage-record features",
    "without_licensed_prescription_or_shortage_history": (
        "Exclude licensed Symphony prescription and historical shortage-record features"
    ),
}

SHORTAGE_HISTORY_FEATURES = {
    "same_ingredient_in_shortage",
    "last_group_episode_duration", "mean_group_episode_duration", "max_group_episode_duration",
    "recent_resolution_rebound", "repeat_shortage_flag", "episode_duration_memory",
    "recurrence_pressure", "recurrence_x_peer_pressure", "rebound_risk",
    "ever_shortage_before", "shortage_starts_past_24m",
    "shortage_ends_past_24m", "labeler_shortage_burden",
}
SHORTAGE_HISTORY_EXACT_EXCLUDE = {
    "time_since_last_shortage", "months_since_group_onset", "months_since_group_resolution",
    "peer_vulnerability_interaction", "quality_peer_interaction",
}
SHORTAGE_HISTORY_PREFIXES = ("peer_shortage", "peer_onset", "labeler_shortage", "ingredient_pressure")
SHORTAGE_HISTORY_TOKENS = ("contagion_stress", "rebound", "recurrence", "repeat_shortage", "group_episode")

# Identifier columns carried onto predictions for subgroup tables and top-risk rankings.
ID_COLS = ["subject_id", "drug_group_key", "drug_group_name", "dosage_form_name", "route",
           "therapeutic_class", "therapeutic_category", "is_generic", "year_month"]

# Default hyperparameters (used when no tuned config is supplied).
HP_DEFAULT = {
    "logistic": {"C": 0.7, "penalty": "l2", "class_weight": None},
    "logistic_time_only": {"C": 1.0, "penalty": "l2", "class_weight": None},
    "lgbm": {"num_leaves": 127, "max_depth": 8, "learning_rate": 0.03, "min_child_samples": 35, "scale_pos_weight_cap": 20.0},
    "lgbm_focal": {"num_leaves": 127, "max_depth": 8, "learning_rate": 0.03, "min_child_samples": 35,
                   "focal_alpha": 0.95, "focal_gamma": 2.0, "neg_pos_ratio": 20.0},
    "transformer": {"d_model": 128, "layers": 4, "lr": 1e-3, "dropout": 0.15, "loss": "focal", "heads": 8},
}

# Coarse per-architecture search grids for the temporal-carve-out tuning phase.
HP_SPACE = {
    "logistic": [
        {"C": 0.1, "penalty": "l2", "class_weight": None},
        {"C": 0.3, "penalty": "l2", "class_weight": None},
        {"C": 0.7, "penalty": "l2", "class_weight": None},
        {"C": 1.0, "penalty": "l2", "class_weight": None},
        {"C": 3.0, "penalty": "l2", "class_weight": None},
    ],
    "logistic_time_only": [
        {"C": 1.0, "penalty": "l2", "class_weight": None},
    ],
    "lgbm": [
        {"num_leaves": 127, "max_depth": 8, "learning_rate": 0.03, "min_child_samples": 35, "scale_pos_weight_cap": 20.0},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.03, "min_child_samples": 50, "scale_pos_weight_cap": 20.0},
        {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05, "min_child_samples": 50, "scale_pos_weight_cap": 50.0},
        {"num_leaves": 127, "max_depth": 8, "learning_rate": 0.05, "min_child_samples": 20, "scale_pos_weight_cap": 50.0},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.03, "min_child_samples": 100, "scale_pos_weight_cap": 10.0},
    ],
    "lgbm_focal": [
        {"num_leaves": 127, "max_depth": 8, "learning_rate": 0.03, "min_child_samples": 35, "focal_alpha": 0.95, "focal_gamma": 2.0, "neg_pos_ratio": 20.0},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.03, "min_child_samples": 50, "focal_alpha": 0.90, "focal_gamma": 2.0, "neg_pos_ratio": 20.0},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.05, "min_child_samples": 50, "focal_alpha": 0.75, "focal_gamma": 2.0, "neg_pos_ratio": 10.0},
        {"num_leaves": 127, "max_depth": 8, "learning_rate": 0.03, "min_child_samples": 35, "focal_alpha": 0.90, "focal_gamma": 1.0, "neg_pos_ratio": 20.0},
        {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.03, "min_child_samples": 50, "focal_alpha": 0.95, "focal_gamma": 2.0, "neg_pos_ratio": 50.0},
    ],
    "transformer": [
        {"d_model": 128, "layers": 4, "lr": 1e-3, "dropout": 0.15, "loss": "focal", "heads": 8},
        {"d_model": 64, "layers": 2, "lr": 1e-3, "dropout": 0.10, "loss": "bce", "heads": 4},
        {"d_model": 128, "layers": 2, "lr": 3e-4, "dropout": 0.20, "loss": "focal", "heads": 8},
        {"d_model": 64, "layers": 4, "lr": 1e-3, "dropout": 0.15, "loss": "focal", "heads": 4},
        {"d_model": 128, "layers": 4, "lr": 3e-4, "dropout": 0.15, "loss": "bce", "heads": 8},
    ],
}

TUNE_METRIC = "auprc"


# ============================================================================================
# Shared numeric helpers
# ============================================================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def set_global_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compute_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
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


def compute_metrics(y, p, ks=RANK_KS) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "mean_prediction": float(p.mean()) if len(p) else 0.0,
    }
    if len(np.unique(y)) == 2:
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auprc"] = np.nan
        out["auroc"] = np.nan
    # AUROC of the per-period event is the discrete-time concordance (C-statistic).
    out["concordance"] = out["auroc"]
    out["brier"] = float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else 0.0
    out["ece"] = compute_ece(y, p)
    order = np.argsort(-p)
    ranked_y = y[order]
    total_pos = max(int(y.sum()), 1)
    for k in ks:
        top = ranked_y[:min(k, len(ranked_y))]
        out[f"precision_at_{k}"] = float(top.mean()) if len(top) else 0.0
        out[f"recall_at_{k}"] = float(top.sum() / total_pos) if len(top) else 0.0
    return out


def score_for_tuning(y, p, metric: str = TUNE_METRIC) -> float:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) != 2:
        return 0.0
    metric = metric.lower()
    if metric == "auprc":
        return float(average_precision_score(y, p))
    if metric in {"auroc", "c_statistic", "concordance"}:
        return float(roc_auc_score(y, p))
    raise ValueError(f"Unsupported tuning metric {metric!r}")


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return pd.DataFrame()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin": i, "n": n,
            "mean_predicted": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def make_focal_binary_objective(alpha: float = 0.95, gamma: float = 2.0):
    alpha = float(np.clip(alpha, FOCAL_EPS, 1.0 - FOCAL_EPS))
    gamma = float(max(gamma, 0.0))

    def objective(preds, dataset):
        y = dataset.get_label().astype(np.float64)
        p = np.clip(sigmoid(preds), FOCAL_EPS, 1.0 - FOCAL_EPS)
        q = 1.0 - p
        grad = np.empty_like(p)
        hess = np.empty_like(p)
        pos = y >= 0.5
        neg = ~pos
        if pos.any():
            pp, qq = p[pos], q[pos]
            logp = np.log(pp)
            b = gamma * pp * logp - qq
            grad[pos] = alpha * (qq ** gamma) * b
            d_b = gamma * (logp + 1.0) + 1.0
            grad_dp = alpha * (-gamma * (qq ** (gamma - 1.0)) * b + (qq ** gamma) * d_b)
            hess[pos] = grad_dp * pp * qq
        if neg.any():
            pp, qq = p[neg], q[neg]
            logq = np.log(qq)
            c = pp - gamma * qq * logq
            na = 1.0 - alpha
            grad[neg] = na * (pp ** gamma) * c
            d_c = 1.0 + gamma * (logq + 1.0)
            grad_dp = na * (gamma * (pp ** (gamma - 1.0)) * c + (pp ** gamma) * d_c)
            hess[neg] = grad_dp * pp * qq
        hess = np.maximum(hess, 1e-6)
        w = dataset.get_weight()
        if w is not None:
            w = np.asarray(w, dtype=np.float64)
            grad *= w
            hess *= w
        return grad, hess

    return objective


def _lgb_auprc_eval(preds, dataset):
    labels = dataset.get_label()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return "auprc", 0.0, True
    return "auprc", float(average_precision_score(labels, preds)), True


def sample_case_control(df: pd.DataFrame, target: str, neg_pos_ratio: float | None):
    """Keep all events and subsample non-events with inverse weights."""
    if neg_pos_ratio is None or neg_pos_ratio <= 0:
        return df.copy(), None
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if pos.empty or neg.empty:
        return df.copy(), None
    n_neg = min(len(neg), int(math.ceil(len(pos) * float(neg_pos_ratio))))
    if n_neg >= len(neg):
        return df.copy(), None
    sampled_neg = neg.sample(n=n_neg, random_state=SEED)
    sampled = pd.concat([pos, sampled_neg], ignore_index=True)
    w = np.ones(len(sampled), dtype=np.float64)
    w[sampled[target].to_numpy() == 0] = len(neg) / max(n_neg, 1)
    perm = sampled.sample(frac=1.0, random_state=SEED).index.to_numpy()
    sampled = sampled.loc[perm].reset_index(drop=True)
    return sampled, w[perm]


# ============================================================================================
# Panel, features, and survival person-period construction
# ============================================================================================
def load_panel() -> pd.DataFrame:
    """Full group-month panel including unevaluable months, cached and re-derived if the source panel is newer."""
    cache = ANALYSIS / "survival_grouped_panel.parquet"
    src = _group_mod.PANEL_PATH
    if cache.exists() and src.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        print(f"  using cached grouped panel: {cache}", flush=True)
        df = pd.read_parquet(cache)
    else:
        grouped_all, _, _ = _group_mod.load_group_panel(include_unevaluable=True, exclude_current_shortage=False)
        df = grouped_all.drop(columns=[c for c in ["period"] if c in grouped_all.columns]).copy()
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        print(f"  cached grouped panel: {cache}", flush=True)
    if "period" in df.columns:
        df = df.drop(columns=["period"])
    df["period"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.sort_values(["drug_group_key", "period"]).reset_index(drop=True)
    df["period_ord"] = df["period"].astype("int64")
    df["month"] = df["period"].dt.month.astype(np.int16)
    return df


def select_features(panel: pd.DataFrame, screen_before: str | None = None) -> list[str]:
    """Select eligible model features using only pre-test months for coverage screening.

    `screen_before` is exclusive and normally equals the first rolling-origin test month.
    The screen is unsupervised, but keeping test-era rows out of the coverage check avoids
    letting future source availability decide which predictors enter the model.
    """
    screen = panel
    if screen_before is not None and "year_month" in panel.columns:
        screen = panel[panel["year_month"].astype(str) < str(screen_before)]
        if screen.empty:
            screen = panel
    feats = []
    for col in panel.columns:
        if col in MODEL_FEATURE_EXCLUSIONS:
            continue
        if col in UNSAFE_CURRENT_UTILIZATION or (col.startswith("symphony_") and col.endswith("_missing")):
            continue
        if any(col.startswith(p) for p in OUTCOME_PREFIXES):
            continue
        if col.startswith("onset_t"):
            continue
        if col in CATEGORICAL:
            if screen[col].notna().mean() > 0.1:
                feats.append(col)
        elif pd.api.types.is_numeric_dtype(panel[col]):
            if screen[col].notna().mean() > 0.1:
                feats.append(col)
    for col in CATEGORICAL:
        if col in feats:
            panel[col] = panel[col].astype("object").fillna("UNKNOWN").astype("category")
    return feats


def is_shortage_history_feature(feature: str) -> bool:
    """Return True when a sensitivity scenario should remove shortage-record history."""
    lower = feature.lower()
    if feature_family(feature) == "shortage_history":
        return True
    if "in_shortage" in lower:
        return True
    return lower in SHORTAGE_HISTORY_EXACT_EXCLUDE


def feature_family(feature: str) -> str:
    """Coarse source domain for the variables-by-domain table and SHAP aggregation."""
    lower_feature = feature.lower()
    if feature == "time_index":
        return "time_history"
    if feature.endswith("_missing"):
        return "missingness_indicator"
    if (lower_feature in SHORTAGE_HISTORY_FEATURES
            or lower_feature.startswith("shortage_")
            or any(tok in lower_feature for tok in SHORTAGE_HISTORY_PREFIXES)
            or any(token in lower_feature for token in SHORTAGE_HISTORY_TOKENS)):
        return "shortage_history"
    if feature.startswith(("future_", "onset_t")):
        return "outcome_target"
    if (feature in {"dosage_form", "route", "therapeutic_class", "marketing_category", "dea_schedule"}
            or feature.startswith(("is_", "active_ingredient"))
            or feature in {"covid_period", "hurricane_season", "year", "month", "quarter"}):
        return "drug_characteristics"
    if feature.startswith(("nadac_", "asp_")) or "nadac" in feature or "asp" in feature or "price_shock" in feature:
        return "pricing"
    if (feature.startswith(("medicaid_", "partd_", "partb_")) or "medicaid" in feature or "partd" in feature
            or "partb" in feature or "symphony" in feature or "utilization" in feature):
        return "utilization"
    if "recall" in feature:
        return "recalls"
    if ("inspection" in feature or "fda483" in feature or "oai" in feature or "warning_letter" in feature
            or feature.startswith("ae_") or "ae_" in feature or "adverse_event" in feature or "quality" in feature):
        return "regulatory_quality"
    if (feature.startswith(("api_", "facility_", "n_facilities", "n_countries", "is_domestic", "n_api_"))
            or "api_" in feature or "supplier" in feature or "supply_chain" in feature or "india_china" in feature
            or feature in {"few_api_suppliers", "has_geo_data", "primary_country"}):
        return "supply_chain_geography"
    if ("manufacturer" in feature or "application" in feature or "sole_source" in feature or "generic" in feature
            or "merger" in feature or "ownership" in feature or "commercial_risk" in feature
            or "low_competition" in feature or "fragmentation" in feature or "labeler" in feature
            or feature in {"capacity_stress", "dominant_labeler_share_ndcs", "has_market_structure_data",
                           "has_merger_data", "labeler_product_count", "market_vulnerability", "n_ndcs",
                           "n_routes_group", "n_therapeutic_classes_group", "ob_match_flag",
                           "peer_vulnerability_interaction"}):
        return "market_structure"
    if "repackager" in feature:
        return "repackagers"
    if "patent" in feature or "exclusivity" in feature:
        return "patents_exclusivity"
    if "disaster" in feature or feature == "disruption_signal":
        return "geographic_disaster"
    if feature.startswith(("months_", "time_since", "years_on_market")):
        return "time_history"
    return "other"


def filter_features_for_scenario(feature_cols: list[str], scenario: str) -> list[str]:
    """Apply prespecified sensitivity feature exclusions to the active model feature set."""
    if scenario not in FEATURE_SCENARIOS:
        raise ValueError(f"Unknown feature scenario {scenario!r}")
    if scenario == "all":
        return list(feature_cols)
    out = []
    for col in feature_cols:
        lower = col.lower()
        is_prescription = "symphony" in lower
        is_shortage_history = is_shortage_history_feature(col)
        if scenario in {"without_licensed_prescription", "without_licensed_prescription_or_shortage_history"} and is_prescription:
            continue
        if scenario in {"without_shortage_history", "without_licensed_prescription_or_shortage_history"} and is_shortage_history:
            continue
        out.append(col)
    return out


def package_versions() -> dict[str, str]:
    versions = {}
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "lightgbm": "lightgbm",
        "torch": "torch",
    }
    try:
        from importlib import metadata
    except Exception:
        metadata = None
    for label, package_name in packages.items():
        try:
            if metadata is None:
                raise RuntimeError("importlib.metadata unavailable")
            versions[label] = metadata.version(package_name)
        except Exception:
            versions[label] = "not_importable"
    return versions


def _logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def fit_platt_scaler(y: np.ndarray, p: np.ndarray, min_positives: int = 10):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    meta = {
        "calibrator": "identity",
        "n": int(len(y)),
        "positives": int(y.sum()),
        "reason": "",
    }
    if len(y) == 0:
        meta["reason"] = "no_validation_rows"
        return None, meta
    if int(y.sum()) < min_positives:
        meta["reason"] = f"fewer_than_{min_positives}_positive_events"
        return None, meta
    if len(np.unique(y)) < 2:
        meta["reason"] = "single_class_validation"
        return None, meta
    model = LogisticRegression(solver="lbfgs", C=1_000_000, max_iter=1000)
    model.fit(_logit_clip(p).reshape(-1, 1), y)
    meta.update({
        "calibrator": "platt",
        "reason": "fit",
        "slope": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    })
    return model, meta


def apply_platt_scaler(model, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if model is None:
        return np.clip(p, FOCAL_EPS, 1.0 - FOCAL_EPS)
    return np.clip(model.predict_proba(_logit_clip(p).reshape(-1, 1))[:, 1], FOCAL_EPS, 1.0 - FOCAL_EPS)


def assert_feature_scenario_clean(feature_cols: list[str], scenario: str) -> None:
    """Fail loudly if a shortage-history sensitivity still carries excluded signals."""
    if scenario not in {"without_shortage_history", "without_licensed_prescription_or_shortage_history"}:
        return
    bad = [c for c in feature_cols if is_shortage_history_feature(c)]
    if bad:
        preview = ", ".join(sorted(bad)[:20])
        raise AssertionError(
            f"{scenario} retained {len(bad)} shortage-history feature(s): {preview}"
        )


def feature_list_md5(feature_cols: list[str]) -> str:
    payload = "\n".join(sorted(feature_cols)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def write_feature_list(out_dir: Path, task: str, scenario: str, feature_cols: list[str],
                       screen_before: str | None) -> dict[str, object]:
    meta = {
        "task": task,
        "feature_scenario": scenario,
        "feature_screen_before": screen_before,
        "n_features": len(feature_cols),
        "feature_md5": feature_list_md5(feature_cols),
        "features": list(feature_cols),
    }
    (out_dir / f"feature_list_{scenario}.json").write_text(json.dumps(meta, indent=2))
    return meta


def compute_shap(task, person, feature_cols, out_dir, max_rows=20000, hp=None):
    """Fit one full-data LightGBM hazard model and write TreeSHAP feature and domain importances."""
    try:
        import shap
    except Exception as exc:
        print(f"  shap unavailable, skipping ({exc})", flush=True)
        return
    if int(person["event"].sum()) == 0:
        print("  no events, skipping shap", flush=True)
        return
    model = fit_lgbm(person, None, feature_cols, hp=hp)
    sample = person if len(person) <= max_rows else person.sample(max_rows, random_state=SEED)
    try:
        sv = shap.TreeExplainer(model).shap_values(sample[feature_cols])
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
    except Exception as exc:
        print(f"  shap failed for {task} ({exc})", flush=True)
        return
    feat_df = pd.DataFrame({"feature": feature_cols, "family": [feature_family(c) for c in feature_cols],
                            "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    feat_df.to_csv(out_dir / "shap_features.csv", index=False)
    dom = feat_df.groupby("family")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()
    dom.to_csv(out_dir / "shap_domains.csv", index=False)
    print(f"  shap: top feature {feat_df.iloc[0]['feature']}, top domain {dom.iloc[0]['family']}", flush=True)


def write_descriptives(panel, feature_cols, out_root):
    """Panel-level tables: variables by domain, feature prevalence/missingness, events by month."""
    d = out_root / "survival_descriptives"
    d.mkdir(parents=True, exist_ok=True)
    desc_features = list(dict.fromkeys(feature_cols + ["time_index"]))
    families = [feature_family(c) for c in desc_features]
    pd.DataFrame({"feature": desc_features, "domain": families}).sort_values(["domain", "feature"]).to_csv(
        d / "variables_by_domain.csv", index=False)
    miss = pd.DataFrame({
        "feature": desc_features, "domain": families,
        "missing_rate": [float(panel[c].isna().mean()) if c in panel.columns else 0.0 for c in desc_features],
        "mean_or_prevalence": [float(pd.to_numeric(panel[c], errors="coerce").mean())
                               if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c]) else np.nan
                               for c in desc_features],
    })
    miss.to_csv(d / "feature_missingness.csv", index=False)
    freq = (panel.groupby("year_month")[["shortage_start", "shortage_end"]].sum()
            .rename(columns={"shortage_start": "onsets", "shortage_end": "offsets"}).reset_index())
    freq.to_csv(d / "event_frequency_by_month.csv", index=False)
    print(f"  descriptives: {len(desc_features)} features across {len(set(families))} domains, "
          f"{len(freq)} months -> {d}", flush=True)


def build_person_periods(panel: pd.DataFrame, task: str) -> pd.DataFrame:
    """One row per at-risk subject-month with a one-step-ahead event label and a survival clock.

    onset:  rows are shortage-free months up to (and including) the month before the first onset.
            event = shortage in the next month. Clock = months since dataset start.
    offset: rows are in-shortage months within each shortage spell.
            event = no shortage in the next month. Clock = months since the spell start.
    Rows whose next month is unobserved are dropped (the censoring point carries no label).
    """
    dataset_start_ord = int(panel["period_ord"].min())
    parts = []
    imputed_end_spells_dropped = 0
    imputed_end_rows_dropped = 0
    for key, g in panel.groupby("drug_group_key", observed=True):
        g = g.sort_values("period_ord")
        po = g["period_ord"].to_numpy()
        sh = g["shortage"].to_numpy().astype(np.int8)
        next_sh = np.r_[sh[1:], -1]
        has_next = np.r_[(np.diff(po) == 1), False]

        if task == "onset":
            first_onset = np.argmax(sh == 1) if (sh == 1).any() else len(sh)
            at_risk = (np.arange(len(sh)) < first_onset) & (sh == 0) & has_next
            if not at_risk.any():
                continue
            sub = g.loc[at_risk].copy()
            sub["event"] = (next_sh[at_risk] == 1).astype(np.int8)
            sub["time_index"] = (po[at_risk] - dataset_start_ord).astype(np.int32)
            sub["subject_id"] = str(key)
            parts.append(sub)
        else:  # offset
            in_short = sh == 1
            if not in_short.any():
                continue
            spell_id = np.cumsum(np.r_[True, (sh[1:] == 1) & (sh[:-1] == 0)])  # increments at each shortage start
            spell_start_ord = {}
            for i in range(len(sh)):
                if in_short[i] and spell_id[i] not in spell_start_ord:
                    spell_start_ord[spell_id[i]] = po[i]
            if "shortage_end_imputed" in g.columns:
                imputed = g["shortage_end_imputed"].fillna(0).to_numpy().astype(np.int8)
            else:
                imputed = np.zeros(len(g), dtype=np.int8)
            imputed_spell_ids = set(spell_id[in_short & (imputed == 1)])
            if imputed_spell_ids:
                imputed_spell = in_short & np.isin(spell_id, list(imputed_spell_ids))
                imputed_end_spells_dropped += len(imputed_spell_ids)
                imputed_end_rows_dropped += int((imputed_spell & has_next).sum())
            else:
                imputed_spell = np.zeros(len(g), dtype=bool)
            at_risk = in_short & has_next & ~imputed_spell
            if not at_risk.any():
                continue
            sub = g.loc[at_risk].copy()
            sub["event"] = (next_sh[at_risk] == 0).astype(np.int8)
            starts = np.array([spell_start_ord[spell_id[i]] for i in np.where(at_risk)[0]])
            sub["time_index"] = (po[at_risk] - starts).astype(np.int32)
            sub["subject_id"] = [f"{key}__s{spell_id[i]}" for i in np.where(at_risk)[0]]
            parts.append(sub)

    if not parts:
        raise RuntimeError(f"No at-risk person-periods for task {task}")
    out = pd.concat(parts, ignore_index=True)
    out.attrs["imputed_end_spells_dropped"] = int(imputed_end_spells_dropped)
    out.attrs["imputed_end_rows_dropped"] = int(imputed_end_rows_dropped)
    return out


def add_time_index(panel: pd.DataFrame, task: str) -> None:
    """Add a panel-wide survival clock so the transformer lookback also carries it.

    onset:  calendar months since dataset start (same value the person-period frame uses).
    offset: months since the start of the current shortage spell, 0 outside a spell.
    """
    if task == "onset":
        start = int(panel["period_ord"].min())
        panel["time_index"] = (panel["period_ord"] - start).astype(np.int32)
        return
    sh = panel["shortage"].to_numpy().astype(np.int8)
    grp = panel["drug_group_key"].to_numpy()
    po = panel["period_ord"].to_numpy()
    same_grp = np.r_[False, grp[1:] == grp[:-1]]
    prev_sh = np.r_[0, sh[:-1]]
    is_start = (sh == 1) & ~(same_grp & (prev_sh == 1))
    start_ord = np.where(is_start, po.astype(float), np.nan)
    start_ff = pd.Series(start_ord, index=panel.index).groupby(panel["drug_group_key"]).ffill().to_numpy()
    ti = np.where(sh == 1, po - start_ff, 0.0)
    panel["time_index"] = np.nan_to_num(ti, nan=0.0).astype(np.int32)


# ============================================================================================
# Tabular architectures
# ============================================================================================
def fit_logistic(train_df, feature_cols, target="event", hp=None):
    hp = hp or HP_DEFAULT["logistic"]
    cat_cols = [c for c in CATEGORICAL if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    prep = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                          ("scale", MaxAbsScaler())]), num_cols),
    ])
    y = train_df[target].to_numpy(dtype=np.int8)
    cw = hp.get("class_weight")
    cw = None if cw in (None, "none") else cw
    model = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(solver="saga", penalty=hp.get("penalty", "l2"), C=hp.get("C", 0.7),
                                   max_iter=800, class_weight=cw, n_jobs=1, random_state=SEED)),
    ])
    model.fit(train_df[feature_cols], y)
    return model


def predict_logistic(model, frame, feature_cols):
    return np.clip(model.predict_proba(frame[feature_cols])[:, 1], FOCAL_EPS, 1 - FOCAL_EPS)


def _lgb_params(prevalence, focal=False, hp=None):
    hp = hp or {}
    p = {
        "metric": "None", "boosting_type": "gbdt", "verbose": -1, "seed": SEED,
        "feature_pre_filter": False, "num_leaves": hp.get("num_leaves", 127), "max_depth": hp.get("max_depth", 8),
        "learning_rate": hp.get("learning_rate", 0.03), "min_child_samples": hp.get("min_child_samples", 35),
        "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.15, "reg_lambda": 0.2, "min_gain_to_split": 0.01,
    }
    if focal:
        p["objective"] = make_focal_binary_objective(hp.get("focal_alpha", 0.95), hp.get("focal_gamma", 2.0))
    else:
        cap = hp.get("scale_pos_weight_cap", 20.0)
        p["objective"] = "binary"
        p["scale_pos_weight"] = min(max((1.0 - prevalence) / max(prevalence, 1e-6), 1.0), cap)
    return p


def fit_lgbm(train_df, val_df, feature_cols, target="event", focal=False, hp=None, final_train_df=None):
    import lightgbm as lgb
    hp = hp or {}
    cat_cols = [c for c in CATEGORICAL if c in feature_cols]

    def make_dataset(source_df):
        fit_df = source_df
        weight = None
        if focal:
            npr = hp.get("neg_pos_ratio", 20.0)
            if npr:
                fit_df, weight = sample_case_control(source_df, target, npr)
        y = fit_df[target].to_numpy(dtype=np.int8)
        prevalence = max(float(y.mean()), 1e-6)
        params = _lgb_params(prevalence, focal=focal, hp=hp)
        ds = lgb.Dataset(fit_df[feature_cols], label=y, weight=weight,
                         categorical_feature=cat_cols, free_raw_data=False)
        return ds, params

    train_ds, params = make_dataset(train_df)
    used_validation = val_df is not None and len(val_df) > 0 and int(val_df[target].sum()) > 0
    if used_validation:
        val_ds = lgb.Dataset(val_df[feature_cols], label=val_df[target].to_numpy(dtype=np.int8),
                             categorical_feature=cat_cols, free_raw_data=False)
        model = lgb.train(params, train_ds, num_boost_round=1000, valid_sets=[val_ds],
                          valid_names=["val"], feval=_lgb_auprc_eval,
                          callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])
        if final_train_df is not None:
            best_iter = int(getattr(model, "best_iteration", 0) or model.current_iteration() or 350)
            final_ds, final_params = make_dataset(final_train_df)
            model = lgb.train(final_params, final_ds, num_boost_round=max(best_iter, 1),
                              valid_sets=[final_ds], callbacks=[lgb.log_evaluation(0)])
    else:
        model = lgb.train(params, train_ds, num_boost_round=350, valid_sets=[train_ds],
                          callbacks=[lgb.log_evaluation(0)])
    model.outputs_raw_score = focal
    return model

def predict_lgbm(model, frame, feature_cols):
    preds = model.predict(frame[feature_cols])
    if getattr(model, "outputs_raw_score", False):
        preds = sigmoid(preds)
    return np.clip(np.asarray(preds, dtype=np.float64), FOCAL_EPS, 1 - FOCAL_EPS)


# ============================================================================================
# Transformer architecture (time-aware encoder over the subject lookback window)
# ============================================================================================
@dataclass
class TConfig:
    lookback: int = LOOKBACK_MONTHS
    max_numeric: int = 0
    neg_pos_ratio: float = 20.0
    batch_size: int = 256
    epochs: int = 12
    patience: int = 3
    d_model: int = 128
    heads: int = 8
    layers: int = 4
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "focal"
    pos_weight_cap: float = 10.0
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    cpu: bool = False


def tconfig_from_hp(hp, base):
    """Return a TConfig copy of `base` with the tunable hyperparameters overridden by `hp`."""
    import dataclasses
    hp = hp or {}
    return dataclasses.replace(
        base,
        d_model=hp.get("d_model", base.d_model), layers=hp.get("layers", base.layers),
        heads=hp.get("heads", base.heads), lr=hp.get("lr", base.lr),
        dropout=hp.get("dropout", base.dropout), loss=hp.get("loss", base.loss),
    )


@dataclass
class TPrep:
    numeric_cols: list
    categorical_cols: list
    medians: np.ndarray
    scales: np.ndarray
    category_maps: dict

    def transform_numeric(self, df):
        x = df[self.numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.medians)
        return ((x - self.medians) / self.scales).astype(np.float32)

    def transform_categorical(self, df):
        if not self.categorical_cols:
            return np.zeros((len(df), 0), dtype=np.int64)
        arr = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
        for j, col in enumerate(self.categorical_cols):
            mp = self.category_maps[col]
            arr[:, j] = [mp.get(v, 0) for v in df[col].astype("object").fillna("UNKNOWN").to_numpy()]
        return arr

    @property
    def category_sizes(self):
        return [len(self.category_maps[c]) + 1 for c in self.categorical_cols]


def fit_transformer_prep(train_df, panel, feature_cols, max_numeric):
    cat_cols = [c for c in CATEGORICAL if c in feature_cols]
    num_cands = [c for c in feature_cols if c not in cat_cols and pd.api.types.is_numeric_dtype(panel[c])]
    if not max_numeric or max_numeric <= 0 or max_numeric >= len(num_cands):
        num_cols = num_cands
    else:
        v = train_df[num_cands].apply(pd.to_numeric, errors="coerce").var(skipna=True)
        v = v.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        num_cols = v.sort_values(ascending=False).head(max_numeric).index.tolist()
    sel = train_df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    medians = np.nanmedian(sel, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(sel), sel, medians)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0).astype(np.float32)
    cmaps = {}
    for col in cat_cols:
        cats = pd.Index(train_df[col].astype("object").fillna("UNKNOWN").unique()).tolist()
        cmaps[col] = {v: i + 1 for i, v in enumerate(cats)}
    return TPrep(num_cols, cat_cols, medians, scales, cmaps)


def build_lookup(panel):
    return {(r.drug_group_key, int(r.period_ord)): int(r.row_id)
            for r in panel[["drug_group_key", "period_ord", "row_id"]].itertuples(index=False)}


class SeqDataset:
    def __init__(self, anchors, panel, num_arr, cat_arr, lookup, lookback, weights=None):
        self.row_ids = anchors["row_id"].to_numpy(np.int64)
        self.targets = anchors["event"].to_numpy(np.float32)
        self.weights = np.ones(len(anchors), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        self.num_arr = num_arr
        self.cat_arr = cat_arr
        self.lookup = lookup
        self.lookback = int(lookback)
        self.month_array = panel["month"].to_numpy(np.int64)
        self.group_array = panel["drug_group_key"].to_numpy()
        self.period_array = panel["period_ord"].to_numpy(np.int64)

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        import torch
        rid = int(self.row_ids[idx])
        group = self.group_array[rid]
        period = int(self.period_array[rid])
        L = self.lookback
        num = np.zeros((L, self.num_arr.shape[1]), dtype=np.float32)
        cat = np.zeros((L, self.cat_arr.shape[1]), dtype=np.int64)
        months = np.zeros(L, dtype=np.int64)
        mask = np.zeros(L, dtype=bool)
        for pos, p_ord in enumerate(range(period - L + 1, period + 1)):
            sid = self.lookup.get((group, p_ord))
            if sid is None:
                continue
            num[pos] = self.num_arr[sid]
            if self.cat_arr.shape[1] > 0:
                cat[pos] = self.cat_arr[sid]
            months[pos] = max(int(self.month_array[sid]) - 1, 0)
            mask[pos] = True
        return {
            "numeric": torch.from_numpy(num), "categorical": torch.from_numpy(cat),
            "month": torch.from_numpy(months), "mask": torch.from_numpy(mask),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }


def weighted_mean(loss_vec, sample_weight):
    import torch
    if sample_weight is None:
        return loss_vec.mean()
    return (loss_vec * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-6)


def focal_bce(logits, targets, alpha, gamma, sample_weight=None):
    import torch.nn.functional as F
    import torch
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return weighted_mean(a_t * (1 - p_t).pow(gamma) * bce, sample_weight)


def build_transformer(n_numeric, category_sizes, cfg):
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.numeric_proj = nn.Linear(n_numeric, d)
            self.numeric_norm = nn.LayerNorm(d)
            emb_dims = [min(16, max(4, int(round(s ** 0.25 * 4)))) for s in category_sizes]
            self.cat_emb = nn.ModuleList([nn.Embedding(s, dim, padding_idx=0)
                                          for s, dim in zip(category_sizes, emb_dims)])
            cat_dim = int(sum(emb_dims))
            self.cat_proj = nn.Linear(cat_dim, d) if cat_dim else None
            self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.month_emb = nn.Embedding(13, d)
            self.pos_emb = nn.Embedding(cfg.lookback + 1, d)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg.heads, dim_feedforward=d * 4,
                                               dropout=cfg.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))

        def forward(self, numeric, categorical, month, mask):
            x = self.numeric_norm(self.numeric_proj(numeric))
            if len(self.cat_emb):
                parts = [emb(categorical[:, :, i]) for i, emb in enumerate(self.cat_emb)]
                x = x + self.cat_proj(torch.cat(parts, dim=-1))
            x = x * self.gate(x)
            positions = torch.arange(numeric.size(1), device=numeric.device)
            x = x + self.month_emb(month.clamp(0, 11)) + self.pos_emb(positions)[None, :, :]
            cls = self.cls.expand(numeric.size(0), -1, -1)
            cls_pos = torch.full((1,), numeric.size(1), device=numeric.device, dtype=torch.long)
            cls_month = torch.full((numeric.size(0), 1), 12, device=numeric.device, dtype=torch.long)
            cls = cls + self.pos_emb(cls_pos)[None, :, :] + self.month_emb(cls_month)
            x = torch.cat([x, cls], dim=1)
            cls_mask = torch.ones((mask.size(0), 1), device=mask.device, dtype=torch.bool)
            kpm = ~torch.cat([mask.bool(), cls_mask], dim=1)
            enc = self.encoder(x, src_key_padding_mask=kpm)
            return self.head(self.norm(enc[:, -1, :])).squeeze(-1)

    return Model()


def _loader(ds, batch, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=False)


def _predict_loader(model, loader, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["numeric"].to(device), b["categorical"].to(device),
                           b["month"].to(device), b["mask"].to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _train_transformer_once(panel, train_df, val_df, feature_cols, cfg, epochs=None, early_stop=True):
    """Train one transformer model. When early_stop is true, return the best validation epoch."""
    import torch
    set_global_seeds()
    epochs = int(epochs or cfg.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    prep = fit_transformer_prep(train_df, panel, feature_cols, cfg.max_numeric)
    num_arr = prep.transform_numeric(panel)
    cat_arr = prep.transform_categorical(panel)
    lookup = build_lookup(panel)
    train_sample, sample_weight = sample_case_control(train_df, "event", cfg.neg_pos_ratio)
    if sample_weight is None:
        sample_weight = np.ones(len(train_sample), dtype=np.float32)
    tr = SeqDataset(train_sample, panel, num_arr, cat_arr, lookup, cfg.lookback, weights=sample_weight)
    va = SeqDataset(val_df, panel, num_arr, cat_arr, lookup, cfg.lookback) if (val_df is not None and len(val_df)) else None
    tr_loader = _loader(tr, cfg.batch_size, True)
    model = build_transformer(len(prep.numeric_cols), prep.category_sizes, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_tr = train_sample["event"].to_numpy(np.int8)
    w_tr = np.asarray(sample_weight, dtype=np.float64)
    neg_w = float(w_tr[y_tr == 0].sum())
    pos_w = float(w_tr[y_tr == 1].sum())
    pos_weight = min(max(neg_w / max(pos_w, 1.0), 1.0), cfg.pos_weight_cap)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
    best_state, best_val, best_epoch, stale = None, -np.inf, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for b in tr_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(b["numeric"].to(device), b["categorical"].to(device),
                           b["month"].to(device), b["mask"].to(device))
            target = b["target"].to(device)
            weight = b["weight"].to(device)
            if cfg.loss == "focal":
                loss = focal_bce(logits, target, cfg.focal_alpha, cfg.focal_gamma, weight)
            else:
                loss = weighted_mean(bce(logits, target), weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if va is not None and early_stop:
            vp = _predict_loader(model, _loader(va, cfg.batch_size * 2, False), device)
            vy = val_df["event"].to_numpy(np.int8)
            vap = average_precision_score(vy, vp) if len(np.unique(vy)) == 2 else 0.0
            if vap > best_val + 1e-5:
                best_val, best_epoch = vap, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    art = {"model": model, "prep": prep, "num_arr": num_arr, "cat_arr": cat_arr, "lookup": lookup,
           "device": device, "selected_epoch": int(best_epoch)}
    return art, int(best_epoch)


def fit_transformer(panel, train_df, val_df, feature_cols, cfg, final_train_df=None):
    """Train once; reusable across a retrain block. Returns cached artifacts for prediction."""
    art, selected_epoch = _train_transformer_once(panel, train_df, val_df, feature_cols, cfg, early_stop=(val_df is not None))
    if final_train_df is not None and val_df is not None and len(val_df) > 0:
        art, _ = _train_transformer_once(panel, final_train_df, None, feature_cols, cfg,
                                         epochs=selected_epoch, early_stop=False)
        art["selected_epoch"] = int(selected_epoch)
    return art

def predict_transformer(art, panel, test_df, cfg):
    te = SeqDataset(test_df, panel, art["num_arr"], art["cat_arr"], art["lookup"], cfg.lookback)
    return _predict_loader(art["model"], _loader(te, cfg.batch_size * 2, False), art["device"])


# ============================================================================================
# Temporal walk-forward validation
# ============================================================================================
def recent_validation_split(train_df):
    months = sorted(train_df["year_month"].unique())
    max_holdout = min(VALIDATION_MAX_MONTHS, max(len(months) - 1, 0))
    for n in range(VALIDATION_RECENT_MONTHS, max_holdout + 1):
        val_months = set(months[-n:])
        val = train_df[train_df["year_month"].isin(val_months)]
        fit = train_df[~train_df["year_month"].isin(val_months)]
        if int(val["event"].sum()) >= VALIDATION_MIN_POSITIVES and int(fit["event"].sum()) > 0:
            return fit.copy(), val.copy()
    return train_df, None


def architecture_features(arch: str, feature_cols: list[str]) -> list[str]:
    if arch == "logistic_time_only":
        return ["time_index"]
    return feature_cols


def predict_fitted_model(model_tuple, panel, frame, feature_cols, tcfg):
    kind, model = model_tuple
    if kind == "logistic":
        return predict_logistic(model, frame, feature_cols)
    if kind in ("lgbm", "lgbm_focal"):
        return predict_lgbm(model, frame, feature_cols)
    return predict_transformer(model, panel, frame, tcfg)


def run_architecture(arch, panel, person, feature_cols, origins, retrain_every, tcfg, hparams):
    """Walk-forward forecasts for one architecture. Returns the person-period test frame with preds."""
    preds_all = []
    last_model = None
    last_calibrator = None
    calibrator_rows = []
    hp = hparams.get(arch, HP_DEFAULT.get(arch, {}))
    active_features = architecture_features(arch, feature_cols)
    fit_kind = "logistic" if arch == "logistic_time_only" else arch
    tc = tconfig_from_hp(hp, tcfg) if fit_kind == "transformer" else tcfg
    for i, origin in enumerate(origins):
        test = person[person["year_month"] == origin]
        if test.empty:
            continue
        if i % retrain_every == 0 or last_model is None:
            train = person[person["year_month"] < origin]
            if int(train["event"].sum()) == 0:
                continue
            fit_df, val_df = recent_validation_split(train)
            if fit_kind == "logistic":
                last_model = ("logistic", fit_logistic(train, active_features, hp=hp))
            elif fit_kind == "lgbm":
                last_model = ("lgbm", fit_lgbm(fit_df, val_df, active_features, hp=hp, final_train_df=train))
            elif fit_kind == "lgbm_focal":
                last_model = ("lgbm_focal", fit_lgbm(fit_df, val_df, active_features, focal=True, hp=hp, final_train_df=train))
            elif fit_kind == "transformer":
                last_model = ("transformer", fit_transformer(panel, fit_df, val_df, active_features, tc, final_train_df=train))
            else:
                raise ValueError(f"Unknown architecture {arch!r}")

            if val_df is None:
                last_calibrator = None
                cal_meta = {
                    "architecture": arch, "retrain_origin": origin, "calibrator": "identity",
                    "reason": "no_inner_validation_split", "n": 0, "positives": 0,
                    "train_rows": int(len(train)), "fit_rows": int(len(fit_df)),
                }
            else:
                pv = predict_fitted_model(last_model, panel, val_df, active_features, tc)
                last_calibrator, cal_meta = fit_platt_scaler(val_df["event"].to_numpy(np.int8), pv)
                cal_meta.update({
                    "architecture": arch, "retrain_origin": origin,
                    "train_rows": int(len(train)), "fit_rows": int(len(fit_df)),
                    "validation_start": str(val_df["year_month"].min()),
                    "validation_end": str(val_df["year_month"].max()),
                })
            calibrator_rows.append(cal_meta)

        p = predict_fitted_model(last_model, panel, test, active_features, tc)
        p_cal = apply_platt_scaler(last_calibrator, p)
        keep = [c for c in ID_COLS if c in test.columns] + ["time_index", "event"]
        block = test[keep].copy()
        block["pred"] = p
        block["pred_calibrated"] = p_cal
        block["origin"] = origin
        preds_all.append(block)
        print(f"      {arch} {origin}: n={len(test):,} events={int(test['event'].sum())} "
              f"mean_p={p.mean():.4f} mean_cal={p_cal.mean():.4f}", flush=True)
    if not preds_all:
        raise RuntimeError(f"No predictions for {arch}")
    return pd.concat(preds_all, ignore_index=True), calibrator_rows


def run_task(task, panel, architectures, rolling_start, rolling_end, retrain_every, tcfg, out_root,
             do_shap=True, hparams=None, feature_scenario="all", feature_screen_before=None,
             hparams_source="defaults"):
    print(f"\n=== task={task} ===", flush=True)
    hparams = hparams or {}
    add_time_index(panel, task)
    feature_cols = select_features(panel, screen_before=feature_screen_before) + ["time_index"]
    feature_cols = filter_features_for_scenario(feature_cols, feature_scenario)
    assert_feature_scenario_clean(feature_cols, feature_scenario)
    person = build_person_periods(panel, task)
    print(f"  person-periods: {len(person):,} rows, {person['subject_id'].nunique():,} subjects, "
          f"events={int(person['event'].sum()):,} ({person['event'].mean():.3%})", flush=True)
    if task == "offset":
        print("  imputed-end offset censoring: "
              f"{person.attrs.get('imputed_end_spells_dropped', 0):,} spells and "
              f"{person.attrs.get('imputed_end_rows_dropped', 0):,} rows dropped", flush=True)

    all_months = sorted(person["year_month"].unique())
    last_origin = all_months[-1]  # every row already has an observed next month
    origins = [m for m in all_months if rolling_start <= m <= (rolling_end or last_origin)]
    print(f"  origins: {len(origins)} from {origins[0]} to {origins[-1]} (retrain every {retrain_every})", flush=True)
    print(f"  feature scenario: {feature_scenario} ({FEATURE_SCENARIOS[feature_scenario]}); "
          f"{len(feature_cols)} features", flush=True)
    if feature_screen_before is not None:
        print(f"  feature coverage screen: months before {feature_screen_before}", flush=True)

    out_dir = out_root / f"survival_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_meta = write_feature_list(out_dir, task, feature_scenario, feature_cols, feature_screen_before)
    pooled_rows, monthly_rows, calib_rows, calibrator_rows_all = [], [], [], []
    for arch in architectures:
        print(f"  -- {arch}  hp={hparams.get(arch, HP_DEFAULT.get(arch))}", flush=True)
        preds, calibrator_rows = run_architecture(arch, panel, person, feature_cols, origins, retrain_every, tcfg, hparams)
        calibrator_rows_all.extend(calibrator_rows)
        y = preds["event"].to_numpy(np.int8)
        p = preds["pred"].to_numpy(np.float64)
        pc = preds["pred_calibrated"].to_numpy(np.float64)
        m = compute_metrics(y, p)
        mc = compute_metrics(y, pc)
        m.update({
            "brier_calibrated": mc["brier"],
            "ece_calibrated": mc["ece"],
            "mean_prediction_calibrated": mc["mean_prediction"],
        })
        m.update({"task": task, "architecture": arch, "feature_scenario": feature_scenario})
        pooled_rows.append(m)
        for origin, sub in preds.groupby("origin"):
            mm = compute_metrics(sub["event"].to_numpy(np.int8), sub["pred"].to_numpy(np.float64))
            mmc = compute_metrics(sub["event"].to_numpy(np.int8), sub["pred_calibrated"].to_numpy(np.float64))
            mm.update({
                "brier_calibrated": mmc["brier"],
                "ece_calibrated": mmc["ece"],
                "mean_prediction_calibrated": mmc["mean_prediction"],
            })
            mm.update({"task": task, "architecture": arch, "origin": origin, "feature_scenario": feature_scenario})
            monthly_rows.append(mm)
        ct = calibration_table(y, p)
        ct["task"], ct["architecture"] = task, arch
        calib_rows.append(ct)
        preds.to_parquet(out_dir / f"predictions_{arch}.parquet", index=False)
        print(f"     pooled: concordance={m['concordance']:.4f} auprc={m['auprc']:.4f} "
              f"brier={m['brier']:.4f} ece={m['ece']:.4f} "
              f"ece_cal={m['ece_calibrated']:.4f} p@100={m['precision_at_100']:.3f}", flush=True)

    pd.DataFrame(pooled_rows).to_csv(out_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out_dir / "origin_metrics.csv", index=False)
    pd.concat(calib_rows, ignore_index=True).to_csv(out_dir / "calibration.csv", index=False)
    summary = {
        "task": task, "run_timestamp": RUN_TS, "architectures": architectures,
        "hparams": {a: hparams.get(a, HP_DEFAULT.get(a)) for a in architectures},
        "feature_scenario": feature_scenario, "feature_scenario_description": FEATURE_SCENARIOS[feature_scenario],
        "feature_screen_before": feature_screen_before,
        "feature_screen_rule": "feature retained if coverage exceeds 10% before feature_screen_before",
        "n_person_periods": int(len(person)), "n_subjects": int(person["subject_id"].nunique()),
        "events": int(person["event"].sum()), "event_rate": float(person["event"].mean()),
        "n_features": len(feature_cols), "feature_md5": feature_meta["feature_md5"],
        "rolling_start": origins[0], "rolling_end": origins[-1],
        "n_origins": len(origins), "retrain_every": retrain_every, "tune_metric": TUNE_METRIC,
        "hparams_source": hparams_source,
        "hparams_reused_across_feature_scenario": bool(feature_scenario != "all" and hparams_source != "tuned_in_run"),
        "imputed_end_spells_dropped": int(person.attrs.get("imputed_end_spells_dropped", 0)),
        "imputed_end_rows_dropped": int(person.attrs.get("imputed_end_rows_dropped", 0)),
        "calibrators": calibrator_rows_all,
        "package_versions": package_versions(),
        "seed": SEED,
        "argv": sys.argv,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if do_shap:
        compute_shap(task, person, feature_cols, out_dir, hp=hparams.get("lgbm", HP_DEFAULT.get("lgbm")))
    print(f"  wrote {out_dir}", flush=True)
    return pooled_rows


def tune_hyperparameters(task, panel, architectures, val_start, val_end, tcfg,
                         feature_scenario="all", feature_screen_before=None):
    """Temporal carve-out tuning. Train on months < val_start, select each architecture's
    config by the prespecified validation metric on [val_start, val_end]. The test window
    must start after val_end."""
    add_time_index(panel, task)
    feature_cols = filter_features_for_scenario(
        select_features(panel, screen_before=feature_screen_before) + ["time_index"], feature_scenario
    )
    assert_feature_scenario_clean(feature_cols, feature_scenario)
    person = build_person_periods(panel, task)
    train = person[person["year_month"] < val_start]
    val = person[(person["year_month"] >= val_start) & (person["year_month"] <= val_end)]
    print(f"  [tune {task}] train<{val_start}: {len(train):,} rows ({int(train['event'].sum())} ev); "
          f"val {val_start}..{val_end}: {len(val):,} rows ({int(val['event'].sum())} ev); "
          f"metric={TUNE_METRIC}; scenario={feature_scenario}", flush=True)
    if int(val["event"].sum()) == 0 or int(train["event"].sum()) == 0:
        print(f"  [tune {task}] insufficient events, using defaults", flush=True)
        return {a: HP_DEFAULT[a] for a in architectures}
    fit_df, val_inner = recent_validation_split(train)
    yv = val["event"].to_numpy(np.int8)
    best = {}
    for arch in architectures:
        active_features = architecture_features(arch, feature_cols)
        fit_kind = "logistic" if arch == "logistic_time_only" else arch
        scored = []
        for hp in HP_SPACE[arch]:
            try:
                if fit_kind == "logistic":
                    p = predict_logistic(fit_logistic(train, active_features, hp=hp), val, active_features)
                elif fit_kind in ("lgbm", "lgbm_focal"):
                    m = fit_lgbm(fit_df, val_inner, active_features, focal=(fit_kind == "lgbm_focal"),
                                 hp=hp, final_train_df=train)
                    p = predict_lgbm(m, val, active_features)
                else:
                    tc = tconfig_from_hp(hp, tcfg)
                    p = predict_transformer(
                        fit_transformer(panel, fit_df, val_inner, active_features, tc, final_train_df=train),
                        panel, val, tc
                    )
                score = score_for_tuning(yv, p, TUNE_METRIC)
                auroc = float(roc_auc_score(yv, p)) if len(np.unique(yv)) == 2 else 0.0
                auprc = float(average_precision_score(yv, p)) if len(np.unique(yv)) == 2 else 0.0
                scored.append((score, hp))
                print(f"    [tune {task}/{arch}] {TUNE_METRIC}={score:.4f} auroc={auroc:.4f} auprc={auprc:.4f}  {hp}", flush=True)
            except Exception as exc:
                print(f"    [tune {task}/{arch}] FAILED ({exc})  {hp}", flush=True)
        scored.sort(key=lambda x: -x[0])
        best[arch] = scored[0][1] if scored else HP_DEFAULT[arch]
        if scored:
            print(f"  [tune {task}] BEST {arch}: {TUNE_METRIC}={scored[0][0]:.4f}  {best[arch]}", flush=True)
        else:
            print(f"  [tune {task}] BEST {arch}: default  {best[arch]}", flush=True)
    return best

def main():
    global RUN_TS
    ap = argparse.ArgumentParser(description="Discrete-time survival benchmarks for onset and offset")
    ap.add_argument("--task", choices=["onset", "offset", "both"], default="both")
    ap.add_argument("--architectures", default=",".join(ARCHITECTURES))
    ap.add_argument("--rolling-start", default=ROLLING_START)
    ap.add_argument("--rolling-end", default=None)
    ap.add_argument("--retrain-every", type=int, default=3)
    ap.add_argument("--output-root", type=Path, default=ANALYSIS)
    ap.add_argument("--hparams-root", type=Path, default=ANALYSIS,
                    help="Root containing survival_<task>/best_hparams.json when reusing tuned settings")
    ap.add_argument("--smoke", action="store_true", help="Fast subset run for verification")
    ap.add_argument("--max-groups", type=int, default=0, help="Limit groups (0 = all)")
    ap.add_argument("--transformer-epochs", type=int, default=12)
    ap.add_argument("--cpu-transformer", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--tune", action="store_true", help="Run temporal carve-out hyperparameter tuning before testing")
    ap.add_argument("--val-start", default="2022-01", help="First month of the tuning validation window")
    ap.add_argument("--val-end", default="2022-12", help="Last month of the tuning validation window")
    ap.add_argument("--feature-scenario", choices=sorted(FEATURE_SCENARIOS), default="all",
                    help="Prespecified feature-exclusion scenario for sensitivity analyses")
    args = ap.parse_args()

    RUN_TS = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    set_global_seeds()
    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    unknown_architectures = [a for a in architectures if a not in ARCHITECTURES]
    if unknown_architectures:
        raise ValueError(f"Unknown architecture(s): {', '.join(unknown_architectures)}")
    tcfg = TConfig(epochs=args.transformer_epochs, cpu=args.cpu_transformer)

    print(f"Loading panel ... ({RUN_TS})", flush=True)
    panel = load_panel()
    if args.smoke:
        args.max_groups = args.max_groups or 600
        if args.rolling_start == ROLLING_START:
            args.rolling_start = "2024-12"
        tcfg.epochs = min(tcfg.epochs, 2)
        args.retrain_every = max(args.retrain_every, 6)
    if args.max_groups:
        keep = pd.Index(sorted(panel["drug_group_key"].unique()))[: args.max_groups]
        panel = panel[panel["drug_group_key"].isin(keep)].reset_index(drop=True)
    panel = panel.reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=np.int64)  # positional id for transformer lookback
    print(f"  panel: {len(panel):,} group-months, {panel['drug_group_key'].nunique():,} groups, "
          f"{panel['year_month'].min()}..{panel['year_month'].max()}", flush=True)

    tasks = ["onset", "offset"] if args.task == "both" else [args.task]

    # When tuning, the test window must start strictly after the tuning validation window.
    if args.tune and pd.Period(args.rolling_start, freq="M") <= pd.Period(args.val_end, freq="M"):
        old_start = args.rolling_start
        args.rolling_start = (pd.Period(args.val_end, freq="M") + 1).strftime("%Y-%m")
        print(f"  tuning enabled: test window moved from {old_start} to {args.rolling_start} "
              f"(after val_end {args.val_end})", flush=True)

    feature_screen_before = args.rolling_start
    add_time_index(panel, "onset" if "onset" in tasks else tasks[0])
    descriptive_features = filter_features_for_scenario(
        select_features(panel, screen_before=feature_screen_before) + ["time_index"],
        args.feature_scenario,
    )
    assert_feature_scenario_clean(descriptive_features, args.feature_scenario)
    write_descriptives(panel, descriptive_features, args.output_root)
    if args.feature_scenario != "all" and not args.tune:
        print("  feature-scenario run is reusing tuned hyperparameters unless --tune is set; "
              "summary.json records the source", flush=True)

    hparams_by_task = {}
    hparams_source_by_task = {}
    for task in tasks:
        out_dir = args.output_root / f"survival_{task}"
        hp_file = out_dir / "best_hparams.json"
        reuse_hp_file = args.hparams_root / f"survival_{task}" / "best_hparams.json"
        if args.tune:
            best = tune_hyperparameters(task, panel, architectures, args.val_start, args.val_end, tcfg,
                                        feature_scenario=args.feature_scenario,
                                        feature_screen_before=feature_screen_before)
            out_dir.mkdir(parents=True, exist_ok=True)
            hp_file.write_text(json.dumps(best, indent=2))
            meta = {
                "run_timestamp": RUN_TS,
                "task": task,
                "architectures": architectures,
                "val_start": args.val_start,
                "val_end": args.val_end,
                "tune_metric": TUNE_METRIC,
                "feature_scenario": args.feature_scenario,
                "feature_screen_before": feature_screen_before,
                "feature_screen_rule": "feature retained if coverage exceeds 10% before feature_screen_before",
                "package_versions": package_versions(),
                "seed": SEED,
                "argv": sys.argv,
                "panel_rows": int(len(panel)),
                "panel_min_month": str(panel["year_month"].min()),
                "panel_max_month": str(panel["year_month"].max()),
            }
            (out_dir / "best_hparams_meta.json").write_text(json.dumps(meta, indent=2))
            hparams_by_task[task] = best
            hparams_source_by_task[task] = "tuned_in_run"
        elif reuse_hp_file.exists() and reuse_hp_file.resolve() != hp_file.resolve():
            hparams_by_task[task] = json.loads(reuse_hp_file.read_text())
            print(f"  loaded tuned hparams for {task} from {reuse_hp_file}", flush=True)
            hparams_source_by_task[task] = str(reuse_hp_file)
        elif hp_file.exists():
            hparams_by_task[task] = json.loads(hp_file.read_text())
            print(f"  loaded tuned hparams for {task} from {hp_file}", flush=True)
            hparams_source_by_task[task] = str(hp_file)
        elif reuse_hp_file.exists():
            hparams_by_task[task] = json.loads(reuse_hp_file.read_text())
            print(f"  loaded tuned hparams for {task} from {reuse_hp_file}", flush=True)
            hparams_source_by_task[task] = str(reuse_hp_file)
        else:
            hparams_source_by_task[task] = "defaults"

    for task in tasks:
        run_task(task, panel, architectures, args.rolling_start, args.rolling_end,
                 args.retrain_every, tcfg, args.output_root, do_shap=not args.skip_shap,
                 hparams=hparams_by_task.get(task), feature_scenario=args.feature_scenario,
                 feature_screen_before=feature_screen_before,
                 hparams_source=hparams_source_by_task.get(task, "defaults"))


if __name__ == "__main__":
    RUN_TS = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    main()
