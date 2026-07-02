"""27_supporting_analyses.py - Supporting rigor for the survival benchmarks.

Two robustness analyses that read the artifacts written by 25_survival_benchmarks.py:

  1. Subject-level Harrell C-index at a landmark origin, as a survival-standard complement to the
     per-period concordance reported in the main tables. For each subject at risk at the landmark,
     the predicted hazard is compared against the observed time to the event (or right-censoring),
     and concordance is the fraction of comparable pairs correctly ordered.
  2. Permutation importance for the LightGBM hazard model, as a cardinality-robust cross-read of the
     TreeSHAP ranking (TreeSHAP inflates high-cardinality categoricals; permutation importance does
     not). For each feature, the drop in out-of-time AUROC when that feature is shuffled is reported.

Run: python Programs/27_supporting_analyses.py [--task both] [--landmark 2023-01] [--test-start 2023-01]
Outputs: Data/analysis/survival_<task>/harrell_c.csv and permutation_importance.csv
"""
import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("survival_benchmarks", HERE / "25_survival_benchmarks.py")
S = module_from_spec(_spec)
sys.modules[_spec.name] = S  # register so dataclass decorators in the module resolve correctly
_spec.loader.exec_module(S)
ANALYSIS = S.ANALYSIS
ARCHS = ["logistic", "logistic_time_only", "lgbm", "lgbm_focal", "transformer"]
SEED = 42


def _period_diff(a, b):
    return (pd.Period(a, freq="M") - pd.Period(b, freq="M")).n


def harrell_c(risk, time, event, max_n=8000):
    """Harrell concordance between predicted risk and observed time-to-event with right-censoring."""
    risk = np.asarray(risk, float); time = np.asarray(time, float); event = np.asarray(event, int)
    n = len(risk)
    if n > max_n:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(n, max_n, replace=False)
        risk, time, event = risk[keep], time[keep], event[keep]
    ti = time[:, None]; tj = time[None, :]
    ei = event[:, None]
    comparable = (ei == 1) & (ti < tj)
    ncomp = comparable.sum()
    if ncomp == 0:
        return np.nan
    ri = risk[:, None]; rj = risk[None, :]
    concordant = (comparable & (ri > rj)).sum()
    tied = (comparable & (ri == rj)).sum()
    return float((concordant + 0.5 * tied) / ncomp)


def landmark_harrell(task, landmark):
    """Per-architecture landmark Harrell C from the saved predictions."""
    rows = []
    for arch in ARCHS:
        f = ANALYSIS / f"survival_{task}" / f"predictions_{arch}.parquet"
        if not f.exists():
            continue
        pred = pd.read_parquet(f)
        at_lm = pred[pred["origin"] == landmark]
        if at_lm.empty:
            landmark = sorted(pred["origin"].unique())[0]
            at_lm = pred[pred["origin"] == landmark]
        risk = at_lm.drop_duplicates("subject_id").set_index("subject_id")["pred"]
        later = pred[pred["subject_id"].isin(risk.index)].copy()
        later = later[later["origin"] >= landmark]
        recs = []
        for sid, g in later.groupby("subject_id"):
            ev = g[g["event"] == 1]
            if len(ev):
                t = _period_diff(ev["origin"].min(), landmark); e = 1
            else:
                t = _period_diff(g["origin"].max(), landmark) + 1; e = 0
            recs.append((risk[sid], max(t, 0), e))
        if not recs:
            continue
        r, t, e = map(np.array, zip(*recs))
        rows.append({"task": task, "architecture": arch, "landmark": landmark,
                     "n_subjects": len(r), "n_events": int(e.sum()), "harrell_c": harrell_c(r, t, e)})
    return pd.DataFrame(rows)


def permutation_importance(task, panel, hp, test_start, top_k=40):
    """LightGBM permutation importance on the out-of-time holdout, as a SHAP cross-read."""
    S.add_time_index(panel, task)
    feats = S.select_features(panel, screen_before=test_start) + ["time_index"]
    person = S.build_person_periods(panel, task)
    train = person[person["year_month"] < test_start]
    hold = person[person["year_month"] >= test_start]
    if int(train["event"].sum()) == 0 or int(hold["event"].sum()) == 0:
        return pd.DataFrame()
    # Evaluate importance on a holdout sample (keep all events) so 421 permuted predictions stay fast.
    if len(hold) > 50000:
        pos = hold[hold["event"] == 1]
        neg = hold[hold["event"] == 0].sample(50000 - len(pos), random_state=SEED)
        hold = pd.concat([pos, neg]).reset_index(drop=True)
    model = S.fit_lgbm(train, None, feats, hp=hp or S.HP_DEFAULT["lgbm"])  # fixed rounds, full-data model
    yh = hold["event"].to_numpy(np.int8)
    base = roc_auc_score(yh, S.predict_lgbm(model, hold, feats))
    rng = np.random.default_rng(SEED)
    Xh = hold[feats].copy()
    rows = []
    for c in feats:
        saved = Xh[c].copy()
        Xh[c] = pd.Series(rng.permutation(saved.to_numpy()), index=Xh.index).astype(saved.dtype)
        auc = roc_auc_score(yh, S.predict_lgbm(model, Xh, feats))
        Xh[c] = saved
        rows.append({"feature": c, "family": S.feature_family(c), "auroc_drop": float(base - auc)})
    out = pd.DataFrame(rows).sort_values("auroc_drop", ascending=False).reset_index(drop=True)
    out.attrs["base_auroc"] = base
    return out


def monthly_budget_metrics(preds, score_col="pred", event_col="label_6m", k=100):
    reviewed = 0
    flagged_events = 0
    total_events = int(preds[event_col].sum())
    for _, sub in preds.groupby("origin"):
        ranked = sub.sort_values(score_col, ascending=False).head(min(k, len(sub)))
        reviewed += len(ranked)
        flagged_events += int(ranked[event_col].sum())
    return {
        "reviewed": int(reviewed),
        "events_flagged": int(flagged_events),
        "total_events": int(total_events),
        "precision": float(flagged_events / reviewed) if reviewed else np.nan,
        "recall": float(flagged_events / total_events) if total_events else np.nan,
    }


def horizon6_onset_metrics(horizon=6):
    """Score saved one-month onset hazards against first onset within the next six months."""
    out_dir = ANALYSIS / "survival_onset"
    panel_path = ANALYSIS / "survival_grouped_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Grouped survival panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path, columns=["drug_group_key", "year_month", "shortage_start"])
    panel = panel.copy()
    panel["period_ord"] = pd.PeriodIndex(panel["year_month"].astype(str), freq="M").astype("int64")
    max_panel_ord = int(panel["period_ord"].max())
    first_onset = (
        panel[panel["shortage_start"].fillna(0).astype(float) > 0]
        .groupby("drug_group_key")["period_ord"]
        .min()
    )

    rows = []
    for arch in ARCHS:
        pred_path = out_dir / f"predictions_{arch}.parquet"
        if not pred_path.exists():
            continue
        preds = pd.read_parquet(pred_path)
        preds = preds.copy()
        preds["origin_ord"] = pd.PeriodIndex(preds["origin"].astype(str), freq="M").astype("int64")
        complete_origin = preds["origin_ord"] + horizon <= max_panel_ord
        dropped_rows = int((~complete_origin).sum())
        dropped_origins = int(preds.loc[~complete_origin, "origin"].nunique())
        eval_df = preds.loc[complete_origin].copy()
        if eval_df.empty:
            continue
        onset_ord = eval_df["drug_group_key"].map(first_onset)
        eval_df["label_6m"] = (
            onset_ord.notna()
            & (onset_ord.to_numpy(dtype=float) > eval_df["origin_ord"].to_numpy(dtype=float))
            & (onset_ord.to_numpy(dtype=float) <= (eval_df["origin_ord"].to_numpy(dtype=float) + horizon))
        ).astype(np.int8)
        y = eval_df["label_6m"].to_numpy(np.int8)
        p = eval_df["pred"].to_numpy(np.float64)
        prevalence = float(y.mean()) if len(y) else np.nan
        auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        auprc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        row = {
            "architecture": arch,
            "score_column": "pred",
            "outcome": f"first shortage onset within {horizon} months",
            "note": "One-month onset hazard re-evaluated at a 6-month horizon, no model was trained or tuned for this horizon.",
            "evaluation_start": str(eval_df["origin"].min()),
            "evaluation_end": str(eval_df["origin"].max()),
            "origins": int(eval_df["origin"].nunique()),
            "dropped_incomplete_origin_rows": dropped_rows,
            "dropped_incomplete_origins": dropped_origins,
            "n": int(len(eval_df)),
            "events": int(y.sum()),
            "prevalence": prevalence,
            "auroc": auroc,
            "auprc": auprc,
        }
        for k in [25, 50, 100]:
            op = monthly_budget_metrics(eval_df, k=k)
            row[f"precision_at_{k}_monthly"] = op["precision"]
            row[f"recall_at_{k}_monthly"] = op["recall"]
            row[f"reviewed_at_{k}_monthly"] = op["reviewed"]
            row[f"events_flagged_at_{k}_monthly"] = op["events_flagged"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "horizon6_metrics.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Supporting rigor: Harrell C and permutation importance")
    ap.add_argument("--task", choices=["onset", "offset", "both"], default="both")
    ap.add_argument("--landmark", default="2023-01")
    ap.add_argument("--test-start", default="2023-01")
    args = ap.parse_args()

    print("Loading panel ...", flush=True)
    panel = S.load_panel()
    panel = panel.reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=np.int64)

    for task in (["onset", "offset"] if args.task == "both" else [args.task]):
        out_dir = ANALYSIS / f"survival_{task}"
        hp_file = out_dir / "best_hparams.json"
        hp = json.loads(hp_file.read_text()).get("lgbm") if hp_file.exists() else None

        hc = landmark_harrell(task, args.landmark)
        if not hc.empty:
            hc.to_csv(out_dir / "harrell_c.csv", index=False)
            print(f"\n=== {task} landmark Harrell C (landmark {args.landmark}) ===")
            print(hc[["architecture", "n_subjects", "n_events", "harrell_c"]].to_string(index=False), flush=True)

        pi = permutation_importance(task, panel, hp, args.test_start)
        if not pi.empty:
            pi.to_csv(out_dir / "permutation_importance.csv", index=False)
            print(f"\n=== {task} permutation importance top 12 (base AUROC {pi.attrs['base_auroc']:.4f}) ===")
            print(pi.head(12).to_string(index=False), flush=True)

        if task == "onset":
            h6 = horizon6_onset_metrics()
            if not h6.empty:
                print("\n=== onset 6-month horizon evaluation of saved one-month scores ===")
                show_cols = [
                    "architecture", "n", "events", "prevalence", "auroc", "auprc",
                    "precision_at_100_monthly", "recall_at_100_monthly",
                    "dropped_incomplete_origins",
                ]
                print(h6[show_cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
