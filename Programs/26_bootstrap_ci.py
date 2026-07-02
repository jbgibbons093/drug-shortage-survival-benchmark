"""26_bootstrap_ci.py - Cluster bootstrap confidence intervals for the survival benchmarks.

Reads the per-architecture out-of-time predictions written by 25_survival_benchmarks.py and
produces percentile confidence intervals for each metric per architecture, plus intervals on the
pairwise difference versus a reference architecture (default logistic). Resampling is clustered by
drug group, because person-months are statistically dependent within a group; a naive row bootstrap
would understate uncertainty.

Run: python Programs/26_bootstrap_ci.py [--task both] [--n-boot 1000] [--reference logistic]
Outputs: Data/analysis/survival_<task>/bootstrap_ci.csv
"""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("utilities", HERE / "00_utilities.py")
_util = module_from_spec(_spec)
_spec.loader.exec_module(_util)
ANALYSIS = _util.ANALYSIS

ARCHS = ["logistic", "logistic_time_only", "lgbm", "lgbm_focal", "transformer"]
METRICS = ["concordance", "auprc", "brier", "ece", "precision_at_100_monthly"]
KEYS = ["drug_group_key", "subject_id", "year_month", "origin"]
SEED = 42


def _compute_ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return np.nan
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


def _precision_at_100_monthly(y, p, origins, k=100):
    reviewed = 0
    flagged_events = 0
    for origin in pd.unique(origins):
        idx = np.where(origins == origin)[0]
        if len(idx) == 0:
            continue
        ranked = idx[np.argsort(-p[idx])[:min(k, len(idx))]]
        reviewed += len(ranked)
        flagged_events += int(y[ranked].sum())
    return float(flagged_events / reviewed) if reviewed else np.nan


def _metrics(y, p, origins):
    out = {
        "concordance": np.nan,
        "auprc": np.nan,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else np.nan,
        "ece": _compute_ece(y, p),
        "precision_at_100_monthly": _precision_at_100_monthly(y, p, origins),
    }
    if len(np.unique(y)) == 2:
        out["concordance"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    return out


def load_aligned(task):
    """Inner-join all architectures' predictions on the shared test rows."""
    base, archs = None, []
    for a in ARCHS:
        f = ANALYSIS / f"survival_{task}" / f"predictions_{a}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)[KEYS + ["event", "pred"]].rename(columns={"pred": f"pred_{a}"})
        archs.append(a)
        base = d if base is None else base.merge(d.drop(columns=["event"]), on=KEYS, how="inner")
    return base, archs


def main():
    ap = argparse.ArgumentParser(description="Cluster bootstrap CIs for survival benchmarks")
    ap.add_argument("--task", choices=["onset", "offset", "both"], default="both")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--reference", default="logistic")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    for task in (["onset", "offset"] if args.task == "both" else [args.task]):
        wide, archs = load_aligned(task)
        if wide is None:
            print(f"{task}: no predictions found, skipping")
            continue
        y = wide["event"].to_numpy(np.int8)
        preds = {a: wide[f"pred_{a}"].to_numpy(np.float64) for a in archs}
        origins = wide["origin"].to_numpy()
        groups = wide["drug_group_key"].to_numpy()
        uniq = pd.unique(groups)
        gidx = {g: np.where(groups == g)[0] for g in uniq}
        ref = args.reference if args.reference in archs else archs[0]
        print(f"{task}: {len(wide):,} aligned rows, {len(uniq):,} groups, {int(y.sum())} events, archs={archs}", flush=True)

        boot = {a: {m: [] for m in METRICS} for a in archs}
        dboot = {a: {m: [] for m in METRICS} for a in archs if a != ref}
        for b in range(args.n_boot):
            samp = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([gidx[g] for g in samp])
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            ob = origins[idx]
            mm = {a: _metrics(yb, preds[a][idx], ob) for a in archs}
            for a in archs:
                for m in METRICS:
                    boot[a][m].append(mm[a][m])
            for a in dboot:
                for m in METRICS:
                    dboot[a][m].append(mm[a][m] - mm[ref][m])

        point = {a: _metrics(y, preds[a], origins) for a in archs}
        rows = []
        for a in archs:
            for m in METRICS:
                arr = np.array(boot[a][m], dtype=float)
                row = {"task": task, "architecture": a, "metric": m,
                       "point": point[a][m],
                       "ci_lo": float(np.nanpercentile(arr, 2.5)),
                       "ci_hi": float(np.nanpercentile(arr, 97.5)),
                       "reference": ref}
                if a != ref:
                    d = np.array(dboot[a][m], dtype=float)
                    row["diff_vs_ref"] = point[a][m] - point[ref][m]
                    row["diff_ci_lo"] = float(np.nanpercentile(d, 2.5))
                    row["diff_ci_hi"] = float(np.nanpercentile(d, 97.5))
                    row["diff_excludes_zero"] = bool(row["diff_ci_lo"] > 0 or row["diff_ci_hi"] < 0)
                rows.append(row)
        out = pd.DataFrame(rows)
        out.to_csv(ANALYSIS / f"survival_{task}" / "bootstrap_ci.csv", index=False)
        show = out[out["metric"] == "concordance"]
        cols = [c for c in ["architecture", "point", "ci_lo", "ci_hi", "diff_vs_ref", "diff_ci_lo", "diff_ci_hi", "diff_excludes_zero"] if c in show.columns]
        print(f"=== {task} concordance ({args.n_boot} cluster-bootstrap resamples, ref={ref}) ===")
        print(show[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
