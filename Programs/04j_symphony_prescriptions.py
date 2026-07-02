"""
Aggregate manually exported Symphony Metys prescription query files.

Inputs:
  Raw Data/Symphony/*.zip, each containing one Report Builder CSV.

Outputs:
  Data/intermediate/symphony_prescriptions_ndc_month_all.parquet
  Data/intermediate/symphony_prescriptions_ndc_month_all.csv.gz
  Data/intermediate/symphony_prescriptions_ndc_month_study.parquet
  Data/intermediate/symphony_prescriptions_ndc_month_study.csv.gz
  Data/intermediate/symphony_prescriptions_file_manifest.csv
  Data/intermediate/symphony_prescriptions_monthly_coverage.csv
"""

import re
import json
import zipfile
from pathlib import Path

import pandas as pd

from importlib.machinery import SourceFileLoader

util = SourceFileLoader("util", str(Path(__file__).with_name("00_utilities.py"))).load_module()

RAW_DIR = util.RAW_DATA / "Symphony"
OUT_ALL = util.INTERMEDIATE / "symphony_prescriptions_ndc_month_all.parquet"
OUT_ALL_CSV = util.INTERMEDIATE / "symphony_prescriptions_ndc_month_all.csv.gz"
OUT_STUDY = util.INTERMEDIATE / "symphony_prescriptions_ndc_month_study.parquet"
OUT_STUDY_CSV = util.INTERMEDIATE / "symphony_prescriptions_ndc_month_study.csv.gz"
OUT_MANIFEST = util.INTERMEDIATE / "symphony_prescriptions_file_manifest.csv"
OUT_COVERAGE = util.INTERMEDIATE / "symphony_prescriptions_monthly_coverage.csv"
OUT_VALIDATION = util.INTERMEDIATE / "symphony_prescriptions_validation.json"

EXCLUDED_SOURCE_COLUMNS = {
    "avg_days_supply",  # Not present in the 2022 Symphony exports; exclude rather than imputing.
}


def snake(name):
    name = str(name).strip()
    if not name:
        return ""
    name = name.replace("/", " per ")
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name


def read_export(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"{zip_path.name}: expected one CSV member, found {members}")
        member = members[0]
        with zf.open(member) as fh:
            df = pd.read_csv(fh, dtype={"NDC 11": "string"}, low_memory=False)

    df = df.loc[:, [c for c in df.columns if str(c).strip()]]
    df = df.rename(columns={c: snake(c) for c in df.columns})
    df = df.rename(columns={"month": "year_month", "ndc_11": "symphony_ndc_11"})
    if "year_month" not in df.columns or "symphony_ndc_11" not in df.columns:
        raise ValueError(f"{zip_path.name}: missing required Month or NDC 11 columns")

    df["year_month"] = df["year_month"].astype("string").str.strip()
    df["symphony_ndc_11"] = df["symphony_ndc_11"].astype("string").str.replace(r"\D", "", regex=True).str.zfill(11)
    df["ndc_11"] = df["symphony_ndc_11"].map(util.format_ndcpackagecode)
    df["source_zip"] = zip_path.name
    df["source_member"] = member
    df = df.drop(columns=[c for c in EXCLUDED_SOURCE_COLUMNS if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed_")])

    id_cols = {"year_month", "symphony_ndc_11", "ndc_11", "source_zip", "source_member"}
    for col in df.columns:
        if col not in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No Symphony ZIP files found in {RAW_DIR}")

    frames = []
    manifest = []
    for z in zips:
        df = read_export(z)
        months = sorted(df["year_month"].dropna().unique().tolist())
        dup = int(df.duplicated(["year_month", "symphony_ndc_11"]).sum())
        manifest.append(
            {
                "source_zip": z.name,
                "rows": len(df),
                "unique_ndc_months": int(df[["year_month", "symphony_ndc_11"]].drop_duplicates().shape[0]),
                "duplicate_ndc_months": dup,
                "first_month": months[0] if months else None,
                "last_month": months[-1] if months else None,
                "n_months": len(months),
                "months": ",".join(months),
            }
        )
        frames.append(df)
        print(f"{z.name}: {len(df):,} rows, {months[0]}..{months[-1]}, dup={dup}")

    all_df = pd.concat(frames, ignore_index=True)
    all_dup = int(all_df.duplicated(["year_month", "symphony_ndc_11"]).sum())
    if all_dup:
        raise ValueError(f"Combined Symphony data has {all_dup:,} duplicate NDC-month rows")

    all_df["year_month"] = all_df["year_month"].astype("string")
    all_df["symphony_ndc_11"] = all_df["symphony_ndc_11"].astype("string")
    all_df["ndc_11"] = all_df["ndc_11"].astype("string")
    all_df = all_df.sort_values(["year_month", "symphony_ndc_11"]).reset_index(drop=True)

    study_df = all_df[(all_df["year_month"] >= util.STUDY_START) & (all_df["year_month"] <= util.STUDY_END)].copy()
    coverage = (
        all_df.groupby("year_month", as_index=False)
        .agg(
            rows=("symphony_ndc_11", "size"),
            unique_ndcs=("symphony_ndc_11", "nunique"),
            trx_count=("trx_count", "sum"),
            nrx_count=("nrx_count", "sum"),
            rrx_count=("rrx_count", "sum"),
            trx_quantity=("trx_quantity", "sum"),
            trx_dollars=("trx_dollars", "sum"),
            trx_mbs_dollars=("trx_mbs_dollars", "sum"),
        )
        .sort_values("year_month")
    )

    pd.DataFrame(manifest).to_csv(OUT_MANIFEST, index=False)
    coverage.to_csv(OUT_COVERAGE, index=False)
    all_df.to_parquet(OUT_ALL, index=False)
    study_df.to_parquet(OUT_STUDY, index=False)
    all_df.to_csv(OUT_ALL_CSV, index=False, compression="gzip")
    study_df.to_csv(OUT_STUDY_CSV, index=False, compression="gzip")

    panel_path = util.ANALYSIS / "drug_shortage_panel.parquet"
    panel_overlap = {}
    if panel_path.exists():
        panel = pd.read_parquet(panel_path, columns=["ndc_11"])
        panel_ndcs = set(panel["ndc_11"].dropna().astype(str).unique())
        study_ndcs = set(study_df["ndc_11"].dropna().astype(str).unique())
        panel_overlap = {
            "panel_unique_ndcs": len(panel_ndcs),
            "study_symphony_ndcs_in_panel": len(study_ndcs & panel_ndcs),
            "study_symphony_ndcs_not_in_panel": len(study_ndcs - panel_ndcs),
            "panel_ndcs_without_symphony_study_rows": len(panel_ndcs - study_ndcs),
        }

    expected_all_months = pd.period_range("2020-01", "2025-12", freq="M").astype(str).tolist()
    expected_study_months = pd.period_range(util.STUDY_START, util.STUDY_END, freq="M").astype(str).tolist()
    validation = {
        "all_rows": len(all_df),
        "all_first_month": str(all_df["year_month"].min()),
        "all_last_month": str(all_df["year_month"].max()),
        "all_n_months": int(all_df["year_month"].nunique()),
        "all_missing_months": sorted(set(expected_all_months) - set(all_df["year_month"].astype(str).unique())),
        "all_duplicate_ndc_months": int(all_df.duplicated(["year_month", "symphony_ndc_11"]).sum()),
        "study_rows": len(study_df),
        "study_first_month": str(study_df["year_month"].min()),
        "study_last_month": str(study_df["year_month"].max()),
        "study_n_months": int(study_df["year_month"].nunique()),
        "study_missing_months": sorted(set(expected_study_months) - set(study_df["year_month"].astype(str).unique())),
        "study_duplicate_ndc_months": int(study_df.duplicated(["year_month", "symphony_ndc_11"]).sum()),
        "study_unique_symphony_ndcs": int(study_df["symphony_ndc_11"].nunique()),
        "study_unique_dashed_ndcs": int(study_df["ndc_11"].nunique()),
        "bad_symphony_ndc_length_rows": int((study_df["symphony_ndc_11"].astype(str).str.len() != 11).sum()),
        "bad_dashed_ndc_format_rows": int((~study_df["ndc_11"].astype(str).str.match(r"^\d{5}-\d{4}-\d{2}$", na=False)).sum()),
        "study_key_metric_missing_rows": {
            col: int(study_df[col].isna().sum())
            for col in ["trx_count", "nrx_count", "rrx_count", "trx_quantity", "trx_dollars", "dacon"]
            if col in study_df.columns
        },
        "excluded_source_columns": sorted(EXCLUDED_SOURCE_COLUMNS),
        "available_columns": sorted(study_df.columns.tolist()),
        "panel_overlap": panel_overlap,
        "output_sizes_bytes": {
            p.name: p.stat().st_size
            for p in [OUT_ALL, OUT_ALL_CSV, OUT_STUDY, OUT_STUDY_CSV, OUT_MANIFEST, OUT_COVERAGE]
            if p.exists()
        },
    }
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print("\nWrote:")
    for path in [OUT_ALL, OUT_ALL_CSV, OUT_STUDY, OUT_STUDY_CSV, OUT_MANIFEST, OUT_COVERAGE, OUT_VALIDATION]:
        print(f"  {path}")
    print(f"\nAll rows: {len(all_df):,}; months {all_df['year_month'].min()}..{all_df['year_month'].max()}; NDCs {all_df['symphony_ndc_11'].nunique():,}")
    print(f"Study rows: {len(study_df):,}; months {study_df['year_month'].min()}..{study_df['year_month'].max()}; NDCs {study_df['symphony_ndc_11'].nunique():,}")


if __name__ == "__main__":
    main()
