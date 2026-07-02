"""
04i_asp_pricing.py - CMS Part B ASP pricing features at NDC x month resolution.

Reads quarterly ASP pricing files and NDC-HCPCS crosswalks from
`Raw Data/CMS ASP/`, joins them per quarter, and expands to monthly
NDC-level features that complement NADAC.

ASP (Average Sales Price) is the Medicare Part B payment basis for
physician-administered drugs. NADAC alone misses most IV / infused agents
(oncology, biologics, vaccines), so ASP fills a real coverage gap for
those products.

Output: `Data/intermediate/asp_pricing.parquet` with columns:
  ndc_11, year_month
  asp_payment_limit          USD per HCPCS billing unit
  asp_billunits_per_pkg      Billing units per NDC package
  asp_payment_per_pkg        Payment limit x billing units (USD per package)
  asp_pkg_size               Package size (e.g., 1 ML)
  asp_quarter                Source quarter (e.g., "2025-Q3")
  has_asp_data               1 if ASP price found for this NDCxmonth, else 0
  asp_is_observed            1 if from a quarter with both pricing+crosswalk; 0 if forward-filled
"""

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "Raw Data"
INTERMEDIATE = ROOT / "Data" / "intermediate"
ASP_DIR = RAW / "CMS ASP"

STUDY_START = "2020-01"
STUDY_END = "2025-09"

QUARTER_MONTHS = {
    "january": (1, 2, 3),
    "april": (4, 5, 6),
    "july": (7, 8, 9),
    "october": (10, 11, 12),
}
QUARTER_IDX = {"january": 1, "april": 2, "july": 3, "october": 4}


def parse_quarter_from_filename(name: str):
    """Return (year, quarter_idx, months_tuple) for a CMS ASP zip filename, else None."""
    m = re.match(r"(january|april|july|october)-(\d{4})-", name.lower())
    if not m:
        return None
    quarter = m.group(1)
    year = int(m.group(2))
    months = QUARTER_MONTHS[quarter]
    return year, quarter, months


def find_section508_csv(zip_path: Path, must_contain: str = "") -> str | None:
    """Return the section-508 CSV inside a zip whose name contains `must_contain`."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.lower().endswith(".csv") and "section 508" in n.lower()]
        if must_contain:
            mc = must_contain.lower()
            filtered = [n for n in candidates if mc in n.lower()]
            if filtered:
                return filtered[0]
        return candidates[0] if candidates else None


def load_pricing_file(zip_path: Path) -> pd.DataFrame:
    """Read an ASP pricing zip → DataFrame [HCPCS Code, Payment Limit, ...]."""
    csv_name = find_section508_csv(zip_path, must_contain="pricing")
    if csv_name is None:
        return pd.DataFrame()
    with zipfile.ZipFile(zip_path) as zf, zf.open(csv_name) as f:
        raw = f.read()
    # The header line is the first one starting with "HCPCS"
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.upper().startswith("HCPCS CODE,")),
        None,
    )
    if header_idx is None:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "HCPCS Code": "hcpcs",
        "Short Description": "hcpcs_desc",
        "HCPCS Code Dosage": "hcpcs_dosage",
        "Payment Limit": "asp_payment_limit",
    })
    df["asp_payment_limit"] = pd.to_numeric(df["asp_payment_limit"], errors="coerce")
    df = df.dropna(subset=["hcpcs", "asp_payment_limit"])
    df = df[["hcpcs", "hcpcs_desc", "hcpcs_dosage", "asp_payment_limit"]]
    return df


def load_crosswalk_file(zip_path: Path) -> pd.DataFrame:
    """Read an ASP NDC-HCPCS crosswalk zip → DataFrame [hcpcs, ndc_11, BILLUNITS, ...]."""
    csv_name = find_section508_csv(zip_path, must_contain="asp ndc-hcpcs")
    if csv_name is None:
        # Some 2024+ Jan files name the section-508 CSV without the "asp" prefix
        csv_name = find_section508_csv(zip_path, must_contain="ndc-hcpcs")
    if csv_name is None:
        return pd.DataFrame()
    with zipfile.ZipFile(zip_path) as zf, zf.open(csv_name) as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # Header line ends with the year-coded HCPCS column name like "_2025_CODE"
    header_idx = next(
        (i for i, line in enumerate(lines) if re.match(r'_?\d{4}_?CODE,', line, re.IGNORECASE)),
        None,
    )
    if header_idx is None:
        # Fallback: header line that mentions BILLUNITS
        header_idx = next(
            (i for i, line in enumerate(lines) if "BILLUNITS" in line.upper()),
            None,
        )
    if header_idx is None:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # First column is the HCPCS code with a year-coded name. Rename to "hcpcs".
    code_col = next((c for c in df.columns if re.match(r'_?\d{4}_?CODE', c, re.IGNORECASE)), None)
    if code_col is None:
        # Some files just use 'HCPCS' or similar
        code_col = next((c for c in df.columns if "code" in c.lower() and "hcpcs" not in c.lower()), df.columns[0])
    rename = {code_col: "hcpcs"}
    if "NDC2" in df.columns:
        rename["NDC2"] = "ndc_dashed"
    elif "NDC" in df.columns:
        rename["NDC"] = "ndc_dashed"
    if "PKG SIZE" in df.columns:
        rename["PKG SIZE"] = "pkg_size"
    if "PKG QTY" in df.columns:
        rename["PKG QTY"] = "pkg_qty"
    if "BILLUNITS" in df.columns:
        rename["BILLUNITS"] = "billunits"
    if "BILLUNITSPKG" in df.columns:
        rename["BILLUNITSPKG"] = "billunits_per_pkg"
    if "Drug Name" in df.columns:
        rename["Drug Name"] = "drug_name"
    if "LABELER NAME" in df.columns:
        rename["LABELER NAME"] = "labeler_name"
    df = df.rename(columns=rename)

    if "ndc_dashed" not in df.columns:
        return pd.DataFrame()
    # Panel uses the 13-char dashed NDC format ("12345-6789-01"). Keep that
    # format here so the merge in 08_assemble_panel.py joins cleanly.
    df["ndc_11"] = df["ndc_dashed"].astype(str).str.strip()
    # Filter to valid 13-char dashed NDCs only (some rows have malformed ids).
    df = df[df["ndc_11"].str.fullmatch(r"\d{5}-\d{4}-\d{2}")]

    keep = ["hcpcs", "ndc_11", "drug_name", "labeler_name",
            "pkg_size", "pkg_qty", "billunits", "billunits_per_pkg"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    for c in ["pkg_qty", "billunits", "billunits_per_pkg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> int:
    print("=" * 70)
    print("04i_asp_pricing.py - Building ASP pricing features")
    print("=" * 70)

    if not ASP_DIR.exists():
        print(f"  No ASP directory at {ASP_DIR}; nothing to process.")
        return 0

    # Index zips by (year, quarter)
    pricing_zips: dict[tuple[int, str], Path] = {}
    crosswalk_zips: dict[tuple[int, str], Path] = {}
    for f in sorted(ASP_DIR.glob("*.zip")):
        info = parse_quarter_from_filename(f.name)
        if info is None:
            continue
        year, quarter, _ = info
        if "pricing-file" in f.name.lower():
            pricing_zips[(year, quarter)] = f
        elif "crosswalk" in f.name.lower():
            crosswalk_zips[(year, quarter)] = f
    print(f"  Pricing files: {len(pricing_zips)}; crosswalks: {len(crosswalk_zips)}")

    # Build per-quarter NDCxASP records, sorted CHRONOLOGICALLY (not alphabetically
    # by quarter name - important for the forward-fill logic to use the most
    # recent prior crosswalk when one is missing).
    quarter_keys = sorted(
        set(pricing_zips.keys()) | set(crosswalk_zips.keys()),
        key=lambda k: (k[0], QUARTER_IDX[k[1]]),
    )
    quarter_frames = []

    # Cache last-seen crosswalk to forward-fill quarters with missing crosswalks
    last_crosswalk: pd.DataFrame | None = None
    last_crosswalk_label = None

    for key in quarter_keys:
        year, quarter = key
        months = QUARTER_MONTHS[quarter]
        label = f"{year}-{quarter}"

        cw_zip = crosswalk_zips.get(key)
        if cw_zip is not None:
            cw = load_crosswalk_file(cw_zip)
            if len(cw) > 0:
                last_crosswalk = cw
                last_crosswalk_label = label
                cw_source = "observed"
            else:
                cw = last_crosswalk
                cw_source = f"forward-fill from {last_crosswalk_label}"
        else:
            cw = last_crosswalk
            cw_source = f"forward-fill from {last_crosswalk_label}"

        if cw is None or len(cw) == 0:
            print(f"    {label}: no crosswalk available yet, skipping")
            continue

        pricing_zip = pricing_zips.get(key)
        if pricing_zip is None:
            # Pricing missing for this quarter - skip (could try to fill, but
            # ASP prices are quarter-specific so forward-fill is risky)
            print(f"    {label}: pricing file missing, skipping")
            continue
        pricing = load_pricing_file(pricing_zip)
        if len(pricing) == 0:
            print(f"    {label}: pricing file empty, skipping")
            continue

        joined = cw.merge(pricing, on="hcpcs", how="inner")
        # Compute payment per package = payment_limit x billing units per pkg
        if "billunits_per_pkg" in joined.columns:
            joined["asp_payment_per_pkg"] = joined["asp_payment_limit"] * joined["billunits_per_pkg"]
        else:
            joined["asp_payment_per_pkg"] = joined["asp_payment_limit"]

        # Pick the smallest payment_per_pkg per NDC if multiple HCPCS rows
        # (some NDCs map to several HCPCS codes; smallest matches the typical
        # billing case)
        joined = joined.sort_values(["ndc_11", "asp_payment_per_pkg"]).drop_duplicates(
            subset=["ndc_11"], keep="first"
        )

        cw_size = len(cw)
        print(f"    {label}: {cw_zip.name if cw_zip else 'forward-filled'} crosswalk ({cw_size:,} rows, {cw_source}), "
              f"pricing {pricing_zip.name} ({len(pricing):,} HCPCS), joined {len(joined):,} NDCs")

        quarter_frames.append({"year": year, "months": months, "data": joined, "label": label,
                               "cw_observed": cw_source == "observed"})

    if not quarter_frames:
        print("  No ASP quarters could be processed; writing empty parquet.")
        empty = pd.DataFrame(columns=["ndc_11", "year_month", "asp_payment_limit",
                                       "asp_billunits_per_pkg", "asp_payment_per_pkg",
                                       "asp_pkg_size", "asp_quarter",
                                       "has_asp_data", "asp_is_observed"])
        INTERMEDIATE.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(INTERMEDIATE / "asp_pricing.parquet", index=False)
        return 0

    # Expand each quarter to its 3 months
    monthly_rows = []
    for q in quarter_frames:
        year = q["year"]
        for m in q["months"]:
            ym = f"{year}-{m:02d}"
            if ym < STUDY_START or ym > STUDY_END:
                continue
            df = q["data"].copy()
            df["year_month"] = ym
            df["asp_quarter"] = q["label"]
            df["asp_is_observed"] = int(q["cw_observed"])
            monthly_rows.append(df)
    monthly = pd.concat(monthly_rows, ignore_index=True)

    # Within ndc_11 x year_month, prefer observed > forward-filled (already
    # mostly handled by quarter ordering but make explicit)
    monthly = monthly.sort_values(
        ["ndc_11", "year_month", "asp_is_observed"], ascending=[True, True, False]
    ).drop_duplicates(subset=["ndc_11", "year_month"], keep="first")
    monthly["has_asp_data"] = 1

    # Final column shape
    out = monthly[[
        "ndc_11", "year_month",
        "asp_payment_limit",
        "billunits_per_pkg",
        "asp_payment_per_pkg",
        "pkg_size",
        "asp_quarter",
        "has_asp_data",
        "asp_is_observed",
    ]].rename(columns={
        "billunits_per_pkg": "asp_billunits_per_pkg",
        "pkg_size": "asp_pkg_size",
    })

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    out_path = INTERMEDIATE / "asp_pricing.parquet"
    out.to_parquet(out_path, index=False)

    print(f"\n  Output: {out_path}")
    print(f"  Shape: {out.shape}")
    print(f"  Unique NDCs with ASP: {out['ndc_11'].nunique():,}")
    print(f"  Year-month coverage: {out['year_month'].min()} to {out['year_month'].max()}")
    print(f"  Observed-quarter rows: {(out['asp_is_observed'] == 1).sum():,}")
    print(f"  Forward-filled-crosswalk rows: {(out['asp_is_observed'] == 0).sum():,}")
    print(f"  Median asp_payment_limit: ${out['asp_payment_limit'].median():.2f}")
    print(f"  Median asp_payment_per_pkg: ${out['asp_payment_per_pkg'].median():.2f}")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
