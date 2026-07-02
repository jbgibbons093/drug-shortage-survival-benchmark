"""
00_utilities.py - Shared utility functions for Drug Shortage Predictions pipeline.

Consolidates reusable functions from the RA notebook and Generic mergers code:
- NDC formatting (5-4, 5-4-2)
- Company name normalization
- Orange Book file reading
- Application number standardization
- NDC extraction from shortage Presentation field
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "Raw Data"
INTERMEDIATE = PROJECT_ROOT / "Data" / "intermediate"
ANALYSIS = PROJECT_ROOT / "Data" / "analysis"
GENERIC_MERGERS = PROJECT_ROOT.parent / "Generic mergers"
HIST_OB = GENERIC_MERGERS / "data" / "raw" / "orange_book" / "nber_historical"
HIST_NDC = GENERIC_MERGERS / "data" / "raw" / "ndc_historical"

# Ensure output directories exist
INTERMEDIATE.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------------
STUDY_START = "2020-01"
STUDY_END = "2025-09"

# ---------------------------------------------------------------------------
# NDC formatting
# ---------------------------------------------------------------------------
def format_productndc(s):
    """Standardize Product NDC to #####-#### (5-digit labeler, 4-digit product)."""
    if pd.isna(s):
        return None
    parts = re.findall(r'\d+', str(s))
    if len(parts) >= 2:
        labeler, product = parts[0], parts[1]
    elif len(parts) == 1:
        d = parts[0]
        if len(d) < 5:
            return None
        labeler, product = d[:-4], d[-4:]
    else:
        return None
    return f"{labeler.zfill(5)}-{product.zfill(4)}"


def format_ndcpackagecode(s):
    """Standardize 11-digit NDC to #####-####-## (5-4-2 with leading zeros)."""
    if pd.isna(s):
        return None
    parts = re.findall(r'\d+', str(s))
    if len(parts) >= 3:
        labeler, product, package = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        labeler, product = parts[0], parts[1]
        package = "00"
    elif len(parts) == 1:
        d = parts[0]
        if len(d) < 7:
            return None
        labeler, product, package = d[:-6], d[-6:-2], d[-2:]
    else:
        return None
    return f"{labeler.zfill(5)}-{product.zfill(4)}-{package.zfill(2)}"


def labeler_code_from_ndc(ndc_11):
    """Extract 5-digit labeler code from formatted 11-digit NDC."""
    if pd.isna(ndc_11):
        return None
    return str(ndc_11).split("-")[0].zfill(5)


# ---------------------------------------------------------------------------
# NDC extraction from drug shortage Presentation field
# ---------------------------------------------------------------------------
NDC_RE = re.compile(
    r'(?i)ndc\s*([0-9]{1,5})[\-\u2010-\u2015\s]+([0-9]{1,4})[\-\u2010-\u2015\s]+([0-9]{1,2})'
)

def extract_ndc_from_presentation(text):
    """Extract raw NDC as 'a-b-c' from the Presentation string, or None."""
    if pd.isna(text):
        return None
    s = str(text)
    # Normalize various unicode dashes to simple hyphen
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    m = NDC_RE.search(s)
    if not m:
        return None
    a, b, c = m.groups()
    return f"{a}-{b}-{c}"


# ---------------------------------------------------------------------------
# Company / manufacturer name normalization
# ---------------------------------------------------------------------------
_BUSINESS_SUFFIXES = re.compile(
    r'\b(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|pllc|pc|lp|llp|ltd|limited'
    r'|pharmaceutical|pharmaceuticals|pharma|pharm|labs|laboratories|laboratory)\b'
)
_DBA_PATTERN = re.compile(r'(d\/b\/a|dba|aka|f\/k\/a|formerly known as)')

def normalize_company_name(s):
    """
    Normalize company/manufacturer name for fuzzy matching.

    Lowercases, strips business suffixes, removes d/b/a phrases,
    replaces & with 'and', removes punctuation, and collapses whitespace.
    """
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace("&", " and ")
    s = _DBA_PATTERN.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _BUSINESS_SUFFIXES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Orange Book file reading
# ---------------------------------------------------------------------------
def fix_appl_no(df):
    """Pad Appl_No to 6 digits with leading zeros. Modifies df in place."""
    df["Appl_No"] = df["Appl_No"].astype(str).str.zfill(6)
    return df


def read_orange_book_products(filepath):
    """
    Read an Orange Book products.txt file (tilde-delimited).
    Handles both directory-based (products.txt) and CSV formats.
    Returns DataFrame with standardized Appl_No.
    """
    filepath = Path(filepath)
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath, dtype=str)
        # CSV files from 2016-2018 have different column names
        col_map = {
            'ingredient': 'Ingredient',
            'dfroute': 'DF;Route',
            'trade_name': 'Trade_Name',
            'applicant': 'Applicant',
            'strength': 'Strength',
            'appl_type': 'Appl_Type',
            'appl_no': 'Appl_No',
            'product_no': 'Product_No',
            'tecode': 'TE_Code',
            'appfull': 'Applicant_Full_Name',
            'rld': 'RLD',
            'type': 'Type',
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    else:
        df = pd.read_csv(filepath, delimiter="~", dtype=str)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    if 'Appl_No' in df.columns:
        df = fix_appl_no(df)
    return df


def read_orange_book_patents(filepath):
    """Read Orange Book patent.txt (tilde-delimited)."""
    df = pd.read_csv(filepath, delimiter="~", dtype=str)
    df.columns = df.columns.str.strip()
    if 'Appl_No' in df.columns:
        df = fix_appl_no(df)
    return df


def read_orange_book_exclusivity(filepath):
    """Read Orange Book exclusivity.txt (tilde-delimited)."""
    df = pd.read_csv(filepath, delimiter="~", dtype=str)
    df.columns = df.columns.str.strip()
    if 'Appl_No' in df.columns:
        df = fix_appl_no(df)
    return df


def read_ndc_product(filepath, encoding=None):
    """
    Read NDC Directory product.txt (tab-delimited).
    Tries multiple encodings if needed.
    """
    encodings = [encoding] if encoding else ['utf-8', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, sep="\t", dtype=str, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Could not read {filepath} with any known encoding")


def read_ndc_package(filepath, encoding=None):
    """Read NDC Directory package.txt (tab-delimited)."""
    encodings = [encoding] if encoding else ['utf-8', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, sep="\t", dtype=str, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Could not read {filepath} with any known encoding")


# ---------------------------------------------------------------------------
# Application number extraction from NDC APPLICATIONNUMBER field
# ---------------------------------------------------------------------------
def extract_appl_no_from_ndc(application_number):
    """
    Extract 6-digit application number from NDC APPLICATIONNUMBER field.
    E.g., 'ANDA076432' -> '076432', 'NDA018781' -> '018781'
    """
    if pd.isna(application_number):
        return None
    digits = re.sub(r"\D", "", str(application_number))
    if not digits:
        return None
    return digits[-6:].zfill(6)


def extract_appl_type_from_ndc(application_number):
    """
    Extract application type from NDC APPLICATIONNUMBER field.
    E.g., 'ANDA076432' -> 'ANDA', 'NDA018781' -> 'NDA'
    """
    if pd.isna(application_number):
        return None
    letters = re.sub(r"[^A-Za-z]", "", str(application_number)).upper()
    if letters in ('ANDA', 'NDA', 'BLA'):
        return letters
    return None


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------
def generate_year_months(start, end):
    """Generate list of year-month strings from start to end (inclusive).

    Args:
        start: 'YYYY-MM' format
        end: 'YYYY-MM' format

    Returns:
        List of 'YYYY-MM' strings
    """
    periods = pd.period_range(start=start, end=end, freq='M')
    return [str(p) for p in periods]


def parse_date_flexible(s):
    """Parse date from various formats found in FDA data."""
    if pd.isna(s) or str(s).strip() == '':
        return pd.NaT
    s = str(s).strip()
    # Try common formats
    for fmt in ['%Y%m%d', '%m/%d/%Y', '%b %d, %Y', '%Y-%m-%d', '%m/%d/%y']:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    # Fallback to pandas inference
    try:
        return pd.to_datetime(s)
    except (ValueError, TypeError):
        return pd.NaT


def date_to_year_month(dt):
    """Convert datetime to 'YYYY-MM' string."""
    if pd.isna(dt):
        return None
    return dt.strftime('%Y-%m')


# ---------------------------------------------------------------------------
# NDC conversion for NADAC / SDUD (11-digit no-dash → 5-4-2 dashed)
# ---------------------------------------------------------------------------
def ndc_nodash_to_panel(ndc_str):
    """Convert 11-digit no-dash NDC (e.g. '24385005452') to panel format '24385-0054-52'."""
    if pd.isna(ndc_str):
        return None
    s = str(ndc_str).strip().zfill(11)
    if len(s) != 11 or not s.isdigit():
        return None
    return f"{s[:5]}-{s[5:9]}-{s[9:11]}"


if __name__ == "__main__":
    # Quick self-test
    assert format_productndc("0378-8700") == "00378-8700"
    assert format_productndc("378-8700") == "00378-8700"
    assert format_ndcpackagecode("0378-8700-06") == "00378-8700-06"
    assert format_ndcpackagecode("378-8700-6") == "00378-8700-06"
    assert extract_ndc_from_presentation("Tablet (NDC 0378-8700-06)") == "0378-8700-06"
    assert normalize_company_name("Teva Pharmaceutical Industries, Inc.") == "teva industries"
    assert extract_appl_no_from_ndc("ANDA076432") == "076432"
    assert extract_appl_type_from_ndc("ANDA076432") == "ANDA"
    print("All utility tests passed.")
