"""
download_ashp_shortages.py - Download drug shortage data from external sources
for independent validation against the FDA Drugshortages.csv.

=============================================================================
DATA SOURCE RESEARCH SUMMARY (March 2026)
=============================================================================

1. ASHP Drug Shortages API (https://www.ashp.org/drug-shortages)
   ---------------------------------------------------------------
   - ASHP maintains the gold-standard drug shortage database, managed by the
     University of Utah Drug Information Service (UUDIS) since 2001.
   - A REST API exists on Firebase: documented at
     https://github.com/ASHP-Software/drugShortagesDoc
   - Endpoints:
       /drugShortages            - all latest shortage versions
       /drugShortages/{key}/latest - single shortage, latest version
       /pastDrugShortages        - all historical versions
   - Data includes: affectedProduct (with NDC, RXCUI), availableProduct,
     shortageStatus, shortageReason, patientCareImplications, safetyNote,
     alternativeAgent, updatedAt (epoch ms), shortageVersion
   - STATUS: ***REQUIRES PAID API KEY***
     Contact softwaresupport@ashp.org for licensing and API key.
     The staging server (ahfs-staging.firebaseio.com) is not kept up to date.
     Production access requires a commercial license.
   - No public bulk download or scraping path is available.
   - The ASHP website at ashp.org/drug-shortages/current-shortages requires
     login and does not expose a machine-readable feed.

2. openFDA Drug Shortages API (https://api.fda.gov/drug/shortages.json)
   ---------------------------------------------------------------------
   - PUBLIC, FREE, NO API KEY REQUIRED (key recommended for heavy use).
   - This is the same data source as our existing Drugshortages.csv but
     accessed through the openFDA REST API with structured JSON + openfda
     harmonized fields (NDC, RxCUI, application_number, brand_name, etc.).
   - Total records: ~1,679 (as of March 2026)
   - Status values: "Current" (1,144), "To Be Discontinued" (494),
     "Resolved" (41)
   - Key fields: generic_name, company_name, status, dosage_form,
     presentation, package_ndc, initial_posting_date, update_date,
     change_date, discontinued_date, shortage_reason, availability,
     therapeutic_category, openfda.* (harmonized identifiers)
   - Supports pagination via skip/limit (up to 26,000) or search_after
     (unlimited) with Link header.
   - Bulk download: https://download.open.fda.gov/drug/shortages/
     drug-shortages-0001-of-0001.json.zip
   - Updated daily.
   - LIMITATION: This is the *same underlying FDA data*, so it is NOT a
     truly independent validation source. However, it provides structured
     access with harmonized NDC/RxCUI identifiers and can serve as a
     canonical reference feed to reconcile against our CSV.

3. FDA Drug Shortage Database (accessdata.fda.gov)
   ------------------------------------------------
   - Interactive web tool at https://www.accessdata.fda.gov/scripts/drugshortages
   - This is the source behind Drugshortages.csv.
   - No public API; CSV must be exported manually from the web interface.

4. HealthData.gov / Data.gov
   --------------------------
   - Dataset: "Current and Resolved Drug Shortages and Discontinuations
     Reported to FDA" (catalog.data.gov)
   - Points back to the same FDA accessdata page; no independent download.

5. ASHP Shortage Statistics (aggregate only)
   -------------------------------------------
   - https://www.ashp.org/drug-shortages/shortage-resources/drug-shortages-statistics
   - Provides quarterly counts (e.g., "216 active shortages") but NOT
     individual drug-level records.

=============================================================================
STRATEGY IMPLEMENTED IN THIS SCRIPT
=============================================================================

Since ASHP requires a paid API key, this script implements TWO download paths:

Path A - openFDA Drug Shortages API (always available):
    Downloads ALL records from the openFDA drug/shortages endpoint using
    the search_after pagination method. Saves structured CSV and Parquet
    with harmonized NDC identifiers. This gives us a machine-readable,
    daily-updated mirror of the FDA shortage database with richer metadata
    than the raw CSV.

Path B - ASHP API (if API key is available):
    If an ASHP API key is provided (via --ashp-key argument or
    ASHP_API_KEY environment variable), downloads the full ASHP shortage
    database including historical versions. This IS a truly independent
    source from UUDIS/ASHP with different editorial decisions.

Output:
    Raw Data/ASHP_Shortages/openfda_drug_shortages.csv
    Raw Data/ASHP_Shortages/openfda_drug_shortages.parquet
    Raw Data/ASHP_Shortages/ashp_drug_shortages.csv         (if key available)
    Raw Data/ASHP_Shortages/ashp_drug_shortages.parquet     (if key available)
    Raw Data/ASHP_Shortages/download_log.txt
"""

import urllib.request
import urllib.error
import urllib.parse
import json
import csv
import time
import sys
import os
import logging
import re
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Raw Data" / "ASHP_Shortages"
LOG_FILE = OUTPUT_DIR / "download_log.txt"

# openFDA settings
OPENFDA_BASE = "https://api.fda.gov/drug/shortages.json"
OPENFDA_BULK_URL = (
    "https://download.open.fda.gov/drug/shortages/"
    "drug-shortages-0001-of-0001.json.zip"
)
OPENFDA_LIMIT = 100  # max per page
OPENFDA_DELAY = 0.35  # ~240 req/min without API key
OPENFDA_MAX_RETRIES = 5
OPENFDA_RETRY_DELAY = 10

# ASHP Firebase settings
ASHP_STAGING_BASE = "https://ahfs-staging.firebaseio.com"
# Production base is not publicly documented; staging is for testing only
ASHP_DELAY = 0.5
ASHP_MAX_RETRIES = 3
ASHP_RETRY_DELAY = 5

# Output files
OPENFDA_CSV = OUTPUT_DIR / "openfda_drug_shortages.csv"
OPENFDA_PARQUET = OUTPUT_DIR / "openfda_drug_shortages.parquet"
ASHP_CSV = OUTPUT_DIR / "ashp_drug_shortages.csv"
ASHP_PARQUET = OUTPUT_DIR / "ashp_drug_shortages.parquet"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging():
    """Configure logging to both console and file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shortage_download")
    logger.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    return logger


log = None  # initialized in main()


# ---------------------------------------------------------------------------
# NDC standardization
# ---------------------------------------------------------------------------
def standardize_ndc_11(ndc_str):
    """Convert an NDC string to 11-digit format (5-4-2).

    Handles common FDA formats:
      - 5-4-2 (already standard)
      - 5-4-1 -> pad package to 2
      - 4-4-2 -> pad labeler to 5
      - 5-3-2 -> pad product to 4
    Also strips leading zeros that exceed segment widths.
    """
    if not ndc_str or not isinstance(ndc_str, str):
        return None
    ndc_str = ndc_str.strip()
    # Remove any non-digit, non-hyphen characters
    ndc_str = re.sub(r"[^\d\-]", "", ndc_str)
    if not ndc_str:
        return None

    parts = ndc_str.split("-")
    if len(parts) == 3:
        labeler, product, package = parts
    elif len(parts) == 2:
        # Treat as labeler-product, no package
        labeler, product = parts
        package = "00"
    elif len(parts) == 1:
        # Try to parse as 11-digit concatenated
        digits = re.sub(r"\D", "", ndc_str)
        if len(digits) == 11:
            labeler = digits[:5]
            product = digits[5:9]
            package = digits[9:11]
        elif len(digits) == 10:
            # Ambiguous; assume 5-4-1 (most common in FDA data)
            labeler = digits[:5]
            product = digits[5:9]
            package = digits[9:10]
        else:
            return None
    else:
        return None

    # Pad to standard widths: 5-4-2
    labeler = labeler.zfill(5)
    product = product.zfill(4)
    package = package.zfill(2)

    return f"{labeler}-{product}-{package}"


def standardize_ndc_9(ndc_str):
    """Convert an NDC string to 9-digit product NDC format (5-4).

    Used for product-level matching against the panel.
    """
    if not ndc_str or not isinstance(ndc_str, str):
        return None
    ndc_str = ndc_str.strip()
    ndc_str = re.sub(r"[^\d\-]", "", ndc_str)
    if not ndc_str:
        return None

    parts = ndc_str.split("-")
    if len(parts) >= 2:
        labeler, product = parts[0], parts[1]
    elif len(parts) == 1:
        digits = re.sub(r"\D", "", ndc_str)
        if len(digits) >= 9:
            labeler = digits[:5]
            product = digits[5:9]
        else:
            return None
    else:
        return None

    labeler = labeler.zfill(5)
    product = product.zfill(4)
    return f"{labeler}-{product}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def fetch_json(url, retries=OPENFDA_MAX_RETRIES, retry_delay=OPENFDA_RETRY_DELAY,
               timeout=60):
    """Fetch a URL and parse JSON response. Returns (data_dict, response_headers)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DrugShortageProject/1.0")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                headers = dict(response.headers)
                data = json.loads(response.read().decode("utf-8"))
                return data, headers
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = retry_delay * (attempt + 1)
                if log:
                    log.warning(f"HTTP {e.code} on {url[:80]}... "
                                f"Retry {attempt+1}/{retries} in {wait}s")
                time.sleep(wait)
            elif e.code == 404:
                if log:
                    log.warning(f"HTTP 404: {url[:120]}")
                return None, {}
            else:
                if log:
                    log.error(f"HTTP {e.code}: {e.reason} - {url[:120]}")
                if attempt == retries - 1:
                    return None, {}
                time.sleep(retry_delay)
        except (urllib.error.URLError, Exception) as e:
            wait = retry_delay * (attempt + 1)
            if log:
                log.warning(f"Error: {e}. Retry {attempt+1}/{retries} in {wait}s")
            time.sleep(wait)
    if log:
        log.error(f"FAILED after {retries} retries: {url[:120]}")
    return None, {}


def fetch_raw(url, retries=3, retry_delay=5, timeout=120):
    """Fetch raw bytes from a URL (for bulk downloads)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DrugShortageProject/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as e:
            wait = retry_delay * (attempt + 1)
            if log:
                log.warning(f"Download error: {e}. Retry {attempt+1}/{retries}")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# openFDA helpers
# ---------------------------------------------------------------------------
def flatten_openfda(openfda_dict, field):
    """Extract a field from the openfda nested object, joining lists with |."""
    if not openfda_dict or field not in openfda_dict:
        return ""
    val = openfda_dict[field]
    if isinstance(val, list):
        return " | ".join(str(v) for v in val)
    return str(val)


def extract_openfda_record(record):
    """Flatten one openFDA drug shortage record into a flat dict."""
    row = {}

    # Top-level fields
    top_fields = [
        "generic_name", "company_name", "status", "dosage_form",
        "presentation", "package_ndc", "contact_info", "update_type",
        "availability", "related_info", "related_info_link",
        "resolved_note", "shortage_reason",
        "initial_posting_date", "update_date", "change_date",
        "discontinued_date", "type",
    ]
    for f in top_fields:
        val = record.get(f, "")
        if isinstance(val, list):
            val = " | ".join(str(v) for v in val)
        row[f] = val if val else ""

    # Therapeutic category (list -> joined)
    tc = record.get("therapeutic_category", [])
    row["therapeutic_category"] = " | ".join(tc) if isinstance(tc, list) else str(tc or "")

    # Strength (list -> joined)
    st = record.get("strength", [])
    row["strength"] = " | ".join(str(s) for s in st) if isinstance(st, list) else str(st or "")

    # openfda harmonized fields
    openfda = record.get("openfda", {})
    openfda_fields = [
        "application_number", "brand_name", "generic_name",
        "manufacturer_name", "product_ndc", "product_type",
        "route", "substance_name", "rxcui", "spl_id", "spl_set_id",
        "package_ndc", "nui", "pharm_class_epc", "pharm_class_moa",
        "unii",
    ]
    for f in openfda_fields:
        row[f"openfda_{f}"] = flatten_openfda(openfda, f)

    # Standardized NDC columns
    # Use the top-level package_ndc first, then openfda package_ndc
    raw_ndc = row.get("package_ndc", "")
    if not raw_ndc:
        raw_ndc = row.get("openfda_package_ndc", "").split(" | ")[0] if row.get("openfda_package_ndc") else ""
    row["ndc_11"] = standardize_ndc_11(raw_ndc)

    # Product NDC from openfda
    product_ndc_raw = row.get("openfda_product_ndc", "").split(" | ")[0] if row.get("openfda_product_ndc") else ""
    row["product_ndc_9"] = standardize_ndc_9(product_ndc_raw) if product_ndc_raw else None

    return row


# ---------------------------------------------------------------------------
# Path A: openFDA Drug Shortages API download
# ---------------------------------------------------------------------------
def download_openfda_via_api():
    """Download all records from openFDA drug/shortages using search_after pagination.

    Uses the Link header rel="next" approach which supports unlimited result sets,
    unlike skip/limit which caps at 26,000.
    """
    log.info("=" * 70)
    log.info("PATH A: Downloading from openFDA Drug Shortages API")
    log.info("=" * 70)

    all_records = []

    # First, get total count
    count_url = f"{OPENFDA_BASE}?limit=1"
    data, _ = fetch_json(count_url)
    if data is None:
        log.error("Cannot reach openFDA API. Trying bulk download fallback.")
        return download_openfda_via_bulk()

    total = data.get("meta", {}).get("results", {}).get("total", 0)
    log.info(f"Total records in openFDA drug shortages: {total:,}")

    if total == 0:
        log.error("No records found in openFDA drug shortages.")
        return []

    # Strategy: If total <= 26,000, use simple skip/limit pagination
    # (search_after requires a sort field; skip/limit is simpler for small datasets)
    if total <= 25000:
        log.info("Using skip/limit pagination (dataset is small enough)")
        skip = 0
        while skip < total:
            url = f"{OPENFDA_BASE}?limit={OPENFDA_LIMIT}&skip={skip}"
            data, _ = fetch_json(url)
            if data is None or "results" not in data:
                log.warning(f"No results at skip={skip}. Stopping.")
                break

            results = data["results"]
            if len(results) == 0:
                break

            for rec in results:
                all_records.append(extract_openfda_record(rec))

            skip += OPENFDA_LIMIT
            log.info(f"  Fetched {len(all_records):,} / {total:,} records")
            time.sleep(OPENFDA_DELAY)
    else:
        # Use search_after for large datasets
        log.info("Using search_after pagination (dataset > 25,000)")
        url = f"{OPENFDA_BASE}?limit={OPENFDA_LIMIT}&sort=initial_posting_date:asc"
        page = 0
        while url:
            page += 1
            data, headers = fetch_json(url)
            if data is None or "results" not in data:
                break

            results = data["results"]
            if len(results) == 0:
                break

            for rec in results:
                all_records.append(extract_openfda_record(rec))

            # Extract next URL from Link header
            link_header = headers.get("Link", "")
            next_url = None
            if link_header:
                # Parse Link header: <url>; rel="next"
                for part in link_header.split(","):
                    part = part.strip()
                    if 'rel="next"' in part or "rel='next'" in part:
                        match = re.search(r"<(.+?)>", part)
                        if match:
                            next_url = match.group(1)
                            break
            url = next_url
            log.info(f"  Page {page}: {len(all_records):,} records so far")
            time.sleep(OPENFDA_DELAY)

    log.info(f"Total openFDA records downloaded: {len(all_records):,}")
    return all_records


def download_openfda_via_bulk():
    """Fallback: download the bulk JSON zip file from openFDA."""
    import zipfile
    import io

    log.info("Attempting bulk download from openFDA...")
    log.info(f"URL: {OPENFDA_BULK_URL}")

    raw_data = fetch_raw(OPENFDA_BULK_URL, retries=3, retry_delay=10, timeout=180)
    if raw_data is None:
        log.error("Bulk download failed.")
        return []

    log.info(f"Downloaded {len(raw_data):,} bytes")

    # Extract JSON from zip
    try:
        with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
            names = zf.namelist()
            log.info(f"Zip contains: {names}")
            if not names:
                log.error("Zip file is empty.")
                return []
            # Read the first JSON file
            json_name = [n for n in names if n.endswith(".json")]
            if not json_name:
                log.error(f"No JSON file in zip. Contents: {names}")
                return []
            with zf.open(json_name[0]) as jf:
                data = json.loads(jf.read().decode("utf-8"))
    except Exception as e:
        log.error(f"Error extracting bulk download: {e}")
        return []

    # The bulk file contains a top-level "results" array
    results = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(results, dict) and "results" not in results:
        # Some bulk files are just an array at top level
        results = [data] if not isinstance(data, list) else data

    if isinstance(results, dict):
        results = results.get("results", [])

    all_records = []
    for rec in results:
        all_records.append(extract_openfda_record(rec))

    log.info(f"Bulk download: extracted {len(all_records):,} records")
    return all_records


def save_openfda_records(records):
    """Save openFDA records to CSV and Parquet."""
    if not records:
        log.warning("No openFDA records to save.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine column order
    all_keys = []
    seen = set()
    for r in records:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Save CSV
    log.info(f"Saving {len(records):,} records to {OPENFDA_CSV}")
    with open(OPENFDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Save Parquet if pandas is available
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        # Parse date columns
        date_cols = ["initial_posting_date", "update_date", "change_date",
                     "discontinued_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed",
                                          dayfirst=False, errors="coerce")
        df.to_parquet(OPENFDA_PARQUET, index=False)
        log.info(f"Saved Parquet: {OPENFDA_PARQUET}")
    except ImportError:
        log.warning("pandas not available; skipping Parquet output.")
    except Exception as e:
        log.warning(f"Parquet save failed: {e}")

    # Summary statistics
    status_counts = {}
    for r in records:
        s = r.get("status", "Unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    has_ndc = sum(1 for r in records if r.get("ndc_11"))
    has_product_ndc = sum(1 for r in records if r.get("product_ndc_9"))

    log.info(f"\nopenFDA Drug Shortages Summary:")
    log.info(f"  Total records: {len(records):,}")
    log.info(f"  By status: {dict(sorted(status_counts.items()))}")
    log.info(f"  Records with standardized NDC-11: {has_ndc:,} "
             f"({100*has_ndc/len(records):.1f}%)")
    log.info(f"  Records with product NDC-9: {has_product_ndc:,} "
             f"({100*has_product_ndc/len(records):.1f}%)")

    # Date range
    dates = [r.get("initial_posting_date", "") for r in records
             if r.get("initial_posting_date")]
    if dates:
        log.info(f"  Initial posting date range: {min(dates)} to {max(dates)}")


# ---------------------------------------------------------------------------
# Path B: ASHP API download (requires API key)
# ---------------------------------------------------------------------------
def download_ashp(api_key, base_url=None):
    """Download shortage data from the ASHP Drug Shortages API.

    Requires a valid API key obtained from softwaresupport@ashp.org.

    The ASHP API is a Firebase REST API:
      /drugShortages.json?auth={key}       - all current shortage latest versions
      /pastDrugShortages.json?auth={key}   - all historical versions

    Returns a list of flattened records.
    """
    log.info("=" * 70)
    log.info("PATH B: Downloading from ASHP Drug Shortages API")
    log.info("=" * 70)

    if base_url is None:
        # Default to staging; production URL would be provided by ASHP
        # with the API key. The staging database is NOT kept up to date.
        base_url = ASHP_STAGING_BASE
        log.warning("Using ASHP staging server - data may be stale.")
        log.warning("For production data, ASHP should provide the production URL "
                     "along with your API key.")

    all_records = []

    # ------------------------------------------------------------------
    # Download current shortages
    # ------------------------------------------------------------------
    log.info("Fetching current drug shortages...")
    url = f"{base_url}/drugShortages.json?auth={api_key}"
    data, _ = fetch_json(url, retries=ASHP_MAX_RETRIES,
                         retry_delay=ASHP_RETRY_DELAY, timeout=120)

    if data is None:
        log.error("Failed to fetch ASHP current shortages. Check your API key.")
        log.error("If using staging, note that the staging server may be down.")
        log.error("Contact softwaresupport@ashp.org for production access.")
        return []

    if isinstance(data, dict):
        log.info(f"  Received dict with {len(data)} keys")
        # Firebase returns a dict keyed by shortage ID
        for shortage_id, shortage_data in data.items():
            if shortage_data is None:
                continue
            # Get the latest version
            if isinstance(shortage_data, dict):
                latest = shortage_data.get("latest", shortage_data)
                record = flatten_ashp_record(shortage_id, latest, "current")
                if record:
                    all_records.append(record)
    elif isinstance(data, list):
        log.info(f"  Received list with {len(data)} entries")
        for i, shortage_data in enumerate(data):
            if shortage_data is None:
                continue
            if isinstance(shortage_data, dict):
                latest = shortage_data.get("latest", shortage_data)
                record = flatten_ashp_record(str(i), latest, "current")
                if record:
                    all_records.append(record)

    log.info(f"  Current shortage records: {len(all_records):,}")
    time.sleep(ASHP_DELAY)

    # ------------------------------------------------------------------
    # Download past/historical shortages
    # ------------------------------------------------------------------
    log.info("Fetching past drug shortages...")
    url = f"{base_url}/pastDrugShortages.json?auth={api_key}"
    data, _ = fetch_json(url, retries=ASHP_MAX_RETRIES,
                         retry_delay=ASHP_RETRY_DELAY, timeout=180)

    past_count = 0
    if data is None:
        log.warning("Failed to fetch ASHP past shortages (may be large).")
    elif isinstance(data, dict):
        for shortage_id, versions in data.items():
            if versions is None:
                continue
            if isinstance(versions, dict):
                # Each shortage has multiple version keys
                for version_key, version_data in versions.items():
                    if version_data is None:
                        continue
                    record = flatten_ashp_record(shortage_id, version_data,
                                                 "historical",
                                                 version_key=version_key)
                    if record:
                        all_records.append(record)
                        past_count += 1
    log.info(f"  Historical shortage records: {past_count:,}")

    log.info(f"Total ASHP records: {len(all_records):,}")
    return all_records


def flatten_ashp_record(shortage_id, data, record_type, version_key=None):
    """Flatten an ASHP shortage record into a flat dict.

    ASHP records have a different structure from FDA:
    - affectedProduct: list of {NDC, RXCUI, textDescription, discontinued, lastChangeDate}
    - availableProduct: same structure for alternatives
    - shortageTitle, shortageStatus, shortageReason, etc.
    """
    if not isinstance(data, dict):
        return None

    row = {
        "ashp_shortage_id": str(shortage_id),
        "record_type": record_type,
        "version_key": str(version_key) if version_key else "",
    }

    # Simple string fields
    simple_fields = [
        "shortageTitle", "shortageStatus", "shortageVersion",
        "lastRevisedDate", "shortageCreateDate", "searchString",
        "updateHistory",
    ]
    for f in simple_fields:
        row[f] = str(data.get(f, "")) if data.get(f) is not None else ""

    # updatedAt is epoch milliseconds
    updated_at = data.get("updatedAt")
    if updated_at and isinstance(updated_at, (int, float)):
        try:
            row["updated_at_utc"] = datetime.utcfromtimestamp(
                updated_at / 1000.0
            ).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            row["updated_at_utc"] = ""
    else:
        row["updated_at_utc"] = ""

    # List/array fields - join as text
    list_fields = [
        "shortageReason", "patientCareImplications", "safetyNote",
        "resupplyEstimateNote", "alternativeAgent", "shortageAuthor",
        "reference",
    ]
    for f in list_fields:
        val = data.get(f, [])
        if isinstance(val, list):
            # Filter out HTML table entries (alternativeAgentTable)
            text_vals = [str(v) for v in val if v]
            row[f] = " | ".join(text_vals)
        elif val:
            row[f] = str(val)
        else:
            row[f] = ""

    # Affected products - extract NDCs and descriptions
    affected = data.get("affectedProduct", [])
    if isinstance(affected, list):
        ndcs = []
        rxcuis = []
        descriptions = []
        for prod in affected:
            if not isinstance(prod, dict):
                continue
            ndc = prod.get("NDC", "")
            if ndc:
                ndcs.append(str(ndc))
            rxcui = prod.get("RXCUI", "")
            if rxcui:
                rxcuis.append(str(rxcui))
            desc = prod.get("textDescription", "")
            if desc:
                descriptions.append(str(desc))

        row["affected_ndcs"] = " | ".join(ndcs)
        row["affected_rxcuis"] = " | ".join(rxcuis)
        row["affected_descriptions"] = " | ".join(descriptions)
        row["affected_product_count"] = len(affected)

        # Standardize first NDC for matching
        if ndcs:
            row["ndc_11"] = standardize_ndc_11(ndcs[0])
            row["product_ndc_9"] = standardize_ndc_9(ndcs[0])
            # Also store all standardized NDCs
            std_ndcs = [standardize_ndc_11(n) for n in ndcs]
            row["all_ndc_11"] = " | ".join(n for n in std_ndcs if n)
        else:
            row["ndc_11"] = None
            row["product_ndc_9"] = None
            row["all_ndc_11"] = ""
    else:
        row["affected_ndcs"] = ""
        row["affected_rxcuis"] = ""
        row["affected_descriptions"] = ""
        row["affected_product_count"] = 0
        row["ndc_11"] = None
        row["product_ndc_9"] = None
        row["all_ndc_11"] = ""

    # Available products
    available = data.get("availableProduct", [])
    if isinstance(available, list):
        row["available_product_count"] = len(available)
        avail_descs = []
        for prod in available:
            if isinstance(prod, dict):
                desc = prod.get("textDescription", "")
                if desc:
                    avail_descs.append(str(desc))
        row["available_descriptions"] = " | ".join(avail_descs)
    else:
        row["available_product_count"] = 0
        row["available_descriptions"] = ""

    return row


def save_ashp_records(records):
    """Save ASHP records to CSV and Parquet."""
    if not records:
        log.warning("No ASHP records to save.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine column order
    all_keys = []
    seen = set()
    for r in records:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Save CSV
    log.info(f"Saving {len(records):,} ASHP records to {ASHP_CSV}")
    with open(ASHP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Save Parquet
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_parquet(ASHP_PARQUET, index=False)
        log.info(f"Saved Parquet: {ASHP_PARQUET}")
    except ImportError:
        log.warning("pandas not available; skipping Parquet output.")
    except Exception as e:
        log.warning(f"Parquet save failed: {e}")

    # Summary
    status_counts = {}
    for r in records:
        s = r.get("shortageStatus", "Unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    type_counts = {}
    for r in records:
        t = r.get("record_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    has_ndc = sum(1 for r in records if r.get("ndc_11"))
    log.info(f"\nASHP Drug Shortages Summary:")
    log.info(f"  Total records: {len(records):,}")
    log.info(f"  By status: {dict(sorted(status_counts.items()))}")
    log.info(f"  By record type: {dict(sorted(type_counts.items()))}")
    log.info(f"  Records with NDC: {has_ndc:,}")


# ---------------------------------------------------------------------------
# Cross-reference report
# ---------------------------------------------------------------------------
def generate_crossref_report(openfda_records, ashp_records):
    """Generate a brief cross-reference report comparing the two sources."""
    log.info("=" * 70)
    log.info("CROSS-REFERENCE REPORT")
    log.info("=" * 70)

    if not openfda_records:
        log.info("No openFDA records available for comparison.")
        return
    if not ashp_records:
        log.info("No ASHP records available for comparison.")
        log.info("To obtain ASHP data, contact softwaresupport@ashp.org for an API key.")
        return

    # Compare by NDC overlap
    openfda_ndcs = {r["ndc_11"] for r in openfda_records if r.get("ndc_11")}
    ashp_ndcs = set()
    for r in ashp_records:
        if r.get("all_ndc_11"):
            for ndc in r["all_ndc_11"].split(" | "):
                ndc = ndc.strip()
                if ndc:
                    ashp_ndcs.add(ndc)
        elif r.get("ndc_11"):
            ashp_ndcs.add(r["ndc_11"])

    overlap = openfda_ndcs & ashp_ndcs
    only_fda = openfda_ndcs - ashp_ndcs
    only_ashp = ashp_ndcs - openfda_ndcs

    log.info(f"  openFDA unique NDCs: {len(openfda_ndcs):,}")
    log.info(f"  ASHP unique NDCs:    {len(ashp_ndcs):,}")
    log.info(f"  Overlapping NDCs:    {len(overlap):,}")
    log.info(f"  Only in openFDA:     {len(only_fda):,}")
    log.info(f"  Only in ASHP:        {len(only_ashp):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download drug shortage data from openFDA and optionally ASHP."
    )
    parser.add_argument(
        "--ashp-key",
        type=str,
        default=None,
        help="ASHP Drug Shortages API key. Also reads ASHP_API_KEY env var."
    )
    parser.add_argument(
        "--ashp-url",
        type=str,
        default=None,
        help="ASHP API base URL (production URL provided with your key)."
    )
    parser.add_argument(
        "--skip-openfda",
        action="store_true",
        help="Skip the openFDA download."
    )
    parser.add_argument(
        "--bulk",
        action="store_true",
        help="Use bulk zip download instead of API pagination for openFDA."
    )
    args = parser.parse_args()

    global log
    log = setup_logging()

    log.info("=" * 70)
    log.info("Drug Shortage External Data Download")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info("=" * 70)

    start_time = time.time()
    openfda_records = []
    ashp_records = []

    # ------------------------------------------------------------------
    # Path A: openFDA
    # ------------------------------------------------------------------
    if not args.skip_openfda:
        try:
            if args.bulk:
                openfda_records = download_openfda_via_bulk()
            else:
                openfda_records = download_openfda_via_api()
                # Fallback to bulk if API pagination returns nothing
                if not openfda_records:
                    log.info("API pagination returned no results; trying bulk download.")
                    openfda_records = download_openfda_via_bulk()
            save_openfda_records(openfda_records)
        except Exception as e:
            log.error(f"openFDA download failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Path B: ASHP (if key available)
    # ------------------------------------------------------------------
    ashp_key = args.ashp_key or os.environ.get("ASHP_API_KEY")
    if ashp_key:
        try:
            ashp_records = download_ashp(ashp_key, base_url=args.ashp_url)
            save_ashp_records(ashp_records)
        except Exception as e:
            log.error(f"ASHP download failed: {e}", exc_info=True)
    else:
        log.info("")
        log.info("=" * 70)
        log.info("ASHP API KEY NOT PROVIDED")
        log.info("=" * 70)
        log.info("The ASHP Drug Shortages API requires a commercial API key.")
        log.info("This is the only truly independent shortage data source")
        log.info("(maintained by U of Utah Drug Information Service since 2001).")
        log.info("")
        log.info("To obtain access:")
        log.info("  1. Email softwaresupport@ashp.org")
        log.info("  2. Request API key and licensing information")
        log.info("  3. Re-run this script with --ashp-key YOUR_KEY")
        log.info("     or set ASHP_API_KEY environment variable")
        log.info("")
        log.info("API documentation: https://github.com/ASHP-Software/drugShortagesDoc")
        log.info("=" * 70)

    # ------------------------------------------------------------------
    # Cross-reference
    # ------------------------------------------------------------------
    generate_crossref_report(openfda_records, ashp_records)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    log.info("")
    log.info("=" * 70)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  openFDA records: {len(openfda_records):,}")
    log.info(f"  ASHP records:    {len(ashp_records):,}")
    log.info(f"  Elapsed time:    {elapsed:.1f}s")
    log.info(f"  Output:          {OUTPUT_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
