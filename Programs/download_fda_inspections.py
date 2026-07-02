"""
download_fda_inspections.py - Download FDA inspections from the FDA dashboard.

Strategy:
1. Use Playwright to export the live "Entire Dataset" workbook from the dashboard
2. Fallback to the legacy ORA/openFDA API probes
3. If all automated paths fail, provide manual download instructions

Output: Raw Data/FDA Inspections/fda_inspections_dashboard_export.xlsx
"""

import urllib.request
import urllib.error
import urllib.parse
import json
import csv
import re
import time
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Raw Data" / "FDA Inspections"
OUTPUT_FILE = OUTPUT_DIR / "fda_inspections_dashboard_export.xlsx"
DASHBOARD_URL = "https://datadashboard.fda.gov/oii/cd/inspections.htm"

DELAY = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 5

CSV_COLUMNS = [
    "firm_name", "fei_number", "city", "state", "country_area",
    "inspection_end_date", "classification", "project_area",
    "center", "product_type", "posted_citations",
]


def find_edge_executable():
    """Find a local Edge installation for Playwright."""
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_url(url, retries=MAX_RETRIES, headers=None):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DrugShortageProject/1.0")
            req.add_header("Accept", "application/json")
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  HTTP {e.code}. Waiting {wait}s (retry {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"  HTTP {e.code}: {e.reason}")
                return None
        except (urllib.error.URLError, Exception) as e:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  Error: {e}. Retry {attempt+1}/{retries}...")
            time.sleep(wait)
    return None


def try_dashboard_export():
    """Download the inspections workbook from the live FDA dashboard."""
    print("\n[1] Trying live FDA dashboard export with Playwright...")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright is not installed.")
        return None

    launch_kwargs = {"headless": True}
    edge_path = find_edge_executable()
    if edge_path is not None:
        launch_kwargs["executable_path"] = str(edge_path)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=90_000)
            # The Qlik page needs a few extra seconds after DOMContentLoaded
            # before the export controls reliably attach their handlers.
            page.wait_for_timeout(10_000)
            page.locator("#export-dropdownData1").wait_for(state="visible", timeout=120_000)
            page.locator("#export-dropdownData1").click(timeout=10_000)
            page.wait_for_timeout(1_500)
            page.locator("#exp-dt5").wait_for(state="visible", timeout=30_000)

            with page.expect_download(timeout=180_000) as download_info:
                page.locator("#exp-dt5").click(timeout=10_000)

            download = download_info.value
            download.save_as(str(OUTPUT_FILE))
            context.close()
            browser.close()

        print(f"  Downloaded dashboard export to: {OUTPUT_FILE}")
        return OUTPUT_FILE
    except Exception as e:
        print(f"  Dashboard export failed: {e}")
        return None


def try_ora_dashboard():
    """Try FDA ORA Data Dashboard API for inspection classifications."""
    print("\n[2] Trying FDA ORA Data Dashboard API...")

    # The ORA Dashboard provides inspection classification data
    # Try the known inspection classifications endpoint
    base_urls = [
        "https://datadashboard.fda.gov/ora/api/inspections",
        "https://datadashboard.fda.gov/api/inspections",
    ]

    for base in base_urls:
        # Test with a small request first
        test_params = urllib.parse.urlencode({
            '$top': 5,
            '$filter': "Center eq 'CDER'",
        })
        test_url = f"{base}?{test_params}"
        print(f"  Testing: {test_url}")
        resp = fetch_url(test_url)
        if resp:
            try:
                data = json.loads(resp)
                if isinstance(data, dict) and "value" in data:
                    print(f"  Found OData endpoint at {base}")
                    return download_from_odata(base, data)
                elif isinstance(data, list) and len(data) > 0:
                    print(f"  Found list endpoint at {base}")
                    return download_from_list(base, data)
            except json.JSONDecodeError:
                continue

        # Try without OData syntax
        test_url2 = f"{base}?{urllib.parse.urlencode({'limit': 5, 'center': 'CDER'})}"
        resp2 = fetch_url(test_url2)
        if resp2:
            try:
                data = json.loads(resp2)
                if isinstance(data, (list, dict)):
                    print(f"  Found endpoint at {base}")
                    return download_from_generic(base, data)
            except json.JSONDecodeError:
                continue

    print("  ORA Dashboard API not accessible.")
    return None


def download_from_odata(base_url, initial_data):
    """Download all CDER inspection records via OData pagination."""
    all_records = []
    page = 0
    page_size = 1000

    while True:
        skip = page * page_size
        params = urllib.parse.urlencode({
            '$top': page_size,
            '$skip': skip,
            '$filter': "Center eq 'CDER' and year(InspectionEndDate) ge 2018",
        })
        url = f"{base_url}?{params}"
        resp = fetch_url(url)
        if not resp:
            break
        try:
            data = json.loads(resp)
            results = data.get("value", [])
        except (json.JSONDecodeError, AttributeError):
            break

        if not results:
            break

        for r in results:
            all_records.append({
                "firm_name": r.get("FirmName", r.get("firm_name", "")),
                "fei_number": r.get("FEINumber", r.get("fei_number", "")),
                "city": r.get("City", r.get("city", "")),
                "state": r.get("State", r.get("state", "")),
                "country_area": r.get("CountryArea", r.get("country_area", "")),
                "inspection_end_date": r.get("InspectionEndDate", r.get("inspection_end_date", "")),
                "classification": r.get("Classification", r.get("classification", "")),
                "project_area": r.get("ProjectArea", r.get("project_area", "")),
                "center": r.get("Center", r.get("center", "")),
                "product_type": r.get("ProductType", r.get("product_type", "")),
                "posted_citations": r.get("PostedCitations", r.get("posted_citations", "")),
            })

        print(f"  Page {page}: {len(results)} records (total: {len(all_records)})")
        page += 1
        if len(results) < page_size:
            break
        time.sleep(DELAY)

    return all_records if all_records else None


def download_from_list(base_url, initial_data):
    """Download from a simple list API."""
    all_records = []
    offset = 0
    limit = 1000

    while True:
        url = f"{base_url}?{urllib.parse.urlencode({'limit': limit, 'offset': offset, 'center': 'CDER'})}"
        resp = fetch_url(url)
        if not resp:
            break
        try:
            data = json.loads(resp)
            if isinstance(data, dict):
                results = data.get("results", data.get("data", []))
            else:
                results = data
        except json.JSONDecodeError:
            break

        if not results:
            break

        for r in results:
            all_records.append({
                "firm_name": r.get("firm_name", r.get("FirmName", "")),
                "fei_number": r.get("fei_number", r.get("FEINumber", "")),
                "city": r.get("city", r.get("City", "")),
                "state": r.get("state", r.get("State", "")),
                "country_area": r.get("country_area", r.get("CountryArea", "")),
                "inspection_end_date": r.get("inspection_end_date", r.get("InspectionEndDate", "")),
                "classification": r.get("classification", r.get("Classification", "")),
                "project_area": r.get("project_area", r.get("ProjectArea", "")),
                "center": r.get("center", r.get("Center", "")),
                "product_type": r.get("product_type", r.get("ProductType", "")),
                "posted_citations": r.get("posted_citations", r.get("PostedCitations", "")),
            })

        print(f"  Offset {offset}: {len(results)} records (total: {len(all_records)})")
        offset += limit
        if len(results) < limit:
            break
        time.sleep(DELAY)

    return all_records if all_records else None


def download_from_generic(base_url, initial_data):
    """Handle unknown API format."""
    print(f"  Unknown format. Sample: {str(initial_data)[:200]}")
    return None


def try_openfda_compliance():
    """Try openFDA for any inspection-related endpoints."""
    print("\n[3] Checking openFDA for inspection data...")

    endpoints = [
        "https://api.fda.gov/drug/drugsfda.json?search=submissions.submission_type:SUPPL&limit=1",
    ]

    for url in endpoints:
        resp = fetch_url(url)
        if resp:
            try:
                data = json.loads(resp)
                if "results" in data:
                    print(f"  Found data at endpoint, but not inspection classifications.")
            except json.JSONDecodeError:
                pass

    print("  No direct inspection classification endpoint found in openFDA.")
    return None


def print_manual_instructions():
    """Print instructions for manual download if API fails."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print("""
The FDA inspection classification data is not available through a simple API.
Please download manually from one of these sources:

1. FDA ORA FOIA Electronic Reading Room:
   https://www.fda.gov/about-fda/office-regulatory-affairs/ora-foia-electronic-reading-room
   -> Look for "Inspection Classifications"
   -> Download the dataset, filter to Product Type = "Drugs"

2. FDA Compliance Actions Dashboard:
   https://datadashboard.fda.gov/oii/cd/inspections.htm
   -> Open "Download Inspections Dataset"
   -> Download "Entire Dataset" as XLSX

3. After downloading, save the file as:
   {output}

Required columns (rename if needed):
   firm_name, fei_number, city, state, country_area,
   inspection_end_date, classification, project_area,
   center, product_type, posted_citations

Classification values should be:
   NAI = No Action Indicated
   VAI = Voluntary Action Indicated
   OAI = Official Action Indicated
""".format(output=str(OUTPUT_FILE)))


def main():
    print("FDA Inspection Classifications Download")
    print(f"Output: {OUTPUT_FILE}\n")

    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_file = try_dashboard_export()
    if export_file is not None and export_file.exists():
        print(f"\nCompleted in {time.time() - start_time:.1f}s")
        sys.exit(0)

    records = try_ora_dashboard()

    if records is None:
        records = try_openfda_compliance()

    if records is None or len(records) == 0:
        print_manual_instructions()

        # Create a minimal placeholder with expected columns
        placeholder = OUTPUT_DIR / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(placeholder, "w") as f:
            f.write("See console output or download_fda_inspections.py for instructions.\n")
            f.write(f"Expected output: {OUTPUT_FILE}\n")
        print(f"\nPlaceholder saved to: {placeholder}")
        sys.exit(0)

    # Filter to CDER if not already filtered. Parse the year robustly: a
    # plain string comparison (date >= "2018") silently drops valid records
    # when dates arrive as MM/DD/YYYY, because "0..." < "2018".
    def _end_year(rec):
        raw = str(rec.get("inspection_end_date", "")).strip()
        m = re.search(r"(19|20)\d{2}", raw)
        return int(m.group(0)) if m else 0

    cder_records = [r for r in records
                    if r.get("center", "").upper() in ("CDER", "")
                    and _end_year(r) >= 2018]
    if cder_records:
        records = cder_records

    # Save
    csv_output = OUTPUT_DIR / "fda_inspections_2018_2025.csv"
    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Summary
    class_counts = {}
    for r in records:
        cl = r.get("classification", "Unknown")
        class_counts[cl] = class_counts.get(cl, 0) + 1

    print(f"\nSaved {len(records)} records to: {csv_output}")
    print(f"By classification: {dict(sorted(class_counts.items()))}")
    print(f"Completed in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
