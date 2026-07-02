"""
download_fda_adverse_events.py - Download FDA adverse event report counts from openFDA.

Queries the openFDA drug/event endpoint to get monthly counts of adverse event
reports by manufacturer (openfda.manufacturer_name) for 2018-2025.

We download COUNTS (not individual reports) using the API's count endpoint,
which is much more efficient than paginating through millions of individual AE reports.

Strategy: For each month, query total AE report count per manufacturer.

Output: Raw Data/FDA Adverse Events/fda_adverse_event_counts.csv
"""

import urllib.request
import urllib.error
import json
import csv
import time
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://api.fda.gov/drug/event.json"
DELAY = 0.35  # openFDA rate limit ~240/min without key
MAX_RETRIES = 5
RETRY_DELAY = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Raw Data" / "FDA Adverse Events"
OUTPUT_FILE = OUTPUT_DIR / "fda_adverse_event_counts.csv"


def fetch_url(url, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "DrugShortageProject/1.0")
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  HTTP {e.code}. Waiting {wait}s (retry {attempt+1}/{retries})...")
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f"  HTTP {e.code}: {e.reason}")
                if attempt == retries - 1:
                    return None
        except (urllib.error.URLError, Exception) as e:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  Error: {e}. Retry {attempt+1}/{retries}...")
            time.sleep(wait)
    return None


def _last_day_of_month(year, month):
    """Return YYYYMMDD string for the last day of the given month."""
    import calendar
    last = calendar.monthrange(year, month)[1]
    return f"{year}{month:02d}{last}"


def get_monthly_manufacturer_counts(year, month):
    """Get AE report counts by manufacturer for a given month.

    Uses the openFDA count endpoint with properly URL-encoded date range.
    The manufacturer field is nested under patient.drug.openfda.
    """
    import urllib.parse
    start = f"{year}{month:02d}01"
    end = _last_day_of_month(year, month)

    search = f"receivedate:[{start} TO {end}]"
    encoded_search = urllib.parse.quote(search)
    url = (f"{BASE_URL}?search={encoded_search}"
           f"&count=patient.drug.openfda.manufacturer_name.exact&limit=1000")

    data = fetch_url(url)
    if data is None or "results" not in data:
        return []

    results = []
    for entry in data["results"]:
        results.append({
            "manufacturer_name": entry.get("term", ""),
            "ae_count": entry.get("count", 0),
        })
    return results


def get_monthly_product_ndc_counts(year, month):
    """Get AE report counts by product NDC for a given month."""
    import urllib.parse
    start = f"{year}{month:02d}01"
    end = _last_day_of_month(year, month)

    search = f"receivedate:[{start} TO {end}]"
    encoded_search = urllib.parse.quote(search)
    url = (f"{BASE_URL}?search={encoded_search}"
           f"&count=patient.drug.openfda.product_ndc.exact&limit=1000")

    data = fetch_url(url)
    if data is None or "results" not in data:
        return []

    results = []
    for entry in data["results"]:
        results.append({
            "product_ndc": entry.get("term", ""),
            "ae_count": entry.get("count", 0),
        })
    return results


def main():
    print("FDA Adverse Event Report Counts Download")
    print(f"Output: {OUTPUT_FILE}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Download manufacturer-level counts per month
    all_mfr_rows = []

    # Query quarterly to reduce API calls (32 quarters vs 93 months)
    quarters = []
    for year in range(2018, 2026):
        for q in range(1, 5):
            start_month = (q - 1) * 3 + 1
            end_month = q * 3
            ym_start = f"{year}-{start_month:02d}"
            ym_end = f"{year}-{end_month:02d}"
            if ym_start > "2025-09":
                break
            quarters.append((year, start_month, end_month))

    for year, start_month, end_month in quarters:
        ym_label = f"{year}-Q{(start_month-1)//3+1}"
        print(f"  {ym_label}...", end=" ")

        # Get manufacturer counts for the quarter (use first month's function
        # but with the full quarter date range)
        import urllib.parse, calendar
        start = f"{year}{start_month:02d}01"
        end_day = calendar.monthrange(year, end_month)[1]
        end = f"{year}{end_month:02d}{end_day}"
        search = f"receivedate:[{start} TO {end}]"
        encoded_search = urllib.parse.quote(search)
        url = (f"{BASE_URL}?search={encoded_search}"
               f"&count=patient.drug.openfda.manufacturer_name.exact&limit=1000")
        data = fetch_url(url)

        if data and "results" in data:
            count = len(data["results"])
            for entry in data["results"]:
                # Assign to middle month of quarter
                mid_month = start_month + 1
                ym = f"{year}-{mid_month:02d}"
                all_mfr_rows.append({
                    "year_month": ym,
                    "manufacturer_name": entry.get("term", ""),
                    "ae_count": entry.get("count", 0),
                })
            print(f"{count} manufacturers")
        else:
            print("no data")

        time.sleep(DELAY)

    # Save manufacturer-level counts
    mfr_file = OUTPUT_DIR / "ae_counts_by_manufacturer.csv"
    if all_mfr_rows:
        with open(mfr_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["year_month", "manufacturer_name", "ae_count"])
            writer.writeheader()
            writer.writerows(all_mfr_rows)
        print(f"\nManufacturer-level: {len(all_mfr_rows):,} rows -> {mfr_file}")
    else:
        print("\nWARNING: No AE data downloaded.")

    # Summary
    print(f"\nCompleted in {time.time() - start_time:.1f}s")
    if all_mfr_rows:
        unique_months = len(set(r['year_month'] for r in all_mfr_rows))
        unique_mfrs = len(set(r['manufacturer_name'] for r in all_mfr_rows))
        print(f"Quarters covered: {unique_months}")
        print(f"Unique manufacturers: {unique_mfrs}")


if __name__ == "__main__":
    main()
