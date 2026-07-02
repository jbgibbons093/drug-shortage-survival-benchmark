"""
download_fda_recalls.py - Download FDA drug recall enforcement data from openFDA API.

Downloads all drug enforcement records from 2018-2025, flattens nested openfda
fields, deduplicates by recall_number, and saves as CSV.

Endpoint: https://api.fda.gov/drug/enforcement.json
Output: Raw Data/FDA Recalls/fda_drug_recalls_2018_2025.csv
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
BASE_URL = "https://api.fda.gov/drug/enforcement.json"
LIMIT = 1000
DELAY = 0.3
MAX_RETRIES = 5
RETRY_DELAY = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Raw Data" / "FDA Recalls"
OUTPUT_FILE = OUTPUT_DIR / "fda_drug_recalls_2018_2025.csv"

TOP_LEVEL_FIELDS = [
    "recall_number", "event_id", "product_description", "reason_for_recall",
    "classification", "status", "voluntary_mandated", "report_date",
    "recall_initiation_date", "center_classification_date", "termination_date",
    "initial_firm_recalling", "recalling_firm", "city", "state", "country",
    "distribution_pattern", "product_quantity", "code_info", "product_type",
]

OPENFDA_FIELDS = [
    "brand_name", "generic_name", "manufacturer_name", "product_ndc",
    "application_number", "substance_name", "spl_id", "package_ndc",
    "unii", "rxcui", "spl_set_id", "pharm_class_epc", "pharm_class_moa",
]

CSV_COLUMNS = TOP_LEVEL_FIELDS + ["openfda_" + f for f in OPENFDA_FIELDS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
                raise
        except (urllib.error.URLError, Exception) as e:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  Error: {e}. Waiting {wait}s (retry {attempt+1}/{retries})...")
            time.sleep(wait)
    print(f"  FAILED after {retries} retries: {url}")
    return None


def flatten_openfda(openfda_dict, field):
    if not openfda_dict or field not in openfda_dict:
        return ""
    val = openfda_dict[field]
    if isinstance(val, list):
        return " | ".join(str(v) for v in val)
    return str(val)


def extract_record(record):
    row = {}
    for field in TOP_LEVEL_FIELDS:
        row[field] = record.get(field, "")
    openfda = record.get("openfda", {})
    for field in OPENFDA_FIELDS:
        row["openfda_" + field] = flatten_openfda(openfda, field)
    return row


# ---------------------------------------------------------------------------
# Main download
# ---------------------------------------------------------------------------
def download_all_records():
    all_records = []

    for year in range(2018, 2026):
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        year_filter = f"report_date:[{start_date}+TO+{end_date}]"

        count_url = f"{BASE_URL}?search={year_filter}&limit=1"
        count_data = fetch_url(count_url)
        if count_data is None or "meta" not in count_data:
            print(f"Year {year}: No data or error. Skipping.")
            continue

        year_total = count_data["meta"]["results"]["total"]
        print(f"Year {year}: {year_total} records")
        if year_total == 0:
            continue

        # Split into half-years if > 25000
        if year_total > 25000:
            date_ranges = [(f"{year}0101", f"{year}0630"),
                           (f"{year}0701", f"{year}1231")]
        else:
            date_ranges = [(start_date, end_date)]

        for range_start, range_end in date_ranges:
            range_filter = f"report_date:[{range_start}+TO+{range_end}]"
            skip = 0

            while True:
                url = f"{BASE_URL}?search={range_filter}&limit={LIMIT}&skip={skip}"
                data = fetch_url(url)
                if data is None or "results" not in data:
                    break

                results = data["results"]
                if len(results) == 0:
                    break

                for record in results:
                    all_records.append(extract_record(record))

                skip += LIMIT
                range_total = data["meta"]["results"]["total"]
                if skip >= range_total or skip >= 25000:
                    break
                time.sleep(DELAY)

            time.sleep(DELAY)

    print(f"\nTotal records downloaded: {len(all_records)}")
    return all_records


def main():
    print("FDA Drug Recall Enforcement Data Download")
    print(f"Date range: 2018-01-01 to 2025-12-31")
    print(f"Output: {OUTPUT_FILE}\n")

    start_time = time.time()
    records = download_all_records()

    if len(records) == 0:
        print("ERROR: No records downloaded.")
        sys.exit(1)

    # Deduplicate by recall_number
    seen = set()
    unique = []
    for r in records:
        rn = r.get("recall_number", "")
        if rn and rn not in seen:
            seen.add(rn)
            unique.append(r)
        elif not rn:
            unique.append(r)
    if len(unique) < len(records):
        print(f"Removed {len(records) - len(unique)} duplicates.")
    records = unique

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to: {OUTPUT_FILE}")

    # Summary
    year_counts = {}
    for r in records:
        yr = r.get("report_date", "")[:4]
        if yr:
            year_counts[yr] = year_counts.get(yr, 0) + 1

    class_counts = {}
    for r in records:
        cl = r.get("classification", "Unknown")
        class_counts[cl] = class_counts.get(cl, 0) + 1

    has_ndc = sum(1 for r in records if r.get("openfda_product_ndc", ""))

    print(f"\nRecords by year: {dict(sorted(year_counts.items()))}")
    print(f"By classification: {dict(sorted(class_counts.items()))}")
    print(f"Records with NDC: {has_ndc} ({100*has_ndc/len(records):.1f}%)")
    print(f"Completed in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
