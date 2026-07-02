"""
download_pricing_utilization.py - Download NADAC, SDUD, and CMS Part D Spending data.

Downloads:
  1. NADAC (National Average Drug Acquisition Cost) - weekly NDC-level prices
  2. SDUD (State Drug Utilization Data) - quarterly Medicaid Rx counts by state x NDC
  3. CMS Medicare Part D Spending by Drug - annual spending & utilization

Output directories:
  Raw Data/NADAC/
  Raw Data/SDUD/
  Raw Data/CMS Drug Spending/
"""

import os
import sys
import urllib.request
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "Raw Data"

NADAC_DIR = RAW_DATA / "NADAC"
SDUD_DIR = RAW_DATA / "SDUD"
CMS_DIR = RAW_DATA / "CMS Drug Spending"

for d in [NADAC_DIR, SDUD_DIR, CMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_file(url, dest_path, description=""):
    """Download a file with progress reporting."""
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  SKIP (already exists, {size_mb:.1f} MB): {dest_path.name}")
        return True

    print(f"  Downloading: {description or dest_path.name}")
    print(f"    URL: {url}")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (research data download)'
        })
        with urllib.request.urlopen(req, timeout=600) as response:
            total = response.headers.get('Content-Length')
            total = int(total) if total else None

            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            start_time = time.time()

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed = downloaded / (1024 * 1024 * max(elapsed, 0.1))

                    if total:
                        pct = downloaded / total * 100
                        total_mb = total / (1024 * 1024)
                        print(f"\r    {downloaded/(1024*1024):.1f} / {total_mb:.1f} MB "
                              f"({pct:.1f}%) - {speed:.1f} MB/s", end="", flush=True)
                    else:
                        print(f"\r    {downloaded/(1024*1024):.1f} MB - {speed:.1f} MB/s",
                              end="", flush=True)

            elapsed = time.time() - start_time
            size_mb = downloaded / (1024 * 1024)
            print(f"\n    Done: {size_mb:.1f} MB in {elapsed:.0f}s")
            return True

    except Exception as e:
        print(f"\n    ERROR: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    print("=" * 70)
    print("Downloading pricing and utilization data")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. NADAC - National Average Drug Acquisition Cost
    # ------------------------------------------------------------------
    print("\n[1/3] NADAC (National Average Drug Acquisition Cost)")
    print("-" * 50)

    # Dataset IDs on data.medicaid.gov (annual compilations)
    nadac_datasets = {
        'nadac_historical': {
            'url': 'https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.c933dc16-7de9-52b6-8971-4b75992673e0.csv',
            'file': 'nadac_historical_through_2021.csv',
            'desc': 'NADAC historical (through mid-2021, covers 2020 data)',
        },
        'nadac_2021': {
            'url': 'https://data.medicaid.gov/api/1/datastore/query/d5eaf378-dcef-5779-83de-acdd8347d68e/0/download?format=csv',
            'file': 'nadac_2021.csv',
            'desc': 'NADAC 2021 (full year)',
        },
        'nadac_2022': {
            'url': 'https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-2022.csv',
            'file': 'nadac_2022.csv',
            'desc': 'NADAC 2022',
        },
        'nadac_2023': {
            'url': 'https://data.medicaid.gov/api/1/datastore/query/4a00010a-132b-4e4d-a611-543c9521280f/0/download?format=csv',
            'file': 'nadac_2023.csv',
            'desc': 'NADAC 2023 (via API)',
        },
        'nadac_2024': {
            'url': 'https://data.medicaid.gov/api/1/datastore/query/99315a95-37ac-4eee-946a-3c523b4c481e/0/download?format=csv',
            'file': 'nadac_2024.csv',
            'desc': 'NADAC 2024 (via API)',
        },
        'nadac_2025': {
            'url': 'https://data.medicaid.gov/api/1/datastore/query/f38d0706-1239-442c-a3cc-40ef1b686ac0/0/download?format=csv',
            'file': 'nadac_2025.csv',
            'desc': 'NADAC 2025 (via API)',
        },
    }

    for key, info in nadac_datasets.items():
        download_file(info['url'], NADAC_DIR / info['file'], info['desc'])

    # ------------------------------------------------------------------
    # 2. SDUD - State Drug Utilization Data
    # ------------------------------------------------------------------
    print("\n[2/3] SDUD (State Drug Utilization Data)")
    print("-" * 50)

    sdud_datasets = {
        'sdud_2019': {
            'url': 'https://download.medicaid.gov/data/state-drug-utilization-data-2019.csv',
            'file': 'sdud_2019.csv',
            'desc': 'SDUD 2019 (for lookback)',
        },
        'sdud_2020': {
            'url': 'https://download.medicaid.gov/data/sdud-2020.csv',
            'file': 'sdud_2020.csv',
            'desc': 'SDUD 2020',
        },
        'sdud_2021': {
            'url': 'https://download.medicaid.gov/data/sdud-2021.csv',
            'file': 'sdud_2021.csv',
            'desc': 'SDUD 2021',
        },
        'sdud_2022': {
            'url': 'https://download.medicaid.gov/data/sdud-2022.csv',
            'file': 'sdud_2022.csv',
            'desc': 'SDUD 2022',
        },
        'sdud_2023': {
            'url': 'https://download.medicaid.gov/data/sdud-2023.csv',
            'file': 'sdud_2023.csv',
            'desc': 'SDUD 2023',
        },
        'sdud_2024': {
            'url': 'https://download.medicaid.gov/data/sdud-2024.csv',
            'file': 'sdud_2024.csv',
            'desc': 'SDUD 2024',
        },
        'sdud_2025': {
            'url': 'https://download.medicaid.gov/data/sdud-2025-updated-dec2025.csv',
            'file': 'sdud_2025.csv',
            'desc': 'SDUD 2025 (through Dec 2025)',
        },
    }

    for key, info in sdud_datasets.items():
        download_file(info['url'], SDUD_DIR / info['file'], info['desc'])

    # ------------------------------------------------------------------
    # 3. CMS Medicare Part D Spending by Drug
    # ------------------------------------------------------------------
    print("\n[3/3] CMS Medicare Part D Spending by Drug")
    print("-" * 50)

    cms_datasets = {
        'partd_spending': {
            'url': 'https://data.cms.gov/sites/default/files/2025-05/56d95a8b-138c-4b60-84a5-613fbab7197f/DSD_PTD_RY25_P04_V10_DY23_BGM.csv',
            'file': 'medicare_partd_spending_by_drug_2019_2023.csv',
            'desc': 'Medicare Part D Spending by Drug (2019-2023)',
        },
    }

    for key, info in cms_datasets.items():
        download_file(info['url'], CMS_DIR / info['file'], info['desc'])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)

    for label, directory in [("NADAC", NADAC_DIR), ("SDUD", SDUD_DIR),
                              ("CMS Drug Spending", CMS_DIR)]:
        files = list(directory.glob("*.csv"))
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"\n  {label}: {len(files)} files, {total_mb:.0f} MB total")
        for f in sorted(files):
            print(f"    {f.name}: {f.stat().st_size/(1024*1024):.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
