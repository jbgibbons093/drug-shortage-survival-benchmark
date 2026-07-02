"""
download_asp_and_partb.py - pulls CMS Part B ASP pricing files and Medicare
Part B quarterly drug spending. Sibling to download_pricing_utilization.py.

Output:
  Raw Data/CMS ASP/<quarter>-asp-pricing-file.zip
  Raw Data/CMS ASP/<quarter>-asp-ndc-hcpcs-crosswalk.zip
  Raw Data/CMS Drug Spending/medicare_partb_spending_quarterly_through_2025q2.csv
"""

import sys
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).parent.parent
RAW = ROOT / "Raw Data"
ASP_DIR = RAW / "CMS ASP"
CMS_DIR = RAW / "CMS Drug Spending"

# All ASP file URLs discovered via probing (2020-Q1 through 2025-Q3 + crosswalks).
# When CMS publishes new files, append URLs here. The script is idempotent - it
# skips files already on disk.
ASP_URLS = [
    # 2020
    "https://www.cms.gov/files/zip/january-2020-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/january-2020-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/april-2020-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/april-2020-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/july-2020-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/october-2020-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/october-2020-asp-ndc-hcpcs-crosswalk.zip",
    # 2021
    "https://www.cms.gov/files/zip/january-2021-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/january-2021-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/april-2021-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/april-2021-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/july-2021-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/july-2021-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/october-2021-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/october-2021-asp-ndc-hcpcs-crosswalk.zip",
    # 2022
    "https://www.cms.gov/files/zip/january-2022-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/january-2022-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/april-2022-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/april-2022-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/july-2022-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/july-2022-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/october-2022-asp-pricing-file.zip",
    # 2023
    "https://www.cms.gov/files/zip/january-2023-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/january-2023-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/april-2023-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/july-2023-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/october-2023-asp-pricing-file.zip",
    # 2024 - note Jan 2024 uses "ndc-hcpcs-crosswalk" without "asp-" prefix
    "https://www.cms.gov/files/zip/january-2024-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/january-2024-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/april-2024-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/april-2024-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/july-2024-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/july-2024-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/october-2024-asp-pricing-file.zip",
    # 2025
    "https://www.cms.gov/files/zip/april-2025-asp-pricing-file.zip",
    "https://www.cms.gov/files/zip/april-2025-asp-ndc-hcpcs-crosswalk.zip",
    "https://www.cms.gov/files/zip/july-2025-asp-pricing-file.zip",
    # 2026
    "https://www.cms.gov/files/zip/january-2026-ndc-hcpcs-crosswalk.zip",
]

# Part B spending files (data.cms.gov catalog, datasets "Medicare Part B
# Spending by Drug" and "Medicare Quarterly Part B Spending by Drug").
# The annual file covers 2019-2023 and fills the training window that the
# quarterly file (2024+) cannot reach; with the one-year publication-lag
# alignment in 04d, the annual file supplies panel months 2020-2024.
PARTB_FILES = [
    ("https://data.cms.gov/sites/default/files/2026-01/"
     "bc1a311b-a338-4205-be8c-0afef9adc475/QDD_PTB_EXP_QTR_QTD202502_280126.csv",
     "medicare_partb_spending_quarterly_through_2025q2.csv"),
    ("https://data.cms.gov/sites/default/files/2026-04/"
     "qddptb_r2602_p06_v10_dqt2503_260424.csv",
     "medicare_partb_spending_quarterly_through_2025q3.csv"),
    ("https://data.cms.gov/sites/default/files/2025-05/"
     "f52d5fcd-8d93-481d-9173-6219813e4efb/DSD_PTB_RY25_P06_V10_DYT23_HCPCS-%20250430.csv",
     "medicare_partb_spending_by_drug_2019_2023.csv"),
]


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False  # already on disk
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return True
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code} on {url}")
        return False
    except Exception as e:
        print(f"    ERROR {type(e).__name__}: {e} on {url}")
        return False


def main() -> int:
    print("=" * 70)
    print("download_asp_and_partb.py")
    print("=" * 70)

    print(f"\n[1/2] ASP pricing + crosswalk files -> {ASP_DIR}")
    ASP_DIR.mkdir(parents=True, exist_ok=True)
    n_dl, n_skip, n_fail = 0, 0, 0

    def _go(url):
        dest = ASP_DIR / url.rsplit("/", 1)[-1]
        existed = dest.exists()
        ok = download(url, dest)
        if existed:
            return ("skip", url)
        return ("ok" if ok else "fail", url)

    with ThreadPoolExecutor(max_workers=6) as exe:
        for status, url in exe.map(_go, ASP_URLS):
            if status == "ok":
                n_dl += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
    print(f"  Downloaded: {n_dl}  Skipped (already present): {n_skip}  Failed: {n_fail}")

    print(f"\n[2/2] Medicare Part B spending files -> {CMS_DIR}")
    for url, name in PARTB_FILES:
        dest = CMS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Already on disk: {dest.name}")
        else:
            ok = download(url, dest)
            print(f"  {'Downloaded' if ok else 'Failed'}: {dest.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
