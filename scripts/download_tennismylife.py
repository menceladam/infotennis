"""Download historical match data from stats.tennismylife.org.

Source: https://stats.tennismylife.org/tennis-match-database
MIT licensed, free to use, updated daily. Covers ATP tour, ATP Challenger
tour, and ATP qualifying, one CSV per season.
"""
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://stats.tennismylife.org/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; senorsanj-tennis-model/1.0)"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LEVELS = {
    "tour": {"url_fmt": "{base}/{year}.csv", "dir": "tour"},
    "challenger": {"url_fmt": "{base}/{year}_challenger.csv", "dir": "challenger"},
    "quali": {"url_fmt": "{base}/atp_quali/{year}_atp_quali.csv", "dir": "quali"},
}


def download_year(level: str, year: int) -> bool:
    cfg = LEVELS[level]
    url = cfg["url_fmt"].format(base=BASE_URL, year=year)
    out_path = DATA_DIR / cfg["dir"] / f"{year}.csv"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return True


def main(start_year: int, end_year: int):
    for level in LEVELS:
        print(f"\n=== {level} ===")
        for year in range(start_year, end_year + 1):
            ok = download_year(level, year)
            status = "OK" if ok else "missing"
            print(f"  {year}: {status}")
            time.sleep(0.2)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    main(start, end)
