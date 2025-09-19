#!/usr/bin/env python3
"""Download data for all tickers listed in `tickers_nasdaq.txt` and `tickers_nyse.txt`.

Place the two files inside `data_management/data/` (one symbol per line). This script will
loop over every ticker and call `monolith_loader.get_downloaded_series` to update parquet files.

Note: This will attempt to download many symbols and may take a long time and hit rate limits.
"""
from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

try:
    from data_management.monolith_loader import get_downloaded_series
except Exception:
    from monolith_loader import get_downloaded_series


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NAS_PATH = DATA_DIR / "tickers_nasdaq.txt"
NYSE_PATH = DATA_DIR / "tickers_nyse.txt"


def read_list(p: Path) -> list[str]:
    if not p.exists():
        return []
    with p.open() as f:
        return [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]


def main():
    nas = read_list(NAS_PATH)
    nyse = read_list(NYSE_PATH)
    if not nas and not nyse:
        print("No ticker files found. Please add 'tickers_nasdaq.txt' and/or 'tickers_nyse.txt' to data_management/data/")
        return 1

    tickers = nas + nyse
    tickers = [t for t in dict.fromkeys(tickers) if t]
    start = "1970-01-01"
    end = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"Downloading {len(tickers)} tickers from {start} to {end} (dtype=close, interval=1d)")

    for i, t in enumerate(tickers, 1):
        if i > 1000:
            try:
                print(f"[{i}/{len(tickers)}] {t}")
                _ = get_downloaded_series([t], start, end, "close", "1d")
                print("  OK")
            except Exception as e:
                print(f"  FAIL: {e}")
        # polite delay
            time.sleep(0.001)


if __name__ == "__main__":
    main()
