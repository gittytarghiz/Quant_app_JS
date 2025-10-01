#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd, requests
from typing import Tuple, Optional

EODHD_KEY = "68da71546b0c49.87033904"  # replace with your API key
BASE_URL = "https://eodhd.com/api/eod/{ticker}"

def eodhd_download(ticker: str, start: str, end: str, interval: str, dtype: str) -> Tuple[bool, Optional[pd.Series]]:
    try:
        dtype = dtype.lower()
        if dtype not in {"open","high","low","close","volume"}: return False, None
        params = {"from": start, "to": end, "period": interval, "fmt": "json", "api_token": EODHD_KEY}
        r = requests.get(BASE_URL.format(ticker=ticker.upper()), params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty: return False, None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns=str.lower).sort_index()
        return (True, df[dtype]) if dtype in df.columns else (False, None)
    except Exception:
        return False, None

if __name__ == "__main__":
    ok, s = eodhd_download("AAPL.US", "2024-01-01", "2025-01-01", "d", "close")
    print("Success:", ok)
    if ok: print(s.head())
