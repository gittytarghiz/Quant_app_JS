#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd, requests
from typing import Tuple, Optional

POLYGON_KEY = "6o6sPhbnZqeVbIOJJL9VmwStKN6mCfUj"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"

def polygon_download(ticker: str, start: str, end: str, interval: str, dtype: str) -> Tuple[bool, Optional[pd.Series]]:
    try:
        dtype = dtype.lower()
        if dtype not in {"open","high","low","close","volume"}: return False, None

        # interval -> (multiplier, timespan)
        if   interval.endswith("d"): mult, span = int(interval[:-1]), "day"
        elif interval.endswith("h"): mult, span = int(interval[:-1]), "hour"
        elif interval.endswith("m"): mult, span = int(interval[:-1]), "minute"
        else: return False, None

        # Free plan cap: ~2 years history
        today = pd.Timestamp.utcnow().normalize()
        min_start = (today - pd.DateOffset(years=2)).date()
        s = max(pd.to_datetime(start).date(), min_start)
        e = pd.to_datetime(end).date()

        url = BASE_URL.format(ticker=ticker.upper(), multiplier=mult, timespan=span, start=str(s), end=str(e))
        params = {"adjusted": "false", "sort": "asc", "limit": 50000, "apiKey": POLYGON_KEY}

        results = []
        while True:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            results += js.get("results", [])
            nxt = js.get("next_url")
            if not nxt: break
            url, params = nxt, None  # next_url already includes the query

        if not results: return False, None

        df = (pd.DataFrame(results)
              .assign(t=lambda x: pd.to_datetime(x["t"], unit="ms"))
              .set_index("t")
              .rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
              .sort_index())
        return (True, df[dtype]) if dtype in df.columns else (False, None)
    except Exception:
        return False, None
if __name__ == "__main__":
    ok, s = polygon_download("AAPL", "2024-01-01", "2025-01-01", "1d", "close")
    print("Success:", ok)
    if ok:
        print(s.head())