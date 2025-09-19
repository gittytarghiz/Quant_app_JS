#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional

BASE_URL = "https://stooq.com/q/d/l/?s={ticker}&i={interval}"

def stooq_download(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    dtype: str
) -> Tuple[bool, Optional[pd.Series]]:
    """
    Download OHLCV data from Stooq for a single ticker.

    Parameters
    ----------
    ticker : str
        e.g. "AAPL" (U.S. tickers must end with .US → "AAPL.US").
    start, end : str
        ISO dates (YYYY-MM-DD).
    interval : str
        Only daily '1d' supported by Stooq.
    dtype : str
        One of {"open","high","low","close","volume"}.

    Returns
    -------
    (success, pd.Series or None)
    """
    dtype = dtype.lower()
    if dtype not in {"open","high","low","close","volume"}:
        return False, None
    if interval != "1d":
        return False, None

    url = BASE_URL.format(ticker=ticker.lower(), interval="d")
    try:
        df = pd.read_csv(url, parse_dates=["Date"])
    except Exception:
        return False, None
    if df is None or df.empty:
        return False, None

    df = df.rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    }).set_index("date").sort_index()

    # Trim to requested window
    df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

    if dtype not in df.columns:
        return False, None

    series = df[dtype]
    if isinstance(series, pd.DataFrame):  # ✅ ensure 1-D
        series = series.squeeze("columns")

    return True, series

if __name__ == "__main__":
    ok, s = stooq_download("aapl.us", "2000-01-01", "2025-01-01", "1d", "close")
    print("Success:", ok)
    if ok:
        print(s.head())
        print(s.tail())
        print("shape:", s.shape)
