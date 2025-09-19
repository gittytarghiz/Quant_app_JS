#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
import requests
from typing import Tuple, Optional

API_KEY = "94DDW7HDSNFQIMU9"
BASE_URL = "https://www.alphavantage.co/query"

def alpha_vantage_download(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    dtype: str
) -> Tuple[bool, Optional[pd.Series]]:
    """
    Download OHLCV data from Alpha Vantage (free API).

    Parameters
    ----------
    ticker : str
        Example: "AAPL".
    start, end : str
        ISO format dates (YYYY-MM-DD).
    interval : str
        Currently only supports "1d".
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

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker.upper(),
        "outputsize": "full",
        "apikey": API_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return False, None

    if "Time Series (Daily)" not in data:
        return False, None

    # Parse into DataFrame
    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index", dtype=float
    )
    df.index = pd.to_datetime(df.index)
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
        }
    ).sort_index()

    # Trim to requested window
    df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

    if dtype not in df.columns:
        return False, None

    series = df[dtype]
    if isinstance(series, pd.DataFrame):
        series = series.squeeze("columns")

    return True, series


if __name__ == "__main__":
    ok, s = alpha_vantage_download("AAPL", "2000-01-01", "2025-01-01", "1d", "close")
    print("Success:", ok)
    if ok:
        print(s.head())
        print(s.tail())
        print("shape:", s.shape)
