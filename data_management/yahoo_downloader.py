#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import Tuple, Optional


def yahoo_download(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    dtype: str
) -> Tuple[bool, Optional[pd.DataFrame]]:
    
    dtype = dtype.lower()
    if dtype not in {"open", "high", "low", "close", "volume"}:
        return False, None

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return False, None

    if df is None or df.empty:
        return False, None

    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    if dtype not in df.columns:
        return False, None
    return True, df[[dtype]]


if __name__ == "__main__":
    ok, s = yahoo_download("MSFT", "2020-01-01", "2021-01-01", "1d", "close")
    print("Success:", ok)
    if ok:
        print(s.head())
