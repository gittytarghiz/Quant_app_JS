#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
from typing import List


import time
try:
    from data_management.partial_reader import partial_read, DATA_DIR
    from data_management.yahoo_downloader import yahoo_download
    from data_management.polygon_dl import polygon_download
    from data_management.stooq_dl import stooq_download
    from data_management.alpha_dl import alpha_vantage_download
except ImportError:
    from partial_reader import partial_read, DATA_DIR
    from yahoo_downloader import yahoo_download
    from polygon_dl import polygon_download
    from stooq_dl import stooq_download
    from alpha_dl import alpha_vantage_download

def get_downloaded_series(tickers: List[str], start: str, end: str, dtype: str, interval: str = "1d") -> pd.DataFrame:
    status = partial_read(tickers, start, end, dtype, interval)
    for t, rng in {k:v for k,v in status.items() if v is not True}.items():
        s, e = rng
        s, e = s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")
        ok, series = False, None
        try:
            stooq_download(t.replace("-", ".") + ".US", s, e, interval, dtype)
            
        except Exception:
            ok, series = False, None
        if not ok or series is None or series.empty:
            try:
                
                ok, series = yahoo_download(t.replace("-", "."), s, e, interval, dtype)
            except Exception:
                ok, series = False, None
        if not ok or series is None or series.empty:
            try:
                ok, series = polygon_download(t.replace("-", "."), s, e, interval, dtype)
            except Exception:
                ok, series = False, None
        if not ok or series is None or series.empty:
            try:
                ok, series = ok, series = alpha_vantage_download(t.replace("-", "."), s, e, interval, dtype)
                
            except Exception:
                ok, series = False, None
        if not ok or series is None or series.empty:
            raise RuntimeError(f"Downloader failed for {t}")
        series = series.squeeze(); series.name = dtype; p = (DATA_DIR / dtype / interval / f"{t}.parquet"); p.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = pd.read_parquet(p); existing.index = pd.to_datetime(existing.index)
        except Exception:
            existing = pd.DataFrame()
        df = pd.concat([existing, series], axis=0).loc[~pd.concat([existing, series], axis=0).index.duplicated(keep="last")].sort_index()
        df.to_parquet(p, index=True)

    out = pd.DataFrame()
    for t in tickers:
        p = DATA_DIR / dtype / interval / f"{t}.parquet"
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        df.index = pd.to_datetime(df.index)
        w = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        out[t] = w.iloc[:, 0] if isinstance(w, pd.DataFrame) else w.squeeze()
    bad = [t for t, res in partial_read(tickers, start, end, dtype, interval).items() if res is not True]
    return out

if __name__ == "__main__":
    ts = ["AAPL","GOOG","MSFT","TSLA","AMZN","META","NVDA","BRK-B","JPM","UNH","V","PG","MA","HD","DIS","PYPL","BAC","VZ","ADBE","CMCSA","NFLX","INTC","T","PFE","KO","CSCO","PEP","XOM","ABT","WMT","CVX","NKE","MRK","WFC","LLY","COST","MCD","DHR","ACN","MDT"]
    ts = ["BAC"]
    for ticker in ts:
        try:
            df = get_downloaded_series([ticker], "1995-01-01", "2025-09-01", "close", "1d")
            print("Shape:", df.shape); print(df.head())
        except RuntimeError as e:
            print("❌", e)
