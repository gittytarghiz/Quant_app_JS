#!/usr/bin/env python3
from __future__ import annotations
import pathlib, pandas as pd
from typing import Dict, List, Tuple, Union

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
BENCH_FILE = DATA_DIR / "close" / "1d" / "SP500.parquet"

def partial_read(
    tickers: List[str],
    start: str,
    end: str,
    dtype: str,
    interval: str = "1d"
) -> Dict[str, Union[bool, Tuple[pd.Timestamp, pd.Timestamp]]]:
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    if not BENCH_FILE.exists():
        raise FileNotFoundError(f"Benchmark file {BENCH_FILE} not found.")
    ref = pd.read_parquet(BENCH_FILE); ref.index = pd.to_datetime(ref.index)
    ref_idx = ref.index[(ref.index >= start) & (ref.index < end)]
    if len(ref_idx) == 0:  # nothing to check
        return {t: True for t in tickers}

    out: Dict[str, Union[bool, Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for t in tickers:
        p = DATA_DIR / dtype / interval / f"{t.upper()}.parquet"
        if not p.exists():
            out[t] = (ref_idx[0], ref_idx[-1]); continue
        try:
            df = pd.read_parquet(p)
            s = pd.Series(df.iloc[:,0].values, index=pd.to_datetime(df.index)).reindex(ref_idx)
        except Exception:
            out[t] = (ref_idx[0], ref_idx[-1]); continue

        first_ok = s.first_valid_index()
        if first_ok is None:                # no data in window
            out[t] = (ref_idx[0], ref_idx[-1])
        elif first_ok > ref_idx[0]:         # enforce only lower edge
            out[t] = (ref_idx[0], first_ok) # end is exclusive in your downloader
        else:
            out[t] = True                   # ignore upper-edge gaps
    return out

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG"]
    out = partial_read(tickers, "2020-01-01", "2021-01-01", "close", "1d")
    for t, res in out.items():
        print(t, ":", res)
