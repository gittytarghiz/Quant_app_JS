# portfolio_optimization/walkforward_equal_weight.py
import numpy as np, pandas as pd
from data_management.monolith_loader import get_downloaded_series

def walkforward_equal_weight(
    tickers, start, end,
    dtype="close", interval="1d", rebalance="monthly",
    costs=None, leverage=1.0, min_weight=0.0, max_weight=1.0,
):
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty: raise ValueError("No returns available.")

    freq = {"daily":"D","weekly":"W-THU","monthly":"ME","quarterly":"Q"}[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    n = len(tickers); bps = sum((costs or {}).get(k,0) for k in ("bps","slippage_bps","spread_bps"))/1e4
    w = np.full(n, 1/n); w = np.clip(w, min_weight, max_weight); w /= w.sum(); w *= leverage

    pnl = pd.Series(0.0, index=rets.index); weights = {}; prev = np.zeros(n)
    for i,t in enumerate(rbd):
        nxt = rbd[i+1] if i+1<len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ w).astype(float)
        if not port.empty:
            tc = (np.abs(w).sum() if i==0 else np.abs(w-prev).sum())*bps
            port.iloc[0] -= tc; pnl.loc[port.index] = port.values
        weights[t], prev = w, w

    W = pd.DataFrame(weights).T; W.index.name="date"; W.columns=tickers
    return {"weights":W,"pnl":pnl.loc[W.index[0]:],
            "details":{"rebalance":rebalance,"leverage":leverage,
                       "bounds":[float(min_weight),float(max_weight)],
                       "costs_bps_total":bps}}
