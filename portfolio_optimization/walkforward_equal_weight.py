# portfolio_optimization/walkforward_equal_weight.py
import numpy as np
import pandas as pd
from data_management.monolith_loader import get_downloaded_series


def walkforward_equal_weight(
    tickers,
    start,
    end,
    dtype="close",
    interval="1d",
    rebalance="monthly",
    costs=None,
    leverage=1.0,
    min_weight=0.0,
    max_weight=1.0,
    interest_rate = 0.04,
):
    # Load & enforce column order == tickers (so rets @ w is correct)
    prices = (
        get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval)
        .dropna()
        .copy()
    )
    daily_rate = interest_rate / 252
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in data: {missing}")
    prices = prices[tickers].astype(float)

    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")

    # Rebalance dates
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    # Costs (bps)
    bps = sum((costs or {}).get(k, 0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4

    # Start weights (equal-weight, clipped, normalized, levered)
    n = len(tickers)
    w = np.full(n, 1.0 / n, dtype=float)
    w = np.clip(w, float(min_weight), float(max_weight))
    w = w / w.sum() * float(leverage)

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    prev = np.zeros(n, dtype=float)

    for i, t in enumerate(rbd):
        nxt = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        # Portfolio returns over [t, nxt]
        port = (rets.loc[t:nxt] @ w).astype(float)
        if not port.empty:
            tc = (np.abs(w).sum() if i == 0 else np.abs(w - prev).sum()) * bps
            port.iloc[0] -= tc

            # Financing penalty
            excess_lev = leverage - 1.0
            if excess_lev > 0:
                port -= excess_lev * daily_rate

        pnl.loc[port.index] = port.values
        # Store weights exactly at the rebalance timestamp t (aligned to tickers order)
        weights_by_date[t] = w.copy()
        prev = w

    # Build weights DataFrame: rows=rebalances, cols=tickers, values=float
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers)
        W.index.name = "date"
        W = W.astype(float)
    else:
        W = pd.DataFrame(columns=tickers, dtype=float)
        W.index.name = "date"

    details = {
        "rebalance": rebalance,
        "leverage": float(leverage),
        "bounds": [float(min_weight), float(max_weight)],
        "costs_bps_total": float(bps),
    }

    # Keep pnl from first weight date onward (nice alignment for frontend)
    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
