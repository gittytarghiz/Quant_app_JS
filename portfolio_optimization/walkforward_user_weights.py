# portfolio_optimization/walkforward_user_weights.py
import numpy as np
import pandas as pd
from data_management.monolith_loader import get_downloaded_series


def walkforward_user_weights(
    tickers,
    start,
    end,
    weights: dict | None = None,
    dtype="close",
    interval="1d",
    rebalance="monthly",
    costs=None,
    leverage=1.0,
    interest_rate=0.04,
):
    """
    Backtest using user-provided fixed weights (dict[ticker -> weight]).
    """
    if not weights:
        raise ValueError("User weights dictionary is required.")

    # Load data
    prices = (
        get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval)
        .dropna()
        .copy()
    )
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in data: {missing}")

    prices = prices[tickers].astype(float)
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")

    # Normalize user weights and apply leverage
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    if w.sum() == 0:
        raise ValueError("All weights are zero.")
    w = w / np.sum(np.abs(w)) * float(leverage)

    # Costs and financing
    bps = sum((costs or {}).get(k, 0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4
    daily_rate = interest_rate / 252
    tc = np.abs(w).sum() * bps
    excess_lev = leverage - 1.0

    # Compute pnl
    pnl = (rets @ w).astype(float)
    pnl.iloc[0] -= tc
    if excess_lev > 0:
        pnl -= excess_lev * daily_rate

    # Store weights snapshot
    W = pd.DataFrame([w], columns=tickers, index=[rets.index[0]])
    W.index.name = "date"

    details = {
        "weights_input": weights,
        "leverage": float(leverage),
        "costs_bps_total": float(bps),
        "interest_rate": float(interest_rate),
    }

    return {"weights": W, "pnl": pnl, "details": details}
