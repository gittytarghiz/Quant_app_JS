# portfolio_optimization/walkforward_equal_weight.py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series


def walkforward_equal_weight(
    tickers: list[str],
    start: str,
    end: str,
    dtype: str = "close",
    interval: str = "1d",
    rebalance: str = "monthly",   # 'daily','weekly','monthly','quarterly'
    costs: dict = None,           # {'bps':x,'slippage_bps':y,'spread_bps':z}
    leverage: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> dict:
    """
    Equal-weight portfolio walkforward backtest.

    Returns
    -------
    dict with:
        weights : pd.DataFrame  (rebalance dates × tickers)
        pnl     : pd.Series     (daily portfolio returns)
        details : dict          (rebalance schedule + config)
    """
    costs = costs or {}
    bps_total = sum(costs.get(k, 0.0) for k in ["bps", "slippage_bps", "spread_bps"])

    # 1. Load returns
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")

    # 2. Build rebalance schedule
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq_map[rebalance])
    rbd = rbd.intersection(rets.index)

    n = len(tickers)
    base_w = np.full(n, 1.0 / n)
    base_w = np.clip(base_w, min_weight, max_weight)
    base_w /= base_w.sum()  # ensure sum=1
    base_w *= leverage

    weights, pnl = {}, pd.Series(0.0, index=rets.index, dtype=float)
    prev_w = np.zeros(n)

    for i, t in enumerate(rbd):
        w = base_w.copy()
        # turnover cost = change in weights × bps
        tc = np.abs(w - prev_w).sum() * (bps_total / 1e4) if i else np.abs(w).sum() * (bps_total / 1e4)
        nxt = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ w).astype(float)
        if not port.empty:
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values
        weights[t] = w
        prev_w = w

    weights_df = pd.DataFrame(weights).T
    weights_df.index.name, weights_df.columns = "date", tickers

    return {
        "weights": weights_df,
        "pnl": pnl.loc[weights_df.index[0]:],
        "details": {
            "rebalance_dates": weights_df.index.tolist(),
            "config": {
                "rebalance": rebalance,
                "costs_bps_total": bps_total,
                "leverage": leverage,
                "min_weight": float(min_weight),
                "max_weight": float(max_weight),
            },
        },
    }
