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
        rebalance: str = "monthly",  # 'daily','weekly','monthly','quarterly'
        costs: dict = None,          # {'bps':x,'slippage_bps':y,'spread_bps':z}
        leverage: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
) -> dict:

    costs = costs or {}
    bps_total = sum(costs.get(k, 0.0) for k in ["bps", "slippage_bps", "spread_bps"])

    # 1. Load prices from monolith
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available for given inputs (check tickers/dates/interval or data source).")

    # 2. Build rebalance schedule (weekly = Thursday to match Yahoo)
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq_map[rebalance])
    rbd = [d for d in rbd if d in rets.index]

    n = len(tickers)

    # build equal weights under simple [min,max] box with sum=1 constraint
    def _eq_with_box(n: int, wmin: float, wmax: float) -> np.ndarray:
        # feasibility checks
        wmin = float(max(0.0, wmin))
        wmax = float(min(1.0, wmax))
        if wmin > wmax:
            wmin, wmax = wmax, wmin
        if n * wmin - 1.0 > 1e-12:
            raise ValueError("Infeasible: n * min_weight > 1")
        if n * wmax + 1e-12 < 1.0:
            raise ValueError("Infeasible: n * max_weight < 1")
        w = np.full(n, 1.0 / n)
        w = np.clip(w, wmin, wmax)
        s = w.sum()
        if abs(s - 1.0) < 1e-12:
            return w
        # water-filling to adjust towards sum=1
        target = 1.0
        for _ in range(8):  # a few iterations sufficient for equal start
            s = w.sum()
            delta = target - s
            if abs(delta) < 1e-9:
                break
            if delta > 0:
                slack = np.maximum(0.0, wmax - w)
            else:
                slack = np.maximum(0.0, w - wmin)
            tot = slack.sum()
            if tot <= 1e-18:
                break
            adj = (delta * (slack / tot))
            w = np.clip(w + adj, wmin, wmax)
        # final normalization within bounds if tiny drift
        s = w.sum()
        if s > 0:
            w = w * (target / s)
            w = np.clip(w, wmin, wmax)
        return w
    weights_list = []
    pnl = pd.Series(0.0, index=rets.index, dtype=float)

    w_prev = None  # track levered weights for turnover consistency
    for i, t in enumerate(rbd):
        w_base = _eq_with_box(n, float(min_weight), float(max_weight))
        w = float(leverage) * w_base
        turnover_cost = (np.abs(w).sum() if w_prev is None else np.abs(w - w_prev).sum()) * (bps_total / 1e4)
        next_t = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        hold = rets.loc[t:next_t]
        port = (hold @ w).astype(float)
        if len(port) > 0:
            port.iloc[0] -= turnover_cost
        pnl.loc[port.index] = port.values
        weights_list.append((t, w))
        w_prev = w

    weights = pd.DataFrame({d: w for d, w in weights_list}).T
    weights.index.name = "date"
    weights.columns = tickers

    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {"rebalance_dates": weights.index.tolist(),
                    "config": {"rebalance": rebalance, "costs_bps_total": bps_total, "leverage": leverage,
                                "min_weight": float(min_weight), "max_weight": float(max_weight)}}
    }
