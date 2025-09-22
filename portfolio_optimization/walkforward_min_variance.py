# portfolio_optimization/walkforward_min_variance_shrink.py
import numpy as np, pandas as pd
from pathlib import Path
import sys
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

def walkforward_min_variance(
    tickers: list[str],
    start: str,
    end: str,
    dtype: str = "close",
    interval: str = "1d",
    rebalance: str = "monthly",   # 'daily','weekly','monthly','quarterly'
    costs: dict | None = None,    # {'bps':x,'slippage_bps':y,'spread_bps':z}
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    min_obs: int = 60,
    leverage: float = 1.0
) -> dict:
    """
    Walk-forward Minimum-Variance with Ledoit-Wolf shrinkage covariance.
    Objective: minimize w' Σ w
    Constraints: sum(w)=1, min_weight <= w_i <= max_weight
    """
    costs = costs or {}
    bps_total = float(sum(costs.get(k, 0.0) for k in ("bps","slippage_bps","spread_bps"))) / 1e4

    # --- data ---
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")

    # --- rebalance dates ---
    freq_map = {"daily":"D","weekly":"W-FRI","monthly":"ME","quarterly":"Q"}
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq_map[rebalance])
    rbd = [d for d in rbd if d in rets.index]
    if not rbd: rbd = [rets.index[0]]

    n = len(tickers); lo, hi = float(min_weight), float(max_weight)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    W, w_prev = [], None

    for i, t in enumerate(rbd):
        win = rets.loc[:t].tail(int(min_obs))
        if len(win) < min_obs:
            w = np.full(n, 1/n)
        else:
            S = LedoitWolf().fit(win.values).covariance_
            try:
                import cvxpy as cp
                wv = cp.Variable(n)
                cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi]
                obj = cp.Minimize(cp.quad_form(wv, S))
                cp.Problem(obj, cons).solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
                w = (wv.value if wv.value is not None else np.full(n,1/n)).ravel()
            except Exception:
                w = np.full(n, 1/n)
        w = np.clip(w, lo, hi); w /= w.sum()
        wL = float(leverage) * w

        nxt = rbd[i+1] if i+1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ wL).astype(float)
        if not port.empty:
            tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * bps_total
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values
        W.append((t, wL)); w_prev = wL

    weights = pd.DataFrame({d:w for d,w in W}).T
    weights.index.name="date"; weights.columns=tickers
    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {
            "method": "min_variance_ledoitwolf",
            "rebalance": rebalance,
            "interval": interval,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "costs_bps_total": bps_total,
            "covariance": "ledoitwolf",
            "min_obs": min_obs,
            "leverage": leverage
        }
    }
