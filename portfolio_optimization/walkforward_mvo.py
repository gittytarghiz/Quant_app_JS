# portfolio_optimization/walkforward_mvo_shrink.py
from __future__ import annotations
import numpy as np, pandas as pd, cvxpy as cp
from sklearn.covariance import LedoitWolf
from data_management.monolith_loader import get_downloaded_series

def walkforward_mvo(
    tickers: list[str], start: str, end: str,
    dtype: str = "close", interval: str = "1d", rebalance: str = "monthly",
    min_weight: float = 0.0, max_weight: float = 1.0,
    min_obs: int = 60, leverage: float = 1.0,
    objective: str = "min_vol", objective_params: dict | None = None,
    **_: object,
) -> dict:
    """
    Walkforward MVO with Ledoit-Wolf shrinkage.
    Objectives: 'min_vol', 'mean_var', 'max_return'.
    """
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty: raise ValueError("No returns available.")
    freq = {"daily":"D","weekly":"W-THU","monthly":"ME","quarterly":"Q"}.get(rebalance.lower(),"ME")
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    n = len(tickers); lo, hi = float(min_weight), float(max_weight)
    lam = float((objective_params or {}).get("risk_aversion", 5.0))
    wv = cp.Variable(n); cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi]
    pnl = pd.Series(0.0, index=rets.index, dtype=float); W = []

    for i, t in enumerate(rbd):
        win = rets.loc[:t].tail(int(min_obs))
        if len(win) < 2:
            w = np.full(n, 1/n)
        else:
            mu = win.mean().values
            S = LedoitWolf().fit(win.values).covariance_
            obj = (cp.Minimize(cp.quad_form(wv,S)) if objective=="min_vol" else
                   cp.Maximize(mu @ wv) if objective=="max_return" else
                   cp.Maximize(mu @ wv - lam * cp.quad_form(wv,S)))
            cp.Problem(obj, cons).solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
            w = (wv.value if wv.value is not None else np.full(n,1/n)).ravel()
        w = float(leverage) * w
        nxt = rbd[i+1] if i+1<len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ w).astype(float)
        if not port.empty: pnl.loc[port.index] = port.values
        W.append((t, w))

    weights = pd.DataFrame({d: w for d,w in W}).T
    weights.index.name="date"; weights.columns=[c.upper() for c in rets.columns]
    return {"weights":weights, "pnl":pnl.loc[weights.index[0]:],
            "details":{"method":"mvo_ledoitwolf","objective":objective,"risk_aversion":lam,
                       "rebalance":rebalance,"min_obs":int(min_obs),"leverage":float(leverage),
                       "bounds":[lo,hi]}}
