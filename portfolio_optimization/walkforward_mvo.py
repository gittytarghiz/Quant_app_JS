from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from data_management.monolith_loader import get_downloaded_series


def walkforward_mvo(
    tickers, start, end,
    dtype="close", interval="1d", rebalance="monthly",
    costs=None, min_weight=0.0, max_weight=1.0, min_obs=60,
    leverage: float = 1.0,
    objective="min_vol", objective_params=None,
    **_: object,
):
    # 1) data
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")

    # 2) schedule
    freq = {"daily": "D", "weekly": "W-FRI", "monthly": "ME", "quarterly": "Q"}.get(str(rebalance).lower(), "ME")
    rbd = [d for d in pd.date_range(rets.index[0], rets.index[-1], freq=freq) if d in rets.index] or [rets.index[0]]

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    lo, hi = max(0.0, lo), min(1.0, hi)
    bps = float((costs or {}).get("bps", 0.0)) / 1e4
    name = str(objective).lower()
    lam = float((objective_params or {}).get("risk_aversion", 5.0)) if isinstance(objective_params, dict) else 5.0

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    W, w_prev = [], None

    for i, t in enumerate(rbd):
        win = rets.loc[:t].tail(int(min_obs))
        if len(win) < 2:
            w = np.full(n, 1.0 / n)
        else:
            mu = win.mean().values.astype(float)
            S = win.cov().values.astype(float)
            # CVXPY solve
            wv = cp.Variable(n)
            cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi]
            if name == "max_return":
                prob = cp.Problem(cp.Maximize(mu @ wv), cons)
            elif name == "min_vol":
                prob = cp.Problem(cp.Minimize(cp.quad_form(wv, S)), cons)
            else:
                # mean-variance with risk_aversion lambda
                prob = cp.Problem(cp.Maximize(mu @ wv - lam * cp.quad_form(wv, S)), cons)
            prob.solve(solver=cp.OSQP, verbose=False)
            w = np.array(wv.value).ravel() if wv.value is not None else np.full(n, 1.0 / n)
        w = float(leverage) * w

        next_t = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        block = rets.loc[t:next_t]
        port = (block @ w).astype(float)
        if len(port) > 0:
            tc = (np.abs(w).sum() if w_prev is None else np.abs(w - w_prev).sum()) * bps
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values
        W.append((t, w))
        w_prev = w

    weights = pd.DataFrame({d: w for d, w in W}).T
    weights.index.name = "date"; weights.columns = [c.upper() for c in rets.columns]
    return {"weights": weights, "pnl": pnl.loc[weights.index[0]:], "details": {"method": "mvo_cvxpy_slim", "objective": name, "risk_aversion": lam}}
