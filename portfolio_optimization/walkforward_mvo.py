# portfolio_optimization/walkforward_mvo_shrink.py
from __future__ import annotations
import numpy as np, pandas as pd, cvxpy as cp
from sklearn.covariance import LedoitWolf
from data_management.monolith_loader import get_downloaded_series

from sklearn.covariance import LedoitWolf
import cvxpy as cp
import numpy as np
import pandas as pd
from data_management.monolith_loader import get_downloaded_series

def walkforward_mvo(
    tickers: list[str],
    start: str,
    end: str,
    dtype: str = "close",
    interval: str = "1d",
    rebalance: str = "monthly",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    min_obs: int = 60,
    leverage: float = 1.0,
    objective: str = "min_vol",   # "min_vol", "max_return", "mean_var"
    objective_params: dict | None = None,
    costs: dict | None = None,
    **_: object,
) -> dict:
    """
    Walkforward Mean–Variance Optimization (Ledoit–Wolf Σ), aligned to EQW contract.
    Returns {"weights": W(DataFrame), "pnl": Series(from first weight date), "details": dict}
    """
    # --- data load & ticker order ---
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

    # --- rebalancing dates (unified) ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance.lower()]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    # --- costs (bps -> fraction) ---
    bps = sum((costs or {}).get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    lam = float((objective_params or {}).get("risk_aversion", 5.0))

    # pre-build CVX variables/constraints
    wv = cp.Variable(n)
    cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi]

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None

    # --- walk-forward loop ---
    for i, t in enumerate(rbd):
        win = rets.loc[:t].tail(int(min_obs))
        if len(win) < int(min_obs):
            w = np.full(n, 1.0 / n, dtype=float)
        else:
            mu = win.mean().values
            S = LedoitWolf().fit(win.values).covariance_
            if objective == "min_vol":
                obj = cp.Minimize(cp.quad_form(wv, S))
            elif objective == "max_return":
                obj = cp.Maximize(mu @ wv)
            else:  # "mean_var"
                obj = cp.Maximize(mu @ wv - lam * cp.quad_form(wv, S))
            cp.Problem(obj, cons).solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
            w = (wv.value if wv.value is not None else np.full(n, 1.0 / n)).ravel()

        # numerical safety: project to box+sum=1 then apply leverage
        w = np.clip(w, lo, hi)
        w = w / w.sum() if w.sum() != 0 else np.full(n, 1.0 / n)
        wL = float(leverage) * w

        nxt = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ wL).astype(float)

        # transaction cost on first bar of the hold
        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * bps
        if not port.empty:
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values

        weights_by_date[t] = wL.copy()
        w_prev = wL

    # --- outputs aligned with EQW ---
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers).astype(float)
        W.index.name = "date"
    else:
        W = pd.DataFrame(columns=tickers, dtype=float); W.index.name = "date"

    details = {
        "method": "mvo_ledoitwolf",
        "interval": interval,
        "rebalance": rebalance,
        "objective": objective,
        "risk_aversion": lam,
        "min_obs": int(min_obs),
        "leverage": float(leverage),
        "bounds": [lo, hi],
        "costs_bps_total": float(bps),
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
