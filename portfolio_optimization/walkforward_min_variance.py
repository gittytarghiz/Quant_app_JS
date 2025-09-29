# portfolio_optimization/walkforward_min_variance_shrink.py
import numpy as np, pandas as pd
from pathlib import Path
import sys
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series
from sklearn.covariance import LedoitWolf

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
    Walk-forward Minimum-Variance (Ledoit–Wolf Σ) aligned to EQW contract.
    Returns { "weights": DataFrame(index='date', cols=tickers),
              "pnl": Series(from first weight date),
              "details": dict }
    """
    costs = costs or {}
    bps = float(sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps"))) / 1e4

    # --- data load & enforce ticker order ---
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

    # --- unified rebalancing ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None

    for i, t in enumerate(rbd):
        win = rets.loc[:t].tail(int(min_obs))
        if len(win) < int(min_obs):
            w = np.full(n, 1.0 / n, dtype=float)
        else:
            S = LedoitWolf().fit(win.values).covariance_
            # QP: min w'Σw  s.t. sum w = 1, lo <= w <= hi
            try:
                import cvxpy as cp
                wv = cp.Variable(n)
                cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi]
                prob = cp.Problem(cp.Minimize(cp.quad_form(wv, S)), cons)
                prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
                w = (wv.value if wv.value is not None else np.full(n, 1.0 / n)).ravel()
            except Exception:
                w = np.full(n, 1.0 / n, dtype=float)

        # project -> leverage
        w = np.clip(w, lo, hi)
        w = w / w.sum()
        wL = float(leverage) * w

        nxt = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ wL).astype(float)

        # TC on first bar
        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * bps
        if not port.empty:
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values

        weights_by_date[t] = wL.copy()  # <-- fix: call copy()

        w_prev = wL

    # --- outputs (EQW contract) ---
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers).astype(float)
        W.index.name = "date"
    else:
        W = pd.DataFrame(columns=tickers, dtype=float); W.index.name = "date"

    details = {
        "method": "min_variance_ledoitwolf",
        "rebalance": rebalance,
        "interval": interval,
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "bounds": [lo, hi],
        "costs_bps_total": float(bps),
        "covariance": "ledoitwolf",
        "min_obs": int(min_obs),
        "leverage": float(leverage),
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
