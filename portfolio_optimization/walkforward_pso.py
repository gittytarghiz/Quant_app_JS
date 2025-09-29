# portfolio_optimization/walkforward_pso.py
import numpy as np, pandas as pd
from pathlib import Path; import sys
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent; sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from data_management.monolith_loader import get_downloaded_series
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from data_management.monolith_loader import get_downloaded_series

def walkforward_pso(
    tickers,
    start,
    end,
    dtype="close",
    interval="1d",
    rebalance="monthly",
    costs=None,
    min_weight=0.0,
    max_weight=1.0,
    min_obs=60,
    leverage=1.0,
    particles=60,
    iters=40,
    c1=1.5,
    c2=1.5,
    w_inertia=0.7,
    seed=42,
    objective="sharpe",
    **_: object,
) -> dict:
    """
    Walk-forward Particle Swarm Optimizer (PSO) with Ledoit–Wolf shrinkage.
    Returns: {"weights": DataFrame(index='date', cols=tickers), "pnl": Series, "details": dict}
    """
    rng = np.random.default_rng(seed)

    # --- load & enforce ticker order ---
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

    # --- rebalance dates ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    # --- costs ---
    bps = sum((costs or {}).get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None

    # --- walk-forward loop ---
    for i, t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < int(min_obs):
            w_best = np.full(n, 1.0 / n, dtype=float)
        else:
            X = rng.uniform(lo, hi, size=(particles, n))
            X /= X.sum(1, keepdims=True)
            V = rng.normal(0, 0.05, (particles, n))

            def eval_sharpe(Wmat):
                R = win.values @ Wmat.T  # (T, particles)
                m, s = R.mean(0), R.std(0) + 1e-12
                return -(m / s) * np.sqrt(252)  # minimize

            fbest = eval_sharpe(X)
            pbest = X.copy()
            gbest = X[fbest.argmin()]

            for _ in range(int(iters)):
                r1, r2 = rng.random(X.shape), rng.random(X.shape)
                V = w_inertia * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
                X = np.clip(X + V, lo, hi)
                X /= X.sum(1, keepdims=True)
                f = eval_sharpe(X)
                mask = f < fbest
                pbest[mask] = X[mask]
                fbest[mask] = f[mask]
                gbest = pbest[fbest.argmin()]

            w_best = gbest

        # leverage + TC like EQW
        wL = float(leverage) * w_best
        nxt = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ wL).astype(float)

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
        W = pd.DataFrame(columns=tickers, dtype=float)
        W.index.name = "date"

    details = {
        "method": "pso",
        "objective": objective,
        "particles": int(particles),
        "iters": int(iters),
        "c1": float(c1),
        "c2": float(c2),
        "w_inertia": float(w_inertia),
        "seed": int(seed),
        "rebalance": rebalance,
        "interval": interval,
        "min_obs": int(min_obs),
        "leverage": float(leverage),
        "bounds": [lo, hi],
        "costs_bps_total": float(bps),
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
