# portfolio_optimization/walkforward_mvo_target.py
import numpy as np, pandas as pd
from pathlib import Path; import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

try:
    from scipy.optimize import differential_evolution as de
    _HAS_DE = True
except Exception:
    _HAS_DE = False

def _stabilize_cov(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    S = 0.5*(S+S.T)
    n = S.shape[0]
    tr = np.trace(S)/max(n,1)
    eps = 1e-8*(tr if np.isfinite(tr) and tr>0 else 1.0)
    return S + eps*np.eye(n)

def _prop_box_sum1(w, lo, hi, iters=6):
    w = np.asarray(w, float); n = w.size
    if n*lo > 1+1e-12 or n*hi < 1-1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")
    s = w.sum(); w = (w/s) if s>1e-12 else np.full(n, 1.0/n)
    w = np.clip(w, lo, hi)
    for _ in range(iters):
        s = w.sum()
        if abs(s-1.0) < 1e-12: break
        if s > 1.0:
            free = w > lo+1e-15
            if not free.any(): break
            w[free] -= (s-1.0) * (w[free]/w[free].sum())
        else:
            free = w < hi-1e-15
            if not free.any(): break
            w[free] += (1.0-s) * (w[free]/w[free].sum())
        w = np.clip(w, lo, hi)
    return w / w.sum()

try:
    from portfolio_optimization._common_fast import prop_box_sum1 as _prop_box_sum1  # type: ignore
except Exception:
    pass
def walkforward_mvo_target_return(
    tickers, start, end,
    dtype="close", interval="1d", rebalance="monthly",
    costs=None, min_weight=0.0, max_weight=1.0, min_obs=60,
    leverage: float = 1.0,
    target_return: float = 0.0,
    cov_shrinkage: float = 0.0,
    cov_estimator: str | None = None,
    # DE fallback params
    de_maxiter=35, de_popsize=20, de_tol=0.01, de_mutation=(0.5,1.0),
    de_recombination=0.7, de_seed=42, de_workers=1, de_polish=False,
):
    """
    Min-variance s.t. E[r] >= target_return.
    Matches EQW contract:
      returns {"weights": W(index='date', cols=tickers),
               "pnl": pnl[from first weight date],
               "details": {...}}
    Uses cvxpy if available, else Differential Evolution with penalty.
    """
    costs = costs or {}
    bps = sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e-4 / 1e4  # keep clarity
    # the line above is equivalent to /1e4; keeping it explicit to avoid mistakes:
    bps = sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4

    # --- data (enforce ticker order) ---
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

    # --- unified mechanics ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    n = len(tickers)
    lo = float(min_weight); hi = float(max_weight)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None

    for i, t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < int(min_obs):
            w = _prop_box_sum1(np.full(n, 1.0 / n), lo, hi)
        else:
            mu = win.mean().values
            S_sample = _stabilize_cov(win.cov().values)
            S = S_sample

            if isinstance(cov_estimator, str):
                ce = cov_estimator.lower()
                if ce in {"diag", "diagonal"}:
                    S = np.diag(np.diag(S_sample))
                elif ce in {"lw", "ledoit", "ledoitwolf"}:
                    try:
                        from sklearn.covariance import LedoitWolf  # type: ignore
                        S = LedoitWolf().fit(win.values).covariance_.astype(float)
                    except Exception:
                        d = np.diag(np.diag(S_sample)); S = 0.9 * S_sample + 0.1 * d

            if float(cov_shrinkage) > 0.0:
                lam = float(cov_shrinkage)
                d = np.diag(np.diag(S))
                S = (1.0 - lam) * S + lam * d

            if _HAS_CVXPY:
                wv = cp.Variable(n)
                cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi, mu @ wv >= float(target_return)]
                prob = cp.Problem(cp.Minimize(cp.quad_form(wv, S)), cons)
                prob.solve(solver=cp.OSQP, verbose=False)
                if wv.value is None:
                    base = np.zeros(n); base[int(np.argmax(mu))] = 1.0
                    w = _prop_box_sum1(base, lo, hi)
                else:
                    w = _prop_box_sum1(np.array(wv.value).ravel(), lo, hi)
            else:
                if not _HAS_DE:
                    raise RuntimeError("SciPy DE not available and cvxpy missing for target-return MVO.")
                bounds = [(lo, hi)] * n

                def obj(y):
                    w_ = _prop_box_sum1(y, lo, hi)
                    v = float(w_ @ S @ w_)
                    pen = max(0.0, float(target_return) - float(np.dot(mu, w_)))
                    return np.sqrt(max(v, 1e-18)) + 1e3 * pen**2

                res = de(
                    obj, bounds=bounds, strategy="best1bin", maxiter=int(de_maxiter),
                    popsize=int(de_popsize), tol=float(de_tol), mutation=de_mutation,
                    recombination=float(de_recombination), seed=int(de_seed),
                    polish=bool(de_polish), updating="deferred",
                    workers=int(de_workers), disp=False
                )
                w = _prop_box_sum1(res.x, lo, hi)

        # leverage after projection; transaction cost on first bar
        wL = float(leverage) * w
        next_t = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:next_t] @ wL).astype(float)

        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * bps
        if not port.empty:
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values

        weights_by_date[t] = wL.copy()
        w_prev = wL

    # --- outputs (EQW contract) ---
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers).astype(float)
        W.index.name = "date"
    else:
        W = pd.DataFrame(columns=tickers, dtype=float); W.index.name = "date"

    details = {
        "method": "mvo_target_return",
        "rebalance": rebalance,
        "interval": interval,
        "target_return": float(target_return),
        "cov_shrinkage": float(cov_shrinkage),
        "cov_estimator": cov_estimator,
        "bounds": [lo, hi],
        "min_obs": int(min_obs),
        "leverage": float(leverage),
        "costs_bps_total": float(bps),
        "de_fallback": {
            "maxiter": int(de_maxiter), "popsize": int(de_popsize), "tol": float(de_tol),
            "mutation": de_mutation, "recombination": float(de_recombination),
            "seed": int(de_seed), "workers": int(de_workers), "polish": bool(de_polish),
        },
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
