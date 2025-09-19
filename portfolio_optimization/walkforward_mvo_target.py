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
    Min-variance subject to expected return >= target_return.
    Uses cvxpy when available; otherwise falls back to DE with a penalty.
    """
    costs = costs or {}
    bps_total = float(sum(costs.get(k,0.0) for k in ("bps","slippage_bps","spread_bps")))

    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty: raise ValueError("No returns available.")

    freq = {"daily":"D","weekly":"W-FRI","monthly":"ME","quarterly":"Q"}[rebalance]
    rbd = [d for d in pd.date_range(rets.index[0], rets.index[-1], freq=freq) if d in rets.index] or [rets.index[0]]
    n = len(tickers); lo=float(min_weight); hi=float(max_weight)
    if n*lo > 1+1e-12 or n*hi < 1-1e-12: raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_list=[]; w_prev=None

    for i, t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < min_obs:
            w = _prop_box_sum1(np.full(n,1.0/n), lo, hi)
        else:
            mu = win.mean().values
            S_sample  = _stabilize_cov(win.cov().values)
            S = S_sample
            if isinstance(cov_estimator, str):
                ce = cov_estimator.lower()
                if ce in {"diag","diagonal"}:
                    S = np.diag(np.diag(S_sample))
                elif ce in {"lw","ledoit","ledoitwolf"}:
                    try:
                        from sklearn.covariance import LedoitWolf  # type: ignore
                        lw = LedoitWolf().fit(win.values)
                        S = lw.covariance_.astype(float)
                    except Exception:
                        d = np.diag(np.diag(S_sample)); S = 0.9*S_sample + 0.1*d
            if cov_shrinkage > 0.0:
                d = np.diag(np.diag(S))
                S = (1.0 - float(cov_shrinkage)) * S + float(cov_shrinkage) * d

            if _HAS_CVXPY:
                wv = cp.Variable(n)
                cons = [cp.sum(wv) == 1, wv >= lo, wv <= hi, mu @ wv >= float(target_return)]
                prob = cp.Problem(cp.Minimize(cp.quad_form(wv, S)), cons)
                prob.solve(solver=cp.OSQP, verbose=False)
                if wv.value is None:
                    # infeasible target -> push to max_return within box
                    base = np.zeros(n); base[np.argmax(mu)] = 1.0
                    w = _prop_box_sum1(base, lo, hi)
                else:
                    w = _prop_box_sum1(np.array(wv.value).ravel(), lo, hi)
            else:
                if not _HAS_DE:
                    raise RuntimeError("SciPy DE not available and cvxpy missing for target-return MVO.")
                bounds = [(lo,hi)]*n
                def obj(y):
                    w = _prop_box_sum1(y, lo, hi)
                    v = float(w @ S @ w)
                    pen = max(0.0, float(target_return) - float(np.dot(mu, w)))
                    return np.sqrt(max(v,1e-18)) + 1e3*pen**2
                res = de(obj, bounds=bounds,
                         strategy="best1bin", maxiter=int(de_maxiter), popsize=int(de_popsize),
                         tol=float(de_tol), mutation=de_mutation, recombination=float(de_recombination),
                         seed=int(de_seed), polish=bool(de_polish), updating="deferred",
                         workers=int(de_workers), disp=False)
                w = _prop_box_sum1(res.x, lo, hi)

        wL = float(leverage) * w
        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL-w_prev).sum()) * (bps_total/1e4)
        next_t = rbd[i+1] if i+1<len(rbd) else rets.index[-1]
        hold = rets.loc[t:next_t]; port=(hold@wL).astype(float)
        if len(port)>0: port.iloc[0]-=tc
        pnl.loc[port.index]=port.values; weights_list.append((t,wL)); w_prev=wL

    weights = pd.DataFrame({d:w for d,w in weights_list}).T
    weights.index.name="date"; weights.columns=tickers
    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {
            "rebalance_dates": weights.index.tolist(),
            "config": {
                "method": "mvo_target_return",
                "rebalance": rebalance, "interval": interval,
                "target_return": float(target_return),
                "min_weight": min_weight, "max_weight": max_weight,
                "costs_bps_total": bps_total, "min_obs": min_obs,
                "has_cvxpy": bool(_HAS_CVXPY), "has_de": bool(_HAS_DE),
                "leverage": leverage, "cov_shrinkage": float(cov_shrinkage),
            }
        }
    }
