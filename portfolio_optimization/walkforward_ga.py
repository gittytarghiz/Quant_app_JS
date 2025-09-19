# portfolio_optimization/walkforward_ga_fast.py
import numpy as np, pandas as pd
from pathlib import Path; import sys
from scipy.optimize import differential_evolution as de

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

# ---------- utils ----------
def _stabilize_cov(S):
    S = np.asarray(S, float); S = 0.5*(S+S.T)
    n = S.shape[0]; tr = np.trace(S)/max(n,1)
    eps = 1e-8*(tr if np.isfinite(tr) and tr>0 else 1.0)
    return S + eps*np.eye(n)

def _prop_box_sum1(w, lo, hi, iters=6):
    """Fast projection onto {sum=1, lo<=w<=hi} via proportional rescaling."""
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

# Optional Cython acceleration
try:
    from portfolio_optimization._common_fast import prop_box_sum1 as _prop_box_sum1  # type: ignore
except Exception:
    pass

# ---------- objective bank ----------
def _ann_factor(interval: str):
    return {"1d":252, "D":252, "daily":252, "1wk":52, "W":52, "weekly":52, "1mo":12, "M":12, "monthly":12}.get(interval, 252)

def _max_drawdown(r):
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    dd = (eq/peak) - 1.0
    return float(-np.min(dd)) if dd.size else 0.0

def _cvar(r, alpha=0.05):
    if r.size == 0: return 0.0
    loss = -r
    k = max(1, int(np.ceil(alpha * loss.size)))
    idx = np.argpartition(loss, -k)[-k:]
    return float(loss[idx].mean())

def _sortino(r, mar=0.0, ann=252):
    if r.size == 0: return 0.0
    excess = r - (mar/ann)
    downside = excess[excess < 0.0]
    ds = np.sqrt(np.mean(np.square(downside))) if downside.size else 1e-12
    mean = float(excess.mean())
    return float((mean / (ds + 1e-12)) * np.sqrt(ann))  # annualized (matches Sharpe style)

def _calmar(r, ann=252):
    if r.size == 0: return 0.0
    ar = (1.0 + r).prod()**(ann/max(1,len(r))) - 1.0
    mdd = _max_drawdown(r)
    return float(ar / (mdd + 1e-12))

def _sharpe(r, rf=0.0, ann=252):
    if r.size == 0: return 0.0
    rp = r - (rf/ann)
    mu, sd = float(rp.mean()), float(rp.std(ddof=0))
    return float((mu / (sd + 1e-12)) * np.sqrt(ann))

def _ann_return(r, ann=252):
    if r.size == 0: return 0.0
    return float((1.0 + r).prod()**(ann/max(1,len(r))) - 1.0)

def _fitness_from_name(name, w, mu, S, rts, params, w_prev=None, ann=252):
    r = rts @ w if rts is not None else None  # (T,)
    if name == "min_vol":         return float(np.sqrt(max(w @ S @ w, 1e-18)))
    if name == "max_return":      return -float(w @ mu)
    if name in ("target_return_min_vol","target_min_vol"):
        target = float(params.get("target", 0.0))
        v = float(w @ S @ w); penalty = max(0.0, target - float(w @ mu))
        return np.sqrt(max(v,1e-18)) + 1e3*penalty**2
    if name == "sharpe":          return -_sharpe(np.asarray(r), rf=float(params.get("rf",0.0)), ann=ann)
    if name == "sortino":         return -_sortino(np.asarray(r), mar=float(params.get("mar",0.0)), ann=ann)
    if name == "calmar":          return -_calmar(np.asarray(r), ann=ann)
    if name == "cvar":            return _cvar(np.asarray(r), alpha=float(params.get("alpha",0.05)))
    if name in ("kelly","max_growth","log_utility"):
        rr = np.asarray(r)
        v = float(np.log(np.maximum(1e-8, 1.0 + rr)).mean()) if rr.size else 0.0
        return -v
    if name in ("diversification","diversification_ratio","max_diversification"):
        sig = np.sqrt(np.clip(np.diag(S), 1e-18, None))
        num = float(np.dot(w, sig))
        den = float(np.sqrt(max(w @ S @ w, 1e-18)))
        dr = num / (den + 1e-18)
        return -dr
    if name == "return_to_drawdown":
        ar = _ann_return(np.asarray(r), ann=ann); mdd = _max_drawdown(np.asarray(r))
        return -(ar / (mdd + 1e-12))
    if name == "return_to_turnover":
        to = 0.05 if w_prev is None else float(np.abs(w - w_prev).sum())
        ar = _ann_return(np.asarray(r), ann=ann)
        return -(ar / (to + 1e-12))
    return -_sharpe(np.asarray(r), rf=float(params.get("rf",0.0)), ann=ann)  # fallback

def _make_obj(name, mu, S, win_returns, lo, hi, params, w_prev, ann):
    name = (name or "sharpe").lower()
    R = np.ascontiguousarray(win_returns, dtype=float)
    def _obj(y):
        w = _prop_box_sum1(y, lo, hi)
        return _fitness_from_name(name, w, mu, S, R, params, w_prev=w_prev, ann=ann)
    return _obj

# ---------- main ----------
def walkforward_ga_fast(
    tickers, start, end, dtype="close", interval="1d",
    rebalance="monthly", costs=None,
    min_weight=0.0, max_weight=1.0, min_obs=60,
    leverage: float = 1.0,
    objective="sharpe", objective_params=None,
    de_maxiter=40, de_popsize=25, de_tol=0.01, de_mutation=(0.5,1.0),
    de_recombination=0.7, de_strategy="best1bin", de_seed=42,
    de_workers=None, de_polish=False
):
    costs = costs or {}
    bps_total = float(sum(costs.get(k,0.0) for k in ("bps","slippage_bps","spread_bps")))
    objective_params = objective_params or {}

    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty: raise ValueError("No returns available.")

    freq = {"daily":"D","weekly":"W-FRI","monthly":"ME","quarterly":"Q"}[rebalance]
    rbd = [d for d in pd.date_range(rets.index[0], rets.index[-1], freq=freq) if d in rets.index] or [rets.index[0]]
    n = len(tickers); lo=float(min_weight); hi=float(max_weight)
    if n*lo > 1+1e-12 or n*hi < 1-1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    bounds=[(lo,hi)]*n; ann = _ann_factor(interval)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_list=[]; w_prev=None  # track levered weights

    for i,t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < min_obs:
            w = _prop_box_sum1(np.full(n,1.0/n), lo, hi)
        else:
            mu = win.mean().values; S = _stabilize_cov(win.cov().values)
            obj = _make_obj(objective, mu, S, win.values, lo, hi, objective_params, w_prev, ann)
            res = de(obj, bounds=bounds, strategy=de_strategy, maxiter=int(de_maxiter), popsize=int(de_popsize),
                     tol=float(de_tol), mutation=de_mutation, recombination=float(de_recombination),
                     seed=int(de_seed), polish=bool(de_polish), updating="deferred",
                     workers=(de_workers if de_workers is not None else 1), disp=False)
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
        "details": {"rebalance_dates": weights.index.tolist(),
            "config":{"method":"ga_scipy_de_fast_prop","rebalance":rebalance,"interval":interval,
                      "objective":objective,"objective_params":objective_params,
                      "min_weight":min_weight,"max_weight":max_weight,"costs_bps_total":bps_total,
                      "min_obs":min_obs,"de_maxiter":de_maxiter,"de_popsize":de_popsize,"de_tol":de_tol,
                      "de_mutation":de_mutation,"de_recombination":de_recombination,"de_strategy":de_strategy,
                      "de_seed":de_seed,"de_workers":de_workers,"de_polish":de_polish,
                      "leverage": leverage}}
    }
