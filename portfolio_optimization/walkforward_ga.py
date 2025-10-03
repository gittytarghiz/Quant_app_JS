# portfolio_optimization/walkforward_ga_fast.py
import numpy as np, pandas as pd
from pathlib import Path; import sys
from scipy.optimize import differential_evolution as de
from sklearn.covariance import LedoitWolf  # <- added

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
    return float((mean / (ds + 1e-12)) * np.sqrt(ann))

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
    r = rts @ w if rts is not None else None
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
    return -_sharpe(np.asarray(r), rf=float(params.get("rf",0.0)), ann=ann)

def _make_obj(name, mu, S, win_returns, lo, hi, params, w_prev, ann):
    name = (name or "sharpe").lower()
    R = np.ascontiguousarray(win_returns, dtype=float)
    def _obj(y):
        w = _prop_box_sum1(y, lo, hi)
        return _fitness_from_name(name, w, mu, S, R, params, w_prev=w_prev, ann=ann)
    return _obj
# ---------- main ----------
def walkforward_ga_fast(
    tickers,
    start,
    end,
    dtype="close",
    interval="1d",
    rebalance="monthly",
    costs=None,
    leverage: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    interest_rate = 0.04,
    min_obs: int = 60,
    objective: str = "sharpe",
    objective_params=None,
    de_maxiter=40, de_popsize=25, de_tol=0.01, de_mutation=(0.5, 1.0),
    de_recombination=0.7, de_strategy="best1bin", de_seed=42,
    de_workers=None, de_polish=False
):
    """
    GA walk-forward optimizer aligned to EQW contract.
    Returns: {"weights": DataFrame(index='date', cols=tickers), "pnl": Series, "details": dict}
    """
    # --- data load & ticker order enforcement ---
    prices = (
        get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval)
        .dropna()
        .copy()
    )
    daily_rate = interest_rate / 252

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in data: {missing}")
    prices = prices[tickers].astype(float)

    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")

    # --- unified mechanics (match EQW) ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    bps = sum((costs or {}).get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4
    n = len(tickers); lo = float(min_weight); hi = float(max_weight)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")
    bounds = [(lo, hi)] * n
    ann = _ann_factor(interval)
    objective_params = objective_params or {}

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None

    # --- walk-forward loop ---
    for i, t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < int(min_obs):
            w = _prop_box_sum1(np.full(n, 1.0 / n), lo, hi)
        else:
            mu = win.mean().values
            S = LedoitWolf().fit(win.values).covariance_
            obj = _make_obj(objective, mu, S, win.values, lo, hi, objective_params, w_prev, ann)
            res = de(
                obj, bounds=bounds, strategy=de_strategy, maxiter=int(de_maxiter),
                popsize=int(de_popsize), tol=float(de_tol), mutation=de_mutation,
                recombination=float(de_recombination), seed=int(de_seed),
                polish=bool(de_polish), updating="deferred",
                workers=(de_workers if de_workers is not None else 1), disp=False
            )
            w = _prop_box_sum1(res.x, lo, hi)

        # Apply leverage AFTER projection; charge TC on first bar
        wL = float(leverage) * w
        next_t = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:next_t] @ wL).astype(float)

        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * bps
        if not port.empty:
            port.iloc[0] -= tc
            # --- Financing penalty ---
            excess_lev = np.abs(wL).sum() - 1.0
            if excess_lev > 0:
                daily_rate = interest_rate / 252  # interest_rate arg, e.g. 0.04
                port -= excess_lev * daily_rate
    # --------------------------
            pnl.loc[port.index] = port.values

        weights_by_date[t] = wL.copy()
        w_prev = wL

    # --- outputs exactly like EQW ---
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers)
        W.index.name = "date"
        W = W.astype(float)
    else:
        W = pd.DataFrame(columns=tickers, dtype=float); W.index.name = "date"

    details = {
        "rebalance": rebalance,
        "interval": interval,
        "objective": objective,
        "objective_params": objective_params,
        "leverage": float(leverage),
        "bounds": [lo, hi],
        "costs_bps_total": float(bps),
        "min_obs": int(min_obs),
        "ga": {
            "maxiter": int(de_maxiter), "popsize": int(de_popsize), "tol": float(de_tol),
            "mutation": de_mutation, "recombination": float(de_recombination),
            "strategy": de_strategy, "seed": int(de_seed),
            "workers": de_workers, "polish": bool(de_polish),
        },
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}

