import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

# Optional lower-precision mode to reduce memory (set NSGA2_FLOAT32=1)
_USE_F32 = os.getenv("NSGA2_FLOAT32", "0") == "1"
_DTYPE = np.float32 if _USE_F32 else np.float64


# ---------- utils ----------
def _ann_factor(interval: str) -> float:
    key = str(interval).lower()
    return {"1d": 252.0, "d": 252.0, "daily": 252.0,
            "1wk": 52.0, "w": 52.0, "weekly": 52.0,
            "1mo": 12.0, "m": 12.0, "monthly": 12.0}.get(key, 252.0)


def _project_box_sum1(w: np.ndarray, lo: float, hi: float, iters: int = 5) -> np.ndarray:
    w = np.asarray(w, float)
    n = w.size
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (lo,hi) vs sum=1.")
    s = w.sum()
    w = np.clip((w / (s if s > 1e-12 else 1.0)), lo, hi)
    for _ in range(iters):
        s = w.sum()
        if abs(s - 1.0) < 1e-12:
            break
        if s > 1.0:
            free = w > lo + 1e-15
            if np.any(free):
                w[free] -= (s - 1.0) * (w[free] / np.maximum(w[free].sum(), 1e-12))
        else:
            free = w < hi - 1e-15
            if np.any(free):
                w[free] += (1.0 - s) * (w[free] / np.maximum(w[free].sum(), 1e-12))
        w = np.clip(w, lo, hi)
    s = w.sum()
    return w / (s if s > 0 else 1.0)

# Optional Cython acceleration (reuse common projection)
try:
    from portfolio_optimization._common_fast import prop_box_sum1 as _cy_prop  # type: ignore
    _project_box_sum1 = _cy_prop  # type: ignore
except Exception:
    pass


def _downside_std(R: np.ndarray) -> np.ndarray:
    neg = np.minimum(R, 0.0)
    return np.sqrt(np.maximum((neg * neg).mean(axis=1), 1e-18))


def _max_drawdown(paths: np.ndarray) -> np.ndarray:
    mx = np.maximum.accumulate(paths, axis=1)
    dd = 1.0 - np.divide(paths, np.maximum(mx, 1e-12))
    return dd.max(axis=1)


def _cvar(R: np.ndarray, alpha: float) -> np.ndarray:
    L = -R
    m = max(1, int(np.floor(alpha * R.shape[1])))
    part = np.partition(L, -m, axis=1)[:, -m:]
    return part.mean(axis=1)


def _metrics_bank(W: np.ndarray, window: pd.DataFrame, interval: str, w_prev: np.ndarray | None,
                  alpha: float, rf: float, mar: float) -> dict:
    af = _ann_factor(interval)
    P = window.values.astype(_DTYPE, copy=False)  # (T,n)
    R = W @ P.T                      # (k,T)
    Rp = R - (rf / af)
    mu_p = Rp.mean(axis=1)
    sd_p = Rp.std(axis=1, ddof=0)
    sharpe = np.divide(mu_p, np.maximum(sd_p, 1e-12)) * np.sqrt(af)

    mu = R.mean(axis=1)
    sd = R.std(axis=1, ddof=0)
    ann_ret = af * mu
    ann_vol = np.sqrt(af) * sd

    Re = R - (mar / af)
    dsd = _downside_std(Re)
    sortino = np.divide(Re.mean(axis=1) * af, np.sqrt(af) * np.maximum(dsd, 1e-12))

    paths = np.maximum(1.0 + np.cumsum(R, axis=1), 1e-9)
    mdd = _max_drawdown(paths)
    calmar = np.divide(ann_ret, np.maximum(mdd, 1e-12))
    cvar = _cvar(R, alpha)
    turnover = (np.abs(W).sum(axis=1) if w_prev is None else np.abs(W - w_prev[None, :]).sum(axis=1))
    rot = np.divide(ann_ret, np.maximum(turnover, 1e-12))
    rod = np.divide(ann_ret, np.maximum(mdd, 1e-12))

    return {
        "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "cvar": cvar, "max_dd": mdd, "turnover": turnover, "ret_over_turnover": rot, "ret_over_drawdown": rod
    }


def _scores_for_objs(metrics: dict, obj_list: list[str]) -> np.ndarray:
    # NSGA-II assumes minimization; flip sign when we "maximize".
    S = []
    for name in obj_list:
        v = metrics[name]
        if name in {"ann_vol", "cvar", "max_dd", "turnover"}:
            S.append(v)          # minimize
        else:
            S.append(-v)         # maximize -> minimize negative
    return np.vstack(S).T        # (k,m)


def _nds_rank_and_fronts(F: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    N = F.shape[0]
    S = [set() for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].add(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    ranks = np.full(N, np.inf)
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            ranks[p] = i
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return ranks, fronts[:-1]


def _crowding(F: np.ndarray, idxs: list[int]) -> np.ndarray:
    if not idxs:
        return np.array([])
    M = F.shape[1]
    d = np.zeros(len(idxs))
    if len(idxs) <= 2:
        d[:] = 1e9
        return d
    X = F[idxs]
    for m in range(M):
        order = np.argsort(X[:, m])
        d[order[0]] = d[order[-1]] = 1e9
        fmin, fmax = X[order[0], m], X[order[-1], m]
        rng = (fmax - fmin) if fmax > fmin else 1.0
        for j in range(1, len(idxs) - 1):
            d[order[j]] += (X[order[j + 1], m] - X[order[j - 1], m]) / rng
    return d


def _tournament(parents: np.ndarray, F: np.ndarray, ranks: np.ndarray, crowds,
                rng: np.random.Generator, k: int = 2) -> np.ndarray:
    sel = np.empty_like(parents)
    N = parents.shape[0]
    for i in range(N):
        cand = rng.choice(N, size=k, replace=False)
        a, b = cand[0], cand[1]
        ra, rb = ranks[a], ranks[b]
        if ra < rb:
            w = a
        elif rb < ra:
            w = b
        else:
            try:
                ca = crowds[a]; cb = crowds[b]
            except Exception:
                ca = crowds.get(a, 0.0); cb = crowds.get(b, 0.0)  # type: ignore[attr-defined]
            w = a if ca > cb else b
        sel[i] = parents[w]
    return sel


def _sbx(p1: np.ndarray, p2: np.ndarray, eta_c: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta_c + 1)), (2 * (1 - u)) ** (-1.0 / (eta_c + 1)))
    c1 = 0.5 * ((p1 + p2) - beta * (p2 - p1))
    c2 = 0.5 * ((p1 + p2) + beta * (p2 - p1))
    return c1, c2


def _poly_mut(x: np.ndarray, eta_m: float, rate: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    for i in range(y.size):
        if rng.random() > rate:
            continue
        u = rng.random()
        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
        y[i] = y[i] + delta
    return y


# ---------- Walkforward NSGA-II ----------
def walkforward_nsga2(
    tickers: list[str], start: str, end: str,
    dtype: str = "close", interval: str = "1d",
    rebalance: str = "monthly", costs: dict | None = None,
    min_weight: float = 0.0, max_weight: float = 1.0, min_obs: int = 60,
    leverage: float = 1.0,
    # simplified: number of random candidates per rebalance
    tries: int = 48, seed: int = 42,
) -> dict:
    """
    Tiny, fast replacement for NSGA-II used in quick testing.
    Samples `tries` random weight vectors per rebalance, evaluates their
    annualized Sharpe over the lookback window and picks the best.
    """
    rng = np.random.default_rng(seed)
    costs = costs or {}
    bps_total = float(sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")))

    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")
    freq_map = {"daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q"}
    freq = freq_map.get(rebalance, "M")
    rbd = [d for d in pd.date_range(rets.index[0], rets.index[-1], freq=freq) if d in rets.index] or [rets.index[0]]

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_list: list[tuple[pd.Timestamp, np.ndarray]] = []
    w_prev = None

    for i, t in enumerate(rbd):
        window = rets.loc[:t]
        if len(window) < min_obs:
            w_opt = np.full(n, 1.0 / n)
        else:
            cand = rng.dirichlet(np.full(n, 1.0), size=tries)
            cand = np.clip(cand, lo, hi)
            cand = np.array([_project_box_sum1(c, lo, hi) for c in cand])
            P = (cand * float(leverage)) @ window.T.values
            mu = P.mean(axis=1)
            sd = P.std(axis=1, ddof=0)
            ann_sharpe = (mu / np.maximum(sd, 1e-12)) * np.sqrt(_ann_factor(interval))
            idx = int(np.nanargmax(ann_sharpe))
            w_opt = cand[idx]

        wL = float(leverage) * w_opt
        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * (bps_total / 1e4)
        next_t = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        hold = rets.loc[t:next_t]
        port = (hold @ wL).astype(float)
        if len(port) > 0:
            port.iloc[0] -= tc
        pnl.loc[port.index] = port.values
        weights_list.append((t, wL))
        w_prev = wL

    weights = pd.DataFrame({d: w for d, w in weights_list}).T
    weights.index.name = "date"
    weights.columns = tickers

    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {"rebalance_dates": weights.index.tolist(), "config": {"method": "nsga2_simple", "tries": tries}},
    }
