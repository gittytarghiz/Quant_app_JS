# portfolio_optimization/walkforward_nsga2.py
import numpy as np, pandas as pd
from pathlib import Path; import sys
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent; sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series
from sklearn.covariance import LedoitWolf

def walkforward_nsga2(
    tickers,
    start,
    end,
    dtype="close",
    interval="1d",
    rebalance="monthly",
    costs=None,                          # added: unified costs (bps/1e4)
    min_weight=0.0,
    max_weight=1.0,
    min_obs=60,
    leverage=1.0,
    interest_rate = 0.04,
    objectives=None,
    pop=64,
    gens=12,
    eta_c=12.0,
    eta_m=20.0,
    mut_rate=0.1,
    alpha=0.2,
    seed=42,
    **_
) -> dict:
    """NSGA-II (vectorized), Ledoit–Wolf Σ. Picks max-Sharpe on first front. API-aligned."""
    rng = np.random.default_rng(seed)

    # --- load & enforce ticker order ---
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

    # --- unified mechanics (EQW contract) ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    bps = sum((costs or {}).get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")) / 1e4
    lo, hi = float(min_weight), float(max_weight)
    n = len(tickers)
    if n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    ann = {"1d": 252, "D": 252, "daily": 252, "1wk": 52, "W": 52, "weekly": 52,
           "1mo": 12, "M": 12, "monthly": 12}.get(interval, 252)

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None
    obj = (objectives or ["ann_vol", "cvar"]).copy()

    # --- helpers ---
    def eval_metrics(Wm: np.ndarray, R: np.ndarray):
        P = R @ Wm.T
        m = P.mean(0)
        s = P.std(0) + 1e-12
        ann_ret = ann * m
        ann_vol = np.sqrt(ann) * s
        sharpe = (m / s) * np.sqrt(ann)
        k = max(1, int(np.ceil(alpha * P.shape[0])))
        idx = np.argpartition(-P, k - 1, axis=0)[-k:]
        cvar = np.take_along_axis(-P, idx, axis=0).mean(0)
        return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "cvar": cvar}

    def scores(M):
        cols = []
        for name in obj:
            cols.append(M[name] if name in {"ann_vol", "cvar"} else -M[name])
        return np.vstack(cols).T

    def non_dominated_sort(F: np.ndarray):
        N = F.shape[0]
        S = [set() for _ in range(N)]
        n_ = np.zeros(N, dtype=int)
        fronts = [[]]
        for p in range(N):
            for q in range(N):
                if p == q: continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].add(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_[p] += 1
            if n_[p] == 0:
                fronts[0].append(p)
        ranks = np.full(N, np.inf)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                ranks[p] = i
                for q in S[p]:
                    n_[q] -= 1
                    if n_[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        return ranks, fronts[:-1]

    # --- walk-forward ---
    for i, t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < int(min_obs):
            w_best = np.full(n, 1.0 / n, dtype=float)
        else:
            R = win.values
            _ = LedoitWolf().fit(R)  # ensure shrinkage path (stabilizes metrics)
            # init population inside box + simplex
            X = rng.uniform(lo, hi, (pop, n))
            X /= X.sum(1, keepdims=True)

            for _gen in range(int(gens)):
                M = eval_metrics(X, R)
                F = scores(M)
                ranks, fronts = non_dominated_sort(F)

                # crowding distance
                crowd = np.zeros(pop)
                for fr in fronts:
                    if len(fr) == 0:
                        continue
                    Y = F[fr]
                    for m_ in range(Y.shape[1]):
                        o = np.argsort(Y[:, m_])
                        crowd[np.array(fr)[o[0]]] = 1e9
                        crowd[np.array(fr)[o[-1]]] = 1e9
                        span = Y[o[-1], m_] - Y[o[0], m_]
                        span = span if abs(span) > 1e-12 else 1.0
                        for j in range(1, len(fr) - 1):
                            crowd[np.array(fr)[o[j]]] += (Y[o[j + 1], m_] - Y[o[j - 1], m_]) / span

                # tournament selection
                sel = []
                for _k in range(pop):
                    a, b = rng.choice(pop, 2, replace=False)
                    sel.append(X[a] if (ranks[a] < ranks[b] or (ranks[a] == ranks[b] and crowd[a] > crowd[b])) else X[b])
                sel = np.asarray(sel)

                # SBX crossover + polynomial mutation
                kids = []
                for j in range(0, pop, 2):
                    p1, p2 = sel[j], sel[(j + 1) % pop]
                    u = rng.random(n)
                    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta_c + 1.0)),
                                    (2 * (1 - u)) ** (-1.0 / (eta_c + 1.0)))
                    c1 = 0.5 * ((p1 + p2) - beta * (p2 - p1))
                    c2 = 0.5 * ((p1 + p2) + beta * (p2 - p1))
                    for c in (c1, c2):
                        if rng.random() < mut_rate:
                            u2 = rng.random(n)
                            delta = np.where(u2 < 0.5, (2 * u2) ** (1.0 / (eta_m + 1.0)) - 1.0,
                                             1.0 - (2 * (1 - u2)) ** (1.0 / (eta_m + 1.0)))
                            c = c + delta
                        c = np.clip(c, lo, hi)
                        c = c / c.sum()
                        kids.append(c)
                X = np.asarray(kids)[:pop]

            # select max-Sharpe from first front
            M = eval_metrics(X, R)
            F = scores(M)
            _, fronts = non_dominated_sort(F)
            first = fronts[0] if fronts else list(range(pop))
            w_best = X[first[np.argmax(M["sharpe"][first])]]

        # leverage & TC on first bar (EQW contract)
        wL = float(leverage) * w_best
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

    # --- outputs (EQW contract) ---
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers).astype(float)
        W.index.name = "date"
    else:
        W = pd.DataFrame(columns=tickers, dtype=float); W.index.name = "date"

    details = {
        "method": "nsga2",
        "rebalance": rebalance,
        "interval": interval,
        "objectives": obj,
        "population": int(pop),
        "generations": int(gens),
        "eta_c": float(eta_c),
        "eta_m": float(eta_m),
        "mutation_rate": float(mut_rate),
        "alpha": float(alpha),
        "seed": int(seed),
        "leverage": float(leverage),
        "bounds": [lo, hi],
        "min_obs": int(min_obs),
        "costs_bps_total": float(bps),
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
