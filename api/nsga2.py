from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data_management.monolith_loader import get_downloaded_series
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics
from portfolio_optimization.walkforward_nsga2 import walkforward_nsga2


router = APIRouter(prefix="/opt", tags=["opt"])


_ANN = {"1d": 252, "daily": 252, "1wk": 52, "weekly": 52, "1mo": 12, "monthly": 12}


class NSGA2SlimRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    min_weight: float = 0.0
    max_weight: float = 1.0
    leverage: float = 1.0

    primary_objective: str = Field(default="sharpe")  # choose best-by
    # algorithm
    pop_size: int = 64
    gens: int = 25
    seed: int = 42

    # params for metrics
    rf: float = 0.0
    mar: float = 0.0
    cvar_alpha: float = 0.05
    target: float = 0.0


def _project_sum1_box(w: np.ndarray, lo: float, hi: float, iters: int = 6) -> None:
    n = w.size
    s = float(w.sum())
    if abs(s) <= 1e-12:
        w[:] = 1.0 / n
    else:
        w[:] = np.clip(w / s, lo, hi)
    for _ in range(iters):
        s = float(w.sum())
        if abs(s - 1.0) < 1e-12:
            break
        if s > 1.0:
            free = w > lo + 1e-15
            if not np.any(free):
                break
            excess = s - 1.0
            wf = w[free]
            w[free] = np.clip(wf - excess * (wf / (wf.sum() + 1e-12)), lo, hi)
        else:
            free = w < hi - 1e-15
            if not np.any(free):
                break
            deficit = 1.0 - s
            wf = w[free]
            w[free] = np.clip(wf + deficit * (wf / (wf.sum() + 1e-12)), lo, hi)
    w /= w.sum() + 1e-12


def _scores(R: np.ndarray, w: np.ndarray, ann: float, rf: float, mar: float, alpha: float, target: float) -> dict[str, float]:
    r = R @ w  # (T,)
    mean = float(r.mean())
    sd = float(r.std(ddof=0))
    sharpe = (mean - rf / ann) / (sd + 1e-12) * np.sqrt(ann)
    min_vol = sd
    # CVaR alpha (historical)
    k = max(1, int(np.ceil(alpha * r.size)))
    loss = -r
    idx = np.argpartition(loss, -k)[-k:]
    cvar = float(loss[idx].mean())
    # extras used by primary if chosen
    sortino = 0.0
    if mar != 0.0:
        excess = r - (mar / ann)
        d = excess[excess < 0.0]
        ds = float(np.sqrt((d * d).mean())) if d.size else 1e-12
        sortino = (float(excess.mean()) / (ds + 1e-12)) * np.sqrt(ann)
    # calmar needs CAGR and MDD
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12)) - 1.0
    mdd = float(-dd.min()) if dd.size else 0.0
    T = max(r.size, 1)
    cagr = float(np.power(max(eq[-1], 1e-12), ann / T) - 1.0)
    calmar = (cagr / (mdd + 1e-12)) if mdd > 0 else 0.0
    # target_min_vol penalty
    tpen = max(0.0, target - mean)
    target_min_vol = min_vol + 1e3 * (tpen * tpen)
    return {
        "sharpe": float(sharpe),
        "min_vol": float(min_vol),
        "cvar": float(cvar),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "target_min_vol": float(target_min_vol),
    }


def _nsga2(R: np.ndarray, lo: float, hi: float, pop: int, gens: int, seed: int,
           ann: float, rf: float, mar: float, alpha: float, target: float,
           objectives: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    T, N = R.shape
    P = pop
    K = len(objectives)
    W = rng.uniform(lo, hi, size=(P, N)).astype(np.float64)
    for i in range(P):
        _project_sum1_box(W[i], lo, hi)
    S = np.empty((P, K), dtype=np.float64)

    def eval_pop(Wm: np.ndarray, Sm: np.ndarray) -> None:
        for i in range(P):
            sc = _scores(R, Wm[i], ann, rf, mar, alpha, target)
            Sm[i] = [sc[o] if o in sc else 0.0 for o in objectives]

    def nsort(Sm: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        Pn, Kk = Sm.shape
        Slist = Sm  # (P,K)
        dom_count = np.zeros(Pn, dtype=np.int32)
        dom_list: list[list[int]] = [[] for _ in range(Pn)]
        fronts: list[list[int]] = []
        F0: list[int] = []
        for p in range(Pn):
            for q in range(Pn):
                if p == q:
                    continue
                sp = Slist[p]; sq = Slist[q]
                p_dom_q = np.all(sp <= sq) and np.any(sp < sq)
                q_dom_p = np.all(sq <= sp) and np.any(sq < sp)
                if p_dom_q:
                    dom_list[p].append(q)
                elif q_dom_p:
                    dom_count[p] += 1
            if dom_count[p] == 0:
                F0.append(p)
        fronts.append(F0)
        i = 0
        while i < len(fronts) and fronts[i]:
            Q: list[int] = []
            for p in fronts[i]:
                for q in dom_list[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
        rank = np.full(Pn, len(fronts), dtype=np.int32)
        for r, F in enumerate(fronts):
            for idx in F:
                rank[idx] = r
        return fronts, rank

    def crowd(Sm: np.ndarray, F: list[int]) -> np.ndarray:
        m = len(F)
        cd = np.zeros(m, dtype=np.float64)
        if m == 0:
            return cd
        for k in range(K):
            vals = [(Sm[i, k], j) for j, i in enumerate(F)]
            vals.sort(key=lambda x: x[0])
            cd[vals[0][1]] = cd[vals[-1][1]] = np.inf
            vmin, vmax = vals[0][0], vals[-1][0]
            if vmax - vmin <= 1e-18:
                continue
            for j in range(1, m - 1):
                prevv, _ = vals[j - 1]
                nextv, _ = vals[j + 1]
                cd[vals[j][1]] += (nextv - prevv) / (vmax - vmin)
        return cd

    eval_pop(W, S)
    for _ in range(gens):
        fronts, rank = nsort(S)
        crowd_all = np.zeros(P, dtype=np.float64)
        for F in fronts:
            cd = crowd(S, F)
            for j, idx in enumerate(F):
                crowd_all[idx] = cd[j]
        # offspring by tournament selection
        Wn = np.empty_like(W)
        for i in range(0, P, 2):
            i1, i2, j1, j2 = rng.integers(0, P, size=4)
            p = i1 if (rank[i1] < rank[i2] or (rank[i1] == rank[i2] and crowd_all[i1] > crowd_all[i2])) else i2
            q = j1 if (rank[j1] < rank[j2] or (rank[j1] == rank[j2] and crowd_all[j1] > crowd_all[j2])) else j2
            a, b = W[p], W[q]
            c1 = 0.5 * (a + b); c2 = 0.5 * (b + a)
            # small mutation within box
            c1 += (rng.random(N) - 0.5) * (hi - lo) * 0.1
            c2 += (rng.random(N) - 0.5) * (hi - lo) * 0.1
            c1 = np.clip(c1, lo, hi); c2 = np.clip(c2, lo, hi)
            _project_sum1_box(c1, lo, hi); _project_sum1_box(c2, lo, hi)
            Wn[i] = c1; Wn[min(i + 1, P - 1)] = c2
        # evaluate children
        Sn = np.empty_like(S)
        eval_pop(Wn, Sn)
        # combine and select next gen
        W_all = np.vstack([W, Wn])
        S_all = np.vstack([S, Sn])
        fronts, r_all = nsort(S_all)
        order = []
        for F in fronts:
            if not F:
                continue
            cd = crowd(S_all, F)
            F_sorted = sorted(range(len(F)), key=lambda t: (cd[t]), reverse=True)
            order.extend([F[t] for t in F_sorted])
            if len(order) >= P:
                break
        order = order[:P]
        W = W_all[order]
        S = S_all[order]
    return W, S


@router.post("/nsga2")
def nsga2(req: NSGA2SlimRequest) -> dict[str, Any]:
    if not req.tickers:
        raise HTTPException(status_code=400, detail="Tickers list is empty")
    prices = get_downloaded_series(req.tickers, req.start, req.end, dtype=req.dtype, interval=req.interval).dropna()
    rets_df = prices.pct_change().dropna()
    if rets_df.empty:
        raise HTTPException(status_code=400, detail="No returns available for given inputs")
    R = rets_df.values.astype(np.float64, copy=False)
    T, N = R.shape
    lo, hi = float(req.min_weight), float(req.max_weight)
    if N * lo > 1 + 1e-12 or N * hi < 1 - 1e-12:
        raise HTTPException(status_code=400, detail="Infeasible (min_weight,max_weight) for sum=1.")
    ann = float(_ANN.get(req.interval.lower(), 252))
    # Use simplified walkforward: sample `tries` candidates and pick best by Sharpe
    res = walkforward_nsga2(
        tickers=list(prices.columns),
        start=req.start,
        end=req.end,
        dtype=req.dtype,
        interval=req.interval,
        rebalance=req.interval,
        costs=None,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        min_obs=60,
        leverage=req.leverage,
        tries=int(req.tries),
        seed=int(req.seed),
    )
    weights_df = res.get("weights")
    pnl = res.get("pnl")
    details = normalize_details(res.get("details"))
    try:
        details["metrics"] = compute_metrics(pnl, weights_df)
    except Exception:
        pass
    return {"weights": df_to_records(weights_df), "pnl": series_to_records(pnl, value_name="pnl"), "details": details}
