# portfolio_optimization/walkforward_pso.py
import numpy as np, pandas as pd
from pathlib import Path; import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series
from portfolio_optimization.walkforward_equal_weight import walkforward_equal_weight


# ---- main ----
def walkforward_pso(
    tickers, start, end, dtype="close", interval="1d",
    rebalance="monthly", costs=None, min_weight=0.0, max_weight=1.0, min_obs=60,
    leverage: float = 1.0,
    pso_particles=60, pso_iters=80, pso_c1=1.5, pso_c2=1.5, pso_w=0.7, pso_seed=42,
    objective="sharpe", cvar_alpha=0.05
):
    """
    PSO optimization fallback: using equal weight strategy as PSO is not implemented.
    """
    result = walkforward_equal_weight(
        tickers, start, end, dtype=dtype, interval=interval,
        rebalance=rebalance, costs=costs, min_weight=min_weight,
        max_weight=max_weight, leverage=leverage
    )
    # Indicate fallback in details
    if "details" in result and isinstance(result["details"], dict):
        result["details"]["pso_fallback"] = "PSO optimization not implemented; equal weight used as fallback."
    return result

    costs = costs or {}
    bps_total = float(sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")))
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty: raise ValueError("No returns available.")

    freq = {"daily": "D", "weekly": "W-FRI", "monthly": "ME", "quarterly": "Q"}[rebalance]
    rbd = [d for d in pd.date_range(rets.index[0], rets.index[-1], freq=freq) if d in rets.index] or [rets.index[0]]

    n = len(tickers); lo, hi = float(min_weight), float(max_weight)
    if n*lo > 1 + 1e-12 or n*hi < 1 - 1e-12:
        raise ValueError("Infeasible (min_weight,max_weight) for sum=1.")

    if pso_seed is not None: np.random.seed(int(pso_seed))
    options = {"c1": float(pso_c1), "c2": float(pso_c2), "w": float(pso_w)}
    ann = _ann_factor(interval)

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    W_mat = np.zeros((len(rbd), n), dtype=float)
    dates = []

    w_prev = None
    for i, t in enumerate(rbd):
        win = rets.loc[:t]
        if len(win) < min_obs:
            w = np.full(n, 1.0/n)
            w = _fast_rebalance_batch(w[None, :], lo, hi)[0]
        else:
            mu = win.mean().values
            S  = _stabilize_cov(win.cov().values)
            R  = np.ascontiguousarray(win.values, dtype=float)
            lb, ub = np.full(n, lo), np.full(n, hi)
            opt = GlobalBestPSO(n_particles=int(pso_particles), dimensions=n, options=options, bounds=(lb, ub))

            if objective == "sharpe":
                obj = lambda X: _neg_sharpe_batch(X, mu, S, lo, hi)
            elif objective == "sortino":
                obj = lambda X: _neg_sortino_batch(X, R, lo, hi, ann)
            elif objective == "calmar":
                obj = lambda X: _neg_calmar_batch(X, R, lo, hi, ann)
            elif objective == "cvar":
                obj = lambda X: _cvar_batch(X, R, lo, hi, float(cvar_alpha))
            elif objective == "ret_turnover":
                obj = lambda X: _neg_ret_turnover_batch(X, R, lo, hi, ann, w_prev)
            elif objective == "ret_drawdown":
                obj = lambda X: _neg_ret_drawdown_batch(X, R, lo, hi, ann)
            elif objective in ("kelly","max_growth","log_utility"):
                obj = lambda X: _neg_kelly_batch(X, R, lo, hi)
            elif objective in ("diversification","diversification_ratio","max_diversification"):
                obj = lambda X: _neg_diversification_batch(X, S, lo, hi)
            else:
                obj = lambda X: _neg_sharpe_batch(X, mu, S, lo, hi)

            _, pos = opt.optimize(obj, iters=int(pso_iters), verbose=False)
            w = _fast_rebalance_batch(pos[None, :], lo, hi)[0]

        # apply leverage and txn cost at rebalance
        wL = float(leverage) * w
        tc = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * (bps_total / 1e4)
        next_t = rbd[i+1] if i+1 < len(rbd) else rets.index[-1]
        hold = rets.loc[t:next_t]
        port = (hold @ wL).astype(float)
        if len(port) > 0:
            port.iloc[0] -= tc
            pnl.loc[port.index] = port.values

        W_mat[i, :] = wL; dates.append(t); w_prev = wL

    weights = pd.DataFrame(W_mat, index=pd.Index(dates, name="date"), columns=tickers)
    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {
            "config": {
                "method": "pso_fast_rebalance",
                "objective": objective,
                "rebalance": rebalance,
                "interval": interval,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "min_obs": min_obs,
                "costs_bps_total": bps_total,
                "pso_particles": pso_particles,
                "pso_iters": pso_iters,
                "pso_c1": pso_c1,
                "pso_c2": pso_c2,
                "pso_w": pso_w,
                "cvar_alpha": cvar_alpha,
                "leverage": leverage
            }
        }
    }
