# portfolio_optimization/walkforward_min_variance.py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Optional cvxpy; fall back to SciPy SLSQP if unavailable
try:
    import cvxpy as cp  # type: ignore
    _HAS_CVXPY = True
except Exception:
    cp = None  # type: ignore
    _HAS_CVXPY = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

def _stabilize_cov(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    S = 0.5 * (S + S.T)
    n = S.shape[0]
    tr = np.trace(S) / max(n, 1)
    eps = 1e-8 * (tr if np.isfinite(tr) and tr > 0 else 1.0)
    return S + eps * np.eye(n)

def walkforward_min_variance(
        tickers: list[str],
        start: str,
        end: str,
        dtype: str = "close",
        interval: str = "1d",
        rebalance: str = "monthly",          # 'daily','weekly','monthly','quarterly'
        costs: dict | None = None,           # {'bps':x,'slippage_bps':y,'spread_bps':z}
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_obs: int = 60,                   # min observations before first optimization
        leverage: float = 1.0
) -> dict:
    """
    Walk-forward Minimum-Variance (sample covariance, expanding window).
    Objective:  minimize w' Σ w
    Constraints: sum(w)=1, min_weight <= w_i <= max_weight
    Returns:
        dict(weights: DataFrame, pnl: Series, details: dict)
    """
    costs = costs or {}
    bps_total = float(sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")))

    # 1) Data
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available for given inputs.")

    # 2) Rebalance dates
    freq_map = {"daily": "D", "weekly": "W-FRI", "monthly": "ME", "quarterly": "Q"}
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq_map[rebalance])
    rbd = [d for d in rbd if d in rets.index]
    if not rbd:
        rbd = [rets.index[0]]

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_list, w_prev = [], None  # track levered weights

    for i, t in enumerate(rbd):
        window = rets.loc[:t]
        if len(window) < min_obs:
            w = np.full(n, 1.0 / n)
            w = np.clip(w, lo, hi); w /= w.sum()
        else:
            S = _stabilize_cov(window.cov().values)
            if _HAS_CVXPY:
                # Solve convex QP with cvxpy if available
                w_var = cp.Variable(n)
                cons = [cp.sum(w_var) == 1, w_var >= lo, w_var <= hi]
                prob = cp.Problem(cp.Minimize(cp.quad_form(w_var, S)), cons)
                try:
                    prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
                except Exception:
                    try:
                        prob.solve(solver=cp.SCS, eps=1e-4, verbose=False)
                    except Exception:
                        pass

                if (getattr(w_var, 'value', None) is None) or (not np.all(np.isfinite(np.asarray(w_var.value)))):
                    w = np.full(n, 1.0 / n)
                    w = np.clip(w, lo, hi); w /= w.sum()
                else:
                    w = np.asarray(w_var.value, float).ravel()
                    w = np.clip(w, lo, hi); w /= w.sum()
            else:
                # Fallback: SciPy SLSQP with bounds + equality constraint
                try:
                    from scipy.optimize import minimize
                except Exception:
                    # If SciPy is somehow unavailable, use equal weights
                    w = np.full(n, 1.0 / n)
                    w = np.clip(w, lo, hi); w /= w.sum()
                else:
                    def obj(x: np.ndarray) -> float:
                        return float(x @ S @ x)

                    x0 = np.full(n, 1.0 / n)
                    bnds = [(lo, hi) for _ in range(n)]
                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)
                    try:
                        res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
                        if (res.x is None) or (not np.all(np.isfinite(res.x))):
                            raise RuntimeError('SLSQP failed')
                        w = np.asarray(res.x, float)
                        w = np.clip(w, lo, hi); s = w.sum(); w = (w / s) if s != 0 else x0
                    except Exception:
                        # Robust fallback
                        w = np.full(n, 1.0 / n)
                        w = np.clip(w, lo, hi); w /= w.sum()

        # apply leverage and turnover cost at rebalance
        wL = float(leverage) * w
        turnover_cost = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * (bps_total / 1e4)

        # hold until next rebalance (or end)
        next_t = rbd[i + 1] if i + 1 < len(rbd) else rets.index[-1]
        hold = rets.loc[t:next_t]
        port = (hold @ wL).astype(float)
        if len(port) > 0:
            port.iloc[0] -= turnover_cost

        pnl.loc[port.index] = port.values
        weights_list.append((t, wL))
        w_prev = wL

    weights = pd.DataFrame({d: w for d, w in weights_list}).T
    weights.index.name = "date"; weights.columns = tickers

    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {
            "rebalance_dates": weights.index.tolist(),
            "config": {
                "method": "min_variance",
                "rebalance": rebalance,
                "interval": interval,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "costs_bps_total": bps_total,
                "covariance": "sample_expanding",
                "min_obs": min_obs,
                "leverage": leverage,
                "has_cvxpy": bool(_HAS_CVXPY)
            }
        }
    }
