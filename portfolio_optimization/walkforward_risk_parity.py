# portfolio_optimization/walkforward_risk_parity.py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

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

def _risk_parity_solver(S: np.ndarray, lo: float, hi: float,
                        max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Simple iterative risk parity solver (variance-based)."""
    n = len(S)
    w = np.ones(n) / n
    w = np.clip(w, lo, hi); w /= w.sum()
    for _ in range(max_iter):
        # marginal risk contributions
        mrc = S @ w
        rc = w * mrc
        rc /= rc.sum()
        # adjust weights toward equal risk
        w *= (1.0 / (rc + 1e-12))
        w = np.clip(w, lo, hi)
        w /= w.sum()
        if np.linalg.norm(rc - 1/n, 1) < tol:
            break
    return w

# Optional Numba-accelerated solver
try:
    import numba as _nb
    import numpy as _np

    @_nb.njit(cache=True)
    def _risk_parity_solver_numba(S: _np.ndarray, lo: float, hi: float,
                                  max_iter: int = 1000, tol: float = 1e-8) -> _np.ndarray:
        n = S.shape[0]
        w = _np.empty(n, dtype=_np.float64)
        inv_n = 1.0 / n if n > 0 else 0.0
        for i in range(n):
            w[i] = inv_n
        # clip and renorm
        ssum = 0.0
        for i in range(n):
            if w[i] < lo:
                w[i] = lo
            elif w[i] > hi:
                w[i] = hi
            ssum += w[i]
        if ssum > 0.0:
            for i in range(n):
                w[i] /= ssum
        for _ in range(max_iter):
            # mrc = S @ w
            mrc = _np.zeros(n, dtype=_np.float64)
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += S[i, j] * w[j]
                mrc[i] = s
            rc = _np.zeros(n, dtype=_np.float64)
            sumrc = 0.0
            for i in range(n):
                rc[i] = w[i] * mrc[i]
                sumrc += rc[i]
            if sumrc <= 0.0:
                sumrc = 1.0
            for i in range(n):
                rc[i] /= sumrc
            # adjust towards equal risk contributions
            for i in range(n):
                w[i] *= 1.0 / (rc[i] + 1e-12)
                if w[i] < lo:
                    w[i] = lo
                elif w[i] > hi:
                    w[i] = hi
            # renormalize
            ssum = 0.0
            for i in range(n):
                ssum += w[i]
            if ssum > 0.0:
                inv = 1.0 / ssum
                for i in range(n):
                    w[i] *= inv
            # convergence on L1 distance
            diff = 0.0
            for i in range(n):
                diff += abs(rc[i] - inv_n)
            if diff < tol:
                break
        return w

    # swap in jitted solver
    _risk_parity_solver = _risk_parity_solver_numba  # type: ignore
except Exception:
    pass

def walkforward_risk_parity(
        tickers: list[str],
        start: str,
        end: str,
        dtype: str = "close",
        interval: str = "1d",
        rebalance: str = "monthly",  # 'daily','weekly','monthly','quarterly'
        costs: dict | None = None,   # {'bps':x,'slippage_bps':y,'spread_bps':z}
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_obs: int = 60,
        leverage: float = 1.0,
        cov_estimator: str | None = None,
) -> dict:
    """
    Walk-forward Risk Parity (variance-based).
    """
    costs = costs or {}
    bps_total = sum(costs.get(k, 0.0) for k in ["bps","slippage_bps","spread_bps"])

    # 1. Load data
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available for given inputs.")

    # 2. Rebalance dates
    freq_map = {"daily":"D","weekly":"W","monthly":"M","quarterly":"Q"}
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq_map[rebalance])
    rbd = [d for d in rbd if d in rets.index]
    if not rbd: rbd = [rets.index[0]]

    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_list, w_prev = [], None  # track levered weights

    for i, t in enumerate(rbd):
        window = rets.loc[:t]
        if len(window) < min_obs:
            w = np.ones(n) / n
            w = np.clip(w, lo, hi); w /= w.sum()
        else:
            S_sample = _stabilize_cov(window.cov().values)
            S = S_sample
            if isinstance(cov_estimator, str):
                ce = cov_estimator.lower()
                if ce in {"diag","diagonal"}:
                    S = np.diag(np.diag(S_sample))
                elif ce in {"lw","ledoit","ledoitwolf"}:
                    try:
                        from sklearn.covariance import LedoitWolf  # type: ignore
                        lw = LedoitWolf().fit(window.values)
                        S = lw.covariance_.astype(float)
                    except Exception:
                        d = np.diag(np.diag(S_sample)); S = 0.9*S_sample + 0.1*d
            w = _risk_parity_solver(S, lo, hi)

        # apply leverage and turnover costs
        wL = float(leverage) * w
        turnover_cost = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * (bps_total/1e4)

        # hold period
        next_t = rbd[i+1] if i+1 < len(rbd) else rets.index[-1]
        hold = rets.loc[t:next_t]
        port = (hold @ wL).astype(float)
        if len(port) > 0: port.iloc[0] -= turnover_cost

        pnl.loc[port.index] = port.values
        weights_list.append((t,wL))
        w_prev = wL

    weights = pd.DataFrame({d:w for d,w in weights_list}).T
    weights.index.name="date"; weights.columns=tickers

    return {
        "weights": weights,
        "pnl": pnl.loc[weights.index[0]:],
        "details": {
            "rebalance_dates": weights.index.tolist(),
            "config": {
                "method":"risk_parity",
                "rebalance":rebalance,
                "interval":interval,
                "min_weight":min_weight,
                "max_weight":max_weight,
                "costs_bps_total":bps_total,
                "covariance":"sample_expanding",
                "min_obs":min_obs,
                "leverage": leverage
            }
        }
    }
