# portfolio_optimization/walkforward_risk_parity.py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

def _stabilize_cov(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    tr = np.trace(S) / max(S.shape[0], 1)
    eps = 1e-8 * (tr if np.isfinite(tr) and tr > 0 else 1.0)
    return S + eps * np.eye(S.shape[0])

def _risk_parity_solver(S: np.ndarray, lo: float, hi: float,
                        max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    n = len(S)
    w = np.full(n, 1.0/n)
    for _ in range(max_iter):
        mrc = S @ w
        rc = w * mrc
        rc /= rc.sum()
        w *= 1.0 / (rc + 1e-12)
        w = np.clip(w, lo, hi); w /= w.sum()
        if np.linalg.norm(rc - 1/n, 1) < tol: break
    return w

def walkforward_risk_parity(
    tickers: list[str],
    start: str,
    end: str,
    dtype: str = "close",
    interval: str = "1d",
    rebalance: str = "monthly",    # 'daily','weekly','monthly','quarterly'
    costs: dict | None = None,     # {'bps':x,'slippage_bps':y,'spread_bps':z}
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    min_obs: int = 60,
    leverage: float = 1.0,
    interest_rate: float = 0.04,
    cov_estimator: str | None = None,
) -> dict:
    """Walk-forward Risk Parity with transaction + financing costs."""
    rets = (
        get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval)
        .dropna()
        .pct_change()
        .dropna()
    )
    if rets.empty:
        raise ValueError("No returns available.")

    # Rebalance schedule
    freq_map = {"daily":"D","weekly":"W","monthly":"M","quarterly":"Q"}
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq_map[rebalance])
    rbd = rbd.intersection(rets.index)
    if rbd.empty: rbd = [rets.index[0]]

    # Params
    n = len(tickers)
    lo, hi = float(min_weight), float(max_weight)
    bps_total = sum((costs or {}).get(k, 0.0) for k in ["bps","slippage_bps","spread_bps"]) / 1e4
    daily_rate = interest_rate / 252

    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date, w_prev = {}, None

    for i, t in enumerate(rbd):
        window = rets.loc[:t]
        if len(window) < min_obs:
            w = np.full(n, 1.0/n)
        else:
            S = _stabilize_cov(window.cov().values)
            if cov_estimator and cov_estimator.lower() in {"diag","diagonal"}:
                S = np.diag(np.diag(S))
            w = _risk_parity_solver(S, lo, hi)

        wL = w / w.sum() * leverage
        turnover = (np.abs(wL).sum() if w_prev is None else np.abs(wL - w_prev).sum()) * bps_total

        nxt = rbd[i+1] if i+1 < len(rbd) else rets.index[-1]
        port = (rets.loc[t:nxt] @ wL).astype(float)
        if not port.empty:
            port.iloc[0] -= turnover
            excess_lev = np.abs(wL).sum() - 1.0
            if excess_lev > 0:
                port -= excess_lev * daily_rate
            pnl.loc[port.index] = port.values

        weights_by_date[t] = wL
        w_prev = wL

    W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers)
    W.index.name = "date"

    return {
        "weights": W,
        "pnl": pnl.loc[W.index[0]:],
        "details": {
            "method": "risk_parity",
            "rebalance": rebalance,
            "interval": interval,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "costs_bps_total": bps_total,
            "min_obs": min_obs,
            "leverage": leverage,
            "interest_rate": interest_rate,
        }
    }
