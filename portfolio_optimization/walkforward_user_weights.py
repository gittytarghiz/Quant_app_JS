# Minimal walkforward for user weights: supports static dict or scheduled DataFrame.
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v.astype(float), 0.0)
    s = v.sum()
    return (v / s) if s > 0 else np.ones_like(v) / max(1, v.size)

def walkforward_user_weights(
    tickers: list[str],
    start: str,
    end: str,
    dtype: str = "close",
    interval: str = "1d",
    rebalance: str = "monthly",
    costs: dict | None = None,
    static_weights: dict | None = None,
    weights_df: pd.DataFrame | None = None,
    normalize: bool = True,
    leverage: float = 1.0,
    interest_rate = 0.04,
) -> dict:
    """
    Walk-forward backtester for user-supplied weights.
    - If `weights_df` provided (date-indexed), use most recent row at each rebalance date.
    - Else if `static_weights` dict given, use it.
    - Else falls back to equal-weight.
    Returns: {"weights": DataFrame(index='date'), "pnl": Series, "details": dict}
    """
    daily_rate = interest_rate / 252

    costs = costs or {}
    bps = float(sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps"))) / 1e4

    # --- load & enforce ticker order ---
    prices = (
        get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval)
        .dropna()
        .copy()
    )
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers in data: {missing}")
    prices = prices[tickers].astype(float)

    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available for given inputs.")

    # --- rebalance dates (unified) ---
    freq_map = {"daily": "D", "weekly": "W-THU", "monthly": "ME", "quarterly": "Q"}
    freq = freq_map[rebalance]
    rbd = pd.date_range(rets.index[0], rets.index[-1], freq=freq).intersection(rets.index)

    n = len(tickers)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    w_prev = None

    # --- prepare static vector if provided ---
    static_vec = None
    if static_weights:
        vec = np.array([float(static_weights.get(t, 0.0)) for t in tickers], dtype=float)
        if normalize:
            s = vec.sum()
            static_vec = vec / s if s > 0 else np.zeros_like(vec)
        else:
            static_vec = np.maximum(vec, 0.0)

    # --- prepare scheduled table if provided ---
    schedule = None
    if weights_df is not None:
        df = weights_df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
        # enforce tickers columns
        for t in tickers:
            if t not in df.columns:
                df[t] = 0.0
        df = df[tickers].sort_index()
        if normalize:
            s = df.sum(axis=1).replace(0.0, np.nan)
            df = df.div(s, axis=0).fillna(0.0)
        schedule = df

    # --- walk-forward ---
    for i, t in enumerate(rbd):
        # decide weights
        if schedule is not None:
            ix = schedule.index.searchsorted(t, side="right") - 1
            if ix < 0:
                row = schedule.iloc[0]
            else:
                row = schedule.iloc[ix]
            w = row.values.astype(float)
            if normalize:
                s = w.sum()
                w = w / s if s > 0 else np.zeros_like(w)
        elif static_vec is not None:
            w = static_vec.copy()
        else:
            w = np.ones(n, dtype=float) / max(1, n)

        # apply leverage + TC
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

    # --- build outputs (EQW contract) ---
    if weights_by_date:
        W = pd.DataFrame.from_dict(weights_by_date, orient="index", columns=tickers).astype(float)
        W.index.name = "date"
    else:
        W = pd.DataFrame(columns=tickers, dtype=float); W.index.name = "date"

    details = {
        "method": "user_weights",
        "rebalance": rebalance,
        "interval": interval,
        "source": "scheduled" if schedule is not None else ("static" if static_vec is not None else "equal_weight"),
        "normalize": bool(normalize),
        "leverage": float(leverage),
        "costs_bps_total": float(bps),
    }

    start_idx = W.index[0] if len(W.index) else rets.index[0]
    return {"weights": W, "pnl": pnl.loc[start_idx:], "details": details}
