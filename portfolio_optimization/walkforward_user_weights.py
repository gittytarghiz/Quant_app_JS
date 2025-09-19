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
    tickers: list[str], start: str, end: str,
    dtype: str = "close", interval: str = "1d",
    rebalance: str = "monthly", costs: dict | None = None,
    static_weights: dict | None = None, weights_df: pd.DataFrame | None = None,
    normalize: bool = True, leverage: float = 1.0,
) -> dict:
    """
    Minimal user-weights backtester.
    - If `weights_df` provided (date-indexed), use the most-recent row at each rebalance date.
    - Else use `static_weights` (dict of ticker->weight). Missing tickers are treated as 0.
    - Falls back to equal-weight when input is empty or invalid.
    """
    costs = costs or {}
    bps_total = float(sum(costs.get(k, 0.0) for k in ("bps", "slippage_bps", "spread_bps")))

    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available for given inputs.")

    freq_map = {"daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q"}
    freq = freq_map.get(rebalance, "M")
    rbd = [d for d in pd.date_range(rets.index[0], rets.index[-1], freq=freq) if d in rets.index] or [rets.index[0]]

    n = len(tickers)
    pnl = pd.Series(0.0, index=rets.index, dtype=float)
    weights_list: list[tuple[pd.Timestamp, np.ndarray]] = []
    w_prev = None

    # prepare static vector
    static_vec = None
    if static_weights:
        vec = np.array([float(static_weights.get(t, 0.0)) for t in tickers], dtype=float)
        static_vec = _norm_vec(vec) if normalize else np.maximum(vec, 0.0)

    # prepare scheduled table if provided
    schedule = None
    if weights_df is not None:
        df = weights_df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
        # keep known columns, add missing as 0
        for t in tickers:
            if t not in df.columns:
                df[t] = 0.0
        df = df[tickers].sort_index()
        if normalize:
            s = df.sum(axis=1).replace(0.0, np.nan)
            df = df.div(s, axis=0).fillna(0.0)
        schedule = df

    for i, t in enumerate(rbd):
        # determine weights at this rebalance
        if schedule is not None:
            ix = schedule.index.searchsorted(t, side="right") - 1
            if ix < 0:
                row = schedule.iloc[0]
            else:
                row = schedule.iloc[ix]
            w = row.values.astype(float)
            if normalize:
                w = _norm_vec(w)
        elif static_vec is not None:
            w = static_vec.copy()
        else:
            w = np.ones(n, dtype=float) / max(1, n)

        wL = float(leverage) * w
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
        "details": {
            "rebalance_dates": weights.index.tolist(),
            "config": {"method": "user_weights_simple", "source": ("scheduled" if schedule is not None else "static"), "leverage": float(leverage)},
        },
    }
