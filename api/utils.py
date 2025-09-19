from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _to_iso(dt: Any) -> Any:
    if isinstance(dt, (pd.Timestamp, datetime)):
        # Ensure naive ISO (no tz info) for frontend friendliness
        return pd.Timestamp(dt).tz_localize(None).isoformat()
    return dt


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a DataFrame with a DatetimeIndex to list-of-dicts.

    - Index is emitted as 'date' (ISO string)
    - Columns are kept as-is; values converted to built-in types
    - Empty frames -> []
    """
    if df is None or len(df) == 0:
        return []
    # Normalize index name and types; force first column to be 'date'
    out = []
    r = df.reset_index()
    if "date" not in r.columns and len(r.columns) > 0:
        r = r.rename(columns={r.columns[0]: "date"})
    # Iterate rows to ensure numpy scalars -> Python types
    for row in r.itertuples(index=False):
        rec = {}
        for k, v in zip(r.columns, row):
            if k == "date":
                rec[k] = _to_iso(v)
            else:
                # Cast numpy scalar to native Python
                if isinstance(v, (np.generic,)):
                    v = v.item()
                rec[str(k)] = None if pd.isna(v) else v
        out.append(rec)
    return out


def series_to_records(s: pd.Series, value_name: str = "value") -> list[dict[str, Any]]:
    """Serialize a Series with a DatetimeIndex to list-of-dicts: [{date, value}]."""
    if s is None or len(s) == 0:
        return []
    idx_name = s.index.name or "date"
    r = s.reset_index()
    if idx_name != "date":
        r = r.rename(columns={idx_name: "date", 0: value_name})
    else:
        r = r.rename(columns={0: value_name})
    out = []
    for row in r.itertuples(index=False):
        d, v = row[0], row[1]
        if isinstance(v, (np.generic,)):
            v = v.item()
        out.append({"date": _to_iso(d), value_name: None if pd.isna(v) else v})
    return out


def normalize_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize details dict for JSON (convert dates to ISO strings)."""
    if not details:
        return {}
    out: dict[str, Any] = {}
    for k, v in details.items():
        if k == "rebalance_dates" and isinstance(v, Iterable):
            out[k] = [_to_iso(x) for x in v]
        else:
            out[k] = v
    return out


# --------- simple portfolio metrics (server-side) ---------
def _annualization_factor(idx: pd.DatetimeIndex) -> int:
    """Infer periods-per-year from a datetime index (rough heuristic like frontend)."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return 252
    diffs = np.diff(idx.view(np.int64)) / 1e9  # seconds
    days = np.array(diffs) / (24 * 3600)
    if len(days) == 0:
        return 252
    med = float(np.median(days))
    if med <= 2:
        return 252
    if med <= 10:
        return 52
    if med <= 40:
        return 12
    return 252


def _compute_drawdown_stats(equity: pd.Series) -> tuple[float, float | None, float | None]:
    """Return (max_drawdown, max_drawdown_days, current_drawdown_days)."""
    if equity is None or len(equity) == 0:
        return 0.0, None, None
    e = pd.Series(pd.to_numeric(equity, errors="coerce"), index=equity.index).dropna()
    if e.empty:
        return 0.0, None, None
    run_max = e.cummax()
    dd = (e / run_max) - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    # Durations if datetime index
    max_days = None
    cur_days = None
    if isinstance(e.index, pd.DatetimeIndex):
        in_dd = e < run_max
        max_len = pd.Timedelta(0)
        cur_len = pd.Timedelta(0)
        start = None
        for t, flag in zip(e.index, in_dd):
            if flag:
                if start is None:
                    start = t
                cur_len = t - start
                if cur_len > max_len:
                    max_len = cur_len
            else:
                start = None
                cur_len = pd.Timedelta(0)
        max_days = float(max_len.days)
        cur_days = float(cur_len.days)
    return mdd, max_days, cur_days


def _compute_turnover(weights: pd.DataFrame | None) -> tuple[float | None, float | None]:
    """Approximate turnover using L1 changes between consecutive weight vectors.

    Returns (avg_turnover, total_turnover)."""
    if weights is None or len(weights) == 0:
        return None, None
    if not isinstance(weights, pd.DataFrame):
        try:
            weights = pd.DataFrame(weights)
        except Exception:
            return None, None
    w = weights.copy()
    # Assume index is chronological
    w = w.sort_index()
    # Keep only numeric columns
    w = w.select_dtypes(include=[np.number])
    if w.shape[1] == 0:
        return None, None
    delta = (w - w.shift(1)).abs().sum(axis=1) / 2.0
    delta = delta.dropna()
    if len(delta) == 0:
        return None, None
    return float(delta.mean()), float(delta.sum())


def compute_metrics(pnl: pd.Series | None, weights: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute easy metrics from PnL series (and optionally weights).

    Metrics: ann_factor, ann_return, ann_vol, sharpe, sortino, total_return,
    cagr, max_drawdown, calmar, avg_turnover, total_turnover, max_drawdown_days,
    current_drawdown_days.
    """
    out: dict[str, Any] = {}
    if pnl is None or len(pnl) == 0:
        return out
    s = pd.Series(pd.to_numeric(pnl, errors="coerce"), index=pnl.index).dropna()
    if s.empty:
        return out
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    ann = _annualization_factor(s.index if isinstance(s.index, pd.DatetimeIndex) else pd.DatetimeIndex([]))
    mean = float(s.mean())
    vol = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ann_vol = vol * np.sqrt(ann)
    ann_return = mean * ann
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0
    downs = s[s < 0]
    ds = float(np.sqrt(np.mean(np.square(downs)))) if len(downs) else 0.0
    sortino = (ann_return / (ds * np.sqrt(ann))) if ds > 0 else 0.0
    equity = (1.0 + s).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    T = max(len(s), 1)
    cagr = float(np.power(max(equity.iloc[-1], 1e-12), ann / T) - 1.0)
    mdd, max_dd_days, cur_dd_days = _compute_drawdown_stats(equity)
    calmar = (cagr / abs(mdd)) if mdd < 0 else None
    avg_turn, tot_turn = _compute_turnover(weights)
    out.update(
        {
            "ann_factor": ann,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": mdd,
            "calmar": calmar,
            "avg_turnover": avg_turn,
            "total_turnover": tot_turn,
            "max_drawdown_days": max_dd_days,
            "current_drawdown_days": cur_dd_days,
        }
    )
    # Historical VaR / CVaR at 95%
    try:
        if len(s) >= 10:
            q = 0.05
            var95 = float(np.quantile(s, q))
            cvar95 = float(s[s <= var95].mean()) if np.any(s <= var95) else None
            out.update({"var_95": var95, "cvar_95": cvar95})
    except Exception:
        pass
    return out
