# api/opt_min_variance.py — equal-weight style hardened endpoint for Minimum-Variance

from __future__ import annotations
from typing import Any, Optional
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
import os, logging, time

from portfolio_optimization.walkforward_min_variance import (
    walkforward_min_variance as _wf_minvar,
)
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics

router = APIRouter(prefix="/opt", tags=["opt"])
_ALLOWED_REBAL = {"daily", "weekly", "monthly", "quarterly"}

# ---------- helpers ----------
def _dedup_upper(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for s in seq or []:
        if not isinstance(s, str):
            continue
        u = s.strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _normalize_costs(costs: Optional[dict[str, float]]) -> Optional[tuple[tuple[str, float], ...]]:
    if not costs:
        return None
    allowed = ("bps", "slippage_bps", "spread_bps")
    items: list[tuple[str, float]] = []
    for k, v in costs.items():
        if k in allowed and v is not None:
            try:
                items.append((k, float(v)))
            except (TypeError, ValueError):
                continue
    return tuple(sorted(items)) or None if items else None

@lru_cache(maxsize=16)
def _minvar_cached(
    tickers_key: tuple[str, ...],
    start: str, end: str, dtype: str, interval: str, rebalance: str,
    costs_key: Optional[tuple[tuple[str, float], ...]],
    min_weight: float, max_weight: float, min_obs: int, leverage: float,
) -> dict[str, Any]:
    res = _wf_minvar(
        tickers=list(tickers_key),
        start=start, end=end,
        dtype=dtype, interval=interval, rebalance=rebalance,
        costs=dict(costs_key) if costs_key else None,
        min_weight=min_weight, max_weight=max_weight, min_obs=min_obs,
        leverage=leverage,
    )
    raw_w = res.get("weights")
    raw_p = res.get("pnl")
    details = normalize_details(res.get("details"))
    try:
        details["metrics"] = compute_metrics(raw_p, raw_w)
    except Exception:
        pass
    return {
        "weights": df_to_records(raw_w),
        "pnl": series_to_records(raw_p, value_name="pnl"),
        "details": details,
    }

# ---------- schema ----------
class MinVarianceRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    costs: Optional[dict[str, float]] = None
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_obs: int = 60
    leverage: float = 1.0

# ---------- endpoint ----------
@router.post("/min-variance")
def min_variance(req: MinVarianceRequest) -> dict[str, Any]:
    log = logging.getLogger("api.minvar")
    t0 = time.time()
    try:
        log.info(
            "POST /opt/min-variance tickers=%d dtype=%s interval=%s rebalance=%s lev=%s window=%s→%s",
            len(req.tickers or []), req.dtype, req.interval, req.rebalance,
            req.leverage, req.start, req.end,
        )
    except Exception:
        pass

    # --- guardrails ---
    tickers = _dedup_upper(req.tickers)
    if not tickers:
        raise HTTPException(400, "tickers must be non-empty")
    if len(tickers) > 64:
        raise HTTPException(400, "too many tickers; limit to 64")

    rebalance = (req.rebalance or "monthly").lower().strip()
    if rebalance not in _ALLOWED_REBAL:
        raise HTTPException(400, f"rebalance must be one of {sorted(_ALLOWED_REBAL)}")

    try:
        lev = float(req.leverage)
    except (TypeError, ValueError):
        raise HTTPException(400, "leverage must be a number")
    lev = max(0.0, min(5.0, lev))

    try:
        min_w = max(0.0, min(1.0, float(req.min_weight)))
        max_w = max(0.0, min(1.0, float(req.max_weight)))
    except (TypeError, ValueError):
        raise HTTPException(400, "min_weight/max_weight must be numbers")
    if min_w > max_w:
        min_w, max_w = max_w, min_w

    costs_key = _normalize_costs(req.costs)

    # --- run with timeout ---
    opt_timeout = float(os.environ.get("OPT_TIMEOUT", "25"))
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            _minvar_cached,
            tuple(tickers), req.start, req.end, req.dtype, req.interval,
            rebalance, costs_key, min_w, max_w, int(req.min_obs), lev,
        )
        try:
            out = fut.result(timeout=opt_timeout)
        except _Timeout:
            raise HTTPException(504, "Min-variance optimization timed out")
        except Exception as e:
            raise HTTPException(400, f"Min-variance failed: {e}")

    try:
        log.info(
            "/opt/min-variance done in %.1fms: weights_rows=%d pnl_rows=%d",
            (time.time() - t0) * 1000.0,
            len(out.get("weights", [])),
            len(out.get("pnl", [])),
        )
    except Exception:
        pass
    return out
