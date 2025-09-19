from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from datetime import datetime
from functools import lru_cache

from portfolio_optimization.walkforward_equal_weight import (
    walkforward_equal_weight as _wf_eqw,
)
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics


# ---- lightweight caching & guardrails ----
_ALLOWED_REBAL = {"daily", "weekly", "monthly", "quarterly"}


def _dedup_upper(seq: list[str]) -> list[str]:
    """Uppercase tickers and de-duplicate while preserving first-seen order."""
    out = []
    seen = set()
    for s in seq or []:
        if not isinstance(s, str):
            continue
        u = s.strip().upper()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _normalize_costs(costs: Optional[dict[str, float]]) -> Optional[tuple[tuple[str, float], ...]]:
    if not costs:
        return None
    # Only accept known keys; coerce to float; sort for stable cache key
    allowed = ("bps", "slippage_bps", "spread_bps")
    items: list[tuple[str, float]] = []
    for k, v in costs.items():
        if k in allowed and v is not None:
            try:
                items.append((k, float(v)))
            except (TypeError, ValueError):
                continue
    if not items:
        return None
    return tuple(sorted(items))


@lru_cache(maxsize=16)
def _eqw_cached(
    tickers_key: tuple[str, ...],
    start: str,
    end: str,
    dtype: str,
    interval: str,
    rebalance: str,
    leverage: float,
    min_weight: float,
    max_weight: float,
    costs_key: Optional[tuple[tuple[str, float], ...]],
) -> dict[str, Any]:
    res = _wf_eqw(
        tickers=list(tickers_key),
        start=start,
        end=end,
        dtype=dtype,
        interval=interval,
        rebalance=rebalance,
        costs=dict(costs_key) if costs_key else None,
        leverage=leverage,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    raw_weights = res.get("weights")
    raw_pnl = res.get("pnl")
    details = normalize_details(res.get("details"))
    try:
        details["metrics"] = compute_metrics(raw_pnl, raw_weights)
    except Exception:
        # Do not fail the request if metrics computation has an issue
        pass
    return {
        "weights": df_to_records(raw_weights),
        "pnl": series_to_records(raw_pnl, value_name="pnl"),
        "details": details,
    }


router = APIRouter(prefix="/opt", tags=["opt"])


class EqualWeightRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")  # daily|weekly|monthly|quarterly
    costs: Optional[dict[str, float]] = None  # {bps, slippage_bps, spread_bps}
    leverage: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0


@router.post("/equal-weight")
def equal_weight(req: EqualWeightRequest) -> dict[str, Any]:
    """Equal weight portfolio optimization endpoint"""
    log = logging.getLogger("api.eqw")
    t0 = time.time()
    try:
        log.info(
            "POST /opt/equal-weight tickers=%d dtype=%s interval=%s rebalance=%s lev=%s window=%s→%s",
            len(req.tickers or []), req.dtype, req.interval, req.rebalance, req.leverage, req.start, req.end,
        )

        # Sanitize inputs
        tickers = _dedup_upper(req.tickers)
        if not tickers:
            raise ValueError("tickers must be a non-empty list")
        if len(tickers) > 64:
            raise ValueError("too many tickers; please limit to 64 or fewer")

        rebalance = (req.rebalance or "monthly").lower().strip()
        if rebalance not in _ALLOWED_REBAL:
            raise ValueError(f"rebalance must be one of {sorted(_ALLOWED_REBAL)}")

        leverage = max(0.0, min(5.0, float(req.leverage)))
        min_w = max(0.0, min(1.0, float(req.min_weight)))
        max_w = max(0.0, min(1.0, float(req.max_weight)))
        if min_w > max_w:
            min_w, max_w = max_w, min_w

        costs_key = _normalize_costs(req.costs)

        # Run optimization with timeout protection
        opt_timeout = float(os.environ.get("OPT_TIMEOUT", "25"))
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                _wf_eqw,
                tickers=tickers,
                start=req.start,
                end=req.end,
                dtype=req.dtype,
                interval=req.interval,
                rebalance=rebalance,
                costs=dict(costs_key) if costs_key else None,
                leverage=leverage,
                min_weight=min_w,
                max_weight=max_w
            )
            try:
                res = fut.result(timeout=opt_timeout)
            except _Timeout:
                raise HTTPException(status_code=504, detail="Equal-weight optimization timed out")
            except Exception as e:
                log.error(f"Optimization error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Process results
        weights = res.get("weights")
        pnl = res.get("pnl")
        details = normalize_details(res.get("details"))
        
        try:
            details["metrics"] = compute_metrics(pnl, weights)
        except Exception as e:
            log.warning(f"Failed to compute metrics: {str(e)}")
            details["metrics"] = {}

        out = {
            "weights": df_to_records(weights),
            "pnl": series_to_records(pnl, value_name="pnl"),
            "details": details
        }

        log.info(
            "/opt/equal-weight done in %.1fms: weights_rows=%d pnl_rows=%d",
            (time.time() - t0) * 1000.0,
            len(weights) if weights is not None else 0,
            len(pnl) if pnl is not None else 0
        )

        return out

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
