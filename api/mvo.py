# api/opt_mvo.py — hardened, copy-style MVO endpoint matching slim walkforward_mvo

from __future__ import annotations
from typing import Any, Optional
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
import os, logging, time

from portfolio_optimization.walkforward_mvo import walkforward_mvo as _wf_mvo
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics

router = APIRouter(prefix="/opt", tags=["opt"])
_ALLOWED_REBAL = {"daily", "weekly", "monthly", "quarterly"}

def _dedup_upper(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for s in seq or []:
        if not isinstance(s, str): continue
        u = s.strip().upper()
        if u and u not in seen: seen.add(u); out.append(u)
    return out

def _normalize_costs(costs: Optional[dict[str, float]]) -> Optional[tuple[tuple[str, float], ...]]:
    if not costs: return None
    allowed = {"bps","slippage_bps","spread_bps"}
    items = []
    for k,v in costs.items():
        if k in allowed and v is not None:
            try: items.append((k, float(v)))
            except (TypeError, ValueError): pass
    return tuple(sorted(items)) or None

def _normalize_obj_params(p: Optional[dict[str, float]]) -> Optional[tuple[tuple[str, float], ...]]:
    if not p: return None
    items = []
    for k,v in p.items():
        if v is None: continue
        try: items.append((str(k), float(v)))
        except (TypeError, ValueError): pass
    return tuple(sorted(items)) or None

@lru_cache(maxsize=16)
def _mvo_cached(
    tickers_key: tuple[str, ...],
    start: str, end: str, dtype: str, interval: str, rebalance: str,
    costs_key: Optional[tuple[tuple[str, float], ...]],
    min_weight: float, max_weight: float, min_obs: int, leverage: float,
    objective: str, obj_params_key: Optional[tuple[tuple[str, float], ...]],
) -> dict[str, Any]:
    res = _wf_mvo(
        tickers=list(tickers_key), start=start, end=end,
        dtype=dtype, interval=interval, rebalance=rebalance,
        costs=dict(costs_key) if costs_key else None,
        min_weight=min_weight, max_weight=max_weight, min_obs=min_obs,
        leverage=leverage, objective=objective,
        objective_params=dict(obj_params_key) if obj_params_key else None,
    )
    raw_w, raw_p = res.get("weights"), res.get("pnl")
    details = normalize_details(res.get("details"))
    try: details["metrics"] = compute_metrics(raw_p, raw_w)
    except Exception: pass
    return {
        "weights": df_to_records(raw_w),
        "pnl": series_to_records(raw_p, value_name="pnl"),
        "details": details,
    }

class MVORequest(BaseModel):
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
    objective: str = Field(default="min_vol")
    objective_params: Optional[dict[str, float]] = None

@router.post("/mvo")
def mvo(req: MVORequest) -> dict[str, Any]:
    log, t0 = logging.getLogger("api.mvo"), time.time()
    try:
        log.info("POST /opt/mvo tickers=%d dtype=%s interval=%s rebalance=%s obj=%s %s→%s",
                 len(req.tickers or []), req.dtype, req.interval, req.rebalance, req.objective, req.start, req.end)
    except Exception: pass

    tickers = _dedup_upper(req.tickers)
    if not tickers: raise HTTPException(400, "tickers must be a non-empty list")
    if len(tickers) > 64: raise HTTPException(400, "too many tickers; limit to 64")

    rebalance = (req.rebalance or "monthly").lower().strip()
    if rebalance not in _ALLOWED_REBAL:
        raise HTTPException(400, f"rebalance must be one of {sorted(_ALLOWED_REBAL)}")

    try: lev = float(req.leverage)
    except (TypeError, ValueError): raise HTTPException(400, "leverage must be a number")
    lev = max(0.0, min(5.0, lev))

    try:
        min_w = max(0.0, min(1.0, float(req.min_weight)))
        max_w = max(0.0, min(1.0, float(req.max_weight)))
    except (TypeError, ValueError):
        raise HTTPException(400, "min_weight/max_weight must be numbers")
    if min_w > max_w: min_w, max_w = max_w, min_w

    costs_key = _normalize_costs(req.costs)
    obj_params_key = _normalize_obj_params(req.objective_params)

    opt_timeout = float(os.environ.get("OPT_TIMEOUT", "25"))
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            _mvo_cached,
            tuple(tickers), req.start, req.end, req.dtype, req.interval, rebalance,
            costs_key, min_w, max_w, int(req.min_obs), lev,
            str(req.objective).lower().strip(), obj_params_key,
        )
        try:
            out = fut.result(timeout=opt_timeout)
        except _Timeout:
            raise HTTPException(status_code=504, detail="MVO optimization timed out")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"MVO failed: {e}")

    try: log.info("/opt/mvo done in %.1fms: weights_rows=%d pnl_rows=%d",
                  (time.time()-t0)*1000.0, len(out.get("weights",[])), len(out.get("pnl",[])))
    except Exception: pass
    return out
