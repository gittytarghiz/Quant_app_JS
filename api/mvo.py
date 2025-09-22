# api/opt_mvo.py — slim API for current walkforward_mvo
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
        if isinstance(s, str):
            u = s.strip().upper()
            if u and u not in seen:
                seen.add(u); out.append(u)
    return out

def _normalize_obj_params(p: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
    if not p: return None
    out = {}
    for k, v in p.items():
        try: out[str(k)] = float(v)
        except (TypeError, ValueError): pass
    return out or None

@lru_cache(maxsize=16)
def _mvo_cached(
    tickers_key: tuple[str, ...],
    start: str, end: str, dtype: str, interval: str, rebalance: str,
    min_weight: float, max_weight: float, min_obs: int, leverage: float,
    objective: str, obj_params: Optional[tuple[tuple[str,float], ...]],
) -> dict[str, Any]:
    res = _wf_mvo(
        tickers=list(tickers_key), start=start, end=end,
        dtype=dtype, interval=interval, rebalance=rebalance,
        min_weight=min_weight, max_weight=max_weight, min_obs=min_obs,
        leverage=leverage, objective=objective,
        objective_params=dict(obj_params) if obj_params else None,
    )
    w, p = res["weights"], res["pnl"]
    details = normalize_details(res.get("details"))
    try: details["metrics"] = compute_metrics(p, w)
    except Exception: pass
    return {"weights": df_to_records(w), "pnl": series_to_records(p, "pnl"), "details": details}

class MVORequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_obs: int = 60
    leverage: float = 1.0
    objective: str = Field(default="min_vol")
    objective_params: Optional[dict[str, float]] = None

@router.post("/mvo")
def mvo(req: MVORequest) -> dict[str, Any]:
    tickers = _dedup_upper(req.tickers)
    if not tickers: raise HTTPException(400, "tickers must be non-empty")
    if len(tickers) > 64: raise HTTPException(400, "too many tickers (max 64)")

    rebalance = (req.rebalance or "monthly").lower()
    if rebalance not in _ALLOWED_REBAL:
        raise HTTPException(400, f"rebalance must be one of {sorted(_ALLOWED_REBAL)}")

    try: lev = max(0.0, min(5.0, float(req.leverage)))
    except: raise HTTPException(400, "invalid leverage")

    try:
        min_w, max_w = float(req.min_weight), float(req.max_weight)
    except: raise HTTPException(400, "invalid weights")
    if min_w > max_w: min_w, max_w = max_w, min_w

    obj_params = _normalize_obj_params(req.objective_params)
    obj_params_key = tuple(sorted(obj_params.items())) if obj_params else None

    timeout = float(os.environ.get("OPT_TIMEOUT", "25"))
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            _mvo_cached, tuple(tickers), req.start, req.end, req.dtype, req.interval,
            rebalance, min_w, max_w, int(req.min_obs), lev,
            req.objective.lower().strip(), obj_params_key
        )
        try: return fut.result(timeout=timeout)
        except _Timeout: raise HTTPException(504, "MVO optimization timed out")
        except Exception as e: raise HTTPException(400, f"MVO failed: {e}")
