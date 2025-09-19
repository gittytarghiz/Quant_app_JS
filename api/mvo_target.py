# api/opt_mvo_target.py — copy-style hardened endpoint for target-return MVO

from __future__ import annotations
from typing import Any, Optional
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
import os, logging, time

from portfolio_optimization.walkforward_mvo_target import (
    walkforward_mvo_target_return as _wf_mvo_target,
)
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics

router = APIRouter(prefix="/opt", tags=["opt"])
_ALLOWED_REBAL = {"daily", "weekly", "monthly", "quarterly"}

# ---------- helpers ----------
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
    for k, v in costs.items():
        if k in allowed and v is not None:
            try: items.append((k, float(v)))
            except (TypeError, ValueError): pass
    return tuple(sorted(items)) or None

@lru_cache(maxsize=16)
def _mvo_target_cached(
    tickers_key: tuple[str, ...],
    start: str, end: str, dtype: str, interval: str, rebalance: str,
    costs_key: Optional[tuple[tuple[str, float], ...]],
    min_weight: float, max_weight: float, min_obs: int, leverage: float,
    target_return: float, cov_shrinkage: float, cov_estimator: Optional[str],
    # DE params
    de_maxiter: int, de_popsize: int, de_tol: float,
    de_mutation_a: float, de_mutation_b: float,
    de_recombination: float, de_seed: int, de_workers: int, de_polish: bool,
) -> dict[str, Any]:
    res = _wf_mvo_target(
        tickers=list(tickers_key),
        start=start, end=end,
        dtype=dtype, interval=interval, rebalance=rebalance,
        costs=dict(costs_key) if costs_key else None,
        min_weight=min_weight, max_weight=max_weight, min_obs=min_obs,
        leverage=leverage,
        target_return=target_return,
        cov_shrinkage=cov_shrinkage,
        cov_estimator=cov_estimator,
        de_maxiter=de_maxiter, de_popsize=de_popsize, de_tol=de_tol,
        de_mutation=(de_mutation_a, de_mutation_b),
        de_recombination=de_recombination, de_seed=de_seed,
        de_workers=de_workers, de_polish=de_polish,
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

# ---------- schema ----------
class MVOTargetRequest(BaseModel):
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

    target_return: float = 0.0
    cov_shrinkage: float = 0.0
    cov_estimator: Optional[str] = Field(default=None, description="sample|diag|lw")

    # DE fallback controls (used only if cvxpy missing)
    de_maxiter: int = 35
    de_popsize: int = 20
    de_tol: float = 0.01
    de_mutation: tuple[float, float] | float = (0.5, 1.0)
    de_recombination: float = 0.7
    de_seed: int = 42
    de_workers: int = 1
    de_polish: bool = False

# ---------- endpoint ----------
@router.post("/mvo-target")
def mvo_target(req: MVOTargetRequest) -> dict[str, Any]:
    log, t0 = logging.getLogger("api.mvo_target"), time.time()
    try:
        log.info("POST /opt/mvo-target tickers=%d dtype=%s interval=%s rebalance=%s target=%s %s→%s",
                 len(req.tickers or []), req.dtype, req.interval, req.rebalance, req.target_return, req.start, req.end)
    except Exception: pass

    # guardrails
    tickers = _dedup_upper(req.tickers)
    if not tickers: raise HTTPException(400, "tickers must be a non-empty list")
    if len(tickers) > 64: raise HTTPException(400, "too many tickers; limit to 64")

    rebalance = (req.rebalance or "monthly").lower().strip()
    if rebalance not in _ALLOWED_REBAL:
        raise HTTPException(400, f"rebalance must be one of {sorted(_ALLOWED_REBAL)}")

    try: lev = max(0.0, min(5.0, float(req.leverage)))
    except (TypeError, ValueError): raise HTTPException(400, "leverage must be a number")

    try:
        min_w = max(0.0, min(1.0, float(req.min_weight)))
        max_w = max(0.0, min(1.0, float(req.max_weight)))
    except (TypeError, ValueError):
        raise HTTPException(400, "min_weight/max_weight must be numbers")
    if min_w > max_w: min_w, max_w = max_w, min_w

    try:
        tgt = float(req.target_return)
        shrink = max(0.0, min(1.0, float(req.cov_shrinkage)))
    except (TypeError, ValueError):
        raise HTTPException(400, "target_return/cov_shrinkage must be numbers")

    # normalize de_mutation -> (a,b)
    dm_a, dm_b = (float(req.de_mutation), float(req.de_mutation)) if isinstance(req.de_mutation, (int, float)) \
                 else (float(req.de_mutation[0]), float(req.de_mutation[1]))

    costs_key = _normalize_costs(req.costs)

    opt_timeout = float(os.environ.get("OPT_TIMEOUT", "25"))
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            _mvo_target_cached,
            tuple(tickers), req.start, req.end, req.dtype, req.interval, rebalance,
            costs_key, min_w, max_w, int(req.min_obs), lev,
            tgt, shrink, (req.cov_estimator.lower().strip() if isinstance(req.cov_estimator, str) else None),
            int(req.de_maxiter), int(req.de_popsize), float(req.de_tol),
            dm_a, dm_b, float(req.de_recombination), int(req.de_seed),
            int(req.de_workers), bool(req.de_polish),
        )
        try:
            out = fut.result(timeout=opt_timeout)
        except _Timeout:
            raise HTTPException(status_code=504, detail="MVO-target optimization timed out")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"MVO-target failed: {e}")

    try:
        log.info("/opt/mvo-target done in %.1fms: weights_rows=%d pnl_rows=%d",
                 (time.time()-t0)*1000.0, len(out.get("weights",[])), len(out.get("pnl",[])))
    except Exception: pass
    return out
